#!/usr/bin/env python3
"""Stage 09 (framework): multi-evidence cis-mQTL variant prioritisation.

Per (variant, CpG) feature vector -> standardise -> two scores:
  (1) weighted additive score   S_i = sum_j W_j * Z_ij        (Ionita-Laza-style)
  (2) elastic-net score: regularised regression of the fine-mapping signal (PIP)
      on the remaining annotations (Gagliano-style), giving data-driven weights.
Convergence between the two rankings (Spearman) is reported as the robustness
criterion, since there is no causal ground truth.

Features:
  fine-mapping : pip
  association  : neg_log10_p, abs_effect      (from --nominal, optional)
                 neg_log10_fdr                (from --fdr, per-CpG, optional)
  variant      : maf, prox_cpg (closeness to the CpG)
  structural   : sv_flag (CpG is SV-implicated)
  functional   : var_ccre, var_cpg_island, prox_tss  (annotated at the variant)

Candidate set = variants in a SuSiE 95% credible set.
"""
import argparse, os, time
import numpy as np
import pandas as pd


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


# ---- weights for the additive model (standardised features). Tune freely. ----
DEFAULT_WEIGHTS = {
    "pip": 2.0, "neg_log10_p": 1.0, "abs_effect": 0.5, "neg_log10_fdr": 0.5,
    "maf": 0.0, "prox_cpg": 0.5, "sv_flag": 1.0,
    "var_ccre": 1.0, "var_cpg_island": 0.5, "prox_tss": 0.5,
}


def load_bed(path, cols):
    if not path or not os.path.exists(path):
        return None
    d = pd.read_csv(path, sep="\t", header=None)
    d.columns = cols[:d.shape[1]]
    d["start"] = d["start"].astype(int)
    d["end"] = d["end"].astype(int)
    return d


def merge_intervals(bed):
    s = bed.sort_values("start")
    starts, ends, cs, ce = [], [], None, None
    for st, en in zip(s["start"].values, s["end"].values):
        if cs is None:
            cs, ce = st, en
        elif st <= ce:
            ce = max(ce, en)
        else:
            starts.append(cs); ends.append(ce); cs, ce = st, en
    if cs is not None:
        starts.append(cs); ends.append(ce)
    return np.array(starts), np.array(ends)


def overlap_flag(pos, bed):
    pos = np.asarray(pos)
    if bed is None or bed.empty:
        return np.zeros(len(pos))
    starts, ends = merge_intervals(bed)
    i = np.searchsorted(starts, pos, side="right") - 1
    out = np.zeros(len(pos))
    ok = i >= 0
    out[ok] = (pos[ok] <= ends[i[ok]]).astype(float)
    return out


def nearest_dist(pos, bed):
    pos = np.asarray(pos)
    if bed is None or bed.empty:
        return np.full(len(pos), np.nan)
    t = np.sort(bed["start"].values)
    i = np.searchsorted(t, pos)
    d = np.full(len(pos), np.inf)
    for off in (-1, 0):
        j = np.clip(i + off, 0, len(t) - 1)
        dd = np.abs(pos - t[j])
        d = np.minimum(d, dd)
    return d


def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    mu, sd = s.mean(), s.std(ddof=0)
    return (s - mu) / sd if sd and np.isfinite(sd) and sd > 0 else s * 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--susie", required=True)
    ap.add_argument("--bim", required=True, help="plink .bim for variant positions")
    ap.add_argument("--sv")
    ap.add_argument("--functional", help="per-CpG functional (for CpG positions / fallback)")
    ap.add_argument("--nominal", help="tensorQTL nominal stats (parquet or tsv); optional")
    ap.add_argument("--fdr", help="per-CpG FDR table with a qval column; optional")
    ap.add_argument("--tss-bed")
    ap.add_argument("--ccre-bed")
    ap.add_argument("--cpg-island-bed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cs-only", action="store_true", default=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    df = pd.read_csv(a.susie, sep="\t")
    if a.cs_only and "cs_id" in df.columns:
        df = df[df["cs_id"].notna() & (df["cs_id"].astype(str) != "-1")].copy()
    log(f"candidate credible-set variants: {df.shape[0]} across {df['phenotype_id'].nunique()} CpGs")

    # --- variant positions from bim ---
    bim = pd.read_csv(a.bim, sep=r"\s+", header=None,
                      names=["chrom", "snp", "cm", "pos", "a1", "a2"])
    vpos = bim.set_index("snp")["pos"]
    df["var_pos"] = df["variant_id"].map(vpos)

    # --- CpG positions: parse from phenotype_id (chr22_<pos>) or functional ---
    def cpg_pos_from_id(pid):
        try:
            return int(str(pid).split("_")[-1])
        except Exception:
            return np.nan
    df["cpg_pos"] = df["phenotype_id"].map(cpg_pos_from_id)
    if df["cpg_pos"].isna().any() and a.functional and os.path.exists(a.functional):
        fa = pd.read_csv(a.functional, sep="\t").set_index("cpg")["pos"]
        df.loc[df["cpg_pos"].isna(), "cpg_pos"] = df["phenotype_id"].map(fa)

    # --- features ---
    feat = pd.DataFrame(index=df.index)
    feat["pip"] = pd.to_numeric(df.get("pip"), errors="coerce")
    af = pd.to_numeric(df.get("af"), errors="coerce")
    feat["maf"] = np.minimum(af, 1 - af)
    feat["prox_cpg"] = -np.log10(np.abs(df["var_pos"] - df["cpg_pos"]).astype(float) + 1)

    # structural
    sv_flag = {}
    if a.sv and os.path.exists(a.sv):
        sv = pd.read_csv(a.sv, sep="\t")
        if "sv_implicated" in sv.columns:
            sv["flag"] = sv["sv_implicated"].astype(str).isin(["True", "true", "1"])
            sv_flag = sv.groupby("phenotype_id")["flag"].any().astype(float).to_dict()
    feat["sv_flag"] = df["phenotype_id"].map(sv_flag).fillna(0.0)

    # variant-level functional
    tss = load_bed(a.tss_bed, ["chrom", "start", "end", "gene", "score", "strand"])
    ccre = load_bed(a.ccre_bed, ["chrom", "start", "end", "label"])
    isl = load_bed(a.cpg_island_bed, ["chrom", "start", "end", "label"])
    vp = df["var_pos"].values
    feat["var_ccre"] = overlap_flag(vp, ccre)
    feat["var_cpg_island"] = overlap_flag(vp, isl)
    feat["prox_tss"] = -np.log10(nearest_dist(vp, tss) + 1)

    # association (optional)
    if a.nominal and os.path.exists(a.nominal):
        if a.nominal.endswith(".parquet"):
            # filtered read: only credible-set variants + needed columns.
            # keeps memory low on a huge all-pairs parquet and scales genome-wide.
            import pyarrow.parquet as pq
            names = pq.read_schema(a.nominal).names
            pcol = next((c for c in names if "pval" in c.lower()), None)
            scol = next((c for c in names if c.lower() in ("slope", "beta", "effect")), None)
            usecols = [c for c in (["phenotype_id", "variant_id", pcol, scol]) if c]
            need = list(pd.unique(df["variant_id"].astype(str)))
            try:
                nom = pd.read_parquet(a.nominal, columns=usecols,
                                      filters=[("variant_id", "in", need)])
            except Exception as e:
                log(f"  filtered read failed ({e}); falling back to column-only read")
                nom = pd.read_parquet(a.nominal, columns=usecols)
        else:
            nom = pd.read_csv(a.nominal, sep="\t")
            pcol = next((c for c in nom.columns if "pval" in c.lower()), None)
            scol = next((c for c in nom.columns if c.lower() in ("slope", "beta", "effect")), None)
        log(f"  nominal rows loaded: {len(nom):,}  (cols={list(nom.columns)})")
        key = ["phenotype_id", "variant_id"]
        if all(k in nom.columns for k in key) and pcol:
            nom = nom[key + [c for c in (pcol, scol) if c]].drop_duplicates(key)
            df = df.merge(nom, on=key, how="left")
            feat["neg_log10_p"] = -np.log10(pd.to_numeric(df[pcol], errors="coerce").clip(lower=1e-300))
            if scol:
                feat["abs_effect"] = pd.to_numeric(df[scol], errors="coerce").abs()
    if a.fdr and os.path.exists(a.fdr):
        fdr = pd.read_csv(a.fdr, sep="\t", index_col=0)
        qcol = next((c for c in fdr.columns if "qval" in c.lower()), None)
        if qcol:
            qmap = fdr[qcol].to_dict()
            feat["neg_log10_fdr"] = -np.log10(
                pd.to_numeric(df["phenotype_id"].map(qmap), errors="coerce").clip(lower=1e-300))

    feat = feat.dropna(axis=1, how="all")
    log(f"features assembled: {list(feat.columns)}")

    # --- standardise ---
    Z = feat.apply(zscore).fillna(0.0)

    # --- (1) weighted additive score ---
    w = {k: DEFAULT_WEIGHTS.get(k, 1.0) for k in Z.columns}
    df["score_weighted"] = sum(w[c] * Z[c] for c in Z.columns)

    # --- (2) elastic-net score: predict PIP from the other annotations ---
    df["score_elasticnet"] = np.nan
    coefs = {}
    try:
        from sklearn.linear_model import ElasticNetCV
        Xcols = [c for c in Z.columns if c != "pip"]
        y = feat["pip"].fillna(feat["pip"].median()).values
        X = Z[Xcols].values
        enet = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, 1.0], cv=5,
                            max_iter=5000, n_jobs=-1)
        enet.fit(X, y)
        df["score_elasticnet"] = enet.predict(X)
        coefs = dict(zip(Xcols, enet.coef_))
        log(f"elastic-net: alpha={enet.alpha_:.4g} l1_ratio={enet.l1_ratio_} R2={enet.score(X, y):.3f}")
        log("  learned coefficients: " + ", ".join(f"{k}={v:+.3f}" for k, v in coefs.items()))
    except Exception as e:
        log(f"elastic-net skipped ({e}); install scikit-learn for the learned model")

    # --- convergence between the two rankings ---
    if df["score_elasticnet"].notna().any():
        from scipy.stats import spearmanr
        rho = spearmanr(df["score_weighted"], df["score_elasticnet"], nan_policy="omit").correlation
        log(f"convergence (Spearman, weighted vs elastic-net): rho = {rho:.3f}")

    out = pd.concat([df, Z.add_prefix("z_")], axis=1).sort_values("score_weighted", ascending=False)
    out.to_csv(a.out, sep="\t", index=False)
    log(f"Wrote {a.out}: {out.shape[0]} ranked variant-CpG pairs, {out.shape[1]} columns")


if __name__ == "__main__":
    main()
