#!/usr/bin/env python3
"""Fine-map FDR-significant cis-mQTL CpGs with SuSiE (tensorQTL wrapper).

For each significant CpG, SuSiE assigns a posterior inclusion probability (PIP)
to every cis variant and groups likely-causal variants into 95% credible sets.
Needs susieR + rpy2 in the env (see docs/DOWNSTREAM.md).

NOTE: the structure of tensorqtl.susie.map_susie's return value has changed
across releases. The parser below is defensive; if your CpGs come back with empty
credible sets, print one entry of `res` and adjust the key names here.

Output: TSV (phenotype_id, variant_id, pip, cs_id).
"""
import argparse, os, time
import numpy as np
import pandas as pd


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def inverse_normal_transform(df):
    from scipy.stats import norm
    r = df.rank(axis=1, method="average")
    n = df.shape[1]
    return pd.DataFrame(norm.ppf((r - 0.5) / n), index=df.index, columns=df.columns)


def flatten_susie(cpg, r):
    """Best-effort flattening of one CpG's SuSiE result into rows."""
    rows = []
    if r is None:
        return rows
    pip = r.get("pip") if isinstance(r, dict) else getattr(r, "pip", None)
    variant_ids = r.get("variant_id") if isinstance(r, dict) else None
    sets = (r.get("sets") if isinstance(r, dict) else {}) or {}
    cs = sets.get("cs", {}) if isinstance(sets, dict) else {}
    idx_to_cs = {}
    if isinstance(cs, dict):
        for cs_id, idxs in cs.items():
            for i in np.atleast_1d(idxs):
                idx_to_cs[int(i)] = cs_id
    if pip is None:
        return rows
    pip = pd.Series(pip)
    ids = variant_ids if variant_ids is not None else list(pip.index)
    for i, v in enumerate(ids):
        rows.append({"phenotype_id": cpg, "variant_id": v,
                     "pip": float(np.asarray(pip)[i]),
                     "cs_id": idx_to_cs.get(i)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plink-prefix", required=True)
    ap.add_argument("--phenotype-bed", required=True)
    ap.add_argument("--covariates", required=True)
    ap.add_argument("--significant-cpgs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cis-window", type=int, default=1000000)
    ap.add_argument("--maf-threshold", type=float, default=0.05)
    ap.add_argument("--max-l", type=int, default=10, help="max causal signals per CpG")
    ap.add_argument("--inverse-normal", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    sig = [l.strip() for l in open(a.significant_cpgs) if l.strip()]
    if not sig:
        log("No significant CpGs -- nothing to fine-map.")
        return

    import torch
    torch.set_num_threads(a.threads)
    import tensorqtl
    from tensorqtl import genotypeio, susie
    log(f"tensorqtl {tensorqtl.__version__} | fine-mapping {len(sig)} CpGs with SuSiE")

    phenotype_df, phenotype_pos_df = tensorqtl.read_phenotype_bed(a.phenotype_bed)
    keep = phenotype_df.index.intersection(sig)
    phenotype_df = phenotype_df.loc[keep]
    phenotype_pos_df = phenotype_pos_df.loc[keep]
    if a.inverse_normal:
        phenotype_df = inverse_normal_transform(phenotype_df)

    covariates_df = pd.read_csv(a.covariates, sep="\t", index_col=0).T
    pr = genotypeio.PlinkReader(a.plink_prefix)
    genotype_df = pr.load_genotypes()
    variant_df = pr.bim.set_index("snp")[["chrom", "pos"]]

    res = susie.map(genotype_df, variant_df, phenotype_df, phenotype_pos_df,
                    covariates_df, L=a.max_l, maf_threshold=a.maf_threshold,
                    window=a.cis_window, coverage=0.95, summary_only=True)

    # map_susie may return a DataFrame (newer tensorQTL) or a dict (older).
    if isinstance(res, pd.DataFrame):
        res.to_csv(a.out, sep="\t", index=True)
        log(f"Wrote {a.out}: DataFrame {res.shape}; columns={list(res.columns)}")
        cs_col = next((c for c in res.columns if "cs" in c.lower()), None)
        if cs_col is not None:
            inc = res[res[cs_col].astype(str).str.strip().replace({"nan": "", "-1": "", "0": ""}) != ""]
            pid = next((c for c in res.columns if "phenotype" in c.lower()), res.columns[0])
            log(f"  rows in a credible set: {inc.shape[0]} across {inc[pid].nunique()} CpGs")
        return

    rows = []
    items = res.items() if isinstance(res, dict) else []
    for cpg, r in items:
        rows.extend(flatten_susie(cpg, r))
    out = pd.DataFrame(rows)
    out.to_csv(a.out, sep="\t", index=False)
    log(f"Wrote {a.out}: {out.shape[0]} variant rows")
    if not out.empty:
        incs = out[out.cs_id.notna()]
        log(f"  variants in a credible set: {incs.shape[0]} across "
            f"{incs.phenotype_id.nunique()} CpGs")


if __name__ == "__main__":
    main()
