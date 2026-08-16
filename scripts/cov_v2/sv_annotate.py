#!/usr/bin/env python3
"""Stage 07b (NOVEL): does a structural variant explain the cis-mQTL signal?

For each FDR-significant CpG:
  (a) DIRECT TEST  -- regress covariate-residualised methylation on each cis SV's
      dosage (OLS) -> SV-mQTL beta and p-value.
  (b) LD TAGGING   -- r^2 between the CpG's lead SNV and each cis SV.
  FLAG the CpG if a cis SV is directly associated (p < --sv-p) OR sits in high LD
  (r^2 >= --r2) with the lead SNV  -> candidate SV-driven mQTL.

Inputs: methylation matrix, covariates, the permutation table (lead SNV per CpG),
the significant-CpG list, the SNV plink set (lead dosage + LD), and the SV dosage
matrix from extract_sv_genotypes.pbs.

Output: TSV (phenotype_id, cpg_pos, lead_snv, sv_id, svtype, sv_pos, distance,
sv_beta, sv_p, r2_lead, flag).
"""
import argparse, gzip, os, time
import numpy as np
import pandas as pd


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def read_matrix(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        hdr = f.readline().rstrip("\n").split("\t")
    samples = hdr[4:]
    df = pd.read_csv(path, sep="\t")
    pos = df.set_index(df.columns[3])[df.columns[1]].astype(int)   # cpg_id -> start
    M = df.iloc[:, 4:].astype(float)
    M.index = df.iloc[:, 3]
    M.columns = samples
    return samples, M, pos


def gt_to_dosage(s):
    s = str(s)
    if s in (".", "./.", ".|.", ""):
        return np.nan
    a = s.replace("|", "/").split("/")
    try:
        return float(int(a[0] != "0") + int(a[1] != "0"))
    except Exception:
        return np.nan


def load_sv_matrix(path, samples):
    df = pd.read_csv(path, sep="\t", dtype=str)
    meta = df[["sv_id", "chrom", "pos", "end", "svtype", "svlen"]].copy()
    meta["pos"] = pd.to_numeric(meta["pos"], errors="coerce")
    gt = df[samples].applymap(gt_to_dosage)   # SVs x samples
    gt.index = df["sv_id"]
    meta = meta.set_index("sv_id")
    return meta, gt


def residualise(y, C):
    Cc = np.column_stack([np.ones(len(y)), C])
    beta, *_ = np.linalg.lstsq(Cc, y, rcond=None)
    return y - Cc @ beta


def ols_p(x, y):
    """Simple slope test of y ~ x (both 1-D, NaNs dropped). Returns beta, p."""
    from scipy.stats import t as tdist
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 10 or np.nanstd(x) == 0:
        return np.nan, np.nan
    x1 = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(x1, y, rcond=None)
    resid = y - x1 @ beta
    dof = len(x) - 2
    s2 = (resid @ resid) / dof
    xtx_inv = np.linalg.inv(x1.T @ x1)
    se = np.sqrt(s2 * xtx_inv[1, 1])
    tstat = beta[1] / se if se > 0 else np.nan
    p = 2 * tdist.sf(abs(tstat), dof) if np.isfinite(tstat) else np.nan
    return float(beta[1]), float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--covariates", required=True)
    ap.add_argument("--permutation", required=True, help="chr22.perm.txt.gz (lead SNV per CpG)")
    ap.add_argument("--significant-cpgs", required=True)
    ap.add_argument("--plink-prefix", required=True, help="SNV plink set (for lead dosage + LD)")
    ap.add_argument("--sv-matrix", required=True, help="chr22.sv.gt.tsv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cis-window", type=int, default=1000000)
    ap.add_argument("--sv-p", type=float, default=1e-3)
    ap.add_argument("--r2", type=float, default=0.8)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    sig = [l.strip() for l in open(a.significant_cpgs) if l.strip()]
    if not sig:
        log("No significant CpGs -- nothing to annotate.")
        pd.DataFrame().to_csv(a.out, sep="\t", index=False)
        return

    samples, M, cpg_pos = read_matrix(a.matrix)
    cov = pd.read_csv(a.covariates, sep="\t", index_col=0).T.reindex(samples)
    C = cov.values.astype(float)

    perm = pd.read_csv(a.permutation, sep="\t", index_col=0)
    lead_col = "variant_id" if "variant_id" in perm.columns else perm.columns[0]

    import tensorqtl
    from tensorqtl import genotypeio
    pr = genotypeio.PlinkReader(a.plink_prefix)
    geno = pr.load_genotypes()[samples]              # SNVs x samples, aligned

    sv_meta, sv_gt = load_sv_matrix(a.sv_matrix, samples)
    log(f"{len(sig)} significant CpGs | {geno.shape[0]:,} SNVs | {sv_gt.shape[0]:,} SVs")

    rows = []
    for cpg in sig:
        if cpg not in M.index or cpg not in cpg_pos.index:
            continue
        pos = int(cpg_pos[cpg])
        cis = sv_meta[(sv_meta["pos"] >= pos - a.cis_window) &
                      (sv_meta["pos"] <= pos + a.cis_window)]
        if cis.empty:
            continue
        y = residualise(M.loc[cpg].values.astype(float), C)
        lead = perm.loc[cpg, lead_col] if cpg in perm.index else None
        lead_dos = geno.loc[lead].values.astype(float) if (lead in geno.index) else None
        for sv_id, mrow in cis.iterrows():
            dos = sv_gt.loc[sv_id].values.astype(float)
            beta, p = ols_p(dos.copy(), y.copy())
            r2 = np.nan
            if lead_dos is not None:
                mm = ~(np.isnan(dos) | np.isnan(lead_dos))
                if mm.sum() > 10 and np.nanstd(dos[mm]) > 0 and np.nanstd(lead_dos[mm]) > 0:
                    r2 = float(np.corrcoef(dos[mm], lead_dos[mm])[0, 1] ** 2)
            flag = ((p is not None and np.isfinite(p) and p < a.sv_p) or
                    (np.isfinite(r2) and r2 >= a.r2))
            rows.append({"phenotype_id": cpg, "cpg_pos": pos, "lead_snv": lead,
                         "sv_id": sv_id, "svtype": mrow["svtype"], "sv_pos": mrow["pos"],
                         "distance": int(mrow["pos"] - pos), "sv_beta": beta,
                         "sv_p": p, "r2_lead": r2, "sv_implicated": bool(flag)})

    out = pd.DataFrame(rows)
    out.to_csv(a.out, sep="\t", index=False)
    nflag = int(out["sv_implicated"].sum()) if not out.empty else 0
    ncpg = out.loc[out["sv_implicated"], "phenotype_id"].nunique() if not out.empty else 0
    log(f"Wrote {a.out}: {out.shape[0]} CpG-SV pairs; {nflag} flagged across {ncpg} CpGs")

    # NEXT STEP (joint fine-mapping): recode these SVs to a plink-friendly biallelic
    # form, merge with the SNV set, and re-run SuSiE so SVs compete with SNVs for PIP.


if __name__ == "__main__":
    main()
