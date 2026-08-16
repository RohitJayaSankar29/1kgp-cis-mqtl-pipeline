#!/usr/bin/env python3
"""cis-mQTL covariate matrix (v2): known covariates + ancestry + residual structure.

  known covariates : sex, platform (R9/R10 batch), and age if provided (PROPHECY)
  genotype PCs     : ancestry / population structure (from plink2 --pca)
  methylation PCs  : hidden structure REMAINING after regressing out the known
                     covariates + genotype PCs -- so sex/platform/ancestry are
                     not double-counted (PEER-style residual factors).

Output: covariates as ROWS, samples as COLUMNS (tensorQTL reads it transposed).
"""
import argparse, gzip
import numpy as np
import pandas as pd


def read_matrix(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        hdr = f.readline().rstrip("\n").split("\t")
    samples = hdr[4:]
    df = pd.read_csv(path, sep="\t")
    M = df.iloc[:, 4:].astype(float)
    M.columns = samples
    return samples, M


def residual_methylation_pcs(M, k, C):
    """PCs of the methylation matrix after regressing out known covariates C."""
    X = M.values.T.astype(float)                 # samples x CpGs
    X = X - X.mean(0, keepdims=True)
    if C is not None and C.shape[1] > 0:
        Cc = np.column_stack([np.ones(C.shape[0]), C])
        beta, *_ = np.linalg.lstsq(Cc, X, rcond=None)
        X = X - Cc @ beta                        # residualise
    G = X @ X.T                                  # samples x samples Gram
    w, V = np.linalg.eigh(G)
    idx = np.argsort(w)[::-1][:k]
    return V[:, idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--geno-pca", required=True, help="plink2 .eigenvec")
    ap.add_argument("--metadata", required=True, help="TSV: sample, sex, platform[, age]")
    ap.add_argument("--n-geno-pc", type=int, default=5)
    ap.add_argument("--n-meth-pc", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    samples, M = read_matrix(a.matrix)
    n = len(samples)
    known = {}   # known covariates only (used both as covariates and to residualise)

    md = pd.read_csv(a.metadata, sep="\t", dtype=str).set_index("sample").reindex(samples)

    if "sex" in md.columns:
        sx = md["sex"].map({"M": 0, "F": 1})
        if sx.isna().any():
            print(f"WARNING: sex missing for {int(sx.isna().sum())}; dropping sex")
        elif sx.nunique() < 2:
            print("NOTE: sex monomorphic; dropping sex")
        else:
            known["sex"] = sx.values.astype(float)

    if "platform" in md.columns and md["platform"].nunique() > 1:
        levels = sorted(md["platform"].dropna().unique())
        for lv in levels[1:]:                     # first level = reference
            known[f"platform_{lv}"] = (md["platform"] == lv).astype(float).values
        print(f"platform levels {levels}; reference={levels[0]}")
    elif "platform" in md.columns:
        print("NOTE: platform monomorphic; dropping platform")

    if "age" in md.columns and md["age"].notna().any():
        age = pd.to_numeric(md["age"], errors="coerce")
        if age.notna().all():
            known["age"] = ((age - age.mean()) / age.std()).values
        else:
            print("NOTE: age incomplete; dropping age")
    else:
        print("age: not provided (expected for 1000G) -- skipping")

    ev = pd.read_csv(a.geno_pca, sep=r"\s+")
    iidcol = "IID" if "IID" in ev.columns else ev.columns[1]
    ev = ev.set_index(iidcol).reindex(samples)
    for c in [c for c in ev.columns if c.upper().startswith("PC")][:a.n_geno_pc]:
        known[f"geno{c.upper()}"] = ev[c].astype(float).values

    C = np.column_stack(list(known.values())) if known else np.empty((n, 0))
    PC = residual_methylation_pcs(M, a.n_meth_pc, C)

    cov = dict(known)
    for i in range(a.n_meth_pc):
        cov[f"methPC{i+1}"] = PC[:, i]

    out = pd.DataFrame(cov, index=samples).T
    out.index.name = "ID"
    out.to_csv(a.out, sep="\t")
    print(f"Wrote {a.out}: {out.shape[0]} covariates x {out.shape[1]} samples")
    print("  " + ", ".join(out.index.tolist()))
    print(f"  residual df ~ {n - out.shape[0] - 2}  (keep this comfortably positive at n={n})")


if __name__ == "__main__":
    main()
