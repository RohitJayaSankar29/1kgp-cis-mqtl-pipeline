#!/usr/bin/env python3
"""
build_matrix_genome.py - genome-wide version of build_matrix.py.
Reads each sample's whole-genome modkit bedMethyl ONCE, splitting all
chromosomes in a single pass, then builds one tensorQTL phenotype BED per
chromosome. Same filters/transform as the chr22 pilot (min-cov, min-frac,
mean-impute, M-value) so results are comparable.

Memory strategy: process ONE chromosome at a time across all samples. For a
given chromosome, read that chrom's rows from every file, assemble the matrix,
filter/impute/transform, write, free, next chrom. Never holds the whole genome
for all samples at once.

Sample sheet TSV (no header): <sample_id>\t<path_to_combined.bed.gz>
"""
import argparse, gzip, os, sys
import numpy as np
import pandas as pd

CHROMS = [f"chr{i}" for i in range(1, 23)]   # autosomes; add chrX if wanted


def load_sample_chrom(path, chrom, min_cov, mod="m"):
    """Return Series {start_pos: beta} for one sample, one chromosome."""
    pos, beta = [], []
    ck = chrom + "\t"
    with gzip.open(path, "rt") as f:
        for line in f:
            if not line.startswith(ck):        # fast chrom prefilter
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 12 or c[3] != mod:
                continue
            try:
                cov = int(c[9])
            except ValueError:
                continue
            if cov < min_cov:
                continue
            pos.append(int(c[1]))
            beta.append(float(c[10]) / 100.0)
    return pd.Series(beta, index=pos, dtype="float64")


def build_chrom(sheet, chrom, a):
    cols = {}
    for _, r in sheet.iterrows():
        cols[r["sid"]] = load_sample_chrom(r["path"], chrom, a.min_cov, a.mod)
    mat = pd.DataFrame(cols)
    if mat.shape[0] == 0:
        print(f"  {chrom}: no CpGs", file=sys.stderr)
        return None
    n = mat.shape[1]
    keep = mat.notna().sum(axis=1) >= a.min_frac * n
    mat = mat.loc[keep].sort_index()
    if mat.shape[0] == 0:
        print(f"  {chrom}: 0 CpGs passed min-frac", file=sys.stderr)
        return None
    vals = mat.to_numpy(dtype="float64", copy=True)
    rmean = np.nanmean(vals, axis=1)
    ni, nj = np.where(np.isnan(vals))
    vals[ni, nj] = rmean[ni]
    if a.mvalue:
        eps = 1e-2
        vals = np.log2((vals + eps) / (1.0 - vals + eps))
    starts = mat.index.astype(int).to_numpy()
    bed = pd.DataFrame(vals, columns=mat.columns)
    bed.insert(0, "phenotype_id", [f"{chrom}_{p}" for p in starts])
    bed.insert(0, "end", starts + 1)
    bed.insert(0, "start", starts)
    bed.insert(0, "#chr", chrom)
    bed = bed.sort_values(["#chr", "start"]).reset_index(drop=True)
    out = os.path.join(a.outdir, f"{chrom}.methylation_Mval.bed")
    bed.to_csv(out, sep="\t", index=False, float_format="%.4f")
    print(f"  {chrom}: {bed.shape[0]} CpGs x {n} samples -> {out}", file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--chroms", default="", help="comma list e.g. chr1,chr2 (default all autosomes)")
    ap.add_argument("--min-cov", type=int, default=5)
    ap.add_argument("--min-frac", type=float, default=0.90)
    ap.add_argument("--mod", default="m")
    ap.add_argument("--mvalue", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    sheet = pd.read_csv(a.sheet, sep="\t", header=None, names=["sid", "path"])
    print(f"{len(sheet)} samples", file=sys.stderr)
    chroms = a.chroms.split(",") if a.chroms else CHROMS
    for chrom in chroms:
        build_chrom(sheet, chrom, a)


if __name__ == "__main__":
    main()
