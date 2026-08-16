#!/usr/bin/env python3
"""
build_matrix_singlepass.py - SU-efficient genome-wide matrix builder.
Reads each sample's whole-genome bedMethyl ONCE, bucketing all chromosomes in
that single pass (vs the per-chromosome version that re-read every file 22x).
Same filters/transform as the pilot: min-cov, min-frac 0.90, mean-impute, M-value.

Memory strategy: accumulate {chrom: {sample: Series}} during the single read
pass, then build+write each chromosome matrix and immediately free it. Peak
memory ~ one chromosome's worth across all samples at write time, plus the raw
per-sample dicts during reading. For 452 samples genome-wide this needs a
biggish node (request ~128GB to be safe).

VALIDATION: if --validate-against DIR is given and a chromosome already exists
there (from the trusted per-chrom build), the script compares its output to the
existing one and ABORTS if they differ - so a position-alignment bug can't slip
through silently.
"""
import argparse, gzip, os, sys
import numpy as np
import pandas as pd

CHROMS = [f"chr{i}" for i in range(1, 23)]


def read_one_file(path, chroms_set, min_cov, mod="m"):
    """Single pass over one file -> {chrom: (pos_list, beta_list)}."""
    out = {c: ([], []) for c in chroms_set}
    with gzip.open(path, "rt") as f:
        for line in f:
            tab = line.find("\t")
            if tab < 0:
                continue
            chrom = line[:tab]
            if chrom not in out:
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
            out[chrom][0].append(int(c[1]))
            out[chrom][1].append(float(c[10]) / 100.0)
    return out


def build_and_write(chrom, per_sample, a):
    """per_sample: {sid: Series(pos->beta)} for ONE chromosome."""
    mat = pd.DataFrame(per_sample)
    if mat.shape[0] == 0:
        print(f"  {chrom}: no CpGs", file=sys.stderr); return None
    n = mat.shape[1]
    keep = mat.notna().sum(axis=1) >= a.min_frac * n
    mat = mat.loc[keep].sort_index()
    if mat.shape[0] == 0:
        print(f"  {chrom}: 0 pass min-frac", file=sys.stderr); return None
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


def validate(chrom, a):
    """Compare freshly written chrom to an existing trusted build; abort if diff."""
    import subprocess, hashlib
    newf = os.path.join(a.outdir, f"{chrom}.methylation_Mval.bed")
    oldgz = os.path.join(a.validate_against, f"{chrom}.methylation_Mval.bed.gz")
    if not os.path.exists(oldgz):
        return  # nothing to compare
    def digest_new(p):
        h = hashlib.md5()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()
    def digest_old(p):
        h = hashlib.md5()
        with gzip.open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()
    dn, do = digest_new(newf), digest_old(oldgz)
    if dn != do:
        sys.exit(f"VALIDATION FAILED on {chrom}: single-pass output differs from "
                 f"trusted build ({a.validate_against}). Aborting - do NOT trust output.")
    print(f"  {chrom}: VALIDATED identical to trusted build", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--chroms", default="", help="comma list (default all autosomes)")
    ap.add_argument("--min-cov", type=int, default=5)
    ap.add_argument("--min-frac", type=float, default=0.90)
    ap.add_argument("--mod", default="m")
    ap.add_argument("--mvalue", action="store_true")
    ap.add_argument("--validate-against", default="",
                    help="dir of trusted per-chrom builds; abort if any chrom differs")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    sheet = pd.read_csv(a.sheet, sep="\t", header=None, names=["sid", "path"])
    chroms = a.chroms.split(",") if a.chroms else CHROMS
    chroms_set = set(chroms)
    print(f"{len(sheet)} samples, {len(chroms)} chromosomes, single pass", file=sys.stderr)

    # accumulate: {chrom: {sid: Series}}
    acc = {c: {} for c in chroms}
    for i, r in enumerate(sheet.itertuples(index=False), 1):
        d = read_one_file(r.path, chroms_set, a.min_cov, a.mod)
        for c in chroms:
            pos, beta = d[c]
            acc[c][r.sid] = pd.Series(beta, index=pos, dtype="float64")
        if i % 25 == 0:
            print(f"  read {i}/{len(sheet)} files", file=sys.stderr)

    for c in chroms:
        build_and_write(c, acc[c], a)
        acc[c] = None  # free
        if a.validate_against:
            validate(c, a)


if __name__ == "__main__":
    main()
