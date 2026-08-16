#!/usr/bin/env python3
"""asm_from_haplotypes.py — call allele-specific methylation (ASM) for one sample
by comparing the two haplotype bedMethyl files from
`modkit pileup --partition-tag HP`.

For each CpG covered on both haplotypes, tests whether the methylated/unmethylated
read counts differ between haplotype 1 and haplotype 2 (Fisher's exact test), and
reports the methylation difference. This is the per-sample ASM signal.

bedMethyl columns used (modkit): 1=chrom 2=start 3=end 4=mod_code ... 10=Nvalid
11=percent_mod 12=Nmod 13=Ncanonical  (1-based).

Usage:
  python3 asm_from_haplotypes.py --hp1 hp_1.bed.gz --hp2 hp_2.bed.gz \
      --min-cov 5 --out SAMPLE.asm.tsv
"""
import argparse
import gzip
import math

from scipy.stats import fisher_exact
import pandas as pd


def read_bedmethyl(path, min_cov):
    op = gzip.open if str(path).endswith(".gz") else open
    rows = {}
    with op(path, "rt") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 13:
                continue
            chrom, start = f[0], int(f[1])
            nmod, ncanon = int(f[11]), int(f[12])
            cov = nmod + ncanon
            if cov >= min_cov:
                rows[(chrom, start)] = (nmod, ncanon)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hp1", required=True)
    ap.add_argument("--hp2", required=True)
    ap.add_argument("--min-cov", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    h1 = read_bedmethyl(a.hp1, a.min_cov)
    h2 = read_bedmethyl(a.hp2, a.min_cov)
    shared = sorted(set(h1) & set(h2))

    recs = []
    for key in shared:
        m1, c1 = h1[key]
        m2, c2 = h2[key]
        meth1 = m1 / (m1 + c1)
        meth2 = m2 / (m2 + c2)
        # Fisher exact on the 2x2 of (mod, canon) between the two haplotypes
        _, p = fisher_exact([[m1, c1], [m2, c2]])
        recs.append((key[0], key[1], m1 + c1, m2 + c2,
                     round(meth1, 4), round(meth2, 4),
                     round(meth1 - meth2, 4), p))

    df = pd.DataFrame(recs, columns=["chrom", "pos", "cov_hp1", "cov_hp2",
                                     "meth_hp1", "meth_hp2", "delta", "p"])
    # BH FDR
    if len(df):
        m = len(df)
        order = df["p"].rank(method="first").astype(int)
        df["q"] = (df["p"] * m / order).clip(upper=1.0)
        df = df.sort_values("p").reset_index(drop=True)
    df.to_csv(a.out, sep="\t", index=False)
    n_asm = int((df["q"] < 0.05).sum()) if len(df) else 0
    print(f"CpGs tested: {len(df):,}   ASM (q<0.05): {n_asm:,}   -> {a.out}")


if __name__ == "__main__":
    main()
