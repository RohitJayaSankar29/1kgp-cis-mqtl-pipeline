#!/usr/bin/env python3
"""combine_asm.py — combine per-sample ASM region tables into one cohort summary.

For each region, aggregates across all samples:
  n_tested        samples where the region had enough fragments to test
  n_asm05/n_asm01 samples significant at q<0.05 / q<0.01
  mean_delta      mean allelic methylation difference among ASM-significant samples
  dir_consistency fraction of ASM samples sharing the majority delta sign
  label           preliminary call:
                    SD-ASM?    n_asm01>=MIN and dir_consistency>=0.8  (genetic-like)
                    imprint?   n_asm01>=MIN and dir_consistency< 0.8  (direction switches)
                    (else)     too few samples to call
Note: this uses cross-sample DIRECTION CONSISTENCY only. The full SD-ASM vs
imprinting split additionally checks ASM appears in het carriers but not
homozygotes -- that needs genotypes and is done in the next step.

Usage: combine_asm.py --glob 'asm/*/*.asm_regions.tsv' --out chr22.asm_combined.tsv
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-samples", type=int, default=3)
    ap.add_argument("--q-asm", type=float, default=0.01)
    a = ap.parse_args()

    files = glob.glob(a.glob)
    print(f"combining {len(files)} sample files")
    frames = []
    for f in files:
        s = os.path.basename(f).split(".")[0]
        d = pd.read_csv(f, sep="\t")
        if len(d):
            d["sample"] = s
            frames.append(d)
    alld = pd.concat(frames, ignore_index=True)
    alld["sig05"] = alld["q"] < 0.05
    alld["sig01"] = alld["q"] < a.q_asm

    rows = []
    for name, g in alld.groupby("name"):
        n_tested = len(g)
        sig = g[g["sig01"]]
        n05 = int(g["sig05"].sum())
        n01 = int(g["sig01"].sum())
        if len(sig):
            signs = np.sign(sig["delta"])
            maj = 1 if (signs >= 0).sum() >= (signs < 0).sum() else -1
            consistency = float((signs == maj).mean())
            mean_delta = float(sig["delta"].mean())
        else:
            consistency, mean_delta = np.nan, np.nan
        label = "insufficient"
        if n01 >= a.min_samples:
            label = "SD-ASM?" if consistency >= 0.8 else "imprint?"
        r0 = g.iloc[0]
        rows.append((r0["chrom"], int(r0["start"]), int(r0["end"]), name,
                     n_tested, n05, n01, round(mean_delta, 3) if mean_delta == mean_delta else np.nan,
                     round(consistency, 3) if consistency == consistency else np.nan, label))

    out = pd.DataFrame(rows, columns=["chrom", "start", "end", "name", "n_tested",
                                      "n_asm05", "n_asm01", "mean_delta",
                                      "dir_consistency", "label"])
    out = out.sort_values(["n_asm01", "n_asm05"], ascending=False).reset_index(drop=True)
    out.to_csv(a.out, sep="\t", index=False)
    vc = out["label"].value_counts().to_dict()
    print(f"regions: {len(out):,}   {vc}   -> {a.out}")
    print("\ntop 15 by recurrence:")
    print(out.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
