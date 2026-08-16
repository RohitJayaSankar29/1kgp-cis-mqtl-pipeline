#!/usr/bin/env python3
"""asm_aggregate.py — corroborate cis-mQTL candidates with allele-specific
methylation across samples.

Genetic ASM logic: if a SNP causally drives methylation at a nearby CpG, then
HETEROZYGOUS carriers should show a large methylation difference between their two
haplotypes at that CpG, while HOMOZYGOUS individuals should not (their two
haplotypes carry the same allele). So for each candidate (variant, CpG) we compare
the per-sample allelic methylation difference |delta| between het and hom carriers.
A one-sided Mann-Whitney (het > hom) gives the ASM support p-value.

This is the within-individual corroboration column for the prioritised shortlist.

Inputs:
  --asm-glob    per-sample ASM tables from asm_from_haplotypes.py (chrom,pos,delta)
  --candidates  TSV with columns: variant_id, chrom, cpg_pos
  --genotypes   TSV: variant_id + one column per sample, values 0/1/2 (het=1)
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asm-glob", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--genotypes", required=True)
    ap.add_argument("--sample-from", default="dir", choices=["dir", "file"])
    ap.add_argument("--min-het", type=int, default=3)
    ap.add_argument("--min-delta", type=float, default=0.20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    # per-sample ASM -> {sample: {(chrom,pos): |delta|}}
    asm = {}
    for f in glob.glob(a.asm_glob):
        sample = (os.path.basename(os.path.dirname(f)) if a.sample_from == "dir"
                  else os.path.basename(f).split(".")[0])
        d = pd.read_csv(f, sep="\t")
        asm[sample] = {(r.chrom, int(r.pos)): abs(r.delta) for r in d.itertuples()}

    geno = pd.read_csv(a.genotypes, sep="\t").set_index("variant_id")
    cand = pd.read_csv(a.candidates, sep="\t")

    rows = []
    for r in cand.itertuples():
        vid = r.variant_id
        key = (r.chrom, int(r.cpg_pos))
        if vid not in geno.index:
            rows.append((vid, r.chrom, r.cpg_pos, 0, 0, np.nan, np.nan, np.nan, False))
            continue
        g = geno.loc[vid]
        het_d = [asm[s][key] for s in g.index
                 if g[s] == 1 and s in asm and key in asm[s]]
        hom_d = [asm[s][key] for s in g.index
                 if g[s] in (0, 2) and s in asm and key in asm[s]]
        p = np.nan
        if len(het_d) >= a.min_het and len(hom_d) >= 1:
            try:
                p = mannwhitneyu(het_d, hom_d, alternative="greater").pvalue
            except ValueError:
                p = np.nan
        supported = (len(het_d) >= a.min_het
                     and (np.mean(het_d) if het_d else 0) >= a.min_delta
                     and p == p and p < 0.05)
        rows.append((vid, r.chrom, r.cpg_pos, len(het_d), len(hom_d),
                     round(np.mean(het_d), 4) if het_d else np.nan,
                     round(np.mean(hom_d), 4) if hom_d else np.nan, p, bool(supported)))

    res = pd.DataFrame(rows, columns=["variant_id", "chrom", "cpg_pos", "n_het",
                                      "n_hom", "mean_abs_delta_het",
                                      "mean_abs_delta_hom", "asm_p", "asm_supported"])
    res.to_csv(a.out, sep="\t", index=False)
    print(f"candidates: {len(res)}   ASM-supported: {int(res['asm_supported'].sum())}"
          f"   -> {a.out}")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
