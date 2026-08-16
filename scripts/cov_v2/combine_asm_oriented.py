#!/usr/bin/env python3
"""combine_asm_oriented.py — allele-oriented cross-sample ASM classification.

Fixes the arbitrary-haplotype-label problem: instead of "was HP1 more methylated?"
(HP1/HP2 are random per sample), it asks "was the ALT allele more methylated?"
using each sample's PHASED genotype. Then classifies each region:

  SD-ASM   : het-carrier ASM with a CONSISTENT allele direction (genetic)
  imprint  : ASM also appears in HOMOZYGOTES (genetics can't cause that)
  ambiguous: het ASM but inconsistent allele direction
  insufficient : too few samples

Inputs:
  --asm-glob      per-sample asm_regions.tsv (name, meth_hp1, meth_hp2, q, ...)
  --regions       regions.bed (chrom,start,end,name)
  --variant-table variant-level shortlist (variant_id, best_cpg, score_weighted)
                  used to map each region -> its lead cis-mQTL SNP
  --geno          phased genotype matrix: col1 variant_id, then per-sample GT ("0|1")
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asm-glob", required=True)
    ap.add_argument("--regions", required=True)
    ap.add_argument("--variant-table", required=True)
    ap.add_argument("--geno", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-het", type=int, default=3)
    ap.add_argument("--min-hom-imprint", type=int, default=3)
    ap.add_argument("--q", type=float, default=0.05)
    a = ap.parse_args()

    # 1. region -> lead SNP (highest-scoring candidate whose CpG falls in the region)
    reg = pd.read_csv(a.regions, sep="\t", header=None,
                      names=["chrom", "start", "end", "name"])
    vt = pd.read_csv(a.variant_table, sep="\t")
    vt["cpg_pos"] = pd.to_numeric(vt["best_cpg"].astype(str).str.split("_").str[-1],
                                  errors="coerce")
    scol = "score_weighted" if "score_weighted" in vt.columns else vt.columns[-1]
    reg_snp = {}
    for r in reg.itertuples():
        cand = vt[(vt["cpg_pos"] >= r.start) & (vt["cpg_pos"] < r.end)]
        if len(cand):
            reg_snp[r.name] = cand.sort_values(scol, ascending=False).iloc[0]["variant_id"]
    print(f"regions mapped to a SNP: {len(reg_snp)}/{len(reg)}")

    # 2. phased genotypes: variant_id -> {sample: 'a|b'}
    g = pd.read_csv(a.geno, sep="\t").set_index("variant_id")
    gsamples = set(g.columns)

    # 3. per sample: orient each region's delta by the ALT allele
    recs = {}
    for f in glob.glob(a.asm_glob):
        S = os.path.basename(f).split(".")[0]
        if S not in gsamples:
            continue
        d = pd.read_csv(f, sep="\t")
        for row in d.itertuples():
            snp = reg_snp.get(row.name)
            if snp is None or snp not in g.index:
                continue
            gt = str(g.at[snp, S])
            if "|" not in gt:
                continue
            aa, bb = gt.split("|")[:2]
            if aa in (".", "") or bb in (".", ""):
                continue
            h1, h2 = int(aa), int(bb)
            het = (h1 != h2)
            delta_hp = row.meth_hp1 - row.meth_hp2
            # orient so + means the ALT allele (1) is more methylated
            delta_alt = delta_hp if h1 == 1 else -delta_hp
            recs.setdefault(row.name, []).append((S, het, delta_alt, row.q < a.q))

    # 4. classify
    out = []
    for r in reg.itertuples():
        lst = recs.get(r.name, [])
        het_sig = [x for x in lst if x[1] and x[3]]
        hom_sig = [x for x in lst if (not x[1]) and x[3]]
        n_het, n_hom = len(het_sig), len(hom_sig)
        if het_sig:
            signs = np.sign([x[2] for x in het_sig])
            maj = 1 if (signs >= 0).sum() >= (signs < 0).sum() else -1
            cons = float((signs == maj).mean())
            md = float(np.mean([x[2] for x in het_sig]))
        else:
            cons, md = np.nan, np.nan
        strong_het = (n_het >= a.min_het and cons >= 0.8)
        if strong_het:
            label = "SD-ASM"
        elif n_hom >= a.min_hom_imprint and n_het < a.min_het:
            label = "imprint"
        elif n_het >= a.min_het:
            label = "ambiguous"
        else:
            label = "insufficient"
        out.append((r.chrom, r.start, r.end, r.name, reg_snp.get(r.name, ""),
                    n_het, n_hom,
                    round(md, 3) if md == md else np.nan,
                    round(cons, 3) if cons == cons else np.nan, label))

    res = pd.DataFrame(out, columns=["chrom", "start", "end", "name", "snp",
                                     "n_het_asm", "n_hom_asm", "mean_delta_alt",
                                     "allele_consistency", "label"])
    res = res.sort_values(["label", "n_het_asm"], ascending=[True, False]).reset_index(drop=True)
    res.to_csv(a.out, sep="\t", index=False)
    print(res["label"].value_counts().to_dict())
    print("\nSD-ASM regions (genetic, allele-consistent):")
    print(res[res.label == "SD-ASM"].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
