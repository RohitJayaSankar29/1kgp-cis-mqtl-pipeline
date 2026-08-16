#!/usr/bin/env python3
"""join_asm_to_enriched.py — add the SD-ASM/imprint classification onto the
enriched variant-level shortlist, keyed by variant_id.

Adds columns: asm_label (SD-ASM/imprint/ambiguous/insufficient/none),
              asm_n_het (het carriers showing ASM), asm_delta (allele-oriented),
              asm_consistency (allele direction agreement).

Usage:
  join_asm_to_enriched.py --enriched chr22.prioritised_variant_level.enriched.txt.gz \
    --asm chr22.asm_classified.tsv --out chr22.prioritised_variant_level.enriched_asm.txt.gz
"""
import argparse
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--enriched", required=True)
ap.add_argument("--asm", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

e = pd.read_csv(a.enriched, sep="\t")
asm = pd.read_csv(a.asm, sep="\t")

# ASM is per region; key by its lead SNP. Keep the strongest region per SNP.
asm = asm[asm["snp"].astype(str).str.len() > 4].copy()
asm = asm.sort_values("n_het_asm", ascending=False).drop_duplicates("snp")
ren = {"snp": "variant_id", "label": "asm_label", "n_het_asm": "asm_n_het",
       "mean_delta_alt": "asm_delta", "allele_consistency": "asm_consistency"}
asm = asm[list(ren)].rename(columns=ren)

m = e.merge(asm, on="variant_id", how="left")
m["asm_label"] = m["asm_label"].fillna("none")
m.to_csv(a.out, sep="\t", index=False)

n_sd = int((m["asm_label"] == "SD-ASM").sum())
print(f"enriched variants: {len(m):,}   corroborated by SD-ASM: {n_sd}   -> {a.out}")
show = [c for c in ["variant_id", "nearest_gene", "ccre", "sv_implicated", "n_cpgs",
                    "pip", "score_weighted", "asm_label", "asm_n_het", "asm_consistency"]
        if c in m.columns]
print("\nvariants with SD-ASM corroboration:")
print(m[m["asm_label"] == "SD-ASM"][show].head(25).to_string(index=False))
