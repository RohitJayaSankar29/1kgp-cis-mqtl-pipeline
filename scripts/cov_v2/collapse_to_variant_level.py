#!/usr/bin/env python3
"""collapse_to_variant_level.py

Fix for the duplicate-variant issue in the prioritised output: the pair-level
table ranks variant-CpG PAIRS, so a variant that regulates a cluster of nearby
CpGs appears once per CpG and floods the top of the list (the "G:A / A:C" repeats).

This collapses to ONE row per distinct candidate variant:
  - keeps the variant's best-scoring CpG as representative
  - adds n_cpgs   = how many distinct CpGs that variant is the credible-set lead for
  - adds cpg_snp  = is the variant ON the CpG with a CpG-altering change (C<->T / G<->A
                    within 1 bp) — a likely methylation-calling artefact to review/exclude

Usage:
  python3 collapse_to_variant_level.py \
      --prioritised chr22.prioritised_framework.txt.gz \
      --out chr22.prioritised_variant_level.txt.gz
"""
import argparse
import numpy as np
import pandas as pd

CPG_ALTERING = {("C", "T"), ("T", "C"), ("G", "A"), ("A", "G")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prioritised", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--score", default="score_weighted",
                    help="column to rank/representative by (default score_weighted)")
    a = ap.parse_args()

    df = pd.read_csv(a.prioritised, sep="\t")
    if a.score not in df.columns:
        a.score = df.columns[-1]

    # parse variant_id  chr:pos:ref:alt  and cpg position from phenotype_id chr22_<pos>
    parts = df["variant_id"].astype(str).str.split(":", expand=True)
    df["_vpos"] = pd.to_numeric(parts[1], errors="coerce")
    df["_ref"] = parts[2]
    df["_alt"] = parts[3]
    df["_cpos"] = pd.to_numeric(df["phenotype_id"].astype(str).str.split("_").str[-1],
                                errors="coerce")

    # CpG-SNP flag: variant within the CpG dinucleotide AND a CpG-altering transition
    on_cpg = (df["_vpos"] - df["_cpos"]).abs() <= 1
    altering = [ (r, al) in CPG_ALTERING for r, al in zip(df["_ref"], df["_alt"]) ]
    df["cpg_snp"] = on_cpg & pd.Series(altering, index=df.index)

    # rank, then take the best row per variant as its representative
    df = df.sort_values(a.score, ascending=False)
    g = df.groupby("variant_id", sort=False)
    rep = df.drop_duplicates("variant_id").copy()                 # best-scoring row per variant
    rep["n_cpgs"] = rep["variant_id"].map(g["phenotype_id"].nunique())
    rep["cpg_snp_any"] = rep["variant_id"].map(g["cpg_snp"].any())
    rep = rep.sort_values(a.score, ascending=False)

    cols = [c for c in ["variant_id", "phenotype_id", "n_cpgs", "cpg_snp_any",
                        "pip", "pval_nominal", "slope", a.score, "score_elasticnet"]
            if c in rep.columns]
    out = rep[cols].rename(columns={"phenotype_id": "best_cpg"})
    out.to_csv(a.out, sep="\t", index=False)

    print(f"pair rows in        : {len(df):,}")
    print(f"distinct variants   : {len(rep):,}")
    print(f"CpG-SNP-flagged vars: {int(rep['cpg_snp_any'].sum())}  (review/exclude these)")
    print("\nTop 15 distinct candidate variants:")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(out.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
