#!/usr/bin/env python3
"""enrich_variant_table.py

Add functional + structural-variant backing columns to the variant-level
shortlist, joined on the representative CpG (best_cpg). Turns the ranked table
into a self-contained evidence table: each candidate variant carries its
statistical rank AND its biological context.

Adds (where available):
  nearest_gene, dist_to_tss, cpg_island, ccre   (from functional annotation)
  sv_implicated, sv_types                        (from SV integration)

Usage:
  python3 enrich_variant_table.py \
    --variant-table chr22.prioritised_variant_level.txt.gz \
    --functional    chr22.functional.txt.gz \
    --sv            chr22.sv_annotation.txt.gz \
    --out           chr22.prioritised_variant_level.enriched.txt.gz
"""
import argparse
import pandas as pd


def keycol(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant-table", required=True)
    ap.add_argument("--functional")
    ap.add_argument("--sv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    v = pd.read_csv(a.variant_table, sep="\t")
    cpgcol = keycol(v, ["best_cpg", "phenotype_id", "cpg"])

    # functional context
    if a.functional:
        fa = pd.read_csv(a.functional, sep="\t")
        fk = keycol(fa, ["cpg", "phenotype_id"])
        keep = [c for c in ["nearest_gene", "dist_to_tss", "cpg_island", "ccre"]
                if c in fa.columns]
        if fk and keep:
            fa = fa[[fk] + keep].drop_duplicates(fk).rename(columns={fk: cpgcol})
            v = v.merge(fa, on=cpgcol, how="left")

    # SV backing
    if a.sv:
        sv = pd.read_csv(a.sv, sep="\t")
        sk = keycol(sv, ["phenotype_id", "cpg"])
        if sk and "sv_implicated" in sv.columns:
            sv["_imp"] = sv["sv_implicated"].astype(str).isin(["True", "true", "1"])
            anyimp = sv.groupby(sk)["_imp"].any().rename("sv_implicated")
            agg = anyimp.to_frame()
            if "svtype" in sv.columns:
                types = (sv[sv["_imp"]].groupby(sk)["svtype"]
                         .agg(lambda s: ",".join(sorted(set(s.astype(str)))))
                         .rename("sv_types"))
                agg = agg.join(types)
            agg = agg.reset_index().rename(columns={sk: cpgcol})
            v = v.merge(agg, on=cpgcol, how="left")
            v["sv_implicated"] = v["sv_implicated"].fillna(False)

    v.to_csv(a.out, sep="\t", index=False)
    print(f"enriched {len(v):,} variants -> {a.out}")
    show = [c for c in ["variant_id", "best_cpg", "n_cpgs", "cpg_snp_any",
                        "nearest_gene", "dist_to_tss", "cpg_island", "ccre",
                        "sv_implicated", "sv_types", "pip", "score_weighted"]
            if c in v.columns]
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(v[show].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
