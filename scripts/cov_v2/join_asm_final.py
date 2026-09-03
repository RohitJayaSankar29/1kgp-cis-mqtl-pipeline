import pandas as pd, numpy as np
G="/g/data/cy94/rs4477/downstream/genome_452"

# 1. aggregated per-CpG ASM (per region)
asm=pd.read_csv(G+"/genome.asm_percpg_aggregated.tsv", sep="\t")

# 2. map regions back to variants (region name = chrom:region_start, region_start = pos-2000)
targ=pd.read_csv("/scratch/cy94/rs4477/asm_stream_test/asm_strat_variants.tsv", sep="\t")
p=targ["variant_id"].str.split(":", expand=True)
targ["chrom"]=p[0]; targ["pos"]=p[1].astype(int)
targ["region_start"]=(targ.pos-2000).clip(lower=0)
targ["name"]=targ.chrom+":"+targ.region_start.astype(str)
# join ASM stats onto the target variants
tv=targ.merge(asm[["name","n_tested","frac_mwu_sig","frac_fisher_sig","mean_delta","mean_abs_delta"]], on="name", how="left")
tv=tv.rename(columns={"frac_mwu_sig":"asm_frac_mwu","frac_fisher_sig":"asm_frac_fisher",
                      "mean_delta":"asm_mean_delta","mean_abs_delta":"asm_mean_abs_delta",
                      "n_tested":"asm_n_samples"})
tv["asm_tested"]=tv["asm_n_samples"].notna()
# ASM "positive" call: significant (MWU) in a meaningful fraction of samples
tv["asm_mwu_pos"]=(tv["asm_frac_mwu"]>0.1).fillna(False)
print("target variants:", len(tv), "with ASM tested:", int(tv.asm_tested.sum()))
print("ASM-positive (MWU>10% samples):", int(tv.asm_mwu_pos.sum()))

# 3. join onto the full enriched prioritised table
enr=pd.read_csv(G+"/genome.prioritised_variant_level.enriched.txt.gz", sep="\t")
asm_cols=["variant_id","asm_n_samples","asm_frac_mwu","asm_frac_fisher","asm_mean_delta","asm_mean_abs_delta","asm_tested","asm_mwu_pos","stratum"]
enr2=enr.merge(tv[asm_cols], on="variant_id", how="left")
enr2["asm_tested"]=enr2["asm_tested"].fillna(False)
enr2["asm_mwu_pos"]=enr2["asm_mwu_pos"].fillna(False)
enr2.to_csv(G+"/genome.prioritised_variant_level.enriched_asm.txt.gz", sep="\t", index=False)
print()
print("final enriched+ASM table:", len(enr2), "variants,", enr2.shape[1], "columns")
print("columns:", list(enr2.columns))
print()
print("=== top 15 prioritised variants with ASM ===")
show=[c for c in ["variant_id","n_cpgs","pip","score_weighted","nearest_gene","ccre","sv_implicated","asm_frac_mwu","asm_mean_abs_delta"] if c in enr2.columns]
print(enr2.head(15)[show].to_string(index=False))
