import pandas as pd, numpy as np
G="/g/data/cy94/rs4477/downstream/genome_452"

# 1. load enriched table + aggregated PIP>0.9 ASM
v=pd.read_csv(G+"/genome.prioritised_variant_level.enriched.txt.gz", sep="\t")
asm=pd.read_csv(G+"/genome.asm_pip09_aggregated.tsv", sep="\t")

# 2. map ASM regions (name=chrom:start, start=pos-1000) back to variants
p=v["variant_id"].str.split(":",expand=True)
v["chrom"]=p[0]; v["pos"]=p[1].astype(int)
v["asm_name"]=v.chrom+":"+(v.pos-1000).clip(lower=0).astype(str)
m=v.merge(asm[["name","n_tested","frac_mwu","frac_fisher","mean_delta","mean_abs_delta"]],
          left_on="asm_name", right_on="name", how="left")
m=m.rename(columns={"frac_mwu":"asm_frac_mwu","frac_fisher":"asm_frac_fisher",
                    "mean_abs_delta":"asm_abs_delta","n_tested":"asm_n"})
m["asm_tested"]=m["asm_n"].notna()
print("variants:", len(m), "| ASM-tested:", int(m.asm_tested.sum()))

# 3. build z-scored features for the new score
def z(s):
    s=pd.to_numeric(s, errors="coerce")
    return (s - s.mean())/s.std(ddof=0)
m["z_pip"]=z(m["pip"])
m["z_negp"]=z(-np.log10(pd.to_numeric(m["pval_nominal"],errors="coerce").clip(lower=1e-300)))
m["z_eff"]=z(m["slope"].abs())
m["z_asm"]=z(m["asm_frac_fisher"].fillna(0))   # ASM feature (Fisher fraction)
m["z_asmdelta"]=z(m["asm_abs_delta"].fillna(0))
m["z_ccre"]=z((m["ccre"].astype(str)!="none").astype(float)) if "ccre" in m else 0
m["z_sv"]=z(m["sv_implicated"].fillna(False).astype(float)) if "sv_implicated" in m else 0
m["z_ncpg"]=z(np.log1p(m["n_cpgs"]))

# 4. REBALANCED weighted score (ASM co-equal with PIP)
W={"z_pip":1.5, "z_asm":1.5, "z_negp":1.0, "z_eff":0.75, "z_ncpg":0.5,
   "z_ccre":0.75, "z_sv":0.5, "z_asmdelta":0.5}
m["score_v2"]=sum(W[c]*m[c].fillna(0) for c in W)

# 5. CONVERGENCE score (count of independent evidence axes)
m["ev_pip"]=(m["pip"]>0.9).astype(int)
m["ev_asm"]=(m["asm_frac_fisher"].fillna(0)>0.25).astype(int)
m["ev_ccre"]=(m["ccre"].astype(str)!="none").astype(int) if "ccre" in m else 0
m["ev_sv"]=m["sv_implicated"].fillna(False).astype(int) if "sv_implicated" in m else 0
m["ev_effect"]=(m["slope"].abs()>m["slope"].abs().quantile(0.75)).astype(int)
m["score_convergence"]=m[["ev_pip","ev_asm","ev_ccre","ev_sv","ev_effect"]].sum(axis=1)

# 6. save + summarize
out=m.drop(columns=["name","asm_name"], errors="ignore")
out.to_csv(G+"/genome.prioritised_final_asm_rescored.txt.gz", sep="\t", index=False)
print("saved final rescored table:", len(out), "variants")
print()
print("=== top 15 by NEW score_v2 (ASM co-equal) ===")
cols=[c for c in ["variant_id","nearest_gene","pip","asm_frac_fisher","score_weighted","score_v2","score_convergence"] if c in out.columns]
print(out.sort_values("score_v2",ascending=False).head(15)[cols].to_string(index=False))
print()
print("=== convergence score distribution ===")
print(out.score_convergence.value_counts().sort_index())
