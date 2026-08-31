import pandas as pd, numpy as np
G="/g/data/cy94/rs4477/downstream/genome_452"
a=pd.read_csv(G+"/genome.asm_percpg_aggregated.tsv", sep="\t")
targ=pd.read_csv("/scratch/cy94/rs4477/asm_stream_test/asm_strat_variants.tsv", sep="\t")
parts=targ["variant_id"].str.split(":", expand=True)
targ["chrom"]=parts[0]
targ["pos"]=parts[1].astype(int)
targ["region_start"]=(targ.pos-2000).clip(lower=0)
targ["name"]=targ.chrom+":"+targ.region_start.astype(str)
m=a.merge(targ[["name","pip","stratum","nearest_gene","neglogp"]], on="name", how="inner")
print("matched regions:", len(m), "of", len(a), "aggregated /", len(targ), "targets")
print()
order=["PIP>0.9","PIP_0.5-0.9","PIP_0.1-0.5","PIP<0.1"]
print("=== ASM by PIP stratum ===")
s=m.groupby("stratum").agg(
    n=("name","size"),
    median_frac_mwu=("frac_mwu_sig","median"),
    mean_frac_mwu=("frac_mwu_sig","mean"),
    pct_asm25=("frac_mwu_sig", lambda x: round((x>0.25).mean(),3)),
    med_neglogp=("neglogp","median")
).reindex(order)
print(s.to_string())
print()
from scipy.stats import kruskal
groups=[m[m.stratum==st].frac_mwu_sig.dropna().values for st in order]
groups=[g for g in groups if len(g)>0]
if len(groups)>1:
    h,p=kruskal(*groups)
    print(f"Kruskal-Wallis: H={h:.2f}, p={p:.2e}")

print()
print("=== same, Fisher-based ===")
sf=m.groupby("stratum").agg(
    median_frac_fisher=("frac_fisher_sig","median"),
    pct_fisher50=("frac_fisher_sig", lambda x: round((x>0.5).mean(),3))
).reindex(order)
print(sf.to_string())
m.to_csv(G+"/genome.asm_by_stratum.tsv", sep="\t", index=False)
print("saved -> genome.asm_by_stratum.tsv")
