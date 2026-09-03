import pandas as pd, glob, numpy as np
G="/g/data/cy94/rs4477/downstream/genome_452/asm_pip09"
files=glob.glob(G+"/*.percpg_asm.tsv")
print("samples:", len(files))
from collections import defaultdict
tested=defaultdict(int); mwu=defaultdict(int); fisher=defaultdict(int)
delta_sum=defaultdict(float); absdelta_sum=defaultdict(float)
for i,f in enumerate(files):
    d=pd.read_csv(f, sep="\t", usecols=["name","delta","p_mwu","p_fisher"])
    nm=d.name.values; dl=d.delta.values; pm=d.p_mwu.values; pf=d.p_fisher.values
    for j in range(len(nm)):
        name=nm[j]; tested[name]+=1
        if pm[j]<0.05: mwu[name]+=1
        if pf[j]<0.05: fisher[name]+=1
        if not np.isnan(dl[j]): delta_sum[name]+=dl[j]; absdelta_sum[name]+=abs(dl[j])
    if (i+1)%50==0: print("  processed", i+1, flush=True)
rows=[]
for name in tested:
    n=tested[name]
    rows.append((name, n, mwu[name], fisher[name], mwu[name]/n, fisher[name]/n,
                 delta_sum[name]/n, absdelta_sum[name]/n))
agg=pd.DataFrame(rows, columns=["name","n_tested","n_mwu","n_fisher","frac_mwu","frac_fisher","mean_delta","mean_abs_delta"])
agg.to_csv("/g/data/cy94/rs4477/downstream/genome_452/genome.asm_pip09_aggregated.tsv", sep="\t", index=False)
print("regions:", len(agg))
print("ASM (MWU) >25% samples:", int((agg.frac_mwu>0.25).sum()))
print("ASM (MWU) >10% samples:", int((agg.frac_mwu>0.1).sum()))
print("ASM (Fisher) >50% samples:", int((agg.frac_fisher>0.5).sum()))
print("ASM (Fisher) >25% samples:", int((agg.frac_fisher>0.25).sum()))
