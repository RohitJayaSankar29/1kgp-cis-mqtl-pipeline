#!/usr/bin/env python3
"""asm_percpg_fast.py - vectorized per-CpG haplotype ASM. Same output as asm_percpg.py
but processes all regions via groupby (fast at 10k+ regions).
Usage: asm_percpg_fast.py HP1_SUBSET HP2_SUBSET REGIONS OUT [MIN_COV] [MIN_CPG]"""
import sys, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, fisher_exact
hp1f, hp2f, regf, out = sys.argv[1:5]
min_cov = int(sys.argv[5]) if len(sys.argv) > 5 else 5
min_cpg = int(sys.argv[6]) if len(sys.argv) > 6 else 3
cols = ["chrom","start","end","mod","score","strand","ts","te","color","cov","pct","nmod","ncanon"]
def load(f):
    d = pd.read_csv(f, sep="\t", header=None, usecols=range(13), names=cols)
    d = d[d["cov"] >= min_cov].copy()
    d["rate"] = d["nmod"]/d["cov"]
    return d[["chrom","start","cov","nmod","rate"]]
h1 = load(hp1f); h2 = load(hp2f)
reg = pd.read_csv(regf, sep="\t", header=None).iloc[:,:3]
reg.columns=["chrom","start","end"]
reg["name"]=reg["chrom"].astype(str)+":"+reg["start"].astype(str)
reg = reg.sort_values(["chrom","start"]).reset_index(drop=True)
def assign(d):
    d=d.copy(); d["region"]=-1
    for ch,g in reg.groupby("chrom"):
        st=g["start"].values; en=g["end"].values; gi=g.index.values
        m=(d["chrom"]==ch).values
        if not m.any(): continue
        pos=d.loc[m,"start"].values
        j=np.searchsorted(st,pos,side="right")-1
        ok=(j>=0)&(j<len(st)); ok2=np.zeros(len(pos),bool); ok2[ok]=pos[ok]<en[j[ok]]
        ri=np.full(len(pos),-1); ri[ok2]=gi[j[ok2]]
        idx=np.where(m)[0]; d.iloc[idx, d.columns.get_loc("region")]=ri
    return d[d["region"]>=0]
h1=assign(h1); h2=assign(h2)
# vectorized per-region aggregates
def agg(d):
    g=d.groupby("region")
    return pd.DataFrame({"ncpg":g.size(),"nmod":g["nmod"].sum(),"cov":g["cov"].sum(),
                         "rates":g["rate"].apply(list)})
a1=agg(h1); a2=agg(h2)
common=a1.index.intersection(a2.index)
rows=[]
for rgi in common:
    r1=a1.loc[rgi]; r2=a2.loc[rgi]
    if r1["ncpg"]<min_cpg or r2["ncpg"]<min_cpg: continue
    v1=np.array(r1["rates"]); v2=np.array(r2["rates"])
    try:
        _,p_mwu=mannwhitneyu(v1,v2,alternative="two-sided")
    except ValueError: p_mwu=1.0
    if np.isnan(p_mwu): p_mwu=1.0
    m1=int(r1["nmod"]); u1=int(r1["cov"]-r1["nmod"])
    m2=int(r2["nmod"]); u2=int(r2["cov"]-r2["nmod"])
    try: _,p_fisher=fisher_exact([[u1,m1],[u2,m2]])
    except: p_fisher=np.nan
    rn=reg.loc[rgi]
    rows.append((rn["chrom"],rn["start"],rn["end"],rn["name"],int(r1["ncpg"]),int(r2["ncpg"]),
                 round(v1.mean(),3),round(v2.mean(),3),round(v1.mean()-v2.mean(),3),
                 m1,u1,m2,u2,p_mwu,p_fisher))
o=pd.DataFrame(rows,columns=["chrom","start","end","name","ncpg_hp1","ncpg_hp2",
    "mean_hp1","mean_hp2","delta","M_hp1","U_hp1","M_hp2","U_hp2","p_mwu","p_fisher"])
o.to_csv(out,sep="\t",index=False)
print(f"regions tested: {len(o)}, MWU p<0.05: {(o.p_mwu<0.05).sum()}, Fisher p<0.05: {(o.p_fisher<0.05).sum()}")
