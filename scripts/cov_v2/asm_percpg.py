#!/usr/bin/env python3
"""asm_percpg.py - per-CpG haplotype ASM from modkit hp1/hp2 pileup subsets.
Per region: MWU (hp1 per-CpG rates vs hp2 per-CpG rates) + pooled Fisher on counts.
Usage: asm_percpg.py HP1_SUBSET HP2_SUBSET REGIONS OUT [MIN_COV] [MIN_CPG]
"""
import sys, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, fisher_exact
hp1f, hp2f, regf, out = sys.argv[1:5]
min_cov = int(sys.argv[5]) if len(sys.argv) > 5 else 5
min_cpg = int(sys.argv[6]) if len(sys.argv) > 6 else 3
# modkit bedmethyl cols: chrom start end mod score strand tstart tend color coverage pct_mod N_mod N_canon ...
cols = ["chrom","start","end","mod","score","strand","ts","te","color","cov","pct","nmod","ncanon"]
def load(f):
    d = pd.read_csv(f, sep="\t", header=None, usecols=range(13), names=cols)
    d = d[d["cov"] >= min_cov].copy()
    d["rate"] = d["nmod"] / d["cov"]
    return d[["chrom","start","cov","nmod","rate"]]
h1 = load(hp1f); h2 = load(hp2f)
reg = pd.read_csv(regf, sep="\t", header=None, names=["chrom","start","end","name"] if pd.read_csv(regf,sep="\t",nrows=1,header=None).shape[1]>3 else ["chrom","start","end"])
if "name" not in reg.columns: reg["name"] = reg["chrom"].astype(str)+":"+reg["start"].astype(str)
reg = reg.sort_values(["chrom","start"]).reset_index(drop=True)
# assign each CpG to a region (chrom-aware)
def assign(d):
    d = d.copy(); d["region"] = -1
    for ch,g in reg.groupby("chrom"):
        st=g["start"].values; en=g["end"].values; gi=g.index.values
        m=(d["chrom"]==ch).values
        if not m.any(): continue
        pos=d.loc[m,"start"].values
        j=np.searchsorted(st,pos,side="right")-1
        ok=(j>=0)&(j<len(st)); ok2=np.zeros(len(pos),bool); ok2[ok]=pos[ok]<en[j[ok]]
        idx=np.where(m)[0]; ri=np.full(len(pos),-1); ri[ok2]=gi[j[ok2]]
        d.iloc[idx, d.columns.get_loc("region")]=ri
    return d[d["region"]>=0]
h1=assign(h1); h2=assign(h2)
rows=[]
for rgi in sorted(set(h1["region"]).union(h2["region"])):
    a=h1[h1["region"]==rgi]; b=h2[h2["region"]==rgi]
    # match CpGs by position for pairing / pooled counts
    if len(a)<min_cpg or len(b)<min_cpg: continue
    # MWU across per-CpG rates
    try:
        _,p_mwu=mannwhitneyu(a["rate"].values,b["rate"].values,alternative="two-sided")
    except ValueError: p_mwu=1.0
    if np.isnan(p_mwu): p_mwu=1.0
    # pooled Fisher on summed counts
    m1=int(a["nmod"].sum()); u1=int((a["cov"]-a["nmod"]).sum())
    m2=int(b["nmod"].sum()); u2=int((b["cov"]-b["nmod"]).sum())
    try: _,p_fisher=fisher_exact([[u1,m1],[u2,m2]])
    except: p_fisher=np.nan
    r=reg.iloc[rgi]
    rows.append((r["chrom"],r["start"],r["end"],r["name"],len(a),len(b),
                 round(a["rate"].mean(),3),round(b["rate"].mean(),3),
                 round(a["rate"].mean()-b["rate"].mean(),3),m1,u1,m2,u2,p_mwu,p_fisher))
o=pd.DataFrame(rows,columns=["chrom","start","end","name","ncpg_hp1","ncpg_hp2",
    "mean_hp1","mean_hp2","delta","M_hp1","U_hp1","M_hp2","U_hp2","p_mwu","p_fisher"])
o.to_csv(out,sep="\t",index=False)
print(f"regions tested: {len(o)}, MWU p<0.05: {(o.p_mwu<0.05).sum()}, Fisher p<0.05: {(o.p_fisher<0.05).sum()}")
