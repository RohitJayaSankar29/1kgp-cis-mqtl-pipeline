#!/usr/bin/env python3
import argparse, os
import numpy as np, pandas as pd
def bh(p):
    p=np.asarray(p,float); n=len(p); o=np.argsort(p)
    r=p[o]*n/(np.arange(n)+1); q=np.minimum.accumulate(r[::-1])[::-1]
    out=np.empty(n); out[o]=np.clip(q,0,1); return out
ap=argparse.ArgumentParser()
ap.add_argument('--permutation',required=True); ap.add_argument('--out-dir',required=True)
ap.add_argument('--fdr',type=float,default=0.05); a=ap.parse_args()
os.makedirs(a.out_dir,exist_ok=True)
from scipy.stats import chi2
df=pd.read_csv(a.permutation,sep='\t',index_col=0)
pc='pval_beta' if 'pval_beta' in df.columns else 'pval_perm'
v=df[df[pc].notna()].copy(); p=v[pc].values
lam=np.median(chi2.isf(p,1))/chi2.ppf(0.5,1)
v['qval']=bh(p); nsig=int((v['qval']<a.fdr).sum())
print(f"CpGs tested: {len(df)} | with p-value: {len(v)}")
print(f"genomic inflation lambda: {lam:.3f}")
print(f"significant CpGs (FDR<{a.fdr}): {nsig}")
v.sort_values(pc).to_csv(f"{a.out_dir}/chr22.fdr.txt.gz",sep='\t')
v.index[v['qval']<a.fdr].to_series().to_csv(f"{a.out_dir}/chr22.significant_cpgs.txt",index=False,header=False)
try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    obs=-np.log10(np.sort(p)); exp=-np.log10((np.arange(1,len(p)+1)-0.5)/len(p))
    plt.figure(figsize=(5,5)); plt.scatter(exp,obs,s=4,alpha=0.5)
    m=max(exp.max(),obs.max()); plt.plot([0,m],[0,m],'r--',lw=1)
    plt.xlabel('Expected -log10(p)'); plt.ylabel('Observed -log10(p)')
    plt.title(f'cis-mQTL QQ  chr22  n=100  lambda={lam:.3f}')
    plt.tight_layout(); plt.savefig(f"{a.out_dir}/chr22.qq.png",dpi=150)
    print(f"QQ plot: {a.out_dir}/chr22.qq.png")
except Exception as e:
    print(f"(QQ plot skipped: {e})")
