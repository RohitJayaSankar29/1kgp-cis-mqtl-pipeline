#!/usr/bin/env python3
"""make_figures.py — regenerate all ASM/pipeline figures from result files.
Reproducible: reads the actual result tables and writes PNGs into figures/.

Usage:
  python make_figures.py \
    --enriched results/chr22.prioritised_variant_level.enriched_asm.txt.gz \
    --nominal  <downstream>/nominal/chr22.sig.cis_qtl_pairs.chr22.parquet \
    --outdir   figures/
If --nominal is omitted, panels needing the slope/distance are skipped.
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest, spearmanr

plt.rcParams.update({"font.family":"sans-serif","font.size":11,"axes.spines.top":False,
    "axes.spines.right":False,"figure.facecolor":"white","savefig.dpi":150,"savefig.bbox":"tight"})
C={"sd":"#2E9E5B","blue":"#3B7DD8","grey":"#B8C0CC","dark":"#2A3140","amber":"#E0A020"}

def dist(df):
    df=df.copy()
    df["snp_pos"]=df.variant_id.str.split(":").str[1].astype(float)
    df["cpg_pos"]=df.best_cpg.str.split("_").str[-1].astype(float)
    df["dist"]=(df.snp_pos-df.cpg_pos).abs()
    return df

def composition_signtest(d,nom,out):
    fig,ax=plt.subplots(1,2,figsize=(11,4.2))
    vc=d.asm_label.value_counts(); order=["SD-ASM","imprint","insufficient","none"]
    vals=[int(vc.get(k,0)) for k in order]
    ax[0].bar(range(4),vals,color=[C["sd"],C["amber"],C["grey"],"#E8ECF2"])
    ax[0].set_yscale("log"); ax[0].set_xticks(range(4))
    ax[0].set_xticklabels(["SD-ASM\n(genetic)","imprint","insufficient","none"],fontsize=9)
    ax[0].set_ylabel("Variants (log scale)")
    ax[0].set_title("ASM classification of prioritised variants",fontweight="bold",fontsize=12)
    for i,v in enumerate(vals): ax[0].text(i,v*1.15,f"{v:,}",ha="center",fontweight="bold",fontsize=9)
    if nom is not None:
        sd=d[d.asm_label=="SD-ASM"][["variant_id","best_cpg","asm_delta"]].dropna()
        n=nom.rename(columns={"phenotype_id":"best_cpg"})
        m=sd.merge(n[["variant_id","best_cpg","slope"]],on=["variant_id","best_cpg"],how="left").dropna()
        m["c"]=np.sign(m.asm_delta)==np.sign(m.slope); k,N=int(m.c.sum()),len(m)
        p=binomtest(k,N,0.5).pvalue if N else float("nan")
        ax[1].pie([1,1e-6],colors=[C["sd"],"white"],startangle=90,counterclock=False,
            wedgeprops=dict(width=0.35,edgecolor="white",linewidth=2))
        ax[1].text(0,0.15,f"{k}/{N}",ha="center",fontsize=26,fontweight="bold",color=C["sd"])
        ax[1].text(0,-0.15,"concordant",ha="center",fontsize=12)
        ax[1].text(0,-0.40,"ASM vs mQTL direction",ha="center",fontsize=9,color=C["dark"])
        ax[1].text(0,-0.58,f"p = {p:.1e}",ha="center",fontsize=10,color=C["dark"])
        ax[1].set_title("Sign test: mQTLs are cis-causal",fontweight="bold",fontsize=12)
    else:
        ax[1].axis("off"); ax[1].text(0.5,0.5,"sign test skipped\n(no --nominal)",ha="center",color=C["grey"])
    plt.tight_layout(); plt.savefig(out); plt.close()

def distance(d,out):
    d=dist(d); fig,ax=plt.subplots(1,2,figsize=(11,4.2))
    sd=d[d.asm_label=="SD-ASM"].dist.dropna()
    ins=d[d.asm_label=="insufficient"].dist.replace(0,np.nan).dropna()
    b=np.logspace(1,6,40)
    ax[0].hist(sd,bins=b,color=C["sd"],alpha=0.85,label=f"SD-ASM (n={len(sd)})")
    ax[0].hist(ins,bins=b,color=C["grey"],alpha=0.6,label=f"insufficient (n={len(ins)})")
    ax[0].set_xscale("log"); ax[0].axvline(sd.median(),color=C["sd"],ls="--",lw=1.5)
    ax[0].set_xlabel("SNP\u2013CpG distance (bp, log)"); ax[0].set_ylabel("variants")
    ax[0].set_title("ASM tests only short-range pairs",fontweight="bold",fontsize=12); ax[0].legend(fontsize=9)
    med=[sd.median(),ins.median()]
    ax[1].barh(["SD-ASM","insufficient"],med,color=[C["sd"],C["grey"]])
    ax[1].set_xscale("log"); ax[1].set_xlabel("median SNP\u2013CpG distance (bp, log)")
    ax[1].set_title("Direct vs distal regulation",fontweight="bold",fontsize=12)
    for i,v in enumerate(med): ax[1].text(v*1.15,i,f"{v:,.0f} bp",va="center",fontweight="bold")
    ax[1].set_xlim(50,1e6); plt.tight_layout(); plt.savefig(out); plt.close()

def pip_distance(d,nom,out):
    if nom is None: return
    n=nom.rename(columns={"phenotype_id":"best_cpg"})
    f=d.merge(n[["variant_id","best_cpg","start_distance"]],on=["variant_id","best_cpg"],how="left")
    f["absdist"]=f.start_distance.abs(); f=f[f.absdist>0]
    fig,ax=plt.subplots(1,2,figsize=(11,4.2))
    f["bin"]=pd.cut(f.absdist,[0,1e3,1e4,1e5,1.1e6],labels=["<1kb","1\u201310kb","10\u2013100kb","100kb\u20131Mb"])
    med=f.groupby("bin",observed=True).pip.median()
    ax[0].bar(range(len(med)),med.values,color=C["blue"])
    ax[0].set_xticks(range(len(med))); ax[0].set_xticklabels(med.index,fontsize=9); ax[0].set_ylabel("median PIP")
    ax[0].set_title("PIP decreases with distance\n(not an LD artefact)",fontweight="bold",fontsize=11)
    for i,v in enumerate(med.values): ax[0].text(i,v+0.002,f"{v:.3f}",ha="center",fontsize=9)
    s=f.sample(min(3000,len(f)),random_state=1)
    ax[1].scatter(s.absdist,s.pip,s=6,alpha=0.25,color=C["blue"],edgecolors="none")
    sdd=f[f.asm_label=="SD-ASM"]
    ax[1].scatter(sdd.absdist,sdd.pip,s=40,color=C["sd"],edgecolors="white",linewidth=0.5,label="SD-ASM",zorder=5)
    rho,p=spearmanr(f.absdist,f.pip)
    ax[1].set_xscale("log"); ax[1].set_xlabel("SNP\u2013CpG distance (bp, log)"); ax[1].set_ylabel("PIP")
    ax[1].set_title("SD-ASM hits: low PIP, short range",fontweight="bold",fontsize=11)
    ax[1].text(0.05,0.9,f"Spearman \u03c1 = {rho:.2f}\np = {p:.1e}",transform=ax[1].transAxes,fontsize=9,
        bbox=dict(boxstyle="round",fc="white",ec=C["grey"])); ax[1].legend(fontsize=9,loc="upper right")
    plt.tight_layout(); plt.savefig(out); plt.close()

def complementarity(d,out):
    fig,ax=plt.subplots(figsize=(6.5,4.5))
    sd=d[d.asm_label=="SD-ASM"].pip.dropna(); top=d.nlargest(50,"score_weighted").pip
    parts=ax.violinplot([sd,top],showmedians=True,widths=0.7)
    for i,pc in enumerate(parts["bodies"]): pc.set_facecolor([C["sd"],C["blue"]][i]); pc.set_alpha(0.7)
    for kk in ["cmedians","cbars","cmins","cmaxes"]: parts[kk].set_color(C["dark"])
    nt=int(d.nlargest(50,"score_weighted").asm_label.eq("SD-ASM").sum())
    ax.set_xticks([1,2]); ax.set_xticklabels(["SD-ASM\ncorroborated","Top-50\nby score"]); ax.set_ylabel("PIP")
    ax.set_title("ASM and fine-mapping are complementary",fontweight="bold",fontsize=12)
    ax.text(1,0.15,f"median\n{sd.median():.2f}",ha="center",fontsize=9,color=C["sd"])
    ax.text(2,0.90,f"median\n{top.median():.3f}",ha="center",fontsize=9,color=C["blue"])
    ax.text(1.5,0.35,f"{nt} SD-ASM variants\nin top-50",ha="center",fontsize=9,style="italic",color=C["dark"])
    plt.tight_layout(); plt.savefig(out); plt.close()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--enriched",required=True)
    ap.add_argument("--nominal",default="")
    ap.add_argument("--outdir",default="figures")
    a=ap.parse_args()
    os.makedirs(a.outdir,exist_ok=True)
    d=pd.read_csv(a.enriched,sep="\t")
    nom=pd.read_parquet(a.nominal,columns=["phenotype_id","variant_id","slope","start_distance"]) \
        if a.nominal and os.path.exists(a.nominal) else None
    composition_signtest(d,nom,os.path.join(a.outdir,"asm_composition_signtest.png"))
    distance(d,os.path.join(a.outdir,"asm_distance.png"))
    pip_distance(d,nom,os.path.join(a.outdir,"pip_vs_distance.png"))
    complementarity(d,os.path.join(a.outdir,"asm_finemap_complementarity.png"))
    print(f"figures -> {a.outdir}/")

if __name__=="__main__":
    main()