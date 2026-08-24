import pandas as pd, numpy as np, matplotlib, os
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False,
    "figure.facecolor":"white","savefig.dpi":150,"savefig.bbox":"tight"})
G="/g/data/cy94/rs4477/downstream/genome_452"
OUT="/scratch/cy94/rs4477/figures_meeting"; os.makedirs(OUT, exist_ok=True)
C={"blue":"#2c6fbb","green":"#2a9d54","orange":"#e07b39","grey":"#9aa0a6","red":"#c0392b","purple":"#7b52ab"}
print("loading..."); v=pd.read_csv(f"{G}/genome.prioritised_variant_level.enriched.txt.gz", sep="\t"); print("loaded",len(v))

fig,ax=plt.subplots(figsize=(7,4.5))
stages=["CpGs\ntested","Significant\nmQTLs","Lead\nvariants","Credible causal\n(PIP>0.9)"]
vals=[25912587,1638801,1019744,192172]
bars=ax.bar(stages,vals,color=[C["grey"],C["blue"],C["green"],C["orange"]])
ax.set_yscale("log"); ax.set_ylabel("count (log scale)")
for b,val in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,val,f"{val:,}",ha="center",va="bottom",fontsize=8.5)
ax.set_title("Genome-wide cis-mQTL discovery funnel (452 samples)")
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_funnel.png"); plt.close(); print("fig1")

fig,ax=plt.subplots(1,2,figsize=(11,4.2))
ax[0].hist(v.pip.dropna(),bins=50,color=C["blue"],edgecolor="white"); ax[0].set_yscale("log")
ax[0].set_xlabel("SuSiE PIP"); ax[0].set_ylabel("variants"); ax[0].set_title("Fine-mapping PIP distribution")
thr=[0.5,0.9,0.95]; cnt=[int((v.pip>t).sum()) for t in thr]
b=ax[1].bar([f"PIP>{t}" for t in thr],cnt,color=[C["grey"],C["green"],C["orange"]])
for bb,c in zip(b,cnt): ax[1].text(bb.get_x()+bb.get_width()/2,c,f"{c:,}",ha="center",va="bottom",fontsize=9)
ax[1].set_ylabel("credible causal variants"); ax[1].set_title("High-confidence causal variants")
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_pip.png"); plt.close(); print("fig2")

fig,ax=plt.subplots(1,2,figsize=(11,4.2))
cc=v.ccre.astype(str).replace("nan","none")
order=[x for x in cc.value_counts().index if x!="none"][:6]; vals=[int((cc==o).sum()) for o in order]
ax[0].barh(order[::-1],vals[::-1],color=C["green"]); ax[0].set_xlabel("variants"); ax[0].set_title("cCRE regulatory class")
in_ccre=(cc!="none"); in_isl=(v.cpg_island=="yes")
comp=[int(in_ccre.sum()),int(in_isl.sum()),int((~in_ccre&~in_isl).sum())]
b=ax[1].bar(["in cCRE","CpG island","neither"],comp,color=[C["green"],C["blue"],C["grey"]])
for bb,c in zip(b,comp): ax[1].text(bb.get_x()+bb.get_width()/2,c,f"{c:,}",ha="center",va="bottom",fontsize=9)
ax[1].set_ylabel("variants"); ax[1].set_title("Genomic context")
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_functional.png"); plt.close(); print("fig3")

fig,ax=plt.subplots(1,2,figsize=(11,4.2))
sv_yes=int(v.sv_implicated.sum()); sv_no=len(v)-sv_yes
ax[0].pie([sv_yes,sv_no],labels=[f"SV-implicated\n{sv_yes:,}\n({sv_yes/len(v)*100:.1f}%)",f"no SV\n{sv_no:,}"],
    colors=[C["orange"],C["grey"]],startangle=90,wedgeprops={"edgecolor":"white"})
ax[0].set_title("Structural variant implication")
top=v.nlargest(12,"score_weighted")
lbl=[f"{r.nearest_gene} ({r.variant_id.split(':')[0]})" for _,r in top.iterrows()]
ax[1].barh(range(len(top)),top.score_weighted.values[::-1],color=C["red"])
ax[1].set_yticks(range(len(top))); ax[1].set_yticklabels(lbl[::-1],fontsize=8)
ax[1].set_xlabel("score_weighted"); ax[1].set_title("Top 12 prioritised variants")
plt.tight_layout(); plt.savefig(f"{OUT}/fig4_sv_tophits.png"); plt.close(); print("fig4")
print("DONE ->",OUT)
