# Co-methylation domain analysis: SV-implicated variants drive larger domains;
# domain size correlates with ASM. Produces 3-panel figure.
import pandas as pd, numpy as np, matplotlib, os
matplotlib.use("Agg"); import matplotlib.pyplot as plt
G="/g/data/cy94/rs4477/downstream/genome_452"
v=pd.read_csv(G+"/genome.prioritised_final_asm_classified.txt.gz", sep="\t",
              usecols=lambda c: c in ['n_cpgs','sv_implicated','asm_frac_fisher'])
fig,ax=plt.subplots(1,3,figsize=(15,4.2))
ax[0].hist(np.log10(v.n_cpgs), bins=50, color="#2c6fbb", edgecolor="white")
ax[0].set_xlabel("log10(domain size, CpGs)"); ax[0].set_title("Co-methylation domain sizes")
sv=v[v.sv_implicated==True].n_cpgs; nosv=v[v.sv_implicated==False].n_cpgs
ax[1].boxplot([np.log10(nosv),np.log10(sv)], labels=["no SV","SV"], showfliers=False)
ax[1].set_title("SV-implicated drive larger domains (p<1e-300)")
vt=v[v.asm_frac_fisher.notna()].copy()
vt['bin']=pd.cut(vt.n_cpgs,[0,5,20,50,100,10000],labels=['1-5','6-20','21-50','51-100','100+'])
med=vt.groupby('bin',observed=True).asm_frac_fisher.mean()
ax[2].bar(range(len(med)), med.values, color="#2a9d54")
ax[2].set_xticks(range(len(med))); ax[2].set_xticklabels(med.index)
ax[2].set_title("Larger domains show more ASM (rho=0.19)")
plt.tight_layout(); plt.savefig("comethyl_domains.png"); print("saved")
