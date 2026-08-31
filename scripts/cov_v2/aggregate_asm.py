import pandas as pd, glob, numpy as np
G="/g/data/cy94/rs4477/downstream/genome_452/asm_percpg"
files=glob.glob(G+"/*.percpg_asm.tsv")
print("aggregating", len(files), "samples...")
frames=[]
for f in files:
    d=pd.read_csv(f, sep="\t", usecols=["chrom","start","end","name","delta","p_mwu","p_fisher"])
    frames.append(d)
alld=pd.concat(frames, ignore_index=True)
alld["sig_mwu"]=(alld.p_mwu<0.05).astype(int)
alld["sig_fisher"]=(alld.p_fisher<0.05).astype(int)
agg=alld.groupby("name").agg(
    n_tested=("name","size"),
    n_mwu_sig=("sig_mwu","sum"),
    n_fisher_sig=("sig_fisher","sum"),
    mean_delta=("delta","mean"),
    mean_abs_delta=("delta", lambda x: np.mean(np.abs(x)))
).reset_index()
agg["frac_mwu_sig"]=agg.n_mwu_sig/agg.n_tested
agg["frac_fisher_sig"]=agg.n_fisher_sig/agg.n_tested
agg.to_csv("/g/data/cy94/rs4477/downstream/genome_452/genome.asm_percpg_aggregated.tsv", sep="\t", index=False)
print("regions:", len(agg))
print("ASM (MWU) in >50% samples:", int((agg.frac_mwu_sig>0.5).sum()))
print("ASM (MWU) in >25% samples:", int((agg.frac_mwu_sig>0.25).sum()))
print("ASM (Fisher) in >50% samples:", int((agg.frac_fisher_sig>0.5).sum()))
print()
print("top 10 most consistent ASM regions:")
print(agg.sort_values("frac_mwu_sig", ascending=False).head(10)[["name","n_tested","frac_mwu_sig","mean_delta"]].to_string(index=False))
