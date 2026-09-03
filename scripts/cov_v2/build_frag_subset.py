import pandas as pd, numpy as np
G="/g/data/cy94/rs4477/downstream/genome_452"
targ=pd.read_csv("/scratch/cy94/rs4477/asm_stream_test/asm_strat_variants.tsv", sep="\t")
parts=targ["variant_id"].str.split(":", expand=True)
targ["chrom"]=parts[0]; targ["pos"]=parts[1].astype(int)
# sample 150 per stratum (subset of the 750 already stratified+matched)
order=["PIP>0.9","PIP_0.5-0.9","PIP_0.1-0.5","PIP<0.1"]
sub=[]
for st in order:
    s=targ[targ.stratum==st].sample(min(150,len(targ[targ.stratum==st])), random_state=1)
    sub.append(s)
sub=pd.concat(sub)
sub["start"]=(sub.pos-2000).clip(lower=0); sub["end"]=sub.pos+2000
sub[["chrom","start","end"]].sort_values(["chrom","start"]).to_csv("/scratch/cy94/rs4477/asm_stream_test/frag_subset_raw.bed", sep="\t", header=False, index=False)
sub[["variant_id","chrom","pos","pip","stratum","nearest_gene","neglogp"]].to_csv("/scratch/cy94/rs4477/asm_stream_test/frag_subset_variants.tsv", sep="\t", index=False)
print("subset:", len(sub), "variants")
print(sub.groupby("stratum").size())
