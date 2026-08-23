import pandas as pd, glob
G="/g/data/cy94/rs4477/downstream/genome_452"
parts=[]
for N in range(1,23):
    f=f"{G}/chr{N}/sv/chr{N}.sv_annotation.txt.gz"
    d=pd.read_csv(f, sep="\t")
    imp_col=[c for c in d.columns if "implicat" in c.lower()][0]
    cpg_col=[c for c in d.columns if "phenotype" in c.lower()][0]
    d["_imp"]=d[imp_col].astype(str).isin(["True","true","1"])
    anyimp=d.groupby(cpg_col)["_imp"].any().rename("sv_implicated")
    agg=anyimp.to_frame()
    if "svtype" in d.columns:
        types=(d[d["_imp"]].groupby(cpg_col)["svtype"]
               .agg(lambda s: ",".join(sorted(set(s.astype(str))))).rename("sv_types"))
        agg=agg.join(types)
    agg=agg.reset_index().rename(columns={cpg_col:"phenotype_id"})
    parts.append(agg)
    print(f"chr{N}: {len(agg)} CpGs, {int(agg.sv_implicated.sum())} SV-implicated", flush=True)
out=pd.concat(parts, ignore_index=True)
out.to_csv(f"{G}/genome.sv_per_cpg.tsv", sep="\t", index=False)
print("total:", len(out), "SV-implicated:", int(out.sv_implicated.sum()))
