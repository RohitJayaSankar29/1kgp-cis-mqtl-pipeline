import pandas as pd, re
B="/scratch/cy94/rs4477"; C=f"{B}/1kgp-cis-mqtl/config"
ped=pd.read_csv(f"{C}/ped_3202.txt",sep=r"\s+")
print("cols:",list(ped.columns))
f=[c for c in ped.columns if "ather" in c][0]; m=[c for c in ped.columns if "other" in c][0]
s=[c for c in ped.columns if c.lower() in ("sampleid","id","sample")][0]
unrel=set(ped.loc[(ped[f].astype(str)=="0")&(ped[m].astype(str)=="0"),s])
print("unrelated founders:",len(unrel))
rows=[]
for L in open(f"{B}/modkit_roster.txt"):
  n=L.strip().rstrip("/")
  if not n: continue
  sm=n.split("-")[0]; ch="R10" if "-R10-" in n else "R9"
  g="guppy" if "guppy" in n else ("dorado_late" if re.search(r"dorado0?[89]",n) else ("dorado_mid" if re.search(r"dorado0?[35]",n) else "dorado_other"))
  rows.append((sm,ch,g,n))
r=pd.DataFrame(rows,columns=["sample","chemistry","basecaller","folder"]).drop_duplicates("sample")
print("roster:",len(r))
r["unrel"]=r["sample"].isin(unrel); co=r[r.unrel].copy()
print("COHORT:",len(co)); print("chem:",co.chemistry.value_counts().to_dict()); print("basecaller:",co.basecaller.value_counts().to_dict()); print("dropped related:",(~r.unrel).sum())
co[["sample","chemistry","basecaller","folder"]].to_csv(f"{C}/cohort_unrelated.tsv",sep="\t",index=False)
print("wrote cohort_unrelated.tsv")
