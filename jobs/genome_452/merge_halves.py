import sys, glob, pandas as pd
c = sys.argv[1]
base = f"/g/data/cy94/rs4477/downstream/genome_452/{c}/nominal"
fs = sorted(glob.glob(f"{base}/{c}.sig.half*.cis_qtl_pairs.{c}.parquet"))
out = f"{base}/{c}.sig.merged.parquet"
pd.concat([pd.read_parquet(f) for f in fs]).to_parquet(out)
print(f"{c} merged {len(fs)} halves -> {out}")
