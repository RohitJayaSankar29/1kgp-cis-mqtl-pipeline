import sys, pandas as pd, numpy as np
from pyliftover import LiftOver
# args: CpG_id  chrom  parquet  gwas_gz  out_prefix
cpg, chrom, parquet, gwas, outp = sys.argv[1:6]
lo = LiftOver('hg38','hg19')

# 1. mQTL stats for this CpG's cis-variants
import pyarrow.parquet as pq
pf = pq.ParquetFile(parquet)
frames=[]
for i in range(pf.num_row_groups):
    t=pf.read_row_group(i, columns=['phenotype_id','variant_id','af','pval_nominal','slope','slope_se']).to_pandas()
    frames.append(t[t.phenotype_id==cpg])
m=pd.concat(frames, ignore_index=True)
print(f"mQTL variants for {cpg}: {len(m)}")
if len(m)==0: sys.exit("no mQTL data for CpG")
# parse hg38 pos + liftover to hg19
p=m.variant_id.str.split(':',expand=True)
m['chrom']=p[0]; m['pos38']=p[1].astype(int); m['ref']=p[2]; m['alt']=p[3]
def lift(c,pos):
    r=lo.convert_coordinate(c,pos-1)
    return r[0][1]+1 if r else np.nan
m['pos19']=[lift(c,pp) for c,pp in zip(m.chrom,m.pos38)]
m=m.dropna(subset=['pos19']); m['pos19']=m.pos19.astype(int)
m['maf']=np.where(m.af>0.5,1-m.af,m.af)

# 2. eGFR GWAS for this region (hg19)
lo_p, hi_p = m.pos19.min()-1000, m.pos19.max()+1000
chrn=chrom.replace('chr','')
g=pd.read_csv(gwas, sep='\t', compression='gzip',
    usecols=['chr','pos','Allele1','Allele2','Freq1','Effect','StdErr','n','RSID'])
g=g[(g.chr.astype(str)==chrn)&(g.pos>=lo_p)&(g.pos<=hi_p)].copy()
print(f"eGFR variants in region: {len(g)}")

# 3. match by pos19 + alleles
g['key']=g.chr.astype(str)+':'+g.pos.astype(str)
m['key']=chrn+':'+m.pos19.astype(str)
merged=m.merge(g, on='key', suffixes=('_m','_g'))
# allele match (mQTL ref/alt vs GWAS Allele1/Allele2, case-insensitive)
def align(r):
    a1,a2=r.Allele1.upper(),r.Allele2.upper()
    if {r.ref,r.alt}=={a1,a2}: return 1 if r.alt==a1 else -1
    return 0
merged['sign']=merged.apply(align,axis=1)
merged=merged[merged.sign!=0].copy()
print(f"matched variants (mQTL & eGFR, alleles align): {len(merged)}")
merged.to_csv(f"{outp}.matched.tsv", sep='\t', index=False)
# write coloc input
out=pd.DataFrame({
  'snp':merged.key,
  'mqtl_beta':merged.slope, 'mqtl_se':merged.slope_se, 'mqtl_maf':merged.maf, 'mqtl_n':452,
  'gwas_beta':merged.Effect*merged.sign, 'gwas_se':merged.StdErr, 'gwas_maf':np.where(merged.Freq1>0.5,1-merged.Freq1,merged.Freq1), 'gwas_n':merged.n
})
out=out.dropna()
out.to_csv(f"{outp}.coloc_input.tsv", sep='\t', index=False)
print(f"coloc-ready variants: {len(out)} -> {outp}.coloc_input.tsv")
