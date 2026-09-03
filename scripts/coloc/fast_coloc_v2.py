import pandas as pd, numpy as np, os, sys
from pyliftover import LiftOver
import pyarrow.parquet as pq
lo=LiftOver('hg38','hg19')
G="/g/data/cy94/rs4477/downstream/genome_452"
CHROM=sys.argv[1]  # process ONE chromosome per job

def coloc_abf(b1,v1,b2,v2,p1=1e-4,p2=1e-4,p12=1e-5):
    def labf(b,v,W=0.15**2):
        r=W/(W+v); return 0.5*np.log(1-r)+0.5*r*(b*b/v)
    l1=labf(b1,v1); l2=labf(b2,v2)
    def lse(x): m=np.max(x); return m+np.log(np.sum(np.exp(x-m)))
    lH1=np.log(p1)+lse(l1); lH2=np.log(p2)+lse(l2)
    lH3=np.log(p1)+np.log(p2)+lse(l1)+lse(l2)
    lH4=np.log(p12)+lse(l1+l2)
    arr=np.array([0.0,lH1,lH2,lH3,lH4]); m=np.max(arr); post=np.exp(arr-m); post/=post.sum()
    return post

gw=pd.read_csv("gwas/eGFR_stanzick2021.gz",sep='\t',compression='gzip',usecols=['chr','pos','Allele1','Allele2','Effect','StdErr'])
gw=gw[gw.chr.astype(str)==CHROM.replace('chr','')]
gw['key']=gw.chr.astype(str)+':'+gw.pos.astype(str)
gw=gw.drop_duplicates('key').set_index('key')
print(f"{CHROM}: eGFR variants {len(gw)}",flush=True)

sel=pd.read_csv("coloc_egfr_overlap_loci.tsv",sep='\t')
sel=sel[sel.chrom==CHROM]
if len(sel)==0: sys.exit(0)
pth=f"{G}/{CHROM}/nominal/{CHROM}.sig.cis_qtl_pairs.{CHROM}.parquet"
if CHROM in ['chr1','chr2']: pth=f"{G}/{CHROM}/nominal/{CHROM}.sig.merged.parquet"
chrn=CHROM.replace('chr','')
want=set(sel.best_cpg)
pf=pq.ParquetFile(pth); frames=[]
for i in range(pf.num_row_groups):
    t=pf.read_row_group(i,columns=['phenotype_id','variant_id','af','slope','slope_se']).to_pandas()
    frames.append(t[t.phenotype_id.isin(want)])
md=pd.concat(frames) if frames else pd.DataFrame()
print(f"{CHROM}: mQTL rows for {len(want)} CpGs: {len(md)}",flush=True)
results=[]
for _,row in sel.iterrows():
    m=md[md.phenotype_id==row.best_cpg].copy()
    if len(m)<20: continue
    p=m.variant_id.str.split(':',expand=True)
    m['pos38']=p[1].astype(int); m['ref']=p[2].str.upper(); m['alt']=p[3].str.upper()
    cpos=int(row.best_cpg.split('_')[1])
    m=m[(m.pos38>cpos-100000)&(m.pos38<cpos+100000)]
    if len(m)<20: continue
    m['pos19']=[(lambda r:r[0][1]+1 if r else 0)(lo.convert_coordinate(CHROM,x-1)) for x in m.pos38]
    m=m[m.pos19>0]; m['key']=chrn+':'+m.pos19.astype(str)
    mm=m[m.key.isin(gw.index)]
    if len(mm)<20: continue
    g=gw.loc[mm.key]
    sign=np.where(mm.alt.values==g.Allele1.str.upper().values,1,np.where(mm.ref.values==g.Allele1.str.upper().values,-1,0))
    ok=sign!=0
    b1=mm.slope.values[ok]; v1=mm.slope_se.values[ok]**2; b2=(g.Effect.values*sign)[ok]; v2=g.StdErr.values[ok]**2
    good=np.isfinite(b1)&np.isfinite(v1)&np.isfinite(b2)&np.isfinite(v2)&(v1>0)&(v2>0)
    if good.sum()<20: continue
    post=coloc_abf(b1[good],v1[good],b2[good],v2[good])
    z1=np.abs(b1[good]/np.sqrt(v1[good])); z2=np.abs(b2[good]/np.sqrt(v2[good]))
    zc=np.corrcoef(z1,z2)[0,1]
    results.append((row.best_cpg,row.nearest_gene,int(good.sum()),round(post[4],3),round(post[3],3),round(zc,3),round(row.asm_frac_fisher,3)))
res=pd.DataFrame(results,columns=['cpg','gene','nsnp','PP4','PP3','z_corr','asm'])
res.to_csv(f"coloc_eGFR_{CHROM}.tsv",sep='\t',index=False)
print(f"{CHROM}: coloc done, {len(res)} loci, PP4>0.8: {int((res.PP4>0.8).sum())}",flush=True)
