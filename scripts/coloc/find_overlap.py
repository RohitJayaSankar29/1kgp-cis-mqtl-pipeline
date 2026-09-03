import pandas as pd, numpy as np
from pyliftover import LiftOver
lo=LiftOver('hg38','hg19')
G='/g/data/cy94/rs4477/downstream/genome_452'
cand=pd.read_csv('/scratch/cy94/rs4477/coloc/coloc_candidate_loci.tsv',sep='\t')
cand=cand[cand.asm_frac_fisher.fillna(0)>0.25].copy()
p=cand.variant_id.str.split(':',expand=True)
cand['chrom']=p[0]
enr=pd.read_csv(G+'/genome.prioritised_variant_level.enriched.txt.gz',sep='\t',usecols=['variant_id','best_cpg'])
cand=cand.merge(enr,on='variant_id',how='left').dropna(subset=['best_cpg']).drop_duplicates('best_cpg')
cand['cpos38']=cand.best_cpg.str.split('_').str[1].astype(int)
def lift(ch,pos):
    r=lo.convert_coordinate(ch,pos-1)
    return r[0][1]+1 if r else np.nan
cand['cpos19']=[lift(c,x) for c,x in zip(cand.chrom,cand.cpos38)]
cand=cand.dropna(subset=['cpos19'])
cand['cpos19']=cand.cpos19.astype(int)
cand['chrn']=cand.chrom.str.replace('chr','')
loci=pd.read_csv('/scratch/cy94/rs4477/coloc/egfr_sig_loci_hg19.tsv',sep='\t')
loci['chr']=loci.chr.astype(str)
ov=[]
for _,r in cand.iterrows():
    hit=loci[(loci.chr==r.chrn)&(loci.start-100000<=r.cpos19)&(loci.end+100000>=r.cpos19)]
    if len(hit): ov.append(r.best_cpg)
sel=cand[cand.best_cpg.isin(ov)]
sel.to_csv('/scratch/cy94/rs4477/coloc/coloc_egfr_overlap_loci.tsv',sep='\t',index=False)
print('candidate methylation loci overlapping eGFR sig regions:', len(sel))
print(sel[['best_cpg','nearest_gene','asm_frac_fisher']].sort_values('asm_frac_fisher',ascending=False).head(25).to_string(index=False))
