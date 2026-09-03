import pandas as pd, numpy as np
G="/g/data/cy94/rs4477/downstream/genome_452"
cols=['variant_id','chrom','pos','nearest_gene','pip','asm_frac_fisher','asm_frac_mwu','asm_abs_delta','score_weighted','score_v2','score_convergence','sv_implicated','ccre','n_cpgs']
v=pd.read_csv(G+"/genome.prioritised_final_asm_rescored.txt.gz", sep="\t", usecols=lambda c: c in cols)
if 'chrom' not in v.columns:
    p=v.variant_id.str.split(':',expand=True); v['chrom']=p[0]; v['pos']=p[1].astype(int)
# known imprinted domains (hg38, major imprinted regions)
imprinted_regions=[
 ('chr11',2000000,2900000),    # KCNQ1/IGF2/H19 (Beckwith-Wiedemann)
 ('chr15',23000000,25700000),  # PWS/AS (SNRPN/SNORD116/PWAR1/UBE3A)
 ('chr20',58800000,58920000),  # GNAS
 ('chr14',100700000,101000000),# DLK1/MEG3
 ('chr7',94200000,94300000),   # PEG10/SGCE
 ('chr7',50800000,50900000),   # GRB10
 ('chr8',140000000,140200000), # PEG13/KCNK9
 ('chr19',53000000,54200000),  # ZNF331 cluster
 ('chr6',143900000,144200000), # PLAGL1
 ('chr1',40000000,40100000),   # (other)
 ('chr13',48300000,48500000),  # RB1 imprinted
]
v['imprinted']=False
for ch,s,e in imprinted_regions:
    v.loc[(v.chrom==ch)&(v.pos>=s)&(v.pos<=e),'imprinted']=True
# classify ASM type
v['asm_class']=np.where(v.imprinted,'imprinting_control',
                np.where(v.asm_frac_fisher.fillna(0)>0.25,'sequence_driven_ASM','no_strong_ASM'))
v.to_csv(G+"/genome.prioritised_final_asm_classified.txt.gz", sep="\t", index=False)
print('total variants:', len(v))
print('ASM-tested:', int(v.asm_frac_fisher.notna().sum()))
print('flagged imprinted (control):', int(v.imprinted.sum()))
print('sequence-driven ASM (frac_fisher>0.25, non-imprinted):', int((v.asm_class=="sequence_driven_ASM").sum()))
print('convergence>=4, non-imprinted:', int(((v.score_convergence>=4)&(~v.imprinted)).sum()))
print()
# sequence-driven candidates
sd=v[(~v.imprinted)&(v.asm_frac_fisher.fillna(0)>0.25)].copy()
print('=== TOP SEQUENCE-DRIVEN cis-mQTL candidates (ASM-corroborated, imprinting excluded) ===')
print(sd.sort_values(['score_convergence','asm_frac_fisher'],ascending=False).head(20)[['variant_id','nearest_gene','pip','asm_frac_fisher','asm_abs_delta','score_convergence']].to_string(index=False))
