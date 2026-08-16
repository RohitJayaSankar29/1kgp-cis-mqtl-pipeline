# 1KGP cis-mQTL CpG-centric prioritisation pipeline

A reproducible CpG-centric cis-mQTL prioritisation framework built on the public
1000 Genomes Project Oxford Nanopore (1000G-ONT) resource.

## Status (genome-wide run)

Completed end-to-end on **452 unrelated 1000G-ONT samples** (R9 + R10), genome-wide:

- Cohort: 452 unrelated founders (500 roster minus 48 relatives)
- Methylation matrices (5mC, M-value): 22/22 chromosomes
- Genotypes (PLINK, MAF>0.05): 22/22, 452 samples
- CpG-SNP artefact filter: 22/22 (~2.7% dropped)
- Covariates: sex + chemistry + basecaller + 10 genoPC + 10 methPC
- cis-mQTL permutation scan: 22/22, +/-1Mb, 100 perms

### Headline result

- CpGs tested genome-wide: 25,912,587
- Significant cis-mQTLs (FDR<0.05, BH): 1,638,801 (~6.3%)
- Top hits reach p ~ 6e-241; driving variants adjacent to their CpGs.

## Method notes

- Methylation from pre-computed Modkit bedMethyl (5mC only).
- CpG-SNP filter uses in-sample variants (cohort .bim), not the full panel.
- Genotype PCs genome-wide (merged, LD-pruned, 10 PCs).
- Permutation pass is CPU-bound; run chunked-parallel (184 chunks) not GPU.
- Discovery: 100 permutations; 1000-perm confirmation only on prioritised hits.
- FDR by Benjamini-Hochberg.

## Pipeline stages

manifest -> methylation matrix -> genotypes -> CpG-SNP filter -> covariates
-> cis-mQTL permutation -> FDR -> nominal-on-significant -> SuSiE fine-mapping
-> SV integration -> functional annotation -> prioritisation -> ASM corroboration

## Safety

Big data (BAM/VCF/BED/parquet/full results) is NOT committed; see .gitignore.
Only scripts, jobs, configs, and small summaries are versioned.
