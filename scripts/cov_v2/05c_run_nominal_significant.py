#!/usr/bin/env python3
"""
05c_run_nominal_significant.py — map_nominal for FDR-significant CpGs only.
This is the memory-safe design: full nominal mapping is run ONLY for CpGs that
passed FDR (not genome-wide), avoiding the all-pairs blow-up in CpG-dense
regions. Output feeds fine-mapping (SuSiE) downstream.
"""
import argparse, os, time
import pandas as pd


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def inverse_normal_transform(df):
    from scipy.stats import norm
    ranks = df.rank(axis=1, method="average")
    n = df.shape[1]
    return pd.DataFrame(norm.ppf((ranks - 0.5) / n),
                        index=df.index, columns=df.columns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plink-prefix", required=True)
    ap.add_argument("--phenotype-bed", required=True)
    ap.add_argument("--covariates", required=True)
    ap.add_argument("--significant-cpgs", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", default="chr22.sig")
    ap.add_argument("--cis-window", type=int, default=1000000)
    ap.add_argument("--maf-threshold", type=float, default=0.05)
    ap.add_argument("--inverse-normal", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    sig = [l.strip() for l in open(a.significant_cpgs) if l.strip()]
    if not sig:
        log("No significant CpGs — nothing to map. Exiting cleanly.")
        return

    import torch
    torch.set_num_threads(a.threads)
    import tensorqtl
    from tensorqtl import genotypeio, cis

    phenotype_df, phenotype_pos_df = tensorqtl.read_phenotype_bed(a.phenotype_bed)
    keep = phenotype_df.index.intersection(sig)
    phenotype_df = phenotype_df.loc[keep]
    phenotype_pos_df = phenotype_pos_df.loc[keep]
    log(f"Mapping nominal for {phenotype_df.shape[0]} significant CpGs")
    if a.inverse_normal:
        phenotype_df = inverse_normal_transform(phenotype_df)

    covariates_df = pd.read_csv(a.covariates, sep="\t", index_col=0).T
    pr = genotypeio.PlinkReader(a.plink_prefix)
    genotype_df = pr.load_genotypes()
    variant_df = pr.bim.set_index("snp")[["chrom", "pos"]]

    cis.map_nominal(
        genotype_df, variant_df, phenotype_df, phenotype_pos_df,
        prefix=a.prefix, covariates_df=covariates_df,
        maf_threshold=a.maf_threshold, window=a.cis_window,
        output_dir=a.out_dir,
    )
    log(f"Nominal results written to {a.out_dir}/{a.prefix}*")


if __name__ == "__main__":
    main()
