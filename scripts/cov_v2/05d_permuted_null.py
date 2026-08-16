#!/usr/bin/env python3
"""05d_permuted_null.py - negative-control calibration check for the cis-mQTL scan.

Breaks the genotype<->phenotype correspondence by permuting the genotype sample
labels: each sample keeps its own phenotype AND covariates, but is paired with a
random sample's genotypes. Under this null there is no real cis effect, so the
permutation p-values should be ~uniform and the QQ should sit on the diagonal
(lambda ~ 1). If the real scan is inflated but this control is flat, the real
inflation is genuine cis-mQTL signal -- not confounding or relatedness.

Runs on a random subset of CpGs by default (calibration needs a sample, not all).
"""
import argparse, time
import numpy as np
import pandas as pd


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def inverse_normal_transform(df):
    from scipy.stats import norm
    r = df.rank(axis=1, method="average")
    n = df.shape[1]
    return pd.DataFrame(norm.ppf((r - 0.5) / n), index=df.index, columns=df.columns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plink-prefix", required=True)
    ap.add_argument("--phenotype-bed", required=True)
    ap.add_argument("--covariates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cis-window", type=int, default=1000000)
    ap.add_argument("--maf-threshold", type=float, default=0.05)
    ap.add_argument("--n-permutations", type=int, default=1000)
    ap.add_argument("--n-phenotypes", type=int, default=20000, help="random CpG subset (0 = all)")
    ap.add_argument("--inverse-normal", action="store_true")
    ap.add_argument("--shuffle-seed", type=int, default=7)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()

    import torch
    torch.set_num_threads(a.threads)
    import tensorqtl
    from tensorqtl import genotypeio, cis
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"tensorqtl {tensorqtl.__version__} | device={dev} | NEGATIVE CONTROL (shuffled genotypes)")

    phenotype_df, phenotype_pos_df = tensorqtl.read_phenotype_bed(a.phenotype_bed)
    if a.n_phenotypes and a.n_phenotypes < phenotype_df.shape[0]:
        rng = np.random.default_rng(a.shuffle_seed)
        idx = np.sort(rng.choice(phenotype_df.shape[0], a.n_phenotypes, replace=False))
        phenotype_df = phenotype_df.iloc[idx]
        phenotype_pos_df = phenotype_pos_df.iloc[idx]
    log(f"Phenotypes (subset): {phenotype_df.shape[0]:,} CpGs x {phenotype_df.shape[1]} samples")
    if a.inverse_normal:
        phenotype_df = inverse_normal_transform(phenotype_df)

    covariates_df = pd.read_csv(a.covariates, sep="\t", index_col=0).T

    pr = genotypeio.PlinkReader(a.plink_prefix)
    genotype_df = pr.load_genotypes()
    variant_df = pr.bim.set_index("snp")[["chrom", "pos"]]

    # --- the negative control: permute genotype sample labels ---
    rng = np.random.default_rng(a.shuffle_seed)
    perm = rng.permutation(genotype_df.shape[1])
    orig = genotype_df.columns.tolist()
    genotype_df = genotype_df.iloc[:, perm]
    genotype_df.columns = orig          # each ID now carries a random sample's genotypes
    nfix = int(np.sum(perm == np.arange(len(perm))))
    log(f"Shuffled {genotype_df.shape[1]} genotype labels (seed={a.shuffle_seed}; {nfix} stayed fixed by chance)")

    cis_df = cis.map_cis(genotype_df, variant_df, phenotype_df, phenotype_pos_df,
                         covariates_df=covariates_df, maf_threshold=a.maf_threshold,
                         nperm=a.n_permutations, window=a.cis_window, seed=42)
    log(f"map_cis done: {cis_df.shape[0]} CpGs")
    cis_df.to_csv(a.out, sep="\t")
    log(f"Written: {a.out}  (run qq_fdr.py on this; expect lambda ~ 1)")


if __name__ == "__main__":
    main()
