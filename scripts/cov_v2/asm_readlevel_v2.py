#!/usr/bin/env python3
"""asm_readlevel_v2.py - fragment-level, region-based ASM.

Computes TWO statistics per region in a single pass:

  fisher : Atlas method. Fragments binned U (<=u_max) / X / M (>=m_min);
           2x2 contingency [U,M] x [hp1,hp2]; Fisher exact. X fragments dropped.
  mwu    : Mann-Whitney U on the CONTINUOUS per-fragment methylation fraction,
           hp1 fragments vs hp2 fragments. All fragments retained (incl. X).

Both are always reported (p_fisher, p_mwu). --stat decides which one populates
the canonical p/q/delta/meth_hp1/meth_hp2 columns, so downstream scripts
(combine_asm_oriented.py etc.) work unchanged either way.

Rationale for mwu: when nearly all fragments on both haplotypes fall in the same
bin (saturation), the 2x2 table carries almost no information and Fisher has no
power, even if a real allelic shift exists. The continuous test keeps that
resolution. This is a deliberate deviation from Atlas, justified by ONT
fragments covering enough CpGs to give a stable per-fragment estimate.

--calls: read_id, chrom, ref_position, call, hp   (hp in 1/2)
--regions: BED chrom,start,end,name
"""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu


def bh_qvalues(p):
    """Benjamini-Hochberg with monotonicity enforced."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]   # enforce monotonicity
    q = np.empty(n, dtype=float)
    q[order] = np.clip(ranked, 0, 1)
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", required=True)
    ap.add_argument("--regions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-cpg", type=int, default=3)
    ap.add_argument("--u-max", type=float, default=0.35)
    ap.add_argument("--m-min", type=float, default=0.65)
    ap.add_argument("--min-frag", type=int, default=5)
    ap.add_argument("--stat", choices=["fisher", "mwu"], default="fisher",
                    help="which statistic populates p/q/delta/meth_hp1/meth_hp2")
    ap.add_argument("--meth-codes", default="m,C+m,1,methylated")
    ap.add_argument("--permute-hp", action="store_true",
                    help="shuffle hp labels within each region (null control)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    meth = set(a.meth_codes.split(","))
    reg = pd.read_csv(a.regions, sep="\t", header=None,
                      names=["chrom", "start", "end", "name"])
    reg = reg.sort_values("start").reset_index(drop=True)

    print("reading calls...", flush=True)
    c = pd.read_csv(a.calls, sep="\t")
    c["is_meth"] = c["call"].astype(str).isin(meth).astype(int)
    c["hp"] = pd.to_numeric(c["hp"], errors="coerce")
    c = c[c["hp"].isin([1, 2])]
    c["ref_position"] = pd.to_numeric(c["ref_position"], errors="coerce")
    c = c.dropna(subset=["ref_position"]).sort_values("ref_position")

    # assign each call to a region (regions sorted, non-overlapping after merge)
    starts = reg["start"].values
    ends = reg["end"].values
    idx = np.searchsorted(starts, c["ref_position"].values, side="right") - 1
    valid = (idx >= 0) & (idx < len(reg))
    pos = c["ref_position"].values
    inside = np.zeros(len(c), dtype=bool)
    inside[valid] = pos[valid] < ends[np.clip(idx[valid], 0, len(reg) - 1)]
    c = c.assign(region=idx)[inside]
    print(f"  calls mapped to regions: {len(c):,}", flush=True)

    # fragment = (region, read): continuous methylation fraction + haplotype
    frag = c.groupby(["region", "read_id"]).agg(
        hp=("hp", "first"), ncpg=("is_meth", "size"), meth=("is_meth", "mean"))
    frag = frag[frag["ncpg"] >= a.min_cpg].reset_index()
    frag["cls"] = np.where(frag["meth"] <= a.u_max, "U",
                  np.where(frag["meth"] >= a.m_min, "M", "X"))

    if a.permute_hp:
        rng = np.random.default_rng(a.seed)
        frag["hp"] = frag.groupby("region")["hp"].transform(
            lambda s: rng.permutation(s.values))
        print("[permute-hp] hp labels shuffled within each region (null)", flush=True)

    rows = []
    for rgi, g in frag.groupby("region"):
        h1 = g[g.hp == 1]
        h2 = g[g.hp == 2]

        # --- Atlas / Fisher path: U and M only ---
        u1 = int((h1.cls == "U").sum()); m1 = int((h1.cls == "M").sum())
        u2 = int((h2.cls == "U").sum()); m2 = int((h2.cls == "M").sum())
        nb1, nb2 = u1 + m1, u2 + m2          # binary-usable fragments

        # --- continuous path: all fragments ---
        v1 = h1["meth"].values
        v2 = h2["meth"].values
        nc1, nc2 = len(v1), len(v2)
        nX = int((g.cls == "X").sum())

        # require enough fragments under whichever statistic is canonical
        need_binary = a.stat == "fisher"
        if need_binary:
            if nb1 < a.min_frag or nb2 < a.min_frag:
                continue
        else:
            if nc1 < a.min_frag or nc2 < a.min_frag:
                continue

        # Fisher (only if the binary table is usable)
        if nb1 >= a.min_frag and nb2 >= a.min_frag:
            _, p_fisher = fisher_exact([[u1, m1], [u2, m2]])
            fracM1, fracM2 = m1 / nb1, m2 / nb2
        else:
            p_fisher = np.nan
            fracM1 = m1 / nb1 if nb1 else np.nan
            fracM2 = m2 / nb2 if nb2 else np.nan

        # Mann-Whitney on continuous fragment methylation
        if nc1 >= a.min_frag and nc2 >= a.min_frag:
            try:
                _, p_mwu = mannwhitneyu(v1, v2, alternative="two-sided")
            except ValueError:      # all values identical -> no evidence
                p_mwu = 1.0
            if np.isnan(p_mwu):
                p_mwu = 1.0
        else:
            p_mwu = np.nan
        mean1 = float(np.mean(v1)) if nc1 else np.nan
        mean2 = float(np.mean(v2)) if nc2 else np.nan

        r = reg.iloc[rgi]
        rows.append((r["chrom"], r["start"], r["end"], r["name"],
                     u1, m1, u2, m2, nX, nb1, nb2, nc1, nc2,
                     round(fracM1, 3) if fracM1 == fracM1 else np.nan,
                     round(fracM2, 3) if fracM2 == fracM2 else np.nan,
                     round(mean1, 3), round(mean2, 3),
                     p_fisher, p_mwu))

    cols = ["chrom", "start", "end", "name",
            "U_hp1", "M_hp1", "U_hp2", "M_hp2", "n_X",
            "nbin_hp1", "nbin_hp2", "nfrag_hp1", "nfrag_hp2",
            "fracM_hp1", "fracM_hp2", "mean_hp1", "mean_hp2",
            "p_fisher", "p_mwu"]
    df = pd.DataFrame(rows, columns=cols)

    if len(df):
        # canonical columns follow --stat so downstream is unchanged
        if a.stat == "fisher":
            df["meth_hp1"] = df["fracM_hp1"]
            df["meth_hp2"] = df["fracM_hp2"]
            df["p"] = df["p_fisher"]
        else:
            df["meth_hp1"] = df["mean_hp1"]
            df["meth_hp2"] = df["mean_hp2"]
            df["p"] = df["p_mwu"]
        df["delta"] = (df["meth_hp1"] - df["meth_hp2"]).round(3)
        df["p"] = df["p"].fillna(1.0)
        df["q"] = bh_qvalues(df["p"].values)
        # q for the other statistic too, for direct comparison
        df["q_fisher"] = bh_qvalues(df["p_fisher"].fillna(1.0).values)
        df["q_mwu"] = bh_qvalues(df["p_mwu"].fillna(1.0).values)
        df = df.sort_values("p").reset_index(drop=True)
    else:
        for extra in ["meth_hp1", "meth_hp2", "p", "delta", "q", "q_fisher", "q_mwu"]:
            df[extra] = []

    df.to_csv(a.out, sep="\t", index=False)

    if len(df):
        n_f = int((df["q_fisher"] < 0.01).sum())
        n_m = int((df["q_mwu"] < 0.01).sum())
        sat = df[(df["fracM_hp1"] > 0.85) & (df["fracM_hp2"] > 0.85)]
        n_f_sat = int((sat["q_fisher"] < 0.01).sum()) if len(sat) else 0
        n_m_sat = int((sat["q_mwu"] < 0.01).sum()) if len(sat) else 0
        print(f"regions tested: {len(df):,}   (canonical stat: {a.stat})")
        print(f"  ASM q<0.01  fisher: {n_f:,}   mwu: {n_m:,}")
        print(f"  saturated regions (fracM>0.85 both haps): {len(sat):,}")
        print(f"    of those, q<0.01  fisher: {n_f_sat:,}   mwu: {n_m_sat:,}")
        print(f"  -> {a.out}")
        print(df.head(12).to_string(index=False))
    else:
        print(f"no regions passed thresholds -> {a.out}")


if __name__ == "__main__":
    main()
