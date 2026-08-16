#!/usr/bin/env python3
"""asm_readlevel.py — fragment-level, region-based ASM (Atlas method).
Optimized: assigns each call to a region once (vectorized), then one groupby
over (region, read) for fragment U/X/M, then contingency + Fisher per region.

--calls: read_id, chrom, ref_position, call, hp   (hp in 1/2)
--regions: BED chrom,start,end,name
"""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", required=True)
    ap.add_argument("--regions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-cpg", type=int, default=3)
    ap.add_argument("--u-max", type=float, default=0.35)
    ap.add_argument("--m-min", type=float, default=0.65)
    ap.add_argument("--min-frag", type=int, default=5)
    ap.add_argument("--meth-codes", default="m,C+m,1,methylated")
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

    # assign each call to a region via interval search (regions are sorted, assume
    # non-overlapping after merge); searchsorted on starts, then check end.
    starts = reg["start"].values
    ends = reg["end"].values
    idx = np.searchsorted(starts, c["ref_position"].values, side="right") - 1
    valid = (idx >= 0) & (idx < len(reg))
    pos = c["ref_position"].values
    inside = np.zeros(len(c), dtype=bool)
    inside[valid] = pos[valid] < ends[np.clip(idx[valid], 0, len(reg) - 1)]
    c = c.assign(region=idx)[inside]
    print(f"  calls mapped to regions: {len(c):,}", flush=True)

    # fragment (region,read) average methylation + hp
    frag = c.groupby(["region", "read_id"]).agg(
        hp=("hp", "first"), ncpg=("is_meth", "size"), meth=("is_meth", "mean"))
    frag = frag[frag["ncpg"] >= a.min_cpg].reset_index()
    frag["cls"] = np.where(frag["meth"] <= a.u_max, "U",
                  np.where(frag["meth"] >= a.m_min, "M", "X"))

    rows = []
    for rgi, g in frag.groupby("region"):
        u1 = int(((g.hp == 1) & (g.cls == "U")).sum())
        m1 = int(((g.hp == 1) & (g.cls == "M")).sum())
        u2 = int(((g.hp == 2) & (g.cls == "U")).sum())
        m2 = int(((g.hp == 2) & (g.cls == "M")).sum())
        n1, n2 = u1 + m1, u2 + m2
        if n1 < a.min_frag or n2 < a.min_frag:
            continue
        _, p = fisher_exact([[u1, m1], [u2, m2]])
        meth1, meth2 = m1 / n1, m2 / n2
        r = reg.iloc[rgi]
        rows.append((r["chrom"], r["start"], r["end"], r["name"],
                     u1, m1, u2, m2, round(meth1, 3), round(meth2, 3),
                     round(meth1 - meth2, 3), p))

    df = pd.DataFrame(rows, columns=["chrom", "start", "end", "name", "U_hp1",
                                     "M_hp1", "U_hp2", "M_hp2", "meth_hp1",
                                     "meth_hp2", "delta", "p"])
    if len(df):
        m = len(df)
        rank = df["p"].rank(method="first")
        df["q"] = (df["p"] * m / rank).clip(upper=1.0)
        df = df.sort_values("p").reset_index(drop=True)
    df.to_csv(a.out, sep="\t", index=False)
    n_asm = int((df["q"] < 0.01).sum()) if len(df) else 0
    print(f"regions tested: {len(df):,}   ASM (q<0.01): {n_asm:,}   -> {a.out}")
    if len(df):
        print(df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
