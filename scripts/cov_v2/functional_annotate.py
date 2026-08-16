#!/usr/bin/env python3
"""Stage 08b: functional annotation of significant CpGs (chr22).
Nearest gene/TSS + distance, CpG-island overlap, ENCODE cCRE overlap.
Pure pandas (no bedtools needed) -- fine at chr22 scale. Any track left
unspecified is simply skipped.
"""
import argparse, gzip, os, time
import numpy as np
import pandas as pd


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def cpg_positions(matrix, sig):
    op = gzip.open if matrix.endswith(".gz") else open
    with op(matrix, "rt"):
        pass
    df = pd.read_csv(matrix, sep="\t", usecols=range(4))
    df.columns = ["chrom", "start", "end", "cpg"][:df.shape[1]]
    df = df[df["cpg"].isin(set(sig))]
    return df[["cpg", "chrom", "start"]].rename(columns={"start": "pos"})


def load_bed(path, cols):
    if not path or not os.path.exists(path):
        return None
    d = pd.read_csv(path, sep="\t", header=None)
    d.columns = cols[:d.shape[1]]
    return d


def overlaps(pos, bed, label_col):
    if bed is None:
        return ""
    hit = bed[(bed["start"] <= pos) & (bed["end"] >= pos)]
    return ";".join(sorted(set(hit[label_col].astype(str)))) if not hit.empty else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--significant-cpgs", required=True)
    ap.add_argument("--tss-bed")
    ap.add_argument("--cpg-island-bed")
    ap.add_argument("--ccre-bed")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    sig = [l.strip() for l in open(a.significant_cpgs) if l.strip()]
    if not sig:
        log("No significant CpGs.")
        pd.DataFrame().to_csv(a.out, sep="\t", index=False)
        return

    cpgs = cpg_positions(a.matrix, sig)
    tss = load_bed(a.tss_bed, ["chrom", "start", "end", "gene", "score", "strand"])
    isl = load_bed(a.cpg_island_bed, ["chrom", "start", "end", "label"])
    ccre = load_bed(a.ccre_bed, ["chrom", "start", "end", "label"])
    tpos = tss.sort_values("start")["start"].values if tss is not None else np.array([])
    tss_sorted = tss.sort_values("start").reset_index(drop=True) if tss is not None else None

    rows = []
    for _, r in cpgs.iterrows():
        p = int(r["pos"])
        rec = {"cpg": r["cpg"], "chrom": r["chrom"], "pos": p,
               "nearest_gene": "", "dist_to_tss": np.nan}
        if len(tpos):
            j = int(np.searchsorted(tpos, p))
            cand = [k for k in (j - 1, j) if 0 <= k < len(tpos)]
            best = min(cand, key=lambda k: abs(int(tpos[k]) - p))
            rec["nearest_gene"] = tss_sorted.iloc[best]["gene"]
            rec["dist_to_tss"] = int(p - int(tpos[best]))
        rec["cpg_island"] = "yes" if overlaps(p, isl, "label") else "no"
        rec["ccre"] = overlaps(p, ccre, "label") or "none"
        rows.append(rec)

    out = pd.DataFrame(rows)
    out.to_csv(a.out, sep="\t", index=False)
    log(f"Wrote {a.out}: {out.shape[0]} CpGs annotated")


if __name__ == "__main__":
    main()
