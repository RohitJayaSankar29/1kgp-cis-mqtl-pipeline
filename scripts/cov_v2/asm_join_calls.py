#!/usr/bin/env python3
"""asm_join_calls.py — memory-safe filter+join: stream a big `modkit extract calls`
table, keep only haplotagged calls inside the regions, attach hp (1/2) per read.
Writes the compact --calls table asm_readlevel.py needs.

Usage: asm_join_calls.py RAW_CALLS HTLIST REGIONS OUT
"""
import sys
import pandas as pd

raw, htlist, regions, out = sys.argv[1:5]

ht = pd.read_csv(htlist, sep="\t")
ht.columns = [c.lstrip("#") for c in ht.columns]
hpmap = {"H1": 1, "H2": 2, "1": 1, "2": 2}
ht["hp"] = ht["haplotype"].astype(str).map(hpmap)
ht = ht.dropna(subset=["hp"]).set_index("readname")["hp"].astype(int)

reg = pd.read_csv(regions, sep="\t", header=None,
                  names=["chrom", "start", "end", "name"])
lo, hi = int(reg["start"].min()), int(reg["end"].max())

head = pd.read_csv(raw, sep="\t", nrows=0)
rid = next((x for x in ("read_id", "read_name") if x in head.columns), head.columns[0])
def pick(opts): return next((x for x in opts if x in head.columns), None)
cc = pick(["chrom", "chromosome"])
pc = pick(["ref_position", "ref_pos", "position"])
ca = pick(["call_code", "call", "mod_code"])
usecols = [rid, cc, pc, ca]

first, kept = True, 0
for chunk in pd.read_csv(raw, sep="\t", usecols=usecols, chunksize=2_000_000):
    chunk = chunk.rename(columns={rid: "read_id", cc: "chrom",
                                  pc: "ref_position", ca: "call"})
    chunk = chunk[(chunk["chrom"] == "chr22") &
                  (chunk["ref_position"] >= lo) & (chunk["ref_position"] < hi)]
    chunk["hp"] = chunk["read_id"].map(ht)
    chunk = chunk.dropna(subset=["hp"])
    if chunk.empty:
        continue
    chunk["hp"] = chunk["hp"].astype(int)
    chunk[["read_id", "chrom", "ref_position", "call", "hp"]].to_csv(
        out, sep="\t", index=False, mode="w" if first else "a", header=first)
    first, kept = False, kept + len(chunk)
print(f"  kept {kept:,} haplotyped calls inside regions -> {out}")
