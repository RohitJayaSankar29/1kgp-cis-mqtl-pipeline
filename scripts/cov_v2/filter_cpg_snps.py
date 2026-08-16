#!/usr/bin/env python3
"""filter_cpg_snps.py — remove CpG phenotypes that are destroyed by a SNP.

A CpG is a C followed by a G. If a variant changes either base, carriers have no
CpG there, so genotype MECHANICALLY forces the methylation readout -- an
artefactual "association" that is not regulation. These artefacts are strong
(they dominate the top of the prioritised list), so they must be removed BEFORE
the QTL scan rather than flagged afterwards.

Filters the phenotype BED: drops any CpG whose C position or G position (pos and
pos+1 by default) overlaps a substitution in the variant source.

Variant source: a VCF/panel (thorough -- catches rare CpG-destroying variants
that MAF filtering removed) or a PLINK .bim (simpler, less complete).

Usage:
  filter_cpg_snps.py --bed methylation_Mval.bed.gz --variants chr22.snp_positions.tsv \
      --out methylation_Mval.cpgfilt.bed.gz [--transitions-only] [--offset 1]

--variants: TSV of  CHROM<tab>POS<tab>REF<tab>ALT  (one per line, no header).
  From a panel:  bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\n' -v snps panel.vcf.gz
  From a bim:    awk '{print $1"\t"$4"\t"$5"\t"$6}' file.bim
"""
import argparse
import gzip

import pandas as pd

TRANSITIONS = {("C", "T"), ("T", "C"), ("G", "A"), ("A", "G")}


def opener(p, mode="rt"):
    return gzip.open(p, mode) if str(p).endswith(".gz") else open(p, mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bed", required=True, help="phenotype BED (tensorQTL format)")
    ap.add_argument("--variants", required=True, help="TSV: chrom pos ref alt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--offset", type=int, default=1,
                    help="CpG spans pos..pos+offset (default 1: C at pos, G at pos+1)")
    ap.add_argument("--transitions-only", action="store_true",
                    help="only C<->T / G<->A (default: ANY substitution at either base)")
    ap.add_argument("--report", help="optional TSV listing the dropped CpGs")
    a = ap.parse_args()

    # build the set of positions carrying a CpG-destroying substitution
    bad = set()
    with opener(a.variants) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 4:
                continue
            chrom, pos, ref, alt = f[0], f[1], f[2].upper(), f[3].upper()
            if len(ref) != 1 or len(alt) != 1:      # substitutions only (skip indels)
                continue
            if a.transitions_only and (ref, alt) not in TRANSITIONS:
                continue
            try:
                bad.add((chrom, int(pos)))
            except ValueError:
                continue
    print(f"CpG-destroying candidate variant positions: {len(bad):,}")

    kept, dropped, header = [], [], None
    with opener(a.bed) as fh:
        for line in fh:
            if line.startswith("#"):
                header = line
                continue
            f = line.split("\t", 4)
            chrom, start = f[0], int(f[1])
            # CpG occupies start .. start+offset  -> a hit at EITHER base kills it
            hit = any((chrom, start + k + 1) in bad for k in range(a.offset + 1)) \
                  or any((chrom, start + k) in bad for k in range(a.offset + 1))
            (dropped if hit else kept).append(line)

    with opener(a.out, "wt") as out:
        if header:
            out.write(header)
        out.writelines(kept)

    tot = len(kept) + len(dropped)
    pct = 100 * len(dropped) / tot if tot else 0
    print(f"CpGs in : {tot:,}")
    print(f"dropped : {len(dropped):,}  ({pct:.2f}%)  <- CpG-SNP artefacts")
    print(f"kept    : {len(kept):,}   -> {a.out}")

    if a.report and dropped:
        with open(a.report, "w") as r:
            r.write("chrom\tstart\tphenotype_id\n")
            for line in dropped:
                f = line.split("\t")
                r.write(f"{f[0]}\t{f[1]}\t{f[3]}\n")
        print(f"dropped list -> {a.report}")


if __name__ == "__main__":
    main()
