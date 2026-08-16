#!/usr/bin/env python3
"""Build sample metadata (sex, platform) for the cis-mQTL covariate model.

  sex      : from a 1000G sample/pedigree file. chr22-only genotypes cannot
             infer sex (no X chromosome), so it must come from metadata.
  platform : R9/R10 from the extraction manifest -- the dominant known batch.
  age      : NOT released for 1000G (de-identified). Left out here; add an
             'age' column for the PROPHECY cohort where it is available.
"""
import argparse, sys
import pandas as pd


def load_sample_list(path):
    if path.endswith(".fam"):
        return pd.read_csv(path, sep=r"\s+", header=None)[1].astype(str).str.strip().tolist()
    import gzip
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        hdr = f.readline().rstrip("\n").split("\t")
    return [s.strip() for s in hdr[4:] if s.strip()]


def gm_to_na(s):
    # GM##### (Coriell) <-> NA##### (1000G); HG##### unchanged
    return "NA" + s[2:] if s.startswith("GM") else s


def load_sex(panel_path):
    df = pd.read_csv(panel_path, sep=r"\s+", dtype=str)
    cols = {c.lower().lstrip("#"): c for c in df.columns}
    idc = next((cols[k] for k in ("sample", "sampleid", "iid", "individualid")
                if k in cols), df.columns[0])
    sexcol = next((cols[k] for k in ("gender", "sex") if k in cols), None)
    if sexcol is None:
        sys.exit(f"No gender/sex column in {panel_path}; columns={list(df.columns)}")
    m = {}
    for _, r in df.iterrows():
        v = str(r[sexcol]).strip().lower()
        sex = "M" if v in ("1", "male", "m") else "F" if v in ("2", "female", "f") else None
        if sex:
            m[str(r[idc]).strip()] = sex
    return m


def load_platform(manifest_path):
    df = pd.read_csv(manifest_path, sep="\t", dtype=str)
    cols = {c.lower(): c for c in df.columns}
    idc = next((cols[k] for k in ("sample", "sample_id", "id", "sid")
                if k in cols), df.columns[0])
    pc = next((cols[k] for k in ("pore", "platform", "chemistry", "flowcell")
               if k in cols), None)
    out = {}
    if pc:
        for _, r in df.iterrows():
            v = str(r[pc]).upper()
            out[str(r[idc]).strip()] = "R10" if "10" in v else "R9" if "9" in v else v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help=".fam or phenotype .bed(.gz) defining the cohort")
    ap.add_argument("--panel", required=True, help="1000G panel/ped file with sex")
    ap.add_argument("--manifest", required=True, help="extraction manifest with pore/platform")
    ap.add_argument("--default-platform", default="R9", help="platform for samples not in the manifest")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    samples = load_sample_list(a.samples)
    sexmap = load_sex(a.panel)
    platmap = load_platform(a.manifest)

    rows, miss_sex, miss_plat = [], [], []
    for s in samples:
        sx = sexmap.get(s) or sexmap.get(gm_to_na(s))
        if sx is None:
            miss_sex.append(s)
        pl = platmap.get(s)
        if pl is None:
            pl = a.default_platform
            miss_plat.append(s)
        rows.append({"sample": s, "sex": sx or "NA", "platform": pl})

    md = pd.DataFrame(rows)
    md.to_csv(a.out, sep="\t", index=False)
    print(f"Wrote {a.out}: {len(md)} samples")
    print(f"  sex:  M={(md.sex=='M').sum()} F={(md.sex=='F').sum()} NA={(md.sex=='NA').sum()}")
    print("  plat: " + " ".join(f"{k}={v}" for k, v in md.platform.value_counts().items()))
    if miss_sex:
        print(f"  WARNING: sex missing for {len(miss_sex)} (try the 3202 ped file): {miss_sex[:10]}")
    if miss_plat:
        print(f"  NOTE: platform defaulted to {a.default_platform} for {len(miss_plat)}: {miss_plat[:10]}")
    print("  age omitted: not released for 1000G; add an 'age' column for PROPHECY")


if __name__ == "__main__":
    main()
