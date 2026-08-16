#!/usr/bin/env python3
"""make_pilot_figures.py — generate the chr22/n=100 pilot figures and the top
prioritised-variant table from the on-disk result files.

Run in the mqtl_tensorqtl env (matplotlib + pandas + numpy + scipy). Writes:
  <outdir>/figures/qq_real_vs_null.png
  <outdir>/figures/score_convergence.png
  <outdir>/figures/elasticnet_coefficients.png
  <outdir>/figures/sv_type_breakdown.png
  <outdir>/figures/pip_distribution.png
  <outdir>/results/chr22.topN_prioritised.tsv  (+ .md)

Each figure is wrapped in try/except so one missing input never kills the rest.
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Elastic-net coefficients from the pilot stage-09 run (update if you re-run it).
PILOT_ENET_COEFS = {
    "neg_log10_p": 0.194, "neg_log10_fdr": -0.176, "prox_cpg": 0.036,
    "prox_tss": 0.022, "abs_effect": 0.005, "var_ccre": 0.005,
    "maf": -0.004, "sv_flag": -0.009, "var_cpg_island": -0.003,
}


def lam_and_p(series):
    from scipy.stats import chi2
    p = pd.to_numeric(series, errors="coerce").dropna().clip(1e-300, 1)
    lam = float(np.median(chi2.isf(p, 1)) / chi2.ppf(0.5, 1))
    return lam, p


def qq_xy(p, thin=20000):
    p = np.sort(p.values)
    n = len(p)
    exp = -np.log10((np.arange(1, n + 1) - 0.5) / n)
    obs = -np.log10(p)
    if n > thin:                       # keep the tail, thin the dense bulk
        keep = np.r_[np.linspace(0, n - 2000, thin - 2000).astype(int), np.arange(n - 2000, n)]
        exp, obs = exp[keep], obs[keep]
    return exp, obs


def to_md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if pd.isna(r[c]) else str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm", required=True, help="real permutation table (pval_beta)")
    ap.add_argument("--null", help="permuted-null permutation table")
    ap.add_argument("--prioritised", required=True, help="chr22.prioritised_framework.txt.gz")
    ap.add_argument("--susie", help="chr22.susie.txt.gz")
    ap.add_argument("--sv", help="chr22.sv_annotation.txt.gz")
    ap.add_argument("--functional", help="chr22.functional.txt.gz")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--top", type=int, default=30)
    a = ap.parse_args()
    fig = os.path.join(a.outdir, "figures"); os.makedirs(fig, exist_ok=True)
    tab = os.path.join(a.outdir, "results"); os.makedirs(tab, exist_ok=True)

    # 1 — QQ: real vs permuted null
    try:
        perm = pd.read_csv(a.perm, sep="\t", index_col=0)
        pcol = "pval_beta" if "pval_beta" in perm.columns else "pval_perm"
        lam_r, pr = lam_and_p(perm[pcol]); ex, ob = qq_xy(pr)
        plt.figure(figsize=(5, 5))
        mx = float(max(ex.max(), ob.max()))
        plt.plot([0, mx], [0, mx], ls="--", lw=1, color="grey")
        plt.scatter(ex, ob, s=5, color="#2471a3", label=f"real   λ = {lam_r:.3f}")
        if a.null and os.path.exists(a.null):
            nu = pd.read_csv(a.null, sep="\t", index_col=0)
            nc = "pval_beta" if "pval_beta" in nu.columns else "pval_perm"
            lam_n, pn = lam_and_p(nu[nc]); exn, obn = qq_xy(pn)
            plt.scatter(exn, obn, s=5, color="#c0392b", label=f"null   λ = {lam_n:.3f}")
        plt.xlabel("expected  −log₁₀(p)"); plt.ylabel("observed  −log₁₀(p)")
        plt.title("cis-mQTL permutation QQ — real vs permuted null")
        plt.legend(); plt.tight_layout()
        plt.savefig(f"{fig}/qq_real_vs_null.png", dpi=150); plt.close()
        print("✓ qq_real_vs_null.png")
    except Exception as e:
        print("QQ failed:", e)

    pri = pd.read_csv(a.prioritised, sep="\t")

    # 2 — score convergence
    try:
        from scipy.stats import spearmanr
        sub = pri[["score_weighted", "score_elasticnet"]].dropna()
        rho = spearmanr(sub["score_weighted"], sub["score_elasticnet"]).correlation
        plt.figure(figsize=(5, 5))
        plt.scatter(sub["score_weighted"], sub["score_elasticnet"], s=4, alpha=0.25, color="#2471a3")
        plt.xlabel("weighted-additive score"); plt.ylabel("elastic-net score")
        plt.title(f"Score convergence   (Spearman ρ = {rho:.3f})")
        plt.tight_layout(); plt.savefig(f"{fig}/score_convergence.png", dpi=150); plt.close()
        print("✓ score_convergence.png")
    except Exception as e:
        print("convergence failed:", e)

    # 3 — elastic-net coefficients
    try:
        items = sorted(PILOT_ENET_COEFS.items(), key=lambda kv: kv[1])
        names = [k for k, _ in items]; vals = [v for _, v in items]
        plt.figure(figsize=(6, 4))
        plt.barh(names, vals, color=["#c0392b" if v < 0 else "#2471a3" for v in vals])
        plt.axvline(0, color="grey", lw=0.8)
        plt.xlabel("elastic-net coefficient (predicting PIP)")
        plt.title("What predicts fine-mapping PIP")
        plt.tight_layout(); plt.savefig(f"{fig}/elasticnet_coefficients.png", dpi=150); plt.close()
        print("✓ elasticnet_coefficients.png")
    except Exception as e:
        print("coef plot failed:", e)

    # 4 — SV-type breakdown (flagged pairs)
    try:
        if a.sv and os.path.exists(a.sv):
            sv = pd.read_csv(a.sv, sep="\t")
            if "sv_implicated" in sv.columns:
                sv = sv[sv["sv_implicated"].astype(str).isin(["True", "true", "1"])]
            counts = sv["svtype"].value_counts()
            plt.figure(figsize=(5, 4))
            plt.bar(counts.index.astype(str), counts.values, color="#16a085")
            for i, v in enumerate(counts.values):
                plt.text(i, v, str(v), ha="center", va="bottom")
            plt.ylabel("flagged CpG–SV pairs"); plt.title("SV types implicated in cis-mQTLs")
            plt.tight_layout(); plt.savefig(f"{fig}/sv_type_breakdown.png", dpi=150); plt.close()
            print("✓ sv_type_breakdown.png")
    except Exception as e:
        print("sv plot failed:", e)

    # 5 — PIP distribution
    try:
        if a.susie and os.path.exists(a.susie):
            su = pd.read_csv(a.susie, sep="\t")
            pip = pd.to_numeric(su.get("pip"), errors="coerce").dropna()
            plt.figure(figsize=(5, 4))
            plt.hist(pip, bins=40, color="#8e44ad")
            plt.yscale("log")
            plt.xlabel("PIP"); plt.ylabel("credible-set variants (log)")
            plt.title("Fine-mapping PIP distribution")
            plt.tight_layout(); plt.savefig(f"{fig}/pip_distribution.png", dpi=150); plt.close()
            print("✓ pip_distribution.png")
    except Exception as e:
        print("pip plot failed:", e)

    # top table
    try:
        scol = "score_weighted" if "score_weighted" in pri.columns else pri.columns[-1]
        top = pri.sort_values(scol, ascending=False).head(a.top).copy()
        if a.functional and os.path.exists(a.functional):
            fa = pd.read_csv(a.functional, sep="\t")
            if "nearest_gene" in fa.columns:
                top["nearest_gene"] = top["phenotype_id"].map(fa.set_index("cpg")["nearest_gene"])
        keep = [c for c in ["phenotype_id", "variant_id", "pip", "pval_nominal", "slope",
                            "z_sv_flag", "nearest_gene", "score_weighted", "score_elasticnet"]
                if c in top.columns]
        top = top[keep]
        for c in ("pip", "slope", "z_sv_flag", "score_weighted", "score_elasticnet"):
            if c in top.columns:
                top[c] = pd.to_numeric(top[c], errors="coerce").round(3)
        if "pval_nominal" in top.columns:
            top["pval_nominal"] = pd.to_numeric(top["pval_nominal"], errors="coerce").map(
                lambda x: f"{x:.2e}" if pd.notna(x) else "")
        top.to_csv(f"{tab}/chr22.top{a.top}_prioritised.tsv", sep="\t", index=False)
        with open(f"{tab}/chr22.top{a.top}_prioritised.md", "w") as f:
            f.write(to_md(top))
        print(f"✓ chr22.top{a.top}_prioritised.tsv + .md")
    except Exception as e:
        print("top table failed:", e)

    print("done →", a.outdir)


if __name__ == "__main__":
    main()
