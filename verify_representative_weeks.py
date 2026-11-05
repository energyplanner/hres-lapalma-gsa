"""
=============================================================================
Verification of the GSA engine against the full-year model
=============================================================================
The GSA evaluates each sample on four seasonal representative weeks. This
script quantifies the approximation: M quasi-random points of the Sobol'
space are evaluated with the 4-week engine and with the full-year (8760 h)
chained model, and the agreement of every KPI is reported.

Outputs in outputs_revision/:
  verify_weeks_vs_year.csv           paired KPI values
  verify_weeks_vs_year_summary.csv   R², Spearman rank correlation, mean |error|
  fig_verify_weeks_vs_year.png       parity plots

A Spearman correlation close to one means the sensitivity rankings obtained
with the 4-week engine are preserved by the annual model.

Usage:  python verify_representative_weeks.py --M 32 --workers 8
        python verify_representative_weeks.py --M 32 --compare-fleets   (fast3 vs scaled_real)
=============================================================================
"""

import argparse, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, qmc

import config_paper as C
from gsa_runner import evaluate_design

OUT = C.OUTPUT_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--compare-fleets", action="store_true")
    a = ap.parse_args()

    man = OUT / "morris_manifest.json"
    keep = json.loads(man.read_text())["retained_main_seed"] if man.exists() else C.NAMES
    idx = [C.NAMES.index(n) for n in keep]
    lo = np.array([C.MORRIS_PROBLEM["bounds"][i][0] for i in idx])
    hi = np.array([C.MORRIS_PROBLEM["bounds"][i][1] for i in idx])
    X = qmc.scale(qmc.Sobol(len(keep), scramble=True, seed=C.SEED).random(a.M), lo, hi)

    if a.compare_fleets:
        A = evaluate_design(X, keep, OUT / "verify_fast3", mode="weeks", fleet="fast3", workers=a.workers)
        B = evaluate_design(X, keep, OUT / "verify_scaledreal", mode="weeks", fleet="scaled_real", workers=a.workers)
        la, lb, tag = "3 equal units", "10 scaled real units", "fleets"
    else:
        A = evaluate_design(X, keep, OUT / "verify_weeks", mode="weeks", workers=a.workers)
        B = evaluate_design(X, keep, OUT / "verify_year", mode="year", workers=a.workers)
        la, lb, tag = "4 representative weeks", "full year", "weeks_vs_year"

    rows, fig, axes = [], *plt.subplots(1, len(C.KPIS), figsize=(4.2 * len(C.KPIS), 4.2))
    pair = pd.DataFrame(X, columns=keep)
    for ax, kpi in zip(axes, C.KPIS):
        x, y = A[kpi].to_numpy(float), B[kpi].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        r2 = 1 - np.sum((y - x) ** 2) / np.sum((y - y.mean()) ** 2)
        rows.append({"KPI": kpi, "R2": r2, "spearman": spearmanr(x, y).correlation,
                     "mean_abs_error": np.mean(np.abs(x - y)),
                     "mean_rel_error_pct": 100 * np.mean(np.abs(x - y) / (np.abs(y) + 1e-12))})
        pair[f"{kpi}_A"], pair[f"{kpi}_B"] = A[kpi].values, B[kpi].values
        ax.scatter(y, x, s=25); lim = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lim, lim, "k--", lw=1); ax.set_xlabel(lb); ax.set_ylabel(la)
        ax.set_title(f"{C.KPIS[kpi]}\nR²={r2:.3f}"); ax.grid(alpha=.3)
    plt.tight_layout(); fig.savefig(OUT / f"fig_verify_{tag}.png", dpi=300, bbox_inches="tight")
    pair.to_csv(OUT / f"verify_{tag}.csv", index=False)
    summ = pd.DataFrame(rows); summ.to_csv(OUT / f"verify_{tag}_summary.csv", index=False)
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
