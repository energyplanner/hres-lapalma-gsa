"""
=============================================================================
Morris screening for the revised manuscript (Reviewer 1, points 2-4)
=============================================================================
Outputs in outputs_revision/:
  morris_s{seed}_X.npy / _Y.csv          design and model outputs
  morris_table.csv                       mu*, mu*_conf, sigma, normalised mu*
                                         for 16 parameters x 5 KPIs + retained flag
  morris_selection.csv                   screening rule applied to every seed
  morris_robustness.csv                  rank correlation between seeds
  fig5_morris_mu_sigma.png               improved Figure 5
  morris_manifest.json                   versions, settings, seeds

Screening rule (stated in the paper): a parameter is retained for Sobol if its
normalised mu* (mu* / max mu* of that KPI) reaches the threshold for AT LEAST
ONE KPI ('any'), or on average across KPIs ('mean'). Threshold and rule are in
config_paper.MORRIS. The rule is applied to the main seed and to the
robustness seeds; a parameter retained in any seed is reported.

Usage
  python run_morris_revision.py --workers 8
  python run_morris_revision.py --workers 8 --robustness      # + extra seeds
  python run_morris_revision.py --test                         # tiny smoke test
=============================================================================
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr
from SALib.sample import morris as morris_sample
from SALib.analyze import morris as morris_analyze

import config_paper as C
from gsa_runner import evaluate_design, write_manifest

OUT = C.OUTPUT_DIR
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12})


def run_seed(seed, args):
    prob = C.MORRIS_PROBLEM
    r = 2 if args.test else C.MORRIS["trajectories"]
    kw = dict(N=r, num_levels=C.MORRIS["num_levels"], seed=seed)
    if C.MORRIS["optimal_candidates"] and not args.test:
        kw["N"] = C.MORRIS["optimal_candidates"]
        kw["optimal_trajectories"] = C.MORRIS["trajectories"]
    X = morris_sample.sample(prob, **kw)
    xf = OUT / f"morris_s{seed}_X.npy"
    if xf.exists():
        old = np.load(xf)
        if old.shape != X.shape or not np.allclose(old, X):
            raise SystemExit(f"{xf.name} was generated with a different parameter set. Move the old "
                             f"morris_s*_X.npy / morris_s*_Y.csv files out of {OUT} before re-running.")
    np.save(xf, X)
    Y = evaluate_design(X, prob["names"], OUT / f"morris_s{seed}",
                        workers=args.workers, fleet=args.fleet,
                        week_hours=24 if args.test else 168)
    rows = []
    for kpi in C.KPIS:
        y = Y[kpi].to_numpy(float)
        if np.isnan(y).any():                       # failed points → mean (logged)
            y = np.where(np.isnan(y), np.nanmean(y), y)
        Si = morris_analyze.analyze(prob, X, y, num_levels=C.MORRIS["num_levels"],
                                    num_resamples=1000, conf_level=0.95, seed=seed)
        mmax = np.max(Si["mu_star"]) + 1e-300
        for j, n in enumerate(prob["names"]):
            rows.append({"seed": seed, "KPI": kpi, "param": n, "symbol": C.SYMBOL[n],
                         "mu": Si["mu"][j], "mu_star": Si["mu_star"][j],
                         "mu_star_conf": Si["mu_star_conf"][j], "sigma": Si["sigma"][j],
                         "mu_star_norm": Si["mu_star"][j] / mmax,
                         "sigma_over_mu_star": Si["sigma"][j] / (Si["mu_star"][j] + 1e-300)})
    return pd.DataFrame(rows), Y


def select(tab, threshold, rule):
    piv = tab.pivot(index="param", columns="KPI", values="mu_star_norm")
    score = piv.max(axis=1) if rule == "any" else piv.mean(axis=1)
    return score, score[score >= threshold].index.tolist()


def plot_mu_sigma(tab, keep, path):
    kpis = list(C.KPIS)
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5))
    for ax, kpi in zip(axes.flatten(), kpis):
        t = tab[tab.KPI == kpi]
        mmax = t.mu_star.max()
        x, y = t.mu_star / mmax, t.sigma / mmax          # normalised axes
        infl = t.param.isin(keep).to_numpy()
        col = np.array([C.CATEGORY_COLORS[C.PARAM_CATEGORY[p]] for p in t.param])
        ax.errorbar(x[infl], y[infl], xerr=(t.mu_star_conf / mmax)[infl],
                    fmt="none", ecolor="0.6", alpha=0.6, zorder=2)
        ax.scatter(x[infl], y[infl], s=120, c=col[infl], edgecolors="white",
                   linewidths=1.0, zorder=3)
        ax.scatter(x[~infl], y[~infl], s=90, facecolors="white",
                   edgecolors=col[~infl], linewidths=1.6, zorder=3)
        # label only parameters that matter in THIS panel, and keep labels apart
        texts = [ax.text(xi, yi, C.tex(sym), fontsize=12)
                 for xi, yi, sym, f in zip(x, y, t.symbol, infl)
                 if f and (xi >= 0.08 or yi >= 0.08)]
        try:
            from adjustText import adjust_text
            adjust_text(texts, ax=ax, expand=(1.3, 1.6),
                        arrowprops=dict(arrowstyle="-", color="0.5", lw=0.6))
        except ImportError:
            for tx in texts:
                tx.set_position((tx.get_position()[0] + 0.01, tx.get_position()[1] + 0.01))
        lim = max(x.max(), y.max()) * 1.15
        ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.6, label=r"$\sigma=\mu^*$")
        ax.plot([0, lim], [0, .5 * lim], "k:", lw=1, alpha=0.6, label=r"$\sigma=0.5\mu^*$")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel(r"normalised $\mu^*$"); ax.set_ylabel(r"normalised $\sigma$")
        ax.set_title(C.KPIS[kpi], fontweight="bold"); ax.grid(alpha=.3)
    import matplotlib.lines as mlines
    h = [mpatches.Patch(color=c, label=k) for k, c in C.CATEGORY_COLORS.items()]
    h += [mlines.Line2D([], [], marker="o", ls="", ms=10, mfc="0.3", mec="white", label="Retained"),
          mlines.Line2D([], [], marker="o", ls="", ms=9, mfc="white", mec="0.3", label="Fixed at nominal"),
          mlines.Line2D([], [], ls="--", color="k", lw=1, label=r"$\sigma=\mu^*$"),
          mlines.Line2D([], [], ls=":", color="k", lw=1, label=r"$\sigma=0.5\mu^*$")]
    axes.flatten()[-1].legend(handles=h, loc="center", fontsize=12, frameon=False,
                              title="Parameter category", title_fontsize=12)
    axes.flatten()[-1].axis("off")
    plt.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--fleet", default=C.GSA_ENGINE["fleet"], choices=["scaled_real", "fast3"])
    ap.add_argument("--robustness", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--plot-only", action="store_true",
                    help="redraw Figure 5 from morris_table.csv without re-running")
    args = ap.parse_args()
    global OUT
    if args.test:                       # separate folder: never mixes with real runs
        args.fleet = "fast3"
        OUT = C.OUTPUT_DIR / "test"; OUT.mkdir(exist_ok=True)
    if args.plot_only:
        tab = pd.read_csv(OUT / "morris_table.csv")
        man = OUT / "morris_manifest.json"
        import json
        keep = (json.loads(man.read_text())["retained_any_seed"] if man.exists()
                else tab[tab.retained].param.unique().tolist())
        plot_mu_sigma(tab, keep, OUT / "fig5_morris_mu_sigma.png")
        print("✓ outputs_revision/fig5_morris_mu_sigma.png redrawn")
        return

    seeds = [C.MORRIS["seed"]] + (C.MORRIS["robustness_seeds"] if args.robustness else [])
    tabs = [run_seed(s, args)[0] for s in seeds]
    tab = pd.concat(tabs)
    thr, rule = C.MORRIS["threshold"], C.MORRIS["rule"]

    sel_rows, keep_union = [], set()
    for s in seeds:
        score, keep = select(tab[tab.seed == s], thr, rule)
        keep_union |= set(keep)
        for n in C.NAMES:
            sel_rows.append({"seed": s, "param": n, "score": score[n], "retained": n in keep})
    sel = pd.DataFrame(sel_rows)
    sel.to_csv(OUT / "morris_selection.csv", index=False)

    main_tab = tab[tab.seed == seeds[0]].copy()
    _, keep_main = select(main_tab, thr, rule)
    main_tab["retained"] = main_tab.param.isin(keep_main)
    main_tab.to_csv(OUT / "morris_table.csv", index=False)
    wide = main_tab.pivot(index="symbol", columns="KPI", values=["mu_star_norm", "sigma_over_mu_star"])
    wide.round(3).to_csv(OUT / "morris_table_wide.csv")

    if len(seeds) > 1:
        rob = []
        for kpi in C.KPIS:
            a = tab[(tab.seed == seeds[0]) & (tab.KPI == kpi)].set_index("param").mu_star
            for s in seeds[1:]:
                b = tab[(tab.seed == s) & (tab.KPI == kpi)].set_index("param").mu_star
                rob.append({"KPI": kpi, "seed_a": seeds[0], "seed_b": s,
                            "spearman_mu_star": spearmanr(a, b.loc[a.index]).correlation})
        pd.DataFrame(rob).to_csv(OUT / "morris_robustness.csv", index=False)
        print(pd.DataFrame(rob).round(3).to_string(index=False))

    # parameters retained in ANY seed go to Sobol' (conservative choice)
    main_tab["retained"] = main_tab.param.isin(keep_union)
    main_tab.to_csv(OUT / "morris_table.csv", index=False)
    plot_mu_sigma(main_tab, sorted(keep_union), OUT / "fig5_morris_mu_sigma.png")
    write_manifest(OUT / "morris_manifest.json",
                   {"seeds": seeds, "rule": rule, "threshold": thr, "fleet": args.fleet,
                    "delta": C.MORRIS["num_levels"] / (2 * (C.MORRIS["num_levels"] - 1)),
                    "retained_main_seed": keep_main,
                    "retained_any_seed": sorted(keep_union)})
    print(f"\nRule '{rule}', threshold {thr}")
    print("Retained (main seed):", [C.SYMBOL[n] for n in keep_main])
    if len(seeds) > 1:
        print("Retained in any seed:", [C.SYMBOL[n] for n in sorted(keep_union)])


if __name__ == "__main__":
    main()
