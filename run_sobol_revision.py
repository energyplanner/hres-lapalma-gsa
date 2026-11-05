"""
=============================================================================
Sobol' analysis for the revised manuscript (Reviewer 1, points 2-5; R3)
=============================================================================
Sampling   : SALib Saltelli/Sobol' sequence, scrambled, fixed seed,
             second-order terms, N base samples (power of two)
Estimators : Saltelli et al. (2010) for S1 and ST, Saltelli (2002) for S2,
             bootstrap confidence intervals (num_resamples, 95 %)
Convergence: the Sobol' design is built so that the first n(2D+2) rows form a
             complete design of base size n. Indices are therefore recomputed
             for n = 64, 128, 256, 512 from the same run, at no extra cost.

Outputs in outputs_revision/:
  sobol_X.npy, sobol_Y.csv             design and outputs (resumable)
  sobol_indices.csv                    S1, ST with CI, ST-S1, per KPI and parameter
  sobol_additivity.csv                 sum S1 per KPI (additive vs interaction regime)
  sobol_S2.csv                         all pairs with CI and significance flag
  sobol_convergence.csv                indices for every n
  fig6_sobol_indices.png               Figure 6 with 95 % CI
  fig_sobol_S2.png                     dominant pairwise interactions
  fig_sobol_convergence.png            convergence of ST
  fig_summary_heatmap.png              ST of every parameter for every KPI
  sobol_manifest.json

Usage
  python run_sobol_revision.py --workers 8               # keep list from Morris
  python run_sobol_revision.py --workers 8 --keep P_PV P_Wind E_batt ...
  python run_sobol_revision.py --analyze-only            # re-analyse saved runs
  python run_sobol_revision.py --test
=============================================================================
"""

import argparse, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

import config_paper as C
from gsa_runner import evaluate_design, write_manifest

OUT = C.OUTPUT_DIR
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12})
SYM2NAME = {v: k for k, v in C.SYMBOL.items()}


def build_problem(keep):
    keep = list(keep)
    if "batt_power_kw" in keep:          # Morris results from the old parametrization
        # E_batt and P_batt are physically linked: sample E_batt and the duration
        keep[keep.index("batt_power_kw")] = "batt_duration_h"
        if "batt_capacity_kwh" not in keep:
            keep.insert(keep.index("batt_duration_h"), "batt_capacity_kwh")
    order = ["pv_capacity_kw", "wind_capacity_kw", "wind_hub_height", "batt_capacity_kwh",
             "batt_duration_h", "batt_power_kw", "diesel_capacity_kw", "converter_eff", "load_scale"]
    keep = sorted(keep, key=lambda n: order.index(n) if n in order else 99)
    return {"num_vars": len(keep), "names": keep,
            "bounds": [C.BOUNDS[n] for n in keep]}


def analyze(problem, Y, N_list, seed):
    D = problem["num_vars"]
    step = 2 * D + 2
    rows, s2rows = [], []
    for n in N_list:
        for kpi in C.KPIS:
            y = Y[kpi].to_numpy(float)[: n * step]
            if np.isnan(y).any():
                y = np.where(np.isnan(y), np.nanmean(y), y)
            Si = sobol_analyze.analyze(problem, y, calc_second_order=True,
                                       num_resamples=C.SOBOL["num_resamples"],
                                       conf_level=C.SOBOL["conf_level"], seed=seed)
            for j, p in enumerate(problem["names"]):
                s1r, str_ = Si["S1"][j], Si["ST"][j]
                c1, cT = Si["S1_conf"][j], Si["ST_conf"][j]
                rows.append({"N": n, "KPI": kpi, "param": p, "symbol": C.SYMBOL[p],
                             # raw estimator output (can fall outside [0, 1] by sampling error)
                             "S1_raw": s1r, "ST_raw": str_,
                             # reported values: indices are variance ratios, so they lie in [0, 1]
                             "S1": float(np.clip(s1r, 0, 1)), "S1_conf": c1,
                             "ST": float(np.clip(str_, 0, 1)), "ST_conf": cT,
                             "ST_minus_S1": float(max(np.clip(str_, 0, 1) - np.clip(s1r, 0, 1), 0)),
                             # True if the CI does not reach the valid range → not converged
                             "S1_out_of_range": bool(s1r + c1 < 0 or s1r - c1 > 1),
                             "ST_out_of_range": bool(str_ + cT < 0 or str_ - cT > 1),
                             # S1 <= ST in theory; flag if violated beyond both CIs
                             "S1_gt_ST": bool(s1r - c1 > str_ + cT)})
                if n == N_list[-1]:
                    for k in range(j + 1, D):
                        v, c = Si["S2"][j, k], Si["S2_conf"][j, k]
                        s2rows.append({"KPI": kpi, "pair": f"{C.SYMBOL[p]} × {C.SYMBOL[problem['names'][k]]}",
                                       "S2_raw": v, "S2": float(np.clip(v, 0, 1)), "S2_conf": c,
                                       "significant": (v - c) > 0,
                                       "out_of_range": bool(v + c < 0)})
    return pd.DataFrame(rows), pd.DataFrame(s2rows)


def figures(ind, s2, conv, problem):
    kpis, syms = list(C.KPIS), [C.SYMBOL[n] for n in problem["names"]]
    lbl = [C.tex(s) for s in syms]
    x = np.arange(len(syms))

    # Figure 6 — S1 and ST with 95 % CI
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    for ax, kpi in zip(axes.flatten(), kpis):
        t = ind[ind.KPI == kpi].set_index("symbol").loc[syms]
        def err(v, c):          # error bars clipped to the admissible range [0, 1]
            return np.vstack([v - np.clip(v - c, 0, 1), np.clip(v + c, 0, 1) - v])
        ax.bar(x - .2, t.S1, .4, yerr=err(t.S1.values, t.S1_conf.values), capsize=3,
               color="#1f4e79", label="$S_i$ (first order)")
        ax.bar(x + .2, t.ST, .4, yerr=err(t.ST.values, t.ST_conf.values), capsize=3,
               color="#9dc3e6", edgecolor="#1f4e79", label="$S_{Ti}$ (total order)")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x); ax.set_xticklabels(lbl, rotation=40, ha="right")
        ax.axhline(0, color="k", lw=.6); ax.set_ylabel("Sobol' index")
        ax.set_title(C.KPIS[kpi], fontweight="bold"); ax.grid(axis="y", alpha=.3)
    axes.flatten()[0].legend(fontsize=10); axes.flatten()[-1].axis("off")
    plt.tight_layout(); fig.savefig(OUT / "fig6_sobol_indices.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    # Second-order interactions
    D = len(syms)
    fig, axes = plt.subplots(2, 3, figsize=(17, 11))
    for ax, kpi in zip(axes.flatten(), kpis):
        M = np.full((D, D), np.nan)
        t = s2[s2.KPI == kpi]
        for _, r in t.iterrows():
            a, b = r.pair.split(" × "); i, j = syms.index(a), syms.index(b)
            M[i, j] = M[j, i] = r.S2
        im = ax.imshow(np.clip(M, 0, None), cmap="Blues", vmin=0, vmax=max(.05, np.nanmax(M)))
        for _, r in t[t.significant].iterrows():
            a, b = r.pair.split(" × "); i, j = syms.index(a), syms.index(b)
            ax.text(j, i, f"{r.S2:.2f}", ha="center", va="center", fontsize=9)
        ax.set_xticks(range(D)); ax.set_yticks(range(D))
        ax.set_xticklabels(lbl, rotation=40, ha="right"); ax.set_yticklabels(lbl)
        ax.set_title(C.KPIS[kpi], fontweight="bold"); fig.colorbar(im, ax=ax, fraction=.046)
    axes.flatten()[-1].axis("off")
    fig.suptitle("Second-order Sobol' indices (values shown where the 95 % CI excludes zero)")
    plt.tight_layout(); fig.savefig(OUT / "fig_sobol_S2.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    # Convergence of ST
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    for ax, kpi in zip(axes.flatten(), kpis):
        for s in syms:
            t = conv[(conv.KPI == kpi) & (conv.symbol == s)]
            ax.plot(t.N, t.ST, marker="o", label=C.tex(s))
            ax.fill_between(t.N, np.clip(t.ST - t.ST_conf, 0, 1),
                            np.clip(t.ST + t.ST_conf, 0, 1), alpha=.12)
        ax.set_xscale("log", base=2); ax.set_xlabel("base sample size N")
        ax.set_ylabel("$S_{Ti}$"); ax.set_ylim(0, 1.05)
        ax.set_title(C.KPIS[kpi], fontweight="bold"); ax.grid(alpha=.3)
    axes.flatten()[0].legend(fontsize=9, ncol=2); axes.flatten()[-1].axis("off")
    plt.tight_layout(); fig.savefig(OUT / "fig_sobol_convergence.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    # Summary heat map (all KPIs at a glance)
    H = ind.pivot(index="KPI", columns="symbol", values="ST").loc[kpis, syms]
    fig, ax = plt.subplots(figsize=(1.2 * D + 2, 4.2))
    im = ax.imshow(np.clip(H.values, 0, 1), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(H.shape[0]):
        for j in range(D):
            v = H.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > .6 else "black", fontsize=10)
    ax.set_xticks(range(D)); ax.set_xticklabels(lbl)
    ax.set_yticks(range(len(kpis))); ax.set_yticklabels([C.KPIS[k] for k in kpis])
    fig.colorbar(im, ax=ax, label="$S_{Ti}$"); plt.tight_layout()
    fig.savefig(OUT / "fig_summary_heatmap.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--N", type=int, default=C.SOBOL["N"])
    ap.add_argument("--seed", type=int, default=C.SOBOL["seed"])
    ap.add_argument("--fleet", default=C.GSA_ENGINE["fleet"], choices=["scaled_real", "fast3"])
    ap.add_argument("--keep", nargs="+", help="symbols, e.g. P_PV P_Wind ... (default: Morris result)")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    if a.test:
        a.fleet, a.N = "fast3", 8
        OUT = C.OUTPUT_DIR / "test"; OUT.mkdir(exist_ok=True)

    if a.keep:
        keep = [SYM2NAME[s] for s in a.keep]
    else:
        man = json.loads((OUT / "morris_manifest.json").read_text())
        keep = man.get("retained_any_seed") or man["retained_main_seed"]
    problem = build_problem(keep)
    keep = problem["names"]
    tag = f"sobol_s{a.seed}"
    xfile = OUT / f"{tag}_X.npy"
    if xfile.exists():
        X = np.load(xfile)
        if X.shape[1] != problem["num_vars"]:
            raise SystemExit(f"{xfile.name} belongs to a different parameter set; move it away first.")
    else:
        X = sobol_sample.sample(problem, a.N, calc_second_order=True,
                                scramble=C.SOBOL["scramble"], seed=a.seed)
        np.save(xfile, X)
    print(f"[Sobol'] D={problem['num_vars']} N={a.N} → {len(X)} evaluations: "
          f"{[C.SYMBOL[n] for n in keep]}")

    if not a.analyze_only:
        Y = evaluate_design(X, keep, OUT / tag, workers=a.workers, fleet=a.fleet,
                            week_hours=24 if a.test else 168)
    else:
        Y = pd.read_csv(OUT / f"{tag}_Y.csv").set_index("i")
    if len(Y) < len(X):
        print(f"Only {len(Y)}/{len(X)} evaluations available — run again to finish.")
        return

    N_list = [n for n in C.SOBOL["convergence_N"] if n < a.N] + [a.N]
    if a.test:
        N_list = [4, 8]
    conv, s2 = analyze(problem, Y, N_list, a.seed)
    ind = conv[conv.N == a.N]
    ind.to_csv(OUT / "sobol_indices.csv", index=False)
    conv.to_csv(OUT / "sobol_convergence.csv", index=False)
    s2.sort_values(["KPI", "S2"], ascending=[True, False]).to_csv(OUT / "sobol_S2.csv", index=False)
    add = ind.groupby("KPI").agg(sum_S1=("S1", "sum"), sum_ST=("ST", "sum")).loc[list(C.KPIS)]
    add.to_csv(OUT / "sobol_additivity.csv")
    figures(ind, s2, conv, problem)

    gap = Y["mip_gap_max"]
    write_manifest(OUT / "sobol_manifest.json",
                   {"keep": keep, "N": a.N, "seed": a.seed, "fleet": a.fleet,
                    "evaluations": len(X), "failed": int((Y["ok"] == 0).sum()),
                    "mip_gap_max_over_runs": float(gap.max()),
                    "mip_gap_mean_over_runs": float(gap.mean()),
                    "estimators": "Saltelli 2010 (S1, ST), Saltelli 2002 (S2), bootstrap CI"})
    bad = ind[ind.S1_out_of_range | ind.ST_out_of_range | ind.S1_gt_ST]
    if len(bad):
        print("\nWARNING: estimates outside [0, 1] or with S1 > ST beyond their CIs (increase N):")
        print(bad[["KPI", "symbol", "S1_raw", "S1_conf", "ST_raw", "ST_conf"]].round(3).to_string(index=False))
    print("\nSum of first-order indices (≈1 additive, ≪1 interaction-dominated):")
    print(add.round(3).to_string())
    print("\nDominant significant interactions:")
    print(s2[s2.significant].sort_values("S2", ascending=False).groupby("KPI").head(3)
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
