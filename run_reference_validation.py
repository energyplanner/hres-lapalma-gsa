"""
=============================================================================
Reference-case validation and BESS scenario (Reviewers 1 and 3, Section 4.1)
=============================================================================
Full-year (8760 h) chained unit commitment with the real Los Guinchos engines
(config_paper.LOS_GUINCHOS_ENGINES_MW) and the 2023 renewable fleet.

Outputs in outputs_revision/:
  validation_annual.csv     model vs official 2023 values (Table 2) + extra indicators
  validation_monthly.csv    monthly demand, PV, wind and diesel energy [GWh]
  bess_annual.csv           same indicators for the BESS scenario
  fig3_dispatch_reference.png / fig4_dispatch_bess.png
                            winter and summer weeks, common axes, SOC panel

Usage:  python run_reference_validation.py            (needs full Gurobi licence)
=============================================================================
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config_paper as C
from gsa_engine import load_data, HRESEngine

OUT = C.OUTPUT_DIR
plt.rcParams.update({"font.size": 12})
BESS_SCENARIO = {"pv_capacity_kw": 30_000, "wind_capacity_kw": 25_000,
                 "batt_capacity_kwh": 80_000, "batt_duration_h": 80 / 30}   # 80 MWh / 30 MW
WEEKS = {"winter": 17, "summer": 198}          # start day of year (same as GSA weeks)


LAST_RAW = {}


def run_case(eng, overrides):
    names = list(overrides)
    k, raw, p = eng.run(np.array([overrides[n] for n in names], float), names,
                        keep_series=True)
    r = raw[0]
    LAST_RAW["raw"] = raw
    idx = eng.w.index[r["slice"]]
    s = pd.DataFrame({"load": r["load"], "pv": r["p_pv"], "wind": r["p_wind"],
                      "dump": r["p_dump"], "diesel": r["p_dg"], "units_on": r["u_dg"],
                      "ch": r["p_ch"], "dis": r["p_dis"], "soc": r["soc"],
                      "unmet": r["p_unmet"]}, index=idx)
    return k, s, p


def diagnose(name, raw, s):
    """Windows that did not reach the target gap and hours with unserved load."""
    g = np.array(raw[0]["mip_gaps"])
    bad = np.where(g > C.SOLVER["MIPGap"] + 1e-6)[0]
    um = s.unmet[s.unmet > 1]
    print(f"\n[{name}] windows above target gap: {len(bad)}/{len(g)}"
          + (f"  (weeks {bad.tolist()}, gaps {np.round(g[bad], 3).tolist()})" if len(bad) else ""))
    print(f"[{name}] hours with unserved load: {len(um)}, total {um.sum()/1e3:.1f} MWh, "
          f"max {um.max()/1e3 if len(um) else 0:.2f} MW"
          + (f", first at {um.index[0]}" if len(um) else ""))


def indicators(k, s, p):
    fleet_kw = p.diesel_capacity_kw
    return {
        "Demand [GWh]": s.load.sum() / 1e6,
        "PV [GWh]": s.pv.sum() / 1e6, "Wind [GWh]": s.wind.sum() / 1e6,
        "Curtailed [GWh]": s.dump.sum() / 1e6,
        "Diesel [GWh]": s.diesel.sum() / 1e6,
        "Renewable fraction [%]": 100 * k["REF"],
        "LPSP [%]": 100 * k["LPSP"], "LCOE [€/MWh]": 1000 * k["LCOE_eur_kWh"],
        "CO2 [kt]": k["CO2_kt"], "Diesel starts [-]": k["Diesel_starts"],
        "Peak diesel output [MW]": s.diesel.max() / 1e3,
        "Mean units online [-]": s.units_on.mean(),
        "Diesel fleet capacity factor [%]": 100 * s.diesel.mean() / fleet_kw,
        "Battery equivalent cycles [-]": k["Equiv_cycles"],
        "Max MIP gap [-]": k["mip_gap_max"],
    }


def plot_weeks(s, title, path, ymax=None):
    fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharex="col",
                             gridspec_kw={"height_ratios": [3, 1]})
    for c, (season, doy) in enumerate(WEEKS.items()):
        w = s.iloc[(doy - 1) * 24:(doy - 1) * 24 + 168] / 1e3          # MW
        h = np.arange(len(w))
        # Energy-served convention: positive layers are clipped in merit order
        # so that their top equals demand; surplus goes below zero.
        re_used = np.clip(w.pv + w.wind - w.dump, 0, None)
        dg_s = np.minimum(w.diesel, w.load)
        bd_s = np.minimum(w.dis, np.clip(w.load - dg_s, 0, None))
        re_s = np.minimum(re_used, np.clip(w.load - dg_s - bd_s, 0, None))
        frac_pv = np.divide(w.pv, w.pv + w.wind, out=np.zeros(len(w)),
                            where=(w.pv + w.wind) > 1e-6)
        ax = axes[0, c]
        ax.stackplot(h, dg_s, bd_s, re_s * (1 - frac_pv), re_s * frac_pv,
                     colors=["#7f7f7f", "#2ca02c", "#1f77b4", "#ffbf00"],
                     labels=["Diesel", "Battery discharge", "Wind", "PV"])
        ax.stackplot(h, -w.ch, -w.dump, colors=["#98df8a", "#d62728"],
                     labels=["Battery charge", "Curtailment"], alpha=.8)
        ax.plot(h, w.load, "k", lw=1.6, label="Demand")
        ax.axhline(0, color="k", lw=.5)
        ax.set_title(f"{title} — {season} week", fontweight="bold")
        ax.set_ylabel("Power [MW]"); ax.grid(alpha=.3)
        if ymax:
            ax.set_ylim(-ymax * .6, ymax)
        axes[1, c].plot(h, 100 * s.soc.iloc[(doy - 1) * 24:(doy - 1) * 24 + 168].values,
                        color="#2ca02c")
        axes[1, c].set_ylim(0, 100); axes[1, c].set_ylabel("SOC [%]")
        axes[1, c].set_xlabel("Hour of week"); axes[1, c].grid(alpha=.3)
    axes[0, 0].legend(loc="lower left", ncol=4, fontsize=9)
    plt.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fleet", default="scaled_real", choices=["scaled_real", "fast3"])
    ap.add_argument("--time-limit", type=float, default=300,
                    help="Gurobi time limit per weekly window [s]")
    ap.add_argument("--gap", type=float, default=C.SOLVER["MIPGap"])
    a = ap.parse_args()
    eng = HRESEngine(load_data(), mode="year", fleet=a.fleet,
                     time_limit=a.time_limit, mip_gap=a.gap, threads=0)

    k, s, p = run_case(eng, {})                                   # nominal = 2023 system
    diagnose("reference", LAST_RAW["raw"], s)
    ind = indicators(k, s, p)
    R = C.REAL_SYSTEM_2023
    real = {"PV [GWh]": R["E_pv_gwh"], "Wind [GWh]": R["E_wind_gwh"],
            "Diesel [GWh]": R["E_diesel_gwh"] + R["E_gt_gwh"],
            "Renewable fraction [%]": 100 * R["renewable_share"]}
    tab = pd.DataFrame({"Model": ind, "Observed 2023": pd.Series(real)})
    tab["Deviation [%]"] = 100 * (tab.Model / tab["Observed 2023"] - 1)
    tab.round(3).to_csv(OUT / "validation_annual.csv")
    print(tab.round(2).to_string())

    m = (s[["load", "pv", "wind", "diesel", "dump"]].resample("MS").sum() / 1e6)
    m.columns = ["Demand [GWh]", "PV [GWh]", "Wind [GWh]", "Diesel [GWh]", "Curtailed [GWh]"]
    m.round(3).to_csv(OUT / "validation_monthly.csv")

    kb, sb, pb = run_case(eng, BESS_SCENARIO)
    diagnose("BESS", LAST_RAW["raw"], sb)
    pd.Series(indicators(kb, sb, pb)).round(3).to_csv(OUT / "bess_annual.csv")
    ymax = 1.1 * max(s.load.max(), sb.load.max()) / 1e3
    plot_weeks(s, "Reference 2023", OUT / "fig3_dispatch_reference.png", ymax)
    plot_weeks(sb, "BESS scenario", OUT / "fig4_dispatch_bess.png", ymax)
    print("\nBESS scenario:\n", pd.Series(indicators(kb, sb, pb)).round(2).to_string())


if __name__ == "__main__":
    main()
