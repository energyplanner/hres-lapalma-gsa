"""
Solver benchmark for the GSA engine (one winter representative week).

Compares formulations and settings on three points of the parameter space and
prints solve time, final gap and the KPIs that matter. Use it to choose the
settings in config_paper.SOLVER / GSA_ENGINE before launching Morris and Sobol'.

Usage:  python bench_solver.py              (≈ 10-20 min, 1 thread per solve)
        python bench_solver.py --limit 120
"""

import argparse, time, warnings
import numpy as np
import pandas as pd

import config_paper as C
from gsa_engine import load_data, HRESEngine

warnings.filterwarnings("ignore")

POINTS = {
    "reference 2023": {},
    "BESS scenario": {"pv_capacity_kw": 30_000, "wind_capacity_kw": 25_000,
                      "batt_capacity_kwh": 80_000, "batt_duration_h": 80 / 30},
    "high RE, small diesel": {"pv_capacity_kw": 50_000, "wind_capacity_kw": 40_000,
                              "batt_capacity_kwh": 150_000, "batt_duration_h": 150 / 40,
                              "diesel_capacity_kw": 35_000, "load_scale": 1.2},
}
CONFIGS = {
    "old min up/down, 10 units": dict(fleet="scaled_real", updown_form="old", mip_focus=0),
    "RT, 10 units":              dict(fleet="scaled_real", updown_form="rt", mip_focus=0),
    "RT + MIPFocus 2, 10 units": dict(fleet="scaled_real", updown_form="rt", mip_focus=2),
    "RT, 3 equal units":         dict(fleet="fast3", updown_form="rt", mip_focus=0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=float, default=60, help="time limit per week [s]")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    data = load_data()
    rows = []
    cfgs = {k: v for k, v in CONFIGS.items() if not a.test or v["fleet"] == "fast3"}
    for cname, cfg in cfgs.items():
        eng = HRESEngine(data, mode="weeks", time_limit=a.limit, threads=1, **cfg)
        eng.blocks = eng.blocks[:1]                     # winter week only
        if a.test:
            eng.blocks = [(slice(eng.blocks[0][0].start, eng.blocks[0][0].start + 24), eng.blocks[0][1] * 7)]
        for pname, ov in POINTS.items():
            names = list(ov) or ["load_scale"]
            x = np.array([ov[n] for n in ov] or [1.0], float)
            t0 = time.time()
            k = eng.evaluate(x, names)
            rows.append({"config": cname, "point": pname,
                         "time_s": round(time.time() - t0, 1),
                         "gap_%": round(100 * k["mip_gap_max"], 2),
                         "CO2_kt_yr": round(k["CO2_kt"], 2),
                         "LPSP_%": round(100 * k["LPSP"], 4),
                         "REF_%": round(100 * k["REF"], 2)})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(C.OUTPUT_DIR / "bench_solver.csv", index=False)
    print("\n", df.pivot(index="config", columns="point", values=["time_s", "gap_%"]).to_string())


if __name__ == "__main__":
    main()
