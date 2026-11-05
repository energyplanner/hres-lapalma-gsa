"""
=============================================================================
HRES La Palma — evaluation engine for the revised GSA
=============================================================================
Replaces fast_evaluator.py. Corrections with respect to the first submission:

  1. KPIs are computed from ANNUALISED energy totals. Each representative
     week is scaled by (days of its season)/7 and the KPIs are evaluated once
     on the annual totals. The old evaluator averaged weekly KPIs, so the
     annual CAPEX·CRF was divided by one week of energy (LCOE ≈ 52× too
     capital-heavy) and CO2 was reported per week.
  2. pv_temp_coeff and pv_inverter_eff are passed to pvlib. They were sampled
     but never used before, so Morris reported them as negligible by
     construction.
  3. Every parameter not sampled takes its nominal value from config_paper.py
     (no silent defaults from HRESParameters).
  4. Representative weeks are solved with a cyclic SOC condition (the
     battery ends each week with at least its initial SOC), so storage cannot
     borrow free energy within a week.
  5. Diesel fleet: the 10 real Los Guinchos engines scaled to P_DG
     ('scaled_real'), or 3 equal units ('fast3') as an explicitly verified
     approximation.
  6. Solver gap and time are returned with every evaluation.

mode = "weeks" : 4 seasonal weeks (GSA)   |   mode = "year" : 8760 h chained
=============================================================================
"""

import numpy as np
from typing import Dict, List, Optional

import config_paper as C
from hres_model import HRESParameters, compute_kpis
from dispatch_v2 import solve_rolling_multiunit, _default_diesel_fleet
from milp_dispatch import MILPCostParameters
from generation_models import pv_power_pvlib, wind_power_windpowerlib

# start day-of-year and number of days represented by each seasonal week
SEASON_WEEKS = {
    "winter": {"start_doy": 17,  "days": 90},
    "spring": {"start_doy": 105, "days": 92},
    "summer": {"start_doy": 198, "days": 92},
    "autumn": {"start_doy": 288, "days": 91},
}


def load_data(year: int = C.YEAR, location: str = C.LOCATION, mock: bool = False):
    from physical_evaluator import build_lapalma_physical
    return build_lapalma_physical(year=year, location=location, mock=mock)


def build_params(x: Optional[np.ndarray] = None,
                 names: Optional[List[str]] = None) -> HRESParameters:
    """Nominal values from config_paper, then the sampled values on top."""
    p = HRESParameters()
    for n, v in C.NOMINAL.items():
        setattr(p, n, float(v))
    p.batt_soc_init = C.FIXED["batt_soc_init"]
    for k, v in C.ECONOMICS.items():
        setattr(p, k, v)
    p.startup_cost_eur = C.MILP_COSTS["startup_cost"]
    if x is not None:
        for n, v in zip(names, x):
            if n not in C.NOMINAL:
                raise KeyError(f"unknown parameter {n}")
            setattr(p, n, float(v))
    # Battery described by energy and duration: power follows as E / D
    p.batt_power_kw = p.batt_capacity_kwh / max(p.batt_duration_h, 1e-9)
    return p


class HRESEngine:
    def __init__(self, data: Dict, mode: str = "weeks",
                 fleet: str = C.GSA_ENGINE["fleet"],
                 mip_gap: float = C.SOLVER["MIPGap"],
                 time_limit: float = C.SOLVER["TimeLimit_s_per_week"],
                 threads: int = C.SOLVER["Threads_per_worker"],
                 cyclic_week: bool = C.GSA_ENGINE["weekly_cyclic_soc"],
                 week_hours: int = 168,
                 updown_form: str = C.SOLVER["updown_form"],
                 mip_focus: int = C.SOLVER["MIPFocus"]):
        self.w = data["weather"]
        self.lat, self.lon = data["lat"], data["lon"]
        self.load = np.asarray(data["demand_kw"], float)
        self.mode, self.fleet = mode, fleet
        self.cost = MILPCostParameters(**{k: v for k, v in C.MILP_COSTS.items()
                                          if k in MILPCostParameters.__dataclass_fields__})
        self.solver = dict(mip_gap=mip_gap, time_limit_block=time_limit,
                           threads=threads, updown_form=updown_form,
                           mip_focus=mip_focus, **C.UC_CONSTRAINTS)
        self.cyclic_week = cyclic_week
        self.week_hours = week_hours
        self.year_window, self.year_step = 192, 168   # 24 h look-ahead, chained
        n = len(self.load)
        if mode == "weeks":
            self.blocks = [(slice((s["start_doy"] - 1) * 24,
                                  min((s["start_doy"] - 1) * 24 + week_hours, n)),
                            s["days"] * 24 / week_hours)
                           for s in SEASON_WEEKS.values()]
        elif mode == "year":
            self.blocks = [(slice(0, n), 1.0)]
        else:
            raise ValueError("mode must be 'weeks' or 'year'")

    # ── generation ──────────────────────────────────────────────────────────
    def _generation(self, p, sl):
        w = self.w.iloc[sl]
        p_pv = pv_power_pvlib(w, self.lat, self.lon,
                              pv_capacity_kw=p.pv_capacity_kw,
                              tilt=C.FIXED["pv_tilt"], azimuth=C.FIXED["pv_azimuth"],
                              gamma_pdc=p.pv_temp_coeff,
                              system_loss=C.FIXED["pv_system_loss"],
                              inverter_eff=p.pv_inverter_eff)
        p_wind = wind_power_windpowerlib(w, wind_capacity_kw=p.wind_capacity_kw,
                                         hub_height=p.wind_hub_height,
                                         roughness_length=C.FIXED["wind_roughness_m"])
        return p_pv, p_wind

    def _fleet(self, p):
        if self.fleet == "fast3":
            return [p.diesel_capacity_kw / 3.0] * 3
        return _default_diesel_fleet(p.diesel_capacity_kw)

    # ── evaluation ──────────────────────────────────────────────────────────
    def run(self, x=None, names=None, keep_series=False):
        p = build_params(x, names)
        series = {k: [] for k in ["load", "p_pv", "p_wind", "p_diesel", "p_unmet",
                                  "p_dump", "fuel_l", "starts", "soc", "p_batt"]}
        raw, gaps, t_solve, short = [], [], 0.0, 0.0
        for sl, scale in self.blocks:
            load = self.load[sl] * p.load_scale
            p_pv, p_wind = self._generation(p, sl)
            if self.mode == "weeks":
                win, step, cyc = len(load), len(load), self.cyclic_week
            else:
                win, step, cyc = self.year_window, self.year_step, False
            res = solve_rolling_multiunit(p, p_pv, p_wind, load, cost=self.cost,
                                          window=win, step=step,
                                          diesel_fleet_kw=self._fleet(p),
                                          cyclic_window=cyc, **self.solver)
            gaps.append(res["mip_gap_max"]); t_solve += res["solve_time_total"]
            short += res["soc_shortfall_kwh"] * scale
            blk = {"load": load, "p_pv": p_pv, "p_wind": p_wind,
                   "p_diesel": res["p_dg"], "p_unmet": res["p_unmet"],
                   "p_dump": res["p_dump"], "fuel_l": res["fuel_l"],
                   "starts": res["y_dg"], "soc": res["soc"], "p_batt": res["p_batt"]}
            for k in series:
                v = np.asarray(blk[k], float)
                series[k].append(v if k == "soc" else v * scale)
            if keep_series:
                res.update(load=load, p_pv=p_pv, p_wind=p_wind, slice=sl)
                raw.append(res)
        tot = {k: np.concatenate(v) for k, v in series.items()}
        k = compute_kpis(tot, p)
        k["CO2_kt"] = k["CO2_kg"] / 1e6
        k["E_Diesel_GWh"] = k["E_Diesel_kWh"] / 1e6
        k["Diesel_starts"] = float(np.sum(tot["starts"]))
        k["mip_gap_max"] = float(np.max(gaps))
        k["solve_time_s"] = t_solve
        k["SOC_shortfall_MWh"] = short / 1e3
        return (k, raw, p) if keep_series else k

    def evaluate(self, x, names) -> Dict[str, float]:
        return self.run(x, names)


if __name__ == "__main__":
    import time, warnings
    warnings.filterwarnings("ignore")
    data = load_data()
    eng = HRESEngine(data, mode="weeks")
    t0 = time.time()
    k = eng.evaluate(C.nominal_vector(), C.NAMES)
    print(f"nominal (weeks) in {time.time() - t0:.1f}s")
    for kk in list(C.KPIS) + ["mip_gap_max"]:
        print(f"  {kk:<14}{k[kk]:.4f}")
