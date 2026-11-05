"""
=============================================================================
HRES La Palma — single source of truth for the revised manuscript
=============================================================================
Every assumption that appears in the paper (Tables 1, 3, 4, solver settings,
seeds, data sources, reference system) is defined here ONCE. All revision
scripts import from this module, and `export_tables()` writes the parameter
table used in the manuscript, so text and code cannot drift apart again.

Units inside the code: kW, kWh, fractions [-], 1/h, 1/°C.
Units in the exported paper table: MW, MWh, %, %/h, %/°C.

Items marked
before the final runs (see REVISION_NOTES.md).
=============================================================================
"""

from pathlib import Path
import numpy as np
import pandas as pd

OUTPUT_DIR = Path("outputs_revision")
OUTPUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  DATA
# ─────────────────────────────────────────────────────────────────────────────
YEAR = 2023
LOCATION = "fuencaliente"          # lat 28.49 N, lon 17.85 W (LA_PALMA_PRESETS)
DATA_SOURCES = {
    "demand": "REE/ESIOS indicator 1342 (non-peninsular real demand), geo_id 8798, hourly, 2023",
    "solar": "Open-Meteo historical API (ERA5 reanalysis): GHI, DHI, DNI, 2 m temperature, 10 m wind",
    "wind": "Renewables.ninja (MERRA-2) wind speed at 80 m reference height, hourly, 2023",
    "preprocessing": "UTC alignment, linear interpolation of gaps up to 3 h, 29 Feb removed, 8760 h",
}

# ─────────────────────────────────────────────────────────────────────────────
#  REAL SYSTEM (2023) — values to be taken from ONE official source
# ─────────────────────────────────────────────────────────────────────────────
# Los Guinchos diesel engines (10 units). Unit sizes are the commonly reported
# ones; total 82.84 MW. Confirm against REE/ESIOS unit list before submission.
LOS_GUINCHOS_ENGINES_MW = [4.32, 4.32, 4.32, 5.04, 7.52, 7.52,
                           12.30, 12.30, 12.60, 12.60]
GAS_TURBINE_MW = 22.5     # excluded from the UC model (2 GWh in 2023, reserve role)
REAL_SYSTEM_2023 = {      # Table 2 — official values
    "pv_mw": 3.3, "wind_mw": 7.0, "hydro_mw": 0.8,
    "diesel_mw": sum(LOS_GUINCHOS_ENGINES_MW), "gas_turbine_mw": GAS_TURBINE_MW,
    "E_pv_gwh": 4.6, "E_wind_gwh": 19.0, "E_diesel_gwh": 236.0, "E_gt_gwh": 2.0,
    "renewable_share": 0.088,
}

# ─────────────────────────────────────────────────────────────────────────────
#  UNCERTAIN PARAMETERS (Table 3) — independent uniform distributions
# ─────────────────────────────────────────────────────────────────────────────
# name, symbol, lower, upper, nominal, internal unit, paper unit, paper scale, ref
PARAMS = [
    # sizing (nominal = existing 2023 system)
    ("pv_capacity_kw",      "P_PV",     1_000,   60_000,  3_300,  "kW",   "MW",   1e-3, "-"),
    ("wind_capacity_kw",    "P_Wind",   1_000,   50_000,  7_000,  "kW",   "MW",   1e-3, "-"),
    ("batt_capacity_kwh",   "E_batt",       0,  200_000,      0,  "kWh",  "MWh",  1e-3, "-"),
    # battery power follows as E_batt / D_batt (physically consistent designs,
    # independent sampled inputs as required by Morris and Sobol')
    ("batt_duration_h",     "D_batt",     1.0,      6.0,    4.0,  "h",    "h",    1.0,  "-"),
    ("diesel_capacity_kw",  "P_DG",    30_000,   90_000, 82_840,  "kW",   "MW",   1e-3, "-"),
    # resource / generation
    ("wind_hub_height",     "H_hub",       40,      100,     80,  "m",    "m",    1.0,  "-"),
    ("pv_temp_coeff",       "alpha_T", -0.005,  -0.0035, -0.004,  "1/°C", "%/°C", 100,  "[33]"),
    ("pv_inverter_eff",     "eta_inv",   0.93,     0.99,   0.96,  "-",    "-",    1.0,  "[34]"),
    # battery
    ("batt_charge_eff",     "eta_ch",    0.88,     0.98,   0.95,  "-",    "-",    1.0,  "Li-ion refs"),
    ("batt_discharge_eff",  "eta_dis",   0.88,     0.98,   0.95,  "-",    "-",    1.0,  "Li-ion refs"),
    ("batt_soc_min",        "SOC_min",   0.05,     0.30,   0.15,  "-",    "%",    100,  "[27]"),
    ("batt_soc_max",        "SOC_max",   0.85,     1.00,   0.95,  "-",    "%",    100,  "[27]"),
    ("batt_self_discharge", "sigma_self", 8.3e-5, 1.0e-4, 9.0e-5, "1/h",  "%/h",  100,  "[36]"),
    # diesel / system / demand
    ("diesel_min_load",     "P_min",     0.20,     0.40,   0.30,  "-",    "%",    100,  "[28]"),
    ("converter_eff",       "eta_conv",  0.93,     0.99,   0.96,  "-",    "-",    1.0,  "-"),
    ("load_scale",          "f_load",    0.80,     1.30,   1.00,  "-",    "-",    1.0,  "-"),
]

# Parameters used only in the Sobol' stage. The battery is described by its
# energy capacity and its duration; the power rating follows as E/D, so every
# sample is a physically consistent storage design while the sampled inputs
# remain independent (required by the Sobol' estimators).
SOBOL_EXTRA = []
BATTERY_PARAMETRIZATION = "duration"     # 'duration' (E_batt, D_batt) or 'power' (E_batt, P_batt)

NAMES = [p[0] for p in PARAMS]
SYMBOL = {p[0]: p[1] for p in PARAMS + SOBOL_EXTRA}
SYMBOL["batt_power_kw"] = "P_batt"          # derived quantity, used in labels only
NOMINAL = {p[0]: p[4] for p in PARAMS + SOBOL_EXTRA}
BOUNDS = {p[0]: [p[2], p[3]] for p in PARAMS + SOBOL_EXTRA}

MORRIS_PROBLEM = {"num_vars": len(PARAMS), "names": NAMES,
                  "bounds": [[p[2], p[3]] for p in PARAMS]}

# Fixed modelling choices (not sampled) — reported in the implementation section
FIXED = {
    "pv_tilt": 20.0, "pv_azimuth": 180.0,                 # deg (paper said 40 — corrected)
    "pv_system_loss": 0.14,                               # extra loss on top of PVWatts losses
    "pvwatts_losses_pct": dict(soiling=2, shading=0, snow=0, mismatch=2, wiring=2,
                               connections=0.5, lid=1.5, nameplate_rating=1,
                               age=0, availability=0),
    "wind_turbine": "Vestas V90-2.0 MW power curve",
    "wind_ref_height_m": 80, "wind_roughness_m": 0.03,
    "wind_speed_model": "logarithmic",
    "batt_soc_init": 0.50,
}

# ─────────────────────────────────────────────────────────────────────────────
#  ECONOMICS (Table 4) and MILP COSTS
# ─────────────────────────────────────────────────────────────────────────────
ECONOMICS = {
    "capex_pv_eur_kw": 650.0, "capex_wind_eur_kw": 1200.0,
    "capex_batt_eur_kwh": 180.0, "capex_batt_eur_kw": 140.0,
    "capex_diesel_eur_kw": 400.0, "fuel_price_eur_l": 1.20,
    "discount_rate": 0.06, "lifetime_years": 20.0,
    "opex_pct": {"pv": 0.02, "wind": 0.025, "batt": 0.01, "diesel": 0.03},
}
MILP_COSTS = {
    "fuel_price": 1.20,          # €/L
    "diesel_a": 0.08415,         # L/kWh, no-load coefficient (× P_rated × u)
    "diesel_b": 0.246,           # L/kWh, marginal coefficient (× p)
    "startup_cost": 15.0,        # €/start
    "startup_fuel": 0.5,         # L/start
    "unmet_penalty": 22.879,     # €/kWh, VOLL for Spain: 22 879 €/MWh, Resolución DGPEM 7 Jul 2025, BOE-A-2025-14438
    "dump_penalty": 0.0,         # €/kWh
    "batt_deg_cost": 0.03,       # €/kWh throughput
    "co2_per_l": 2.65,           # kg CO2/L
}
UC_CONSTRAINTS = {"ramp_rate": 0.30, "min_up_time": 3, "min_down_time": 2}

# ─────────────────────────────────────────────────────────────────────────────
#  KPIs analysed in the paper
# ─────────────────────────────────────────────────────────────────────────────
KPIS = {
    "LCOE_eur_kWh": "LCOE [€/kWh]",
    "LPSP": "LPSP [-]",
    "REF": "Renewable fraction [-]",
    "Curtailment": "Curtailment [-]",
    "CO2_kt": "CO$_2$ [kt/yr]",
}

# ─────────────────────────────────────────────────────────────────────────────
#  GSA / SOLVER SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
MORRIS = {"trajectories": 20, "num_levels": 4, "seed": SEED,
          "robustness_seeds": [7, 2026],      # repeated screening
          "optimal_candidates": None,         # e.g. 100 → Campolongo optimisation
          "threshold": 0.10,                  # normalised mu* threshold
          "rule": "any"}                      # 'any' KPI or 'mean' over KPIs
SOBOL = {"N": 512, "seed": SEED, "scramble": True, "second_order": True,
         "num_resamples": 1000, "conf_level": 0.95,
         "convergence_N": [64, 128, 256, 512]}
SOLVER = {"solver": "Gurobi", "MIPGap": 0.01, "TimeLimit_s_per_week": 60,
          "Threads_per_worker": 1,
          "updown_form": "rt",        # Rajan-Takriti min up/down inequalities
          "MIPFocus": 2}              # bound focus: best gap/time in bench_solver.py
GSA_ENGINE = {
    "representative_weeks": True,             # 4 seasonal weeks, weighted to one year
    "fleet": "scaled_real",                   # 'scaled_real' (10 units) or 'fast3'
    "weekly_cyclic_soc": True,                # SOC(end of week) = SOC(start)
}


def nominal_vector(names=None):
    names = names or NAMES
    return np.array([NOMINAL[n] for n in names], float)


def export_tables(path=OUTPUT_DIR / "table3_parameters.csv"):
    """Writes Table 3 in paper units (from the same numbers the code uses)."""
    rows = []
    for name, sym, lo, hi, nom, u_in, u_out, sc, ref in PARAMS + SOBOL_EXTRA:
        rows.append({"Parameter": name, "Symbol": sym,
                     "Lower": round(lo * sc, 6), "Upper": round(hi * sc, 6),
                     "Nominal": round(nom * sc, 6), "Unit": u_out,
                     "Distribution": "Uniform, independent", "Ref.": ref,
                     "Nominal inside range": lo <= nom <= hi,
                     "Stage": "Sobol' only" if name in dict((p[0], 1) for p in SOBOL_EXTRA) else "Morris and Sobol'"})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    bad = df[~df["Nominal inside range"]]
    if len(bad):
        print("WARNING nominal outside range:", bad["Parameter"].tolist())
    return df


if __name__ == "__main__":
    print(export_tables().to_string(index=False))


def tex(symbol: str) -> str:
    """'P_Wind' → '$P_{Wind}$', 'eta_conv' → '$\\eta_{conv}$' for figure labels."""
    greek = {"eta": r"\eta", "alpha": r"\alpha", "sigma": r"\sigma"}
    base, _, sub = symbol.partition("_")
    base = greek.get(base, base)
    return f"${base}_{{{sub}}}$" if sub else f"${base}$"


# Parameter categories and colours for figures (same palette as the first submission)
PARAM_CATEGORY = {
    "pv_capacity_kw": "PV", "pv_temp_coeff": "PV", "pv_inverter_eff": "PV",
    "wind_capacity_kw": "Wind", "wind_hub_height": "Wind",
    "batt_capacity_kwh": "Battery", "batt_power_kw": "Battery", "batt_duration_h": "Battery",
    "batt_charge_eff": "Battery", "batt_discharge_eff": "Battery",
    "batt_soc_min": "Battery", "batt_soc_max": "Battery", "batt_self_discharge": "Battery",
    "diesel_capacity_kw": "Diesel", "diesel_min_load": "Diesel",
    "converter_eff": "System", "load_scale": "Demand", "batt_duration_h": "Battery",
}
CATEGORY_COLORS = {"PV": "#E8A838", "Wind": "#3B82F6", "Battery": "#10B981",
                   "Diesel": "#EF4444", "System": "#6B7280", "Demand": "#8B5CF6"}
