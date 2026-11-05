"""
=============================================================================
HRES La Palma — Dataset + Evaluador con pvlib/windpowerlib
=============================================================================
Integra el pipeline definitivo:

  Open-Meteo (GHI/DHI/DNI/viento/temp)  +  ESIOS (demanda La Palma)
        │                                        │
        ▼                                        ▼
  pvlib PVWatts (PV)  +  windpowerlib (eólica, curva real)
        │
        ▼
  series de generación [kW]  →  despacho MILP  →  KPIs  →  GSA

A diferencia del evaluador basado en CF, aquí el dataset guarda el WEATHER
CRUDO y la generación se reconstruye en cada evaluación con pvlib/windpowerlib.
Así, los parámetros técnicos (tilt, azimut, gamma_pdc, turbine_type,
hub_height, roughness) SÍ influyen en el GSA.

Parámetros de diseño/técnicos disponibles para Morris/Sobol/NSGA:
    pv_capacity_kw, wind_capacity_kw, batt_capacity_kwh, batt_power_kw,
    diesel_capacity_kw, pv_tilt, pv_azimuth, pv_gamma_pdc, pv_system_loss,
    wind_hub_height, wind_roughness, batt_charge_eff, batt_discharge_eff,
    batt_soc_min, ... (todo lo que el despacho/generación acepten)
=============================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from hres_model import HRESParameters, compute_kpis
from dispatch_v2 import dispatch_from_dataset
from milp_dispatch import MILPCostParameters
from generation_models import (get_weather_openmeteo, pv_power_pvlib,
                               wind_power_windpowerlib, _mock_weather)
from lapalma_data import fetch_lapalma_demand, LA_PALMA_PRESETS


# ─────────────────────────────────────────────────────────────────────────────
#  CONSTRUCTOR DE DATASET (weather crudo + demanda)
# ─────────────────────────────────────────────────────────────────────────────

def build_lapalma_physical(year: int = 2019,
                           location: str = "fuencaliente",
                           esios_token: Optional[str] = None,
                           geo_id: Optional[int] = None,
                           use_cache: bool = True,
                           mock: bool = False) -> Dict:
    """
    Construye el dataset para el motor pvlib/windpowerlib.

    Devuelve un dict con:
        weather   : DataFrame Open-Meteo (GHI/DHI/DNI/viento/temp/pressure)
        demand_kw : Series de demanda real de La Palma [kW]
        lat, lon  : coordenadas del emplazamiento
    Las tres series alineadas por timestamp y recortadas a 8760 h.
    """
    preset = LA_PALMA_PRESETS[location]
    lat, lon = preset["lat"], preset["lon"]

    if mock:
        weather = _mock_weather(year=year)
        weather["wind_speed_ninja"] = weather["wind_speed_100m"].values * 2.5  # mock realista
        weather.attrs["ninja_ref_height"] = 80
        # demanda mock ~40 MW pico
        from lapalma_data import build_lapalma_mock
        dem = build_lapalma_mock(year=year, peak_kw=40_000)["demand_kw"]
        dem.index = weather.index[:len(dem)]
    else:
        weather = get_weather_openmeteo(lat, lon, year, use_cache=use_cache)
        # Viento realista de Renewables.ninja (Open-Meteo subestima en islas)
        from generation_models import get_ninja_wind_speed
        wind_ninja = get_ninja_wind_speed(
            lat, lon, year, ref_height=80.0,
            turbine=preset.get("wind_turbine", "Vestas V90 2000"),
            use_cache=use_cache)
        weather["wind_speed_ninja"] = wind_ninja.reindex(weather.index).interpolate(limit=3)
        weather.attrs["ninja_ref_height"] = 80
        dem_df = fetch_lapalma_demand(year, geo_id=geo_id,
                                      esios_token=esios_token, use_cache=use_cache)
        dem = dem_df["demand_mw"] * 1000.0  # MW → kW (isla completa)

    # Alinear por timestamp
    df = weather.copy()
    df.attrs["ninja_ref_height"] = weather.attrs.get("ninja_ref_height", 80)
    df["demand_kw"] = dem.reindex(df.index).interpolate(limit=3)
    df = df.dropna(subset=["demand_kw"])
    # Recortar 29-feb
    df = df[~((df.index.month == 2) & (df.index.day == 29))]

    return {"weather": df, "demand_kw": df["demand_kw"].values,
            "lat": lat, "lon": lon, "location": location}


# ─────────────────────────────────────────────────────────────────────────────
#  EVALUADOR FÍSICO (pvlib + windpowerlib + MILP)
# ─────────────────────────────────────────────────────────────────────────────

class HRESPhysicalEvaluator:
    """
    Evaluador que reconstruye la generación con pvlib/windpowerlib en cada
    llamada, de modo que los parámetros técnicos influyan en el GSA.
    """

    def __init__(self, data: Dict, method: str = "annual",
                 cost: Optional[MILPCostParameters] = None,
                 turbine_type: str = "V90/2000",
                 **dispatch_kwargs):
        self.weather = data["weather"]
        self.lat, self.lon = data["lat"], data["lon"]
        self.load = data["demand_kw"]
        self.method = method
        self.cost = cost or MILPCostParameters()
        self.turbine_type = turbine_type
        self.dispatch_kwargs = dispatch_kwargs

    def evaluate(self, param_vector: np.ndarray,
                 param_names: List[str]) -> Dict[str, float]:
        # Construir parámetros
        p = HRESParameters()
        # Atributos extra (no estándar de HRESParameters) para generación
        extra = {"pv_tilt": 20.0, "pv_azimuth": 180.0, "pv_gamma_pdc": -0.004,
                 "pv_system_loss": 0.14, "wind_roughness": 0.03,
                 "load_scale": 1.0}
        for name, val in zip(param_names, param_vector):
            if hasattr(p, name):
                setattr(p, name, float(val))
            else:
                extra[name] = float(val)

        # ── Generación con librerías validadas ─────────────────────────────
        p_pv = pv_power_pvlib(
            self.weather, self.lat, self.lon,
            pv_capacity_kw=p.pv_capacity_kw,
            tilt=extra.get("pv_tilt", 20.0),
            azimuth=extra.get("pv_azimuth", 180.0),
            gamma_pdc=extra.get("pv_gamma_pdc", -0.004),
            system_loss=extra.get("pv_system_loss", 0.14))

        p_wind = wind_power_windpowerlib(
            self.weather, wind_capacity_kw=p.wind_capacity_kw,
            turbine_type=self.turbine_type,
            hub_height=getattr(p, "wind_hub_height", 80.0),
            roughness_length=extra.get("wind_roughness", 0.03))

        load = self.load * getattr(p, "load_scale", extra.get("load_scale", 1.0))

        # ── Despacho MILP ──────────────────────────────────────────────────
        df_pt = pd.DataFrame({
            "pv_cf": p_pv / max(p.pv_capacity_kw, 1e-9),
            "wind_cf": p_wind / max(p.wind_capacity_kw, 1e-9),
            "demand_kw": load,
        }, index=self.weather.index)

        res = dispatch_from_dataset(p, df_pt, cost=self.cost,
                                    method=self.method, **self.dispatch_kwargs)

        kpi_in = {
            "load": load, "p_pv": p_pv, "p_wind": p_wind,
            "p_diesel": res["p_dg"], "p_unmet": res["p_unmet"],
            "p_dump": res["p_dump"], "fuel_l": res["fuel_l"],
            "starts": res["y_dg"], "soc": res["soc"], "p_batt": res["p_batt"],
        }
        return compute_kpis(kpi_in, p)

    def dispatch_full(self, param_vector: np.ndarray, param_names: List[str]):
        """Como evaluate pero devuelve también las series de despacho (para plots)."""
        p = HRESParameters()
        extra = {"pv_tilt": 20.0, "pv_azimuth": 180.0, "pv_gamma_pdc": -0.004,
                 "pv_system_loss": 0.14, "wind_roughness": 0.03, "load_scale": 1.0}
        for name, val in zip(param_names, param_vector):
            if hasattr(p, name):
                setattr(p, name, float(val))
            else:
                extra[name] = float(val)

        p_pv = pv_power_pvlib(self.weather, self.lat, self.lon,
                              pv_capacity_kw=p.pv_capacity_kw,
                              tilt=extra["pv_tilt"], azimuth=extra["pv_azimuth"],
                              gamma_pdc=extra["pv_gamma_pdc"],
                              system_loss=extra["pv_system_loss"])
        p_wind = wind_power_windpowerlib(self.weather, wind_capacity_kw=p.wind_capacity_kw,
                                         turbine_type=self.turbine_type,
                                         hub_height=getattr(p, "wind_hub_height", 80.0),
                                         roughness_length=extra["wind_roughness"])
        load = self.load * getattr(p, "load_scale", extra.get("load_scale", 1.0))
        df_pt = pd.DataFrame({"pv_cf": p_pv/max(p.pv_capacity_kw,1e-9),
                              "wind_cf": p_wind/max(p.wind_capacity_kw,1e-9),
                              "demand_kw": load}, index=self.weather.index)
        res = dispatch_from_dataset(p, df_pt, cost=self.cost,
                                    method=self.method, **self.dispatch_kwargs)
        res["p_pv"], res["p_wind"], res["load"] = p_pv, p_wind, load
        kpi_in = {"load":load,"p_pv":p_pv,"p_wind":p_wind,"p_diesel":res["p_dg"],
                  "p_unmet":res["p_unmet"],"p_dump":res["p_dump"],"fuel_l":res["fuel_l"],
                  "starts":res["y_dg"],"soc":res["soc"],"p_batt":res["p_batt"]}
        kpis = compute_kpis(kpi_in, p)
        return res, kpis


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    print("Demo del evaluador físico (mock)...")
    data = build_lapalma_physical(mock=True)
    print(f"Weather: {data['weather'].shape} | demanda pico "
          f"{data['demand_kw'].max()/1000:.1f} MW")

    ev = HRESPhysicalEvaluator(data, method="rolling",
                               window=168, step=120, time_limit_block=5)
    names = ["pv_capacity_kw", "wind_capacity_kw", "batt_capacity_kwh",
             "diesel_capacity_kw", "batt_power_kw", "wind_hub_height",
             "pv_tilt", "pv_gamma_pdc"]
    x = np.array([30000, 25000, 80000, 100000, 30000, 80, 20, -0.004])
    k = ev.evaluate(x, names)
    print("\nKPIs:")
    for kk in ["LPSP", "REF", "Curtailment", "LCOE_eur_kWh", "CO2_kg",
               "CF_PV", "CF_Wind"]:
        print(f"  {kk:<16}{k[kk]:.4f}")
