"""
=============================================================================
HRES — Modelos de generación PV (pvlib) y eólica (windpowerlib)
=============================================================================
Reemplaza los modelos físicos simplificados por librerías validadas:

  • PV   : pvlib ModelChain con PVWatts (la temperatura de módulo, el coef.
           gamma_pdc, tilt, azimut y pdc0 influyen → entran en el GSA).
  • Wind : windpowerlib ModelChain con la curva de potencia REAL de la turbina
           (oedb). El tipo de turbina, hub_height y la rugosidad influyen.

Recurso meteorológico: Open-Meteo (archivo histórico ERA5), que entrega
GHI/DHI/DNI directamente en W/m², viento, temperatura y presión.

Estas funciones devuelven series horarias de potencia [kW] listas para el
despacho MILP, y están parametrizadas para el GSA (cada parámetro técnico es
un argumento explícito).

Turbinas oedb verificadas (has_power_curve=True): 'V90/2000', 'V100/1800',
'V112/3000', 'E-101/3050', etc. La Vestas V90-2MW es 'V90/2000'.
=============================================================================
"""

import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import hashlib
import json

CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  WEATHER — OPEN-METEO (archivo histórico ERA5)
# ─────────────────────────────────────────────────────────────────────────────

def get_weather_openmeteo(lat: float, lon: float, year: int,
                          use_cache: bool = True,
                          timeout: int = 120) -> pd.DataFrame:
    """
    Descarga weather horario de Open-Meteo (gratis, sin token).

    Devuelve DataFrame [UTC] con columnas:
        GHI, DHI, DNI [W/m²], wind_speed_10m, wind_speed_100m [m/s],
        temp_air [°C], pressure [Pa], temp_air_k [K]
    """
    key = hashlib.md5(f"om_{lat}_{lon}_{year}".encode()).hexdigest()[:12]
    cache = CACHE_DIR / f"openmeteo_{key}.parquet"
    if use_cache and cache.exists():
        try:
            return pd.read_parquet(cache)
        except Exception:
            pass

    hourly_vars = [
        "shortwave_radiation",       # GHI
        "diffuse_radiation",         # DHI
        "direct_normal_irradiance",  # DNI
        "wind_speed_10m",
        "wind_speed_100m",
        "temperature_2m",
        "pressure_msl",
    ]
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": f"{year}-01-01", "end_date": f"{year}-12-31",
        "hourly": ",".join(hourly_vars),
        "timezone": "UTC",
        "wind_speed_unit": "ms", "temperature_unit": "celsius",
        "pressure_unit": "hpa",
    }
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()["hourly"]
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").rename(columns={
        "shortwave_radiation": "GHI",
        "diffuse_radiation": "DHI",
        "direct_normal_irradiance": "DNI",
        "wind_speed_10m": "wind_speed_10m",
        "wind_speed_100m": "wind_speed_100m",
        "temperature_2m": "temp_air",
        "pressure_msl": "pressure",
    })
    df["pressure"] = df["pressure"] * 100.0      # hPa → Pa
    df["temp_air_k"] = df["temp_air"] + 273.15

    if use_cache:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache)
        except Exception:
            try:
                df.to_csv(cache.with_suffix(".csv"))
            except Exception:
                pass  # si no se puede cachear, seguir sin caché
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  PV — pvlib ModelChain (PVWatts)
# ─────────────────────────────────────────────────────────────────────────────

def pv_power_pvlib(weather: pd.DataFrame,
                   lat: float, lon: float,
                   pv_capacity_kw: float,
                   tilt: float = 20.0,
                   azimuth: float = 180.0,
                   gamma_pdc: float = -0.004,
                   temp_model: str = "open_rack_glass_glass",
                   system_loss: float = 0.14,
                   inverter_eff: float = 0.96) -> np.ndarray:
    """
    Potencia PV [kW] con pvlib PVWatts.

    Parámetros que influyen en el GSA:
        pv_capacity_kw : potencia DC nominal [kW]
        tilt, azimuth  : orientación del array [°]
        gamma_pdc      : coef. temperatura de potencia [1/°C]
        system_loss    : pérdidas globales del sistema [-]
    """
    from pvlib.location import Location
    from pvlib.pvsystem import PVSystem
    from pvlib.modelchain import ModelChain
    from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

    loc = Location(lat, lon, tz="UTC")
    temp_params = TEMPERATURE_MODEL_PARAMETERS["sapm"][temp_model]

    system = PVSystem(
        surface_tilt=tilt, surface_azimuth=azimuth,
        module_parameters={"pdc0": pv_capacity_kw * 1000.0,
                           "gamma_pdc": gamma_pdc},
        inverter_parameters={"pdc0": pv_capacity_kw * 1000.0,
                             "eta_inv_nom": inverter_eff},
        temperature_model_parameters=temp_params,
        losses_parameters={"pvwatts": {"soiling": 2, "shading": 0,
                                       "snow": 0, "mismatch": 2, "wiring": 2,
                                       "connections": 0.5, "lid": 1.5,
                                       "nameplate_rating": 1, "age": 0,
                                       "availability": 0}},
    )
    mc = ModelChain(system, loc,
                    dc_model="pvwatts", ac_model="pvwatts",
                    aoi_model="physical", spectral_model="no_loss",
                    temperature_model="sapm", losses_model="pvwatts")

    w = pd.DataFrame({
        "ghi": weather["GHI"].values,
        "dhi": weather["DHI"].values,
        "dni": weather["DNI"].values,
        "temp_air": weather["temp_air"].values,
        "wind_speed": weather["wind_speed_10m"].values,
    }, index=weather.index)

    mc.run_model(w)
    ac_w = mc.results.ac.fillna(0.0).clip(lower=0).values  # W
    # Aplicar pérdidas DC globales adicionales no incluidas
    p_kw = ac_w / 1000.0 * (1 - system_loss)
    return np.clip(p_kw, 0, pv_capacity_kw)


# ─────────────────────────────────────────────────────────────────────────────
#  WIND — windpowerlib ModelChain (curva de potencia real)
# ─────────────────────────────────────────────────────────────────────────────

# Curva de potencia de respaldo de la Vestas V90-2.0MW [m/s → W], por si la
# oedb no estuviera disponible localmente (datos de catálogo público).
V90_2000_POWER_CURVE = pd.DataFrame({
    "wind_speed": [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                   15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
    "value":     [0, 0, 75e3, 175e3, 350e3, 600e3, 950e3, 1300e3,
                  1650e3, 1875e3, 1975e3, 2000e3, 2000e3, 2000e3,
                  2000e3, 2000e3, 2000e3, 2000e3, 2000e3, 2000e3,
                  2000e3, 2000e3, 2000e3, 2000e3],
})


def wind_power_windpowerlib(weather: pd.DataFrame,
                            wind_capacity_kw: float,
                            turbine_type: str = "V90/2000",
                            hub_height: float = 80.0,
                            roughness_length: float = 0.03,
                            nominal_power_kw: float = 2000.0,
                            wind_col: str = "wind_speed_ninja") -> np.ndarray:
    """
    Potencia eólica [kW] con windpowerlib y la curva de potencia REAL.

    El viento se toma de la columna `wind_col` del DataFrame `weather`, que
    debe estar YA a la altura de buje (caso de Renewables.ninja con raw=true,
    que devuelve la velocidad a la 'height' solicitada). Por eso NO se
    extrapola: se pasa a windpowerlib a la altura 'hub_height' directamente.

    Si `wind_col` no existe, recurre a 'wind_speed_100m' (Open-Meteo) y deja
    que windpowerlib extrapole — pero OJO: Open-Meteo ERA5 subestima el viento
    en orografía insular; se recomienda usar el viento de Ninja.

    Parámetros que influyen en el GSA:
        wind_capacity_kw, turbine_type, hub_height, roughness_length
    """
    from windpowerlib import WindTurbine, ModelChain

    try:
        turbine = WindTurbine(turbine_type=turbine_type, hub_height=hub_height)
        if turbine.power_curve is None:
            raise ValueError("sin curva")
        nominal_w = turbine.nominal_power
    except Exception:
        turbine = WindTurbine(hub_height=hub_height,
                              nominal_power=nominal_power_kw * 1000.0,
                              power_curve=V90_2000_POWER_CURVE)
        nominal_w = nominal_power_kw * 1000.0

    if wind_col in weather.columns:
        # Viento de Ninja a una altura de REFERENCIA fija (ninja_ref_height).
        # Se la damos a windpowerlib como la altura del dato; si hub_height
        # difiere, windpowerlib extrapola (log) desde ref a hub → H_hub influye.
        ref_h = int(weather.attrs.get("ninja_ref_height", 80))
        cols = pd.MultiIndex.from_tuples([
            ("wind_speed", ref_h),
            ("temperature", 2), ("pressure", 0), ("roughness_length", 0),
        ])
        wm = pd.DataFrame(index=weather.index, columns=cols, dtype=float)
        wm[("wind_speed", ref_h)] = weather[wind_col].values
    else:
        # Respaldo: viento de Open-Meteo a 100 m (windpowerlib extrapola)
        cols = pd.MultiIndex.from_tuples([
            ("wind_speed", 100), ("temperature", 2),
            ("pressure", 0), ("roughness_length", 0),
        ])
        wm = pd.DataFrame(index=weather.index, columns=cols, dtype=float)
        wm[("wind_speed", 100)] = weather["wind_speed_100m"].values

    wm[("temperature", 2)]  = weather["temp_air_k"].values
    wm[("pressure", 0)]     = weather["pressure"].values
    wm[("roughness_length", 0)] = roughness_length

    mc = ModelChain(turbine,
                    wind_speed_model="logarithmic",
                    density_model="ideal_gas",
                    power_output_model="power_curve")
    mc.run_model(wm)

    p_single_w = mc.power_output.fillna(0.0).clip(lower=0).values
    n_turbines = (wind_capacity_kw * 1000.0) / nominal_w
    p_total_kw = p_single_w * n_turbines / 1000.0
    return np.clip(p_total_kw, 0, wind_capacity_kw)


def get_ninja_wind_speed(lat: float, lon: float, year: int,
                         ref_height: float = 80.0,
                         turbine: str = "Vestas V90 2000",
                         token: Optional[str] = None,
                         use_cache: bool = True) -> pd.Series:
    """
    Descarga la velocidad de viento de Renewables.ninja a una altura de
    referencia fija (ref_height). Devuelve una Serie [m/s] indexada en UTC.

    Se usa porque Open-Meteo ERA5 subestima fuertemente el viento en orografía
    insular (La Palma): Ninja (MERRA-2) da ~7 m/s a 80 m, realista para los
    alisios, frente a ~2.8 m/s de Open-Meteo. El viento se pide a ref_height
    fija; windpowerlib luego extrapola a la hub_height variable del GSA.
    """
    from data_sources import fetch_ninja_wind_raw
    df = fetch_ninja_wind_raw(lat, lon, year, height=ref_height,
                              turbine=turbine, token=token, use_cache=use_cache)
    return df["wind_ms"]


# ─────────────────────────────────────────────────────────────────────────────
#  DEMO / VALIDACIÓN OFFLINE
# ─────────────────────────────────────────────────────────────────────────────

def _mock_weather(year=2019, seed=42):
    """Weather sintético con la estructura de Open-Meteo (para tests offline)."""
    idx = pd.date_range(f"{year}-01-01", periods=8760, freq="h", tz="UTC")
    rng = np.random.default_rng(seed)
    hod = idx.hour.values; doy = idx.dayofyear.values
    decl = 23.45*np.sin(2*np.pi*(doy-81)/365); ha = 15*(hod-12)
    cosz = np.clip(np.sin(np.radians(28.5))*np.sin(np.radians(decl)) +
                   np.cos(np.radians(28.5))*np.cos(np.radians(decl))*np.cos(np.radians(ha)),0,1)
    ghi = np.clip(1000*cosz*(0.78+0.08*np.sin(2*np.pi*doy/365-np.pi/2)),0,1100); ghi[ghi<10]=0
    dhi = ghi*0.25; dni = np.where(cosz>0.05, (ghi-dhi)/cosz, 0)
    w100 = np.clip(8.5*rng.weibull(2.4,8760)*(1+0.15*np.sin(2*np.pi*doy/365)),0,32)
    w10 = w100*0.75
    tair = 20+5*np.sin(2*np.pi*(doy-80)/365)+4*np.sin(np.pi*(hod-6)/12)
    return pd.DataFrame({"GHI":ghi,"DHI":dhi,"DNI":np.clip(dni,0,1000),
                         "wind_speed_10m":w10,"wind_speed_100m":w100,
                         "temp_air":tair,"pressure":101325.0,
                         "temp_air_k":tair+273.15}, index=idx)


if __name__ == "__main__":
    print("Validación offline (mock weather)...")
    w = _mock_weather()
    print(f"Weather mock: {w.shape}, GHI máx {w['GHI'].max():.0f} W/m², "
          f"viento100 medio {w['wind_speed_100m'].mean():.1f} m/s")

    ppv = pv_power_pvlib(w, 28.49, -17.85, pv_capacity_kw=30000,
                         tilt=20, azimuth=180)
    print(f"\nPV: pico {ppv.max()/1000:.1f} MW | "
          f"CF {ppv.mean()/30000:.3f}  (esperado ~0.18-0.22)")

    pw = wind_power_windpowerlib(w, wind_capacity_kw=25000,
                                 turbine_type="V90/2000", hub_height=80)
    print(f"Eólica: pico {pw.max()/1000:.1f} MW | "
          f"CF {pw.mean()/25000:.3f}  (esperado ~0.30-0.40)")
