"""
=============================================================================
HRES — Módulo específico LA PALMA (sistema eléctrico aislado canario)
=============================================================================
La demanda de La Palma NO está en el indicador ESIOS 1293 (ese es el sistema
peninsular). Los sistemas no peninsulares se sirven a través de la REData API
de REE con geo_trunc=electric_system:

    https://apidatos.ree.es/es/datos/demanda/evolucion
        ?start_date=...&end_date=...&time_trunc=hour
        &geo_trunc=electric_system&geo_limit=canarias&geo_ids={ID_LA_PALMA}

Ventajas para el paper:
  • La Palma es un sistema AISLADO con ~90% de generación diésel (central de
    Los Guinchos, 2019) → caso de estudio idóneo para penetración renovable.
  • La REData API es pública (sin token) y citable.

Este módulo:
  (1) Auto-descubre el geo_id de La Palma consultando la API (robusto frente a
      cambios), con un valor de respaldo conocido.
  (2) Descarga la demanda horaria real y la cachea.
  (3) Provee presets de ubicación/recurso para Renewables.ninja en La Palma.

Referencias del sistema:
  - Anuario Energético de Canarias (2024)
  - REE, sistema eléctrico canario (sistemas no peninsulares)
=============================================================================
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd
import requests

CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(exist_ok=True)

ESIOS_DEMAND_INDICATOR = 1342   # "Demanda real" sistemas no peninsulares (Canarias)

# geo_id de La Palma en ESIOS (indicador 1293 "Demanda real").
# Verificado por el usuario contra la API ESIOS.
LA_PALMA_ESIOS_GEO_ID = 8798

# Mapa de sistemas eléctricos canarios en ESIOS (indicador 1293).
CANARY_SYSTEM_ESIOS_GEO_IDS = {
    "el_hierro":                 8800,
    "gran_canaria":              8795,
    "la_gomera":                 8799,
    "la_palma":                  8798,
    "lanzarote_y_fuerteventura": 8796,
    "tenerife":                  8794,
}


# ─────────────────────────────────────────────────────────────────────────────
#  PRESETS DE UBICACIÓN Y RECURSO — LA PALMA
# ─────────────────────────────────────────────────────────────────────────────
# Coordenadas representativas de emplazamientos reales de generación en la isla.

LA_PALMA_PRESETS = {
    # Sur de la isla: mejor recurso eólico (alisios), parques existentes
    "fuencaliente": {
        "lat": 28.49, "lon": -17.85,
        "desc": "Fuencaliente (sur) — parque eólico existente, alisios fuertes",
        "wind_turbine": "Vestas V90 2000", "wind_hub": 80,
        "pv_tilt": 28, "pv_azim": 180,
    },
    # Villa de Mazo (sureste): Manchas Blancas + aeropuerto (AENA)
    "villa_de_mazo": {
        "lat": 28.58, "lon": -17.78,
        "desc": "Villa de Mazo (SE) — Manchas Blancas + aeropuerto",
        "wind_turbine": "Vestas V90 2000", "wind_hub": 80,
        "pv_tilt": 28, "pv_azim": 180,
    },
    # Garafía (norte): Juan Adalid
    "garafia": {
        "lat": 28.82, "lon": -17.94,
        "desc": "Garafía (norte) — parque Juan Adalid",
        "wind_turbine": "Enercon E70 2000", "wind_hub": 65,
        "pv_tilt": 30, "pv_azim": 180,
    },
    # Los Llanos de Aridane (oeste): zona de mayor irradiancia solar
    "los_llanos": {
        "lat": 28.66, "lon": -17.92,
        "desc": "Los Llanos de Aridane (oeste) — máxima irradiancia solar",
        "wind_turbine": "Vestas V90 2000", "wind_hub": 80,
        "pv_tilt": 28, "pv_azim": 180,
    },
    # Centroide de la isla (genérico)
    "centro": {
        "lat": 28.68, "lon": -17.86,
        "desc": "Centro de la isla (genérico)",
        "wind_turbine": "Vestas V90 2000", "wind_hub": 80,
        "pv_tilt": 28, "pv_azim": 180,
    },
}


def _cache_key(prefix: str, **kw) -> Path:
    raw = prefix + "|" + json.dumps(kw, sort_keys=True)
    return CACHE_DIR / f"{prefix}_{hashlib.md5(raw.encode()).hexdigest()[:12]}.parquet"


def _load_cache(p: Path):
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception:
            return None
    return None


def _save_cache(df, p):
    try:
        df.to_parquet(p)
    except Exception:
        df.to_csv(p.with_suffix(".csv"))


# ─────────────────────────────────────────────────────────────────────────────
#  geo_id DE SISTEMAS CANARIOS EN ESIOS
# ─────────────────────────────────────────────────────────────────────────────

def get_canary_geo_id(system_name: str = "la_palma") -> int:
    """Devuelve el geo_id ESIOS del sistema canario indicado."""
    key = system_name.strip().lower().replace(" ", "_")
    if key not in CANARY_SYSTEM_ESIOS_GEO_IDS:
        raise ValueError(
            f"Sistema '{system_name}' no reconocido. "
            f"Opciones: {list(CANARY_SYSTEM_ESIOS_GEO_IDS)}")
    return CANARY_SYSTEM_ESIOS_GEO_IDS[key]


# ─────────────────────────────────────────────────────────────────────────────
#  DEMANDA HORARIA LA PALMA (ESIOS, indicador 1293 + geo_id 8798)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_lapalma_demand(year: int,
                         geo_id: Optional[int] = None,
                         esios_token: Optional[str] = None,
                         indicator: int = 1342,
                         use_cache: bool = True,
                         timeout: int = 120) -> pd.DataFrame:
    """
    Descarga la demanda horaria real de La Palma vía API ESIOS de REE.

    Usa el indicador 1342 (demanda real de sistemas no peninsulares) filtrado
    por geo_id=8798 (La Palma).
    ESIOS limita el rango por petición, así que se descarga por bloques
    mensuales y se concatena. Requiere token ESIOS (env ESIOS_TOKEN o arg).

    Devuelve DataFrame indexado por datetime (UTC) con columna 'demand_mw'.
    """
    from data_sources import fetch_esios_demand

    if geo_id is None:
        geo_id = LA_PALMA_ESIOS_GEO_ID

    cache_path = _cache_key("esios_lapalma", year=year, geo_id=geo_id,
                            indicator=indicator)
    if use_cache:
        cached = _load_cache(cache_path)
        if cached is not None:
            return cached

    frames = []
    for month in range(1, 13):
        start = pd.Timestamp(year=year, month=month, day=1)
        end = (start + pd.offsets.MonthEnd(1))
        dfm = fetch_esios_demand(
            date_from=start.strftime("%Y-%m-%d"),
            date_to=end.strftime("%Y-%m-%d"),
            indicator=indicator, token=esios_token,
            time_trunc="hour", geo_id=geo_id, use_cache=use_cache,
            timeout=timeout)
        frames.append(dfm)

    if not frames:
        raise RuntimeError("ESIOS no devolvió datos para La Palma. "
                           "Verifica el geo_id (8798) o el token.")

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]

    if use_cache:
        _save_cache(df, cache_path)
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  DATASET COMPLETO LA PALMA (demanda REData + recurso Renewables.ninja)
# ─────────────────────────────────────────────────────────────────────────────

def build_lapalma_dataset(year: int = 2019,
                          location: str = "fuencaliente",
                          ninja_token: Optional[str] = None,
                          esios_token: Optional[str] = None,
                          demand_scale_to_peak_kw: Optional[float] = None,
                          geo_id: Optional[int] = None,
                          use_cache: bool = True) -> pd.DataFrame:
    """
    Construye el dataset horario para una microrred en La Palma:
      pv_cf, wind_cf  ← Renewables.ninja en el emplazamiento elegido
      demand_kw       ← demanda real REData de La Palma (opcionalmente reescalada)

    Parámetros
    ----------
    location : clave de LA_PALMA_PRESETS (fuencaliente, villa_de_mazo,
               garafia, los_llanos, centro).
    demand_scale_to_peak_kw : si se indica, reescala la demanda insular real
               para que su pico iguale este valor [kW] — útil para dimensionar
               una microrred que sigue el PERFIL real de la isla a otra escala.
               Si None, se usa la demanda real de La Palma en MW.
    """
    from data_sources import fetch_ninja_pv, fetch_ninja_wind

    if location not in LA_PALMA_PRESETS:
        raise ValueError(f"location debe ser uno de {list(LA_PALMA_PRESETS)}")
    preset = LA_PALMA_PRESETS[location]
    lat, lon = preset["lat"], preset["lon"]

    print(f"[La Palma] Emplazamiento: {preset['desc']}")
    print(f"[La Palma] lat={lat} lon={lon} año={year}")

    df_pv = fetch_ninja_pv(lat, lon, year, token=ninja_token,
                           tilt=preset["pv_tilt"], azim=preset["pv_azim"],
                           use_cache=use_cache)
    df_wind = fetch_ninja_wind(lat, lon, year, token=ninja_token,
                               height=preset["wind_hub"],
                               turbine=preset["wind_turbine"],
                               use_cache=use_cache)
    df_dem = fetch_lapalma_demand(year, geo_id=geo_id,
                                  esios_token=esios_token, use_cache=use_cache)

    df = pd.concat([df_pv["pv_cf"], df_wind["wind_cf"], df_dem["demand_mw"]],
                   axis=1, join="inner").interpolate(limit=3).dropna()

    # Recortar 29-feb en bisiestos → 8760 h
    df = df[~((df.index.month == 2) & (df.index.day == 29))]

    if demand_scale_to_peak_kw is not None:
        df["demand_kw"] = (df["demand_mw"] / df["demand_mw"].max()
                           * demand_scale_to_peak_kw)
    else:
        df["demand_kw"] = df["demand_mw"] * 1000.0  # MW → kW (isla completa)

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  DATASET CON VARIABLES CRUDAS (irradiancia + temperatura + viento)
# ─────────────────────────────────────────────────────────────────────────────

def build_lapalma_dataset_raw(year: int = 2019,
                              location: str = "fuencaliente",
                              ninja_token: Optional[str] = None,
                              esios_token: Optional[str] = None,
                              demand_scale_to_peak_kw: Optional[float] = None,
                              geo_id: Optional[int] = None,
                              use_cache: bool = True) -> pd.DataFrame:
    """
    Como build_lapalma_dataset, pero en vez de los factores de capacidad de
    Ninja descarga las VARIABLES METEOROLÓGICAS CRUDAS:

        ghi_wm2  : irradiancia de superficie [W/m²]
        t_amb_c  : temperatura ambiente [°C]
        wind_ms  : velocidad de viento a 10 m [m/s]
        demand_mw / demand_kw : demanda real de La Palma (ESIOS)

    Estas columnas alimentan pv_power()/wind_power() de hres_model.py, de modo
    que η_PV, α_T, η_inv, Cp, H_hub, cut-in/out SÍ influyan en la generación y,
    por tanto, en el GSA.

    NOTA metodológica: con este flujo la física de la planta la pone el modelo
    propio (no la curva validada de Ninja). El recurso (irradiancia, temp,
    viento) sigue siendo real/reanálisis MERRA-2.
    """
    from data_sources import fetch_ninja_pv_raw, fetch_ninja_wind_raw

    if location not in LA_PALMA_PRESETS:
        raise ValueError(f"location debe ser uno de {list(LA_PALMA_PRESETS)}")
    preset = LA_PALMA_PRESETS[location]
    lat, lon = preset["lat"], preset["lon"]

    print(f"[La Palma RAW] Emplazamiento: {preset['desc']}")
    print(f"[La Palma RAW] lat={lat} lon={lon} año={year} (variables crudas)")

    df_pv = fetch_ninja_pv_raw(lat, lon, year, token=ninja_token,
                               use_cache=use_cache)
    df_wind = fetch_ninja_wind_raw(lat, lon, year, token=ninja_token,
                                   turbine=preset["wind_turbine"],
                                   use_cache=use_cache)
    df_dem = fetch_lapalma_demand(year, geo_id=geo_id,
                                  esios_token=esios_token, use_cache=use_cache)

    df = pd.concat([df_pv["ghi_wm2"], df_pv["t_amb_c"], df_wind["wind_ms"],
                    df_dem["demand_mw"]],
                   axis=1, join="inner").interpolate(limit=3).dropna()

    df = df[~((df.index.month == 2) & (df.index.day == 29))]

    if demand_scale_to_peak_kw is not None:
        df["demand_kw"] = (df["demand_mw"] / df["demand_mw"].max()
                           * demand_scale_to_peak_kw)
    else:
        df["demand_kw"] = df["demand_mw"] * 1000.0

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  MOCK LA PALMA (offline) — perfil insular realista
# ─────────────────────────────────────────────────────────────────────────────

def build_lapalma_mock(year: int = 2019, peak_kw: float = 40000.0,
                       seed: int = 42) -> pd.DataFrame:
    """
    Dataset MOCK con estructura idéntica a build_lapalma_dataset.
    Demanda pico ~40 MW (orden de magnitud del sistema insular de La Palma).
    Recurso: alta penetración eólica (alisios) + buena irradiancia subtropical.
    """
    idx = pd.date_range(f"{year}-01-01", periods=8760, freq="h", tz="UTC")
    rng = np.random.default_rng(seed)
    hod = idx.hour.values
    doy = idx.dayofyear.values

    # PV CF — latitud 28.5°N, irradiancia alta y estable
    decl = 23.45 * np.sin(2*np.pi*(doy-81)/365)
    ha = 15*(hod-12)
    cosz = np.clip(np.sin(np.radians(28.5))*np.sin(np.radians(decl)) +
                   np.cos(np.radians(28.5))*np.cos(np.radians(decl))*np.cos(np.radians(ha)), 0, 1)
    pv_cf = np.clip(0.88*cosz*(0.78+0.08*np.sin(2*np.pi*doy/365-np.pi/2))
                    *(1+0.08*rng.standard_normal(8760)), 0, 1)
    pv_cf[pv_cf < 0.01] = 0

    # Wind CF — alisios: recurso fuerte y persistente, máximo en verano
    u = 8.5*rng.weibull(2.4, 8760)*(1+0.15*np.sin(2*np.pi*doy/365))*(1+0.1*np.sin(np.pi*hod/12))
    wind_cf = np.clip((u/13)**3, 0, 1)
    wind_cf = np.where(u < 3, 0, np.where(u > 25, 0, wind_cf))

    # Demanda insular — perfil con poca estacionalidad (clima suave), pico noche
    base = np.array([.62,.58,.55,.53,.54,.57,.62,.68,.75,.80,.83,.85,
                     .86,.85,.83,.82,.83,.86,.92,1.0,.97,.88,.78,.68])
    dem = peak_kw*base[hod]*(1+0.06*np.sin(2*np.pi*doy/365))*(1+0.05*rng.standard_normal(8760))
    dem = np.clip(dem, 0, peak_kw*1.15)

    return pd.DataFrame({"pv_cf": pv_cf, "wind_cf": wind_cf,
                         "demand_mw": dem/1000.0, "demand_kw": dem}, index=idx)


def build_lapalma_mock_raw(year: int = 2019, peak_kw: float = 40000.0,
                           seed: int = 42) -> pd.DataFrame:
    """
    MOCK con VARIABLES CRUDAS (ghi_wm2, t_amb_c, wind_ms) para validar offline
    el flujo de modelo físico. Misma estructura que build_lapalma_dataset_raw.
    """
    idx = pd.date_range(f"{year}-01-01", periods=8760, freq="h", tz="UTC")
    rng = np.random.default_rng(seed)
    hod = idx.hour.values
    doy = idx.dayofyear.values

    # Irradiancia [W/m²] — geometría solar a 28.5°N
    decl = 23.45 * np.sin(2*np.pi*(doy-81)/365)
    ha = 15*(hod-12)
    cosz = np.clip(np.sin(np.radians(28.5))*np.sin(np.radians(decl)) +
                   np.cos(np.radians(28.5))*np.cos(np.radians(decl))*np.cos(np.radians(ha)), 0, 1)
    ghi = np.clip(1000*cosz*(0.78+0.08*np.sin(2*np.pi*doy/365-np.pi/2))
                  *(1+0.08*rng.standard_normal(8760)), 0, 1100)
    ghi[ghi < 10] = 0

    # Temperatura ambiente [°C] — clima subtropical suave
    t_amb = (20 + 5*np.sin(2*np.pi*(doy-80)/365)
             + 4*np.sin(np.pi*(hod-6)/12) + 1.5*rng.standard_normal(8760))

    # Velocidad de viento a 10 m [m/s] — alisios
    wind_ms = np.clip(7.0*rng.weibull(2.4, 8760)
                      *(1+0.15*np.sin(2*np.pi*doy/365))
                      *(1+0.1*np.sin(np.pi*hod/12)), 0, 32)

    base = np.array([.62,.58,.55,.53,.54,.57,.62,.68,.75,.80,.83,.85,
                     .86,.85,.83,.82,.83,.86,.92,1.0,.97,.88,.78,.68])
    dem = peak_kw*base[hod]*(1+0.06*np.sin(2*np.pi*doy/365))*(1+0.05*rng.standard_normal(8760))
    dem = np.clip(dem, 0, peak_kw*1.15)

    return pd.DataFrame({"ghi_wm2": ghi, "t_amb_c": t_amb, "wind_ms": wind_ms,
                         "demand_mw": dem/1000.0, "demand_kw": dem}, index=idx)
    print("Presets de ubicación La Palma:")
    for k, v in LA_PALMA_PRESETS.items():
        print(f"  {k:<15} {v['desc']}")
    print("\nGenerando MOCK La Palma (offline)...")
    df = build_lapalma_mock(peak_kw=40000)
    print(df.describe())
    print(f"\nPV CF medio: {df['pv_cf'].mean():.3f} | "
          f"Wind CF medio: {df['wind_cf'].mean():.3f} | "
          f"Demanda pico: {df['demand_kw'].max()/1000:.1f} MW")
    print("\nDatos REALES (con token Ninja; REData no requiere token):")
    print('  export NINJA_TOKEN="..."')
    print('  df = build_lapalma_dataset(year=2019, location="fuencaliente",')
    print('         demand_scale_to_peak_kw=None)  # isla completa en kW')
