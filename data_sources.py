"""
=============================================================================
HRES — Adquisición de datos reales (ESIOS + Renewables.ninja)
=============================================================================
Sustituye los perfiles sintéticos por series reales:

  • DEMANDA   : API ESIOS de Red Eléctrica de España (REE)
                Indicador 1293 = "Demanda real" peninsular [MW]
                https://api.esios.ree.es/indicators/{id}
                Header: x-api-key

  • RECURSO   : API Renewables.ninja (perfiles horarios PV y eólica)
  SOLAR/EÓLICO  Endpoints api/data/pv y api/data/wind, capacity=1 → factor
                de capacidad normalizado [0-1] (se escala por la potencia
                instalada en el modelo de despacho).
                Header: Authorization: Token {token}
                Límite: 1 año por petición; rate-limited (registrados).

Todas las descargas se CACHEAN en disco (parquet/csv) para no repetir
llamadas en cada evaluación de Sobol / NSGA-II. Pon los tokens en variables
de entorno:
    export ESIOS_TOKEN="...."
    export NINJA_TOKEN="...."

Referencias de datos:
  - Pfenninger & Staffell (2016) Energy 114:1251-1265  (PV, MERRA-2)
  - Staffell & Pfenninger (2016) Energy 114:1224-1239  (eólica)
=============================================================================
"""

import os
import time
import json
import hashlib
from io import StringIO
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests

CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  UTILIDADES DE CACHÉ
# ─────────────────────────────────────────────────────────────────────────────

def _cache_key(prefix: str, **kwargs) -> Path:
    """Genera una ruta de caché determinista a partir de los parámetros."""
    raw = prefix + "|" + json.dumps(kwargs, sort_keys=True)
    h = hashlib.md5(raw.encode()).hexdigest()[:12]
    return CACHE_DIR / f"{prefix}_{h}.parquet"


def _load_cache(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            return None
    return None


def _save_cache(df: pd.DataFrame, path: Path):
    try:
        df.to_parquet(path)
    except Exception:
        # Fallback a CSV si no hay engine parquet
        df.to_csv(path.with_suffix(".csv"))


# ─────────────────────────────────────────────────────────────────────────────
#  ESIOS — DEMANDA REAL (REE)
# ─────────────────────────────────────────────────────────────────────────────

ESIOS_BASE = "https://api.esios.ree.es"
ESIOS_INDICATORS = {
    "demanda_real":     1293,   # Demanda real peninsular [MW]
    "demanda_prevista": 460,    # Demanda prevista [MW]
    "precio_spot":      600,    # Precio mercado diario (SPOT) [€/MWh]
    "pvpc":             1001,   # PVPC 2.0TD [€/MWh]
}


def fetch_esios_demand(date_from: str,
                       date_to: str,
                       indicator: int = 1293,
                       token: Optional[str] = None,
                       time_trunc: str = "hour",
                       geo_id: Optional[int] = None,
                       use_cache: bool = True,
                       timeout: int = 60) -> pd.DataFrame:
    """
    Descarga una serie de la API ESIOS de REE.

    Parámetros
    ----------
    date_from, date_to : 'YYYY-MM-DD' (inclusive)
    indicator          : ID de indicador ESIOS (1293 = demanda real)
    token              : x-api-key (si None → variable de entorno ESIOS_TOKEN)
    time_trunc         : 'hour' | 'day' | ...
    geo_id             : opcional, filtra por zona geográfica (None = peninsular)

    Devuelve
    --------
    DataFrame indexado por datetime (UTC→local) con columna 'demand_mw'.
    """
    token = token or os.environ.get("ESIOS_TOKEN")
    if not token:
        raise ValueError("Falta token ESIOS. Define ESIOS_TOKEN o pásalo como arg.")

    cache_path = _cache_key("esios", date_from=date_from, date_to=date_to,
                            indicator=indicator, time_trunc=time_trunc,
                            geo_id=geo_id)
    if use_cache:
        cached = _load_cache(cache_path)
        if cached is not None:
            return cached

    headers = {
        "Accept": "application/json; application/vnd.esios-api-v1+json",
        "Content-Type": "application/json",
        "x-api-key": token,
    }
    params = {
        "start_date": f"{date_from}T00:00:00",
        "end_date":   f"{date_to}T23:59:59",
        "time_trunc": time_trunc,
    }
    if geo_id is not None:
        params["geo_ids[]"] = geo_id

    url = f"{ESIOS_BASE}/indicators/{indicator}"
    resp = requests.get(url, headers=headers, params=params, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()

    values = payload.get("indicator", {}).get("values", [])
    if not values:
        raise RuntimeError(
            f"ESIOS indicador {indicator} devolvió 0 valores para "
            f"{date_from}..{date_to} (geo_id={geo_id}). Verifica indicador, "
            f"rango de fechas o que el token tenga acceso a este indicador.")
    df = pd.DataFrame(values)

    # La marca de tiempo puede venir como 'datetime_utc' o solo 'datetime'
    # (con offset horario incluido). Usamos la que exista.
    if "datetime_utc" in df.columns:
        ts = pd.to_datetime(df["datetime_utc"], utc=True)
    elif "datetime" in df.columns:
        ts = pd.to_datetime(df["datetime"], utc=True)
    else:
        raise RuntimeError(
            f"Respuesta ESIOS sin columna de fecha. Columnas: {list(df.columns)}")
    df["datetime"] = ts
    df = df.set_index("datetime").sort_index()

    # Filtrar/agregar por geo_id
    if "geo_id" in df.columns:
        if geo_id is not None:
            # Quedarnos SOLO con el sistema pedido (ESIOS puede devolver varios).
            # Comparación robusta a int/str (ESIOS a veces da geo_id como texto).
            gid_match = df["geo_id"].astype(str) == str(geo_id)
            if gid_match.any():
                df = df[gid_match][["value"]].rename(columns={"value": "demand_mw"})
            else:
                # ESIOS ya filtró y no repite geo_id, o solo hay un sistema:
                # usar todos los valores devueltos.
                df = df[["value"]].rename(columns={"value": "demand_mw"})
        elif df["geo_id"].nunique() > 1:
            df = df.groupby(level=0)["value"].sum().to_frame("demand_mw")
        else:
            df = df[["value"]].rename(columns={"value": "demand_mw"})
    else:
        df = df[["value"]].rename(columns={"value": "demand_mw"})

    df = df[~df.index.duplicated(keep="first")]

    if use_cache:
        _save_cache(df, cache_path)
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  RENEWABLES.NINJA — PERFILES PV Y EÓLICA
# ─────────────────────────────────────────────────────────────────────────────

NINJA_BASE = "https://www.renewables.ninja/api"


def _ninja_session(token: Optional[str] = None) -> requests.Session:
    token = token or os.environ.get("NINJA_TOKEN")
    if not token:
        raise ValueError("Falta token Ninja. Define NINJA_TOKEN o pásalo como arg.")
    s = requests.session()
    s.headers = {"Authorization": f"Token {token}"}
    return s


def _ninja_check(resp: requests.Response):
    """
    Verifica la respuesta de Renewables.ninja y, ante error, expone el MENSAJE
    que devuelve la API en el cuerpo (raise_for_status lo oculta). La mayoría de
    los 400 traen aquí el motivo exacto (parámetro fuera de rango, turbina
    desconocida, límite de fecha, etc.).
    """
    if resp.status_code == 429:
        raise RuntimeError(
            "Renewables.ninja: rate limit (429). Sube rate_pause "
            "(registrados ~6 req/min) o espera unos minutos.")
    if resp.status_code >= 400:
        body = resp.text[:500] if resp.text else "(sin cuerpo)"
        raise RuntimeError(
            f"Renewables.ninja HTTP {resp.status_code} en {resp.url}\n"
            f"  → Mensaje de la API: {body}")
    return resp


def _ninja_to_dataframe(data: dict, value_col: str) -> pd.DataFrame:
    """
    Normaliza la respuesta JSON de Ninja a un DataFrame con columna `value_col`,
    soportando los formatos posibles de la API:
      • Envuelto:  {"data": {<clave>: {"electricity": x}}, "metadata": {...}}
      • Plano:     {<clave>: {"electricity": x}, ...}
    Las CLAVES de tiempo pueden venir como:
      • epoch en milisegundos (int/str numérico, p.ej. "1546300800000")
      • cadena de fecha ("2019-01-01 00:00")
    Se detecta automáticamente y se convierte a datetime UTC.
    """
    # Si viene envuelto en 'data' (formato por defecto), desempaquetar
    if "data" in data and isinstance(data["data"], dict):
        data = data["data"]
    df = pd.DataFrame.from_dict(data, orient="index")

    # Detección del tipo de clave temporal
    idx_raw = df.index.astype(str)
    if idx_raw.str.fullmatch(r"\d+").all():
        # Claves numéricas → epoch. ms si tiene 13 dígitos, s si tiene 10.
        as_int = idx_raw.astype("int64")
        unit = "ms" if idx_raw.str.len().max() >= 12 else "s"
        df.index = pd.to_datetime(as_int, unit=unit, utc=True)
    else:
        df.index = pd.to_datetime(idx_raw, utc=True)

    df = df.sort_index()
    # La columna de salida puede llamarse 'electricity' (o única columna)
    if "electricity" in df.columns:
        df = df[["electricity"]].rename(columns={"electricity": value_col})
    else:
        df = df.iloc[:, [0]]
        df.columns = [value_col]
    return df


def fetch_ninja_pv(lat: float, lon: float,
                   year: int,
                   tilt: float = 35,
                   azim: float = 180,
                   system_loss: float = 0.1,
                   tracking: int = 0,
                   dataset: str = "merra2",
                   token: Optional[str] = None,
                   use_cache: bool = True,
                   rate_pause: float = 12.0,
                   timeout: int = 120) -> pd.DataFrame:
    """
    Perfil horario de PV (factor de capacidad [0-1]) desde Renewables.ninja.

    capacity=1 → la columna 'electricity' es directamente el factor de
    capacidad; se escala luego por params.pv_capacity_kw en el despacho.

    rate_pause : pausa tras la llamada para respetar el rate-limit de la API
                 (registrados ~6 req/min). Súbelo si recibes HTTP 429.
    """
    cache_path = _cache_key("ninja_pv", lat=lat, lon=lon, year=year,
                            tilt=tilt, azim=azim, system_loss=system_loss,
                            tracking=tracking, dataset=dataset)
    if use_cache:
        cached = _load_cache(cache_path)
        if cached is not None:
            return cached

    s = _ninja_session(token)
    url = f"{NINJA_BASE}/data/pv"
    # IMPORTANTE: enteros como int (no float). 'capacity=1.0' provoca HTTP 400
    # en la API; debe ir como '1'. Booleanos como cadena minúscula 'false'.
    args = {
        "lat": round(float(lat), 4), "lon": round(float(lon), 4),
        "date_from": f"{year}-01-01", "date_to": f"{year}-12-31",
        "dataset": dataset, "capacity": 1,
        "system_loss": float(system_loss), "tracking": int(tracking),
        "tilt": float(tilt), "azim": float(azim),
        "format": "json",
    }
    resp = _ninja_check(s.get(url, params=args, timeout=timeout))
    data = resp.json()
    df = _ninja_to_dataframe(data, "pv_cf")

    if use_cache:
        _save_cache(df, cache_path)
    time.sleep(rate_pause)
    return df


def fetch_ninja_wind(lat: float, lon: float,
                     year: int,
                     height: float = 80,
                     turbine: str = "Vestas V90 2000",
                     dataset: str = "merra2",
                     token: Optional[str] = None,
                     use_cache: bool = True,
                     rate_pause: float = 12.0,
                     timeout: int = 120) -> pd.DataFrame:
    """
    Perfil horario eólico (factor de capacidad [0-1]) desde Renewables.ninja.

    Parámetros
    ----------
    height  : altura del buje [m]
    turbine : modelo de turbina (catálogo Ninja, p.ej. 'Vestas V90 2000',
              'Vestas V80 2000', 'Enercon E82 2000', 'Gamesa G87 2000', ...)
    """
    cache_path = _cache_key("ninja_wind", lat=lat, lon=lon, year=year,
                            height=height, turbine=turbine, dataset=dataset)
    if use_cache:
        cached = _load_cache(cache_path)
        if cached is not None:
            return cached

    s = _ninja_session(token)
    url = f"{NINJA_BASE}/data/wind"
    args = {
        "lat": round(float(lat), 4), "lon": round(float(lon), 4),
        "date_from": f"{year}-01-01", "date_to": f"{year}-12-31",
        "capacity": 1, "height": float(height), "turbine": turbine,
        "dataset": dataset, "format": "json",
    }
    resp = _ninja_check(s.get(url, params=args, timeout=timeout))
    data = resp.json()
    df = _ninja_to_dataframe(data, "wind_cf")

    if use_cache:
        _save_cache(df, cache_path)
    time.sleep(rate_pause)
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  RENEWABLES.NINJA — VARIABLES METEOROLÓGICAS CRUDAS (raw=true)
# ─────────────────────────────────────────────────────────────────────────────
# Con raw=true, la API devuelve, además de 'electricity', las variables que
# alimentan el modelo: irradiancia y temperatura (PV); velocidad de viento
# (eólica). Esto permite reconstruir la generación con un modelo físico propio,
# de modo que η_PV, Cp, H_hub, cut-in/out, etc. SÍ influyan en el GSA.
#
# Los nombres de columna que devuelve Ninja con raw=true pueden variar entre
# versiones (p.ej. 'irradiance_surface', 'temperature', 'wind_speed',
# 'windspeed', 'speed'...). El parser de abajo los detecta por patrones y
# normaliza a columnas fijas. Si una llamada real trae nombres distintos, basta
# con ampliar las listas de patrones.

def _ninja_raw_to_dataframe(data: dict) -> pd.DataFrame:
    """
    Convierte la respuesta raw de Ninja en un DataFrame con índice datetime y
    TODAS las columnas devueltas (sin renombrar). El desempaquetado de claves
    temporales reutiliza la lógica robusta de _ninja_to_dataframe.
    """
    if "data" in data and isinstance(data["data"], dict):
        data = data["data"]
    df = pd.DataFrame.from_dict(data, orient="index")
    idx_raw = df.index.astype(str)
    if idx_raw.str.fullmatch(r"\d+").all():
        as_int = idx_raw.astype("int64")
        unit = "ms" if idx_raw.str.len().max() >= 12 else "s"
        df.index = pd.to_datetime(as_int, unit=unit, utc=True)
    else:
        df.index = pd.to_datetime(idx_raw, utc=True)
    return df.sort_index()


def _pick_column(df: pd.DataFrame, patterns, what: str) -> str:
    """Devuelve el nombre de la primera columna que casa con algún patrón."""
    low = {c.lower(): c for c in df.columns}
    for pat in patterns:
        for lc, orig in low.items():
            if pat in lc:
                return orig
    raise RuntimeError(
        f"No encuentro la columna de {what} en la respuesta raw de Ninja. "
        f"Columnas disponibles: {list(df.columns)}. "
        f"Amplía los patrones en _pick_column.")


def fetch_ninja_pv_raw(lat: float, lon: float, year: int,
                       dataset: str = "merra2",
                       token: Optional[str] = None,
                       use_cache: bool = True,
                       rate_pause: float = 12.0,
                       timeout: int = 120) -> pd.DataFrame:
    """
    Descarga irradiancia y temperatura horarias (variables crudas) para PV.

    Devuelve DataFrame con columnas:
        ghi_wm2  : irradiancia a nivel de superficie [W/m²]
        t_amb_c  : temperatura ambiente a 2 m [°C]

    Estas alimentan pv_power() de hres_model.py, de modo que η_PV, α_T y η_inv
    influyan en la generación (y por tanto en el GSA).
    """
    cache_path = _cache_key("ninja_pv_raw", lat=lat, lon=lon, year=year,
                            dataset=dataset)
    if use_cache:
        cached = _load_cache(cache_path)
        if cached is not None:
            return cached

    s = _ninja_session(token)
    url = f"{NINJA_BASE}/data/pv"
    args = {
        "lat": round(float(lat), 4), "lon": round(float(lon), 4),
        "date_from": f"{year}-01-01", "date_to": f"{year}-12-31",
        "dataset": dataset, "capacity": 1,
        "system_loss": 0.1, "tracking": 0, "tilt": 35, "azim": 180,
        "format": "json", "raw": "true",
    }
    resp = _ninja_check(s.get(url, params=args, timeout=timeout))
    raw = _ninja_raw_to_dataframe(resp.json())

    # Detectar columnas (irradiancia de superficie y temperatura)
    irr_col = _pick_column(
        raw, ["irradiance_surface", "irradiance", "swgdn", "ghi", "radiation"],
        "irradiancia")
    tmp_col = _pick_column(
        raw, ["temperature", "t2m", "temp"], "temperatura")

    df = pd.DataFrame({
        "ghi_wm2": pd.to_numeric(raw[irr_col], errors="coerce"),
        "t_amb_c": pd.to_numeric(raw[tmp_col], errors="coerce"),
    }, index=raw.index)

    if use_cache:
        _save_cache(df, cache_path)
    time.sleep(rate_pause)
    return df


def fetch_ninja_wind_raw(lat: float, lon: float, year: int,
                         height: float = 80,
                         turbine: str = "Vestas V90 2000",
                         dataset: str = "merra2",
                         token: Optional[str] = None,
                         use_cache: bool = True,
                         rate_pause: float = 12.0,
                         timeout: int = 120) -> pd.DataFrame:
    """
    Descarga la velocidad de viento horaria (variable cruda) para eólica.

    Devuelve DataFrame con columna:
        wind_ms : velocidad de viento [m/s]

    OJO con la altura: Ninja devuelve la velocidad a la 'height' solicitada.
    Para que H_hub influya en el GSA mediante extrapolación logarítmica, hay
    que pedir el viento a una altura de REFERENCIA fija (aquí 10 m) y dejar que
    wind_power() de hres_model.py lo extrapole a la altura de buje variable.
    Por eso se fuerza height=10 en la petición raw.
    """
    REF_HEIGHT = 10.0
    cache_path = _cache_key("ninja_wind_raw", lat=lat, lon=lon, year=year,
                            ref_height=REF_HEIGHT, dataset=dataset)
    if use_cache:
        cached = _load_cache(cache_path)
        if cached is not None:
            return cached

    s = _ninja_session(token)
    url = f"{NINJA_BASE}/data/wind"
    args = {
        "lat": round(float(lat), 4), "lon": round(float(lon), 4),
        "date_from": f"{year}-01-01", "date_to": f"{year}-12-31",
        "capacity": 1, "height": REF_HEIGHT, "turbine": turbine,
        "dataset": dataset, "format": "json", "raw": "true",
    }
    resp = _ninja_check(s.get(url, params=args, timeout=timeout))
    raw = _ninja_raw_to_dataframe(resp.json())

    wind_col = _pick_column(
        raw, ["wind_speed", "windspeed", "wind", "speed"], "velocidad de viento")

    df = pd.DataFrame({
        "wind_ms": pd.to_numeric(raw[wind_col], errors="coerce"),
    }, index=raw.index)
    # La columna ya está a REF_HEIGHT=10 m (wind_power extrapola a H_hub)
    df.attrs["ref_height"] = REF_HEIGHT

    if use_cache:
        _save_cache(df, cache_path)
    time.sleep(rate_pause)
    return df

def build_real_dataset(lat: float, lon: float, year: int,
                       pv_kwargs: dict = None,
                       wind_kwargs: dict = None,
                       esios_indicator: int = 1293,
                       esios_token: Optional[str] = None,
                       ninja_token: Optional[str] = None,
                       demand_scale_to_peak_kw: Optional[float] = None,
                       use_cache: bool = True) -> pd.DataFrame:
    """
    Construye el dataset horario real y alineado para alimentar el despacho.

    Devuelve un DataFrame con columnas:
        pv_cf     : factor de capacidad PV [0-1]
        wind_cf   : factor de capacidad eólica [0-1]
        demand_mw : demanda real ESIOS [MW]  (o reescalada, ver abajo)

    Parámetros
    ----------
    demand_scale_to_peak_kw : si se indica, la demanda peninsular (GW) se
        reescala linealmente para que su pico iguale este valor [kW]. Útil
        para estudiar una microrred que SIGUE EL PERFIL real de demanda
        nacional pero a la escala de tu sistema. Si None, se devuelve la
        demanda real sin reescalar (en las unidades originales del indicador).

    Notas
    -----
    Las tres series se alinean por intersección de timestamps (UTC) y se
    interpola/elimina cualquier hueco menor. El año bisiesto se recorta a 8760
    descartando el 29-feb para mantener longitud constante en el GSA.
    """
    pv_kwargs = pv_kwargs or {}
    wind_kwargs = wind_kwargs or {}

    df_pv   = fetch_ninja_pv(lat, lon, year, token=ninja_token,
                             use_cache=use_cache, **pv_kwargs)
    df_wind = fetch_ninja_wind(lat, lon, year, token=ninja_token,
                               use_cache=use_cache, **wind_kwargs)
    df_dem  = fetch_esios_demand(f"{year}-01-01", f"{year}-12-31",
                                 indicator=esios_indicator, token=esios_token,
                                 use_cache=use_cache)

    # Alinear por timestamp (intersección)
    df = pd.concat([df_pv["pv_cf"], df_wind["wind_cf"], df_dem["demand_mw"]],
                   axis=1, join="inner")
    df = df.interpolate(limit=3).dropna()

    # Recortar 29-feb en años bisiestos → 8760 h
    mask_leap = ~((df.index.month == 2) & (df.index.day == 29))
    df = df[mask_leap]

    # Reescalado opcional de la demanda al tamaño de la microrred
    if demand_scale_to_peak_kw is not None:
        peak = df["demand_mw"].max()
        df["demand_kw"] = df["demand_mw"] / peak * demand_scale_to_peak_kw
    else:
        df["demand_kw"] = df["demand_mw"]  # asume ya en kW si el indicador lo da

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  GENERADOR MOCK (para validar sin red ni tokens)
# ─────────────────────────────────────────────────────────────────────────────

def build_mock_dataset(year: int = 2019, peak_kw: float = 25.0,
                       seed: int = 42) -> pd.DataFrame:
    """
    Genera un dataset con la MISMA estructura que build_real_dataset pero con
    datos sintéticos. Permite probar todo el pipeline offline; en producción
    se sustituye por build_real_dataset() con tokens reales.
    """
    idx = pd.date_range(f"{year}-01-01", periods=8760, freq="h", tz="UTC")
    rng = np.random.default_rng(seed)
    hod = idx.hour.values
    doy = idx.dayofyear.values

    # PV CF
    decl = 23.45 * np.sin(2*np.pi*(doy-81)/365)
    ha = 15*(hod-12)
    cosz = np.clip(np.sin(np.radians(37))*np.sin(np.radians(decl)) +
                   np.cos(np.radians(37))*np.cos(np.radians(decl))*np.cos(np.radians(ha)), 0, 1)
    pv_cf = np.clip(0.85*cosz*(0.7+0.15*np.sin(2*np.pi*doy/365-np.pi/2))
                    *(1+0.1*rng.standard_normal(8760)), 0, 1)
    pv_cf[pv_cf < 0.01] = 0

    # Wind CF
    u = 7.0*rng.weibull(2.0, 8760)*(1+0.2*np.cos(2*np.pi*doy/365))
    wind_cf = np.clip((u/14)**3, 0, 1)
    wind_cf = np.where(u < 3, 0, np.where(u > 25, 0, wind_cf))

    # Demanda
    base = np.array([.35,.30,.28,.27,.30,.40,.55,.70,.80,.82,.80,.78,
                     .75,.70,.68,.70,.78,.92,1.0,.98,.90,.80,.65,.50])
    dem = peak_kw*base[hod]*(1+0.15*np.sin(2*np.pi*doy/365-np.pi/2))*(1+0.08*rng.standard_normal(8760))
    dem = np.clip(dem, 0, peak_kw*1.2)

    return pd.DataFrame({"pv_cf": pv_cf, "wind_cf": wind_cf,
                         "demand_mw": dem, "demand_kw": dem}, index=idx)


if __name__ == "__main__":
    # Demostración offline con datos mock
    print("Generando dataset MOCK (offline)...")
    df = build_mock_dataset(peak_kw=25.0)
    print(df.describe())
    print(f"\nHoras: {len(df)}  |  PV CF medio: {df['pv_cf'].mean():.3f}  "
          f"| Wind CF medio: {df['wind_cf'].mean():.3f}  "
          f"| Demanda pico: {df['demand_kw'].max():.1f} kW")
    print("\nPara datos REALES:")
    print('  export ESIOS_TOKEN="..."; export NINJA_TOKEN="..."')
    print('  df = build_real_dataset(lat=37.4, lon=-5.98, year=2019,')
    print('         demand_scale_to_peak_kw=25.0)')
