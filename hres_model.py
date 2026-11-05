"""
=============================================================================
HRES Energy Dispatch Model - Hybrid Renewable Energy System
=============================================================================
Sistema: PV + Eólica + Batería + Generador Diésel de respaldo
Despacho: Estrategia de prioridad (renovables primero) con seguimiento de carga
=============================================================================
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Tuple, Dict


# ─────────────────────────────────────────────────────────────────────────────
#  PARÁMETROS DEL SISTEMA
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HRESParameters:
    """Parámetros completos del sistema HRES."""

    # ── Solar PV ─────────────────────────────────────────────────────────────
    pv_capacity_kw: float = 50.0        # Potencia instalada PV [kW]
    pv_efficiency: float = 0.18         # Eficiencia del módulo [-]
    pv_temp_coeff: float = -0.004       # Coef. temperatura [1/°C]
    pv_noct: float = 45.0               # NOCT [°C]
    pv_degradation: float = 0.005       # Degradación anual [-]
    pv_inverter_eff: float = 0.96       # Eficiencia inversor PV [-]
    pv_mismatch_loss: float = 0.02      # Pérdidas por desajuste [-]
    pv_wiring_loss: float = 0.02        # Pérdidas cableado [-]

    # ── Eólica ────────────────────────────────────────────────────────────────
    wind_capacity_kw: float = 30.0      # Potencia instalada eólica [kW]
    wind_efficiency: float = 0.40       # Eficiencia aerogenerador (Cp) [-]
    wind_cut_in: float = 2.5            # Velocidad de arranque [m/s]
    wind_rated: float = 12.0            # Velocidad nominal [m/s]
    wind_cut_out: float = 25.0          # Velocidad de parada [m/s]
    wind_hub_height: float = 30.0       # Altura del buje [m]
    wind_roughness: float = 0.15        # Longitud de rugosidad [m]
    wind_gearbox_eff: float = 0.97      # Eficiencia caja multiplicadora [-]
    wind_generator_eff: float = 0.95    # Eficiencia generador eléctrico [-]

    # ── Batería (BESS) ────────────────────────────────────────────────────────
    batt_capacity_kwh: float = 60.0     # Capacidad energía [kWh]
    batt_power_kw: float = 20.0         # Potencia máx carga/descarga [kW]
    batt_charge_eff: float = 0.95       # Eficiencia de carga [-]
    batt_discharge_eff: float = 0.95    # Eficiencia de descarga [-]
    batt_soc_min: float = 0.20          # SOC mínimo (DoD máx) [-]
    batt_soc_max: float = 0.95          # SOC máximo [-]
    batt_soc_init: float = 0.50         # SOC inicial [-]
    batt_self_discharge: float = 0.0002 # Autodescarga por hora [-]
    batt_degradation: float = 0.02      # Degradación anual [-]

    # ── Generador Diésel (backup) ─────────────────────────────────────────────
    diesel_capacity_kw: float = 20.0    # Potencia nominal [kW]
    diesel_min_load: float = 0.30       # Carga mínima eficiente [-]
    diesel_eff_rated: float = 0.35      # Eficiencia a plena carga [-]
    diesel_fuel_lhv: float = 10.0       # PCI del diésel [kWh/L]
    diesel_startup_fuel: float = 0.5    # Combustible arranque [L]

    # ── Carga ─────────────────────────────────────────────────────────────────
    load_scale: float = 1.0             # Factor de escala de carga [-]
    load_peak_kw: float = 25.0          # Pico de demanda [kW]

    # ── Convertidor DC/AC ─────────────────────────────────────────────────────
    converter_eff: float = 0.97         # Eficiencia convertidor general [-]

    # ── Económicos (para LCOE; defaults = valores de referencia ~2024) ─────────
    capex_pv_eur_kw: float = 650.0      # CAPEX PV [€/kW]
    capex_wind_eur_kw: float = 1200.0   # CAPEX eólica [€/kW]
    capex_batt_eur_kwh: float = 180.0   # CAPEX batería energía [€/kWh]
    capex_batt_eur_kw: float = 140.0    # CAPEX batería potencia [€/kW]
    capex_diesel_eur_kw: float = 400.0  # CAPEX diésel [€/kW]
    fuel_price_eur_l: float = 1.20      # Precio combustible [€/L]
    discount_rate: float = 0.06         # Tasa de descuento [-]
    lifetime_years: float = 20.0        # Vida útil del proyecto [años]


# ─────────────────────────────────────────────────────────────────────────────
#  GENERACIÓN DE DATOS METEOROLÓGICOS SINTÉTICOS
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_weather(hours: int = 8760, seed: int = 42) -> pd.DataFrame:
    """
    Genera datos meteorológicos horarios sintéticos representativos de un
    emplazamiento de clima templado-semiárido (p.ej. sur de España).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(hours)
    day = t // 24
    hour_of_day = t % 24

    # ── Irradiancia solar [W/m²] ─────────────────────────────────────────────
    # Variación estacional (día del año) + variación diaria (ángulo horario)
    day_angle = 2 * np.pi * day / 365
    declination = 23.45 * np.sin(2 * np.pi * (day - 81) / 365)
    hour_angle = 15 * (hour_of_day - 12)  # grados
    lat = 37.0
    cos_zenith = (
        np.sin(np.radians(lat)) * np.sin(np.radians(declination))
        + np.cos(np.radians(lat))
        * np.cos(np.radians(declination))
        * np.cos(np.radians(hour_angle))
    )
    cos_zenith = np.clip(cos_zenith, 0, 1)
    ghi_clear = 950 * cos_zenith  # Irradiancia en cielo despejado
    # Índice de claridad estacional
    kt_seasonal = 0.65 + 0.15 * np.sin(day_angle - np.pi / 2)
    kt = kt_seasonal * (1 + 0.1 * rng.standard_normal(hours))
    kt = np.clip(kt, 0.2, 1.0)
    ghi = ghi_clear * kt
    ghi[ghi < 10] = 0

    # ── Temperatura ambiente [°C] ──────────────────────────────────────────────
    t_seasonal = 18 + 10 * np.sin(day_angle - np.pi / 2)
    t_daily = 6 * np.sin(np.pi * (hour_of_day - 6) / 12)
    t_noise = 2 * rng.standard_normal(hours)
    t_amb = t_seasonal + t_daily + t_noise

    # ── Velocidad de viento [m/s] a 10 m ──────────────────────────────────────
    # Distribución de Weibull (k=2, c=7 típico costa mediterránea)
    k_weibull, c_weibull = 2.0, 7.0
    u_base = c_weibull * rng.weibull(k_weibull, hours)
    # Variación diaria (máximo al mediodía)
    u_daily = 1 + 0.3 * np.sin(np.pi * hour_of_day / 12)
    # Variación estacional (más viento en invierno)
    u_seasonal = 1 + 0.2 * np.cos(day_angle)
    wind_10m = u_base * u_daily * u_seasonal
    wind_10m = np.clip(wind_10m, 0, 35)

    return pd.DataFrame({
        "ghi_wm2": ghi,
        "t_amb_c": t_amb,
        "wind_10m_ms": wind_10m,
    }, index=pd.date_range("2024-01-01", periods=hours, freq="h"))


def generate_load_profile(hours: int = 8760, peak_kw: float = 25.0,
                           scale: float = 1.0, seed: int = 42) -> np.ndarray:
    """
    Perfil de carga horario con patrón residencial/comercial mixto.
    """
    rng = np.random.default_rng(seed + 1)
    t = np.arange(hours)
    hour_of_day = t % 24
    day = t // 24
    day_of_week = day % 7

    # Perfil base horario normalizado (0-1)
    base_profile = np.array([
        0.35, 0.30, 0.28, 0.27, 0.30, 0.40,
        0.55, 0.70, 0.80, 0.82, 0.80, 0.78,
        0.75, 0.70, 0.68, 0.70, 0.78, 0.92,
        1.00, 0.98, 0.90, 0.80, 0.65, 0.50,
    ])
    load_base = base_profile[hour_of_day]

    # Factor fin de semana
    weekend_factor = np.where(day_of_week >= 5, 0.85, 1.0)
    # Factor estacional (más consumo en verano por refrigeración)
    day_angle = 2 * np.pi * day / 365
    seasonal_factor = 1 + 0.15 * np.sin(2 * np.pi * day / 365 - np.pi / 2)
    # Ruido estocástico
    noise = 1 + 0.08 * rng.standard_normal(hours)

    load = peak_kw * scale * load_base * weekend_factor * seasonal_factor * noise
    return np.clip(load, 0, peak_kw * scale * 1.2)


# ─────────────────────────────────────────────────────────────────────────────
#  MODELOS DE GENERACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def pv_power(params: HRESParameters, ghi: np.ndarray,
             t_amb: np.ndarray) -> np.ndarray:
    """
    Modelo PV con corrección de temperatura (modelo de un diodo simplificado).
    P_pv = P_stc * (G/G_stc) * η_temp * η_inv * (1 - L_mismatch) * (1 - L_wire)
    """
    G_stc = 1000.0  # [W/m²]
    T_stc = 25.0    # [°C]

    # Temperatura de célula (modelo NOCT)
    t_cell = t_amb + (params.pv_noct - 20.0) * ghi / 800.0

    # Factor de temperatura
    eta_temp = 1 + params.pv_temp_coeff * (t_cell - T_stc)
    eta_temp = np.clip(eta_temp, 0.5, 1.1)

    # Pérdidas combinadas
    system_losses = (1 - params.pv_mismatch_loss) * (1 - params.pv_wiring_loss)

    p_pv = (params.pv_capacity_kw
            * (ghi / G_stc)
            * eta_temp
            * params.pv_inverter_eff
            * system_losses)

    return np.clip(p_pv, 0, params.pv_capacity_kw)


def wind_power(params: HRESParameters, wind_10m: np.ndarray) -> np.ndarray:
    """
    Curva de potencia aerogenerador con extrapolación logarítmica de altura.
    Zona cúbica entre cut-in y nominal, saturación entre nominal y cut-out.
    """
    # Extrapolación logarítmica de altura
    if params.wind_hub_height != 10.0:
        alpha = np.log(params.wind_hub_height / 10.0) / np.log(params.wind_hub_height / params.wind_roughness)
        wind_hub = wind_10m * (params.wind_hub_height / 10.0) ** alpha
    else:
        wind_hub = wind_10m.copy()

    u = wind_hub
    p_rated = params.wind_capacity_kw
    u_ci = params.wind_cut_in
    u_r = params.wind_rated
    u_co = params.wind_cut_out
    eta_mech = params.wind_gearbox_eff * params.wind_generator_eff

    # Curva de potencia por tramos
    p_wind = np.where(
        u < u_ci, 0.0,
        np.where(
            u < u_r,
            p_rated * eta_mech * params.wind_efficiency * ((u - u_ci) / (u_r - u_ci)) ** 3,
            np.where(u <= u_co, p_rated * eta_mech, 0.0)
        )
    )
    return np.clip(p_wind, 0, p_rated)


# ─────────────────────────────────────────────────────────────────────────────
#  DESPACHO ENERGÉTICO (ESTRATEGIA LOAD-FOLLOWING CON PRIORIDAD RENOVABLE)
# ─────────────────────────────────────────────────────────────────────────────

def dispatch(params: HRESParameters,
             p_pv: np.ndarray,
             p_wind: np.ndarray,
             load: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Algoritmo de despacho horario para HRES.

    Estrategia (prioridad renovable):
    1. Renovables satisfacen la carga directamente
    2. El excedente carga la batería
    3. Si no hay suficiente renovable → descarga la batería
    4. Si batería insuficiente → arranca generador diésel
    5. Excedente sobrante → vertido a red / carga forzada
    """
    n = len(load)
    dt = 1.0  # paso temporal [h]

    # ── Arrays de salida ───────────────────────────────────────────────────────
    soc       = np.zeros(n + 1)
    p_batt    = np.zeros(n)   # >0: descarga, <0: carga
    p_diesel  = np.zeros(n)
    p_dump    = np.zeros(n)   # vertido / excedente no utilizado
    p_unmet   = np.zeros(n)   # carga no suministrada (LPSP)
    fuel_l    = np.zeros(n)   # consumo combustible [L]
    starts    = np.zeros(n)   # arranques del generador

    # SOC inicial
    soc[0] = params.batt_soc_init
    # Guarda numérica: evita división por cero si la batería es nula
    E_batt_safe = params.batt_capacity_kwh if params.batt_capacity_kwh > 1e-9 else 1e-9

    # Eficiencias de batería
    eta_c = params.batt_charge_eff
    eta_d = params.batt_discharge_eff

    prev_diesel_on = False

    for i in range(n):
        # Autodescarga de la batería
        soc_avail = soc[i] * (1 - params.batt_self_discharge)

        # Balance neto renovable vs carga
        p_re = (p_pv[i] + p_wind[i]) * params.converter_eff
        surplus = p_re - load[i]

        if surplus >= 0:
            # ── CASO 1: Excedente renovable ─────────────────────────────────
            # Intentar cargar batería
            p_charge = min(surplus, params.batt_power_kw)
            energy_store = p_charge * eta_c * dt
            soc_new = soc_avail + energy_store / E_batt_safe
            soc_new = min(soc_new, params.batt_soc_max)
            actual_charge = (soc_new - soc_avail) * params.batt_capacity_kwh / (eta_c * dt)
            p_batt[i] = -actual_charge
            p_dump[i] = max(0, surplus - actual_charge)
            soc[i + 1] = soc_new

        else:
            # ── CASO 2: Déficit energético ──────────────────────────────────
            deficit = -surplus

            # 2a. Descargar batería
            energy_needed = deficit * dt / eta_d
            energy_avail = (soc_avail - params.batt_soc_min) * params.batt_capacity_kwh
            energy_from_batt = min(energy_needed, energy_avail,
                                   params.batt_power_kw * dt / eta_d)
            p_batt_d = energy_from_batt * eta_d / dt
            p_batt[i] = p_batt_d
            deficit_remaining = deficit - p_batt_d
            soc[i + 1] = soc_avail - energy_from_batt / E_batt_safe

            # 2b. Arrancar generador diésel si el déficit persiste
            if deficit_remaining > 1e-3:
                # Carga mínima eficiente del generador
                p_min_d = params.diesel_min_load * params.diesel_capacity_kw
                p_diesel_i = max(p_min_d, min(deficit_remaining, params.diesel_capacity_kw))
                p_diesel[i] = p_diesel_i

                # Consumo de combustible (curva lineal: FC = a + b·P)
                # Coeficientes para motor típico (Jawad & Bhattacharya 2020)
                a_coef = 0.08415 # [L/kWh] no-load coefficient
                b_coef = 0.246   # [L/kWh] marginal coefficient
                fuel_l[i] = (a_coef * params.diesel_capacity_kw
                              + b_coef * p_diesel_i) * dt

                # Coste de arranque
                if not prev_diesel_on:
                    fuel_l[i] += params.diesel_startup_fuel
                    starts[i] = 1
                prev_diesel_on = True

                # Excedente del generador va a la batería
                p_dg_surplus = p_diesel_i - deficit_remaining
                if p_dg_surplus > 0:
                    energy_extra = p_dg_surplus * eta_c * dt
                    soc_extra = soc[i + 1] + energy_extra / E_batt_safe
                    soc[i + 1] = min(soc_extra, params.batt_soc_max)
                    p_batt[i] -= p_dg_surplus  # carga adicional

                # Carga no suministrada residual
                p_unmet[i] = max(0, deficit_remaining - p_diesel_i)
            else:
                prev_diesel_on = False
                p_unmet[i] = 0.0

        # Clip SOC
        soc[i + 1] = np.clip(soc[i + 1], params.batt_soc_min, params.batt_soc_max)

    return {
        "soc":      soc[:-1],
        "p_batt":   p_batt,
        "p_diesel": p_diesel,
        "p_dump":   p_dump,
        "p_unmet":  p_unmet,
        "fuel_l":   fuel_l,
        "starts":   starts,
        "p_pv":     p_pv,
        "p_wind":   p_wind,
        "load":     load,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  INDICADORES DE RENDIMIENTO (KPIs)
# ─────────────────────────────────────────────────────────────────────────────

def compute_kpis(results: Dict[str, np.ndarray],
                 params: HRESParameters) -> Dict[str, float]:
    """
    Calcula los KPIs técnico-económicos del sistema HRES.
    """
    load      = results["load"]
    p_pv      = results["p_pv"]
    p_wind    = results["p_wind"]
    p_diesel  = results["p_diesel"]
    p_unmet   = results["p_unmet"]
    p_dump    = results["p_dump"]
    fuel_l    = results["fuel_l"]
    starts    = results["starts"]
    soc       = results["soc"]
    n         = len(load)

    # ── Energías totales [kWh] ─────────────────────────────────────────────
    E_load    = np.sum(load)
    E_pv      = np.sum(p_pv)
    E_wind    = np.sum(p_wind)
    E_diesel  = np.sum(p_diesel)
    E_unmet   = np.sum(p_unmet)
    E_dump    = np.sum(p_dump)
    E_re      = E_pv + E_wind

    # ── Indicadores técnicos ───────────────────────────────────────────────
    # LPSP: Loss of Power Supply Probability
    lpsp = E_unmet / E_load if E_load > 0 else 0.0

    # REF: Renewable Energy Fraction
    ref = min(1.0, (E_re - E_dump) / E_load) if E_load > 0 else 0.0

    # Self-sufficiency ratio
    ssr = 1.0 - lpsp

    # Tasa de excedente / curtailment
    curtailment = E_dump / (E_re) if E_re > 0 else 0.0

    # Ciclos equivalentes de batería
    p_batt_neg = np.abs(np.minimum(results["p_batt"], 0))  # energía cargada
    equiv_cycles = np.sum(p_batt_neg) / (params.batt_capacity_kwh + 1e-9)

    # Factor de capacidad PV y eólica
    cf_pv   = E_pv / (params.pv_capacity_kw * n) if params.pv_capacity_kw > 0 else 0
    cf_wind = E_wind / (params.wind_capacity_kw * n) if params.wind_capacity_kw > 0 else 0

    # SOC estadísticos
    soc_mean = np.mean(soc)
    soc_std  = np.std(soc)

    # ── Indicadores económicos (LCOE) — costes desde params (parametrizables)
    fuel_price   = getattr(params, "fuel_price_eur_l", 1.20)
    lifetime_yr  = getattr(params, "lifetime_years", 20)
    discount_rate= getattr(params, "discount_rate", 0.06)
    capex_pv     = getattr(params, "capex_pv_eur_kw", 650)
    capex_wind   = getattr(params, "capex_wind_eur_kw", 1200)
    capex_batt_e = getattr(params, "capex_batt_eur_kwh", 180)
    capex_batt_p = getattr(params, "capex_batt_eur_kw", 140)
    capex_diesel = getattr(params, "capex_diesel_eur_kw", 400)
    opex_pct = getattr(params, "opex_pct", None) or {
        "pv": 0.02, "wind": 0.025, "batt": 0.01, "diesel": 0.03}   # Table 4
    opex_pv_pct     = opex_pct["pv"]
    opex_wind_pct   = opex_pct["wind"]
    opex_batt_pct   = opex_pct["batt"]
    opex_diesel_pct = opex_pct["diesel"]

    # Coste de la batería: energía (€/kWh) + potencia (€/kW)
    capex_batt_total = (params.batt_capacity_kwh * capex_batt_e
                        + params.batt_power_kw * capex_batt_p)

    # CAPEX total [€]
    capex = (params.pv_capacity_kw   * capex_pv
           + params.wind_capacity_kw  * capex_wind
           + capex_batt_total
           + params.diesel_capacity_kw * capex_diesel)

    # OPEX anual [€/año]
    opex_annual = (params.pv_capacity_kw   * capex_pv   * opex_pv_pct
                 + params.wind_capacity_kw  * capex_wind  * opex_wind_pct
                 + capex_batt_total * opex_batt_pct
                 + params.diesel_capacity_kw * capex_diesel * opex_diesel_pct
                 + np.sum(fuel_l) * fuel_price
                 + np.sum(starts) * getattr(params, "startup_cost_eur", 15.0))

    # CRF: Capital Recovery Factor
    crf = (discount_rate * (1 + discount_rate) ** lifetime_yr /
           ((1 + discount_rate) ** lifetime_yr - 1))

    lcoe = (capex * crf + opex_annual) / (E_load - E_unmet + 1e-9)

    # Emisiones CO2 [kg]
    co2_kg = np.sum(fuel_l) * 2.65  # 2.65 kg CO2/L diésel

    return {
        # Técnicos
        "LPSP":          lpsp,
        "REF":           ref,
        "SSR":           ssr,
        "Curtailment":   curtailment,
        "CF_PV":         cf_pv,
        "CF_Wind":       cf_wind,
        "SOC_mean":      soc_mean,
        "SOC_std":       soc_std,
        "Equiv_cycles":  equiv_cycles,
        # Energías
        "E_load_kWh":    E_load,
        "E_PV_kWh":      E_pv,
        "E_Wind_kWh":    E_wind,
        "E_Diesel_kWh":  E_diesel,
        "E_Unmet_kWh":   E_unmet,
        "E_Dump_kWh":    E_dump,
        # Económicos
        "LCOE_eur_kWh":  lcoe,
        "Fuel_L":        np.sum(fuel_l),
        "CO2_kg":        co2_kg,
        "CAPEX_eur":     capex,
        "OPEX_annual":   opex_annual,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  FUNCIÓN DE SIMULACIÓN COMPLETA
# ─────────────────────────────────────────────────────────────────────────────

def run_hres_simulation(param_vector: np.ndarray,
                         weather: pd.DataFrame,
                         param_names: list) -> Dict[str, float]:
    """
    Función wrapper para la simulación HRES dado un vector de parámetros.
    Usada por el análisis de sensibilidad de Morris.
    """
    # Construir objeto params desde vector
    p = HRESParameters()
    for name, val in zip(param_names, param_vector):
        setattr(p, name, float(val))

    # Generación de energía
    ghi   = weather["ghi_wm2"].values
    t_amb = weather["t_amb_c"].values
    wind  = weather["wind_10m_ms"].values

    p_pv_h   = pv_power(p, ghi, t_amb)
    p_wind_h = wind_power(p, wind)
    load_h   = generate_load_profile(len(ghi),
                                      peak_kw=p.load_peak_kw,
                                      scale=p.load_scale)

    # Despacho
    results = dispatch(p, p_pv_h, p_wind_h, load_h)

    # KPIs
    return compute_kpis(results, p)
