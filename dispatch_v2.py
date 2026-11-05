"""
=============================================================================
HRES Dispatch v2 — MILP (Gurobi) con DATOS REALES
=============================================================================
Versión adaptada para:
  (1) Recurso PV/eólico desde factores de capacidad reales (Renewables.ninja).
  (2) Demanda real (ESIOS/REE).
  (3) Licencia Gurobi ACADÉMICA completa → despacho ANUAL MONOLÍTICO por
      defecto (solve_annual_milp). El rolling horizon queda disponible como
      alternativa (dispatch_milp_rolling, en milp_dispatch.py) por si se desea
      acelerar barridos masivos del GSA.

Diferencia clave con la v1: la generación renovable ya NO se modela con
ecuaciones físicas internas (NOCT, curva de potencia, etc.), sino que se toma
directamente de los factores de capacidad medidos/simulados:

        P_pv(t)   = cf_pv(t)   · P_pv_instalada
        P_wind(t) = cf_wind(t) · P_wind_instalada

Esto es lo correcto al usar Renewables.ninja, cuyos perfiles ya incorporan la
física del recurso (irradiancia, temperatura, curva de turbina) validada
contra mediciones (Pfenninger & Staffell 2016).
=============================================================================
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from dataclasses import dataclass
from typing import Dict, Optional

from milp_dispatch import MILPCostParameters


# ─────────────────────────────────────────────────────────────────────────────
#  GENERACIÓN A PARTIR DE FACTORES DE CAPACIDAD REALES
# ─────────────────────────────────────────────────────────────────────────────

def power_from_cf(cf_pv: np.ndarray, cf_wind: np.ndarray,
                  pv_capacity_kw: float, wind_capacity_kw: float
                  ) -> tuple:
    """
    Convierte factores de capacidad [0-1] en potencia [kW].
    Reemplaza pv_power()/wind_power() del modelo sintético.
    """
    p_pv   = np.clip(cf_pv, 0, 1) * pv_capacity_kw
    p_wind = np.clip(cf_wind, 0, 1) * wind_capacity_kw
    return p_pv, p_wind


# ─────────────────────────────────────────────────────────────────────────────
#  DESPACHO MILP ANUAL MONOLÍTICO (requiere licencia completa)
# ─────────────────────────────────────────────────────────────────────────────

def _default_diesel_fleet(total_kw: float) -> list:
    """
    Flota diésel representativa de Los Guinchos (grupos de distinto tamaño),
    escalada para que la suma ≈ total_kw. Estructura de referencia (proporción
    de tamaños fiel a la central real: pequeños + medianos + grande).
    """
    # Real Los Guinchos diesel engines (10 units, 82.84 MW), gas turbine
    # excluded. Unit sizes are defined once in config_paper.py. The fleet is
    # scaled proportionally so that its sum equals total_kw.
    from config_paper import LOS_GUINCHOS_ENGINES_MW
    ref = np.array(LOS_GUINCHOS_ENGINES_MW, float)
    fleet = ref / ref.sum() * total_kw
    return list(fleet)


def solve_annual_milp(params,
                      p_pv: np.ndarray,
                      p_wind: np.ndarray,
                      load: np.ndarray,
                      cost: Optional[MILPCostParameters] = None,
                      cyclic_soc: bool = True,
                      ramp_rate: float = 0.30,
                      min_up_time: int = 3,
                      min_down_time: int = 2,
                      diesel_fleet_kw: Optional[list] = None,
                      soc_start: Optional[float] = None,
                      u_start: Optional[np.ndarray] = None,
                      commit_hours: Optional[int] = None,
                      soc_end: Optional[float] = None,
                      updown_form: str = "rt",
                      mip_focus: int = 0,
                      mip_gap: float = 0.005,
                      time_limit: float = 300.0,
                      threads: int = 0,
                      verbose: bool = False) -> Dict[str, np.ndarray]:
    """
    Despacho óptimo ANUAL en un único MILP con flota diésel MULTI-GRUPO.

    El diésel se modela como varios grupos independientes (como Los Guinchos),
    cada uno con su mínimo técnico, rampa y tiempos mínimos. El MILP decide
    cuántos grupos encender cada hora → despacho escalonado realista (sin los
    picos no físicos de un único grupo monolítico).

        ramp_rate     : rampa máx por grupo [fracción de P_grupo/h] (0.30=30%/h)
        min_up_time   : horas mínimas encendido por grupo tras arranque
        min_down_time : horas mínimas apagado por grupo tras parada
        diesel_fleet_kw : lista de potencias [kW] de cada grupo; si None, se
                          usa la flota de Los Guinchos escalada a la capacidad.

    cyclic_soc : si True, fuerza soc[H] == soc[0] (cierre energético anual).
    Requiere licencia Gurobi sin límite de tamaño (académica/comercial).
    """
    if cost is None:
        cost = MILPCostParameters()

    H = len(load)
    dt = 1.0

    E_batt   = params.batt_capacity_kwh
    P_batt   = params.batt_power_kw
    # Coherencia física: si no hay capacidad de almacenamiento, no puede haber
    # potencia de carga/descarga (evita que el MILP use la batería como un
    # "cable" instantáneo de potencia sin energía almacenada).
    if params.batt_capacity_kwh <= 1e-9:
        P_batt = 0.0
    # A battery without charge/discharge power cannot exchange energy: treat it
    # as absent (otherwise self-discharge makes the terminal SOC infeasible).
    if P_batt <= 1e-9:
        P_batt, E_batt = 0.0, 0.0
    eta_c    = params.batt_charge_eff
    eta_d    = params.batt_discharge_eff
    soc_min  = params.batt_soc_min
    soc_max  = params.batt_soc_max
    P_dg_max = params.diesel_capacity_kw
    P_dg_min = params.diesel_min_load * params.diesel_capacity_kw
    eta_conv = params.converter_eff

    m = gp.Model("HRES_annual")
    m.setParam("OutputFlag", 1 if verbose else 0)
    m.setParam("MIPGap", mip_gap)
    m.setParam("TimeLimit", time_limit)
    if threads:
        m.setParam("Threads", threads)
    if mip_focus:
        m.setParam("MIPFocus", mip_focus)

    # ── FLOTA DIÉSEL MULTI-GRUPO (Los Guinchos escalada a la capacidad dada)
    # Estructura realista: grupos de distinto tamaño. Se escala la flota de
    # referencia para que su suma ≈ params.diesel_capacity_kw.
    fleet_ref = diesel_fleet_kw if diesel_fleet_kw is not None else \
        _default_diesel_fleet(params.diesel_capacity_kw)
    n_units = len(fleet_ref)
    P_unit = np.array(fleet_ref, float)          # potencia de cada grupo [kW]
    P_unit_min = params.diesel_min_load * P_unit  # mínimo técnico por grupo

    p_re_used = m.addVars(H, lb=0, name="p_re_used")
    p_ch  = m.addVars(H, lb=0, ub=P_batt, name="p_ch")
    p_dis = m.addVars(H, lb=0, ub=P_batt, name="p_dis")
    soc   = m.addVars(H + 1, lb=soc_min, ub=soc_max, name="soc")
    # Variables por grupo g y hora t
    p_g  = m.addVars(n_units, H, lb=0, name="p_g")        # potencia grupo g
    u_g  = m.addVars(n_units, H, vtype=GRB.BINARY, name="u_g")  # ON/OFF
    y_g  = m.addVars(n_units, H, vtype=GRB.BINARY, name="y_g")  # arranque
    z_g  = m.addVars(n_units, H, vtype=GRB.BINARY, name="z_g")  # parada
    p_unmet = m.addVars(H, lb=0, name="p_unmet")
    p_dump  = m.addVars(H, lb=0, name="p_dump")

    # Potencia diésel total por hora (suma de grupos)
    p_dg = {t: gp.quicksum(p_g[g, t] for g in range(n_units)) for t in range(H)}

    p_re_avail = p_pv + p_wind

    soc_ini = soc_start if soc_start is not None else params.batt_soc_init
    m.addConstr(soc[0] == soc_ini, name="soc_init")
    if cyclic_soc:
        m.addConstr(soc[H] == params.batt_soc_init, name="soc_cyclic")
    soc_short = None
    if not cyclic_soc and soc_end is not None and E_batt > 1e-9:
        # Terminal condition: the window must return the energy it borrowed.
        # Soft constraint (shortfall penalised at VOLL) so the MILP is always
        # feasible, e.g. when self-discharge exceeds the available recharge.
        soc_short = m.addVar(lb=0, name="soc_shortfall")
        m.addConstr(soc[H] + soc_short >= soc_end, name="soc_terminal")

    for t in range(H):
        m.addConstr(p_re_used[t] <= p_re_avail[t])
        m.addConstr(p_dump[t] == p_re_avail[t] - p_re_used[t])
        m.addConstr(
            eta_conv * (p_re_used[t] + p_dis[t] + p_dg[t]) - p_ch[t]
            == load[t] - p_unmet[t])
        m.addConstr(
            soc[t + 1] * E_batt ==
            soc[t] * E_batt * (1 - params.batt_self_discharge)
            + eta_c * p_ch[t] * dt - (p_dis[t] / eta_d) * dt)

        # ── Restricciones POR GRUPO ─────────────────────────────────────────
        for g in range(n_units):
            # Límites técnicos (mínimo y máximo de cada grupo)
            m.addConstr(p_g[g, t] >= P_unit_min[g] * u_g[g, t])
            m.addConstr(p_g[g, t] <= P_unit[g] * u_g[g, t])

            # Arranque/parada acoplados (u_start = estado previo de la ventana)
            if t == 0:
                u_prev = (float(u_start[g]) if u_start is not None else 0)
            else:
                u_prev = u_g[g, t - 1]
            m.addConstr(y_g[g, t] - z_g[g, t] == u_g[g, t] - u_prev)
            m.addConstr(y_g[g, t] + z_g[g, t] <= 1)

            # Rampas por grupo (≤ ramp_rate·P_grupo cuando sigue ON;
            # salto al mínimo técnico permitido en arranque/parada)
            if t > 0:
                Rg = ramp_rate * P_unit[g]
                m.addConstr(p_g[g, t] - p_g[g, t - 1]
                            <= Rg * u_g[g, t - 1] + P_unit_min[g] * y_g[g, t])
                m.addConstr(p_g[g, t - 1] - p_g[g, t]
                            <= Rg * u_g[g, t] + P_unit_min[g] * z_g[g, t])

    # ── Tiempos mínimos de operación por grupo ─────────────────────────────
    for g in range(n_units):
        for t in range(H):
            if updown_form == "rt":
                # Rajan & Takriti (2005) turn-on/turn-off inequalities: same
                # integer solutions, much tighter LP relaxation.
                if min_up_time > 1:
                    m.addConstr(gp.quicksum(y_g[g, k] for k in range(max(0, t - min_up_time + 1), t + 1))
                                <= u_g[g, t])
                if min_down_time > 1:
                    m.addConstr(gp.quicksum(z_g[g, k] for k in range(max(0, t - min_down_time + 1), t + 1))
                                <= 1 - u_g[g, t])
            else:
                if min_up_time > 1:
                    for k in range(t, min(t + min_up_time, H)):
                        m.addConstr(u_g[g, k] >= y_g[g, t])
                if min_down_time > 1:
                    for k in range(t, min(t + min_down_time, H)):
                        m.addConstr(1 - u_g[g, k] >= z_g[g, t])

    # ── Romper simetría entre grupos del mismo tamaño (acelera el MILP) ─────
    # Si dos grupos consecutivos son idénticos, fuerza orden de encendido.
    for g in range(n_units - 1):
        if abs(P_unit[g] - P_unit[g + 1]) < 1e-6:
            for t in range(H):
                m.addConstr(u_g[g, t] >= u_g[g + 1, t])


    # Combustible por grupo: no-load proporcional a P_grupo·u + marginal·p
    fuel = gp.quicksum(
        cost.fuel_price * (cost.diesel_a * P_unit[g] * u_g[g, t]
                           + cost.diesel_b * p_g[g, t]) * dt
        for g in range(n_units) for t in range(H))
    starts = gp.quicksum(
        (cost.startup_cost + cost.fuel_price * cost.startup_fuel) * y_g[g, t]
        for g in range(n_units) for t in range(H))
    unmet = gp.quicksum(cost.unmet_penalty*p_unmet[t]*dt for t in range(H))
    dump  = gp.quicksum(cost.dump_penalty*p_dump[t]*dt for t in range(H))
    deg   = gp.quicksum(cost.batt_deg_cost*(p_ch[t]+p_dis[t])*dt for t in range(H))
    term = (cost.unmet_penalty * E_batt * soc_short) if soc_short is not None else 0
    m.setObjective(fuel+starts+unmet+dump+deg+term, GRB.MINIMIZE)

    m.optimize()
    if m.status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL) or m.SolCount == 0:
        raise RuntimeError(f"MILP not solved. Status {m.status}, solutions {m.SolCount}")

    def arr(v): return np.array([v[t].X for t in range(H)])
    # Agregados de la flota
    p_dg_s = np.array([sum(p_g[g, t].X for g in range(n_units)) for t in range(H)])
    u_s = np.array([sum(u_g[g, t].X for g in range(n_units)) for t in range(H)])  # nº grupos ON
    y_s = np.array([sum(y_g[g, t].X for g in range(n_units)) for t in range(H)])  # arranques totales
    # Combustible por hora (suma de grupos)
    fuel_l = np.array([
        sum(cost.diesel_a * P_unit[g] * u_g[g, t].X
            + cost.diesel_b * p_g[g, t].X for g in range(n_units)) * dt
        + cost.startup_fuel * sum(y_g[g, t].X for g in range(n_units))
        for t in range(H)])
    # Potencia por grupo (para análisis/figuras detalladas)
    p_per_unit = np.array([[p_g[g, t].X for t in range(H)] for g in range(n_units)])

    return {
        "p_pv": p_pv, "p_wind": p_wind, "load": load,
        "p_re_used": arr(p_re_used),
        "p_ch": arr(p_ch), "p_dis": arr(p_dis),
        "p_batt": arr(p_dis) - arr(p_ch),
        "p_dg": p_dg_s, "u_dg": u_s, "y_dg": y_s, "starts": y_s,
        "p_per_unit": p_per_unit, "fleet_kw": P_unit,
        "p_unmet": arr(p_unmet), "p_dump": arr(p_dump),
        "fuel_l": fuel_l,
        "soc": np.array([soc[t].X for t in range(H)]),
        "obj": m.objVal,
        "mip_gap": m.MIPGap,
        "solve_time": m.Runtime,
        "status": int(m.status),
        "soc_shortfall_kwh": (soc_short.X * E_batt) if soc_short is not None else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  ROLLING HORIZON MULTI-GRUPO
# ─────────────────────────────────────────────────────────────────────────────

def solve_rolling_multiunit(params,
                            p_pv: np.ndarray,
                            p_wind: np.ndarray,
                            load: np.ndarray,
                            cost: Optional[MILPCostParameters] = None,
                            window: int = 168,
                            step: int = 168,
                            ramp_rate: float = 0.30,
                            min_up_time: int = 3,
                            min_down_time: int = 2,
                            diesel_fleet_kw: Optional[list] = None,
                            mip_gap: float = 0.01,
                            time_limit_block: float = 60.0,
                            cyclic_window: bool = False,
                            threads: int = 0,
                            updown_form: str = "rt",
                            mip_focus: int = 0,
                            verbose: bool = False) -> Dict[str, np.ndarray]:
    """
    Despacho anual por HORIZONTE DESLIZANTE con flota diésel multi-grupo.

    Trocea el año en ventanas de `window` horas (p.ej. 168 = 1 semana),
    resuelve cada una con la flota completa y encadena el SOC y el estado
    ON/OFF de los grupos entre ventanas. Cada ventana es un MILP pequeño que
    resuelve rápido, evitando el coste prohibitivo del problema anual completo.

    window : horas por ventana (168 = semana). step ≤ window.
    cyclic_window : if True, each window must end with at least the SOC it
                    started with (used for independent representative weeks).
    """
    if cost is None:
        cost = MILPCostParameters()
    H = len(load)
    fleet = (diesel_fleet_kw if diesel_fleet_kw is not None
             else _default_diesel_fleet(params.diesel_capacity_kw))
    n_units = len(fleet)

    # Acumuladores de resultados
    keys_series = ["p_pv", "p_wind", "load", "p_re_used", "p_ch", "p_dis",
                   "p_batt", "p_dg", "u_dg", "y_dg", "p_unmet", "p_dump",
                   "fuel_l"]
    out = {k: np.zeros(H) for k in keys_series}
    out["soc"] = np.zeros(H)
    p_per_unit = np.zeros((n_units, H))

    soc_state = params.batt_soc_init
    u_state = np.zeros(n_units)   # todos los grupos OFF al inicio

    pos = 0
    n_windows = 0
    gaps, times, short = [], [], []
    while pos < H:
        w = min(window, H - pos)
        sl = slice(pos, pos + w)
        # Resolver la ventana (sin SOC cíclico; encadenamos manualmente)
        res = solve_annual_milp(
            params, p_pv[sl], p_wind[sl], load[sl], cost=cost,
            cyclic_soc=False, ramp_rate=ramp_rate, min_up_time=min_up_time,
            min_down_time=min_down_time, diesel_fleet_kw=fleet,
            soc_start=soc_state, u_start=u_state,
            soc_end=(soc_state if cyclic_window else None),
            mip_gap=mip_gap, time_limit=time_limit_block, threads=threads,
            updown_form=updown_form, mip_focus=mip_focus,
            verbose=verbose)
        gaps.append(res["mip_gap"]); times.append(res["solve_time"])
        short.append(res.get("soc_shortfall_kwh", 0.0))

        # Comprometer 'step' horas (o toda la ventana si step>=w)
        commit = min(step, w)
        for k in keys_series:
            out[k][pos:pos + commit] = res[k][:commit]
        out["soc"][pos:pos + commit] = res["soc"][:commit]
        p_per_unit[:, pos:pos + commit] = res["p_per_unit"][:, :commit]

        # Encadenar estado para la siguiente ventana
        soc_state = res["soc"][commit] if commit < len(res["soc"]) else res["soc"][-1]
        # Estado ON/OFF de cada grupo al final del tramo comprometido
        u_state = (res["p_per_unit"][:, commit - 1] > 1e-3).astype(float)

        pos += commit
        n_windows += 1

    if verbose:
        print(f"[rolling] {n_windows} ventanas de {window}h resueltas")

    out["p_per_unit"] = p_per_unit
    out["fleet_kw"] = np.array(fleet)
    out["starts"] = out["y_dg"]
    out["mip_gap_max"] = float(np.max(gaps)) if gaps else 0.0
    out["mip_gaps"] = list(gaps)
    out["solve_time_total"] = float(np.sum(times)) if times else 0.0
    out["soc_shortfall_kwh"] = float(np.sum(short)) if short else 0.0
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  WRAPPER UNIFICADO
# ─────────────────────────────────────────────────────────────────────────────

def dispatch_from_dataset(params, df, cost=None,
                          method: str = "annual",
                          **kwargs) -> Dict[str, np.ndarray]:
    """
    Punto de entrada único para el despacho con datos reales.

    method : 'annual'   → MILP monolítico (un solo grupo grande; rápido pero
                          el diésel sale poco realista)
             'rolling'  → horizonte deslizante con flota multi-grupo
                          (recomendado: realista y tratable)

    Devuelve el dict estándar de resultados de despacho.
    """
    p_pv, p_wind = power_from_cf(df["pv_cf"].values, df["wind_cf"].values,
                                 params.pv_capacity_kw, params.wind_capacity_kw)
    load = df["demand_kw"].values

    if method == "annual":
        return solve_annual_milp(params, p_pv, p_wind, load, cost=cost, **kwargs)
    elif method == "rolling":
        return solve_rolling_multiunit(params, p_pv, p_wind, load, cost=cost,
                                       **kwargs)
    elif method == "rolling_singleunit":
        from milp_dispatch import dispatch_milp_rolling
        return dispatch_milp_rolling(params, p_pv, p_wind, load, cost=cost, **kwargs)
    else:
        raise ValueError("method debe ser 'annual', 'rolling' o 'rolling_singleunit'")
