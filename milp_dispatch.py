"""
=============================================================================
HRES Dispatch — Optimización MILP con Gurobi (Rolling Horizon)
=============================================================================
Despacho óptimo de un sistema híbrido renovable PV + Eólica + BESS + Diésel.

Formulación: Mixed-Integer Linear Programming (MILP)
  - Variables binarias: estado ON/OFF del generador diésel, arranques
  - Variables continuas: potencias, SOC, vertido, déficit
  - Objetivo: minimizar coste operativo (combustible + arranques + déficit +
              degradación de batería)
  - Restricciones: balance de potencia, dinámica del SOC, límites de equipos,
              rampa del diésel, carga mínima

Estrategia rolling horizon: el año (8760 h) se resuelve en bloques solapados
de ventana W con paso de avance S, transfiriendo el SOC entre bloques. Esto:
  (1) es metodológicamente estándar en despacho energético (reduce miopía vs.
      despacho hora-a-hora y evita el coste de un MILP anual monolítico);
  (2) mantiene cada subproblema dentro del límite de la licencia académica.

Referencias:
  - Morales-España et al. (2013) IEEE Trans. Power Syst. (tight UC formulation)
  - Silvente et al. (2015) Applied Energy (rolling horizon microgrids)
=============================================================================
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from dataclasses import dataclass
from typing import Dict, Optional


# ─────────────────────────────────────────────────────────────────────────────
#  PARÁMETROS (compatibles con hres_model.HRESParameters)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MILPCostParameters:
    """Coeficientes económicos para la función objetivo del despacho."""
    fuel_price:       float = 1.20    # €/L
    diesel_a:         float = 0.08415 # L/kWh, no-load coefficient (x P_rated x u)  [Dufo-López 2008]
    diesel_b:         float = 0.246   # L/kWh, marginal coefficient (x p)           [Dufo-López 2008]
    startup_cost:     float = 15.0    # €/arranque
    startup_fuel:     float = 0.5     # L/arranque
    unmet_penalty:    float = 5.0     # €/kWh (VOLL — value of lost load)
    dump_penalty:     float = 0.0     # €/kWh (penalización vertido, opcional)
    batt_deg_cost:    float = 0.03    # €/kWh throughput (degradación BESS)
    co2_per_l:        float = 2.65    # kg CO₂/L diésel


# ─────────────────────────────────────────────────────────────────────────────
#  DESPACHO MILP DE UN BLOQUE (VENTANA)
# ─────────────────────────────────────────────────────────────────────────────

def solve_block_milp(params,
                     cost: MILPCostParameters,
                     p_pv: np.ndarray,
                     p_wind: np.ndarray,
                     load: np.ndarray,
                     soc_init: float,
                     diesel_on_init: int = 0,
                     soc_final_target: Optional[float] = None,
                     mip_gap: float = 0.005,
                     time_limit: float = 30.0,
                     verbose: bool = False) -> Dict[str, np.ndarray]:
    """
    Resuelve el despacho óptimo de un bloque temporal mediante MILP.

    Parámetros
    ----------
    params  : HRESParameters (capacidades, eficiencias, límites)
    cost    : MILPCostParameters (coeficientes económicos)
    p_pv    : array potencia PV disponible [kW] (longitud H)
    p_wind  : array potencia eólica disponible [kW] (longitud H)
    load    : array demanda [kW] (longitud H)
    soc_init: SOC al inicio del bloque [-]
    soc_final_target : si se especifica, fuerza SOC final (continuidad cíclica)

    Devuelve
    --------
    dict con las series temporales óptimas del bloque.
    """
    H = len(load)
    dt = 1.0  # h

    # ── Parámetros derivados ──────────────────────────────────────────────
    E_batt   = params.batt_capacity_kwh
    P_batt   = params.batt_power_kw
    eta_c    = params.batt_charge_eff
    eta_d    = params.batt_discharge_eff
    soc_min  = params.batt_soc_min
    soc_max  = params.batt_soc_max
    P_dg_max = params.diesel_capacity_kw
    P_dg_min = params.diesel_min_load * params.diesel_capacity_kw
    eta_conv = params.converter_eff

    m = gp.Model("HRES_block")
    m.setParam("OutputFlag", 1 if verbose else 0)
    m.setParam("MIPGap", mip_gap)
    m.setParam("TimeLimit", time_limit)

    # ── Variables de decisión ─────────────────────────────────────────────
    # Potencia renovable usada (puede recortarse → curtailment)
    p_re_used = m.addVars(H, lb=0, name="p_re_used")
    # Batería: carga (≥0) y descarga (≥0) separadas
    p_ch  = m.addVars(H, lb=0, ub=P_batt, name="p_ch")
    p_dis = m.addVars(H, lb=0, ub=P_batt, name="p_dis")
    # Estado de carga
    soc   = m.addVars(H + 1, lb=soc_min, ub=soc_max, name="soc")
    # Diésel
    p_dg  = m.addVars(H, lb=0, ub=P_dg_max, name="p_dg")
    u_dg  = m.addVars(H, vtype=GRB.BINARY, name="u_dg")        # ON/OFF
    y_dg  = m.addVars(H, vtype=GRB.BINARY, name="y_start")     # arranque
    # Holguras: déficit y vertido
    p_unmet = m.addVars(H, lb=0, name="p_unmet")
    p_dump  = m.addVars(H, lb=0, name="p_dump")

    # Potencia renovable total disponible por hora
    p_re_avail = p_pv + p_wind

    # ── Restricciones ──────────────────────────────────────────────────────
    # (1) SOC inicial
    m.addConstr(soc[0] == soc_init, name="soc_init")

    for t in range(H):
        # (2) Recurso renovable usado ≤ disponible
        m.addConstr(p_re_used[t] <= p_re_avail[t], name=f"re_avail_{t}")
        # Vertido = disponible - usado
        m.addConstr(p_dump[t] == p_re_avail[t] - p_re_used[t], name=f"dump_{t}")

        # (3) Balance de potencia (lado AC, tras convertidor)
        #     generación que sirve carga = demanda - déficit
        m.addConstr(
            eta_conv * (p_re_used[t] + p_dis[t] + p_dg[t]) - p_ch[t]
            == load[t] - p_unmet[t],
            name=f"balance_{t}")

        # (4) Dinámica del SOC
        m.addConstr(
            soc[t + 1] * E_batt ==
            soc[t] * E_batt * (1 - params.batt_self_discharge)
            + eta_c * p_ch[t] * dt
            - (p_dis[t] / eta_d) * dt,
            name=f"soc_dyn_{t}")

        # (5) No carga y descarga simultáneas (linealizado por costes;
        #     reforzado con binaria opcional si se desea estrictamente)
        #     Aquí se evita con penalización de throughput en el objetivo.

        # (6) Límites del diésel (carga mínima cuando ON)
        m.addConstr(p_dg[t] >= P_dg_min * u_dg[t], name=f"dg_min_{t}")
        m.addConstr(p_dg[t] <= P_dg_max * u_dg[t], name=f"dg_max_{t}")

        # (7) Detección de arranque: y_dg[t] >= u_dg[t] - u_dg[t-1]
        if t == 0:
            m.addConstr(y_dg[t] >= u_dg[t] - diesel_on_init, name=f"start_{t}")
        else:
            m.addConstr(y_dg[t] >= u_dg[t] - u_dg[t - 1], name=f"start_{t}")

    # (8) Restricción de SOC final (continuidad / no vaciado oportunista)
    if soc_final_target is not None:
        m.addConstr(soc[H] >= soc_final_target, name="soc_final")

    # ── Función objetivo: coste operativo total ───────────────────────────
    fuel_cost = gp.quicksum(
        cost.fuel_price * (cost.diesel_a * P_dg_max * u_dg[t]
                           + cost.diesel_b * p_dg[t]) * dt
        for t in range(H))
    startup_cost = gp.quicksum(
        (cost.startup_cost + cost.fuel_price * cost.startup_fuel) * y_dg[t]
        for t in range(H))
    unmet_cost = gp.quicksum(cost.unmet_penalty * p_unmet[t] * dt
                             for t in range(H))
    dump_cost = gp.quicksum(cost.dump_penalty * p_dump[t] * dt
                            for t in range(H))
    deg_cost = gp.quicksum(cost.batt_deg_cost * (p_ch[t] + p_dis[t]) * dt
                           for t in range(H))

    m.setObjective(fuel_cost + startup_cost + unmet_cost + dump_cost + deg_cost,
                   GRB.MINIMIZE)

    # ── Resolver ──────────────────────────────────────────────────────────
    m.optimize()

    if m.status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        raise RuntimeError(f"MILP no resuelto. Status Gurobi: {m.status}")

    # ── Extraer solución ──────────────────────────────────────────────────
    def arr(v): return np.array([v[t].X for t in range(H)])

    p_dg_sol   = arr(p_dg)
    u_dg_sol   = arr(u_dg)
    y_dg_sol   = arr(y_dg)
    fuel_l_sol = (cost.diesel_a * P_dg_max * u_dg_sol
                  + cost.diesel_b * p_dg_sol) * dt + cost.startup_fuel * y_dg_sol

    return {
        "p_re_used": arr(p_re_used),
        "p_ch":      arr(p_ch),
        "p_dis":     arr(p_dis),
        "p_batt":    arr(p_dis) - arr(p_ch),   # >0 descarga, <0 carga
        "p_dg":      p_dg_sol,
        "u_dg":      u_dg_sol,
        "y_dg":      y_dg_sol,
        "p_unmet":   arr(p_unmet),
        "p_dump":    arr(p_dump),
        "fuel_l":    fuel_l_sol,
        "soc":       np.array([soc[t].X for t in range(H)]),
        "soc_end":   soc[H].X,
        "diesel_on_end": int(round(u_dg_sol[-1])),
        "obj":       m.objVal,
        "mip_gap":   m.MIPGap if m.status == GRB.OPTIMAL else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  DESPACHO ANUAL POR HORIZONTE DESLIZANTE (ROLLING HORIZON)
# ─────────────────────────────────────────────────────────────────────────────

def dispatch_milp_rolling(params,
                          p_pv: np.ndarray,
                          p_wind: np.ndarray,
                          load: np.ndarray,
                          cost: Optional[MILPCostParameters] = None,
                          window: int = 168,
                          step: int = 120,
                          mip_gap: float = 0.01,
                          time_limit_block: float = 15.0,
                          verbose: bool = False) -> Dict[str, np.ndarray]:
    """
    Despacho anual mediante optimización MILP con horizonte deslizante.

    Parámetros
    ----------
    window : longitud de la ventana de optimización [h] (≤ límite licencia)
    step   : horas que se "comprometen" antes de avanzar (step ≤ window).
             El solapamiento (window-step) evita efectos de borde / miopía.

    Devuelve
    --------
    dict con las series anuales óptimas (misma interfaz que el rule-based).
    """
    if cost is None:
        cost = MILPCostParameters()

    n = len(load)
    # Acumuladores anuales
    out = {k: np.zeros(n) for k in
           ["p_re_used", "p_ch", "p_dis", "p_batt", "p_dg",
            "u_dg", "y_dg", "p_unmet", "p_dump", "fuel_l", "soc"]}
    out["p_pv"]   = p_pv.copy()
    out["p_wind"] = p_wind.copy()
    out["load"]   = load.copy()

    soc_cur = params.batt_soc_init
    dg_on   = 0
    total_obj = 0.0
    n_blocks = 0

    pos = 0
    while pos < n:
        w = min(window, n - pos)
        s = min(step, w)

        block = solve_block_milp(
            params, cost,
            p_pv[pos:pos + w], p_wind[pos:pos + w], load[pos:pos + w],
            soc_init=soc_cur, diesel_on_init=dg_on,
            soc_final_target=None,
            mip_gap=mip_gap, time_limit=time_limit_block, verbose=verbose)

        # Comprometer solo las primeras 's' horas del bloque
        for k in ["p_re_used", "p_ch", "p_dis", "p_batt", "p_dg",
                  "u_dg", "y_dg", "p_unmet", "p_dump", "fuel_l", "soc"]:
            out[k][pos:pos + s] = block[k][:s]

        # Estado para el siguiente bloque = SOC tras las 's' horas comprometidas
        soc_cur = block["soc"][s] if s < len(block["soc"]) else block["soc_end"]
        dg_on   = int(round(block["u_dg"][s - 1]))
        total_obj += block["obj"] * (s / w)  # prorrateo aproximado
        n_blocks += 1
        pos += s

        if verbose:
            print(f"  Bloque {n_blocks}: h[{pos-s}:{pos}]  "
                  f"SOC={soc_cur:.3f}  obj_block={block['obj']:.1f}")

    out["n_blocks"] = n_blocks
    out["total_obj"] = total_obj
    out["starts"] = out["y_dg"]  # alias para compatibilidad
    return out
