"""
Parallel, resumable evaluation of a GSA design + reproducibility manifest.

Each worker builds its own HRESEngine once (data read from data_cache/) and
uses one Gurobi thread. Results are flushed to disk every `flush` points, so
an interrupted run continues where it stopped when relaunched with the same
output prefix. Works on Windows (spawn) when called under
`if __name__ == "__main__":`.
"""

import json, os, platform, sys, time, hashlib
from pathlib import Path
from multiprocessing import get_context

import numpy as np
import pandas as pd

import config_paper as C

_ENGINE = None
_NAMES = None


def _init_worker(mode, fleet, week_hours, names):
    global _ENGINE, _NAMES
    import warnings
    warnings.filterwarnings("ignore")
    from gsa_engine import load_data, HRESEngine
    _ENGINE = HRESEngine(load_data(), mode=mode, fleet=fleet, week_hours=week_hours)
    _NAMES = names


def _eval_one(args):
    i, x = args
    try:
        k = _ENGINE.evaluate(x, _NAMES)
        row = {kk: k[kk] for kk in list(C.KPIS) + ["Equiv_cycles", "E_Diesel_GWh",
                                                    "mip_gap_max", "solve_time_s",
                                                    "SOC_shortfall_MWh"]}
        row["ok"] = 1
    except Exception as e:                       # keep the row, flag it
        row = {kk: np.nan for kk in list(C.KPIS)}
        row.update(ok=0, error=str(e)[:200])
    row["i"] = i
    return row


def evaluate_design(X, names, out_prefix, mode="weeks", fleet=C.GSA_ENGINE["fleet"],
                    workers=1, flush=20, week_hours=168):
    """Evaluates every row of X; returns a DataFrame indexed by row number."""
    out_csv = Path(f"{out_prefix}_Y.csv")
    done = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()
    if len(done):                       # failed points are always re-evaluated
        n_failed = int((done["ok"] == 0).sum())
        if n_failed:
            print(f"[runner] retrying {n_failed} previously failed evaluations")
        done = done[done["ok"] == 1].drop(columns=["error"], errors="ignore")
    done_idx = set(done["i"]) if len(done) else set()
    todo = [(i, X[i]) for i in range(len(X)) if i not in done_idx]
    print(f"[runner] {len(X)} points, {len(done_idx)} already done, "
          f"{len(todo)} to go, {workers} worker(s)", flush=True)
    rows, t0 = [], time.time()

    def _flush():
        nonlocal done, rows
        if rows:
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            done.sort_values("i").to_csv(out_csv, index=False)
            rows = []

    if todo:
        if workers > 1:
            ctx = get_context("spawn")
            with ctx.Pool(workers, initializer=_init_worker,
                          initargs=(mode, fleet, week_hours, names)) as pool:
                for n, row in enumerate(pool.imap_unordered(_eval_one, todo), 1):
                    rows.append(row)
                    if n % flush == 0:
                        _flush()
                        el = time.time() - t0
                        print(f"  {n}/{len(todo)}  {el/60:.1f} min  "
                              f"(~{el/n*(len(todo)-n)/3600:.1f} h left)", flush=True)
        else:
            _init_worker(mode, fleet, week_hours, names)
            for n, item in enumerate(todo, 1):
                rows.append(_eval_one(item))
                if n % flush == 0:
                    _flush()
                    el = time.time() - t0
                    print(f"  {n}/{len(todo)}  {el/60:.1f} min  "
                          f"(~{el/n*(len(todo)-n)/3600:.1f} h left)", flush=True)
        _flush()
    Y = done.sort_values("i").set_index("i")
    bad = int((Y["ok"] == 0).sum())
    if bad:
        print(f"[runner] WARNING {bad} failed evaluations (see 'error' column)")
    return Y


def write_manifest(path, extra=None):
    """Software versions, solver settings, seeds and data fingerprints."""
    def ver(mod):
        from importlib.metadata import version
        try:
            return version(mod)
        except Exception:
            return "n/a"
    try:
        import gurobipy as gp
        grb = ".".join(map(str, gp.gurobi.version()))
    except Exception:
        grb = "n/a"
    cache = {}
    for f in sorted(Path("data_cache").glob("*.parquet")):
        cache[f.name] = hashlib.md5(f.read_bytes()).hexdigest()[:12]
    man = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "python": sys.version.split()[0], "os": platform.platform(),
        "packages": {m: ver(m) for m in ["numpy", "pandas", "pvlib",
                                         "windpowerlib", "SALib", "matplotlib",
                                         "scipy", "gurobipy"]},
        "gurobi": grb, "solver_settings": C.SOLVER, "uc_constraints": C.UC_CONSTRAINTS,
        "milp_costs": C.MILP_COSTS, "gsa_engine": C.GSA_ENGINE,
        "morris": C.MORRIS, "sobol": C.SOBOL, "year": C.YEAR,
        "location": C.LOCATION, "data_sources": C.DATA_SOURCES,
        "fixed": {k: v for k, v in C.FIXED.items()}, "economics": C.ECONOMICS,
        "data_cache_md5": cache,
    }
    if extra:
        man.update(extra)
    Path(path).write_text(json.dumps(man, indent=2, default=str))
    return man
