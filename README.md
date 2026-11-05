# Global sensitivity analysis of an hourly unit commitment model for an island hybrid renewable energy system

Code and input data for the article

> L. Álvarez-Piñeiro, C. Berna-Escriche, D. Blanco-Muelas, P. Bastida-Molina.
> Global Sensitivity Analysis of an Hourly Unit Commitment Model for an Island Hybrid Renewable Energy System.
> Energy 

The model couples hourly photovoltaic and wind generation (pvlib, windpowerlib) with a
mixed-integer linear unit commitment of the ten diesel engines of the Los Guinchos power
station on La Palma (Canary Islands, Spain), battery storage, curtailment and unserved load.
A two-stage global sensitivity analysis (Morris screening followed by Sobol' indices)
quantifies how sixteen uncertain inputs drive five performance indicators: levelized cost of
electricity, loss of power supply probability, renewable fraction, curtailment and annual
carbon dioxide emissions.

## Repository structure

| File | Role |
|---|---|
| `config_paper.py` | Every assumption of the paper: parameter ranges (Table 3), economics (Table 4), unit commitment constraints, solver settings, seeds, real diesel fleet |
| `gsa_engine.py` | Evaluation engine: generation, dispatch and annualized KPIs, on four representative weeks or the full year |
| `dispatch_v2.py`, `milp_dispatch.py` | MILP unit commitment (Gurobi) and cost coefficients |
| `hres_model.py` | System parameters and KPI definitions |
| `generation_models.py`, `physical_evaluator.py`, `lapalma_data.py`, `data_sources.py` | Resource and demand data, PV and wind models |
| `gsa_runner.py` | Parallel, resumable evaluation of a sampling design and run manifest |
| `run_reference_validation.py` | Reference case 2023 and storage scenario (Section 4.1, Figures 3 and 4, Table S1) |
| `run_morris_revision.py` | Morris screening with three seeds (Section 4.2, Figure 5, Table S2) |
| `run_sobol_revision.py` | Sobol' indices with confidence intervals, second-order indices and convergence (Section 4.3, Figures 6 and 7, Tables S3 and S4, Figure S1) |
| `verify_representative_weeks.py` | Optional comparison of the representative-week engine with the full-year model |
| `bench_solver.py` | Solver benchmark used to choose the MILP settings |
| `data_cache/` | Processed hourly input data for 2023 (see `data_cache/README.md`) |

## Installation

```bash
pip install -r requirements.txt
```

The unit commitment is solved with **Gurobi**, which is not distributed with this repository.
Free academic licenses are available from Gurobi Optimization. The `pip` package `gurobipy`
includes a size-limited license that is enough for the `--test` runs below but not for the
full model. Results were obtained with the software versions recorded in the
`*_manifest.json` files written by each script.

No API tokens are needed to reproduce the paper: the input data are included in `data_cache/`.
Downloading other years or locations requires the environment variables `ESIOS_TOKEN`
(REE/ESIOS) and `NINJA_TOKEN` (Renewables.ninja).

## Reproducing the results

```bash
python config_paper.py                                   # writes Table 3
python run_reference_validation.py --time-limit 120      # Section 4.1
python run_morris_revision.py --workers 5 --robustness   # Section 4.2, 3 x 340 evaluations
python run_sobol_revision.py --workers 5 --N 256         # Section 4.3, 4608 evaluations
```

Every script writes to `outputs_revision/` and can be resumed after an interruption by
running the same command again. `--workers` sets the number of parallel processes, each
using one Gurobi thread. With five workers on a desktop computer the Morris screening takes
about 7 h and the Sobol' analysis about 33 h. `python run_sobol_revision.py --analyze-only`
recomputes the indices and figures from saved model outputs without new simulations.

A quick check of the whole pipeline, with 24 h representative periods and a reduced fleet:

```bash
python run_morris_revision.py --test
python run_sobol_revision.py --test
```

## Main settings
- La Palma Island (Spain), year 2023, hourly resolution, Fuencaliente (28.49° N, 17.85° W).
- Ten diesel engines of 82.84 MW in total; the 22.5 MW gas turbine is not represented.
- Sensitivity analysis with Morris and Sobol
- MILP relative gap 1%

## License

Code: MIT License (see `LICENSE`). Input data keep the licenses of their sources
(see `data_cache/README.md`).

## Citation

See `CITATION.cff`. Please cite the article when using this code.
