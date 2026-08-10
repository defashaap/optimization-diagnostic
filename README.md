# Optimization Diagnostic

This repository contains a configuration-driven pumped-storage hydropower optimal scheduling and diagnosis workflow built with Pyomo.  
It compares a baseline optimization against a user-defined alternative scenario, then explains feasible and infeasible outcomes.

## What The Code Does

The codebase solves pumped-storage hydropower operation problems with:

- reservoirs and tailwaters
- turbines, pumps, and spill units
- lexicographic objectives
- user-forced ON/OFF schedules
- diagnostic reporting for feasible and infeasible alternatives

Typical workflow:

1. Define a baseline system configuration JSON.
2. Define a user scenario JSON with forced unit schedules.
3. Select the solver and output folder in `configuration.json`.
4. Run the optimizer.
5. Read the generated diagnostic JSON report.

## Quick Start
To run the project on a new computer:

1. Create and activate a new virtual environment.
2. Use Python `3.13` if possible. Python `>=3.11` should also work.
3. Install the dependencies from `requirements.txt`.
4. Check that the `baseline` and `alternative` paths in `configuration.json` exactly match the filenames and directories in this repository.
5. Run `python cfg_main.py`.

Example setup:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python --version
pip install -r requirements.txt
python cfg_main.py
```

Edit 'configuration.json':

```json
{
  "baseline": "Configuration/baseline_config.json",
  "alternative": "Scenario/Module_Test/01_B.json",
  "solver": "gurobi",
  "output_folder": "gurobi_output"
}
```

Run:

```powershell
python cfg_main.py
```

The run will:

- load the baseline configuration
- load the user alternative scenario
- solve the baseline case
- solve the constrained alternative case
- generate a diagnostic report in the selected output folder

## Main Modules

### Core workflow

- `cfg_main.py`  
  Main entry point. Loads `configuration.json`, reads the baseline and alternative files, runs both scenarios, prints terminal output, and writes the final diagnostic JSON report.

- `cfg_model_builder.py`
  Builds the Pyomo model from JSON input. Parses reservoirs, topology, components, inflows, spill logic, user constraints, and lexicographic objectives.

- `cfg_model_runner.py`
  Solves the model. Handles solver selection, lexicographic stage-by-stage optimization, result extraction, and baseline-versus-alternative execution.

- `cfg_diagnostic_analysis.py`
  Generates analysis after optimization. For feasible cases, it compares mode changes, power deltas, and binding constraints. For infeasible cases, it computes IIS explanations and relaxation pathways.

- `cfg_diagnostic_report.py`
  Converts optimization and analysis output into a structured JSON report with explanations.

## Solvers

Supported solvers in the code:

- `gurobi`
- `cplex`
- `highs`

## Dependencies

Install the project dependencies with:

```powershell
pip install -r requirements.txt
```

