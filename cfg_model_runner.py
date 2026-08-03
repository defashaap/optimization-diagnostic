import pyomo.environ as pyo

import warnings
from time import perf_counter

from cfg_model_builder import OBJECTIVE_SPECS, build_model, default_spill_control

# Supported solver names
SUPPORTED_SOLVERS = {"gurobi", "highs", "appsi_highs", "cplex"}
POWER_TOL = 1e-4
P1_LOCK_TOL = 1e-5


def normalize_solver_name(solver_name):
    name = str(solver_name or "gurobi").strip().lower()
    if name in {"appsi_highs", "appsi-highs"}:
        name = "highs"
    if name not in SUPPORTED_SOLVERS:
        supported = "gurobi, highs, appsi_highs, cplex"
        raise ValueError(f"Unsupported solver '{solver_name}'. Choose one of: {supported}.")
    return name


def apply_highs_options(solver):
    solver.options["mip_rel_gap"] = 0
    solver.options["mip_abs_gap"] = 0
    solver.options["random_seed"] = 1
    solver.options["presolve"] = "off"
    solver.options["threads"] = 1
    return solver


def create_solver(solver_name="gurobi"):
    solver_name = normalize_solver_name(solver_name)

    if solver_name == "gurobi":
        solver = pyo.SolverFactory("gurobi")
        if not solver.available(exception_flag=False):
            raise RuntimeError("Gurobi solver is not available.")
        solver.options["MIPGap"] = 0.00
        solver.options["MIPGapAbs"] = 0.00
        solver.options["Seed"] = 1
        solver.options["Threads"] = 1
        return solver

    if solver_name == "highs":
        solver = pyo.SolverFactory("appsi_highs")
        if not solver.available(exception_flag=False):
            raise RuntimeError("HiGHS solver is not available.")
        return apply_highs_options(solver)

    solver = pyo.SolverFactory("cplex_direct")
    if not solver.available(exception_flag=False):
        raise RuntimeError("CPLEX solver is not available.")
    solver.options["mip_tolerances_mipgap"] = 0.0
    solver.options["mip_tolerances_absmipgap"] = 0.0
    solver.options["randomseed"] = 1
    solver.options["threads"] = 1
    return solver


def result_attr(result, section_name, attr_name):
    section = getattr(result, section_name, None)
    if section is None:
        return None
    value = getattr(section, attr_name, None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def solver_diagnostics(result):
    return {
        "objective": result_attr(result, "problem", "objective"),
        "lower_bound": result_attr(result, "problem", "lower_bound"),
        "upper_bound": result_attr(result, "problem", "upper_bound"),
        "gap": result_attr(result, "problem", "gap"),
    }


def deactivate_all_lexicographic_objectives(model):
    for component_name in getattr(model, "lexicographic_objective_components", []):
        getattr(model, component_name).deactivate()


def objective_is_minimize(objective_name):
    return OBJECTIVE_SPECS[objective_name]["sense"] == "minimize"


def objective_value_from_model(model, objective_name):
    expression_attr = OBJECTIVE_SPECS[objective_name]["expression_attr"]
    return pyo.value(getattr(model, expression_attr))


def prepare_stage_model(parsed_config, user_constraints, spill_control, locked_objective_values, stage_index):
    model = build_model(
        parsed_config,
        user_constraints=user_constraints,
        spill_control=spill_control,
    )
    deactivate_all_lexicographic_objectives(model)

    model.lexicographic_stage_locks = pyo.ConstraintList()
    for objective_name in model.lexicographic_objective_names[:stage_index - 1]:
        best_value = locked_objective_values[objective_name]
        expression = getattr(model, OBJECTIVE_SPECS[objective_name]["expression_attr"])
        if objective_is_minimize(objective_name):
            model.lexicographic_stage_locks.add(expression <= best_value + P1_LOCK_TOL)
        else:
            model.lexicographic_stage_locks.add(expression >= best_value - P1_LOCK_TOL)

    target_component = getattr(model, model.lexicographic_objective_components[stage_index - 1])
    target_component.activate()
    model.lexicographic_locked_values = dict(locked_objective_values)
    model.lexicographic_lock_tolerance = P1_LOCK_TOL
    return model


def solve_lexicographic(
    parsed_config,
    user_constraints=None,
    spill_control=None,
    solver_name="gurobi",
):
    solver_name = normalize_solver_name(solver_name)
    sc = spill_control if spill_control is not None else default_spill_control()
    objective_names = list(parsed_config["lexicographic_objectives"])
    locked_objective_values = {}
    stage_results = []
    final_model = None

    for stage_index, objective_name in enumerate(objective_names, start=1):
        model = prepare_stage_model(
            parsed_config,
            user_constraints,
            sc,
            locked_objective_values,
            stage_index,
        )
        solver = create_solver(solver_name)
        result = solver.solve(model, tee=False, load_solutions=False)
        if result.solver.termination_condition != pyo.TerminationCondition.optimal:
            raise RuntimeError(
                f"No optimal solution found in priority {stage_index} ({objective_name}). "
                f"Status={result.solver.status}, "
                f"Term={result.solver.termination_condition}"
            )

        model.solutions.load_from(result)
        locked_objective_values[objective_name] = objective_value_from_model(model, objective_name)
        stage_results.append(
            {
                "priority": stage_index,
                "objective": objective_name,
                "status": str(result.solver.status),
                "termination": str(result.solver.termination_condition),
                "value": locked_objective_values[objective_name],
                "solver_diagnostics": solver_diagnostics(result),
            }
        )
        final_model = model

    final_model.lexicographic_best_values = dict(locked_objective_values)
    return final_model, stage_results


# Collect results
def collect_results(model, stage_results):
    pc = model.parsed_config
    N_p = pc["N_p"]
    comp_ids = list(pc["comp_ids"])
    turbine_ids = list(pc["turbines"])
    pump_ids = list(pc["pumps"])
    active_ids = list(pc["active_comps"])
    res_ids = sorted(pc["reservoirs"].keys())

    hourly = []
    for t in model.T: 
        row = {"hour": int(t)}

        #flow per component
        for cid in comp_ids:
            row[f"Q_{cid}"] = pyo.value(model.Q[cid, t])
            row[f"d_{cid}"] = int(round(pyo.value(model.d[cid, t])))
        
        #power per component
        for cid in active_ids:
            row[f"P_{cid}"] = pyo.value(model.P[cid, t])

        #head differences
        for cid in active_ids:
            row[f"dH_{cid}"] = pyo.value(model.dH[cid, t])
        
        #reservoir levels
        for rid in res_ids:
            row[f"h_{rid}"] = pyo.value(model.H[rid, t])  
        
        #net power
        gen = sum(pyo.value(model.P[tid, t]) for tid in turbine_ids)
        pump = sum(pyo.value(model.P[pid, t]) for pid in pump_ids)
        row["net_power"] = gen - pump

        #unit mode
        row["mode_tuple"] = tuple(row[f"d_{cid}"] for cid in comp_ids)

        hourly.append(row)

    #power totals per component
    power_totals = {}
    for cid in active_ids:
        power_totals[cid] = sum(row[f"P_{cid}"] for row in hourly)
    total_gen = sum(power_totals[tid] for tid in turbine_ids)
    total_pump = sum(power_totals[pid] for pid in pump_ids)

    #final reservoir levels
    final_h = {rid: pyo.value(model.H[rid, N_p]) for rid in res_ids}
    delta_h = {rid: pyo.value(model.H[rid, N_p]) - pyo.value(model.H[rid, 0]) for rid in res_ids}
    objective_values = {
        objective_name: objective_value_from_model(model, objective_name)
        for objective_name in model.lexicographic_objective_names
    }

    return {
        "p1_status": stage_results[0]["status"] if len(stage_results) >= 1 else None,
        "p1_termination": stage_results[0]["termination"] if len(stage_results) >= 1 else None,
        "p2_status": stage_results[1]["status"] if len(stage_results) >= 2 else None,
        "p2_termination": stage_results[1]["termination"] if len(stage_results) >= 2 else None,
        "p3_status": stage_results[2]["status"] if len(stage_results) >= 3 else None,
        "p3_termination": stage_results[2]["termination"] if len(stage_results) >= 3 else None,
        "p1_solver_diagnostics": stage_results[0]["solver_diagnostics"] if len(stage_results) >= 1 else None,
        "p2_solver_diagnostics": stage_results[1]["solver_diagnostics"] if len(stage_results) >= 2 else None,
        "p3_solver_diagnostics": stage_results[2]["solver_diagnostics"] if len(stage_results) >= 3 else None,
        "p1_best_safety_violation": objective_values.get("min_total_safety_violation"),
        "p1_lock_tolerance": getattr(model, "lexicographic_lock_tolerance", None),
        "total_safety_violation": pyo.value(model.total_safety_violation),
        "total_generation_power": pyo.value(model.total_generation_power),
        "total_revenue_eur": pyo.value(model.total_revenue_eur),
        "total_gen": total_gen,
        "total_pump": total_pump,
        "power_totals": power_totals,
        "final_h": final_h,
        "delta_h": delta_h,
        "hourly": hourly,
        "comp_ids": comp_ids,
        "turbine_ids": turbine_ids,
        "pump_ids": pump_ids,
        "active_ids": active_ids,
        "res_ids": res_ids,
        "solver_stages": stage_results,
        "lexicographic_objectives": list(model.lexicographic_objective_names),
        "objective_values": objective_values,
    }

# Baseline and User's alternative scenario runner
def run_two_scenarios(parsed_config, user_constraints, spill_control=None, solver_name="gurobi"):
    timings = {}
    spill_control = default_spill_control() if spill_control is None else spill_control
    solver_name = normalize_solver_name(solver_name)

    # Baseline
    t0 = perf_counter()
    try:
        baseline_model, baseline_stage_results = solve_lexicographic(
            parsed_config,
            user_constraints=None,
            spill_control=spill_control,
            solver_name=solver_name,
        )
        baseline_result = collect_results(baseline_model, baseline_stage_results)
        baseline_out = {
            "is_feasible": True,
            "termination": baseline_stage_results[-1]["termination"],
        }
    except RuntimeError as e:
        baseline_model = None
        baseline_result = None
        baseline_out = {
            "is_feasible": False,
            "termination": str(e),
        }
    finally:
        timings["1) Baseline optimization"] = perf_counter() - t0

    # User's alternative
    alt_out = {
        "is_feasible": False,
        "termination": None,
        "model": None,
        "result": None,
    }

    if not baseline_out["is_feasible"]:
        alt_out["termination"] = "not_run: baseline optimization is infeasible"
        return {
            "solver_name": solver_name,
            "baseline": baseline_out,
            "baseline_model": baseline_model,
            "baseline_result": baseline_result,
            "alternative": alt_out,
            "timings": timings,
        }

    t0 = perf_counter()
    try:
        alt_model, alt_stage_results = solve_lexicographic(
            parsed_config,
            user_constraints=user_constraints,
            spill_control=spill_control,
            solver_name=solver_name,
        )
        alt_result = collect_results(alt_model, alt_stage_results)
        alt_out.update({
            "is_feasible": True,
            "termination": alt_stage_results[-1]["termination"],
            "model": alt_model,
            "result": alt_result,
        })
    except RuntimeError as e:
        alt_out.update({
            "is_feasible": False,
            "termination": str(e),
            "model": None,
            "result": None,
        })
    finally: 
        timings["2) Alternative optimization"] = perf_counter() - t0

    # A feasible alternative is also feasible for the unconstrained baseline.
    # If this warning fires, the selected solver did not reproduce the Phase 2
    # dominance relationship and the result should not be trusted.
    if alt_out["is_feasible"] and alt_out["result"] is not None:
        objectives = parsed_config["lexicographic_objectives"]
        base_values = baseline_result["objective_values"]
        alt_values = alt_out["result"]["objective_values"]
        alt_dominates = True
        alt_strictly_better = False

        for objective_name in objectives:
            base_value = base_values[objective_name]
            alt_value = alt_values[objective_name]
            if objective_is_minimize(objective_name):
                if alt_value > base_value + POWER_TOL:
                    alt_dominates = False
                    break
                if alt_value < base_value - POWER_TOL:
                    alt_strictly_better = True
            else:
                if alt_value < base_value - POWER_TOL:
                    alt_dominates = False
                    break
                if alt_value > base_value + POWER_TOL:
                    alt_strictly_better = True

        if alt_dominates and alt_strictly_better:
            warnings.warn(
                "Sanity check FAILED: the constrained alternative lexicographically "
                "improves on the unconstrained baseline. This indicates the solver "
                f"path for {solver_name} did not reproduce the true baseline optimum.",
                RuntimeWarning,
                stacklevel=2,
            )

    return {
        "solver_name": solver_name,
        "baseline": baseline_out,
        "baseline_model": baseline_model,
        "baseline_result": baseline_result,
        "alternative": alt_out,
        "timings": timings,
    }
