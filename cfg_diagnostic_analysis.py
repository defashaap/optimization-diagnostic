import pyomo.environ as pyo


from cfg_model_builder import(
    build_model, 
    collect_raw_forced_assignments,
    default_spill_control,
    normalize_user_constraints,
)
from cfg_model_runner import (
    create_solver,
    deactivate_all_lexicographic_objectives,
    normalize_solver_name,
    objective_value_from_model,
    prepare_stage_model,
)

def is_termination_feasible(term):
    return term in {
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.feasible,
    }

# Feasible Case Analysis

# Delta decomposition of two scenarios
def compute_hourly_deltas(base_result, alt_result, parsed_config):
    pc = parsed_config
    comp_ids = list(pc["comp_ids"])
    active_ids = sorted(pc["active_comps"])
    res_ids = sorted(pc["reservoirs"].keys())

    rows = []
    for t in range(len(base_result["hourly"])):
        b = base_result["hourly"][t]
        a = alt_result["hourly"][t]
        row = {"hour": t}

        #power deltas per active component
        for cid in active_ids:
            row[f"delta_P_{cid}"] = a.get(f"P_{cid}", 0) - b.get(f"P_{cid}", 0)

        #net power delta
        row["delta_net_power"] = a.get("net_power", 0) - b.get("net_power", 0)
        row["abs_delta_net_power"] = abs(row["delta_net_power"])

        #mode tuple delta
        row["base_mode"] = b["mode_tuple"]
        row["alt_mode"] = a["mode_tuple"]

        rows.append(row)

    rows.sort(key=lambda r: r["abs_delta_net_power"], reverse=True)
    return rows

# Build hour-indexed labels of user-defined component modes
def build_mode_labels(user_constraints, compo_ids):
    forced = {}
    normalized = normalize_user_constraints(user_constraints, compo_ids)
    for cid, hour_map in normalized.items():
        for hour, value in hour_map.items():
            forced.setdefault(hour, []).append(f"{cid}={value}")
    return forced

# Binding Constraints
def get_binding_constraints(model, parsed_config, tol=1e-5):
    results = []
    for con in model.component_objects(pyo.Constraint, active=True):
        for idx in con: 
            c = con[idx]
            if c.body is None: 
                continue
            body_val = pyo.value(c.body, exception=False)
            if body_val is None:
                continue
            lb = pyo.value(c.lower) if c.lower is not None else None
            ub = pyo.value(c.upper) if c.upper is not None else None

            #skip equality constraints since they are always binding
            if lb is not None and ub is not None and abs(lb - ub) <= tol:
                continue

            if lb is not None:
                slack = body_val - lb
                results.append({
                    "constraint": con.name, "index": idx, "side": "lower", "slack": slack, "binding": abs(slack) <= tol,
                })
            
            if ub is not None:
                slack = ub - body_val
                results.append({
                    "constraint": con.name, "index": idx, "side": "upper", "slack": slack, "binding": abs(slack) <= tol,
                })
    return results

# Extracts the hour from constraints
def constraint_hour(row):
    idx = row["index"]
    if isinstance(idx, tuple):
        idx = idx[-1]
    try:
        return int(idx)
    except (TypeError, ValueError):
        return None

# Extracts the component ID from constraints
def component_from_constraint_name(name, prefixes):
    for prefix in prefixes:
        if name.startswith(prefix):
            return name[len(prefix):]
    return None

# Reads binary mode variable values
def mode_value(model, cid, hour):
    if hour is None or not hasattr(model, "d"):
        return None
    try:
        return pyo.value(model.d[cid, hour], exception=False)
    except Exception:
        return None

# Detects if a binding constraint should be ignored because the component is OFF
def is_inactive_mode_artifact(row, model):
    name = row["constraint"]
    hour = constraint_hour(row)

    cid = component_from_constraint_name(
        name,
        ("q_upper_", "q_min_on_", "turbine_plane_", "pump_plane_"),
    )
    if cid is None:
        cid = component_from_constraint_name(name, ("flow_trigger_", "head_trigger_"))
    if cid is None:
        return False

    value = mode_value(model, cid, hour)
    return value is not None and value <= 0.5


# Build the set of constraints that are structural (linking constraints) to be excluded from the binding constraint analysis
def build_structural_set(parsed_config):
    pc = parsed_config
    structural = {"keep_p1"}
    for cid in pc["active_comps"]:
        structural.update({
            f"P_link_up_mode_{cid}", f"P_link_low_mode_{cid}", f"P_link_up_aux_{cid}", f"P_link_low_aux_{cid}",
        })
    return structural

# Compare binding constraints between two scenarios
def compare_binding_constraints(base_model, alt_model, parsed_config, tol=1e-5, exclude_structural=True):
    structural = build_structural_set(parsed_config) if exclude_structural else set()
    base_binding = get_binding_constraints(base_model, parsed_config, tol)
    alt_binding = get_binding_constraints(alt_model, parsed_config, tol)
    def make_key(row):
        return (row["constraint"], row["index"], row["side"])
    def include(row, model):
        if exclude_structural and row["constraint"] in structural:
            return False
        if is_inactive_mode_artifact(row, model):
            return False
        return True
    base_lookup = {
        make_key(row): row
        for row in base_binding
        if row["binding"] and include(row, base_model)
    }
    alt_lookup = {
        make_key(row): row
        for row in alt_binding
        if row["binding"] and include(row, alt_model)
    }
    base_set = set(base_lookup)
    alt_set = set(alt_lookup)
    only_base = sorted(base_set - alt_set)
    only_alt = sorted(alt_set - base_set)
    in_both = sorted(base_set & alt_set)
    return (
        only_base,
        only_alt,
        in_both,
        {key: base_lookup[key]["slack"] for key in only_base},
        {key: alt_lookup[key]["slack"] for key in only_alt},
        {
            key: {
                "baseline": base_lookup[key]["slack"],
                "alternative": alt_lookup[key]["slack"],
            }
            for key in in_both
        },
        tol,
    )

def feasible_analysis(base_model, base_result, alt_model, alt_result, user_constraints, parsed_config, top_k=8):
    ranked = compute_hourly_deltas(base_result, alt_result, parsed_config)
    top_rows = ranked[:min(top_k, len(ranked))]
    (
        only_base,
        only_alt,
        in_both,
        only_base_slacks,
        only_alt_slacks,
        in_both_slacks,
        binding_tolerance,
    ) = compare_binding_constraints(base_model, alt_model, parsed_config)
    comp_ids = sorted(parsed_config["comp_ids"])
    return {
        "all_rows": ranked,
        "top_rows": top_rows,
        "only_base": only_base,
        "only_alt": only_alt,
        "in_both": in_both,
        "only_base_slacks": only_base_slacks,
        "only_alt_slacks": only_alt_slacks,
        "in_both_slacks": in_both_slacks,
        "binding_tolerance": binding_tolerance,
        "forced_mode": build_mode_labels(user_constraints, comp_ids),
    }
    
# Infeasible Case Analysis

# Create a merged list of forced constraints
def build_atoms(user_constraints, spill_control, parsed_config):
    comp_ids = sorted(parsed_config["comp_ids"])
    spill_ids = sorted(parsed_config["spills"])
    horizon = parsed_config["N_p"]
    raw = collect_raw_forced_assignments(user_constraints, comp_ids, spill_control, spill_ids, horizon)
    merged = {}
    for item in raw:
        key = (item["var"], int(item["hour"]), int(item["value"]))
        row = merged.setdefault(key, {"var": item["var"], "hour": int(item["hour"]), "value": int(item["value"]), "sources": set()})
        row["sources"].add(item["source"])
    atoms = []
    for atom_id, key in enumerate(sorted(merged.keys())):
        row = merged[key]
        atoms.append({"id": atom_id, "var": row["var"], "hour": row["hour"], "value": row["value"], "sources": sorted(row["sources"])})
    return atoms

# Apply forced constraints to the model 
def apply_forced_atoms(model, atoms, relaxed_ids=None):
    relaxed_ids = set(relaxed_ids or[])
    model.hard_forced = pyo.ConstraintList()
    for atom_id, atom in enumerate(atoms):
        if atom_id in relaxed_ids:
            continue
        model.hard_forced.add(model.d[atom["var"], atom["hour"]] == atom["value"])

# Check the feasibility of the user's forced scenario
def is_forced_set_feasible(atoms, parsed_config, solver_name="gurobi"):
    model = build_model(parsed_config, user_constraints=None, spill_control=default_spill_control())
    apply_forced_atoms(model, atoms)
    result = create_solver(solver_name).solve(model, tee=False, load_solutions=False)
    term = result.solver.termination_condition
    return is_termination_feasible(term), str(term)

# Format constraint names for reporting
def format_constraint_name(name, idx):
    if idx is None:
        return name
    if isinstance(idx, tuple):
        return f"{name}(" + ",".join(str(x) for x in idx) + ")"
    return f"{name}({idx})"

# Constraints' name handling
def extract_solver_to_pyomo_constraint_map(solver):
    solver_to_pyomo = getattr(solver, "_solver_con_to_pyomo_con_map", None)
    if solver_to_pyomo is not None:
        return dict(solver_to_pyomo)

    pyomo_to_solver = getattr(solver, "_pyomo_to_solver_con_map", None)
    #handle different naming across pyomo version
    if pyomo_to_solver is None:
        pyomo_to_solver = getattr(solver, "_pyomo_con_to_solver_con_map", None)
    if pyomo_to_solver is None:
        return None
    return {solver_con: pyomo_con for pyomo_con, solver_con in pyomo_to_solver.items()}
def normalize_solver_constraint_name(name):
    if not isinstance(name, str):
        return str(name)
    text = name.strip()
    if "[" in text and text.endswith("]"):
        base, idx = text.split("[", 1)
        idx = idx[:-1]
        return f"{base}({idx})"
    return text

# Extracts the forced-constraint ID
def parse_atom_id_from_constraint_name(name, component_name):
    normalized = normalize_solver_constraint_name(name)
    prefix = f"{component_name}("
    if not normalized.startswith(prefix) or not normalized.endswith(")"):
        return None
    idx_text = normalized[len(prefix):-1].split(",")[0].strip()
    try:
        return int(idx_text)
    except (TypeError, ValueError):
        return None

# Classifies IIS constraint (user/model)
def add_pyomo_conflict_constraint(pyomo_con, atoms, forced_atom_ids, non_forced_constraints):
    component = pyomo_con.parent_component()
    component_name = component.name
    idx = pyomo_con.index()

    if component_name == "hard_forced_iis":
        atom_id = idx[0] if isinstance(idx, tuple) else idx
        if atom_id is not None:
            forced_atom_ids.add(int(atom_id))
        return

    non_forced_constraints.add(format_constraint_name(component_name, idx))

# Handle potential differences in how solvers report constraint names 
def possible_solver_constraint_names(solver_con):
    names = []
    if isinstance(solver_con, str):
        names.append(solver_con)
    else:
        for attr in ("name", "ConstrName"):
            value = getattr(solver_con, attr, None)
            if value:
                names.append(value)
        names.append(str(solver_con))
    return {normalize_solver_constraint_name(name) for name in names if name}

# Take solver reported name then find the original Pyomo constraint
def extract_solver_name_to_pyomo_constraint_map(solver):
    solver_to_pyomo = extract_solver_to_pyomo_constraint_map(solver)
    if solver_to_pyomo is None:
        return {}

    out = {}
    for solver_con, pyomo_con in solver_to_pyomo.items():
        for name in possible_solver_constraint_names(solver_con):
            out[name] = pyomo_con
    return out 


def suppress_gurobi_terminal_output(gmodel):
    for param_name, value in (("OutputFlag", 0), ("LogToConsole", 0)):
        try:
            gmodel.setParam(param_name, value)
        except Exception:
            pass


# Build a model with hard atom constraints and find IIS using Gurobi IIS extraction
def find_irreducible_infeasible_set_gurobi(atoms, parsed_config):
    #create persistent Gurobi solver
    solver = pyo.SolverFactory("gurobi_persistent")
    if solver is None or not solver.available(exception_flag=False):
        return None, None, "gurobi_persistent solver not available"

    model = build_model(parsed_config, user_constraints=None, spill_control=default_spill_control())
    model.hard_forced_iis_idx = pyo.RangeSet(0, len(atoms) - 1)

    # add hard constraints for each atom to the model
    def hard_forced_iis_rule(m, i):
        atom = atoms[int(i)]
        return m.d[atom["var"], atom["hour"]] == atom["value"]

    model.hard_forced_iis = pyo.Constraint(model.hard_forced_iis_idx, rule=hard_forced_iis_rule)

    try:
        solver.set_instance(model)
    except Exception as exc:
        return None, None, f"cannot set gurobi_persistent instance ({exc})"

    #compute IIS using Gurobi
    gmodel = solver._solver_model
    suppress_gurobi_terminal_output(gmodel)
    try:
        gmodel.computeIIS()
    except Exception as exc:
        return None, None, f"Gurobi IIS failed ({exc})"

    #retrieve the mapping from Gurobi constraints back to Pyomo constraints
    solver_to_pyomo = extract_solver_to_pyomo_constraint_map(solver)
    if solver_to_pyomo is None:
        return None, None, "gurobi_persistent constraint map unavailable"

    forced_atom_ids = set()
    non_forced_constraints = set()

    for constr in gmodel.getConstrs():
        if constr.IISConstr != 1:
            continue
        pyomo_con = solver_to_pyomo.get(constr)
        if pyomo_con is None:
            continue
        add_pyomo_conflict_constraint(
            pyomo_con,
            atoms,
            forced_atom_ids,
            non_forced_constraints,
        )

    if not forced_atom_ids:
        return None, sorted(non_forced_constraints), "IIS computed but no hard_forced_iis rows identified"

    return (
        [atoms[i] for i in sorted(forced_atom_ids)],
        sorted(non_forced_constraints),
        "ok",
    )

# CPLEX Block
# List Pyomo constraints -> list CPLEX constraints in order -> match names to build mapping from CPLEX constraint index to Pyomo constraint
def add_pyomo_constraint_order_display_names(model, cpx, solver_name_to_pyomo):
    pyomo_constraints = list(model.component_data_objects(pyo.Constraint, active=True, descend_into=True))
    for idx, pyomo_con in enumerate(pyomo_constraints[:cpx.linear_constraints.get_num()]):
        try:
            cplex_name = normalize_solver_constraint_name(cpx.linear_constraints.get_names(idx))
        except Exception:
            continue
        solver_name_to_pyomo.setdefault(cplex_name, pyomo_con)
    return solver_name_to_pyomo

# Lookup list for CPLEX constraint types (linear/lower bound/upper bound)
def cplex_type_name_lookup(cpx):
    type_obj = cpx.conflict.constraint_type
    lookup = {}
    for attr in dir(type_obj):
        if attr.startswith("_"):
            continue
        value = getattr(type_obj, attr)
        if isinstance(value, int):
            lookup[value] = attr
    return lookup

# Handle possibilities of different naming in constraint type
def cplex_constraint_type(cpx, *names):
    type_obj = cpx.conflict.constraint_type
    for name in names:
        if hasattr(type_obj, name):
            return getattr(type_obj, name)
    return None

# Add a conflict group for a single constraint (ctype, index)
def cplex_add_conflict_group(cpx, groups, members, ctype, idx):
    if ctype is None:
        return
    groups.append((1.0, ((ctype, idx),)))
    members.append((ctype, idx))

# Build constraint groups for CPLEX conflict refiner with one constraint per group
def build_cplex_single_member_conflict_groups(cpx):
    groups = []
    members = build_cplex_conflict_member_order(cpx)

    for ctype, idx in members:
        cplex_add_conflict_group(cpx, groups, [], ctype, idx)

    return groups, members

# Build the list of all constraints in the order they are indexed in CPLEX, along with their type
def build_cplex_conflict_member_order(cpx):
    members = []
    linear_type = cplex_constraint_type(cpx, "linear", "linear_constraint")
    for idx in range(cpx.linear_constraints.get_num()):
        if linear_type is not None:
            members.append((linear_type, idx))

    lower_type = cplex_constraint_type(cpx, "lower_bound", "lower")
    upper_type = cplex_constraint_type(cpx, "upper_bound", "upper")
    for idx in range(cpx.variables.get_num()):
        if lower_type is not None:
            members.append((lower_type, idx))
        if upper_type is not None:
            members.append((upper_type, idx))

    indicator_type = cplex_constraint_type(cpx, "indicator", "indicator_constraint")
    if indicator_type is not None and hasattr(cpx, "indicator_constraints"):
        for idx in range(cpx.indicator_constraints.get_num()):
            members.append((indicator_type, idx))

    quadratic_type = cplex_constraint_type(cpx, "quadratic", "quadratic_constraint")
    if quadratic_type is not None and hasattr(cpx, "quadratic_constraints"):
        for idx in range(cpx.quadratic_constraints.get_num()):
            members.append((quadratic_type, idx))

    sos_type = cplex_constraint_type(cpx, "SOS", "sos", "sos_constraint")
    if sos_type is not None and hasattr(cpx, "SOS"):
        for idx in range(cpx.SOS.get_num()):
            members.append((sos_type, idx))

    return members

# Turn off CPLEX terminal output 
def suppress_cplex_terminal_output(cpx):
    for stream_name in ("results", "log", "warning", "error"):
        setter = getattr(cpx, f"set_{stream_name}_stream", None)
        if setter is not None:
            try:
                setter(None)
            except Exception:
                pass

    conflict_params = getattr(getattr(cpx, "parameters", None), "conflict", None)
    conflict_display = getattr(conflict_params, "display", None)
    if conflict_display is not None:
        try:
            conflict_display.set(0)
        except Exception:
            pass

# Compute the conflict (IIS)
def refine_cplex_conflict(cpx):
    suppress_cplex_terminal_output(cpx)
    groups, group_members = build_cplex_single_member_conflict_groups(cpx)
    try:
        cpx.conflict.refine(groups)
        return cpx.conflict.get(), group_members, None
    except Exception as list_exc:
        try:
            cpx.conflict.refine(*groups)
            return cpx.conflict.get(), group_members, None
        except Exception as splat_exc:
            tuple_exc = f"as list: {list_exc}; as positional groups: {splat_exc}"

        try:
            cpx.conflict.refine(cpx.conflict.all_constraints())
            statuses = cpx.conflict.get()
            all_members = build_cplex_conflict_member_order(cpx)
            return statuses, all_members[:len(statuses)], None
        except Exception as all_exc:
            return None, None, (
                "explicit groups failed "
                f"({tuple_exc}); all_constraints failed ({all_exc})"
            )


# Converts CPLEX's conflict name to a readable name
def cplex_member_display_name(cpx, type_name, member_idx):
    try:
        if "linear" in type_name:
            return normalize_solver_constraint_name(cpx.linear_constraints.get_names(member_idx))
        if "indicator" in type_name:
            return normalize_solver_constraint_name(cpx.indicator_constraints.get_names(member_idx))
        if "quadratic" in type_name:
            return normalize_solver_constraint_name(cpx.quadratic_constraints.get_names(member_idx))
        if "sos" in type_name:
            return normalize_solver_constraint_name(cpx.SOS.get_names(member_idx))
    except Exception:
        pass
    return f"{type_name}({member_idx})"

# Classify a CPLEX conflict member as a user-forced constraint (add to forced_atom_ids) vs a non-forced model constraint (add to non_forced_constraints) vs ignore (if it's a bound constraint that doesn't directly correspond to a user-forced constraint)
def add_cplex_conflict_member(
    cpx,
    solver_name_to_pyomo,
    ctype,
    member_idx,
    atoms,
    forced_atom_ids,
    non_forced_constraints,
):
    type_lookup = cplex_type_name_lookup(cpx)
    type_name = type_lookup.get(ctype, str(ctype))

    if "lower" in type_name or "upper" in type_name:
        return

    member_name = cplex_member_display_name(cpx, type_name, member_idx)

    pyomo_con = solver_name_to_pyomo.get(normalize_solver_constraint_name(member_name))
    if pyomo_con is not None:
        add_pyomo_conflict_constraint(pyomo_con, atoms, forced_atom_ids, non_forced_constraints)
        return

    atom_id = parse_atom_id_from_constraint_name(member_name, "hard_forced_iis")
    if atom_id is not None:
        forced_atom_ids.add(atom_id)
        return

    non_forced_constraints.add(member_name)

# Find a CPLEX conflict, then translate it into IIS-style report format
def find_irreducible_infeasible_set_cplex(atoms, parsed_config):
    solver = pyo.SolverFactory("cplex_persistent")
    if solver is None or not solver.available(exception_flag=False):
        return None, None, "cplex_persistent solver not available"

    model = build_model(parsed_config, user_constraints=None, spill_control=default_spill_control())
    model.hard_forced_iis_idx = pyo.RangeSet(0, len(atoms) - 1)

    def hard_forced_iis_rule(m, i):
        atom = atoms[int(i)]
        return m.d[atom["var"], atom["hour"]] == atom["value"]

    model.hard_forced_iis = pyo.Constraint(model.hard_forced_iis_idx, rule=hard_forced_iis_rule)

    try:
        solver.set_instance(model)
    except Exception as exc:
        return None, None, f"cannot set cplex_persistent instance ({exc})"

    cpx = solver._solver_model
    conflict_statuses, group_members, conflict_error = refine_cplex_conflict(cpx)
    if conflict_error:
        return None, None, f"CPLEX conflict refiner failed ({conflict_error})"

    member_statuses = {
        cpx.conflict.group_status.member,
        cpx.conflict.group_status.possible_member,
    }

    forced_atom_ids = set()
    non_forced_constraints = set()
    solver_name_to_pyomo = add_pyomo_constraint_order_display_names(
        model,
        cpx,
        extract_solver_name_to_pyomo_constraint_map(solver),
    )

    for (ctype, member_idx), status in zip(group_members, conflict_statuses):
        if status not in member_statuses:
            continue
        add_cplex_conflict_member(
            cpx,
            solver_name_to_pyomo,
            ctype,
            member_idx,
            atoms,
            forced_atom_ids,
            non_forced_constraints,
        )

    if not forced_atom_ids:
        return (
            None,
            sorted(non_forced_constraints),
            "conflict computed but no hard_forced_iis rows identified",
        )

    return (
        [atoms[i] for i in sorted(forced_atom_ids)],
        sorted(non_forced_constraints),
        "ok",
    )

# Call Gurobi IIS extractor and return the identified IIS user's constraints along with any model constraints that were part of the IIS
def find_irreducible_infeasible_set(atoms, parsed_config, solver_name="gurobi"):
    solver_name = normalize_solver_name(solver_name)
    if not atoms:
        return [], [], "none", "No forced atoms were provided."

    if solver_name == "highs":
        return [], [], "none", "HiGHS does not provide IIS extraction. The forced scenario is infeasible."

    if solver_name == "gurobi":
        gurobi_iis, gurobi_constraints, gurobi_note = find_irreducible_infeasible_set_gurobi(atoms, parsed_config)
        if gurobi_iis is not None:
            return gurobi_iis, (gurobi_constraints or []), "gurobi_iis", "Gurobi returned one IIS successfully. IIS subsets are not unique."
        return [], [], "none", f"Gurobi IIS unavailable: {gurobi_note}."

    cplex_iis, cplex_constraints, cplex_note = find_irreducible_infeasible_set_cplex(atoms, parsed_config)
    if cplex_iis is not None:
        return cplex_iis, (cplex_constraints or []), "cplex_conflict_refiner", "CPLEX returned one IIS-style conflict successfully. IIS/conflict subsets are not unique."
    return [], [], "none", f"CPLEX conflict refiner unavailable: {cplex_note}."

# Build relaxation model to find relaxation pathway
def build_relaxation_model(atoms, parsed_config, relaxable_ids, forbidden_relax_sets):
    model = build_model(
        parsed_config,
        user_constraints=None,
        spill_control=default_spill_control(),
    )

    model.relax_idx = pyo.RangeSet(0, len(atoms) - 1)
    model.relax = pyo.Var(model.relax_idx, domain=pyo.Binary)
    model.relax_fixed = pyo.ConstraintList() # enforce that non-relaxable atoms are fixed to their value
    model.soft_forced = pyo.ConstraintList() # enforce that relaxable atoms can be relaxed to either value
    model.no_good = pyo.ConstraintList() # enforce that forbidden combinations of relaxations are not allowed

    #release force constraints for value = 1, keep force constraints for value = 0
    for i, atom in enumerate(atoms):
        var = model.d[atom["var"], atom["hour"]]

        if i not in relaxable_ids:
            model.relax_fixed.add(model.relax[i] == 0)

        if atom["value"] == 1:
            model.soft_forced.add(var >= 1 - model.relax[i])
        else:
            model.soft_forced.add(var <= model.relax[i])
    
    #prevent forbidden combinations of relaxations
    for forbidden in forbidden_relax_sets:
        if forbidden:
            model.no_good.add(
                sum(model.relax[i] for i in forbidden) <= len(forbidden) - 1
            )

    #turn off the model objective and find the smallest repair
    deactivate_all_lexicographic_objectives(model)
    model.min_relax_obj = pyo.Objective(
        expr=sum(model.relax[i] for i in relaxable_ids),
        sense=pyo.minimize,
    )
    return model

# Solver settings for finding minimal relaxations
def relaxation_solver(model, solver_name="gurobi"):
    try:
        result = create_solver(solver_name).solve(model, tee=False, load_solutions=False)
        note = {
            "gurobi": "Gurobi used for relaxation pathways.",
            "highs": "HiGHS used for relaxation pathways.",
            "cplex": "CPLEX used for relaxation pathways.",
        }[normalize_solver_name(solver_name)]
        return result, normalize_solver_name(solver_name), note
    except Exception as exc:
        chosen = normalize_solver_name(solver_name)
        if chosen == "highs":
            raise
        note = f"{chosen.capitalize()} unavailable for relaxation pathways ({exc})."

    result = create_solver("highs").solve(model, tee=False, load_solutions=False)
    return result, "highs", f"{note} Fallback to HiGHS."

# Function to solve the original lexicographic model after relaxing a subset of forced atoms
def solve_lexicographic_with_atoms(atoms, parsed_config, relaxed_ids=None, solver_name="gurobi"):
    locked_objective_values = {}
    stage_terminations = []

    for stage_index, objective_name in enumerate(parsed_config["lexicographic_objectives"], start=1):
        model = prepare_stage_model(
            parsed_config,
            user_constraints=None,
            spill_control=default_spill_control(),
            locked_objective_values=locked_objective_values,
            stage_index=stage_index,
        )
        apply_forced_atoms(model, atoms, relaxed_ids=relaxed_ids)

        solver = create_solver(solver_name)
        result = solver.solve(model, tee=False, load_solutions=False)
        term = str(result.solver.termination_condition)
        stage_terminations.append(term)
        if result.solver.termination_condition != pyo.TerminationCondition.optimal:
            return False, stage_terminations

        model.solutions.load_from(result)
        locked_objective_values[objective_name] = objective_value_from_model(model, objective_name)

    return True, stage_terminations

# Solve the relaxation model and extract the identified minimal relaxations
def solve_relaxation_model(atoms, parsed_config, relaxable_ids=None, forbidden_relax_sets=None, solver_name="gurobi"):
    forbidden_relax_sets = forbidden_relax_sets or [] #previously found relaxation sets
    relaxable_ids = set(range(len(atoms))) if relaxable_ids is None else set(relaxable_ids)

    if not atoms:
        return {"termination": "optimal", "relaxed_atoms": [], "min_relax_count": 0}

    model = build_relaxation_model(atoms, parsed_config, relaxable_ids, forbidden_relax_sets)
    result, solver_method, solver_note = relaxation_solver(model, solver_name=solver_name)

    term = result.solver.termination_condition
    if not is_termination_feasible(term):
        return {
            "termination": str(term),
            "relaxed_atoms": None,
            "min_relax_count": None,
            "solver_method": solver_method,
            "solver_note": solver_note,
            "lexicographic_feasible": None,
            "lexicographic_p1_termination": None,
            "lexicographic_p2_termination": None,
        }

    model.solutions.load_from(result)
    relaxed_ids = [i for i in range(len(atoms)) if pyo.value(model.relax[i]) > 0.5]
    relaxed_atoms = [atoms[i] for i in relaxed_ids]

    #rebuild original model with only the identified relaxations to verify feasibility and check lexicographic termination
    lex_feasible, lex_stage_terms = solve_lexicographic_with_atoms(
        atoms, parsed_config, relaxed_ids=relaxed_ids, solver_name=solver_name
    )

    return {
        "termination": str(term),
        "relaxed_atoms": relaxed_atoms,
        "relaxed_ids": set(relaxed_ids),
        "min_relax_count": len(relaxed_atoms),
        "solver_method": solver_method,
        "solver_note": solver_note,
        "lexicographic_feasible": lex_feasible,
        "lexicographic_stage_terminations": lex_stage_terms,
        "lexicographic_p1_termination": lex_stage_terms[0] if len(lex_stage_terms) >= 1 else None,
        "lexicographic_p2_termination": lex_stage_terms[1] if len(lex_stage_terms) >= 2 else None,
        "lexicographic_p3_termination": lex_stage_terms[2] if len(lex_stage_terms) >= 3 else None,
    }

# Enumerate multiple different minimal relaxation pathway 
def find_relaxation_pathways(atoms, parsed_config, max_pathways=3, relaxable_ids=None, solver_name="gurobi"):
    pathways = []
    forbidden_sets = []
    method = "unknown"
    note = ""

    for _ in range(max_pathways):
        result = solve_relaxation_model(
            atoms,
            parsed_config,
            relaxable_ids=relaxable_ids,
            forbidden_relax_sets=forbidden_sets,
            solver_name=solver_name,
        )
        method = result.get("solver_method", method)
        note = result.get("solver_note", note)

        if result["relaxed_atoms"] is None:
            break

        pathways.append(result)
        forbidden_sets.append(result["relaxed_ids"])

    return pathways, method, note

def infeasible_analysis(user_constraints, spill_control, parsed_config, solver_name="gurobi"):
    """Run full infeasibility diagnosis: IIS and minimal relaxation pathways."""
    solver_name = normalize_solver_name(solver_name)
    atoms = build_atoms(user_constraints, spill_control, parsed_config)

    feasible, term = is_forced_set_feasible(atoms, parsed_config, solver_name=solver_name)
    if feasible:
        return {
            "is_infeasible": False, "termination": term,
            "iis_atoms": [], "iis_forced": [], "iis_non_forced_constraints": [],
            "iis_method": "none",
            "iis_note": "Forced set is feasible; IIS is not required.",
            "relaxation_method": "none",
            "relaxation_note": "Forced set is feasible; relaxation pathways are not required.",
            "relaxation_pathways": [],
        }

    iis_forced, iis_nf_cons, iis_method, iis_note = find_irreducible_infeasible_set(
        atoms, parsed_config, solver_name=solver_name
    )
    pathways, relax_method, relax_note = find_relaxation_pathways(
        atoms, parsed_config, max_pathways=3, solver_name=solver_name
    )

    return {
        "is_infeasible": True, "termination": term,
        "iis_atoms": iis_forced, "iis_forced": iis_forced,
        "iis_non_forced_constraints": iis_nf_cons,
        "iis_method": iis_method, "iis_note": iis_note,
        "relaxation_method": relax_method, "relaxation_note": relax_note,
        "relaxation_pathways": pathways,
    }
