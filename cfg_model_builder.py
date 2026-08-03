import json

import pyomo.environ as pyo

# Baseline configuration 

OBJECTIVE_SPECS = {
    "min_total_safety_violation": {
        "sense": "minimize",
        "expression_attr": "total_safety_violation",
    },
    "max_net_power": {
        "sense": "maximize",
        "expression_attr": "total_generation_power",
    },
    "max_revenue": {
        "sense": "maximize",
        "expression_attr": "total_revenue_eur",
    },
}

OBJECTIVE_ALIASES = {
    "min_total_safety_violation": "min_total_safety_violation",
    "max_net_power": "max_net_power",
    "max_revenue": "max_revenue",
    "max_revenue_eur": "max_revenue",
}

# Hyperplane power equations: alpha*P + beta*Q + gamma*H + delta = 0
# Convert to P = a + b*Q + c*H

def plane_to_abc(alpha, beta, gamma, delta):
    return (-delta/alpha, -beta/alpha, -gamma/alpha)

def line_corner_max(a, b, c, q_lo, q_hi, h_lo, h_hi):
    return max (
        a + b*q_lo + c*h_lo,
        a + b*q_lo + c*h_hi,
        a + b*q_hi + c*h_lo,
        a + b*q_hi + c*h_hi
    )

def line_corner_min(a, b, c, q_lo, q_hi, h_lo, h_hi):
    return min (
        a + b*q_lo + c*h_lo,
        a + b*q_lo + c*h_hi,
        a + b*q_hi + c*h_lo,
        a + b*q_hi + c*h_hi
    )

# Load configuration from JSON file
def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg =json.load(f)
    return cfg

def normalize_hourly_series(raw_series, horizon, series_name):
    if raw_series is None:
        return {t: 0.0 for t in range(horizon)}

    values = raw_series
    if isinstance(raw_series, dict):
        if "values" in raw_series:
            values = raw_series["values"]
        else:
            normalized = {t: 0.0 for t in range(horizon)}
            for t, value in raw_series.items():
                hour = int(t)
                if 0 <= hour < horizon:
                    normalized[hour] = float(value)
            return normalized

    values = [float(value) for value in values]
    normalized = {}
    for t in range(horizon):
        normalized[t] = values[t] if t < len(values) else 0.0
    return normalized

def normalize_lexicographic_objectives(cfg):
    raw_objectives = (cfg.get("objectives", {}) or {}).get("lexicographic")
    if not raw_objectives:
        raw_objectives = ["min_total_safety_violation", "max_net_power"]

    normalized = []
    for objective_name in raw_objectives:
        key = OBJECTIVE_ALIASES.get(str(objective_name).strip().lower())
        if key is None:
            supported = ", ".join(sorted(OBJECTIVE_SPECS))
            raise ValueError(
                f"Unsupported objective '{objective_name}'. Choose from: {supported}."
            )
        normalized.append(key)

    if normalized[0] != "min_total_safety_violation":
        raise ValueError(
            "The first lexicographic objective must be 'min_total_safety_violation'."
        )

    if len(normalized) != len(set(normalized)):
        raise ValueError("Duplicate objective names are not allowed in the lexicographic list.")

    return normalized

# Build Pyomo model from configuration
def parse_config(cfg): 
    pc = {}

    #time horizon
    pc["N_p"] = int(cfg["horizon"]["hours"])
    pc["dt"] = float(cfg["horizon"]["dt_s"])
    pc["dt_hours"] = pc["dt"] / 3600.0
    pc["lexicographic_objectives"] = normalize_lexicographic_objectives(cfg)
    pc["price_eur_per_mwh"] = normalize_hourly_series(
        cfg.get("price_eur_per_mwh"),
        pc["N_p"],
        "price_eur_per_mwh",
    )

    #reservoirs
    pc["reservoirs"] = {}
    for rid, res in cfg["reservoirs"].items():
        pc["reservoirs"][rid] = {
            "area": float(res["area_m2"]),
            "h_min": float(res["h_min"]),
            "h_max": float(res["h_max"]),
            "h0": float(res["h0"]),
            "V0": float(res["area_m2"] * res["h0"] / 10000),
            "V_min": float(res["area_m2"] * res["h_min"]),
            "V_max": float(res["area_m2"] * res["h_max"]),
            "V_range": float(res["area_m2"] * (res["h_max"] - res["h_min"])),
            "V0_norm": float(
                (res["area_m2"] * res["h0"] - res["area_m2"] * res["h_min"])
                / (res["area_m2"] * (res["h_max"] - res["h_min"]))),
        }

    #tailwaters
    pc["tailwaters"] = {}
    for tid, tw in cfg.get("tailwaters", {}).items():
        pc["tailwaters"][tid] = {"head": float(tw["head"])}

    #initial head of each node
    def initial_head(node_id):
        if node_id in pc["reservoirs"]:
            return pc["reservoirs"][node_id]["h0"]
        elif node_id in pc["tailwaters"]:
            return pc["tailwaters"][node_id]["head"]
        else:
            raise ValueError(f"Node {node_id} not found in reservoirs or tailwaters")
        
    #head bounds for each node
    def head_bounds(node_id):
        if node_id in pc["reservoirs"]:
            r = pc["reservoirs"][node_id]
            return (r["h_min"], r["h_max"])
        elif node_id in pc["tailwaters"]:
            head = pc["tailwaters"][node_id]["head"]
            return (head, head)  # fixed head for tailwaters
        else:
            raise ValueError(f"Node {node_id} not found in reservoirs or tailwaters")
    
    #edges
    pc["edges"] = {}
    for eid in cfg["topology"]["edges"]:
        pc["edges"][eid["id"]] = {"from": eid["from"], "to": eid["to"]}

    #components
    pc["components"] = {}
    pc["turbines"] = []
    pc["pumps"] = []
    pc["spills"] = []
    for cid, cd in cfg["components"].items():
        edge_id = cd["edge"]
        edge = pc["edges"][edge_id]
        from_node = edge["from"]
        to_node = edge["to"]

        comp = {
            "id": cid,
            "type": cd["type"],
            "edge": edge_id,
            "from": from_node,
            "to": to_node,
            "q_max": float(cd["q_max"]),
        }

        if cd["type"] in ("turbine", "pump"):
            comp["min_on_ratio"] = float(cd["min_on_ratio"])
            comp["q_min_on"] = comp["min_on_ratio"] * comp["q_max"]

            #power-plane coefficients
            raw_planes = cd["power_planes"]["planes"]
            coefs = {i: plane_to_abc(*p) for i, p in enumerate(raw_planes)}
            comp["coefs"] = coefs
            comp["n_planes"] = len(raw_planes)

            #head-difference
            h_from = initial_head(from_node)
            h_to = initial_head(to_node)
            high_node, low_node = (from_node, to_node) if h_from >= h_to else (to_node, from_node)
            comp["dH_high_node"] = high_node
            comp["dH_low_node"] = low_node
            hh_min, hh_max = head_bounds(high_node)
            hl_min, hl_max = head_bounds(low_node)
            comp["dH_min"] = hh_min - hl_max
            comp["dH_max"] = hh_max - hl_min

            #P_aux bounds
            aux_max = max(0.0, max(line_corner_max(*coefs[i], 0.0, comp["q_max"], comp["dH_min"], comp["dH_max"]) for i in coefs))
            aux_min = min(0.0, min(line_corner_min(*coefs[i], 0.0, comp["q_max"], comp["dH_min"], comp["dH_max"]) for i in coefs))
            comp["P_aux_max"] = aux_max
            comp["P_aux_min"] = aux_min
            comp["P_aux_ub_eff"] = aux_max
            comp["P_aux_lb_eff"] = 0.0 if cd["type"] == "pump" else aux_min

            if cd["type"] == "turbine":
                pc["turbines"].append(cid)
            else:
                pc["pumps"].append(cid)

        elif cd["type"] == "spill":
            comp["trigger"] = cd.get("trigger", {})
            pc["spills"].append(cid)

        pc["components"][cid] = comp

    # Match the legacy Phase 2 model ordering as closely as possible:
    # lower turbines to tailwater, upper turbines, pumps, then spills.
    tailwater_ids = set(pc["tailwaters"].keys())
    lower_turbines = [
        cid for cid in pc["turbines"]
        if pc["components"][cid]["to"] in tailwater_ids
    ]
    upper_turbines = [
        cid for cid in pc["turbines"]
        if cid not in lower_turbines
    ]
    pc["turbines"] = lower_turbines + upper_turbines
    pc["active_comps"] = pc["turbines"] + pc["pumps"]
    pc["comp_ids"] = (
        pc["turbines"]
        + pc["pumps"]
        + pc["spills"]
        + [
            cid for cid in pc["components"]
            if cid not in pc["active_comps"] and cid not in pc["spills"]
        ]
    )
    
    pc["inflows"] = {}
    for node_id, inflow in cfg.get("inflows", {}).items():
        pc["inflows"][node_id] = normalize_hourly_series(inflow, pc["N_p"], f"inflows.{node_id}")

    #initial flows
    # Numeric values fix Q[cid, 0]; null/"auto" leave hour-0 flow to the solver.
    def parse_initial_flow(value):
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in {"auto", "free", "solver"}:
            return None
        return float(value)

    pc["initial_flows"] = {
        cid: parse_initial_flow(q)
        for cid, q in cfg.get("initial_flows", {}).items()
    }

    #constraints from config
    pc["exclusive_pairs"] = cfg.get("constraints", {}).get("exclusive", [])
    pc["soft_bound_reservoirs"] = (cfg.get("constraints", {}).get("soft_bounds", {})).get("reservoirs", [])

    return pc

# User-alternative configuration

def normalize_user_constraints(raw_const, comp_ids):
    normalized = {cid: {} for cid in comp_ids}
    if raw_const is None:
        return normalized
    for cid in comp_ids: 
        user_const = raw_const.get(cid, {})
        for hour in user_const.get("on", []):
            normalized[cid][int(hour)] = 1
        for hour in user_const.get("off", []):
            normalized[cid][int(hour)] = 0
    return normalized

def collect_raw_forced_assignments(raw_const, comp_ids, raw_spill_control, spill_ids, horizon):
    assignments = []
    constraints = {} if raw_const is None else raw_const
    for cid in comp_ids:
        user_const = constraints.get(cid, {})
        for hour in user_const.get("on", []):
            assignments.append({"var": cid, "hour": int(hour), "value": 1, "source": "USER_ALTERNATIVE_CONSTRAINTS"})
        for hour in user_const.get("off", []):
            assignments.append({"var": cid, "hour": int(hour), "value": 0, "source": "USER_ALTERNATIVE_CONSTRAINTS"})

    #spill/weir control
    if raw_spill_control is not None and spill_ids:
        for sid in spill_ids:
            spill_const = raw_spill_control.get(sid, raw_spill_control)
            enabled = bool(spill_const.get("enabled", True)) if isinstance(spill_const, dict) else True
            if not enabled:
                for hour in range(horizon):
                    assignments.append({"var": sid, "hour": int(hour), "value": 0, "source": "USER_SPILL_CONTROL"})
                continue
            for hour in spill_const.get("on", []):
                assignments.append({"var": sid, "hour": int(hour), "value": 1, "source": "USER_SPILL_CONTROL"})
            for hour in spill_const.get("off", []):
                assignments.append({"var": sid, "hour": int(hour), "value": 0, "source": "USER_SPILL_CONTROL"})

    return assignments

def default_spill_control():
    return {"enabled": True, "on": [], "off": []}

def normalize_spill_control(raw_spill_control, spill_ids, horizon):
    normalized = {sid: {} for sid in spill_ids}
    if raw_spill_control is None:
        return normalized
    for sid in spill_ids:
        spill_const = raw_spill_control.get(sid, raw_spill_control)
        if not isinstance(spill_const, dict):
            continue
        if not bool(spill_const.get("enabled", True)):
            normalized[sid] = {t: 0 for t in range(horizon)}
            continue
        for hour in spill_const.get("on", []):
            normalized[sid][int(hour)] = 1
        for hour in spill_const.get("off", []):
            normalized[sid][int(hour)] = 0
    return normalized

# Model builder
def build_model(pc, user_constraints=None, spill_control=None):
    N_p = pc["N_p"]
    dt = pc["dt"]
    model = pyo.ConcreteModel()

    # Sets
    model.T = pyo.RangeSet(0, N_p-1)
    model.Tp1 = pyo.RangeSet(0, N_p)  # for initial conditions

    res_ids = sorted(pc["reservoirs"].keys())
    comp_ids = list(pc["comp_ids"])
    turbine_ids = list(pc["turbines"])
    pump_ids = list(pc["pumps"])
    spill_ids = list(pc["spills"])
    active_ids = list(pc["active_comps"])

    model.Reservoirs = pyo.Set(initialize=res_ids)
    model.Components = pyo.Set(initialize=comp_ids)
    model.Turbines = pyo.Set(initialize=turbine_ids)
    model.Pumps = pyo.Set(initialize=pump_ids)
    model.Spills = pyo.Set(initialize=spill_ids)
    model.ActiveComps = pyo.Set(initialize=active_ids)

    # Plane coefficients index
    for cid in active_ids:
        n = pc["components"][cid]["n_planes"]
        model.add_component(f"L_{cid}", pyo.RangeSet(0, n-1))

    # Parameters
    model.dt = pyo.Param(initialize=dt)

    #reservoir parameters
    model.Area = pyo.Param(model.Reservoirs, initialize={rid: pc["reservoirs"][rid]["area"] for rid in res_ids})
    model.H_min = pyo.Param(model.Reservoirs, initialize={rid: pc["reservoirs"][rid]["h_min"] for rid in res_ids})
    model.H_max = pyo.Param(model.Reservoirs, initialize={rid: pc["reservoirs"][rid]["h_max"] for rid in res_ids})
    model.H0 = pyo.Param(model.Reservoirs, initialize={rid: pc["reservoirs"][rid]["h0"] for rid in res_ids})
    model.V0 = pyo.Param(model.Reservoirs, initialize={rid: pc["reservoirs"][rid]["V0"] for rid in res_ids})
    
    

    #tailwater parameters
    model.TW_Head = {tid: pc["tailwaters"][tid]["head"] for tid in pc["tailwaters"]}

    #component parameters
    model.Q_max = pyo.Param(model.Components, initialize={cid: pc["components"][cid]["q_max"] for cid in comp_ids})

    q_min_on = {}
    for cid in active_ids:
        comp = pc["components"][cid]
        q_min_on[cid] = comp["q_min_on"] if "q_min_on" in comp else 0.0
    for sid in spill_ids:
        q_min_on[sid] = 0.0
    model.Q_min_on = pyo.Param(model.Components, initialize=q_min_on)

    #power-plane coefficients
    for cid in active_ids:
        coefs = pc["components"][cid]["coefs"]
        plane_set = getattr(model, f"L_{cid}")
        model.add_component(f"a_{cid}", pyo.Param(plane_set, initialize={i: coefs[i][0] for i in coefs}))
        model.add_component(f"b_{cid}", pyo.Param(plane_set, initialize={i: coefs[i][1] for i in coefs}))
        model.add_component(f"c_{cid}", pyo.Param(plane_set, initialize={i: coefs[i][2] for i in coefs}))

    #inflows
    for rid in res_ids:
        inflow_data = pc["inflows"].get(rid, {t: 0.0 for t in range(N_p)})
        model.add_component(f"Inflow_{rid}", pyo.Param(model.T, initialize=inflow_data))

    #initial flows
    model.InitialFlow = pyo.Param(
        model.Components,
        initialize={
            cid: (
                0.0
                if pc["initial_flows"].get(cid) is None
                else pc["initial_flows"].get(cid, 0.0)
            )
            for cid in comp_ids
        },
    )

    # Variables
    # reservoir volumes and heads
    def v_init(model, rid, t):
        return pc["reservoirs"][rid]["V0_norm"]

    def h_bounds(model, rid, t):
        r = pc["reservoirs"][rid]
        return r["h_min"], r["h_max"]

    def h_init(model, rid, t):
        return pc["reservoirs"][rid]["h0"]

    model.V = pyo.Var(model.Reservoirs, model.Tp1, bounds=(0.0, 1.0), initialize=v_init)
    model.H = pyo.Var(model.Reservoirs, model.Tp1, bounds=h_bounds, initialize=h_init)
    model.V_range = pyo.Param(model.Reservoirs, initialize={rid: pc["reservoirs"][rid]["V_range"] for rid in res_ids},)

    # flows and binary modes
    def q_bounds(model, cid, t):
        return 0.0, pc["components"][cid]["q_max"]

    def q_init(model, cid, t):
        initial_flow = pc["initial_flows"].get(cid)
        return 0.0 if initial_flow is None else initial_flow

    model.Q = pyo.Var(
        model.Components,
        model.T,
        domain=pyo.NonNegativeReals,
        bounds=q_bounds,
        initialize=q_init,
    )
    model.d = pyo.Var(model.Components, model.T, domain=pyo.Binary, initialize=0)

    # power variables for active components
    def p_bounds(model, cid, t):
        if pc["components"][cid]["type"] == "pump":
            return 0.0, None
        return None, None

    def p_aux_bounds(model, cid, t):
        comp = pc["components"][cid]
        return comp["P_aux_lb_eff"], comp["P_aux_ub_eff"]

    model.P = pyo.Var(model.ActiveComps, model.T, domain=pyo.Reals, bounds=p_bounds)
    model.P_aux = pyo.Var(model.ActiveComps, model.T, domain=pyo.Reals, bounds=p_aux_bounds)

    #soft-bound violation variables for reservoirs
    model.v_low = pyo.Var(model.Reservoirs, model.Tp1, domain=pyo.NonNegativeReals)
    model.v_high = pyo.Var(model.Reservoirs, model.Tp1, domain=pyo.NonNegativeReals)

    #head difference variables for active components
    def node_head(model, node_id,t):
        if node_id in pc["reservoirs"]:
            return model.H[node_id,t]
        elif node_id in pc["tailwaters"]:
            return pc["tailwaters"][node_id]["head"]
        else:
            raise ValueError(f"Node {node_id} not found in reservoirs or tailwaters")
    def dH_rule(model, cid, t):
        comp = pc["components"][cid]
        return node_head(model, comp["dH_high_node"], t) - node_head(model, comp["dH_low_node"], t)
    model.dH = pyo.Expression(model.ActiveComps, model.T, rule=dH_rule)

    #initial conditions
    model.init_cond = pyo.ConstraintList()
    for rid in res_ids:
        # model.V is normalized to [0, 1], so the initial storage must use
        # the normalized reservoir value as well.
        model.init_cond.add(model.V[rid, 0] == pc["reservoirs"][rid]["V0_norm"])
        model.init_cond.add(model.H[rid, 0] == pc["reservoirs"][rid]["h0"])
    for cid in comp_ids:
        if pc["components"][cid]["type"] == "spill":
            continue
        initial_flow = pc["initial_flows"].get(cid)
        if initial_flow is not None:
            model.init_cond.add(model.Q[cid, 0] == initial_flow)

    #water balance constraints
    def water_balance_rule(model, rid, t):
        net_flow = getattr(model, f"Inflow_{rid}")[t]
        for cid, comp in pc["components"].items():
            if comp["from"] == rid: #outflow from reservoir
                net_flow -= model.Q[cid, t]
            if comp["to"] == rid: #inflow to reservoir
                net_flow += model.Q[cid, t]
        return model.V[rid, t+1] == model.V[rid, t] + net_flow * model.dt / model.V_range[rid]
    model.water_balance = pyo.Constraint(model.Reservoirs, model.T, rule=water_balance_rule)

    #head-volume relationship
    def head_volume_rule(model, rid, t):
        return model.H[rid, t] == model.H_min[rid] + model.V[rid, t] * (model.H_max[rid] - model.H_min[rid])
    model.head_eq = pyo.Constraint(model.Reservoirs, model.Tp1, rule=head_volume_rule)

    #soft bounds on reservoir heads
    for rid in pc["soft_bound_reservoirs"]:
        def soft_low_rule(model, t, rid=rid):
            return model.H[rid, t] + model.v_low[rid, t] >= model.H_min[rid]
        model.add_component(f"soft_low_{rid}", pyo.Constraint(model.Tp1, rule=soft_low_rule))
        def soft_high_rule(model, t, rid=rid):
            return model.H[rid, t] - model.v_high[rid, t] <= model.H_max[rid]
        model.add_component(f"soft_high_{rid}", pyo.Constraint(model.Tp1, rule=soft_high_rule))

    #mutual exclusivity constraints
    if pc["exclusive_pairs"]:
        pairs = pc["exclusive_pairs"]
        model.exclusive_pairs = pyo.RangeSet(0, len(pairs)-1)
        def exclusive_rule(model, i, t):
            return sum(model.d[cid, t] for cid in pairs[i]) <= 1
        model.exclusive_const = pyo.Constraint(model.exclusive_pairs, model.T, rule=exclusive_rule)

    #flow-mode coupling constraints
    for cid in comp_ids:
        comp = pc["components"][cid]
        def q_upper_rule(model, t, cid=cid):
            return model.Q[cid, t] <= model.Q_max[cid] * model.d[cid, t]
        model.add_component(f"q_upper_{cid}", pyo.Constraint(model.T, rule=q_upper_rule))
        if comp["type"] == "spill":
            continue
        def q_min_on_rule(model, t, cid=cid, Q_min=comp["q_min_on"]):
            return model.Q[cid, t] >= Q_min * model.d[cid, t]
        model.add_component(f"q_min_on_{cid}", pyo.Constraint(model.T, rule=q_min_on_rule))

    #spill trigger constraints
    for sid in spill_ids:
        spill = pc["components"][sid]
        spill_trigger = spill.get("trigger", {})

        #flow trigger
        Q_trigger = spill_trigger.get("q_ge")
        Q_component = spill_trigger.get("q_component")
        if Q_trigger and Q_component is not None: 
            Q_comp_max = pc["components"][Q_component]["q_max"]
            def flow_trigger_rule(
                model,
                t,
                sid=sid,
                Q_component=Q_component,
                Q_trigger=Q_trigger,
                Q_comp_max=Q_comp_max,
            ):
                return model.Q[Q_component, t] >= Q_trigger - Q_comp_max * (1 - model.d[sid, t])
            model.add_component(f"flow_trigger_{sid}", pyo.Constraint(model.T, rule=flow_trigger_rule))

        #head trigger
        hp_trigger = spill_trigger.get("hp_ge")
        head_res = spill_trigger.get("head_reservoir")
        if head_res and hp_trigger is not None:
            def head_trigger_rule(
                model,
                t,
                sid=sid,
                head_res=head_res,
                hp_trigger=hp_trigger,
            ):
                big_m = model.H_max[head_res] - model.H_min[head_res]
                return model.H[head_res, t] >= hp_trigger - big_m * (1 - model.d[sid, t])
            model.add_component(f"head_trigger_{sid}", pyo.Constraint(model.T, rule=head_trigger_rule))

    #power-plane constraints for active components
    for cid in active_ids:
        comp = pc["components"][cid]
        plane_set = getattr(model, f"L_{cid}")
        a = getattr(model, f"a_{cid}")
        b = getattr(model, f"b_{cid}")
        c = getattr(model, f"c_{cid}")

        # turbine: P_aux <= a + b*Q + c*dH
        if comp["type"] == "turbine":
            def turbine_plane_rule(model, i, t, cid=cid, a=a, b=b, c=c):
                return model.P_aux[cid, t] <= a[i] + b[i]*model.Q[cid, t] + c[i]*model.dH[cid, t]
            model.add_component(f"turbine_plane_{cid}", pyo.Constraint(plane_set, model.T, rule=turbine_plane_rule))

        # pump: P_aux >= a + b*Q + c*dH
        elif comp["type"] == "pump":
            def pump_plane_rule(model, i, t, cid=cid, a=a, b=b, c=c):
                return model.P_aux[cid, t] >= a[i] + b[i]*model.Q[cid, t] + c[i]*model.dH[cid, t]
            model.add_component(f"pump_plane_{cid}", pyo.Constraint(plane_set, model.T, rule=pump_plane_rule))

    for cid in active_ids: 
        comp = pc["components"][cid]
        P_aux_ub = comp["P_aux_ub_eff"]
        P_aux_lb = comp["P_aux_lb_eff"]

        # P <= P_upper_bound * d
        def link_up_mode(model, t, cid=cid, ub=P_aux_ub):
            return model.P[cid, t] <= ub * model.d[cid, t]
        model.add_component(f"P_link_up_mode_{cid}", pyo.Constraint(model.T, rule=link_up_mode))

        # P >= P_lower_bound * d
        if comp["type"] != "pump":
            def link_low_mode(model, t, cid=cid, lb=P_aux_lb):
                return model.P[cid, t] >= lb * model.d[cid, t]
            model.add_component(f"P_link_low_mode_{cid}", pyo.Constraint(model.T, rule=link_low_mode))

        # P <= P_aux - P_lower_bound * (1 - d)
        def link_up_aux(model, t, cid=cid, lb=P_aux_lb):
            return model.P[cid, t] <= model.P_aux[cid, t] - lb * (1 - model.d[cid, t])
        model.add_component(f"P_link_up_aux_{cid}", pyo.Constraint(model.T, rule=link_up_aux))

        # P >= P_aux - P_upper_bound * (1 - d)
        def link_low_aux(model, t, cid=cid, ub=P_aux_ub):
            return model.P[cid, t] >= model.P_aux[cid, t] - ub * (1 - model.d[cid, t])
        model.add_component(f"P_link_low_aux_{cid}", pyo.Constraint(model.T, rule=link_low_aux))

    #user forced constraints
    norm_user = normalize_user_constraints(user_constraints, comp_ids)
    model.user_forced = pyo.ConstraintList()
    for cid, hour_map in norm_user.items(): 
        for hour, forced_value in sorted(hour_map.items()):
            model.user_forced.add(model.d[cid, hour] == forced_value)

    #user spill control constraints 
    actual_spill_control = default_spill_control() if spill_control is None else spill_control
    norm_spill = normalize_spill_control(actual_spill_control, spill_ids, N_p)
    model.spill_control = pyo.ConstraintList()
    for sid, hour_map in norm_spill.items():
        for hour, forced_value in sorted(hour_map.items()):
            model.spill_control.add(model.d[sid, hour] == forced_value)

    #objective function 
    model.total_safety_violation = pyo.Expression(expr=sum(model.v_low[rid, t] + model.v_high[rid, t] for rid in pc["soft_bound_reservoirs"] for t in model.Tp1))
    model.total_generation_power = pyo.Expression(expr=sum(model.P[tid, t] for tid in turbine_ids for t in model.T) - sum(model.P[pid, t] for pid in pump_ids for t in model.T))
    model.total_revenue_eur = pyo.Expression(
        expr=sum(
            pc["price_eur_per_mwh"][t]
            * (
                sum(model.P[tid, t] for tid in turbine_ids)
                - sum(model.P[pid, t] for pid in pump_ids)
            )
            * pc["dt_hours"]
            for t in model.T
        )
    )

    model.lexicographic_objective_names = list(pc["lexicographic_objectives"])
    model.lexicographic_objective_components = []
    for stage_idx, objective_name in enumerate(model.lexicographic_objective_names, start=1):
        spec = OBJECTIVE_SPECS[objective_name]
        objective = pyo.Objective(
            expr=getattr(model, spec["expression_attr"]),
            sense=pyo.minimize if spec["sense"] == "minimize" else pyo.maximize,
        )
        component_name = f"obj_p{stage_idx}"
        model.add_component(component_name, objective)
        if stage_idx > 1:
            objective.deactivate()
        model.lexicographic_objective_components.append(component_name)
    model.parsed_config = pc

    return model
