import json
from collections import defaultdict
from pathlib import Path

from cfg_model_builder import load_config, parse_config, default_spill_control
from cfg_model_runner import normalize_solver_name, run_two_scenarios
from cfg_diagnostic_analysis import feasible_analysis, infeasible_analysis
from cfg_diagnostic_report import generate_diagnostic_report

CONFIGURATION_PATH = Path("configuration.json")


def load_runtime_paths(path):
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    payload = json.loads(file_path.read_text(encoding="utf-8"))
    baseline = payload.get("baseline")
    alternative = payload.get("alternative")
    solver_name = normalize_solver_name(payload.get("solver", "gurobi"))
    output_folder = payload.get("output_folder", "reports") or "reports"

    if not baseline:
        raise ValueError("Missing 'baseline' entry in configuration.json")
    if not alternative:
        raise ValueError("Missing 'alternative' entry in configuration.json")

    return Path(baseline), Path(alternative), solver_name, Path(output_folder)

# Load alternative scenario
def load_user_scenario(path):
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"User scenario file not found: {file_path}")
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    return {
        "user_constraints": payload.get("user_constraints", payload.get("user_alternative_constraints", {})),
        "user_spill_control": payload.get("spill_control", payload.get("user_spill_control", default_spill_control())),
    }

# Print helper
def print_result(title, result_data, parsed_config):
    N_p = parsed_config["N_p"]
    comp_ids = sorted(parsed_config["comp_ids"])
    active_ids = sorted(parsed_config["active_comps"])
    res_ids = sorted(parsed_config["reservoirs"].keys())

    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)

    print("P1 Solver status:", result_data["p1_status"])
    print("P1 Termination condition:", result_data["p1_termination"])
    print("P2 Solver status:", result_data["p2_status"])
    print("P2 Termination condition:", result_data["p2_termination"])
    if result_data.get("p3_status") is not None:
        print("P3 Solver status:", result_data["p3_status"])
        print("P3 Termination condition:", result_data["p3_termination"])
    print("Lexicographic objectives:", " -> ".join(result_data.get("lexicographic_objectives", [])))
    print(f"Priority 1 (total safety violation): {result_data['total_safety_violation']:.6f}")
    print(f"Net generation power over {N_p} h: {result_data['total_generation_power']:.4f}")
    print(f"Revenue over {N_p} h: EUR {result_data['total_revenue_eur']:.2f}")
    print(f"Total generated power: {result_data['total_gen']:.2f}")
    print(f"Total consumed power (pump): {result_data['total_pump']:.2f}")

    for rid in res_ids:
        print(f"Final level {rid}: h[{N_p}]={result_data['final_h'][rid]:.2f}  "
              f"delta={result_data['delta_h'][rid]:+.2f}")

    # hourly table header
    mode_hdr = " ".join(f"d_{cid:>3s}" for cid in comp_ids)
    flow_hdr = " | ".join(f"Q_{cid:>3s}" for cid in comp_ids)
    power_hdr = " | ".join(f"P_{cid:>3s}" for cid in active_ids)

    print(f"\nHour | {mode_hdr} | {flow_hdr} | {power_hdr} | NetPower")
    for row in result_data["hourly"]:
        modes = " ".join(f"{row[f'd_{cid}']:>5d}" for cid in comp_ids)
        flows = " | ".join(f"{row[f'Q_{cid}']:>5.2f}" for cid in comp_ids)
        powers = " | ".join(f"{row[f'P_{cid}']:>5.2f}" for cid in active_ids)
        print(f"{row['hour']:>4d} | {modes} | {flows} | {powers} | {row['net_power']:>8.2f}")

def print_comparison(base_result, alt_result, parsed_config,
                     title="COMPARISON: BASELINE VS MODE-CONSTRAINED ALTERNATIVE"):
    N_p = parsed_config["N_p"]
    comp_ids = sorted(parsed_config["comp_ids"])
    res_ids = sorted(parsed_config["reservoirs"].keys())

    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)

    delta_violation = alt_result["total_safety_violation"] - base_result["total_safety_violation"]
    delta_power = alt_result["total_generation_power"] - base_result["total_generation_power"]
    delta_revenue = alt_result["total_revenue_eur"] - base_result["total_revenue_eur"]
    print(f"Delta total safety violation (alt - baseline): {delta_violation:+.6f}")
    print(f"Delta total generation power (alt - baseline): {delta_power:+.4f}")
    print(f"Delta total revenue (alt - baseline): EUR {delta_revenue:+.2f}")

    for rid in res_ids:
        dh = alt_result["final_h"][rid] - base_result["final_h"][rid]
        print(f"Delta final h_{rid} (alt - baseline): {dh:+.4f}")

    changed = 0
    for t in range(N_p):
        if base_result["hourly"][t]["mode_tuple"] != alt_result["hourly"][t]["mode_tuple"]:
            changed += 1
    print(f"Hours with changed mode tuple: {changed}/{N_p}")

    mode_str = lambda row: "(" + ",".join(str(row[f"d_{cid}"]) for cid in comp_ids) + ")"
    print(f"\nHour | Base mode | Alt mode  | BaseNetP | AltNetP  | DeltaNetP")
    for t in range(N_p):
        b = base_result["hourly"][t]
        a = alt_result["hourly"][t]
        dn = a["net_power"] - b["net_power"]
        print(
            f"{t:>4d} | {mode_str(b):>9s} | {mode_str(a):>9s} | "
            f"{b['net_power']:>8.2f} | {a['net_power']:>8.2f} | {dn:>+9.2f}"
        )


def print_top_mode_impacts(feasible_diag, parsed_config, top_k=8):
    print("\n" + "=" * 72)
    print("TOP HOURS / MODE RESTRICTIONS FROM USER ALTERNATIVE")
    print("=" * 72)

    forced_lookup = feasible_diag.get("forced_mode_lookup", feasible_diag.get("forced_mode", {}))
    ranked = feasible_diag.get("top_rows", feasible_diag.get("all_rows", []))
    top_rows = ranked[:min(top_k, len(ranked))]

    active_ids = sorted(parsed_config["active_comps"])
    delta_hdrs = " | ".join(f"dP_{cid}" for cid in active_ids)
    print(f"Rank | Hour | |DeltaNet| | DeltaNet | {delta_hdrs} | ForcedMode")
    for i, row in enumerate(top_rows, start=1):
        forced_txt = ", ".join(forced_lookup.get(row["hour"], ["-"]))
        deltas = " | ".join(f"{row.get(f'delta_P_{cid}', 0.0):>+9.2f}" for cid in active_ids)
        abs_delta_net = row.get("abs_delta_net", row.get("abs_delta_net_power", 0.0))
        delta_net = row.get("delta_net", row.get("delta_net_power", 0.0))
        print(
            f"{i:>4d} | {row['hour']:>4d} | {abs_delta_net:>10.2f} | "
            f"{delta_net:>+9.2f} | {deltas} | {forced_txt}"
        )


def print_binding_constraint_comparison(feasible_diag):
    only_base = feasible_diag.get("only_base", [])
    only_alt = feasible_diag.get("only_alt", [])
    in_both = feasible_diag.get("in_both", [])

    print("\n" + "=" * 72)
    print("BINDING CONSTRAINT COMPARISON: BASELINE vs ALTERNATIVE")
    print("=" * 72)
    print(f"\nConstraints binding in BOTH: {len(in_both)}")
    print(f"Constraints binding ONLY in baseline (relaxed): {len(only_base)}")
    print(f"Constraints binding ONLY in alternative (new bottlenecks): {len(only_alt)}")

    if only_alt:
        print("\n--- NEW BOTTLENECKS ---")
        by_family = defaultdict(list)
        for name, idx, side in only_alt:
            by_family[name].append((idx, side))
        for family in sorted(by_family):
            print(f"  {family}:")
            for idx, side in sorted(by_family[family]):
                print(f"    [{idx}] {side} bound")

    if only_base:
        print("\n--- RELAXED CONSTRAINTS ---")
        by_family = defaultdict(list)
        for name, idx, side in only_base:
            by_family[name].append((idx, side))
        for family in sorted(by_family):
            print(f"  {family}:")
            for idx, side in sorted(by_family[family]):
                print(f"    [{idx}] {side} bound")


def print_infeasibility_diagnosis(diagnosis):
    print("\n" + "=" * 72)
    print("INFEASIBILITY DIAGNOSIS")
    print("=" * 72)
    print(f"Alternative termination condition: {diagnosis.get('termination')}")

    iis_atoms = diagnosis.get("iis_atoms", [])
    iis_method = diagnosis.get("iis_method", "none")
    iis_note = diagnosis.get("iis_note", "")
    print(f"\n1) Irreducible infeasible set ({iis_method})")
    if iis_note:
        print(f"   Note: {iis_note}")
    if not iis_atoms:
        print("   No IIS subset identified.")
    else:
        for atom in sorted(iis_atoms, key=lambda a: (a["hour"], a["var"], a["value"])):
            src = ",".join(atom.get("sources", []))
            print(f"   d[{atom['var']},{atom['hour']}] = {atom['value']} (source: {src})")

    iis_nf = diagnosis.get("iis_non_forced_constraints", [])
    print("   Conflicting non-forced constraints in IIS:")
    if not iis_nf:
        print("     None identified.")
    else:
        for con_name in iis_nf:
            print(f"     {con_name}")

    print("\n2) Relaxation pathways")
    pathways = diagnosis.get("relaxation_pathways", [])
    if not pathways:
        print("   No relaxation pathway found.")
    else:
        for i, path in enumerate(pathways, start=1):
            print(f"   Pathway {i}: relax {path['min_relax_count']} forced constraints")
            for atom in sorted(path["relaxed_atoms"], key=lambda a: (a["hour"], a["var"])):
                print(f"     release d[{atom['var']},{atom['hour']}] = {atom['value']}")

# Main function
def main():
    baseline_config_path, alternative_scenario_path, solver_name, output_folder = load_runtime_paths(CONFIGURATION_PATH)

    # Load and parse configuration
    cfg = load_config(baseline_config_path)
    parsed_cfg = parse_config(cfg)

    # Load user constraints
    alt_payload = load_user_scenario(alternative_scenario_path)
    user_constraints = alt_payload["user_constraints"]
    user_spill_control = alt_payload["user_spill_control"]

    # Run both scenarios and collect results
    run_outputs = run_two_scenarios(
        parsed_cfg,
        user_constraints,
        spill_control=user_spill_control,
        solver_name=solver_name,
    )
    feasible_diag = None
    infeasible_diag = None

    # Print baseline results
    print_result("1) BASELINE OPTIMIZATION RESULT", run_outputs["baseline_result"], parsed_cfg)

    # Print alternative scenario results
    if run_outputs["alternative"]["is_feasible"]:
        print_result(
            "2) USER'S ALTERNATIVE SCENARIO RESULT",
            run_outputs["alternative"]["result"],
            parsed_cfg,
        )
    else:
        print("\n" + "=" * 72)
        print("2) USER'S ALTERNATIVE SCENARIO RESULT")
        print("=" * 72)
        print("- Optimization result: INFEASIBLE")
        print(f"- Reason: {run_outputs['alternative']['termination']}")

    # Print comparison of baseline vs alternative
    if run_outputs["alternative"]["is_feasible"]:
        feasible_diag = feasible_analysis(
            base_model=run_outputs["baseline_model"],
            base_result=run_outputs["baseline_result"],
            alt_model=run_outputs["alternative"]["model"],
            alt_result=run_outputs["alternative"]["result"],
            user_constraints=user_constraints,
            parsed_config=parsed_cfg,
            top_k=8,
        )
        print_comparison(
            run_outputs["baseline_result"],
            run_outputs["alternative"]["result"],
            parsed_cfg,
        )
    
    # Print feasible diagnosis
    if run_outputs["alternative"]["is_feasible"]:
        print_top_mode_impacts(feasible_diag, parsed_cfg)
        print_binding_constraint_comparison(feasible_diag)
        print("Alternative scenario status: FEASIBLE")
    
    # Print infeasibility diagnosis
    else:
        infeasible_diag = infeasible_analysis(
            user_constraints=user_constraints,
            spill_control=user_spill_control,
            parsed_config=parsed_cfg,
            solver_name=solver_name,
        )
        print_infeasibility_diagnosis(infeasible_diag)
        print("Alternative scenario status: INFEASIBLE")

    # Print a full diagnostic report as a JSON file
    report_out = generate_diagnostic_report(
        run_outputs=run_outputs,
        user_constraints=user_constraints,
        spill_control=user_spill_control,
        parsed_config=parsed_cfg,
        feasible_diag=feasible_diag,
        infeasible_diag=infeasible_diag,
        report_prefix=alternative_scenario_path.stem,
        output_folder=output_folder,
    )

    print(f"\nConfiguration loaded from: {CONFIGURATION_PATH}")
    print(f"Baseline loaded from: {baseline_config_path}")
    print(f"Alternative loaded from: {alternative_scenario_path}")
    print(f"Solver selected: {solver_name}")
    print(f"Report folder: {output_folder}")
    print(f"Summary JSON written to: {report_out['json_path']}")

    timings = run_outputs.get("timings", {})
    if timings:
        print("\n" + "=" * 72)
        print("RUNTIME SUMMARY")
        print("=" * 72)
        for stage_name, seconds in timings.items():
            print(f"{stage_name}: {seconds:.4f} s")
        print(f"Total: {sum(timings.values()):.4f} s")

if __name__ == "__main__":
    main()
