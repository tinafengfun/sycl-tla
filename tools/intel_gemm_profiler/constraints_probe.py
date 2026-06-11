#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import copy
import re


def apply_static_probe_constraints(base_constraints, env_caps):
    constraints = copy.deepcopy(base_constraints)
    constraints["constraint_source"] = "phase_a_static_probe"
    actions = []
    if not env_caps["executables"]["streamk_example_available"]:
        constraints["limits"]["max_split_k"] = 1
        constraints["allowed_values"]["split_k"] = [1]
        actions.append(
            {
                "action": "limit_split_k",
                "reason": "streamk_example_unavailable",
                "max_split_k": 1,
            }
        )
    constraints["probe_feedback"] = {
        "mode": "static",
        "probe_rows": 0,
        "passed_probe_rows": 0,
        "failed_probe_rows": 0,
        "actions": actions,
    }
    return constraints


def blocked_rule_for_row(row):
    return {
        "rule_id": f"probe.blocked.{row['candidate_id']}",
        "match": {
            "tile_m": int(re.search(r"_tm(\d+)_", row["candidate_id"]).group(1)),
            "tile_n": int(re.search(r"_tn(\d+)_", row["candidate_id"]).group(1)),
            "tile_k": int(re.search(r"_tk(\d+)_", row["candidate_id"]).group(1)),
            "sg_m": int(re.search(r"_sg(\d+)x", row["candidate_id"]).group(1)),
            "sg_n": int(re.search(r"x(\d+)_st", row["candidate_id"]).group(1)),
            "split_k": int(row["split_k"]),
        },
        "reason": row.get("failure_reason", "") or row.get("status", "probe_failure"),
        "source": "phase_a_probe_failure",
        "evidence": {
            "candidate_id": row["candidate_id"],
            "shape_id": row.get("shape_id", ""),
            "status": row["status"],
            "stdout_log": row.get("stdout_log", ""),
        },
    }


def apply_run_probe_constraints(static_constraints, probe_rows, anomaly_report=None):
    constraints = copy.deepcopy(static_constraints)
    constraints["constraint_source"] = "phase_a_run_probe"
    constraints["limits"]["max_slm_kb"] = min(
        constraints["limits"].get("max_slm_kb", 64),
        64,
    )
    actions = []
    if not any(row["status"] == "pass" and int(row["split_k"]) > 1 for row in probe_rows):
        constraints["limits"]["max_split_k"] = 1
        constraints["allowed_values"]["split_k"] = [1]
        actions.append(
            {
                "action": "limit_split_k",
                "reason": "no_successful_split_k_probe",
                "max_split_k": 1,
            }
        )
    failures = [row for row in probe_rows if row["status"] != "pass"]
    existing_ids = {rule.get("rule_id") for rule in constraints.get("blocked_rules", [])}
    for row in failures:
        rule = blocked_rule_for_row(row)
        if rule["rule_id"] not in existing_ids:
            constraints["blocked_rules"].append(rule)
            existing_ids.add(rule["rule_id"])
            actions.append(
                {
                    "action": "block_candidate",
                    "reason": "probe_failure",
                    "rule_id": rule["rule_id"],
                    "candidate_id": row["candidate_id"],
                    "shape_id": row.get("shape_id", ""),
                    "status": row["status"],
                }
            )
    for rule in (anomaly_report or {}).get("auto_block_rules", []):
        if rule["rule_id"] not in existing_ids:
            constraints["blocked_rules"].append(rule)
            existing_ids.add(rule["rule_id"])
            actions.append(
                {
                    "action": "block_candidate",
                    "reason": rule.get("reason", "probe_anomaly"),
                    "rule_id": rule["rule_id"],
                    "candidate_id": rule["rule_id"].replace("probe.auto_block.anomaly.", ""),
                    "source": "probe_anomaly",
                }
            )
    constraints["probe_feedback"] = {
        "mode": "run",
        "probe_rows": len(probe_rows),
        "passed_probe_rows": sum(1 for row in probe_rows if row["status"] == "pass"),
        "failed_probe_rows": len(failures),
        "anomaly_count": len((anomaly_report or {}).get("anomalies", [])),
        "auto_block_rule_count": len((anomaly_report or {}).get("auto_block_rules", [])),
        "blocked_rule_count": len(constraints.get("blocked_rules", [])),
        "actions": actions,
    }
    return constraints


def apply_probe_results_to_profiles(profiles, compiler_probe_summary):
    updated = copy.deepcopy(profiles)
    result_by_id = {item["compiler_profile_id"]: item for item in compiler_probe_summary.get("results", [])}
    selected_profile_ids = set(compiler_probe_summary.get("selected_profile_ids", {}).values())
    for profile in updated["profiles"]:
        result = result_by_id.get(profile["compiler_profile_id"])
        if result:
            profile["probe_status"] = result["status"]
            profile["probe_avg_tflops"] = result["avg_tflops"]
            profile["probe_avg_runtime_ms"] = result["avg_runtime_ms"]
        profile["probe_selected"] = profile["compiler_profile_id"] in selected_profile_ids
    return updated
