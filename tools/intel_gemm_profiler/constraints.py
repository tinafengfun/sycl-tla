#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import copy
import re

from .config import (
    DEFAULT_BUILD_CONFIG_PATH,
    DEFAULT_RUNTIME_CONFIG_PATH,
    list_compile_variants,
    list_runtime_variants,
    load_persisted_build_config,
    load_persisted_runtime_config,
    selected_compile_env,
    selected_runtime_env,
    update_build_config_variant,
    update_runtime_config_variant,
)
from .schemas import SCHEMA_VERSION
from .utils import now_iso, read_json


def default_constraints():
    """
    Default B70 (BMG G31) constraints.
    sg_m × sg_n restricted to {16, 32} (B70 max subgroup = 32, SG8×8=64 is illegal).
    Set ``valid_subgroup_sizes`` to null for unrestricted (≤32), or pass
    a device-specific constraints JSON (e.g. constraints_b60.json).
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "constraint_source": "b70_default",
        "device_arch": "bmg",
        "description": "B70 (BMG G31) restricted: sg_m × sg_n ∈ {16, 32}",
        "limits": {
            "max_slm_kb": 64,
            "subgroup_size": 16,
            "max_split_k": 6,
            "max_stages": 3,
            "valid_subgroup_sizes": [16, 32],
        },
        "allowed_values": {
            "tile_m": [8, 16, 32, 64, 128, 256, 512],
            "tile_n": [32, 64, 96, 128, 192, 256, 512],
            "tile_k": [16, 32, 64],
            "sg_m": [1, 2, 4, 8],
            "sg_n": [2, 4, 8],
            "stages": [0, 1, 2, 3],
            "split_k": [1, 2, 3, 4, 6],
            "grf_mode": [256],
        },
        "blocked_rules": [],
        "probe_feedback": {
            "mode": "default",
            "probe_rows": 0,
            "passed_probe_rows": 0,
            "failed_probe_rows": 0,
            "actions": [],
        },
    }


def default_compiler_profiles():
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "build_config": load_persisted_build_config(),
        "runtime_config": load_persisted_runtime_config(),
        "profiles": [
            {
                "compiler_profile_id": "bmg.small_tile.default",
                "candidate_class": "small_tile",
                "description": "Default BMG profile for small tiles.",
                "selector": {"tile_m_max": 16, "sg_count_max": 8},
                "runtime_env_override": {},
            },
            {
                "compiler_profile_id": "bmg.medium_tile.default",
                "candidate_class": "medium_tile",
                "description": "Default BMG profile for medium tiles.",
                "selector": {"tile_m_min": 32, "tile_m_max": 64, "sg_count_max": 16},
                "runtime_env_override": {},
            },
            {
                "compiler_profile_id": "bmg.large_tile.default",
                "candidate_class": "large_tile",
                "description": "Default BMG profile for large tiles.",
                "selector": {"tile_m_min": 128, "sg_count_min": 16},
                "runtime_env_override": {},
            },
        ],
    }


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


def blocked(seed, constraints):
    """
    Seed-level filter using constraints JSON.
    
    Subgroup product gate (limits.valid_subgroup_sizes):
      [16,32]    → B70 default: sg_m × sg_n ∈ {16, 32}
      [32,64]    → B60:         sg_m × sg_n ∈ {32, 64}
      null       → legacy:      sg_m × sg_n ≤ 32
    """
    valid_sg_sizes = (constraints.get("limits") or {}).get("valid_subgroup_sizes")
    sg_product = seed["sg_m"] * seed["sg_n"]
    if valid_sg_sizes is not None:
        if sg_product not in valid_sg_sizes:
            return True
    elif sg_product > 32:
        return True
    allowed = constraints["allowed_values"]
    for key in ("tile_m", "tile_n", "tile_k", "sg_m", "sg_n", "stages", "split_k", "grf_mode"):
        if seed.get(key) not in allowed.get(key, []):
            return True
    for rule in constraints.get("blocked_rules", []):
        match = rule.get("match", {})
        if all(seed.get(name) == value for name, value in match.items()):
            return True
    return False


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
