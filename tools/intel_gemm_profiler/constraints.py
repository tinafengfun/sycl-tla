#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import copy

from .constraints_probe import (
    apply_probe_results_to_profiles,
    apply_run_probe_constraints,
    apply_static_probe_constraints,
    blocked_rule_for_row,
)
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
