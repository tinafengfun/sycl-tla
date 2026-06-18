#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

from __future__ import annotations

import csv
import math
from pathlib import Path

from .hw_specs import analyze_efficiency, resolve_hw_reference_spec
from .utils import now_iso, read_json, write_json


PRIORITY_STATE_SCHEMA_VERSION = "exact-shape-priority-v1"
DEFAULT_PRIORITY_STRATEGY = "b70_learned_sg_tiers_v3"
DEFAULT_PRIORITY_STATE_FILENAME = "exact_shape_priority_state.json"
WORKGROUP_SUBGROUP_WIDTH = 32
HARD_FILTER_SUBGROUPS = {"1x4", "1x8"}
LOW_PRIORITY_SCHEDULER_SUBGROUP = "2x8"
LOW_PRIORITY_GEMM_SUBGROUP = "2x4"
TIER_ORDER = {"high": 0, "normal": 1, "low": 2}
TIMEOUT_HARD_FILTER_FAMILIES = (
    {
        "name": "timeout_streamk_512x128x64_sg4x8_rcr",
        "scheduler": {"streamk"},
        "layouts": {"rcr"},
        "tile": (512, 128, 64),
        "subgroup": "4x8",
        "stages": {1, 2, 3},
    },
    {
        "name": "timeout_streamk_512x128x64_sg8x4_rcr",
        "scheduler": {"streamk"},
        "layouts": {"rcr"},
        "tile": (512, 128, 64),
        "subgroup": "8x4",
        "stages": {2, 3},
    },
)
TIMEOUT_LOW_PRIORITY_FAMILIES = (
    {
        "name": "timeout_prone_64x512x64_sg8x2",
        "scheduler": {"gemm", "data_parallel", "splitk"},
        "tile": (64, 512, 64),
        "subgroup": "8x2",
        "stages": {1, 2, 3},
    },
    {
        "name": "timeout_prone_gemm_512x64x64_sg4x4",
        "scheduler": {"gemm"},
        "tile": (512, 64, 64),
        "subgroup": "4x4",
        "stages": {3},
    },
    {
        "name": "timeout_prone_gemm_512x256x64_sg8x4",
        "scheduler": {"gemm"},
        "tile": (512, 256, 64),
        "subgroup": "8x4",
        "stages": set(),
    },
)

BUCKET_PRIORS = {
    "tiny_n": {
        "scheduler": {"gemm": 3.5, "data_parallel": 1.25, "splitk": 1.0, "streamk": 0.25},
        "wg": {512: 2.0, 1024: 2.0},
        "tile_k": {64: 2.5, 32: 1.0},
        "stages": {1: 2.0, 2: 1.5, 3: 0.5},
        "subgroup": {"8x4": 2.5, "4x4": 2.25, "4x8": 1.5, "2x8": 1.25, "8x2": 0.5},
        "orientation": {"balanced": 2.0, "n_wide": 1.0, "m_wide": 0.5},
        "out_per_work_item_target": 32.0,
    },
    "small_shape": {
        "scheduler": {"gemm": 3.0, "data_parallel": 1.5, "splitk": 1.25, "streamk": 0.5},
        "wg": {512: 2.5, 1024: 1.0},
        "tile_k": {32: 2.0, 64: 0.75},
        "stages": {3: 2.0, 2: 1.0},
        "subgroup": {"8x2": 2.5, "8x4": 0.75, "4x8": 0.75},
        "orientation": {"m_wide": 1.5, "balanced": 1.0, "n_wide": 0.5},
        "out_per_work_item_target": 48.0,
    },
    "skinny_n": {
        "scheduler": {"gemm": 3.0, "splitk": 1.75, "data_parallel": 1.5, "streamk": 1.0},
        "wg": {1024: 2.5, 512: 1.0},
        "tile_k": {64: 2.0, 32: 1.0},
        "stages": {2: 2.0, 3: 1.0},
        "subgroup": {"4x8": 2.0, "8x4": 1.5, "8x2": 0.5},
        "orientation": {"balanced": 2.0, "m_wide": 1.0, "n_wide": 0.5},
        "out_per_work_item_target": 16.0,
    },
    "huge_n": {
        "scheduler": {"gemm": 3.5, "data_parallel": 1.5, "splitk": 1.5, "streamk": 1.0},
        "wg": {1024: 2.5, 512: 0.75},
        "tile_k": {32: 2.0, 64: 0.75},
        "stages": {3: 2.0, 2: 1.0},
        "subgroup": {"4x8": 2.0, "8x4": 1.5, "8x2": 0.5},
        "orientation": {"n_wide": 2.5, "balanced": 0.75, "m_wide": 0.0},
        "out_per_work_item_target": 64.0,
    },
    "deep_k_mid_n": {
        "scheduler": {"data_parallel": 3.25, "splitk": 3.25, "gemm": 0.75, "streamk": 0.25},
        "wg": {512: 2.5, 1024: 1.0},
        "tile_k": {32: 2.0, 64: 0.75},
        "stages": {3: 2.0, 2: 0.75},
        "subgroup": {"8x2": 2.5, "8x4": 1.25, "4x8": 0.75},
        "orientation": {"m_wide": 2.25, "balanced": 1.0, "n_wide": 0.0},
        "out_per_work_item_target": 64.0,
    },
    "balanced": {
        "scheduler": {"gemm": 2.0, "data_parallel": 1.5, "splitk": 1.5, "streamk": 1.0},
        "wg": {1024: 2.0, 512: 1.5},
        "tile_k": {32: 1.5, 64: 1.5},
        "stages": {2: 1.25, 3: 1.25},
        "subgroup": {"4x8": 1.5, "8x4": 1.5, "8x2": 1.0},
        "orientation": {"balanced": 1.5, "m_wide": 1.0, "n_wide": 1.0},
        "out_per_work_item_target": 48.0,
    },
}


def default_exact_shape_priority_state():
    return {
        "schema_version": PRIORITY_STATE_SCHEMA_VERSION,
        "updated_at": "",
        "learned_runs": [],
        "bucket_stats": {},
    }


def load_exact_shape_priority_state(path):
    state_path = Path(path)
    if not state_path.exists():
        return default_exact_shape_priority_state()
    state = read_json(state_path)
    if state.get("schema_version") != PRIORITY_STATE_SCHEMA_VERSION:
        return default_exact_shape_priority_state()
    state.setdefault("updated_at", "")
    state.setdefault("learned_runs", [])
    state.setdefault("bucket_stats", {})
    return state


def write_exact_shape_priority_state(path, state):
    payload = dict(state)
    payload["schema_version"] = PRIORITY_STATE_SCHEMA_VERSION
    payload["updated_at"] = now_iso()
    write_json(Path(path), payload)


def _shape_tuple(shape):
    if isinstance(shape, dict):
        return int(shape["m"]), int(shape["n"]), int(shape["k"])
    if isinstance(shape, (tuple, list)) and len(shape) == 3:
        return int(shape[0]), int(shape[1]), int(shape[2])
    if isinstance(shape, str):
        parts = shape.lower().replace("x", "_").split("_")
        if len(parts) == 3 and all(part.strip("-").isdigit() for part in parts):
            return int(parts[0]), int(parts[1]), int(parts[2])
    raise ValueError(f"Unsupported shape payload: {shape!r}")


def classify_exact_shape_bucket(shape):
    m, n, k = _shape_tuple(shape)
    if n <= 384:
        return "tiny_n"
    if m <= 4096 and n <= 2048:
        return "small_shape"
    if n <= 512 and m >= 4096:
        return "skinny_n"
    if n >= max(32768, m * 2):
        return "huge_n"
    if k >= 8192 and n >= 4096:
        return "deep_k_mid_n"
    return "balanced"


def exact_shape_scheduler_family(entry):
    streamk_mode = str(entry.get("streamk_mode", "")).strip().lower()
    decomposition_mode = str(entry.get("decomposition_mode", "")).strip().lower()
    scheduler_family = str(entry.get("scheduler_family", entry.get("tile_scheduler", ""))).strip().lower()
    kernel_name = str(entry.get("kernel_name", entry.get("kernel", ""))).strip().lower()
    combined = " ".join((streamk_mode, decomposition_mode, scheduler_family, kernel_name))
    if "data_parallel" in combined or "dataparallel" in combined:
        return "data_parallel"
    if "splitk" in combined or "split_k" in combined:
        return "splitk"
    if "streamk" in combined or "stream_k" in combined:
        return "streamk"
    return "gemm"


def exact_shape_workgroup_size(entry):
    sg_m = int(entry.get("sg_m", 0) or 0)
    sg_n = int(entry.get("sg_n", 0) or 0)
    if sg_m <= 0 or sg_n <= 0:
        return 0
    return WORKGROUP_SUBGROUP_WIDTH * sg_m * sg_n


def exact_shape_subgroup(entry):
    return f"{int(entry.get('sg_m', 0) or 0)}x{int(entry.get('sg_n', 0) or 0)}"


def exact_shape_workgroup_subgroups(entry):
    wg = exact_shape_workgroup_size(entry)
    return max(wg // WORKGROUP_SUBGROUP_WIDTH, 1) if wg else 0


def exact_shape_tile_orientation(entry):
    tile_m = int(entry.get("tile_m", 0) or 0)
    tile_n = int(entry.get("tile_n", 0) or 0)
    if tile_m <= 0 or tile_n <= 0:
        return "unknown"
    if tile_n >= tile_m * 2:
        return "n_wide"
    if tile_m >= tile_n * 2:
        return "m_wide"
    return "balanced"


def exact_shape_out_per_work_item(entry):
    tile_m = int(entry.get("tile_m", 0) or 0)
    tile_n = int(entry.get("tile_n", 0) or 0)
    wg = exact_shape_workgroup_size(entry)
    if tile_m <= 0 or tile_n <= 0 or wg <= 0:
        return 0.0
    return (tile_m * tile_n) / float(wg)


def exact_shape_thread_wave_proxy(shape, entry, hw_spec):
    m, n, _ = _shape_tuple(shape)
    tile_m = int(entry.get("tile_m", 0) or 0)
    tile_n = int(entry.get("tile_n", 0) or 0)
    if tile_m <= 0 or tile_n <= 0:
        return math.inf
    total_tiles = math.ceil(m / tile_m) * math.ceil(n / tile_n)
    sg_per_wg = exact_shape_workgroup_subgroups(entry)
    concurrent_sgs_per_xe_core = max(int(hw_spec.get("concurrent_sgs_per_xe_core_256grf", 8)), 1)
    xe_cores = max(int(hw_spec.get("xe_cores", 32)), 1)
    concurrent_wg = max((xe_cores * concurrent_sgs_per_xe_core) / max(sg_per_wg, 1), 1.0)
    return total_tiles / concurrent_wg


def exact_shape_wave_fill_efficiency(shape, entry, hw_spec):
    m, n, _ = _shape_tuple(shape)
    tile_m = int(entry.get("tile_m", 0) or 0)
    tile_n = int(entry.get("tile_n", 0) or 0)
    if tile_m <= 0 or tile_n <= 0:
        return 0.0
    total_tiles = math.ceil(m / tile_m) * math.ceil(n / tile_n)
    sg_per_wg = exact_shape_workgroup_subgroups(entry)
    concurrent_sgs_per_xe_core = max(int(hw_spec.get("concurrent_sgs_per_xe_core_256grf", 8)), 1)
    xe_cores = max(int(hw_spec.get("xe_cores", 32)), 1)
    concurrent_wg = max((xe_cores * concurrent_sgs_per_xe_core) / max(sg_per_wg, 1), 1.0)
    waves = max(math.ceil(total_tiles / concurrent_wg), 1)
    return total_tiles / float(waves * concurrent_wg)


def _feature_keys(entry):
    return [
        f"scheduler:{exact_shape_scheduler_family(entry)}",
        f"wg:{exact_shape_workgroup_size(entry)}",
        f"tile_k:{int(entry.get('tile_k', 0) or 0)}",
        f"stages:{int(entry.get('stages', 0) or 0)}",
        f"subgroup:{exact_shape_subgroup(entry)}",
        f"orientation:{exact_shape_tile_orientation(entry)}",
        f"layout:{entry.get('layout', '')}",
        f"out_per_wi:{int(round(exact_shape_out_per_work_item(entry)))}",
    ]


def _learned_bonus(entry, bucket, state):
    bucket_stats = state.get("bucket_stats", {}).get(bucket)
    if not bucket_stats or bucket_stats.get("run_count", 0) <= 0:
        return 0.0
    feature_weights = bucket_stats.get("feature_weights", {})
    run_count = max(int(bucket_stats.get("run_count", 0)), 1)
    bonus = 0.0
    for key in _feature_keys(entry):
        bonus += feature_weights.get(key, 0.0) / run_count
    return bonus * 0.75


def _bucket_bonus(priors, category, key):
    return float(priors.get(category, {}).get(key, 0.0))


def _refer_seed_match(entry, shape):
    m, n, _ = _shape_tuple(shape)
    if exact_shape_scheduler_family(entry) != "gemm":
        return None
    tile = (
        int(entry.get("tile_m", 0) or 0),
        int(entry.get("tile_n", 0) or 0),
        int(entry.get("tile_k", 0) or 0),
    )
    subgroup = exact_shape_subgroup(entry)
    stages = int(entry.get("stages", 0) or 0)
    if n <= 384 and m > 256:
        if m >= 12288 and tile == (96, 128, 64) and subgroup == "4x4":
            return "refer_tiny_n_cfg16", 4.75 + (0.5 if stages == 2 else 0.0)
        if m >= 6144 and tile == (128, 128, 64) and subgroup == "4x4":
            return "refer_tiny_n_cfg10", 4.75 + (1.0 if stages == 1 else 0.0)
        if m >= 3072 and tile == (128, 128, 64) and subgroup == "8x4":
            return "refer_tiny_n_cfg11", 4.75 + (0.5 if stages == 2 else 0.0)
        if tile == (256, 192, 32) and subgroup == "8x4":
            return "refer_tiny_n_cfg18", 4.5 + (0.5 if stages == 2 else 0.0)
    if m <= 64:
        if n <= 3072 and tile == (64, 128, 64) and subgroup == "2x4":
            return "refer_small_m_64x128", 2.0
        if n > 3072 and tile == (64, 256, 64) and subgroup == "2x8":
            return "refer_small_m_64x256", 2.0
    if m <= 128:
        if n <= 3072 and tile == (128, 128, 64) and subgroup == "2x8":
            return "refer_small_m_128x128", 2.0
        if n > 3072 and tile == (128, 256, 64) and subgroup == "4x8":
            return "refer_small_m_128x256", 2.0
    if tile == (256, 256, 32) and subgroup == "8x4":
        return "refer_default_256x256", 1.5
    return None


def _timeout_family_match(entry, families):
    scheduler = exact_shape_scheduler_family(entry)
    subgroup = exact_shape_subgroup(entry)
    tile = (
        int(entry.get("tile_m", 0) or 0),
        int(entry.get("tile_n", 0) or 0),
        int(entry.get("tile_k", 0) or 0),
    )
    stages = int(entry.get("stages", 0) or 0)
    layout = str(entry.get("layout", "")).strip().lower()
    for family in families:
        if scheduler not in family["scheduler"]:
            continue
        if tile != family["tile"]:
            continue
        if subgroup != family["subgroup"]:
            continue
        if family.get("layouts") and layout not in family["layouts"]:
            continue
        if family.get("stages") and stages not in family["stages"]:
            continue
        return family["name"]
    return ""


def _preferred_high_priority_subgroups(bucket, scheduler):
    if bucket == "tiny_n":
        if scheduler == "gemm":
            return {"8x4", "4x4", "4x8"}
        return {"8x2", "8x4"}
    if bucket == "small_shape":
        if scheduler == "gemm":
            return {"8x2", "4x4", "4x8"}
        return {"8x2", "8x4"}
    if bucket == "skinny_n":
        if scheduler == "gemm":
            return {"4x8", "8x4", "4x4", "2x8"}
        return {"8x2", "8x4"}
    if bucket == "huge_n":
        if scheduler == "gemm":
            return {"4x8", "8x4", "2x8", "8x2"}
        return {"8x2", "8x4"}
    if bucket == "deep_k_mid_n":
        if scheduler in {"data_parallel", "splitk"}:
            return {"8x2", "8x4"}
        return {"8x2", "4x8"}
    return {"8x2", "8x4", "4x8"}


def classify_exact_shape_priority_rule(entry, shape):
    bucket = classify_exact_shape_bucket(shape)
    scheduler = exact_shape_scheduler_family(entry)
    subgroup = exact_shape_subgroup(entry)
    timeout_filter = _timeout_family_match(entry, TIMEOUT_HARD_FILTER_FAMILIES)
    timeout_low_priority = _timeout_family_match(entry, TIMEOUT_LOW_PRIORITY_FAMILIES)
    refer_seed = _refer_seed_match(entry, shape)
    if timeout_filter:
        return {
            "tier": "filtered",
            "reason": f"timeout_family_blacklist:{timeout_filter}",
            "bucket": bucket,
            "scheduler": scheduler,
            "subgroup": subgroup,
        }
    if subgroup in HARD_FILTER_SUBGROUPS:
        return {
            "tier": "filtered",
            "reason": f"hard_filter_subgroup:{subgroup}",
            "bucket": bucket,
            "scheduler": scheduler,
            "subgroup": subgroup,
        }
    if timeout_low_priority:
        return {
            "tier": "low",
            "reason": f"timeout_family_probation:{timeout_low_priority}",
            "bucket": bucket,
            "scheduler": scheduler,
            "subgroup": subgroup,
        }
    if refer_seed is not None:
        return {
            "tier": "high",
            "reason": f"refer_seed:{refer_seed[0]}",
            "bucket": bucket,
            "scheduler": scheduler,
            "subgroup": subgroup,
        }
    if scheduler == "gemm" and subgroup == LOW_PRIORITY_GEMM_SUBGROUP:
        return {
            "tier": "low",
            "reason": "low_priority_gemm_subgroup:2x4",
            "bucket": bucket,
            "scheduler": scheduler,
            "subgroup": subgroup,
        }
    if scheduler in {"data_parallel", "splitk", "streamk"} and subgroup == LOW_PRIORITY_SCHEDULER_SUBGROUP:
        return {
            "tier": "low",
            "reason": "low_priority_scheduler_subgroup:2x8",
            "bucket": bucket,
            "scheduler": scheduler,
            "subgroup": subgroup,
        }
    preferred = _preferred_high_priority_subgroups(bucket, scheduler)
    if subgroup in preferred:
        return {
            "tier": "high",
            "reason": f"preferred_bucket_scheduler_band:{bucket}:{scheduler}",
            "bucket": bucket,
            "scheduler": scheduler,
            "subgroup": subgroup,
        }
    return {
        "tier": "normal",
        "reason": f"fallback_band:{bucket}:{scheduler}",
        "bucket": bucket,
        "scheduler": scheduler,
        "subgroup": subgroup,
    }


def score_exact_shape_kernel(entry, shape, hw_spec=None, state=None):
    hw = dict(hw_spec or resolve_hw_reference_spec(device_arch="bmg", hw_spec_id="bmg_g31"))
    learning_state = state or default_exact_shape_priority_state()
    bucket = classify_exact_shape_bucket(shape)
    priors = BUCKET_PRIORS[bucket]
    analysis = analyze_efficiency({"m": _shape_tuple(shape)[0], "n": _shape_tuple(shape)[1], "k": _shape_tuple(shape)[2]}, entry, hw)
    scheduler = exact_shape_scheduler_family(entry)
    wg = exact_shape_workgroup_size(entry)
    tile_k = int(entry.get("tile_k", 0) or 0)
    stages = int(entry.get("stages", 0) or 0)
    subgroup = exact_shape_subgroup(entry)
    orientation = exact_shape_tile_orientation(entry)
    out_per_wi = exact_shape_out_per_work_item(entry)
    target_out = float(priors.get("out_per_work_item_target", 48.0))
    out_bonus = max(0.0, 2.5 - abs(out_per_wi - target_out) / 16.0)
    wave_proxy = exact_shape_thread_wave_proxy(shape, entry, hw)
    wave_fill_efficiency = exact_shape_wave_fill_efficiency(shape, entry, hw)
    wave_penalty = math.log2(max(wave_proxy, 1.0)) * 0.85
    tier_rule = classify_exact_shape_priority_rule(entry, shape)
    refer_seed = _refer_seed_match(entry, shape)
    score = 0.0
    score += analysis.get("max_expected_efficiency", 0.0) * 12.0
    score += analysis.get("wave_efficiency", 0.0) * 4.0
    score += wave_fill_efficiency * 3.0
    score += analysis.get("tile_efficiency", 0.0) * 2.0
    score += analysis.get("cross_xe_core_penalty", 0.0) * 2.0
    score += _bucket_bonus(priors, "scheduler", scheduler)
    score += _bucket_bonus(priors, "wg", wg)
    score += _bucket_bonus(priors, "tile_k", tile_k)
    score += _bucket_bonus(priors, "stages", stages)
    score += _bucket_bonus(priors, "subgroup", subgroup)
    score += _bucket_bonus(priors, "orientation", orientation)
    score += out_bonus
    score += _learned_bonus(entry, bucket, learning_state)
    if refer_seed is not None:
        score += refer_seed[1]
    score -= wave_penalty
    score -= max(analysis.get("xe_cores_per_wg", 1) - 1, 0) * 1.5
    if tier_rule["tier"] == "high":
        score += 1.75
    elif tier_rule["tier"] == "low":
        score -= 3.5
    if not analysis.get("slm_ok", True):
        score -= 6.0
    if analysis.get("grf_pressure", 0.0) > 1.0:
        score -= 4.0
    return {
        "score": score,
        "bucket": bucket,
        "scheduler": scheduler,
        "workgroup_size": wg,
        "out_per_work_item": out_per_wi,
        "wave_proxy": wave_proxy,
        "wave_fill_efficiency": wave_fill_efficiency,
        "analysis": analysis,
        "tier": tier_rule["tier"],
        "tier_reason": tier_rule["reason"],
        "subgroup": subgroup,
        "refer_seed": "" if refer_seed is None else refer_seed[0],
    }


def prioritize_exact_shape_kernels(entries, shapes, hw_spec=None, state=None, top_k=128):
    if not shapes:
        raise ValueError("shapes must not be empty")
    normalized_shapes = [{"m": _shape_tuple(shape)[0], "n": _shape_tuple(shape)[1], "k": _shape_tuple(shape)[2]} for shape in shapes]
    hw = dict(hw_spec or resolve_hw_reference_spec(device_arch="bmg", hw_spec_id="bmg_g31"))
    learning_state = state or default_exact_shape_priority_state()
    ranked = []
    filtered = []
    for entry in entries:
        shape_scores = [score_exact_shape_kernel(entry, shape, hw, learning_state) for shape in normalized_shapes]
        average_score = sum(item["score"] for item in shape_scores) / len(shape_scores)
        primary = shape_scores[0]
        ranked_entry = dict(entry)
        ranked_entry["priority_strategy"] = DEFAULT_PRIORITY_STRATEGY
        ranked_entry["priority_score"] = average_score
        ranked_entry["priority_bucket"] = primary["bucket"]
        ranked_entry["priority_scheduler"] = primary["scheduler"]
        ranked_entry["priority_subgroup"] = primary["subgroup"]
        ranked_entry["priority_wave_proxy"] = sum(item["wave_proxy"] for item in shape_scores) / len(shape_scores)
        ranked_entry["priority_wave_fill"] = sum(item["wave_fill_efficiency"] for item in shape_scores) / len(shape_scores)
        ranked_entry["priority_out_per_work_item"] = sum(item["out_per_work_item"] for item in shape_scores) / len(shape_scores)
        ranked_entry["priority_refer_seed"] = primary["refer_seed"]
        tier_ranks = [TIER_ORDER.get(item["tier"], len(TIER_ORDER)) for item in shape_scores if item["tier"] != "filtered"]
        if not tier_ranks:
            ranked_entry["priority_tier"] = "filtered"
            ranked_entry["priority_tier_rank"] = len(TIER_ORDER)
            ranked_entry["priority_filter_reason"] = primary["tier_reason"]
            filtered.append(ranked_entry)
            continue
        ranked_entry["priority_tier"] = primary["tier"]
        ranked_entry["priority_tier_rank"] = min(tier_ranks)
        ranked_entry["priority_filter_reason"] = ""
        ranked_entry["priority_tier_reason"] = primary["tier_reason"]
        ranked.append(ranked_entry)
    ranked.sort(key=lambda item: (item["priority_tier_rank"], -item["priority_score"], item["kernel_name"]))
    for index, entry in enumerate(ranked, start=1):
        entry["priority_rank"] = index
    filtered.sort(key=lambda item: (item["priority_filter_reason"], item["kernel_name"]))
    tier_counts = {"high": 0, "normal": 0, "low": 0, "filtered": len(filtered)}
    for entry in ranked:
        tier_counts[entry["priority_tier"]] += 1
    return {
        "strategy": DEFAULT_PRIORITY_STRATEGY,
        "shapes": normalized_shapes,
        "top_k": top_k,
        "input_entry_count": len(entries),
        "kept_entry_count": len(ranked),
        "filtered_entry_count": len(filtered),
        "tier_counts": tier_counts,
        "ranked_entries": ranked,
        "filtered_entries": [
            {
                "kernel_name": entry["kernel_name"],
                "priority_bucket": entry["priority_bucket"],
                "priority_scheduler": entry["priority_scheduler"],
                "priority_subgroup": entry["priority_subgroup"],
                "priority_filter_reason": entry["priority_filter_reason"],
                "tile_m": entry.get("tile_m", 0),
                "tile_n": entry.get("tile_n", 0),
                "tile_k": entry.get("tile_k", 0),
                "stages": entry.get("stages", 0),
            }
            for entry in filtered
        ],
        "top_candidates": [
            {
                "priority_rank": entry["priority_rank"],
                "kernel_name": entry["kernel_name"],
                "priority_score": entry["priority_score"],
                "priority_bucket": entry["priority_bucket"],
                "priority_scheduler": entry["priority_scheduler"],
                "priority_tier": entry["priority_tier"],
                "priority_tier_reason": entry["priority_tier_reason"],
                "priority_refer_seed": entry["priority_refer_seed"],
                "tile_m": entry.get("tile_m", 0),
                "tile_n": entry.get("tile_n", 0),
                "tile_k": entry.get("tile_k", 0),
                "sg_m": entry.get("sg_m", 0),
                "sg_n": entry.get("sg_n", 0),
                "stages": entry.get("stages", 0),
                "priority_wave_proxy": entry["priority_wave_proxy"],
                "priority_wave_fill": entry["priority_wave_fill"],
            }
            for entry in ranked[:top_k]
        ],
    }


def learn_exact_shape_priority_state(shape, merged_rows, state, run_signature):
    if run_signature in state.get("learned_runs", []):
        return state, {"run_signature": run_signature, "skipped": True, "reason": "already_learned"}
    ok_rows = [row for row in merged_rows if str(row.get("status", "")).upper() == "OK" and float(row.get("tflops", 0.0) or 0.0) > 0.0]
    if not ok_rows:
        return state, {"run_signature": run_signature, "skipped": True, "reason": "no_ok_rows"}
    bucket = classify_exact_shape_bucket(shape)
    bucket_stats = state.setdefault("bucket_stats", {}).setdefault(
        bucket,
        {"run_count": 0, "feature_weights": {}, "winner_history": []},
    )
    sorted_rows = sorted(ok_rows, key=lambda row: (-float(row["tflops"]), float(row.get("total_runtime_ms", 0.0) or 0.0)))
    best_tflops = float(sorted_rows[0]["tflops"])
    cutoff = best_tflops * 0.985
    positive_rows = [row for row in sorted_rows if float(row["tflops"]) >= cutoff]
    if len(positive_rows) < min(8, len(sorted_rows)):
        positive_rows = sorted_rows[: min(8, len(sorted_rows))]
    if len(positive_rows) > 32:
        positive_rows = positive_rows[:32]
    for row in positive_rows:
        weight = float(row["tflops"]) / best_tflops
        for key in _feature_keys(row):
            bucket_stats["feature_weights"][key] = bucket_stats["feature_weights"].get(key, 0.0) + weight
    bucket_stats["run_count"] += 1
    winner = sorted_rows[0]
    bucket_stats["winner_history"].append(
        {
            "run_signature": run_signature,
            "shape_tag": shape if isinstance(shape, str) else "_".join(str(x) for x in _shape_tuple(shape)),
            "kernel_name": winner["kernel_name"],
            "tflops": float(winner["tflops"]),
            "priority_rank": int(winner.get("priority_rank", 0) or 0),
            "priority_score": float(winner.get("priority_score", 0.0) or 0.0),
        }
    )
    state.setdefault("learned_runs", []).append(run_signature)
    summary = {
        "run_signature": run_signature,
        "bucket": bucket,
        "best_kernel": winner["kernel_name"],
        "best_tflops": float(winner["tflops"]),
        "winner_priority_rank": int(winner.get("priority_rank", 0) or 0),
        "positive_kernel_count": len(positive_rows),
        "top_kernel_names": [row["kernel_name"] for row in positive_rows[:8]],
    }
    return state, summary


def _read_shape_result_rows(run_dir, shape_tag):
    rows = []
    shape_dir = Path(run_dir) / "results" / shape_tag
    for csv_path in sorted(shape_dir.glob("*.csv")):
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row["result_csv"] = str(csv_path)
                row["shape_tag"] = shape_tag
                rows.append(row)
    return rows


def update_exact_shape_priority_state_from_run(run_dir, shape_tag, state_path, hw_spec=None):
    run_path = Path(run_dir)
    state = load_exact_shape_priority_state(state_path)
    kernel_metadata = read_json(run_path / "kernel_metadata.json")
    merged_rows = []
    for row in _read_shape_result_rows(run_path, shape_tag):
        kernel_name = row.get("kernel", "")
        metadata = kernel_metadata.get(kernel_name)
        if metadata is None:
            continue
        merged = dict(metadata)
        merged["kernel_name"] = kernel_name
        merged["tflops"] = float(row.get("tflops", 0.0) or 0.0)
        merged["status"] = row.get("status", "")
        merged["total_runtime_ms"] = float(row.get("total_runtime_ms", 0.0) or 0.0)
        merged["priority_rank"] = int(metadata.get("priority_rank", 0) or 0)
        merged["priority_score"] = float(metadata.get("priority_score", 0.0) or 0.0)
        merged_rows.append(merged)
    updated_state, summary = learn_exact_shape_priority_state(
        shape_tag,
        merged_rows,
        state,
        run_signature=f"{run_path.name}:{shape_tag}",
    )
    write_exact_shape_priority_state(state_path, updated_state)
    feedback_path = run_path / f"priority_feedback_{shape_tag}.json"
    payload = dict(summary)
    payload["state_path"] = str(state_path)
    payload["hw_spec_id"] = (hw_spec or {}).get("device_id", "bmg_g31")
    payload["updated_at"] = now_iso()
    write_json(feedback_path, payload)
    return payload
