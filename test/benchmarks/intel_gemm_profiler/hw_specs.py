#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

"""Microbenchmark-calibrated hardware specifications for Intel GPU anomaly detection.

All values are derived from actual microbenchmark measurements on B60
(intel-b60-microbench repo REPORT.md), NOT from paper/datasheet specs.
"""

import copy
import json
import math
from pathlib import Path

_HW_REFERENCE_SPECS_JSON = Path(__file__).with_name("hw_reference_specs.json")

HW_REFERENCE_SPECS = {
    "bmg_g21": {
        # Topology
        "device_id": "bmg_g21",
        "device_name": "Intel Arc Pro B60",
        "xe_cores": 20,
        "eus_per_xe_core": 8,
        "xmx_engines": 160,
        "subgroup_size": 16,
        # Clock — confirmed by zeDeviceGetProperties, NOT 2850
        "clock_mhz": 2400,
        # DPAS/XMX microarchitecture (from microbenchmark)
        "dpas_flops_per_instruction": 4096,  # dpas.8x8: 8×16×16×2
        "dpas_latency_cycles": 33,  # Benchmark 1 slope method
        "dpas_reciprocal_throughput_cycles": 16.1,  # Benchmark 4 full-GPU
        "dpas_pipeline_depth": 2,
        "ilp_saturation_point": 14,
        # Peak compute (measured)
        "peak_xmx_tflops": 97.66,  # raw XMX, zero memory pressure
        "peak_gemm_bf16_tflops": 89.77,  # custom SYCL GEMM, 92% of raw
        "peak_gemm_efficiency": 0.92,
        "peak_bf16_tflops": 97.66,  # ceiling for anomaly detection
        "peak_f16_tflops": 97.66,  # same XMX path as BF16
        # Memory (GDDR6, NOT LPDDR5X!)
        "memory_type": "GDDR6",
        "memory_bus_width_bits": 256,
        "peak_memory_bw_gbps": 576,  # theoretical: 256-bit × 18Gbps / 8
        "measured_read_bw_gbps": 538,  # vectorized float4, 93% of theoretical
        "measured_write_bw_gbps": 442,
        # Cache & SLM (measured)
        "l1_data_cache_per_xe_core_kb": 128,
        "l1_latency_cycles_min": 71,
        "l2_cache_total_mb": 18,
        "l2_latency_cycles_min": 162,
        "global_memory_latency_cycles": 261,
        "slm_per_xe_core_kb": 64,  # ONLY 64 KB usable, not 128!
        "slm_latency_cycles_min": 46,
        # GRF
        "grf_registers_per_thread": 256,
        "grf_bytes_per_thread": 8192,
        "accumulator_grf_per_tile": 16,  # 8×16 float = 512B = 16 regs
        # Scheduling overhead (measured)
        "dispatch_fixed_overhead_ns": 3744,
        "dispatch_per_wg_overhead_ns": 40.1,
        "barrier_overhead_cycles_min": 2,
        "barrier_overhead_cycles_max": 11,
        "store_tile_overhead_cycles": 25,
        "alu_free_budget_during_dpas": 16,
        # 256-GRF mode: 1 concurrent thread per EU
        "concurrent_threads_per_eu_256grf": 1,
        # GEMM tuning guidance
        "optimal_barrier_interval": [4, 8],
        "optimal_sgs_per_wg": [4, 8],
    },
}

# Arch-string aliases to canonical spec keys
_ARCH_ALIASES = {
    "bmg": "bmg_g21",
    "bmg_g21": "bmg_g21",
    "xe20": "bmg_g21",
}


def get_hw_spec(device_arch: str) -> dict:
    """Return the hardware spec dict for *device_arch*.

    First tries to load from the JSON file alongside this module so that
    specs can be updated independently of the Python source.  Falls back to
    the hardcoded ``HW_REFERENCE_SPECS`` dictionary.

    Args:
        device_arch: architecture string, e.g. ``"bmg"`` or ``"bmg_g21"``.

    Returns:
        A copy of the matching spec dictionary.

    Raises:
        KeyError: if *device_arch* is not recognised.
    """
    canonical = _ARCH_ALIASES.get(device_arch.lower())
    if canonical is None:
        raise KeyError(f"Unknown device_arch {device_arch!r}. Known aliases: {sorted(_ARCH_ALIASES)}")

    # Try loading from the JSON file first
    try:
        with open(_HW_REFERENCE_SPECS_JSON, encoding="utf-8") as fh:
            from_json = json.load(fh)
        if canonical in from_json:
            return copy.deepcopy(from_json[canonical])
    except (OSError, json.JSONDecodeError, KeyError):
        pass

    return copy.deepcopy(HW_REFERENCE_SPECS[canonical])


def compute_efficiency_bounds(shape: dict, candidate: dict, hw_spec: dict) -> tuple:
    """Return ``(min_expected_efficiency, max_expected_efficiency)`` as fractions 0–1.

    The bounds are determined by a lightweight roofline + wave-utilisation
    model calibrated to the microbenchmark data in *hw_spec*.

    Args:
        shape:     shape dict with keys ``m``, ``n``, ``k``.
        candidate: candidate dict with keys ``tile_m``, ``tile_n``, ``tile_k``,
                   ``sg_m``, ``sg_n``, ``stages``.
        hw_spec:   hardware spec dict (e.g. from :func:`get_hw_spec`).

    Returns:
        ``(min_efficiency, max_efficiency)`` both in [0, 1].
    """
    m = shape["m"]
    n = shape["n"]
    k = shape["k"]
    tile_m = candidate["tile_m"]
    tile_n = candidate["tile_n"]
    tile_k = candidate.get("tile_k", 32)
    sg_m = candidate["sg_m"]
    sg_n = candidate["sg_n"]
    stages = candidate.get("stages", 2)

    xe_cores = hw_spec["xe_cores"]
    eus_per_xe_core = hw_spec["eus_per_xe_core"]
    slm_per_xe_core_kb = hw_spec["slm_per_xe_core_kb"]
    grf_bytes_per_thread = hw_spec["grf_bytes_per_thread"]
    peak_xmx_tflops = hw_spec["peak_xmx_tflops"]
    peak_gemm_efficiency = hw_spec["peak_gemm_efficiency"]
    measured_read_bw_gbps = hw_spec["measured_read_bw_gbps"]

    # 256-GRF mode: 1 concurrent thread per EU
    concurrent_sgs_per_xe_core = eus_per_xe_core  # = 8
    max_concurrent_sgs = xe_cores * concurrent_sgs_per_xe_core  # = 160

    sg_count = sg_m * sg_n
    max_concurrent_wg = max_concurrent_sgs // sg_count if sg_count > 0 else 0

    # Pathological: no wave can fully fill the GPU
    if max_concurrent_wg == 0:
        return (0.01, 0.10)

    # SLM overflow check — use dtype factor 2 bytes (BF16/F16)
    slm_needed_kb = (tile_m * tile_k + tile_n * tile_k) * 2 * stages / 1024
    if slm_needed_kb > slm_per_xe_core_kb:
        return (0.01, 0.10)

    # GRF accumulator check — 8×16 fp32 accumulators per sg (crude estimate)
    accum_bytes = tile_m * tile_n * 4 // sg_count
    if accum_bytes > grf_bytes_per_thread:
        return (0.01, 0.10)

    # Tile utilisation
    m_util = m / (math.ceil(m / tile_m) * tile_m)
    n_util = n / (math.ceil(n / tile_n) * tile_n)
    k_util = k / (math.ceil(k / tile_k) * tile_k)
    tile_util = m_util * n_util * k_util

    # Grid & wave analysis
    num_wg = math.ceil(m / tile_m) * math.ceil(n / tile_n)
    waves = math.ceil(num_wg / max_concurrent_wg)
    last_wave_wg = num_wg % max_concurrent_wg or max_concurrent_wg
    last_wave_util = last_wave_wg / max_concurrent_wg
    wave_util = (waves - 1 + last_wave_util) / waves if waves > 0 else 1.0

    # Cross-Xe-core penalty (WG spans multiple Xe-cores)
    xe_cores_per_wg = math.ceil(sg_count / concurrent_sgs_per_xe_core)
    cross_xe_penalty = 0.85 ** (xe_cores_per_wg - 1)

    base_efficiency = tile_util * wave_util * cross_xe_penalty

    # Roofline analysis
    # flops per element pair = 2 (multiply-add)
    flops = 2 * m * n * k
    # bytes read: A (m×k) + B (k×n), 2 bytes per element (BF16/F16)
    bytes_read = (m * k + k * n) * 2
    arithmetic_intensity = flops / bytes_read if bytes_read > 0 else float("inf")
    # Ridge point in flops/byte
    ridge_point = (peak_xmx_tflops * 1e12) / (measured_read_bw_gbps * 1e9)

    if arithmetic_intensity >= ridge_point:
        # Compute-bound
        min_eff = base_efficiency * 0.25
        max_eff = base_efficiency * peak_gemm_efficiency
    else:
        # Memory-bound
        ratio = arithmetic_intensity / ridge_point
        min_eff = ratio * 0.3
        max_eff = ratio * 1.1

    # Clamp to [0, 1]
    min_eff = max(0.0, min(1.0, min_eff))
    max_eff = max(0.0, min(1.0, max_eff))

    return (min_eff, max_eff)


def detect_probe_anomalies(probe_rows: list, shapes_doc: dict, candidate_space: dict, hw_spec: dict) -> dict:
    """Analyse probe results and flag anomalies relative to hardware specs.

    Runs two checks:

    1. **Spec-based**: compares ``actual_tflops / peak_tflops`` against the
       efficiency bounds returned by :func:`compute_efficiency_bounds`.
    2. **Cross-comparison**: detects cases where a larger tile on a larger M
       is slower than a smaller tile.

    Args:
        probe_rows:      list of probe result dicts (each has ``candidate_id``,
                         ``shape_id``, ``avg_tflops``, ``status``).
        shapes_doc:      shapes document dict.
        candidate_space: candidate space dict (``candidates`` list).
        hw_spec:         hardware spec dict from :func:`get_hw_spec`.

    Returns:
        Dict with keys ``anomalies``, ``auto_block_rules``, ``hw_spec_used``.
    """
    peak_tflops = hw_spec["peak_xmx_tflops"]
    eus_per_xe_core = hw_spec["eus_per_xe_core"]
    xe_cores = hw_spec["xe_cores"]
    concurrent_sgs_per_xe_core = eus_per_xe_core  # 256-GRF: 1 thread/EU

    shape_map = {shape["shape_id"]: shape for shape in shapes_doc.get("shapes", [])}
    candidate_map = {cand["candidate_id"]: cand for cand in candidate_space.get("candidates", [])}

    anomalies = []
    auto_block_rules = []

    passed_rows = [row for row in probe_rows if row.get("status") == "pass"]

    # --- Spec-based check ---
    for row in passed_rows:
        shape = shape_map.get(row.get("shape_id"))
        candidate = candidate_map.get(row.get("candidate_id"))
        if shape is None or candidate is None:
            continue

        try:
            actual_tflops = float(row.get("avg_tflops") or 0.0)
        except (TypeError, ValueError):
            continue

        actual_efficiency = actual_tflops / peak_tflops

        min_eff, max_eff = compute_efficiency_bounds(shape, candidate, hw_spec)

        anomaly_type = None
        if actual_efficiency < min_eff * 0.5:
            anomaly_type = "severely_below_spec"
        elif actual_efficiency < min_eff:
            anomaly_type = "below_spec"
        elif actual_efficiency > max_eff * 1.5:
            anomaly_type = "above_expected"

        if anomaly_type:
            sg_count = candidate.get("sg_m", 1) * candidate.get("sg_n", 4)
            hint = None
            if sg_count > concurrent_sgs_per_xe_core:
                hint = f"sg_count={sg_count} exceeds concurrent_sgs_per_xe_core={concurrent_sgs_per_xe_core}; WG spans multiple Xe-cores"

            anomaly = {
                "anomaly_type": anomaly_type,
                "candidate_id": row["candidate_id"],
                "shape_id": row["shape_id"],
                "actual_tflops": actual_tflops,
                "actual_efficiency": round(actual_efficiency, 6),
                "min_expected_efficiency": round(min_eff, 6),
                "max_expected_efficiency": round(max_eff, 6),
                "root_cause_hint": hint,
            }
            anomalies.append(anomaly)

            if anomaly_type in ("severely_below_spec", "cross_anomaly"):
                rule = {
                    "rule_id": f"anomaly.block.{row['candidate_id']}@{row['shape_id']}",
                    "source": "anomaly_detection",
                    "anomaly_type": anomaly_type,
                    "match": _extract_candidate_match(candidate),
                }
                auto_block_rules.append(rule)

    # --- Cross-comparison check ---
    # Build lookup: (dtype, tile_m) → list of (m, tflops, candidate_id, shape_id)
    by_tile: dict = {}
    for row in passed_rows:
        shape = shape_map.get(row.get("shape_id"))
        candidate = candidate_map.get(row.get("candidate_id"))
        if shape is None or candidate is None:
            continue
        try:
            tflops = float(row.get("avg_tflops") or 0.0)
        except (TypeError, ValueError):
            continue
        key = (candidate.get("dtype_a", ""), candidate["tile_m"])
        by_tile.setdefault(key, []).append({
            "m": shape["m"],
            "tflops": tflops,
            "candidate_id": row["candidate_id"],
            "shape_id": row["shape_id"],
        })

    # Check: larger tile_m on larger M should not be slower than smaller tile
    all_tile_m_vals = sorted({k[1] for k in by_tile})
    for idx, large_tile_m in enumerate(all_tile_m_vals):
        for small_tile_m in all_tile_m_vals[:idx]:
            for dtype in {k[0] for k in by_tile}:
                large_entries = by_tile.get((dtype, large_tile_m), [])
                small_entries = by_tile.get((dtype, small_tile_m), [])
                for lg in large_entries:
                    for sm in small_entries:
                        # Only compare when shape_m is larger for the large tile
                        if lg["m"] > sm["m"] and lg["tflops"] < sm["tflops"]:
                            anomaly = {
                                "anomaly_type": "cross_anomaly",
                                "candidate_id": lg["candidate_id"],
                                "shape_id": lg["shape_id"],
                                "compared_to_candidate_id": sm["candidate_id"],
                                "compared_to_shape_id": sm["shape_id"],
                                "actual_tflops": lg["tflops"],
                                "compared_tflops": sm["tflops"],
                                "root_cause_hint": f"tile_m={large_tile_m} on m={lg['m']} is slower than tile_m={small_tile_m} on m={sm['m']}",
                            }
                            anomalies.append(anomaly)
                            candidate = candidate_map.get(lg["candidate_id"])
                            if candidate is not None:
                                rule = {
                                    "rule_id": f"anomaly.block.{lg['candidate_id']}@{lg['shape_id']}",
                                    "source": "anomaly_detection",
                                    "anomaly_type": "cross_anomaly",
                                    "match": _extract_candidate_match(candidate),
                                }
                                # Avoid duplicate rules
                                existing_ids = {r["rule_id"] for r in auto_block_rules}
                                if rule["rule_id"] not in existing_ids:
                                    auto_block_rules.append(rule)

    return {
        "anomalies": anomalies,
        "auto_block_rules": auto_block_rules,
        "hw_spec_used": hw_spec.get("device_id", "unknown"),
    }


def _extract_candidate_match(candidate: dict) -> dict:
    """Extract the match dict for a blocked rule from a candidate dict."""
    match = {}
    for key in ("tile_m", "tile_n", "tile_k", "sg_m", "sg_n", "split_k"):
        if key in candidate:
            match[key] = candidate[key]
    return match
