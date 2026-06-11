#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

from .schemas import SCHEMA_VERSION
from .source_templates import is_valid_xe2_tile_sg
from .utils import now_iso


SCHEDULER_BRUTEFORCE_CONFIG_FIELDS = [
    "candidate_id",
    "kernel_id",
    "layout",
    "dtype_a",
    "dtype_b",
    "dtype_c",
    "dtype_d",
    "dtype_acc",
    "tile_m",
    "tile_n",
    "tile_k",
    "sg_m",
    "sg_n",
    "stages",
    "streamk_mode",
    "decomposition_mode",
    "reduction_mode",
    "kernel_schedule",
    "tile_scheduler",
    "runner",
]


REGULAR_GEMM_FULL_CONFIG_FIELDS = [
    "candidate_id",
    "kernel_id",
    "source",
    "layout",
    "dtype_a",
    "dtype_b",
    "dtype_c",
    "dtype_d",
    "dtype_acc",
    "tile_m",
    "tile_n",
    "tile_k",
    "sg_m",
    "sg_n",
    "stages",
    "kernel_schedule",
    "tile_scheduler",
    "runner",
]


def collect_scheduler_bruteforce_full_config_rows(candidate_space):
    scheduler_candidates = [
        candidate
        for candidate in candidate_space.get("candidates", [])
        if candidate.get("runner", "benchmark") == "benchmark"
        and candidate.get("streamk_mode")
        and candidate.get("dtype_a") == "bf16"
    ]
    rows = []
    duplicates = []
    seen = set()
    for candidate in scheduler_candidates:
        row = {field: candidate.get(field, "") for field in SCHEDULER_BRUTEFORCE_CONFIG_FIELDS}
        dedupe_key = tuple(row[field] for field in SCHEDULER_BRUTEFORCE_CONFIG_FIELDS if field != "candidate_id")
        if dedupe_key in seen:
            duplicates.append(row)
            continue
        seen.add(dedupe_key)
        rows.append(row)
    rows.sort(
        key=lambda row: (
            row["layout"],
            row["tile_m"],
            row["tile_n"],
            row["tile_k"],
            row["sg_m"],
            row["sg_n"],
            row["stages"],
            row["streamk_mode"],
        )
    )
    return rows, duplicates


def collect_regular_gemm_full_config_rows(candidate_space):
    regular_candidates = [
        candidate
        for candidate in candidate_space.get("candidates", [])
        if candidate.get("runner", "benchmark") == "benchmark"
        and not candidate.get("streamk_mode")
    ]
    rows = []
    duplicates = []
    seen = set()
    for candidate in regular_candidates:
        row = {field: candidate.get(field, "") for field in REGULAR_GEMM_FULL_CONFIG_FIELDS}
        dedupe_key = tuple(row[field] for field in REGULAR_GEMM_FULL_CONFIG_FIELDS if field != "candidate_id")
        if dedupe_key in seen:
            duplicates.append(row)
            continue
        seen.add(dedupe_key)
        rows.append(row)
    rows.sort(
        key=lambda row: (
            row["layout"],
            row["dtype_a"],
            row["tile_m"],
            row["tile_n"],
            row["tile_k"],
            row["sg_m"],
            row["sg_n"],
            row["stages"],
        )
    )
    return rows, duplicates


def build_scheduler_bruteforce_gap_scan(config_rows, duplicate_rows=None):
    duplicate_rows = duplicate_rows or []
    expected_modes = {"streamk", "data_parallel", "splitk"}
    grouped_modes = {}
    for row in config_rows:
        base_key = (
            row["layout"],
            row["dtype_a"],
            row["dtype_b"],
            row["dtype_c"],
            row["dtype_d"],
            row["dtype_acc"],
            row["tile_m"],
            row["tile_n"],
            row["tile_k"],
            row["sg_m"],
            row["sg_n"],
            row["stages"],
        )
        grouped_modes.setdefault(base_key, set()).add(row["streamk_mode"])

    incomplete_groups = []
    for base_key, modes in sorted(grouped_modes.items()):
        if modes != expected_modes:
            incomplete_groups.append(
                {
                    "layout": base_key[0],
                    "dtype_a": base_key[1],
                    "dtype_b": base_key[2],
                    "dtype_c": base_key[3],
                    "dtype_d": base_key[4],
                    "dtype_acc": base_key[5],
                    "tile_m": base_key[6],
                    "tile_n": base_key[7],
                    "tile_k": base_key[8],
                    "sg_m": base_key[9],
                    "sg_n": base_key[10],
                    "stages": base_key[11],
                    "present_modes": sorted(modes),
                    "missing_modes": sorted(expected_modes - modes),
                }
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "row_count": len(config_rows),
        "duplicate_rows_removed": len(duplicate_rows),
        "base_config_group_count": len(grouped_modes),
        "expected_modes_per_base_group": sorted(expected_modes),
        "incomplete_mode_group_count": len(incomplete_groups),
        "incomplete_mode_groups": incomplete_groups[:100],
    }


def build_regular_gemm_gap_scan(config_rows, constraints, duplicate_rows=None):
    duplicate_rows = duplicate_rows or []
    exhaustive_rows = [row for row in config_rows if row.get("source") == "exhaustive_regular_gemm_catalog"]
    actual_regular_stage_space = {
        (
            row["layout"],
            row["dtype_a"],
            row["dtype_b"],
            row["dtype_c"],
            row["dtype_d"],
            row["dtype_acc"],
            int(row["tile_m"]),
            int(row["tile_n"]),
            int(row["tile_k"]),
            int(row["sg_m"]),
            int(row["sg_n"]),
            int(row["stages"]),
        )
        for row in config_rows
        if int(row["stages"]) in (1, 2, 3)
    }
    signatures = sorted(
        {
            (
                row["layout"],
                row["dtype_a"],
                row["dtype_b"],
                row["dtype_c"],
                row["dtype_d"],
                row["dtype_acc"],
            )
            for row in exhaustive_rows
        }
    )
    allowed = (constraints or {}).get("allowed_values", {})
    limits = (constraints or {}).get("limits", {})
    valid_sg_sizes = limits.get("valid_subgroup_sizes")
    expected_exhaustive = set()
    for layout, dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc in signatures:
        for tile_m in allowed.get("tile_m", []):
            for tile_n in allowed.get("tile_n", []):
                for tile_k in allowed.get("tile_k", []):
                    for sg_m in allowed.get("sg_m", []):
                        for sg_n in allowed.get("sg_n", []):
                            if not is_valid_xe2_tile_sg(
                                (tile_m, tile_n, tile_k),
                                (sg_m, sg_n, 1),
                                sg_product_set=valid_sg_sizes,
                            ):
                                continue
                            for stage in [stage for stage in allowed.get("stages", []) if stage in (1, 2, 3)]:
                                expected_exhaustive.add(
                                    (
                                        layout,
                                        dtype_a,
                                        dtype_b,
                                        dtype_c,
                                        dtype_d,
                                        dtype_acc,
                                        tile_m,
                                        tile_n,
                                        tile_k,
                                        sg_m,
                                        sg_n,
                                        stage,
                                    )
                                )

    missing = sorted(expected_exhaustive - actual_regular_stage_space)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "row_count": len(config_rows),
        "duplicate_rows_removed": len(duplicate_rows),
        "exhaustive_regular_row_count": len(exhaustive_rows),
        "expected_exhaustive_config_count": len(expected_exhaustive),
        "actual_exhaustive_config_count": len(actual_regular_stage_space),
        "missing_exhaustive_config_count": len(missing),
        "missing_exhaustive_configs": [
            {
                "layout": item[0],
                "dtype_a": item[1],
                "dtype_b": item[2],
                "dtype_c": item[3],
                "dtype_d": item[4],
                "dtype_acc": item[5],
                "tile_m": item[6],
                "tile_n": item[7],
                "tile_k": item[8],
                "sg_m": item[9],
                "sg_n": item[10],
                "stages": item[11],
            }
            for item in missing[:100]
        ],
        "config_count_by_source": {
            str(source): sum(1 for row in config_rows if row.get("source", "") == source)
            for source in sorted({row.get("source", "") for row in config_rows})
        },
    }
