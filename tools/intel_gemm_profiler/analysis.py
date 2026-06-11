#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

from .analysis_gap import (
    REGULAR_GEMM_FULL_CONFIG_FIELDS,
    SCHEDULER_BRUTEFORCE_CONFIG_FIELDS,
    build_regular_gemm_gap_scan,
    build_scheduler_bruteforce_gap_scan,
    collect_regular_gemm_full_config_rows,
    collect_scheduler_bruteforce_full_config_rows,
)
from .schemas import SCHEMA_VERSION
from .utils import now_iso


def count_by(items, field):
    counts = {}
    for item in items:
        value = item.get(field, "")
        if value == "":
            value = "<empty>"
        counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items()))


def build_scheduler_bruteforce_plan(candidate_space, args, build_manifest=None, candidate_build_plan=None):
    candidates = list(candidate_space.get("candidates", []))
    scheduler_candidates = [
        candidate
        for candidate in candidates
        if candidate.get("runner", "benchmark") == "benchmark" and candidate.get("streamk_mode")
    ]
    scheduler_bf16_candidates = [
        candidate for candidate in scheduler_candidates if candidate.get("dtype_a") == "bf16"
    ]
    regular_benchmark_candidates = [
        candidate
        for candidate in candidates
        if candidate.get("runner", "benchmark") == "benchmark" and not candidate.get("streamk_mode")
    ]
    enabled = bool(
        getattr(args, "search_strategy", "") == "bruteforce_scheduler"
        or getattr(args, "bruteforce_scheduler_search", False)
        or getattr(args, "kernel_catalog_source", "") == "layered_bmg_scheduler_expanded"
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "enabled": enabled,
        "search_strategy": getattr(args, "search_strategy", ""),
        "kernel_catalog_source": getattr(args, "kernel_catalog_source", ""),
        "constraint_source": candidate_space.get("constraint_source", ""),
        "design_summary": {
            "goal": "Preserve the regular GEMM universe while widening benchmark-backed BF16 scheduler search across legal subgroup and stage combinations.",
            "regular_gemm_space_preserved": getattr(args, "kernel_catalog_source", "") == "layered_bmg_scheduler_expanded",
            "scheduler_candidates_routed_through_preflight": bool(getattr(args, "use_candidate_build_preflight_benchmarks", False)),
            "prefilter_disabled": getattr(args, "prefilter", "none") == "none",
        },
        "execution_routing": {
            "run_candidate_build_preflight": bool(getattr(args, "run_candidate_build_preflight", False)),
            "use_candidate_build_preflight_benchmarks": bool(getattr(args, "use_candidate_build_preflight_benchmarks", False)),
            "build_candidate_benchmark": bool(getattr(args, "build_candidate_benchmark", False)),
            "candidate_build_batch_size": int(getattr(args, "candidate_build_batch_size", 0)),
            "candidate_build_parallelism": int(getattr(args, "candidate_build_parallelism", 0)),
            "aggregate_build_parallelism": int(candidate_build_plan.get("build_parallelism", 0)) if candidate_build_plan else 0,
            "preflight_build_parallelism": int(candidate_build_plan.get("batch_build_parallelism", 0)) if candidate_build_plan else 0,
            "prefilter": getattr(args, "prefilter", "none"),
            "skip_run": bool(getattr(args, "skip_run", False)),
            "dry_run": bool(getattr(args, "dry_run", False)),
        },
        "candidate_counts": {
            "total_candidates": len(candidates),
            "benchmark_candidates": sum(
                1 for candidate in candidates if candidate.get("runner", "benchmark") == "benchmark"
            ),
            "regular_benchmark_candidates": len(regular_benchmark_candidates),
            "scheduler_benchmark_candidates": len(scheduler_candidates),
            "scheduler_bf16_benchmark_candidates": len(scheduler_bf16_candidates),
            "selected_kernel_count": build_manifest.get("selected_kernel_count", 0) if build_manifest else 0,
            "selected_kernel_batch_count": len(build_manifest.get("selected_kernel_batches", [])) if build_manifest else 0,
            "preflight_batch_count": len(candidate_build_plan.get("batch_preflight_plans", [])) if candidate_build_plan else 0,
        },
        "scheduler_search_axes": {
            "layouts": sorted({candidate.get("layout", "") for candidate in scheduler_bf16_candidates}),
            "streamk_modes": sorted({candidate.get("streamk_mode", "") for candidate in scheduler_bf16_candidates}),
            "sg_layouts": [
                [sg_m, sg_n]
                for sg_m, sg_n in sorted(
                    {(candidate.get("sg_m", 0), candidate.get("sg_n", 0)) for candidate in scheduler_bf16_candidates}
                )
            ],
            "stages": sorted({int(candidate.get("stages", 0)) for candidate in scheduler_bf16_candidates}),
            "tile_shape_count": len(
                {
                    (candidate.get("layout", ""), candidate.get("tile_m", 0), candidate.get("tile_n", 0), candidate.get("tile_k", 0))
                    for candidate in scheduler_bf16_candidates
                }
            ),
            "candidate_count_by_layout": count_by(scheduler_bf16_candidates, "layout"),
            "candidate_count_by_streamk_mode": count_by(scheduler_bf16_candidates, "streamk_mode"),
            "candidate_count_by_decomposition_mode": count_by(scheduler_bf16_candidates, "decomposition_mode"),
            "candidate_count_by_stage": count_by(scheduler_bf16_candidates, "stages"),
            "candidate_count_by_sg": dict(
                sorted(
                    (
                        f"{sg_m}x{sg_n}",
                        sum(
                            1
                            for candidate in scheduler_bf16_candidates
                            if candidate.get("sg_m", 0) == sg_m and candidate.get("sg_n", 0) == sg_n
                        ),
                    )
                    for sg_m, sg_n in {
                        (candidate.get("sg_m", 0), candidate.get("sg_n", 0))
                        for candidate in scheduler_bf16_candidates
                    }
                )
            ),
        },
    }
