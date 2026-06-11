#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import csv
from pathlib import Path

from .analysis import (
    REGULAR_GEMM_FULL_CONFIG_FIELDS,
    SCHEDULER_BRUTEFORCE_CONFIG_FIELDS,
    build_regular_gemm_gap_scan,
    build_scheduler_bruteforce_gap_scan,
    build_scheduler_bruteforce_plan,
    collect_regular_gemm_full_config_rows,
    collect_scheduler_bruteforce_full_config_rows,
)
from .build_plan import (
    build_candidate_build_plan,
    detect_available_vcpus,
    resolve_candidate_build_jobs,
)
from .candidates import build_candidate_build_manifest
from .utils import write_json
def prepare_candidate_artifacts(
    args,
    workspace,
    reports_dir,
    candidate_space,
    profiles,
    constraints,
    *,
    build_manifest_fn=build_candidate_build_manifest,
    build_plan_fn=build_candidate_build_plan,
    detect_vcpus_fn=detect_available_vcpus,
    resolve_jobs_fn=resolve_candidate_build_jobs,
):
    build_manifest = build_manifest_fn(
        candidate_space,
        selected_kernel_batch_size=args.candidate_build_batch_size,
        build_config=profiles.get("build_config", {}),
    )
    selected_kernel_list_path = reports_dir / "selected_kernel_list.txt"
    selected_kernel_filter_path = reports_dir / "selected_kernel_filter.list"
    candidate_build_cmake_config_path = reports_dir / "candidate_build_cmake_config.json"
    candidate_build_plan_path = reports_dir / "candidate_build_plan.json"
    selected_kernel_list_path.write_text("\n".join(build_manifest["selected_kernel_list"]) + "\n", encoding="utf-8")
    selected_kernel_filter_path.write_text("\n".join(build_manifest["kernel_filter_file"]["lines"]) + "\n", encoding="utf-8")
    for batch in build_manifest.get("selected_kernel_batches", []):
        batch_filter_path = reports_dir / f"selected_kernel_filter_part{batch['batch_index']:03d}.list"
        batch_filter_path.write_text("\n".join(batch["kernel_filter_file"]["lines"]) + "\n", encoding="utf-8")
        batch["kernel_filter_path"] = str(batch_filter_path)
    build_manifest_path = reports_dir / "candidate_build_manifest.json"
    write_json(build_manifest_path, build_manifest)
    write_json(candidate_build_cmake_config_path, build_manifest["cmake_config"])

    source_dir = Path(args.cmake_source_dir).resolve() if args.cmake_source_dir else (Path(args.cwd).resolve() if args.cwd else Path.cwd().resolve())
    build_dir = Path(args.benchmark_build_dir).resolve() if args.benchmark_build_dir else workspace / "build" / "candidate_benchmarks"
    googlebenchmark_dir = Path(args.googlebenchmark_dir).resolve() if args.googlebenchmark_dir else None
    googlebenchmark_build_dir = Path(args.googlebenchmark_build_dir).resolve() if args.googlebenchmark_build_dir else None
    detected_vcpus = detect_vcpus_fn()
    candidate_build_workers = max(1, int(getattr(args, "candidate_build_parallelism", 1) or 1))
    aggregate_build_parallelism = detected_vcpus
    batch_build_parallelism = resolve_jobs_fn(candidate_build_workers, total_vcpus=detected_vcpus)
    candidate_build_plan = build_plan_fn(
        build_manifest,
        source_dir,
        build_dir,
        selected_kernel_filter_path,
        googlebenchmark_dir,
        googlebenchmark_build_dir,
        args.cmake_cxx_compiler,
        build_parallelism=aggregate_build_parallelism,
        batch_build_parallelism=batch_build_parallelism,
    )
    write_json(candidate_build_plan_path, candidate_build_plan)

    regular_gemm_full_config_path = reports_dir / "regular_gemm_full_config.csv"
    regular_gemm_gap_scan_path = reports_dir / "regular_gemm_gap_scan.json"
    regular_full_config_rows, regular_duplicate_rows = collect_regular_gemm_full_config_rows(candidate_space)
    with open(regular_gemm_full_config_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGULAR_GEMM_FULL_CONFIG_FIELDS)
        writer.writeheader()
        writer.writerows(regular_full_config_rows)
    write_json(
        regular_gemm_gap_scan_path,
        build_regular_gemm_gap_scan(
            regular_full_config_rows,
            constraints,
            duplicate_rows=regular_duplicate_rows,
        ),
    )

    scheduler_bruteforce_full_config_path = reports_dir / "scheduler_bruteforce_full_config.csv"
    scheduler_bruteforce_gap_scan_path = reports_dir / "scheduler_bruteforce_gap_scan.json"
    scheduler_full_config_rows, scheduler_duplicate_rows = collect_scheduler_bruteforce_full_config_rows(candidate_space)
    with open(scheduler_bruteforce_full_config_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCHEDULER_BRUTEFORCE_CONFIG_FIELDS)
        writer.writeheader()
        writer.writerows(scheduler_full_config_rows)
    write_json(
        scheduler_bruteforce_gap_scan_path,
        build_scheduler_bruteforce_gap_scan(
            scheduler_full_config_rows,
            duplicate_rows=scheduler_duplicate_rows,
        ),
    )

    scheduler_bruteforce_plan_path = reports_dir / "scheduler_bruteforce_plan.json"
    write_json(
        scheduler_bruteforce_plan_path,
        build_scheduler_bruteforce_plan(
            candidate_space,
            args,
            build_manifest=build_manifest,
            candidate_build_plan=candidate_build_plan,
        ),
    )
    return {
        "build_manifest": build_manifest,
        "build_manifest_path": build_manifest_path,
        "selected_kernel_list_path": selected_kernel_list_path,
        "selected_kernel_filter_path": selected_kernel_filter_path,
        "candidate_build_cmake_config_path": candidate_build_cmake_config_path,
        "candidate_build_plan_path": candidate_build_plan_path,
        "candidate_build_plan": candidate_build_plan,
        "candidate_build_workers": candidate_build_workers,
        "regular_gemm_full_config_path": regular_gemm_full_config_path,
        "regular_gemm_gap_scan_path": regular_gemm_gap_scan_path,
        "scheduler_bruteforce_full_config_path": scheduler_bruteforce_full_config_path,
        "scheduler_bruteforce_gap_scan_path": scheduler_bruteforce_gap_scan_path,
        "scheduler_bruteforce_plan_path": scheduler_bruteforce_plan_path,
    }
