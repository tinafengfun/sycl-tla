#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import os
from pathlib import Path

from .build_exec import (
    benchmark_batch_plan_by_kernel_id,
    benchmark_command_strings,
    benchmark_log_paths,
    execute_candidate_build_plan,
    execute_candidate_build_preflight_plans,
    run_entries_with_batch_benchmarks,
    validate_candidate_auto_build_mode,
)
from .utils import shell_join


def benchmark_exe_for_build_plan(build_dir, build_target):
    return str(Path(build_dir) / "benchmarks" / "gemm" / build_target)


def detect_available_vcpus():
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        return max(1, os.cpu_count() or 1)


def resolve_candidate_build_jobs(max_workers, total_vcpus=None):
    total = max(1, int(total_vcpus or detect_available_vcpus()))
    workers = max(1, int(max_workers or 1))
    return max(1, total // workers)


def candidate_build_commands(source_dir, build_dir, build_target, cmake_vars, build_parallelism=0):
    configure_command = ["cmake", "-S", str(source_dir), "-B", str(build_dir)]
    configure_command.extend(f"-D{name}={value}" for name, value in sorted(cmake_vars.items()))
    build_command = [
        "cmake",
        "--build",
        str(build_dir),
        "--target",
        build_target,
    ]
    if int(build_parallelism or 0) > 0:
        build_command.extend(["--parallel", str(int(build_parallelism))])
    else:
        build_command.append("--parallel")
    return configure_command, build_command


def build_batch_preflight_plans(build_manifest, source_dir, build_dir, base_cmake_vars, build_parallelism=0):
    cmake_config = build_manifest["cmake_config"]
    build_target = cmake_config["build_target"]
    kernel_filter_var = cmake_config["kernel_filter_cmake_var"]
    plans = []
    for batch in build_manifest.get("selected_kernel_batches", []):
        batch_filter_path = batch.get("kernel_filter_path", "")
        if not batch_filter_path:
            continue
        batch_build_dir = Path(build_dir) / "candidate_batch_preflight" / batch["batch_id"]
        batch_cmake_vars = dict(base_cmake_vars)
        batch_cmake_vars[kernel_filter_var] = batch_filter_path
        configure_command, build_command = candidate_build_commands(
            source_dir,
            batch_build_dir,
            build_target,
            batch_cmake_vars,
            build_parallelism=build_parallelism,
        )
        plans.append(
            {
                "schema_version": build_manifest["schema_version"],
                "generated_at": build_manifest["generated_at"],
                "build_target": build_target,
                "batch_id": batch["batch_id"],
                "batch_index": batch["batch_index"],
                "kernel_count": batch["kernel_count"],
                "selected_kernel_list": batch["selected_kernel_list"],
                "kernel_filter_file": batch_filter_path,
                "build_dir": str(batch_build_dir),
                "benchmark_exe": benchmark_exe_for_build_plan(batch_build_dir, build_target),
                "build_parallelism": int(build_parallelism or 0),
                "cmake_vars": batch_cmake_vars,
                "configure_command": configure_command,
                "build_command": build_command,
                "configure_command_line": shell_join(configure_command),
                "build_command_line": shell_join(build_command),
            }
        )
    return plans


def build_candidate_build_plan(
    build_manifest,
    source_dir,
    build_dir,
    kernel_filter_path,
    googlebenchmark_dir=None,
    googlebenchmark_build_dir=None,
    cmake_cxx_compiler="",
    build_parallelism=0,
    batch_build_parallelism=0,
):
    cmake_config = build_manifest["cmake_config"]
    cmake_vars = dict(cmake_config["cmake_vars"])
    kernel_filter_var = cmake_config["kernel_filter_cmake_var"]
    cmake_vars[kernel_filter_var] = str(kernel_filter_path)
    if googlebenchmark_dir:
        cmake_vars["GOOGLEBENCHMARK_DIR"] = str(googlebenchmark_dir)
    if googlebenchmark_build_dir:
        cmake_vars["GOOGLEBENCHMARK_BUILD_DIR"] = str(googlebenchmark_build_dir)
    if cmake_cxx_compiler:
        cmake_vars["CMAKE_CXX_COMPILER"] = cmake_cxx_compiler
    configure_command, build_command = candidate_build_commands(
        source_dir,
        build_dir,
        cmake_config["build_target"],
        cmake_vars,
        build_parallelism=build_parallelism,
    )
    return {
        "schema_version": build_manifest["schema_version"],
        "generated_at": build_manifest["generated_at"],
        "build_target": cmake_config["build_target"],
        "source_dir": str(source_dir),
        "build_dir": str(build_dir),
        "benchmark_exe": benchmark_exe_for_build_plan(build_dir, cmake_config["build_target"]),
        "kernel_filter_file": str(kernel_filter_path),
        "googlebenchmark_dir": str(googlebenchmark_dir) if googlebenchmark_dir else "",
        "googlebenchmark_build_dir": str(googlebenchmark_build_dir) if googlebenchmark_build_dir else "",
        "cmake_cxx_compiler": cmake_cxx_compiler,
        "build_parallelism": int(build_parallelism or 0),
        "batch_build_parallelism": int(batch_build_parallelism or 0),
        "selected_kernel_count": build_manifest["selected_kernel_count"],
        "selected_kernel_batch_size": build_manifest.get("selected_kernel_batch_size", 0),
        "selected_kernel_batches": build_manifest.get("selected_kernel_batches", []),
        "batch_preflight_plans": build_batch_preflight_plans(
            build_manifest,
            source_dir,
            build_dir,
            cmake_vars,
            build_parallelism=batch_build_parallelism,
        ),
        "cmake_vars": cmake_vars,
        "configure_command": configure_command,
        "build_command": build_command,
        "configure_command_line": shell_join(configure_command),
        "build_command_line": shell_join(build_command),
    }
