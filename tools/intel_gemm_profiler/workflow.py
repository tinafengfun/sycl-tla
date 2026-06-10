#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import copy
import csv
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    PACKAGE_ROOT = Path(__file__).resolve().parents[1]
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    from intel_gemm_profiler.catalog import SEED_KERNELS, build_kernel_catalog
    from intel_gemm_profiler.candidates import (
        build_candidate_build_manifest,
        build_compiler_profile_probe_entries,
        build_dpas_probe_entry,
        build_phase_a_probe_entries,
        build_screening_entries,
        default_shapes,
        dry_run_shapes,
        generate_candidate_space,
        generate_confirmation_entries,
    )
    from intel_gemm_profiler.constraints import (
        apply_probe_results_to_profiles,
        apply_run_probe_constraints,
        apply_static_probe_constraints,
        default_compiler_profiles,
        default_constraints,
        selected_compile_env,
        selected_runtime_env,
    )
    from intel_gemm_profiler.ali_dataset import build_ali_gemm_docs
    from intel_gemm_profiler.dispatch import DISPATCH_KEY_FIELDS, load_dispatch_table, lookup_dispatch_entry
    from intel_gemm_profiler.device_target import resolve_profiles_device_target
    from intel_gemm_profiler.hw_specs import resolve_hw_reference_spec
    from intel_gemm_profiler.phase_a import (
        build_compiler_flags_probe_summary,
        empty_anomaly_report,
        run_phase_a_probe,
    )
    from intel_gemm_profiler.runner import collect_environment_metadata, run_benchmark, run_entries_with_benchmark, run_entries_with_streamk_example
    from intel_gemm_profiler.selector import build_candidate_coverage_report, build_dispatch_table, build_phase_a_summary, build_phase_b_summary, build_reference_comparison, build_run_summary, write_results_csv
    from intel_gemm_profiler.source_templates import is_valid_xe2_tile_sg
    from intel_gemm_profiler.utils import ensure_dir, now_iso, read_json, resolve_executable, shell_init_with_env, shell_join, write_json
    from intel_gemm_profiler.schemas import SCHEMA_VERSION, SEARCH_RUNTIME_SCHEMA
else:
    from .catalog import SEED_KERNELS, build_kernel_catalog
    from .candidates import (
        build_candidate_build_manifest,
        build_compiler_profile_probe_entries,
        build_dpas_probe_entry,
        build_phase_a_probe_entries,
        build_screening_entries,
        default_shapes,
        dry_run_shapes,
        generate_candidate_space,
        generate_confirmation_entries,
    )
    from .constraints import (
        apply_probe_results_to_profiles,
        apply_run_probe_constraints,
        apply_static_probe_constraints,
        default_compiler_profiles,
        default_constraints,
        selected_compile_env,
        selected_runtime_env,
    )
    from .ali_dataset import build_ali_gemm_docs
    from .dispatch import DISPATCH_KEY_FIELDS, load_dispatch_table, lookup_dispatch_entry
    from .device_target import resolve_profiles_device_target
    from .hw_specs import resolve_hw_reference_spec
    from .phase_a import (
        build_compiler_flags_probe_summary,
        empty_anomaly_report,
        run_phase_a_probe,
    )
    from .runner import collect_environment_metadata, run_benchmark, run_entries_with_benchmark, run_entries_with_streamk_example
    from .selector import build_candidate_coverage_report, build_dispatch_table, build_phase_a_summary, build_phase_b_summary, build_reference_comparison, build_run_summary, write_results_csv
    from .source_templates import is_valid_xe2_tile_sg
    from .utils import ensure_dir, now_iso, read_json, resolve_executable, shell_init_with_env, shell_join, write_json
    from .schemas import SCHEMA_VERSION, SEARCH_RUNTIME_SCHEMA


def load_compiled_kernel_list(path):
    if not path:
        return None
    kernels = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        if item.startswith("^") and item.endswith("$"):
            item = item[1:-1]
        kernels.append(item)
    return kernels


def filter_candidate_space_by_compiled_kernels(candidate_space, compiled_kernels):
    if compiled_kernels is None:
        return candidate_space
    compiled = set(compiled_kernels)
    filtered = copy.deepcopy(candidate_space)
    filtered["candidates"] = [
        candidate for candidate in candidate_space["candidates"]
        if candidate.get("runner", "benchmark") != "benchmark" or candidate["kernel_id"] in compiled
    ]
    filtered["compiled_kernel_filter"] = {
        "source": "compiled_kernel_list",
        "kernel_count": len(compiled),
        "matched_candidate_count": len(filtered["candidates"]),
    }
    if candidate_space["candidates"] and not filtered["candidates"]:
        raise ValueError("Compiled kernel list does not match any generated benchmark candidates.")
    return filtered


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


def execute_candidate_build_plan(build_plan, log_dir, shell_init="", timeout=None, log_prefix="candidate_build"):
    ensure_dir(Path(log_dir))
    steps = [
        ("configure", build_plan["configure_command"], Path(log_dir) / f"{log_prefix}_configure.log"),
        ("build", build_plan["build_command"], Path(log_dir) / f"{log_prefix}.log"),
    ]
    results = []
    for step, command, log_path in steps:
        process, timed_out, timeout_reason = run_benchmark(command, log_path, shell_init=shell_init, timeout=timeout)
        status = "timeout" if timed_out else ("pass" if process.returncode == 0 else "fail")
        item = {
            "step": step,
            "status": status,
            "returncode": process.returncode,
            "command": shell_join(command),
            "log": str(log_path),
        }
        if timed_out:
            item["timeout_reason"] = timeout_reason
        results.append(item)
        if status != "pass":
            return {
                "schema_version": build_plan["schema_version"],
                "generated_at": build_plan["generated_at"],
                "status": status,
                "failure_step": step,
                "failure_reason": f"Candidate benchmark {step} failed with status {status}. See {log_path}.",
                "build_target": build_plan["build_target"],
                "benchmark_exe": build_plan["benchmark_exe"],
                "build_parallelism": int(build_plan.get("build_parallelism", 0)),
                "selected_kernel_count": build_plan.get("selected_kernel_count", ""),
                "kernel_filter_file": build_plan.get("kernel_filter_file", ""),
                "batch_id": build_plan.get("batch_id", ""),
                "steps": results,
            }
    return {
        "schema_version": build_plan["schema_version"],
        "generated_at": build_plan["generated_at"],
        "status": "pass",
        "build_target": build_plan["build_target"],
        "benchmark_exe": build_plan["benchmark_exe"],
        "selected_kernel_count": build_plan.get("selected_kernel_count", ""),
        "kernel_filter_file": build_plan.get("kernel_filter_file", ""),
        "batch_id": build_plan.get("batch_id", ""),
        "steps": results,
    }


def _batch_build_already_done(build_log_path, batch_plan):
    """Check if a batch was already successfully built (idempotent check)."""
    import os
    log = Path(build_log_path) if not isinstance(build_log_path, Path) else build_log_path
    exe = Path(batch_plan.get("benchmark_exe", ""))
    if not log.exists():
        return False
    if not exe.exists():
        # log exists but no executable → partial build, must rebuild
        return False
    # Check last 3 lines for build success marker
    try:
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-5:]
        for line in tail:
            if "Build succeeded" in line or "[100%] Built target" in line:
                return True
    except Exception:
        pass
    return False


def _load_preflight_progress(progress_path):
    """Load batch completion state from JSON progress file."""
    try:
        return read_json(progress_path).get("completed_batches", {})
    except Exception:
        return {}


def _save_preflight_progress(progress_path, completed_batches):
    """Atomically save batch completion state."""
    tmp = Path(str(progress_path) + ".tmp")
    write_json(tmp, {
        "schema_version": "1.0",
        "generated_at": now_iso(),
        "completed_batches": completed_batches,
    })
    os.replace(tmp, progress_path)


def execute_candidate_build_preflight_plans(build_plan, log_dir, shell_init="", timeout=None, max_workers=1, resume=False, progress_path=None):
    preflight_plans = build_plan.get("batch_preflight_plans", [])
    if not preflight_plans:
        return {
            "schema_version": build_plan["schema_version"],
            "generated_at": build_plan["generated_at"],
            "status": "not_run",
            "reason": "no batch_preflight_plans",
            "batch_count": 0,
            "batches": [],
        }
    # Sort plans by batch_index for predictable output
    plans_sorted = sorted(preflight_plans, key=lambda p: p["batch_index"])

    # Resume support: skip batches already built successfully
    resumed_batches = {}
    if resume and progress_path:
        progress_state = _load_preflight_progress(Path(progress_path))
        for plan in plans_sorted:
            bid = plan["batch_id"]
            log_path = Path(log_dir) / f"candidate_build_preflight_{bid}.log"
            # Check persisted progress first, then fallback to log inspection
            if progress_state.get(bid) == "pass" and _batch_build_already_done(log_path, plan):
                resumed_batches[plan["batch_index"]] = {
                    "batch_id": bid,
                    "batch_index": plan["batch_index"],
                    "kernel_count": plan["kernel_count"],
                    "status": "pass",
                    "resumed": True,
                }
        if resumed_batches:
            remaining = [p for p in plans_sorted if p["batch_index"] not in resumed_batches]
            print(f"  [resume] {len(resumed_batches)} batches already built, {len(remaining)} remaining")
            plans_sorted = remaining

    def _build_one(plan):
        summary = execute_candidate_build_plan(
            plan,
            log_dir,
            shell_init=shell_init,
            timeout=timeout,
            log_prefix=f"candidate_build_preflight_{plan['batch_id']}",
        )
        summary["batch_id"] = plan["batch_id"]
        summary["batch_index"] = plan["batch_index"]
        summary["kernel_count"] = plan["kernel_count"]
        return summary

    if max_workers <= 1:
        batches = [_build_one(plan) for plan in plans_sorted]
        # Save progress once after all builds complete
        if progress_path:
            all_done = dict(resumed_batches)
            for b in batches:
                if b.get("status") == "pass":
                    all_done[b["batch_id"]] = "pass"
            if all_done:
                _save_preflight_progress(Path(progress_path), all_done)
    else:
        import concurrent.futures
        import threading

        total_vcpus = detect_available_vcpus()
        cores_per_worker = resolve_candidate_build_jobs(max_workers, total_vcpus=total_vcpus)
        lock = threading.Lock()
        results_by_index = {}
        completed = [0]

        def _build_parallel(plan):
            result = _build_one(plan)
            with lock:
                results_by_index[plan["batch_index"]] = result
                completed[0] += 1
                if completed[0] % 50 == 0 or completed[0] <= 5:
                    passed = sum(1 for r in results_by_index.values() if r["status"] == "pass")
                    print(f"  [preflight] {completed[0]}/{len(plans_sorted)} batches ({passed} passed, {max_workers} workers)", flush=True)
            return result

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for plan in plans_sorted:
                futures.append(executor.submit(_build_parallel, plan))
            concurrent.futures.wait(futures)

        # Reassemble in batch_index order (merge resumed + newly built)
        batches = [results_by_index[i] for i in sorted(results_by_index)]
        # Prepend resumed batches (they go first in order)
        if resumed_batches:
            batches = [resumed_batches[i] for i in sorted(resumed_batches)] + batches

        # Save progress once after all builds complete (avoids race condition)
        if progress_path:
            all_done = dict(resumed_batches)
            for b in batches:
                if b.get("status") == "pass":
                    all_done[b["batch_id"]] = "pass"
            if all_done:
                _save_preflight_progress(Path(progress_path), all_done)

    failed = [batch for batch in batches if batch["status"] != "pass"]
    status = "pass" if not failed else ("timeout" if any(batch["status"] == "timeout" for batch in failed) else "fail")
    result = {
        "schema_version": build_plan["schema_version"],
        "generated_at": build_plan["generated_at"],
        "status": status,
        "batch_count": len(batches),
        "passed_batches": sum(1 for batch in batches if batch["status"] == "pass"),
        "failed_batches": len(failed),
        "failure_reason": failed[0].get("failure_reason", "") if failed else "",
        "batches": batches,
        "max_workers": max_workers,
        "resumed_batches": len(resumed_batches),
        "effective_build_parallelism": int(build_plan.get("batch_build_parallelism", 0) or cores_per_worker),
    }
    try:
        result["total_vcpus"] = total_vcpus
        result["cores_per_worker"] = cores_per_worker
    except NameError:
        result["total_vcpus"] = 1
        result["cores_per_worker"] = 1
    return result



def benchmark_batch_plan_by_kernel_id(build_plan):
    mapping = {}
    for plan in build_plan.get("batch_preflight_plans", []):
        for kernel_id in plan.get("selected_kernel_list", []):
            mapping[kernel_id] = plan
    return mapping


def batched_stage_path(path, batch_id):
    return path.with_name(f"{path.stem}_{batch_id}{path.suffix}")


def run_entries_with_batch_benchmarks(entries, config_path, manifest_path, log_path, batch_plan_by_kernel_id, cwd=None, shell_init=None, timeout=None, chunk_size=0):
    grouped = {}
    for entry in entries:
        kernel_id = entry["candidate"]["kernel_id"]
        plan = batch_plan_by_kernel_id.get(kernel_id)
        if plan is None:
            raise ValueError(f"No batch preflight benchmark plan found for kernel '{kernel_id}'.")
        grouped.setdefault(plan["batch_id"], {"plan": plan, "entries": []})["entries"].append(entry)
    rows = []
    commands = []
    log_paths = []
    for batch_id, item in sorted(grouped.items()):
        batch_log_path = batched_stage_path(log_path, batch_id)
        batch_rows, command = run_entries_with_benchmark(
            item["entries"],
            batched_stage_path(config_path, batch_id),
            batched_stage_path(manifest_path, batch_id),
            batch_log_path,
            item["plan"]["benchmark_exe"],
            cwd=cwd,
            shell_init=shell_init,
            timeout=timeout,
            chunk_size=chunk_size,
        )
        rows.extend(batch_rows)
        commands.extend(command if command and isinstance(command[0], (list, tuple)) else [command])
        log_paths.extend(benchmark_log_paths(batch_log_path, command))
    return rows, commands, log_paths


def validate_candidate_auto_build_mode(args, dry_run_mode, probe_mode):
    if not args.build_candidate_benchmark or dry_run_mode or args.skip_run or args.constraints_json:
        return
    if probe_mode not in {"auto", "run"}:
        return
    if resolve_executable(args.benchmark_exe, cwd=args.cwd):
        return
    raise ValueError(
        "--build-candidate-benchmark builds the generated benchmark after Phase A. "
        "Use --probe-mode=off or --constraints-json when no prebuilt --benchmark-exe is available for Phase A probes."
    )


SEARCH_STRATEGY_PRESETS = {
    "manual": {},
    "baseline": {
        "kernel_catalog_source": "persisted",
        "prefilter": "none",
        "run_candidate_build_preflight": False,
        "use_candidate_build_preflight_benchmarks": False,
    },
    "expanded_bmg": {
        "kernel_catalog_source": "expanded_bmg",
        "prefilter": "none",
        "run_candidate_build_preflight": False,
        "use_candidate_build_preflight_benchmarks": False,
    },
    "layered_exhaustive": {
        "kernel_catalog_source": "layered_bmg",
        "prefilter": "none",
        "run_candidate_build_preflight": False,
        "use_candidate_build_preflight_benchmarks": False,
    },
    "bruteforce_scheduler": {
        "kernel_catalog_source": "layered_bmg_scheduler_expanded",
        "prefilter": "none",
        "run_candidate_build_preflight": True,
        "use_candidate_build_preflight_benchmarks": True,
    },
}


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

    def _count_by(items, field):
        counts = {}
        for item in items:
            value = item.get(field, "")
            if value == "":
                value = "<empty>"
            counts[str(value)] = counts.get(str(value), 0) + 1
        return dict(sorted(counts.items()))

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
            "candidate_count_by_layout": _count_by(scheduler_bf16_candidates, "layout"),
            "candidate_count_by_streamk_mode": _count_by(scheduler_bf16_candidates, "streamk_mode"),
            "candidate_count_by_decomposition_mode": _count_by(scheduler_bf16_candidates, "decomposition_mode"),
            "candidate_count_by_stage": _count_by(scheduler_bf16_candidates, "stages"),
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


def apply_search_strategy_defaults(args):
    strategy = getattr(args, "search_strategy", "manual") or "manual"
    if getattr(args, "bruteforce_scheduler_search", False) and strategy == "manual":
        strategy = "bruteforce_scheduler"
    preset = SEARCH_STRATEGY_PRESETS.get(strategy, {})
    if preset:
        args.kernel_catalog_source = preset["kernel_catalog_source"]
        args.prefilter = preset["prefilter"]
        args.run_candidate_build_preflight = preset["run_candidate_build_preflight"]
        args.use_candidate_build_preflight_benchmarks = preset["use_candidate_build_preflight_benchmarks"]
    if strategy == "bruteforce_scheduler" and getattr(args, "candidate_build_batch_size", 0) <= 0:
        args.candidate_build_batch_size = 1
    if strategy == "bruteforce_scheduler" and (getattr(args, "skip_run", False) or getattr(args, "dry_run", False)):
        args.run_candidate_build_preflight = False
        args.use_candidate_build_preflight_benchmarks = False
    args.search_strategy = strategy
    return args


def apply_bruteforce_scheduler_search_defaults(args):
    args.bruteforce_scheduler_search = True
    return apply_search_strategy_defaults(args)


def load_target_shapes_and_reference(args, dry_run_mode):
    if args.ali_workbook:
        if args.shapes_json:
            raise ValueError("--ali-workbook and --shapes-json are mutually exclusive.")
        if args.reference_json:
            raise ValueError("--ali-workbook and --reference-json are mutually exclusive.")
        shapes_doc, reference_doc = build_ali_gemm_docs(args.ali_workbook)
        return limit_shapes_and_reference(shapes_doc, reference_doc, args.max_shapes)
    shapes_doc = read_json(args.shapes_json) if args.shapes_json else (dry_run_shapes(args.dtype) if dry_run_mode else default_shapes(args.dtype))
    reference_doc = read_json(args.reference_json) if args.reference_json else None
    return limit_shapes_and_reference(shapes_doc, reference_doc, args.max_shapes)


def limit_shapes_and_reference(shapes_doc, reference_doc=None, max_shapes=0):
    if max_shapes is None or max_shapes == 0:
        return shapes_doc, reference_doc
    if max_shapes < 0:
        raise ValueError("--max-shapes must be non-negative.")
    limited_shapes_doc = copy.deepcopy(shapes_doc)
    selected_shapes = limited_shapes_doc.get("shapes", [])[:max_shapes]
    limited_shapes_doc["shapes"] = selected_shapes
    limited_shapes_doc["shape_limit"] = max_shapes
    limited_shapes_doc["unlimited_shape_count"] = len(shapes_doc.get("shapes", []))
    if reference_doc is None:
        return limited_shapes_doc, None
    selected_shape_ids = {shape["shape_id"] for shape in selected_shapes}
    selected_shape_keys = {
        (shape.get("dtype_a"), shape.get("m"), shape.get("n"), shape.get("k"))
        for shape in selected_shapes
    }
    limited_reference_doc = copy.deepcopy(reference_doc)
    limited_reference_doc["entries"] = [
        entry for entry in limited_reference_doc.get("entries", [])
        if entry.get("shape_id") in selected_shape_ids
    ]
    limited_reference_doc["skipped_entries"] = [
        entry for entry in limited_reference_doc.get("skipped_entries", [])
        if (entry.get("dtype"), entry.get("m"), entry.get("n"), entry.get("k")) in selected_shape_keys
    ]
    limited_reference_doc["shape_limit"] = max_shapes
    limited_reference_doc["unlimited_reference_entries"] = len(reference_doc.get("entries", []))
    return limited_shapes_doc, limited_reference_doc


def benchmark_command_strings(command_or_commands):
    if not command_or_commands:
        return []
    if isinstance(command_or_commands[0], (list, tuple)):
        return [shell_join(command) for command in command_or_commands]
    return [shell_join(command_or_commands)]


def benchmark_log_paths(log_path, command_or_commands):
    if not command_or_commands or not isinstance(command_or_commands[0], (list, tuple)):
        return [str(log_path)]
    return [
        str(log_path.with_name(f"{log_path.stem}_part{chunk_index:03d}{log_path.suffix}"))
        for chunk_index in range(len(command_or_commands))
    ]


def artifact_record(name, path, purpose, required=True):
    if not path:
        return {
            "name": name,
            "path": "",
            "required": required,
            "exists": False,
            "purpose": purpose,
        }
    artifact_path = Path(path)
    exists = artifact_path.exists()
    return {
        "name": name,
        "path": str(artifact_path),
        "required": required,
        "exists": exists,
        "size_bytes": artifact_path.stat().st_size if exists else "",
        "sha256": file_sha256(artifact_path) if exists else "",
        "purpose": purpose,
    }


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_artifact_bundle_manifest(workspace, artifacts):
    required_artifacts = [
        artifact_record("gemm_target_shapes", artifacts["target_shapes"], "Requested exact GEMM shape set."),
        artifact_record("safe_search_constraints", artifacts["constraints"], "Safe Phase A / Phase B search boundary."),
        artifact_record("compiler_profiles", artifacts["compiler_profiles"], "Compiler and runtime profile presets."),
        artifact_record("kernel_catalog", artifacts["kernel_catalog"], "Candidate kernel catalog used for this run."),
        artifact_record("safe_candidates", artifacts["safe_candidates"], "Filtered buildable/searchable candidate space."),
        artifact_record("candidate_build_manifest", artifacts["build_manifest"], "Selected generated kernels and build metadata."),
        artifact_record("gemm_profile_results", artifacts["results_csv"], "Normalized benchmark/profile result rows."),
        artifact_record("gemm_dispatch_table", artifacts["dispatch_table"], "Selected exact-shape dispatch entries."),
        artifact_record("optimal_dispatch_table", artifacts["optimal_dispatch_table"], "Product-facing best dispatch artifact."),
        artifact_record("run_summary", artifacts["run_summary"], "Run row counts, commands, and logs."),
        artifact_record("phase_a_summary", artifacts["phase_a_summary"], "Probe and hardware constraint summary."),
        artifact_record("phase_b_summary", artifacts["phase_b_summary"], "Candidate/search/dispatch summary."),
    ]
    optional_artifacts = [
        artifact_record("reference_comparison", artifacts["reference_comparison"], "Optional reference-vs-dispatch comparison.", required=False),
        artifact_record("candidate_build_summary", artifacts["candidate_build_summary"], "Candidate benchmark aggregate build status.", required=False),
        artifact_record("candidate_build_preflight_summary", artifacts["candidate_build_preflight_summary"], "Per-batch candidate build preflight status.", required=False),
        artifact_record("scheduler_bruteforce_plan", artifacts["scheduler_bruteforce_plan"], "Scheduler brute-force search plan, search axes, and execution routing.", required=False),
        artifact_record("regular_gemm_full_config", artifacts["regular_gemm_full_config"], "Deduplicated full regular GEMM configuration list.", required=False),
        artifact_record("regular_gemm_gap_scan", artifacts["regular_gemm_gap_scan"], "Duplicate-removal and exhaustive-coverage scan for regular GEMM configs.", required=False),
        artifact_record("scheduler_bruteforce_full_config", artifacts["scheduler_bruteforce_full_config"], "Deduplicated full BF16 benchmark-backed scheduler configuration list.", required=False),
        artifact_record("scheduler_bruteforce_gap_scan", artifacts["scheduler_bruteforce_gap_scan"], "Duplicate-removal and missing-mode scan for the scheduler brute-force configuration list.", required=False),
        artifact_record("device_target_detection", artifacts["device_target_detection"], "Auto-detected SYCL device target used for candidate benchmark builds.", required=False),
        artifact_record("verified_hw_caps", artifacts["verified_hw_caps"], "Collected or probed hardware capability metadata.", required=False),
    ]
    lookup_args = [
        "python3",
        "test/benchmarks/intel_gemm_profiler.py",
        "--lookup-dispatch-table",
        artifacts["optimal_dispatch_table"],
        "--lookup-layout",
        "<layout>",
        "--lookup-dtype-a",
        "<dtype_a>",
        "--lookup-dtype-b",
        "<dtype_b>",
        "--lookup-dtype-c",
        "<dtype_c>",
        "--lookup-dtype-d",
        "<dtype_d>",
        "--lookup-dtype-acc",
        "<dtype_acc>",
        "--lookup-m",
        "<m>",
        "--lookup-n",
        "<n>",
        "--lookup-k",
        "<k>",
        "--lookup-batch-count",
        "<batch_count>",
        "--fallback-candidate-id",
        "<optional_fallback_candidate_id>",
    ]
    return {
        "schema_version": SEARCH_RUNTIME_SCHEMA["schema_version"],
        "generated_at": now_iso(),
        "bundle_id": f"intel_gemm_product_bundle_{Path(workspace).name}",
        "workspace": str(workspace),
        "required_artifacts": required_artifacts,
        "optional_artifacts": optional_artifacts,
        "missing_required_artifacts": [
            artifact["name"] for artifact in required_artifacts if not artifact["exists"]
        ],
        "missing_optional_artifacts": [
            artifact["name"] for artifact in optional_artifacts if not artifact["exists"]
        ],
        "runtime_lookup": {
            "dispatch_table": artifacts["optimal_dispatch_table"],
            "key_fields": list(DISPATCH_KEY_FIELDS),
            "cli_args_template": lookup_args,
            "cli_template": shell_join(lookup_args),
            "fallback_behavior": "Exact shape miss returns status=missing unless --fallback-candidate-id is set, in which case status=fallback is returned with reason=shape_not_found.",
        },
        "handoff_notes": [
            "Use optimal_dispatch_table.json as the product-facing dispatch artifact.",
            "Keep gemm_profile_results.csv and phase_b_summary.json with the dispatch table for auditability.",
            "Do not silently substitute a kernel on lookup miss; consume the explicit missing/fallback status.",
        ],
    }


def validate_product_bundle_manifest(bundle_manifest_or_path):
    bundle = read_json(bundle_manifest_or_path) if isinstance(bundle_manifest_or_path, (str, Path)) else bundle_manifest_or_path
    errors = []
    warnings = []
    required_artifacts = bundle.get("required_artifacts", [])
    optional_artifacts = bundle.get("optional_artifacts", [])
    if not isinstance(required_artifacts, list):
        errors.append("required_artifacts must be a list")
        required_artifacts = []
    if not isinstance(optional_artifacts, list):
        errors.append("optional_artifacts must be a list")
        optional_artifacts = []
    missing_required = []
    integrity_errors = []
    integrity_warnings = []

    def validate_artifact_integrity(artifact, mismatch_target):
        artifact_name = artifact.get("name", "")
        artifact_path = artifact.get("path", "")
        path = Path(artifact_path) if artifact_path else None
        if not path or not path.exists():
            return False
        expected_size = artifact.get("size_bytes", "")
        if expected_size != "":
            actual_size = path.stat().st_size
            if int(expected_size) != actual_size:
                mismatch_target.append(f"{artifact_name or artifact_path}: size_bytes expected {expected_size}, got {actual_size}")
        expected_sha256 = artifact.get("sha256", "")
        if expected_sha256:
            actual_sha256 = file_sha256(path)
            if expected_sha256 != actual_sha256:
                mismatch_target.append(f"{artifact_name or artifact_path}: sha256 mismatch")
        return True

    for artifact in required_artifacts:
        artifact_name = artifact.get("name", "")
        artifact_path = artifact.get("path", "")
        if not artifact_path or not Path(artifact_path).exists():
            missing_required.append(artifact_name or artifact_path or "<unnamed>")
        else:
            validate_artifact_integrity(artifact, integrity_errors)
    missing_optional = []
    for artifact in optional_artifacts:
        artifact_name = artifact.get("name", "")
        artifact_path = artifact.get("path", "")
        if not artifact_path or not Path(artifact_path).exists():
            missing_optional.append(artifact_name or artifact_path or "<unnamed>")
        else:
            validate_artifact_integrity(artifact, integrity_warnings)
    if missing_required:
        errors.append(f"missing required artifacts: {', '.join(missing_required)}")
    if missing_optional:
        warnings.append(f"missing optional artifacts: {', '.join(missing_optional)}")
    errors.extend(integrity_errors)
    warnings.extend(integrity_warnings)
    runtime_lookup = bundle.get("runtime_lookup", {})
    key_fields = runtime_lookup.get("key_fields")
    if key_fields != list(DISPATCH_KEY_FIELDS):
        errors.append("runtime_lookup.key_fields does not match dispatch key contract")
    dispatch_table_path = runtime_lookup.get("dispatch_table", "")
    dispatch_entry_count = 0
    if not dispatch_table_path:
        errors.append("runtime_lookup.dispatch_table is missing")
    elif not Path(dispatch_table_path).exists():
        errors.append(f"runtime lookup dispatch table does not exist: {dispatch_table_path}")
    else:
        try:
            dispatch_table = load_dispatch_table(dispatch_table_path)
            dispatch_entry_count = len(dispatch_table["entries"])
        except Exception as exc:
            errors.append(f"dispatch table validation failed: {exc}")
    cli_args_template = runtime_lookup.get("cli_args_template", [])
    if "--lookup-dispatch-table" not in cli_args_template:
        errors.append("runtime_lookup.cli_args_template is missing --lookup-dispatch-table")
    status = "fail" if errors else "pass"
    return {
        "schema_version": SEARCH_RUNTIME_SCHEMA["schema_version"],
        "generated_at": now_iso(),
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "required_artifact_count": len(required_artifacts),
        "optional_artifact_count": len(optional_artifacts),
        "missing_required_artifacts": missing_required,
        "missing_optional_artifacts": missing_optional,
        "integrity_errors": integrity_errors,
        "integrity_warnings": integrity_warnings,
        "dispatch_table": dispatch_table_path,
        "dispatch_entry_count": dispatch_entry_count,
    }


def export_product_bundle_manifest(bundle_manifest_or_path, output_dir):
    bundle = read_json(bundle_manifest_or_path)
    output_dir = ensure_dir(Path(output_dir))
    artifacts_dir = ensure_dir(output_dir / "artifacts")
    exported = copy.deepcopy(bundle)
    copied_by_name = {}

    def export_records(records):
        exported_records = []
        used_names = set()
        for artifact in records:
            source_path = Path(artifact.get("path", "")) if artifact.get("path", "") else None
            if not source_path or not source_path.exists():
                exported_records.append(
                    artifact_record(
                        artifact.get("name", ""),
                        "",
                        artifact.get("purpose", ""),
                        required=artifact.get("required", False),
                    )
                )
                continue
            filename = source_path.name
            if filename in used_names:
                filename = f"{artifact.get('name', source_path.stem)}_{source_path.name}"
            used_names.add(filename)
            destination = artifacts_dir / filename
            if source_path.resolve() != destination.resolve():
                shutil.copy2(source_path, destination)
            exported_record = artifact_record(
                artifact.get("name", ""),
                destination,
                artifact.get("purpose", ""),
                required=artifact.get("required", False),
            )
            exported_records.append(exported_record)
            copied_by_name[exported_record["name"]] = exported_record
        return exported_records

    exported["workspace"] = str(output_dir.resolve())
    exported["required_artifacts"] = export_records(bundle.get("required_artifacts", []))
    exported["optional_artifacts"] = export_records(bundle.get("optional_artifacts", []))
    exported["missing_required_artifacts"] = [
        artifact["name"] for artifact in exported["required_artifacts"] if not artifact["exists"]
    ]
    exported["missing_optional_artifacts"] = [
        artifact["name"] for artifact in exported["optional_artifacts"] if not artifact["exists"]
    ]
    optimal_dispatch = copied_by_name.get("optimal_dispatch_table")
    if optimal_dispatch:
        exported["runtime_lookup"]["dispatch_table"] = optimal_dispatch["path"]
        cli_args = list(exported["runtime_lookup"].get("cli_args_template", []))
        if "--lookup-dispatch-table" in cli_args:
            cli_args[cli_args.index("--lookup-dispatch-table") + 1] = optimal_dispatch["path"]
        exported["runtime_lookup"]["cli_args_template"] = cli_args
        exported["runtime_lookup"]["cli_template"] = shell_join(cli_args)
    exported_manifest_path = output_dir / "gemm_product_bundle_manifest.json"
    write_json(exported_manifest_path, exported)
    validation = validate_product_bundle_manifest(exported_manifest_path)
    return {
        "schema_version": SEARCH_RUNTIME_SCHEMA["schema_version"],
        "generated_at": now_iso(),
        "status": validation["status"],
        "source_manifest": str(Path(bundle_manifest_or_path)),
        "export_dir": str(output_dir.resolve()),
        "exported_manifest": str(exported_manifest_path),
        "artifact_count": len(exported["required_artifacts"]) + len(exported["optional_artifacts"]),
        "validation": validation,
    }


def workflow(args):
    if not args.workspace:
        raise ValueError("--workspace is required unless --lookup-dispatch-table is used.")
    workspace = ensure_dir(Path(args.workspace).resolve())
    inputs_dir = ensure_dir(workspace / "inputs")
    generated_dir = ensure_dir(workspace / "generated")
    configs_dir = ensure_dir(generated_dir / "configs")
    manifests_dir = ensure_dir(generated_dir / "manifests")
    logs_dir = ensure_dir(workspace / "logs")
    reports_dir = ensure_dir(workspace / "reports")
    profiles = read_json(args.compiler_profiles_json) if args.compiler_profiles_json else default_compiler_profiles()

    # --- Handle variant list/update operations ---
    from .constraints import (
        list_compile_variants,
        list_runtime_variants,
        update_build_config_variant,
        update_runtime_config_variant,
    )
    if args.list_compile_variants:
        import json as _json
        print(_json.dumps(list_compile_variants(), indent=2))
        return {}
    if args.list_runtime_variants:
        import json as _json
        print(_json.dumps(list_runtime_variants(), indent=2))
        return {}
    if args.update_compile_variant:
        update_build_config_variant(args.update_compile_variant)
        print(f"Updated build_config_bmg_perf.json selected_compile_variant → {args.update_compile_variant}")
    if args.update_runtime_variant:
        update_runtime_config_variant(args.update_runtime_variant)
        print(f"Updated runtime_config_bmg_perf.json selected_runtime_variant → {args.update_runtime_variant}")
    if args.update_compile_variant or args.update_runtime_variant:
        profiles = default_compiler_profiles()  # reload after update
    args = apply_search_strategy_defaults(args)
    profiles, device_target_detection = resolve_profiles_device_target(profiles, shell_init=args.shell_init)
    dry_run_mode = getattr(args, "dry_run", False)
    shapes_doc, reference_doc = load_target_shapes_and_reference(args, dry_run_mode)
    base_constraints = read_json(args.constraints_json) if args.constraints_json else default_constraints()
    top_k = min(args.top_k, 1) if dry_run_mode else args.top_k
    confirm_runs = 0 if dry_run_mode else args.confirm_runs
    probe_mode = "off" if dry_run_mode else args.probe_mode
    validate_candidate_auto_build_mode(args, dry_run_mode, probe_mode)
    probe_rows = []
    probe_logs = []
    probe_commands = []
    benchmark_commands = []
    base_runtime_shell_init = shell_init_with_env(args.shell_init, selected_runtime_env(profiles, variant_override=args.runtime_variant or None))
    compile_shell_init = shell_init_with_env(args.shell_init, selected_compile_env(profiles, variant_override=args.compile_variant or None))
    if args.constraints_json or probe_mode == "off":
        constraints = copy.deepcopy(base_constraints)
        env_caps = collect_environment_metadata(args.shell_init, args.benchmark_exe, args.streamk_example_exe, cwd=args.cwd)
        hw_spec = resolve_hw_reference_spec(
            constraints["device_arch"],
            getattr(args, "hw_spec_id", "") or device_target_detection.get("resolved_hw_spec_id", ""),
        )
        env_caps["probe_mode"] = "dry_run_off" if dry_run_mode else ("off" if probe_mode == "off" else "external_constraints")
        env_caps["device_target_detection"] = device_target_detection
        env_caps["hw_reference_spec_id"] = device_target_detection.get("resolved_hw_spec_id", hw_spec["device_id"])
        env_caps["hw_reference_spec"] = hw_spec
        env_caps["constraint_source"] = constraints["constraint_source"]
        env_caps["anomaly_report"] = empty_anomaly_report(hw_spec)
        env_caps["probe_results"] = []
        verified_hw_caps_path = reports_dir / "verified_hw_caps.json"
        write_json(verified_hw_caps_path, env_caps)
    else:
        constraints, env_caps, verified_hw_caps_path, probe_rows, probe_logs, probe_commands = run_phase_a_probe(args, shapes_doc, base_constraints, profiles, reports_dir, configs_dir, manifests_dir, logs_dir)
        profiles = apply_probe_results_to_profiles(profiles, env_caps.get("compiler_flags_probe", {}))
        env_caps["device_target_detection"] = device_target_detection
    allowed_runners = ("benchmark", "streamk_example") if env_caps["executables"].get("streamk_example_available") else ("benchmark",)
    write_json(inputs_dir / "safe_search_constraints.json", constraints)
    device_target_detection_path = reports_dir / "device_target_detection.json"
    write_json(device_target_detection_path, device_target_detection)
    write_json(inputs_dir / "compiler_profiles.json", profiles)
    write_json(inputs_dir / "gemm_target_shapes.json", shapes_doc)
    reference_doc_path = reports_dir / "ali_reference.json"
    if reference_doc is not None:
        write_json(reference_doc_path, reference_doc)
    write_json(inputs_dir / "search_runtime_schema.json", SEARCH_RUNTIME_SCHEMA)
    kernel_catalog = build_kernel_catalog(
        dtypes=sorted({shape["dtype_a"] for shape in shapes_doc["shapes"]}),
        allowed_runners=allowed_runners,
        catalog_path=Path(args.kernel_catalog_path) if args.kernel_catalog_path else None,
        catalog_source=args.kernel_catalog_source,
        generator_arch=args.generator_arch,
        generator_instantiation_level=args.generator_instantiation_level,
    )
    write_json(reports_dir / "kernel_catalog.json", kernel_catalog)
    candidate_space = generate_candidate_space(
        shapes_doc,
        constraints,
        profiles,
        allowed_runners=allowed_runners,
        catalog_path=Path(args.kernel_catalog_path) if args.kernel_catalog_path else None,
        catalog_source=args.kernel_catalog_source,
        generator_arch=args.generator_arch,
        generator_instantiation_level=args.generator_instantiation_level,
        prefilter_strategy=getattr(args, "prefilter", "none"),
    )
    candidate_space = filter_candidate_space_by_compiled_kernels(
        candidate_space,
        load_compiled_kernel_list(args.compiled_kernel_list),
    )
    write_json(reports_dir / "gemm_candidate_space.json", candidate_space)
    write_json(reports_dir / "bmg_safe_candidates.json", candidate_space)
    candidate_coverage_report_path = reports_dir / "candidate_coverage_report.json"
    write_json(candidate_coverage_report_path, build_candidate_coverage_report(candidate_space))
    build_manifest = build_candidate_build_manifest(
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
    write_json(reports_dir / "candidate_build_manifest.json", build_manifest)
    write_json(candidate_build_cmake_config_path, build_manifest["cmake_config"])
    source_dir = Path(args.cmake_source_dir).resolve() if args.cmake_source_dir else (Path(args.cwd).resolve() if args.cwd else Path.cwd().resolve())
    build_dir = Path(args.benchmark_build_dir).resolve() if args.benchmark_build_dir else workspace / "build" / "candidate_benchmarks"
    googlebenchmark_dir = Path(args.googlebenchmark_dir).resolve() if args.googlebenchmark_dir else None
    googlebenchmark_build_dir = (
        Path(args.googlebenchmark_build_dir).resolve() if args.googlebenchmark_build_dir else None
    )
    detected_vcpus = detect_available_vcpus()
    candidate_build_workers = max(1, int(getattr(args, "candidate_build_parallelism", 1) or 1))
    aggregate_build_parallelism = detected_vcpus
    batch_build_parallelism = resolve_candidate_build_jobs(candidate_build_workers, total_vcpus=detected_vcpus)
    candidate_build_plan = build_candidate_build_plan(
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
    regular_gap_scan = build_regular_gemm_gap_scan(
        regular_full_config_rows,
        constraints,
        duplicate_rows=regular_duplicate_rows,
    )
    write_json(regular_gemm_gap_scan_path, regular_gap_scan)
    scheduler_bruteforce_full_config_path = reports_dir / "scheduler_bruteforce_full_config.csv"
    scheduler_bruteforce_gap_scan_path = reports_dir / "scheduler_bruteforce_gap_scan.json"
    scheduler_full_config_rows, scheduler_duplicate_rows = collect_scheduler_bruteforce_full_config_rows(candidate_space)
    with open(scheduler_bruteforce_full_config_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCHEDULER_BRUTEFORCE_CONFIG_FIELDS)
        writer.writeheader()
        writer.writerows(scheduler_full_config_rows)
    scheduler_gap_scan = build_scheduler_bruteforce_gap_scan(
        scheduler_full_config_rows,
        duplicate_rows=scheduler_duplicate_rows,
    )
    write_json(scheduler_bruteforce_gap_scan_path, scheduler_gap_scan)
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
    candidate_build_summary_path = reports_dir / "candidate_build_summary.json"
    candidate_build_preflight_summary_path = reports_dir / "candidate_build_preflight_summary.json"
    candidate_build_summary = {"status": "not_run", "reason": "build_candidate_benchmark disabled"}
    candidate_build_preflight_summary = {"status": "not_run", "reason": "run_candidate_build_preflight disabled"}
    build_timeout = args.build_timeout or args.timeout
    effective_benchmark_exe = args.benchmark_exe
    if args.run_candidate_build_preflight:
        candidate_build_preflight_summary = execute_candidate_build_preflight_plans(
            candidate_build_plan,
            logs_dir,
            shell_init=compile_shell_init,
            timeout=build_timeout,
            max_workers=candidate_build_workers,
            resume=getattr(args, "resume_candidate_build_preflight", False),
            progress_path=str(reports_dir / "preflight_progress.json"),
        )
        write_json(candidate_build_preflight_summary_path, candidate_build_preflight_summary)
        if candidate_build_preflight_summary.get("status") not in {"pass", "not_run"}:
            raise RuntimeError(candidate_build_preflight_summary["failure_reason"])
    else:
        write_json(candidate_build_preflight_summary_path, candidate_build_preflight_summary)
    if args.use_candidate_build_preflight_benchmarks and candidate_build_preflight_summary.get("status") != "pass":
        raise ValueError("--use-candidate-build-preflight-benchmarks requires successful --run-candidate-build-preflight.")
    if args.build_candidate_benchmark:
        candidate_build_summary = execute_candidate_build_plan(
            candidate_build_plan,
            logs_dir,
            shell_init=compile_shell_init,
            timeout=build_timeout,
        )
        write_json(candidate_build_summary_path, candidate_build_summary)
        if candidate_build_summary.get("status") != "pass":
            raise RuntimeError(candidate_build_summary["failure_reason"])
        effective_benchmark_exe = candidate_build_plan["benchmark_exe"]
        env_caps["executables"]["benchmark_exe"] = effective_benchmark_exe
        env_caps["executables"]["benchmark_available"] = True
        env_caps["candidate_build_summary"] = candidate_build_summary
        write_json(verified_hw_caps_path, env_caps)
    else:
        write_json(candidate_build_summary_path, candidate_build_summary)
    screening_entries = build_screening_entries(shapes_doc, candidate_space)
    all_rows = list(probe_rows)
    log_paths = list(probe_logs)
    benchmark_commands.extend(probe_commands)
    if candidate_build_summary.get("status") == "pass":
        log_paths.extend(step["log"] for step in candidate_build_summary["steps"])
        benchmark_commands.extend(step["command"] for step in candidate_build_summary["steps"])
    if candidate_build_preflight_summary.get("status") == "pass":
        for batch in candidate_build_preflight_summary["batches"]:
            for step in batch.get("steps", []):
                log_paths.append(step.get("log", ""))
                benchmark_commands.append(step.get("command", ""))
    batch_plan_by_kernel = benchmark_batch_plan_by_kernel_id(candidate_build_plan) if args.use_candidate_build_preflight_benchmarks else {}
    if not args.skip_run:
        screening_benchmark_entries = [entry for entry in screening_entries if entry["candidate"].get("runner", "benchmark") == "benchmark"]
        screening_streamk_entries = [entry for entry in screening_entries if entry["candidate"].get("runner") == "streamk_example"]
        screening_rows = []
        if screening_benchmark_entries:
            screening_log = logs_dir / "screening.log"
            if args.use_candidate_build_preflight_benchmarks:
                rows, command, batch_logs = run_entries_with_batch_benchmarks(screening_benchmark_entries, configs_dir / "screening.in", manifests_dir / "screening_manifest.json", screening_log, batch_plan_by_kernel, cwd=args.cwd, shell_init=base_runtime_shell_init, timeout=args.timeout, chunk_size=args.benchmark_entry_chunk_size)
            else:
                rows, command = run_entries_with_benchmark(screening_benchmark_entries, configs_dir / "screening.in", manifests_dir / "screening_manifest.json", screening_log, effective_benchmark_exe, cwd=args.cwd, shell_init=base_runtime_shell_init, timeout=args.timeout, chunk_size=args.benchmark_entry_chunk_size)
                batch_logs = benchmark_log_paths(screening_log, command)
            screening_rows.extend(rows)
            log_paths.extend(batch_logs)
            benchmark_commands.extend(benchmark_command_strings(command))
        if screening_streamk_entries:
            rows, commands = run_entries_with_streamk_example(screening_streamk_entries, logs_dir, args.streamk_example_exe, cwd=args.cwd, shell_init=base_runtime_shell_init, timeout=args.timeout)
            screening_rows.extend(rows)
            log_paths.extend(str(logs_dir / f"{entry['bm_name']}.log") for entry in screening_streamk_entries)
            benchmark_commands.extend(commands)
        all_rows.extend(screening_rows)
        if confirm_runs > 0:
            confirm_entries = generate_confirmation_entries(screening_rows, candidate_space, shapes_doc, top_k=top_k, confirm_runs=confirm_runs)
            if confirm_entries:
                confirm_benchmark_entries = [entry for entry in confirm_entries if entry["candidate"].get("runner", "benchmark") == "benchmark"]
                confirm_streamk_entries = [entry for entry in confirm_entries if entry["candidate"].get("runner") == "streamk_example"]
                confirm_rows = []
                if confirm_benchmark_entries:
                    confirm_log = logs_dir / "confirm.log"
                    if args.use_candidate_build_preflight_benchmarks:
                        rows, command, batch_logs = run_entries_with_batch_benchmarks(confirm_benchmark_entries, configs_dir / "confirm.in", manifests_dir / "confirm_manifest.json", confirm_log, batch_plan_by_kernel, cwd=args.cwd, shell_init=base_runtime_shell_init, timeout=args.timeout, chunk_size=args.benchmark_entry_chunk_size)
                    else:
                        rows, command = run_entries_with_benchmark(confirm_benchmark_entries, configs_dir / "confirm.in", manifests_dir / "confirm_manifest.json", confirm_log, effective_benchmark_exe, cwd=args.cwd, shell_init=base_runtime_shell_init, timeout=args.timeout, chunk_size=args.benchmark_entry_chunk_size)
                        batch_logs = benchmark_log_paths(confirm_log, command)
                    confirm_rows.extend(rows)
                    log_paths.extend(batch_logs)
                    benchmark_commands.extend(benchmark_command_strings(command))
                if confirm_streamk_entries:
                    rows, commands = run_entries_with_streamk_example(confirm_streamk_entries, logs_dir, args.streamk_example_exe, cwd=args.cwd, shell_init=base_runtime_shell_init, timeout=args.timeout)
                    confirm_rows.extend(rows)
                    log_paths.extend(str(logs_dir / f"{entry['bm_name']}.log") for entry in confirm_streamk_entries)
                    benchmark_commands.extend(commands)
                all_rows.extend(confirm_rows)
    write_results_csv(all_rows, reports_dir / "gemm_profile_results.csv")
    dispatch_table = build_dispatch_table(
        all_rows,
        shapes_doc,
        top_k=top_k,
        confirm_runs=confirm_runs,
        close_call_threshold=args.close_call_threshold,
        candidate_space=candidate_space,
        hw_spec=env_caps.get("hw_reference_spec"),
    )
    write_json(reports_dir / "gemm_dispatch_table.json", dispatch_table)
    write_json(reports_dir / "optimal_dispatch_table.json", dispatch_table)
    if reference_doc is not None:
        write_json(
            reports_dir / "reference_comparison.json",
            build_reference_comparison(dispatch_table, reference_doc),
        )
    summary = build_run_summary(all_rows, dispatch_table, benchmark_commands, log_paths)
    write_json(reports_dir / "run_summary.json", summary)
    phase_a_summary_path = reports_dir / "phase_a_summary.json"
    phase_b_summary_path = reports_dir / "phase_b_summary.json"
    run_summary_path = reports_dir / "run_summary.json"
    write_json(phase_a_summary_path, build_phase_a_summary(env_caps, constraints, probe_rows))
    write_json(phase_b_summary_path, build_phase_b_summary(candidate_space, dispatch_table, summary))
    outputs = {
        "workspace": str(workspace),
        "search_runtime_schema": str(inputs_dir / "search_runtime_schema.json"),
        "target_shapes": str(inputs_dir / "gemm_target_shapes.json"),
        "constraints": str(inputs_dir / "safe_search_constraints.json"),
        "compiler_profiles": str(inputs_dir / "compiler_profiles.json"),
        "kernel_catalog": str(reports_dir / "kernel_catalog.json"),
        "candidate_space": str(reports_dir / "gemm_candidate_space.json"),
        "candidate_coverage_report": str(candidate_coverage_report_path),
        "build_manifest": str(reports_dir / "candidate_build_manifest.json"),
        "selected_kernel_list": str(selected_kernel_list_path),
        "selected_kernel_filter": str(selected_kernel_filter_path),
        "candidate_build_cmake_config": str(candidate_build_cmake_config_path),
        "candidate_build_plan": str(candidate_build_plan_path),
        "candidate_build_summary": str(candidate_build_summary_path),
        "candidate_build_preflight_summary": str(candidate_build_preflight_summary_path),
        "scheduler_bruteforce_plan": str(scheduler_bruteforce_plan_path),
        "regular_gemm_full_config": str(regular_gemm_full_config_path),
        "regular_gemm_gap_scan": str(regular_gemm_gap_scan_path),
        "scheduler_bruteforce_full_config": str(scheduler_bruteforce_full_config_path),
        "scheduler_bruteforce_gap_scan": str(scheduler_bruteforce_gap_scan_path),
        "device_target_detection": str(device_target_detection_path),
        "safe_candidates": str(reports_dir / "bmg_safe_candidates.json"),
        "verified_hw_caps": str(verified_hw_caps_path),
        "results_csv": str(reports_dir / "gemm_profile_results.csv"),
        "dispatch_table": str(reports_dir / "gemm_dispatch_table.json"),
        "optimal_dispatch_table": str(reports_dir / "optimal_dispatch_table.json"),
        "reference_doc": str(reference_doc_path) if reference_doc is not None else "",
        "reference_comparison": str(reports_dir / "reference_comparison.json") if reference_doc is not None else "",
        "phase_a_summary": str(phase_a_summary_path),
        "phase_b_summary": str(phase_b_summary_path),
        "run_summary": str(run_summary_path),
        "summary": str(run_summary_path),
        "dry_run": dry_run_mode,
    }
    artifact_bundle_manifest_path = reports_dir / "gemm_product_bundle_manifest.json"
    write_json(artifact_bundle_manifest_path, build_artifact_bundle_manifest(workspace, outputs))
    outputs["artifact_bundle_manifest"] = str(artifact_bundle_manifest_path)
    return outputs


def build_parser():
    if __package__ in (None, ""):
        from intel_gemm_profiler.cli import build_parser as _build_parser
    else:
        from .cli import build_parser as _build_parser

    return _build_parser()


def dispatch_lookup_from_args(args):
    if __package__ in (None, ""):
        from intel_gemm_profiler.cli import dispatch_lookup_from_args as _dispatch_lookup_from_args
    else:
        from .cli import dispatch_lookup_from_args as _dispatch_lookup_from_args

    return _dispatch_lookup_from_args(args)


def main():
    if __package__ in (None, ""):
        from intel_gemm_profiler.cli import main as _main
    else:
        from .cli import main as _main

    return _main()


if __name__ == "__main__":
    main()
