#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import os
from pathlib import Path

from .runner import run_benchmark, run_entries_with_benchmark
from .utils import ensure_dir, now_iso, read_json, resolve_executable, shell_join, write_json


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
    log = Path(build_log_path) if not isinstance(build_log_path, Path) else build_log_path
    exe = Path(batch_plan.get("benchmark_exe", ""))
    if not log.exists() or not exe.exists():
        return False
    try:
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-5:]
        for line in tail:
            if "Build succeeded" in line or "[100%] Built target" in line:
                return True
    except Exception:
        pass
    return False


def _load_preflight_progress(progress_path):
    try:
        return read_json(progress_path).get("completed_batches", {})
    except Exception:
        return {}


def _save_preflight_progress(progress_path, completed_batches):
    tmp = Path(str(progress_path) + ".tmp")
    write_json(
        tmp,
        {
            "schema_version": "1.0",
            "generated_at": now_iso(),
            "completed_batches": completed_batches,
        },
    )
    os.replace(tmp, progress_path)


def execute_candidate_build_preflight_plans(build_plan, log_dir, shell_init="", timeout=None, max_workers=1, resume=False, progress_path=None, detect_available_vcpus_fn=None, resolve_candidate_build_jobs_fn=None):
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
    plans_sorted = sorted(preflight_plans, key=lambda p: p["batch_index"])
    resumed_batches = {}
    if resume and progress_path:
        progress_state = _load_preflight_progress(Path(progress_path))
        for plan in plans_sorted:
            bid = plan["batch_id"]
            log_path = Path(log_dir) / f"candidate_build_preflight_{bid}.log"
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

    total_vcpus = 1
    cores_per_worker = 1
    if max_workers <= 1:
        batches = [_build_one(plan) for plan in plans_sorted]
        if resumed_batches:
            batches = [resumed_batches[i] for i in sorted(resumed_batches)] + batches
        if progress_path:
            all_done = dict(resumed_batches)
            for batch in batches:
                if batch.get("status") == "pass":
                    all_done[batch["batch_id"]] = "pass"
            if all_done:
                _save_preflight_progress(Path(progress_path), all_done)
    else:
        import concurrent.futures
        import threading

        total_vcpus = detect_available_vcpus_fn() if detect_available_vcpus_fn else 1
        cores_per_worker = (
            resolve_candidate_build_jobs_fn(max_workers, total_vcpus=total_vcpus)
            if resolve_candidate_build_jobs_fn
            else 1
        )
        lock = threading.Lock()
        results_by_index = {}
        completed = [0]

        def _build_parallel(plan):
            result = _build_one(plan)
            with lock:
                results_by_index[plan["batch_index"]] = result
                completed[0] += 1
                if completed[0] % 50 == 0 or completed[0] <= 5:
                    passed = sum(1 for item in results_by_index.values() if item["status"] == "pass")
                    print(
                        f"  [preflight] {completed[0]}/{len(plans_sorted)} batches ({passed} passed, {max_workers} workers)",
                        flush=True,
                    )
            return result

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_build_parallel, plan) for plan in plans_sorted]
            concurrent.futures.wait(futures)

        batches = [results_by_index[i] for i in sorted(results_by_index)]
        if resumed_batches:
            batches = [resumed_batches[i] for i in sorted(resumed_batches)] + batches
        if progress_path:
            all_done = dict(resumed_batches)
            for batch in batches:
                if batch.get("status") == "pass":
                    all_done[batch["batch_id"]] = "pass"
            if all_done:
                _save_preflight_progress(Path(progress_path), all_done)

    failed = [batch for batch in batches if batch["status"] != "pass"]
    status = "pass" if not failed else ("timeout" if any(batch["status"] == "timeout" for batch in failed) else "fail")
    return {
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
        "total_vcpus": total_vcpus,
        "cores_per_worker": cores_per_worker,
    }


def benchmark_batch_plan_by_kernel_id(build_plan):
    mapping = {}
    for plan in build_plan.get("batch_preflight_plans", []):
        for kernel_id in plan.get("selected_kernel_list", []):
            mapping[kernel_id] = plan
    return mapping


def batched_stage_path(path, batch_id):
    return path.with_name(f"{path.stem}_{batch_id}{path.suffix}")


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
