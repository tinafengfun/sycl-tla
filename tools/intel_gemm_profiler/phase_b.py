#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

from .build_plan import (
    benchmark_batch_plan_by_kernel_id,
    benchmark_command_strings,
    benchmark_log_paths,
    run_entries_with_batch_benchmarks,
)
from .build_exec import execute_candidate_build_plan, execute_candidate_build_preflight_plans
from .candidate_entries import build_screening_entries, generate_confirmation_entries
from .phase_b_outputs import finalize_phase_b_outputs
from .runner import run_entries_with_benchmark, run_entries_with_streamk_example
from .utils import write_json


def execute_phase_b(
    args,
    *,
    shapes_doc,
    candidate_space,
    logs_dir,
    configs_dir,
    manifests_dir,
    reports_dir,
    base_runtime_shell_init,
    compile_shell_init,
    candidate_build_plan,
    candidate_build_workers,
    probe_rows,
    probe_logs,
    probe_commands,
    top_k,
    confirm_runs,
    env_caps,
    verified_hw_caps_path,
    execute_preflight_fn=execute_candidate_build_preflight_plans,
    execute_build_fn=execute_candidate_build_plan,
    build_screening_entries_fn=build_screening_entries,
    benchmark_batch_plan_by_kernel_id_fn=benchmark_batch_plan_by_kernel_id,
    run_entries_with_batch_benchmarks_fn=run_entries_with_batch_benchmarks,
    run_entries_with_benchmark_fn=run_entries_with_benchmark,
    benchmark_log_paths_fn=benchmark_log_paths,
    benchmark_command_strings_fn=benchmark_command_strings,
    run_entries_with_streamk_example_fn=run_entries_with_streamk_example,
    generate_confirmation_entries_fn=generate_confirmation_entries,
):
    candidate_build_summary_path = reports_dir / "candidate_build_summary.json"
    candidate_build_preflight_summary_path = reports_dir / "candidate_build_preflight_summary.json"
    candidate_build_summary = {"status": "not_run", "reason": "build_candidate_benchmark disabled"}
    candidate_build_preflight_summary = {"status": "not_run", "reason": "run_candidate_build_preflight disabled"}
    build_timeout = args.build_timeout or args.timeout
    effective_benchmark_exe = args.benchmark_exe

    if args.run_candidate_build_preflight:
        candidate_build_preflight_summary = execute_preflight_fn(
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
        candidate_build_summary = execute_build_fn(
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

    screening_entries = build_screening_entries_fn(shapes_doc, candidate_space)
    all_rows = list(probe_rows)
    log_paths = list(probe_logs)
    benchmark_commands = list(probe_commands)
    if candidate_build_summary.get("status") == "pass":
        log_paths.extend(step["log"] for step in candidate_build_summary["steps"])
        benchmark_commands.extend(step["command"] for step in candidate_build_summary["steps"])
    if candidate_build_preflight_summary.get("status") == "pass":
        for batch in candidate_build_preflight_summary["batches"]:
            for step in batch.get("steps", []):
                log_paths.append(step.get("log", ""))
                benchmark_commands.append(step.get("command", ""))

    batch_plan_by_kernel = (
        benchmark_batch_plan_by_kernel_id_fn(candidate_build_plan)
        if args.use_candidate_build_preflight_benchmarks
        else {}
    )

    if not args.skip_run:
        screening_benchmark_entries = [
            entry for entry in screening_entries if entry["candidate"].get("runner", "benchmark") == "benchmark"
        ]
        screening_streamk_entries = [
            entry for entry in screening_entries if entry["candidate"].get("runner") == "streamk_example"
        ]
        screening_rows = []
        if screening_benchmark_entries:
            screening_log = logs_dir / "screening.log"
            if args.use_candidate_build_preflight_benchmarks:
                rows, command, batch_logs = run_entries_with_batch_benchmarks_fn(
                    screening_benchmark_entries,
                    configs_dir / "screening.in",
                    manifests_dir / "screening_manifest.json",
                    screening_log,
                    batch_plan_by_kernel,
                    cwd=args.cwd,
                    shell_init=base_runtime_shell_init,
                    timeout=args.timeout,
                    chunk_size=args.benchmark_entry_chunk_size,
                )
            else:
                rows, command = run_entries_with_benchmark_fn(
                    screening_benchmark_entries,
                    configs_dir / "screening.in",
                    manifests_dir / "screening_manifest.json",
                    screening_log,
                    effective_benchmark_exe,
                    cwd=args.cwd,
                    shell_init=base_runtime_shell_init,
                    timeout=args.timeout,
                    chunk_size=args.benchmark_entry_chunk_size,
                )
                batch_logs = benchmark_log_paths_fn(screening_log, command)
            screening_rows.extend(rows)
            log_paths.extend(batch_logs)
            benchmark_commands.extend(benchmark_command_strings_fn(command))
        if screening_streamk_entries:
            rows, commands = run_entries_with_streamk_example_fn(
                screening_streamk_entries,
                logs_dir,
                args.streamk_example_exe,
                cwd=args.cwd,
                shell_init=base_runtime_shell_init,
                timeout=args.timeout,
            )
            screening_rows.extend(rows)
            log_paths.extend(str(logs_dir / f"{entry['bm_name']}.log") for entry in screening_streamk_entries)
            benchmark_commands.extend(commands)
        all_rows.extend(screening_rows)

        if confirm_runs > 0:
            confirm_entries = generate_confirmation_entries_fn(
                screening_rows,
                candidate_space,
                shapes_doc,
                top_k=top_k,
                confirm_runs=confirm_runs,
            )
            if confirm_entries:
                confirm_benchmark_entries = [
                    entry for entry in confirm_entries if entry["candidate"].get("runner", "benchmark") == "benchmark"
                ]
                confirm_streamk_entries = [
                    entry for entry in confirm_entries if entry["candidate"].get("runner") == "streamk_example"
                ]
                confirm_rows = []
                if confirm_benchmark_entries:
                    confirm_log = logs_dir / "confirm.log"
                    if args.use_candidate_build_preflight_benchmarks:
                        rows, command, batch_logs = run_entries_with_batch_benchmarks_fn(
                            confirm_benchmark_entries,
                            configs_dir / "confirm.in",
                            manifests_dir / "confirm_manifest.json",
                            confirm_log,
                            batch_plan_by_kernel,
                            cwd=args.cwd,
                            shell_init=base_runtime_shell_init,
                            timeout=args.timeout,
                            chunk_size=args.benchmark_entry_chunk_size,
                        )
                    else:
                        rows, command = run_entries_with_benchmark_fn(
                            confirm_benchmark_entries,
                            configs_dir / "confirm.in",
                            manifests_dir / "confirm_manifest.json",
                            confirm_log,
                            effective_benchmark_exe,
                            cwd=args.cwd,
                            shell_init=base_runtime_shell_init,
                            timeout=args.timeout,
                            chunk_size=args.benchmark_entry_chunk_size,
                        )
                        batch_logs = benchmark_log_paths_fn(confirm_log, command)
                    confirm_rows.extend(rows)
                    log_paths.extend(batch_logs)
                    benchmark_commands.extend(benchmark_command_strings_fn(command))
                if confirm_streamk_entries:
                    rows, commands = run_entries_with_streamk_example_fn(
                        confirm_streamk_entries,
                        logs_dir,
                        args.streamk_example_exe,
                        cwd=args.cwd,
                        shell_init=base_runtime_shell_init,
                        timeout=args.timeout,
                    )
                    confirm_rows.extend(rows)
                    log_paths.extend(str(logs_dir / f"{entry['bm_name']}.log") for entry in confirm_streamk_entries)
                    benchmark_commands.extend(commands)
                all_rows.extend(confirm_rows)

    return {
        "candidate_build_summary_path": candidate_build_summary_path,
        "candidate_build_preflight_summary_path": candidate_build_preflight_summary_path,
        "candidate_build_summary": candidate_build_summary,
        "candidate_build_preflight_summary": candidate_build_preflight_summary,
        "effective_benchmark_exe": effective_benchmark_exe,
        "all_rows": all_rows,
        "log_paths": log_paths,
        "benchmark_commands": benchmark_commands,
    }
