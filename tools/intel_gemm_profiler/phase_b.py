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
from .runner import (
    run_entries_with_benchmark,
    run_entries_with_streamk_example,
    run_entries_with_weight_only_example,
    run_entries_with_batch_weight_only_codegen,
    run_entries_with_weight_only_codegen,
)
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
    run_entries_with_weight_only_example_fn=run_entries_with_weight_only_example,
    run_entries_with_batch_weight_only_codegen_fn=run_entries_with_batch_weight_only_codegen,
    run_entries_with_weight_only_codegen_fn=run_entries_with_weight_only_codegen,
    generate_confirmation_entries_fn=generate_confirmation_entries,
):
    candidate_build_summary_path = reports_dir / "candidate_build_summary.json"
    candidate_build_preflight_summary_path = reports_dir / "candidate_build_preflight_summary.json"
    candidate_build_summary = {"status": "not_run", "reason": "build_candidate_benchmark disabled"}
    candidate_build_preflight_summary = {"status": "not_run", "reason": "run_candidate_build_preflight disabled"}
    build_timeout = args.build_timeout or args.timeout
    effective_benchmark_exe = args.benchmark_exe
    mixed_codegen_only = (
        bool(candidate_space.get("candidates"))
        and all(candidate.get("runner") == "mixed_dtype_codegen" for candidate in candidate_space["candidates"])
    )

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
            preflight_pass_count = candidate_build_preflight_summary.get("passed_batches", 0)
            if (
                mixed_codegen_only
                and args.use_candidate_build_preflight_benchmarks
                and preflight_pass_count > 0
            ):
                pass
            else:
                raise RuntimeError(candidate_build_preflight_summary["failure_reason"])
    else:
        write_json(candidate_build_preflight_summary_path, candidate_build_preflight_summary)

    if args.use_candidate_build_preflight_benchmarks and candidate_build_preflight_summary.get("status") != "pass":
        if not (
            mixed_codegen_only
            and candidate_build_preflight_summary.get("passed_batches", 0) > 0
        ):
            raise ValueError("--use-candidate-build-preflight-benchmarks requires successful --run-candidate-build-preflight.")

    if args.build_candidate_benchmark and not (
        mixed_codegen_only
        and args.run_candidate_build_preflight
        and args.use_candidate_build_preflight_benchmarks
    ):
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
        if (
            args.build_candidate_benchmark
            and mixed_codegen_only
            and args.run_candidate_build_preflight
            and args.use_candidate_build_preflight_benchmarks
        ):
            candidate_build_summary = {
                "status": "not_run",
                "reason": "aggregate build skipped; using successful mixed-dtype preflight batches",
            }
        write_json(candidate_build_summary_path, candidate_build_summary)

    successful_batch_ids = {
        batch["batch_id"]
        for batch in candidate_build_preflight_summary.get("batches", [])
        if batch.get("status") == "pass"
    }

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
        benchmark_batch_plan_by_kernel_id_fn(candidate_build_plan, successful_batch_ids=successful_batch_ids or None)
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
        screening_mixed_bf16_s8_entries = [
            entry for entry in screening_entries if entry["candidate"].get("runner") == "mixed_bf16_s8_example"
        ]
        screening_mixed_f16_s8_entries = [
            entry for entry in screening_entries if entry["candidate"].get("runner") == "mixed_f16_s8_example"
        ]
        screening_mixed_codegen_entries = [
            entry for entry in screening_entries if entry["candidate"].get("runner") == "mixed_dtype_codegen"
        ]
        if args.use_candidate_build_preflight_benchmarks:
            screening_mixed_codegen_entries = [
                entry for entry in screening_mixed_codegen_entries
                if entry["candidate"]["kernel_id"] in batch_plan_by_kernel
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
        if screening_mixed_bf16_s8_entries:
            rows, commands = run_entries_with_weight_only_example_fn(
                screening_mixed_bf16_s8_entries,
                logs_dir,
                args.mixed_bf16_s8_example_exe,
                cwd=args.cwd,
                shell_init=base_runtime_shell_init,
                timeout=args.timeout,
            )
            screening_rows.extend(rows)
            log_paths.extend(str(logs_dir / f"{entry['bm_name']}.log") for entry in screening_mixed_bf16_s8_entries)
            benchmark_commands.extend(commands)
        if screening_mixed_f16_s8_entries:
            rows, commands = run_entries_with_weight_only_example_fn(
                screening_mixed_f16_s8_entries,
                logs_dir,
                args.mixed_f16_s8_example_exe,
                cwd=args.cwd,
                shell_init=base_runtime_shell_init,
                timeout=args.timeout,
            )
            screening_rows.extend(rows)
            log_paths.extend(str(logs_dir / f"{entry['bm_name']}.log") for entry in screening_mixed_f16_s8_entries)
            benchmark_commands.extend(commands)
        if screening_mixed_codegen_entries:
            if args.use_candidate_build_preflight_benchmarks:
                rows, commands = run_entries_with_batch_weight_only_codegen_fn(
                    screening_mixed_codegen_entries,
                    logs_dir,
                    batch_plan_by_kernel,
                    cwd=args.cwd,
                    shell_init=base_runtime_shell_init,
                    timeout=args.timeout,
                )
            else:
                rows, commands = run_entries_with_weight_only_codegen_fn(
                    screening_mixed_codegen_entries,
                    logs_dir,
                    candidate_build_plan,
                    cwd=args.cwd,
                    shell_init=base_runtime_shell_init,
                    timeout=args.timeout,
                )
            screening_rows.extend(rows)
            log_paths.extend(str(logs_dir / f"{entry['bm_name']}.log") for entry in screening_mixed_codegen_entries)
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
                confirm_mixed_bf16_s8_entries = [
                    entry for entry in confirm_entries if entry["candidate"].get("runner") == "mixed_bf16_s8_example"
                ]
                confirm_mixed_f16_s8_entries = [
                    entry for entry in confirm_entries if entry["candidate"].get("runner") == "mixed_f16_s8_example"
                ]
                confirm_mixed_codegen_entries = [
                    entry for entry in confirm_entries if entry["candidate"].get("runner") == "mixed_dtype_codegen"
                ]
                if args.use_candidate_build_preflight_benchmarks:
                    confirm_mixed_codegen_entries = [
                        entry for entry in confirm_mixed_codegen_entries
                        if entry["candidate"]["kernel_id"] in batch_plan_by_kernel
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
                if confirm_mixed_bf16_s8_entries:
                    rows, commands = run_entries_with_weight_only_example_fn(
                        confirm_mixed_bf16_s8_entries,
                        logs_dir,
                        args.mixed_bf16_s8_example_exe,
                        cwd=args.cwd,
                        shell_init=base_runtime_shell_init,
                        timeout=args.timeout,
                    )
                    confirm_rows.extend(rows)
                    log_paths.extend(str(logs_dir / f"{entry['bm_name']}.log") for entry in confirm_mixed_bf16_s8_entries)
                    benchmark_commands.extend(commands)
                if confirm_mixed_f16_s8_entries:
                    rows, commands = run_entries_with_weight_only_example_fn(
                        confirm_mixed_f16_s8_entries,
                        logs_dir,
                        args.mixed_f16_s8_example_exe,
                        cwd=args.cwd,
                        shell_init=base_runtime_shell_init,
                        timeout=args.timeout,
                    )
                    confirm_rows.extend(rows)
                    log_paths.extend(str(logs_dir / f"{entry['bm_name']}.log") for entry in confirm_mixed_f16_s8_entries)
                    benchmark_commands.extend(commands)
                if confirm_mixed_codegen_entries:
                    if args.use_candidate_build_preflight_benchmarks:
                        rows, commands = run_entries_with_batch_weight_only_codegen_fn(
                            confirm_mixed_codegen_entries,
                            logs_dir,
                            batch_plan_by_kernel,
                            cwd=args.cwd,
                            shell_init=base_runtime_shell_init,
                            timeout=args.timeout,
                        )
                    else:
                        rows, commands = run_entries_with_weight_only_codegen_fn(
                            confirm_mixed_codegen_entries,
                            logs_dir,
                            candidate_build_plan,
                            cwd=args.cwd,
                            shell_init=base_runtime_shell_init,
                            timeout=args.timeout,
                        )
                    confirm_rows.extend(rows)
                    log_paths.extend(str(logs_dir / f"{entry['bm_name']}.log") for entry in confirm_mixed_codegen_entries)
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
