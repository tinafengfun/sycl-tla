#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

from .build_plan import (
    benchmark_batch_plan_by_kernel_id,
    benchmark_command_strings,
    benchmark_log_paths,
    execute_candidate_build_plan,
    execute_candidate_build_preflight_plans,
    run_entries_with_batch_benchmarks,
)
from .bundle import build_artifact_bundle_manifest
from .candidate_entries import build_screening_entries, generate_confirmation_entries
from .runner import run_entries_with_benchmark, run_entries_with_streamk_example
from .selector import (
    build_dispatch_table,
    build_phase_a_summary,
    build_phase_b_summary,
    build_reference_comparison,
    build_run_summary,
    write_results_csv,
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


def finalize_phase_b_outputs(
    *,
    workspace,
    inputs_dir,
    reports_dir,
    artifact_paths,
    candidate_coverage_report_path,
    device_target_detection_path,
    verified_hw_caps_path,
    reference_doc_path,
    reference_doc,
    env_caps,
    constraints,
    probe_rows,
    candidate_space,
    shapes_doc,
    top_k,
    confirm_runs,
    close_call_threshold,
    all_rows,
    benchmark_commands,
    log_paths,
    dry_run_mode,
    write_results_csv_fn=write_results_csv,
    build_dispatch_table_fn=build_dispatch_table,
    build_reference_comparison_fn=build_reference_comparison,
    build_run_summary_fn=build_run_summary,
    build_phase_a_summary_fn=build_phase_a_summary,
    build_phase_b_summary_fn=build_phase_b_summary,
    build_artifact_bundle_manifest_fn=build_artifact_bundle_manifest,
):
    results_csv_path = reports_dir / "gemm_profile_results.csv"
    dispatch_table_path = reports_dir / "gemm_dispatch_table.json"
    optimal_dispatch_table_path = reports_dir / "optimal_dispatch_table.json"
    reference_comparison_path = reports_dir / "reference_comparison.json"
    run_summary_path = reports_dir / "run_summary.json"
    phase_a_summary_path = reports_dir / "phase_a_summary.json"
    phase_b_summary_path = reports_dir / "phase_b_summary.json"

    write_results_csv_fn(all_rows, results_csv_path)
    dispatch_table = build_dispatch_table_fn(
        all_rows,
        shapes_doc,
        top_k=top_k,
        confirm_runs=confirm_runs,
        close_call_threshold=close_call_threshold,
        candidate_space=candidate_space,
        hw_spec=env_caps.get("hw_reference_spec"),
    )
    write_json(dispatch_table_path, dispatch_table)
    write_json(optimal_dispatch_table_path, dispatch_table)
    if reference_doc is not None:
        write_json(
            reference_comparison_path,
            build_reference_comparison_fn(dispatch_table, reference_doc),
        )
    summary = build_run_summary_fn(all_rows, dispatch_table, benchmark_commands, log_paths)
    write_json(run_summary_path, summary)
    write_json(phase_a_summary_path, build_phase_a_summary_fn(env_caps, constraints, probe_rows))
    write_json(phase_b_summary_path, build_phase_b_summary_fn(candidate_space, dispatch_table, summary))

    outputs = {
        "workspace": str(workspace),
        "search_runtime_schema": str(inputs_dir / "search_runtime_schema.json"),
        "target_shapes": str(inputs_dir / "gemm_target_shapes.json"),
        "constraints": str(inputs_dir / "safe_search_constraints.json"),
        "compiler_profiles": str(inputs_dir / "compiler_profiles.json"),
        "kernel_catalog": str(reports_dir / "kernel_catalog.json"),
        "candidate_space": str(reports_dir / "gemm_candidate_space.json"),
        "candidate_coverage_report": str(candidate_coverage_report_path),
        "build_manifest": str(artifact_paths["build_manifest_path"]),
        "selected_kernel_list": str(artifact_paths["selected_kernel_list_path"]),
        "selected_kernel_filter": str(artifact_paths["selected_kernel_filter_path"]),
        "candidate_build_cmake_config": str(artifact_paths["candidate_build_cmake_config_path"]),
        "candidate_build_plan": str(artifact_paths["candidate_build_plan_path"]),
        "candidate_build_summary": str(reports_dir / "candidate_build_summary.json"),
        "candidate_build_preflight_summary": str(reports_dir / "candidate_build_preflight_summary.json"),
        "scheduler_bruteforce_plan": str(artifact_paths["scheduler_bruteforce_plan_path"]),
        "regular_gemm_full_config": str(artifact_paths["regular_gemm_full_config_path"]),
        "regular_gemm_gap_scan": str(artifact_paths["regular_gemm_gap_scan_path"]),
        "scheduler_bruteforce_full_config": str(artifact_paths["scheduler_bruteforce_full_config_path"]),
        "scheduler_bruteforce_gap_scan": str(artifact_paths["scheduler_bruteforce_gap_scan_path"]),
        "device_target_detection": str(device_target_detection_path),
        "safe_candidates": str(reports_dir / "bmg_safe_candidates.json"),
        "verified_hw_caps": str(verified_hw_caps_path),
        "results_csv": str(results_csv_path),
        "dispatch_table": str(dispatch_table_path),
        "optimal_dispatch_table": str(optimal_dispatch_table_path),
        "reference_doc": str(reference_doc_path) if reference_doc is not None else "",
        "reference_comparison": str(reference_comparison_path) if reference_doc is not None else "",
        "phase_a_summary": str(phase_a_summary_path),
        "phase_b_summary": str(phase_b_summary_path),
        "run_summary": str(run_summary_path),
        "summary": str(run_summary_path),
        "dry_run": dry_run_mode,
    }
    artifact_bundle_manifest_path = reports_dir / "gemm_product_bundle_manifest.json"
    write_json(artifact_bundle_manifest_path, build_artifact_bundle_manifest_fn(workspace, outputs))
    outputs["artifact_bundle_manifest"] = str(artifact_bundle_manifest_path)
    return outputs
