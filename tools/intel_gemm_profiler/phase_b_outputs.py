#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

from .bundle import build_artifact_bundle_manifest
from .selector import (
    build_dispatch_table,
    build_phase_a_summary,
    build_phase_b_summary,
    build_reference_comparison,
    build_run_summary,
    write_results_csv,
)
from .utils import write_json


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
