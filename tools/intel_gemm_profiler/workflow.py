#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import copy
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    PACKAGE_ROOT = Path(__file__).resolve().parents[1]
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    from intel_gemm_profiler.catalog import build_kernel_catalog
    from intel_gemm_profiler.candidates import (
        build_candidate_build_manifest,
        build_screening_entries,
        generate_candidate_space,
        generate_confirmation_entries,
    )
    from intel_gemm_profiler.constraints import (
        apply_probe_results_to_profiles,
        default_compiler_profiles,
        default_constraints,
        selected_compile_env,
        selected_runtime_env,
    )
    from intel_gemm_profiler.artifacts import prepare_candidate_artifacts
    from intel_gemm_profiler.bundle import (
        build_artifact_bundle_manifest,
        export_product_bundle_manifest,
        validate_product_bundle_manifest,
    )
    from intel_gemm_profiler.build_plan import (
        benchmark_batch_plan_by_kernel_id,
        benchmark_command_strings,
        benchmark_log_paths,
        build_candidate_build_plan,
        detect_available_vcpus,
        execute_candidate_build_plan,
        execute_candidate_build_preflight_plans,
        resolve_candidate_build_jobs,
        run_entries_with_batch_benchmarks,
        validate_candidate_auto_build_mode,
    )
    from intel_gemm_profiler.inputs import (
        SEARCH_STRATEGY_PRESETS,
        apply_bruteforce_scheduler_search_defaults,
        apply_search_strategy_defaults,
        filter_candidate_space_by_compiled_kernels,
        limit_shapes_and_reference,
        load_compiled_kernel_list,
        load_target_shapes_and_reference,
    )
    from intel_gemm_profiler.dispatch import DISPATCH_KEY_FIELDS, load_dispatch_table, lookup_dispatch_entry
    from intel_gemm_profiler.device_target import resolve_profiles_device_target
    from intel_gemm_profiler.hw_specs import resolve_hw_reference_spec
    from intel_gemm_profiler.phase_a import (
        empty_anomaly_report,
        run_phase_a_probe,
    )
    from intel_gemm_profiler.phase_b import execute_phase_b, finalize_phase_b_outputs
    from intel_gemm_profiler.runner import collect_environment_metadata, run_entries_with_benchmark, run_entries_with_streamk_example
    from intel_gemm_profiler.selector import build_candidate_coverage_report, build_dispatch_table, build_phase_a_summary, build_phase_b_summary, build_reference_comparison, build_run_summary, write_results_csv
    from intel_gemm_profiler.utils import ensure_dir, read_json, shell_init_with_env, write_json
    from intel_gemm_profiler.schemas import SEARCH_RUNTIME_SCHEMA
else:
    from .catalog import build_kernel_catalog
    from .candidates import (
        build_candidate_build_manifest,
        build_screening_entries,
        generate_candidate_space,
        generate_confirmation_entries,
    )
    from .constraints import (
        apply_probe_results_to_profiles,
        default_compiler_profiles,
        default_constraints,
        selected_compile_env,
        selected_runtime_env,
    )
    from .artifacts import prepare_candidate_artifacts
    from .bundle import (
        build_artifact_bundle_manifest,
        export_product_bundle_manifest,
        validate_product_bundle_manifest,
    )
    from .build_plan import (
        benchmark_batch_plan_by_kernel_id,
        benchmark_command_strings,
        benchmark_log_paths,
        build_candidate_build_plan,
        detect_available_vcpus,
        execute_candidate_build_plan,
        execute_candidate_build_preflight_plans,
        resolve_candidate_build_jobs,
        run_entries_with_batch_benchmarks,
        validate_candidate_auto_build_mode,
    )
    from .inputs import (
        SEARCH_STRATEGY_PRESETS,
        apply_bruteforce_scheduler_search_defaults,
        apply_search_strategy_defaults,
        filter_candidate_space_by_compiled_kernels,
        limit_shapes_and_reference,
        load_compiled_kernel_list,
        load_target_shapes_and_reference,
    )
    from .dispatch import DISPATCH_KEY_FIELDS, load_dispatch_table, lookup_dispatch_entry
    from .device_target import resolve_profiles_device_target
    from .hw_specs import resolve_hw_reference_spec
    from .phase_a import (
        empty_anomaly_report,
        run_phase_a_probe,
    )
    from .phase_b import execute_phase_b, finalize_phase_b_outputs
    from .runner import collect_environment_metadata, run_entries_with_benchmark, run_entries_with_streamk_example
    from .selector import build_candidate_coverage_report, build_dispatch_table, build_phase_a_summary, build_phase_b_summary, build_reference_comparison, build_run_summary, write_results_csv
    from .utils import ensure_dir, read_json, shell_init_with_env, write_json
    from .schemas import SEARCH_RUNTIME_SCHEMA


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
    artifact_paths = prepare_candidate_artifacts(
        args,
        workspace,
        reports_dir,
        candidate_space,
        profiles,
        constraints,
        build_plan_fn=build_candidate_build_plan,
        detect_vcpus_fn=detect_available_vcpus,
        resolve_jobs_fn=resolve_candidate_build_jobs,
    )
    build_manifest = artifact_paths["build_manifest"]
    candidate_build_plan = artifact_paths["candidate_build_plan"]
    candidate_build_workers = artifact_paths["candidate_build_workers"]
    phase_b_results = execute_phase_b(
        args,
        shapes_doc=shapes_doc,
        candidate_space=candidate_space,
        logs_dir=logs_dir,
        configs_dir=configs_dir,
        manifests_dir=manifests_dir,
        reports_dir=reports_dir,
        base_runtime_shell_init=base_runtime_shell_init,
        compile_shell_init=compile_shell_init,
        candidate_build_plan=candidate_build_plan,
        candidate_build_workers=candidate_build_workers,
        probe_rows=probe_rows,
        probe_logs=probe_logs,
        probe_commands=probe_commands,
        top_k=top_k,
        confirm_runs=confirm_runs,
        env_caps=env_caps,
        verified_hw_caps_path=verified_hw_caps_path,
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
    )
    return finalize_phase_b_outputs(
        workspace=workspace,
        inputs_dir=inputs_dir,
        reports_dir=reports_dir,
        artifact_paths=artifact_paths,
        candidate_coverage_report_path=candidate_coverage_report_path,
        device_target_detection_path=device_target_detection_path,
        verified_hw_caps_path=verified_hw_caps_path,
        reference_doc_path=reference_doc_path,
        reference_doc=reference_doc,
        env_caps=env_caps,
        constraints=constraints,
        probe_rows=probe_rows,
        candidate_space=candidate_space,
        shapes_doc=shapes_doc,
        top_k=top_k,
        confirm_runs=confirm_runs,
        close_call_threshold=args.close_call_threshold,
        all_rows=phase_b_results["all_rows"],
        benchmark_commands=phase_b_results["benchmark_commands"],
        log_paths=phase_b_results["log_paths"],
        dry_run_mode=dry_run_mode,
        write_results_csv_fn=write_results_csv,
        build_dispatch_table_fn=build_dispatch_table,
        build_reference_comparison_fn=build_reference_comparison,
        build_run_summary_fn=build_run_summary,
        build_phase_a_summary_fn=build_phase_a_summary,
        build_phase_b_summary_fn=build_phase_b_summary,
        build_artifact_bundle_manifest_fn=build_artifact_bundle_manifest,
    )


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
