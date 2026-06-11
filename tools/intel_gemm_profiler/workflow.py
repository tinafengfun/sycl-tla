#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import copy
import csv
import json
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
    from intel_gemm_profiler.analysis import (
        REGULAR_GEMM_FULL_CONFIG_FIELDS,
        SCHEDULER_BRUTEFORCE_CONFIG_FIELDS,
        build_regular_gemm_gap_scan,
        build_scheduler_bruteforce_gap_scan,
        build_scheduler_bruteforce_plan,
        collect_regular_gemm_full_config_rows,
        collect_scheduler_bruteforce_full_config_rows,
    )
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
    from intel_gemm_profiler.dispatch import DISPATCH_KEY_FIELDS, load_dispatch_table, lookup_dispatch_entry
    from intel_gemm_profiler.device_target import resolve_profiles_device_target
    from intel_gemm_profiler.hw_specs import resolve_hw_reference_spec
    from intel_gemm_profiler.phase_a import (
        build_compiler_flags_probe_summary,
        empty_anomaly_report,
        run_phase_a_probe,
    )
    from intel_gemm_profiler.runner import collect_environment_metadata, run_entries_with_benchmark, run_entries_with_streamk_example
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
    from .analysis import (
        REGULAR_GEMM_FULL_CONFIG_FIELDS,
        SCHEDULER_BRUTEFORCE_CONFIG_FIELDS,
        build_regular_gemm_gap_scan,
        build_scheduler_bruteforce_gap_scan,
        build_scheduler_bruteforce_plan,
        collect_regular_gemm_full_config_rows,
        collect_scheduler_bruteforce_full_config_rows,
    )
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
    from .dispatch import DISPATCH_KEY_FIELDS, load_dispatch_table, lookup_dispatch_entry
    from .device_target import resolve_profiles_device_target
    from .hw_specs import resolve_hw_reference_spec
    from .phase_a import (
        build_compiler_flags_probe_summary,
        empty_anomaly_report,
        run_phase_a_probe,
    )
    from .runner import collect_environment_metadata, run_entries_with_benchmark, run_entries_with_streamk_example
    from .selector import build_candidate_coverage_report, build_dispatch_table, build_phase_a_summary, build_phase_b_summary, build_reference_comparison, build_run_summary, write_results_csv
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
