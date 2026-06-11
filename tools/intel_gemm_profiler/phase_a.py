#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import statistics

from .candidate_entries import (
    build_compiler_profile_probe_entries,
    build_dpas_probe_entry,
    build_phase_a_probe_entries,
)
from .candidates import (
    generate_candidate_space,
)
from .constraints import (
    apply_run_probe_constraints,
    apply_static_probe_constraints,
    selected_compile_env,
    selected_runtime_env,
)
from .hw_specs import detect_probe_anomalies, resolve_hw_reference_spec
from .runner import (
    collect_environment_metadata,
    run_entries_with_benchmark,
    run_entries_with_streamk_example,
)
from .utils import shell_init_with_env, shell_join, write_json


def build_compiler_flags_probe_summary(rows, profiles=None):
    profile_class_map = {
        profile["compiler_profile_id"]: profile.get("candidate_class", "")
        for profile in (profiles or {}).get("profiles", [])
    }
    by_profile = {}
    for row in rows:
        profile_id = row["compiler_profile_id"]
        by_profile.setdefault(profile_id, []).append(
            {
                "compiler_profile_id": profile_id,
                "candidate_class": row.get("candidate_class", profile_class_map.get(profile_id, "")),
                "status": row["status"],
                "avg_tflops": row["avg_tflops"],
                "avg_runtime_ms": row["avg_runtime_ms"],
                "candidate_id": row["candidate_id"],
                "shape_id": row["shape_id"],
                "log": row["stdout_log"],
            }
        )
    summarized = []
    for profile_id, items in by_profile.items():
        passed = [item for item in items if item["status"] == "pass"]
        best_pass = max(passed, key=lambda item: float(item["avg_tflops"] or 0.0), default=None)
        status = "pass" if passed else items[0]["status"]
        avg_tflops = ""
        avg_runtime_ms = ""
        if passed:
            tflops_values = [float(item["avg_tflops"]) for item in passed if item["avg_tflops"] != ""]
            runtime_values = [float(item["avg_runtime_ms"]) for item in passed if item["avg_runtime_ms"] != ""]
            avg_tflops = str(statistics.median(tflops_values)) if tflops_values else ""
            avg_runtime_ms = str(statistics.median(runtime_values)) if runtime_values else ""
        reference = best_pass or items[0]
        summarized.append(
            {
                "compiler_profile_id": profile_id,
                "candidate_class": reference["candidate_class"],
                "status": status,
                "avg_tflops": avg_tflops,
                "avg_runtime_ms": avg_runtime_ms,
                "candidate_id": reference["candidate_id"],
                "shape_id": reference["shape_id"],
                "log": reference["log"],
                "samples": len(items),
            }
        )
    grouped = {}
    for item in summarized:
        grouped.setdefault(item["candidate_class"], []).append(item)
    selected = {}
    for candidate_class, items in grouped.items():
        passed = [item for item in items if item["status"] == "pass"]
        if passed:
            selected[candidate_class] = max(
                passed, key=lambda item: float(item["avg_tflops"] or 0.0)
            )["compiler_profile_id"]
    return {"results": summarized, "selected_profile_ids": selected}


def empty_anomaly_report(hw_spec):
    return {
        "hw_spec": hw_spec["device_id"],
        "hw_spec_calibration_status": hw_spec.get("calibration_status", "unknown"),
        "peak_tflops": hw_spec.get("peak_xmx_tflops", 0.0),
        "anomalies": [],
        "auto_block_rules": [],
    }


def run_phase_a_probe(args, shapes_doc, base_constraints, profiles, reports_dir, configs_dir, manifests_dir, logs_dir):
    base_runtime_shell_init = shell_init_with_env(
        args.shell_init,
        selected_runtime_env(profiles, variant_override=getattr(args, "runtime_variant", None) or None),
    )
    compile_shell_init = shell_init_with_env(
        args.shell_init,
        selected_compile_env(profiles, variant_override=getattr(args, "compile_variant", None) or None),
    )
    env_caps = collect_environment_metadata(
        args.shell_init,
        args.benchmark_exe,
        args.streamk_example_exe,
        cwd=args.cwd,
    )
    static_constraints = apply_static_probe_constraints(base_constraints, env_caps)
    hw_spec = resolve_hw_reference_spec(
        static_constraints["device_arch"],
        getattr(args, "hw_spec_id", "") or profiles.get("device_target_detection", {}).get("resolved_hw_spec_id", ""),
    )
    allowed_runners = (
        ("benchmark", "streamk_example")
        if env_caps["executables"].get("streamk_example_available")
        else ("benchmark",)
    )
    static_candidate_space = generate_candidate_space(
        shapes_doc,
        static_constraints,
        profiles,
        allowed_runners=allowed_runners,
        prefilter_strategy=getattr(args, "prefilter", "none"),
    )
    probe_rows = []
    probe_logs = []
    probe_commands = []
    probe_entries = build_phase_a_probe_entries(shapes_doc, static_candidate_space)
    effective_probe_mode = args.probe_mode
    if effective_probe_mode == "auto":
        effective_probe_mode = "static" if args.skip_run else "run"
    if effective_probe_mode == "run" and not args.skip_run and probe_entries:
        probe_benchmark_entries = [
            entry for entry in probe_entries if entry["candidate"].get("runner", "benchmark") == "benchmark"
        ]
        probe_streamk_entries = [
            entry for entry in probe_entries if entry["candidate"].get("runner") == "streamk_example"
        ]
        if probe_benchmark_entries:
            probe_log = logs_dir / "probe.log"
            rows, command = run_entries_with_benchmark(
                probe_benchmark_entries,
                configs_dir / "probe.in",
                manifests_dir / "probe_manifest.json",
                probe_log,
                args.benchmark_exe,
                cwd=args.cwd,
                shell_init=base_runtime_shell_init,
                timeout=args.timeout,
            )
            probe_rows.extend(rows)
            probe_logs.append(str(probe_log))
            probe_commands.append(shell_join(command))
        if probe_streamk_entries:
            rows, commands = run_entries_with_streamk_example(
                probe_streamk_entries,
                logs_dir,
                args.streamk_example_exe,
                cwd=args.cwd,
                shell_init=base_runtime_shell_init,
                timeout=args.timeout,
            )
            probe_rows.extend(rows)
            probe_logs.extend(str(logs_dir / f"{entry['bm_name']}.log") for entry in probe_streamk_entries)
            probe_commands.extend(commands)
    dpas_probe = {"status": "skipped", "reason": "probe mode disabled or benchmark unavailable"}
    compiler_flags_probe = {"results": [], "selected_profile_ids": {}}
    if effective_probe_mode == "run" and not args.skip_run and env_caps["executables"]["benchmark_available"]:
        dpas_entry = build_dpas_probe_entry(shapes_doc, static_candidate_space)
        if dpas_entry:
            dpas_log = logs_dir / "dpas_probe.log"
            rows, command = run_entries_with_benchmark(
                [dpas_entry],
                configs_dir / "dpas_probe.in",
                manifests_dir / "dpas_probe_manifest.json",
                dpas_log,
                args.benchmark_exe,
                cwd=args.cwd,
                shell_init=base_runtime_shell_init,
                timeout=args.timeout,
            )
            if rows:
                probe_rows.extend(rows)
                probe_logs.append(str(dpas_log))
                probe_commands.append(shell_join(command))
                row = rows[0]
                dpas_probe = {
                    "status": row["status"],
                    "candidate_id": row["candidate_id"],
                    "shape_id": row["shape_id"],
                    "avg_tflops": row["avg_tflops"],
                    "avg_runtime_ms": row["avg_runtime_ms"],
                    "log": str(dpas_log),
                }
            else:
                dpas_probe = {"status": "fail", "reason": "missing benchmark row", "log": str(dpas_log)}
        compiler_probe_entries = build_compiler_profile_probe_entries(
            shapes_doc,
            static_candidate_space,
            profiles,
        )
        compiler_probe_rows = []
        for entry in compiler_probe_entries:
            profile = next(
                profile
                for profile in profiles["profiles"]
                if profile["compiler_profile_id"] == entry["compiler_profile_probe_id"]
            )
            compiler_log = logs_dir / f"{entry['compiler_profile_probe_id'].replace('.', '_')}.log"
            runtime_shell_init = shell_init_with_env(args.shell_init, selected_runtime_env(profiles, profile))
            rows, command = run_entries_with_benchmark(
                [entry],
                configs_dir / f"{entry['compiler_profile_probe_id'].replace('.', '_')}.in",
                manifests_dir / f"{entry['compiler_profile_probe_id'].replace('.', '_')}_manifest.json",
                compiler_log,
                args.benchmark_exe,
                cwd=args.cwd,
                shell_init=runtime_shell_init,
                timeout=args.timeout,
            )
            compiler_probe_rows.extend(rows)
            probe_logs.append(str(compiler_log))
            probe_commands.append(shell_join(command))
        compiler_flags_probe = build_compiler_flags_probe_summary(compiler_probe_rows, profiles)
    anomaly_report = (
        detect_probe_anomalies(probe_rows, shapes_doc, static_candidate_space, hw_spec)
        if probe_rows
        else empty_anomaly_report(hw_spec)
    )
    constraints = (
        apply_run_probe_constraints(static_constraints, probe_rows, anomaly_report=anomaly_report)
        if probe_rows
        else static_constraints
    )
    env_caps["probe_mode"] = effective_probe_mode
    env_caps["hw_reference_spec_id"] = hw_spec["device_id"]
    env_caps["hw_reference_spec"] = hw_spec
    env_caps["constraint_source"] = constraints["constraint_source"]
    env_caps["dpas_baseline_probe"] = dpas_probe
    env_caps["compiler_flags_probe"] = compiler_flags_probe
    env_caps["anomaly_report"] = anomaly_report
    env_caps["probe_results"] = [
        {
            "candidate_id": row["candidate_id"],
            "shape_id": row["shape_id"],
            "status": row["status"],
            "avg_tflops": row["avg_tflops"],
            "split_k": row["split_k"],
        }
        for row in probe_rows
    ]
    verified_hw_caps_path = reports_dir / "verified_hw_caps.json"
    write_json(verified_hw_caps_path, env_caps)
    return constraints, env_caps, verified_hw_caps_path, probe_rows, probe_logs, probe_commands
