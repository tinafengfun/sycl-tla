#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import copy
import hashlib
import shutil
from pathlib import Path

from .dispatch import DISPATCH_KEY_FIELDS, load_dispatch_table
from .schemas import SEARCH_RUNTIME_SCHEMA
from .utils import ensure_dir, now_iso, read_json, shell_join, write_json


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
                mismatch_target.append(
                    f"{artifact_name or artifact_path}: size_bytes expected {expected_size}, got {actual_size}"
                )
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
