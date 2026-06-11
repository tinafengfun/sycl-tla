#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import csv
import math
import statistics
from collections import defaultdict

from .schemas import CSV_FIELDS, REPORT_TRACKED_DIMENSIONS, SCHEMA_VERSION
from .utils import now_iso


def write_results_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def numeric_values(values):
    numeric = []
    for value in values:
        if str(value) == "":
            continue
        parsed = float(value)
        if math.isfinite(parsed):
            numeric.append(parsed)
    return numeric


def median_or_nan(values):
    numeric = numeric_values(values)
    if not numeric:
        return math.nan
    return statistics.median(numeric)


def pstdev_or_nan(values):
    numeric = numeric_values(values)
    if len(numeric) < 2:
        return math.nan
    return statistics.pstdev(numeric)


def round_or_empty(value):
    return round(value, 6) if not math.isnan(value) else ""


def tracked_metadata(source):
    return {field: source.get(field, "") for field in REPORT_TRACKED_DIMENSIONS if field in source}


def summarize_dimension_values(items):
    items = list(items)
    summary = {}
    for field in REPORT_TRACKED_DIMENSIONS:
        counts = defaultdict(int)
        for item in items:
            value = item.get(field, "")
            if value == "":
                value = "<empty>"
            counts[str(value)] += 1
        if counts:
            summary[field] = {
                "unique_count": len(counts),
                "values": dict(sorted(counts.items())),
            }
    return summary


def build_candidate_coverage_report(candidate_space):
    candidates = candidate_space.get("candidates", [])
    exceptions = candidate_space.get("candidate_exceptions", [])
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "candidate_count": len(candidates),
        "candidate_exception_count": len(exceptions),
        "accepted_dimension_values": summarize_dimension_values(candidates),
        "exception_dimension_values": summarize_dimension_values(exceptions),
        "exception_reasons": candidate_space.get("candidate_exception_summary", []),
    }


def build_run_summary(rows, dispatch_table, build_command, log_paths):
    passed = sum(1 for row in rows if row["status"] == "pass")
    failed = sum(1 for row in rows if row["status"] != "pass")
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "rows": len(rows),
        "passed": passed,
        "failed": failed,
        "dispatch_entries": len(dispatch_table["entries"]),
        "benchmark_command": build_command,
        "logs": log_paths,
    }


def build_phase_a_summary(verified_hw_caps, constraints, probe_rows):
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "probe_mode": verified_hw_caps.get("probe_mode", "off"),
        "hw_reference_spec_id": verified_hw_caps.get("hw_reference_spec_id", ""),
        "constraint_source": constraints["constraint_source"],
        "dpas_baseline_probe": verified_hw_caps.get("dpas_baseline_probe", {}),
        "compiler_flags_probe": verified_hw_caps.get("compiler_flags_probe", {}),
        "anomaly_report": verified_hw_caps.get("anomaly_report", {"anomalies": [], "auto_block_rules": []}),
        "probe_results": len(probe_rows),
        "successful_probe_results": sum(1 for row in probe_rows if row["status"] == "pass"),
        "allowed_values": constraints["allowed_values"],
        "limits": constraints["limits"],
        "blocked_rules": constraints.get("blocked_rules", []),
        "probe_feedback": constraints.get("probe_feedback", {}),
    }


def build_phase_b_summary(candidate_space, dispatch_table, summary):
    low_efficiency_warnings = [
        {
            "shape_id": entry["shape_id"],
            "candidate_id": entry["candidate_id"],
            "selected_efficiency": entry["selected_efficiency"],
            "peak_tflops": entry["peak_tflops"],
            "warning": entry["efficiency_warning"],
        }
        for entry in dispatch_table["entries"]
        if entry.get("efficiency_warning")
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "candidate_count": len(candidate_space["candidates"]),
        "catalog_version": candidate_space.get("kernel_catalog", {}).get("catalog_version", ""),
        "dispatch_entries": len(dispatch_table["entries"]),
        "rows": summary["rows"],
        "passed": summary["passed"],
        "failed": summary["failed"],
        "candidate_dimension_coverage": build_candidate_coverage_report(candidate_space),
        "selected_dimension_values": summarize_dimension_values(
            [entry.get("selected_candidate_metadata", {}) for entry in dispatch_table["entries"]]
        ),
        "low_efficiency_warnings": low_efficiency_warnings,
    }


def build_reference_comparison(dispatch_table, reference_doc):
    reference_by_shape = {entry["shape_id"]: entry for entry in reference_doc.get("entries", []) if entry.get("supported", True)}
    dispatch_by_shape = {entry["shape_id"]: entry for entry in dispatch_table.get("entries", [])}
    entries = []
    matched = 0
    missing_dispatch = 0
    for shape_id, reference in reference_by_shape.items():
        dispatch_entry = dispatch_by_shape.get(shape_id)
        if dispatch_entry is None:
            missing_dispatch += 1
            entries.append(
                {
                    "shape_id": shape_id,
                    "reference_provider": reference["reference_provider"],
                    "reference_tflops": reference["reference_tflops"],
                    "selected_candidate_id": "",
                    "selected_candidate_metadata": {},
                    "selected_tflops": "",
                    "selected_vs_reference_ratio": "",
                    "status": "missing_dispatch",
                }
            )
            continue
        matched += 1
        selected_tflops = dispatch_entry["selected_metric"]
        reference_tflops = reference["reference_tflops"]
        ratio = round(selected_tflops / reference_tflops, 6) if reference_tflops else ""
        entries.append(
            {
                "shape_id": shape_id,
                "reference_provider": reference["reference_provider"],
                "reference_tflops": reference_tflops,
                "selected_candidate_id": dispatch_entry["candidate_id"],
                "selected_candidate_metadata": dispatch_entry.get("selected_candidate_metadata", {}),
                "selected_tflops": selected_tflops,
                "selected_vs_reference_ratio": ratio,
                "status": "matched",
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "dataset_id": reference_doc.get("dataset_id", ""),
        "summary": {
            "reference_entries": len(reference_by_shape),
            "matched": matched,
            "missing_dispatch": missing_dispatch,
        },
        "entries": entries,
    }
