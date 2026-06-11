#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import re
from pathlib import Path

from .schemas import BENCHMARK_ERROR_RE, RESULT_METADATA_FIELDS, infer_epilogue_metadata, infer_scheduler_metadata


def parse_metric(line, key):
    match = re.search(rf"{re.escape(key)}=([0-9.]+)", line)
    return match.group(1) if match else ""


def row_result_metadata(metadata):
    defaults = infer_scheduler_metadata(metadata)
    defaults.update(
        {
            "runner": "benchmark",
            "benchmark_target": "",
            "streamk_mode": "",
            "support_status": "supported",
            "support_reason": "",
            "mma_atom": "XE_DPAS_TT",
            "gmem_copy_atom_a": "auto",
            "gmem_copy_atom_b": "auto",
            "epilogue_op": "LinearCombination",
            "epilogue_tile": "auto",
            "epilogue_copy_atom_c": "auto",
            "epilogue_copy_atom_d": "auto",
        }
    )
    defaults.update(infer_epilogue_metadata(metadata))
    return {field: metadata.get(field, defaults.get(field, "")) for field in RESULT_METADATA_FIELDS}


def with_result_metadata(row, metadata):
    row.update(row_result_metadata(metadata))
    return row


def parse_benchmark_log(log_path, metadata_by_bm_name, run_id):
    rows = []
    text = Path(log_path).read_text(encoding="utf-8")
    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if "manual_time" not in line and not BENCHMARK_ERROR_RE.search(stripped):
                continue
            parts = stripped.split()
            if not parts:
                continue
            token = parts[0]
            segments = token.split("/")
            if len(segments) < 2:
                continue
            metadata = metadata_by_bm_name.get(segments[1])
            if not metadata:
                continue
            failure = bool(BENCHMARK_ERROR_RE.search(stripped))
            rows.append(
                with_result_metadata(
                    {
                        "run_id": run_id,
                        "stage": metadata["stage"],
                        "attempt_index": metadata["attempt_index"],
                        "shape_id": metadata["shape_id"],
                        "candidate_id": metadata["candidate_id"],
                        "compiler_profile_id": metadata["compiler_profile_id"],
                        "status": "fail" if failure else "pass",
                        "verify_status": "fail" if failure else "pass",
                        "layout": metadata["layout"],
                        "dtype_a": metadata["dtype_a"],
                        "dtype_b": metadata["dtype_b"],
                        "dtype_c": metadata["dtype_c"],
                        "dtype_d": metadata.get("dtype_d", metadata["dtype_c"]),
                        "dtype_acc": metadata["dtype_acc"],
                        "m": metadata["m"],
                        "n": metadata["n"],
                        "k": metadata["k"],
                        "batch_count": metadata.get("batch_count", 1),
                        "split_k": metadata.get("split_k", 1),
                        "avg_runtime_ms": parse_metric(stripped, "runtime_trimmed_mean_ms") or parse_metric(stripped, "avg_runtime_ms"),
                        "best_runtime_ms": parse_metric(stripped, "runtime_min_ms") or parse_metric(stripped, "best_runtime_ms"),
                        "worst_runtime_ms": parse_metric(stripped, "runtime_max_ms") or parse_metric(stripped, "worst_runtime_ms"),
                        "runtime_median_ms": parse_metric(stripped, "runtime_median_ms"),
                        "runtime_stddev_ms": parse_metric(stripped, "runtime_stddev_ms"),
                        "warmup_iters": parse_metric(stripped, "warmup_iters"),
                        "measure_iters": parse_metric(stripped, "measure_iters"),
                        "avg_tflops": parse_metric(stripped, "avg_tflops"),
                        "median_tflops": parse_metric(stripped, "median_tflops"),
                        "avg_throughput": parse_metric(stripped, "avg_throughput"),
                        "max_error": "",
                        "close_call_group": "",
                        "failure_reason": stripped if failure else "",
                        "stdout_log": str(log_path),
                    },
                    metadata,
                )
            )
    if not rows and "Benchmark not found" in text:
        for _, metadata in metadata_by_bm_name.items():
            rows.append(
                with_result_metadata(
                    {
                        "run_id": run_id,
                        "stage": metadata["stage"],
                        "attempt_index": metadata["attempt_index"],
                        "shape_id": metadata["shape_id"],
                        "candidate_id": metadata["candidate_id"],
                        "compiler_profile_id": metadata["compiler_profile_id"],
                        "status": "fail",
                        "verify_status": "fail",
                        "layout": metadata["layout"],
                        "dtype_a": metadata["dtype_a"],
                        "dtype_b": metadata["dtype_b"],
                        "dtype_c": metadata["dtype_c"],
                        "dtype_d": metadata.get("dtype_d", metadata["dtype_c"]),
                        "dtype_acc": metadata["dtype_acc"],
                        "m": metadata["m"],
                        "n": metadata["n"],
                        "k": metadata["k"],
                        "batch_count": metadata.get("batch_count", 1),
                        "split_k": metadata.get("split_k", 1),
                        "avg_runtime_ms": "",
                        "best_runtime_ms": "",
                        "worst_runtime_ms": "",
                        "runtime_median_ms": "",
                        "runtime_stddev_ms": "",
                        "warmup_iters": "",
                        "measure_iters": "",
                        "avg_tflops": "",
                        "median_tflops": "",
                        "avg_throughput": "",
                        "max_error": "",
                        "close_call_group": "",
                        "failure_reason": "benchmark registry entry not found for generated kernel",
                        "stdout_log": str(log_path),
                    },
                    metadata,
                )
            )
    return rows


def timeout_rows(entries, log_path, reason):
    rows = []
    for entry in entries:
        candidate = entry["candidate"]
        shape = entry["shape"]
        rows.append(
            with_result_metadata(
                {
                    "run_id": entry["stage"],
                    "stage": entry["stage"],
                    "attempt_index": entry["attempt_index"],
                    "shape_id": shape["shape_id"],
                    "candidate_id": candidate["candidate_id"],
                    "compiler_profile_id": candidate["compiler_profile_id"],
                    "status": "timeout",
                    "verify_status": "fail",
                    "layout": shape["layout"],
                    "dtype_a": shape["dtype_a"],
                    "dtype_b": shape["dtype_b"],
                    "dtype_c": shape["dtype_c"],
                    "dtype_d": shape.get("dtype_d", shape["dtype_c"]),
                    "dtype_acc": shape["dtype_acc"],
                    "m": shape["m"],
                    "n": shape["n"],
                    "k": shape["k"],
                    "batch_count": shape.get("batch_count", 1),
                    "split_k": candidate.get("split_k", 1),
                    "avg_runtime_ms": "",
                    "best_runtime_ms": "",
                    "worst_runtime_ms": "",
                    "avg_tflops": "",
                    "avg_throughput": "",
                    "max_error": "",
                    "close_call_group": "",
                    "failure_reason": reason,
                    "stdout_log": str(log_path),
                },
                candidate,
            )
        )
    return rows
