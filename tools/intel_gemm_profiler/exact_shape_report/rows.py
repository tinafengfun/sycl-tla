#!/usr/bin/env python3
"""Row loading and ranking helpers for exact-shape search reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Iterable


BASE_FIELDS = [
    "shape_tag",
    "batch_id",
    "result_csv",
    "kernel",
    "tflops",
    "avg_runtime_ms",
    "total_runtime_ms",
    "measure_iters",
    "warmup_iters",
    "input_mode",
    "workspace_bytes",
    "input_bytes_per_buffer",
    "input_pool_target_bytes",
    "input_pool_buffers",
    "fixed_vram_input",
    "prebuilt_variants",
    "workspace_reuse_enabled",
    "latency_source",
    "status",
    "gpu",
    "m",
    "n",
    "k",
]

PREFERRED_METADATA_FIELDS = [
    "kernel_name",
    "kernel_id",
    "layout",
    "runner",
    "benchmark_target",
    "scheduler_family",
    "operator_family",
    "tile_scheduler",
    "kernel_schedule",
    "mainloop_dispatch_policy",
    "epilogue_dispatch_policy",
    "decomposition_mode",
    "streamk_mode",
    "streamk_dtype_preset",
    "reduction_mode",
    "tile_m",
    "tile_n",
    "tile_k",
    "sg_m",
    "sg_n",
    "stages",
    "split_k",
    "grf_mode",
    "ilp_class",
    "batch_count",
    "allowed_runtime_sweeps",
    "runtime_defaults",
    "dtype_family",
    "dtype_a",
    "dtype_b",
    "dtype_c",
    "dtype_d",
    "dtype_acc",
    "mma_atom",
    "gmem_copy_atom_a",
    "gmem_copy_atom_b",
    "epilogue_op",
    "epilogue_tile",
    "epilogue_copy_atom_c",
    "epilogue_copy_atom_d",
    "element_output_epilogue",
    "element_compute_epilogue",
    "element_source_epilogue",
    "element_scalar_epilogue",
    "source",
    "instantiation_level",
    "support_status",
    "support_reason",
    "support_detail",
    "support_future_enable_condition",
    "example_family",
    "padding_mode",
    "activation",
    "bias_mode",
    "quant_mode",
    "scale_mode",
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_optional_json(path: Path) -> dict | list:
    if not path.exists():
        return {}
    return load_json(path)


def load_run_meta(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    doc: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        doc[key] = value
    return doc


def iter_shape_tags(run_dir: Path, requested_shapes: Path, explicit_shape_tag: str) -> list[str]:
    if explicit_shape_tag:
        return [explicit_shape_tag]
    if requested_shapes.exists():
        payload = load_json(requested_shapes)
        return [f"{shape['m']}_{shape['n']}_{shape['k']}" for shape in payload.get("shapes", [])]
    results_dir = run_dir / "results"
    return sorted(path.name for path in results_dir.iterdir() if path.is_dir())


def safe_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_scalar(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def derive_latency_fields(row: dict) -> dict:
    avg_runtime_ms = safe_float(row.get("avg_runtime_ms"))
    total_runtime_ms = safe_float(row.get("total_runtime_ms"))
    measure_iters = safe_int(row.get("measure_iters"))
    warmup_iters = safe_int(row.get("warmup_iters"))

    if avg_runtime_ms is not None or total_runtime_ms is not None:
        if measure_iters is None and avg_runtime_ms is not None and total_runtime_ms is not None and avg_runtime_ms > 0:
            measure_iters = int(round(total_runtime_ms / avg_runtime_ms))
        return {
            "avg_runtime_ms": "" if avg_runtime_ms is None else f"{avg_runtime_ms:.6f}",
            "total_runtime_ms": "" if total_runtime_ms is None else f"{total_runtime_ms:.6f}",
            "measure_iters": "" if measure_iters is None else str(measure_iters),
            "warmup_iters": "" if warmup_iters is None else str(warmup_iters),
            "latency_source": row.get("latency_source") or "reported",
        }

    tflops = safe_float(row.get("tflops"))
    m = safe_int(row.get("m"))
    n = safe_int(row.get("n"))
    k = safe_int(row.get("k"))
    if tflops is None or tflops <= 0 or m is None or n is None or k is None:
        return {
            "avg_runtime_ms": "",
            "total_runtime_ms": "",
            "measure_iters": "",
            "warmup_iters": "",
            "latency_source": "",
        }

    measure_iters = measure_iters or 100
    warmup_iters = warmup_iters or 100
    total_flops = 2.0 * float(m) * float(n) * float(k)
    avg_runtime_ms = (total_flops / (tflops * 1.0e12)) * 1.0e3
    total_runtime_ms = avg_runtime_ms * float(measure_iters)
    return {
        "avg_runtime_ms": f"{avg_runtime_ms:.6f}",
        "total_runtime_ms": f"{total_runtime_ms:.6f}",
        "measure_iters": str(measure_iters),
        "warmup_iters": str(warmup_iters),
        "latency_source": "derived_from_tflops",
    }


def read_rows(csv_paths: Iterable[Path], kernel_metadata: dict[str, dict], shape_tag: str) -> list[dict]:
    rows: list[dict] = []
    for csv_path in sorted(csv_paths):
        batch_id = csv_path.stem.rsplit("_gpu", 1)[0]
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                kernel = row.get("kernel", "")
                merged = {
                    "shape_tag": shape_tag,
                    "batch_id": batch_id,
                    "result_csv": str(csv_path),
                    "kernel": kernel,
                    "tflops": row.get("tflops", ""),
                    "status": row.get("status", ""),
                    "gpu": row.get("gpu", ""),
                    "m": row.get("m", ""),
                    "n": row.get("n", ""),
                    "k": row.get("k", ""),
                    "avg_runtime_ms": row.get("avg_runtime_ms", ""),
                    "total_runtime_ms": row.get("total_runtime_ms", ""),
                    "measure_iters": row.get("measure_iters", ""),
                    "warmup_iters": row.get("warmup_iters", ""),
                    "input_mode": row.get("input_mode", ""),
                    "workspace_bytes": row.get("workspace_bytes", ""),
                    "input_bytes_per_buffer": row.get("input_bytes_per_buffer", ""),
                    "input_pool_target_bytes": row.get("input_pool_target_bytes", ""),
                    "input_pool_buffers": row.get("input_pool_buffers", "") or row.get("pool_buffers", ""),
                    "fixed_vram_input": row.get("fixed_vram_input", ""),
                    "prebuilt_variants": row.get("prebuilt_variants", ""),
                    "workspace_reuse_enabled": row.get("workspace_reuse_enabled", ""),
                    "latency_source": row.get("latency_source", ""),
                }
                merged.update({key: normalize_scalar(value) for key, value in kernel_metadata.get(kernel, {}).items()})
                merged.update(derive_latency_fields(merged))
                rows.append(merged)
    return rows


def ok_rows(rows: Iterable[dict]) -> list[dict]:
    ranked = []
    for row in rows:
        if row.get("status") != "OK":
            continue
        tflops = safe_float(row.get("tflops"))
        if tflops is None:
            continue
        avg_runtime_ms = safe_float(row.get("avg_runtime_ms"))
        total_runtime_ms = safe_float(row.get("total_runtime_ms"))
        ranked.append(
            {
                **row,
                "_tflops": tflops,
                "_avg_runtime_ms": avg_runtime_ms,
                "_total_runtime_ms": total_runtime_ms,
            }
        )
    return ranked


def merged_fields(rows: list[dict]) -> list[str]:
    extras = set()
    for row in rows:
        extras.update(row.keys())
    extras.difference_update(BASE_FIELDS)

    ordered = []
    for field in PREFERRED_METADATA_FIELDS:
        if field in extras:
            ordered.append(field)
            extras.remove(field)
    ordered.extend(sorted(extras))
    return BASE_FIELDS + ordered


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def trim_rank_rows(rows: list[dict], fieldnames: list[str]) -> list[dict]:
    trimmed = []
    for row in rows:
        copy = {field: row.get(field, "") for field in fieldnames}
        trimmed.append(copy)
    return trimmed


def summarize_numeric(rows: list[dict], field: str) -> dict:
    values = [safe_float(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return {}
    return {
        "min": round(min(values), 6),
        "median": round(median(values), 6),
        "mean": round(mean(values), 6),
        "max": round(max(values), 6),
    }


__all__ = [
    "iter_shape_tags",
    "load_json",
    "load_optional_json",
    "load_run_meta",
    "merged_fields",
    "ok_rows",
    "read_rows",
    "safe_float",
    "summarize_numeric",
    "trim_rank_rows",
    "write_csv",
]
