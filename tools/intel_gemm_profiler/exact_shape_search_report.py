#!/usr/bin/env python3
"""Summarize remote exact-shape search results."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

try:
    from .exact_shape_search_report_artifacts import (
        infer_search_limitations,
        write_export_bundles,
        write_repro_artifacts,
    )
    from .exact_shape_search_report_rows import (
        iter_shape_tags,
        load_json,
        load_optional_json,
        load_run_meta,
        merged_fields,
        ok_rows,
        read_rows,
        safe_float,
        summarize_numeric,
        trim_rank_rows,
        write_csv,
    )
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from exact_shape_search_report_artifacts import (  # type: ignore
        infer_search_limitations,
        write_export_bundles,
        write_repro_artifacts,
    )
    from exact_shape_search_report_rows import (  # type: ignore
        iter_shape_tags,
        load_json,
        load_optional_json,
        load_run_meta,
        merged_fields,
        ok_rows,
        read_rows,
        safe_float,
        summarize_numeric,
        trim_rank_rows,
        write_csv,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize exact-shape search results.")
    parser.add_argument("--run-dir", required=True, help="Run directory created by tools/intel_gemm_profiler/remote_exact_shape_search.sh")
    parser.add_argument("--shape-tag", default="", help="Optional shape tag like 8192_384_3584. Defaults to all shapes.")
    parser.add_argument("--output-dir", default="", help="Optional report output directory. Defaults under run_dir/reports.")
    return parser.parse_args()


def summarize_shape(run_dir: Path, shape_tag: str, output_dir: Path) -> dict:
    kernel_metadata_path = run_dir / "kernel_metadata.json"
    kernel_metadata = load_json(kernel_metadata_path) if kernel_metadata_path.exists() else {}
    rows = read_rows((run_dir / "results" / shape_tag).glob("*.csv"), kernel_metadata, shape_tag)
    fieldnames = merged_fields(rows)
    status_counts = Counter(row.get("status", "") for row in rows)
    ranked_ok = ok_rows(rows)
    ranked_desc = sorted(ranked_ok, key=lambda row: row["_tflops"], reverse=True)
    ranked_asc = sorted(ranked_ok, key=lambda row: row["_tflops"])
    ranked_latency = sorted(
        [row for row in ranked_ok if row["_total_runtime_ms"] is not None],
        key=lambda row: row["_total_runtime_ms"],
    )
    ranked_latency_desc = list(reversed(ranked_latency))
    ranked_rcr = [row for row in ranked_desc if row.get("layout") == "rcr"]
    ranked_rcr_latency = [row for row in ranked_latency if row.get("layout") == "rcr"]

    merged_csv = output_dir / "merged_results.csv"
    ranked_by_tflops_csv = output_dir / "ranked_by_tflops.csv"
    ranked_by_total_runtime_csv = output_dir / "ranked_by_total_runtime.csv"
    top5_csv = output_dir / "top5.csv"
    worst5_csv = output_dir / "worst5.csv"
    top5_rcr_csv = output_dir / "top5_rcr.csv"
    fastest5_latency_csv = output_dir / "fastest5_latency.csv"
    slowest5_latency_csv = output_dir / "slowest5_latency.csv"
    fastest5_rcr_latency_csv = output_dir / "fastest5_rcr_latency.csv"
    write_csv(merged_csv, rows, fieldnames)
    write_csv(ranked_by_tflops_csv, trim_rank_rows(ranked_desc, fieldnames), fieldnames)
    write_csv(ranked_by_total_runtime_csv, trim_rank_rows(ranked_latency, fieldnames), fieldnames)
    write_csv(top5_csv, trim_rank_rows(ranked_desc[:5], fieldnames), fieldnames)
    write_csv(worst5_csv, trim_rank_rows(ranked_asc[:5], fieldnames), fieldnames)
    write_csv(top5_rcr_csv, trim_rank_rows(ranked_rcr[:5], fieldnames), fieldnames)
    write_csv(fastest5_latency_csv, trim_rank_rows(ranked_latency[:5], fieldnames), fieldnames)
    write_csv(slowest5_latency_csv, trim_rank_rows(ranked_latency_desc[:5], fieldnames), fieldnames)
    write_csv(fastest5_rcr_latency_csv, trim_rank_rows(ranked_rcr_latency[:5], fieldnames), fieldnames)

    run_meta_path = run_dir / "run_meta.txt"
    manifest_path = run_dir / "manifest.json"
    benchmark_config_path = run_dir / "benchmark_config.json"
    synced_sources_path = run_dir / "synced_sources.json"
    run_meta = load_run_meta(run_meta_path)
    benchmark_config = load_optional_json(benchmark_config_path)
    synced_sources = load_optional_json(synced_sources_path)
    repro_artifacts = write_repro_artifacts(
        output_dir,
        shape_tag,
        ranked_desc,
        run_meta,
        benchmark_config,
        synced_sources_path,
    )
    export_bundles = write_export_bundles(
        output_dir,
        shape_tag,
        ranked_desc,
        run_meta,
        benchmark_config,
        synced_sources_path,
    )

    summary = {
        "shape_tag": shape_tag,
        "row_count": len(rows),
        "ok_row_count": len(ranked_ok),
        "status_counts": dict(sorted(status_counts.items())),
        "latency_stats": {
            "avg_runtime_ms": summarize_numeric(ranked_ok, "avg_runtime_ms"),
            "total_runtime_ms": summarize_numeric(ranked_ok, "total_runtime_ms"),
        },
        "merged_fields": fieldnames,
        "kernel_metadata_path": str(kernel_metadata_path),
        "run_meta_path": str(run_meta_path),
        "run_meta": run_meta,
        "benchmark_config_path": str(benchmark_config_path),
        "benchmark_config": benchmark_config,
        "synced_sources_path": str(synced_sources_path),
        "synced_sources": synced_sources,
        "manifest_path": str(manifest_path),
        "manifest": load_json(manifest_path) if manifest_path.exists() else {},
        "search_limitations": infer_search_limitations(rows, run_meta),
        "merged_results_csv": str(merged_csv),
        "ranked_by_tflops_csv": str(ranked_by_tflops_csv),
        "ranked_by_total_runtime_csv": str(ranked_by_total_runtime_csv),
        "top5_csv": str(top5_csv),
        "worst5_csv": str(worst5_csv),
        "top5_rcr_csv": str(top5_rcr_csv),
        "fastest5_latency_csv": str(fastest5_latency_csv),
        "slowest5_latency_csv": str(slowest5_latency_csv),
        "fastest5_rcr_latency_csv": str(fastest5_rcr_latency_csv),
        "top5": trim_rank_rows(ranked_desc[:5], fieldnames),
        "worst5": trim_rank_rows(ranked_asc[:5], fieldnames),
        "top5_rcr": trim_rank_rows(ranked_rcr[:5], fieldnames),
        "fastest5_latency": trim_rank_rows(ranked_latency[:5], fieldnames),
        "slowest5_latency": trim_rank_rows(ranked_latency_desc[:5], fieldnames),
        "fastest5_rcr_latency": trim_rank_rows(ranked_rcr_latency[:5], fieldnames),
        "repro_artifacts": repro_artifacts,
        "export_bundles": export_bundles,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Run dir not found: {run_dir}")

    requested_shapes = run_dir / "requested_shapes.json"
    shape_tags = iter_shape_tags(run_dir, requested_shapes, args.shape_tag)
    if not shape_tags:
        raise SystemExit(f"No shape tags found in {run_dir}")

    base_output = Path(args.output_dir).resolve() if args.output_dir else (run_dir / "reports")
    summaries = []
    for shape_tag in shape_tags:
        result_dir = run_dir / "results" / shape_tag
        if not result_dir.is_dir():
            raise SystemExit(f"Result dir not found for shape {shape_tag}: {result_dir}")
        output_dir = base_output / shape_tag
        summaries.append(summarize_shape(run_dir, shape_tag, output_dir))

    print(json.dumps({"run_dir": str(run_dir), "shape_summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
