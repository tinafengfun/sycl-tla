#!/usr/bin/env python3
"""Compatibility wrapper for exact-shape report repro helpers."""

from __future__ import annotations

try:
    from .exact_shape_report.repro import (
        build_repro_manifest,
        format_numeric_cli_arg,
        infer_search_limitations,
        make_benchmark_config_line,
        row_runtime_arguments,
        strip_internal_fields,
        write_repro_artifacts,
        write_repro_config,
        write_repro_filter,
    )
except ImportError:
    from exact_shape_report.repro import (  # type: ignore
        build_repro_manifest,
        format_numeric_cli_arg,
        infer_search_limitations,
        make_benchmark_config_line,
        row_runtime_arguments,
        strip_internal_fields,
        write_repro_artifacts,
        write_repro_config,
        write_repro_filter,
    )
__all__ = [
    "build_repro_manifest",
    "format_numeric_cli_arg",
    "infer_search_limitations",
    "make_benchmark_config_line",
    "row_runtime_arguments",
    "strip_internal_fields",
    "write_repro_artifacts",
    "write_repro_config",
    "write_repro_filter",
]
