#!/usr/bin/env python3
"""Compatibility wrapper for exact-shape report row helpers."""

from __future__ import annotations

try:
    from .exact_shape_report.rows import (
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
    from exact_shape_report.rows import (  # type: ignore
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
