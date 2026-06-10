#!/usr/bin/env python3
"""Compatibility wrapper for the relocated Intel profiler exact-shape report script."""

from pathlib import Path
import runpy


runpy.run_path(
    str(Path(__file__).resolve().parent / "intel_gemm_profiler" / "exact_shape_search_report.py"),
    run_name="__main__",
)
