#!/usr/bin/env python3
"""Compatibility wrapper for the relocated Intel profiler gen_mini_hpp script."""

from pathlib import Path
import runpy


runpy.run_path(
    str(Path(__file__).resolve().parent / "intel_gemm_profiler" / "gen_mini_hpp.py"),
    run_name="__main__",
)
