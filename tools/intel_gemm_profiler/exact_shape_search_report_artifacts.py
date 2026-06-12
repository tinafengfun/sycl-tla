#!/usr/bin/env python3
"""Compatibility exports for exact-shape report artifact writers."""

from __future__ import annotations

try:
    from .exact_shape_search_report_export import write_export_bundle, write_export_bundles
    from .exact_shape_search_report_repro import infer_search_limitations, write_repro_artifacts
except ImportError:
    from exact_shape_search_report_export import write_export_bundle, write_export_bundles  # type: ignore
    from exact_shape_search_report_repro import infer_search_limitations, write_repro_artifacts  # type: ignore

__all__ = [
    "infer_search_limitations",
    "write_export_bundle",
    "write_export_bundles",
    "write_repro_artifacts",
]
