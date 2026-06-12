#!/usr/bin/env python3
"""Compatibility wrapper for exact-shape report export helpers."""

from __future__ import annotations

try:
    from .exact_shape_report.export import write_export_bundle, write_export_bundles
except ImportError:
    from exact_shape_report.export import write_export_bundle, write_export_bundles  # type: ignore


__all__ = [
    "write_export_bundle",
    "write_export_bundles",
]
