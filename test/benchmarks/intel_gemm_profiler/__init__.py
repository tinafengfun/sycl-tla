#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

"""Compatibility package shim for the relocated Intel GEMM profiler implementation."""

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[3] / "tools" / "intel_gemm_profiler"
PACKAGE_INIT = PACKAGE_DIR / "__init__.py"

__path__ = [str(PACKAGE_DIR)]
__file__ = str(PACKAGE_INIT)

exec(compile(PACKAGE_INIT.read_text(encoding="utf-8"), str(PACKAGE_INIT), "exec"))
