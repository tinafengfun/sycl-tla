#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

from pathlib import Path

from .schemas import SCHEMA_VERSION
from .utils import read_json, write_json


DEFAULT_BUILD_CONFIG_PATH = Path(__file__).resolve().parent / "build_config_bmg_perf.json"
DEFAULT_RUNTIME_CONFIG_PATH = Path(__file__).resolve().parent / "runtime_config_bmg_perf.json"


def _default_build_config():
    return {
        "schema_version": SCHEMA_VERSION,
        "device_arch": "bmg",
        "purpose": "optimal_performance_profiling",
        "cmake_vars": {
            "CUTLASS_ENABLE_SYCL": "ON",
            "DPCPP_SYCL_TARGET": "auto",
            "DPCPP_HOST_COMPILER": "g++-13",
            "CMAKE_BUILD_TYPE": "Release",
            "CUTLASS_SYCL_PROFILING_ENABLED": "OFF",
            "CUTLASS_ENABLE_BENCHMARKS": "ON",
            "CUTLASS_ENABLE_EXAMPLES": "OFF",
            "CUTLASS_ENABLE_TESTS": "ON",
        },
        "device_target_detection": {
            "mode": "auto",
            "cmake_var": "DPCPP_SYCL_TARGET",
            "fallback_target": "bmg",
            "strict": False,
            "selected_device_env": "ZE_AFFINITY_MASK",
        },
        "compile_env": {
            "CC": "icx",
            "CXX": "icpx",
            "IGC_ExtraOCLOptions": "-cl-intel-256-GRF-per-thread",
            "IGC_VectorAliasBBThreshold": "10000",
            "SYCL_PROGRAM_COMPILE_OPTIONS": "-ze-opt-large-register-file -gline-tables-only",
        },
        "compile_env_variants": {
            "perf_default": {
                "IGC_ExtraOCLOptions": "-cl-intel-256-GRF-per-thread",
                "IGC_VectorAliasBBThreshold": "10000",
                "SYCL_PROGRAM_COMPILE_OPTIONS": "-ze-opt-large-register-file -gline-tables-only",
            },
            "perf_perfmodel": {
                "IGC_ExtraOCLOptions": "-cl-intel-256-GRF-per-thread",
                "IGC_VectorAliasBBThreshold": "10000",
                "IGC_VISAOptions": "-perfmodel",
                "SYCL_PROGRAM_COMPILE_OPTIONS": "-ze-opt-large-register-file -gline-tables-only",
            },
            "perf_128grf_experiment": {
                "IGC_VectorAliasBBThreshold": "10000",
                "IGC_TotalGRFNum": "128",
            },
            "perf_enableBCR": {
                "IGC_ExtraOCLOptions": "-cl-intel-256-GRF-per-thread",
                "IGC_VectorAliasBBThreshold": "10000",
                "IGC_VISAOptions": "-enableBCR",
                "SYCL_PROGRAM_COMPILE_OPTIONS": "-ze-opt-large-register-file -gline-tables-only",
            },
            "debug_with_lines": {
                "IGC_ExtraOCLOptions": "-cl-intel-256-GRF-per-thread",
                "IGC_VectorAliasBBThreshold": "10000",
                "SYCL_PROGRAM_COMPILE_OPTIONS": "-ze-opt-large-register-file -gline-tables-only",
            },
        },
        "compile_env_variant_metadata": {
            "perf_default": {
                "status": "validated",
                "notes": "Validated on BMG G31: g++-13 host compiler, 256-GRF, large-register-file, and line tables at compile and runtime.",
            },
            "perf_perfmodel": {
                "status": "experimental",
                "notes": "Optional perfmodel sweep. Not selected by default and not enforced by the workflow.",
            },
            "perf_128grf_experiment": {
                "status": "needs_validation",
                "notes": (
                    "Experimental 128-GRF trial. Leaves both 256-GRF hints unset on purpose: "
                    "do not pass -cl-intel-256-GRF-per-thread and do not pass "
                    "-ze-opt-large-register-file. Advisory only; do not use in production until "
                    "B60 A/B validation confirms benefit over perf_default."
                ),
            },
            "perf_enableBCR": {
                "status": "experimental",
                "notes": "Optional enableBCR sweep. Not selected by default and not enforced by the workflow.",
            },
            "debug_with_lines": {
                "status": "debug_only",
                "notes": "Adds line tables for debug or profiling workflows.",
            },
        },
        "selected_compile_variant": "perf_default",
    }


def _default_runtime_config():
    return {
        "schema_version": SCHEMA_VERSION,
        "device_arch": "bmg",
        "runtime_env": {
            "ONEAPI_DEVICE_SELECTOR": "level_zero:gpu",
            "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS": "1",
            "ZE_FLAT_DEVICE_HIERARCHY": "COMPOSITE",
            "IGC_ExtraOCLOptions": "-cl-intel-256-GRF-per-thread",
            "IGC_VectorAliasBBThreshold": "10000",
            "SYCL_PROGRAM_COMPILE_OPTIONS": "-ze-opt-large-register-file -gline-tables-only",
        },
        "runtime_env_variants": {
            "default": {
                "ONEAPI_DEVICE_SELECTOR": "level_zero:gpu",
                "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS": "1",
                "ZE_FLAT_DEVICE_HIERARCHY": "COMPOSITE",
                "IGC_ExtraOCLOptions": "-cl-intel-256-GRF-per-thread",
                "IGC_VectorAliasBBThreshold": "10000",
                "SYCL_PROGRAM_COMPILE_OPTIONS": "-ze-opt-large-register-file -gline-tables-only",
            },
            "ze_affinity_0": {
                "ONEAPI_DEVICE_SELECTOR": "level_zero:gpu",
                "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS": "1",
                "ZE_FLAT_DEVICE_HIERARCHY": "COMPOSITE",
                "IGC_ExtraOCLOptions": "-cl-intel-256-GRF-per-thread",
                "IGC_VectorAliasBBThreshold": "10000",
                "SYCL_PROGRAM_COMPILE_OPTIONS": "-ze-opt-large-register-file -gline-tables-only",
                "ZE_AFFINITY_MASK": "0",
            },
            "ze_affinity_1": {
                "ONEAPI_DEVICE_SELECTOR": "level_zero:gpu",
                "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS": "1",
                "ZE_FLAT_DEVICE_HIERARCHY": "COMPOSITE",
                "IGC_ExtraOCLOptions": "-cl-intel-256-GRF-per-thread",
                "IGC_VectorAliasBBThreshold": "10000",
                "SYCL_PROGRAM_COMPILE_OPTIONS": "-ze-opt-large-register-file -gline-tables-only",
                "ZE_AFFINITY_MASK": "1",
            },
            "ze_affinity_7": {
                "ONEAPI_DEVICE_SELECTOR": "level_zero:gpu",
                "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS": "1",
                "ZE_FLAT_DEVICE_HIERARCHY": "COMPOSITE",
                "IGC_ExtraOCLOptions": "-cl-intel-256-GRF-per-thread",
                "IGC_VectorAliasBBThreshold": "10000",
                "SYCL_PROGRAM_COMPILE_OPTIONS": "-ze-opt-large-register-file -gline-tables-only",
                "ZE_AFFINITY_MASK": "7",
            },
        },
        "selected_runtime_variant": "default",
    }


def load_persisted_build_config(path=DEFAULT_BUILD_CONFIG_PATH):
    return read_json(path) if path.exists() else _default_build_config()


def load_persisted_runtime_config(path=DEFAULT_RUNTIME_CONFIG_PATH):
    return read_json(path) if path.exists() else _default_runtime_config()


def update_build_config_variant(variant_name, path=DEFAULT_BUILD_CONFIG_PATH):
    config = load_persisted_build_config(path)
    config["selected_compile_variant"] = variant_name
    write_json(path, config)
    return config


def update_runtime_config_variant(variant_name, path=DEFAULT_RUNTIME_CONFIG_PATH):
    config = load_persisted_runtime_config(path)
    config["selected_runtime_variant"] = variant_name
    write_json(path, config)
    return config


def list_compile_variants(path=DEFAULT_BUILD_CONFIG_PATH):
    config = load_persisted_build_config(path)
    variants = config.get("compile_env_variants", {})
    metadata = config.get("compile_env_variant_metadata", {})
    result = []
    for name, env in variants.items():
        meta = metadata.get(name, {})
        result.append(
            {
                "name": name,
                "status": meta.get("status", "unknown"),
                "notes": meta.get("notes", ""),
                "env_keys": sorted(env.keys()),
            }
        )
    return result


def list_runtime_variants(path=DEFAULT_RUNTIME_CONFIG_PATH):
    config = load_persisted_runtime_config(path)
    variants = config.get("runtime_env_variants", {})
    return [{"name": name, "env_keys": sorted(env.keys())} for name, env in variants.items()]


def selected_runtime_env(profiles, profile=None, variant_override=None):
    runtime_config = profiles.get("runtime_config", {})
    runtime_env = dict(runtime_config.get("runtime_env", {}))
    selected_variant = variant_override or runtime_config.get("selected_runtime_variant")
    variant_overrides = runtime_config.get("runtime_env_variants", {}).get(selected_variant, {})
    runtime_env.update(variant_overrides)
    if profile:
        runtime_env.update(profile.get("runtime_env_override", {}))
    return runtime_env


def selected_compile_env(profiles, variant_override=None):
    build_config = profiles.get("build_config", {})
    compile_env = dict(build_config.get("compile_env", {}))
    selected_variant = variant_override or build_config.get("selected_compile_variant")
    variant_overrides = build_config.get("compile_env_variants", {}).get(selected_variant, {})
    compile_env.update(variant_overrides)
    return compile_env
