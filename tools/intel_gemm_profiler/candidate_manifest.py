#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

from .schemas import SCHEMA_VERSION, SEARCH_RUNTIME_SCHEMA
from .utils import now_iso


def build_selected_kernel_batches(selected_kernel_list, batch_size):
    if batch_size <= 0:
        return []
    batches = []
    for batch_index, start in enumerate(range(0, len(selected_kernel_list), batch_size)):
        kernels = selected_kernel_list[start:start + batch_size]
        batches.append(
            {
                "batch_id": f"selected_kernel_batch_{batch_index:03d}",
                "batch_index": batch_index,
                "kernel_count": len(kernels),
                "selected_kernel_list": kernels,
                "kernel_filter_file": {
                    "format": "python-regex-lines",
                    "recommended_cmake_var": "KERNEL_FILTER_FILE",
                    "lines": [f"^{kernel_id}$" for kernel_id in kernels],
                },
            }
        )
    return batches


def build_candidate_build_variant(candidate):
    return {
        "candidate_id": candidate["candidate_id"],
        "kernel_id": candidate["kernel_id"],
        "benchmark_target": candidate["benchmark_target"],
        "runner": candidate["runner"],
        "compile_time_variant": {
            "layout": candidate["layout"],
            "dtype_a": candidate["dtype_a"],
            "dtype_b": candidate["dtype_b"],
            "dtype_c": candidate["dtype_c"],
            "dtype_acc": candidate["dtype_acc"],
            "dtype_d": candidate.get("dtype_d", candidate["dtype_c"]),
            "tile_m": candidate["tile_m"],
            "tile_n": candidate["tile_n"],
            "tile_k": candidate["tile_k"],
            "sg_m": candidate["sg_m"],
            "sg_n": candidate["sg_n"],
            "stages": candidate["stages"],
            "split_k": candidate["split_k"],
            "streamk_mode": candidate.get("streamk_mode", ""),
            "streamk_dtype_preset": candidate.get("streamk_dtype_preset", ""),
            "scheduler_family": candidate.get("scheduler_family", ""),
            "operator_family": candidate.get("operator_family", ""),
            "grf_mode": candidate["grf_mode"],
            "mma_atom": candidate.get("mma_atom", "XE_DPAS_TT"),
            "gmem_copy_atom_a": candidate.get("gmem_copy_atom_a", "auto"),
            "gmem_copy_atom_b": candidate.get("gmem_copy_atom_b", "auto"),
            "element_output_epilogue": candidate.get("element_output_epilogue", ""),
            "element_compute_epilogue": candidate.get("element_compute_epilogue", ""),
            "element_source_epilogue": candidate.get("element_source_epilogue", ""),
            "element_scalar_epilogue": candidate.get("element_scalar_epilogue", ""),
            "epilogue_op": candidate.get("epilogue_op", "LinearCombination"),
            "epilogue_tile": candidate.get("epilogue_tile", "auto"),
            "epilogue_copy_atom_c": candidate.get("epilogue_copy_atom_c", "auto"),
            "epilogue_copy_atom_d": candidate.get("epilogue_copy_atom_d", "auto"),
            "mainloop_dispatch_policy": candidate.get("mainloop_dispatch_policy", ""),
            "kernel_schedule": candidate.get("kernel_schedule", ""),
            "tile_scheduler": candidate.get("tile_scheduler", ""),
            "decomposition_mode": candidate.get("decomposition_mode", ""),
            "reduction_mode": candidate.get("reduction_mode", ""),
            "epilogue_dispatch_policy": candidate.get("epilogue_dispatch_policy", ""),
            "example_family": candidate.get("example_family", ""),
            "padding_mode": candidate.get("padding_mode", ""),
            "activation": candidate.get("activation", ""),
            "bias_mode": candidate.get("bias_mode", ""),
            "quant_mode": candidate.get("quant_mode", ""),
            "scale_mode": candidate.get("scale_mode", ""),
            "ilp_class": candidate["ilp_class"],
            "instantiation_level": candidate["instantiation_level"],
        },
        "runtime_sweep": {"allowed_fields": candidate["allowed_runtime_sweeps"], "defaults": candidate["runtime_defaults"]},
        "compiler_profile_id": candidate["compiler_profile_id"],
    }


def build_candidate_build_cmake_vars(candidate_space, build_config=None):
    kernel_catalog = candidate_space.get("kernel_catalog", {})
    catalog_source = kernel_catalog.get("catalog_source", "persisted")
    generator_level = int(kernel_catalog.get("generator_instantiation_level", 0)) if catalog_source == "generator" else 0
    cmake_vars = {
        "CUTLASS_ENABLE_SYCL": "ON",
        "CUTLASS_ENABLE_BENCHMARKS": "ON",
        "CUTLASS_ENABLE_TESTS": "OFF",
        "CUTLASS_ENABLE_EXAMPLES": "OFF",
        "CUTLASS_LIBRARY_OPERATIONS": "gemm",
        "CUTLASS_LIBRARY_INSTANTIATION_LEVEL": str(generator_level),
        "SYCL_INTEL_TARGET": candidate_space["device_arch"],
        "BENCHMARK_ENABLE_TESTING": "OFF",
        "BENCHMARK_ENABLE_GTEST_TESTS": "OFF",
    }
    cmake_vars.update((build_config or {}).get("cmake_vars", {}))
    cmake_vars["CUTLASS_LIBRARY_OPERATIONS"] = "gemm"
    cmake_vars["CUTLASS_LIBRARY_INSTANTIATION_LEVEL"] = str(generator_level)
    cmake_vars["BENCHMARK_ENABLE_TESTING"] = "OFF"
    cmake_vars["BENCHMARK_ENABLE_GTEST_TESTS"] = "OFF"
    if catalog_source in {"expanded_streamk", "expanded_bmg", "layered_bmg", "layered_bmg_scheduler_expanded"}:
        cmake_vars["CUTLASS_BENCHMARK_EXPANDED_BMG_STREAMK"] = "ON"
    if catalog_source == "layered_bmg":
        cmake_vars["CUTLASS_BENCHMARK_EXHAUSTIVE_GEMM"] = "ON"
    if catalog_source == "layered_bmg_scheduler_expanded":
        cmake_vars["CUTLASS_BENCHMARK_EXHAUSTIVE_GEMM"] = "ON"
        cmake_vars["CUTLASS_BENCHMARK_EXHAUSTIVE_STREAMK"] = "ON"
    if cmake_vars.get("DPCPP_SYCL_TARGET"):
        cmake_vars.pop("SYCL_INTEL_TARGET", None)
    return cmake_vars


def build_candidate_build_manifest(candidate_space, selected_kernel_batch_size=0, build_config=None):
    selected_kernel_list = []
    seen_kernel_ids = set()
    variants = []
    for candidate in candidate_space["candidates"]:
        if candidate["runner"] == "benchmark" and candidate["kernel_id"] not in seen_kernel_ids:
            seen_kernel_ids.add(candidate["kernel_id"])
            selected_kernel_list.append(candidate["kernel_id"])
        variants.append(build_candidate_build_variant(candidate))

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "device_arch": candidate_space["device_arch"],
        "constraint_source": candidate_space["constraint_source"],
        "search_runtime_schema": SEARCH_RUNTIME_SCHEMA,
        "kernel_catalog": candidate_space.get("kernel_catalog", {}),
        "selected_kernel_list": selected_kernel_list,
        "selected_kernel_count": len(selected_kernel_list),
        "kernel_filter_file": {
            "format": "python-regex-lines",
            "recommended_cmake_var": "KERNEL_FILTER_FILE",
            "lines": [f"^{kernel_id}$" for kernel_id in selected_kernel_list],
        },
        "selected_kernel_batch_size": selected_kernel_batch_size,
        "selected_kernel_batches": build_selected_kernel_batches(selected_kernel_list, selected_kernel_batch_size),
        "cmake_config": {
            "build_target": "cutlass_benchmarks_gemm_sycl",
            "cmake_vars": build_candidate_build_cmake_vars(candidate_space, build_config=build_config),
            "kernel_filter_cmake_var": "KERNEL_FILTER_FILE",
        },
        "variants": variants,
    }
