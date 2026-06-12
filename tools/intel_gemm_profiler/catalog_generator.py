#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

from __future__ import annotations

import sys
from pathlib import Path

from .catalog_space import apply_epilogue_metadata, apply_scheduler_metadata, ilp_class
from .schemas import SCHEMA_VERSION, SEARCH_RUNTIME_SCHEMA


def _import_cutlass_generator_modules():
    try:
        from cutlass_library.generator import GenerateIntelXe
        from cutlass_library.library import DataTypeNames, LayoutType, TileSchedulerType
        from cutlass_library.manifest import Manifest, Options
    except ImportError:
        python_root = Path(__file__).resolve().parents[2] / "python"
        if str(python_root) not in sys.path:
            sys.path.insert(0, str(python_root))
        from cutlass_library.generator import GenerateIntelXe
        from cutlass_library.library import DataTypeNames, LayoutType, TileSchedulerType
        from cutlass_library.manifest import Manifest, Options
    return GenerateIntelXe, DataTypeNames, LayoutType, TileSchedulerType, Manifest, Options


def _generator_arch_details(generator_arch):
    arch_key = str(generator_arch).lower()
    arch_map = {
        "bmg": ("bmg", 20),
        "xe20": ("bmg", 20),
        "20": ("bmg", 20),
        "pvc": ("pvc", 12),
        "xe12": ("pvc", 12),
        "12": ("pvc", 12),
    }
    if arch_key not in arch_map:
        raise ValueError(f"Unsupported Intel Xe generator arch: {generator_arch}")
    return arch_map[arch_key]


def _generator_layout_name(layout_type):
    return "r" if layout_type.name.startswith("RowMajor") else "c"


def _generator_dtype_family(dtype_a):
    if dtype_a in {"bf16", "f16"}:
        return "16b"
    if dtype_a in {"e4m3", "e5m2"}:
        return "fp8"
    if dtype_a in {"s8"}:
        return "int8"
    return dtype_a


def _generator_kernel_catalog_entry(operation, data_type_names, tile_scheduler_type, instantiation_level):
    tile_shape = operation.tile_description.tile_shape
    sg_shape = operation.tile_description.warp_count
    streamk_mode = "streamk" if operation.tile_scheduler == tile_scheduler_type.StreamK else ""
    dtype_a = data_type_names[operation.A.element]
    dtype_b = data_type_names[operation.B.element]
    dtype_c = data_type_names[operation.C.element]
    dtype_d = data_type_names[getattr(operation, "D", operation.C).element]
    dtype_acc = data_type_names[operation.accumulator_type()]
    entry = {
        "kernel_name": operation.procedural_name(),
        "layout": "".join(
            _generator_layout_name(layout_type)
            for layout_type in (operation.A.layout, operation.B.layout, operation.C.layout)
        ),
        "dtype_a": dtype_a,
        "dtype_b": dtype_b,
        "dtype_c": dtype_c,
        "dtype_d": dtype_d,
        "dtype_acc": dtype_acc,
        "tile_m": tile_shape[0],
        "tile_n": tile_shape[1],
        "tile_k": tile_shape[2],
        "sg_m": sg_shape[0],
        "sg_n": sg_shape[1],
        "stages": int(operation.tile_description.stages),
        "split_k": 1,
        "runner": "benchmark",
        "kernel_id": operation.procedural_name(),
        "instantiation_level": instantiation_level,
        "benchmark_target": "cutlass_benchmarks_gemm_sycl",
        "grf_mode": 256,
        "streamk_mode": streamk_mode,
        "streamk_dtype_preset": "",
        "support_status": "supported",
        "support_reason": "",
        "runtime_defaults": {},
        "allowed_runtime_sweeps": ["shape_id", "m", "n", "k", "batch_count"],
        "source": "generator_manifest",
        "mma_atom": "XE_DPAS_TT",
        "gmem_copy_atom_a": "auto",
        "gmem_copy_atom_b": "auto",
        "epilogue_op": "LinearCombination",
        "epilogue_tile": "auto",
        "epilogue_copy_atom_c": "auto",
        "epilogue_copy_atom_d": "auto",
    }
    apply_epilogue_metadata(entry)
    apply_scheduler_metadata(entry)
    entry["ilp_class"] = ilp_class(entry)
    entry["dtype_family"] = _generator_dtype_family(dtype_a)
    return entry


def generated_generator_kernel_catalog(generator_arch="bmg", generator_instantiation_level=0):
    GenerateIntelXe, DataTypeNames, _, TileSchedulerType, Manifest, Options = _import_cutlass_generator_modules()
    arch_name, arch_value = _generator_arch_details(generator_arch)
    args = Options()
    args.kernels = ""
    args.curr_build_dir = "."
    args.architectures = arch_name
    args.filter_by_cc = "true"
    args.operations = "gemm"
    args.ignore_kernels = ""
    args.exclude_kernels = ""
    args.kernel_filter_file = None
    args.disable_full_archs_compilation = False
    args.instantiation_level = str(generator_instantiation_level)
    manifest = Manifest(args)
    GenerateIntelXe(manifest, cuda_version=None, arch=arch_value)
    kernels = []
    for operation in sorted(manifest.operations_by_name.values(), key=lambda op: op.procedural_name()):
        kernels.append(
            _generator_kernel_catalog_entry(
                operation,
                DataTypeNames,
                TileSchedulerType,
                int(generator_instantiation_level),
            )
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "catalog_version": f"generator-{arch_name}-level{int(generator_instantiation_level)}",
        "instantiation_levels": {
            "0": "generator conservative Intel Xe catalog",
            "1": "expanded tile, stage, and scheduler Intel Xe catalog",
            "2": "generator-expanded Intel Xe catalog including fp8/int8 families",
        },
        "generator_arch": arch_name,
        "generator_instantiation_level": int(generator_instantiation_level),
        "search_runtime_schema": SEARCH_RUNTIME_SCHEMA,
        "kernels": kernels,
    }
