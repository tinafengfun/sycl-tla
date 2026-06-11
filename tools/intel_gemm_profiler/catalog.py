#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import copy
import sys
from pathlib import Path

from .catalog_space import (
    BENCHMARK_STREAMK_TILE_SHAPES,
    EXPANDED_GEMM_TILE_SHAPES,
    EXPANDED_STREAMK_TILE_SHAPES,
    EXHAUSTIVE_REGULAR_GEMM_STAGES,
    SEED_KERNELS,
    STREAMK_TILE_SHAPES,
    _get_exhaustive_8x4_tiles,
    apply_epilogue_metadata,
    apply_scheduler_metadata,
    benchmark_gemm_tile_candidates,
    benchmark_streamk_tile_candidates,
    dedupe_kernel_entries,
    exhaustive_regular_gemm_tile_candidates,
    exhaustive_streamk_tile_candidates,
    exhaustive_streamk_tile_stage_candidates,
    ilp_class,
    kernel_catalog_entry,
    normalize_benchmark_streamk_splitk,
    source_template_gemm_tile_candidates,
)
from .schemas import SCHEMA_VERSION, SEARCH_RUNTIME_SCHEMA
from .source_templates import observed_bmg_template_space
from .utils import now_iso, read_json


DEFAULT_KERNEL_CATALOG_PATH = Path(__file__).resolve().parent / "intel_gemm_kernel_catalog_level0.json"
def generated_level0_kernel_catalog():
    catalog = []
    for dtype in sorted(SEED_KERNELS.keys()):
        for seed in SEED_KERNELS.get(dtype, []):
            catalog.append(kernel_catalog_entry(dtype, seed))
    return {
        "schema_version": SCHEMA_VERSION,
        "catalog_version": "level0-seed-catalog",
        "instantiation_levels": {
            "0": "existing validated benchmark-backed kernels",
            "1": "expanded tile and subgroup layouts",
            "2": "full autotuning catalog including copy/epilogue variants",
        },
        "search_runtime_schema": SEARCH_RUNTIME_SCHEMA,
        "kernels": catalog,
    }


def generated_expanded_streamk_kernel_catalog():
    expanded = copy.deepcopy(SEED_KERNELS)
    source_template_space = observed_bmg_template_space()
    expanded_gemm = {
        "bf16": [
            *benchmark_gemm_tile_candidates(
                "BmgGemmBF16BF16FP32",
                "bf16",
                "bf16",
                "f32",
                "f32",
                layout="rcr",
            ),
            *benchmark_gemm_tile_candidates(
                "BmgGemmBF16BF16FP32",
                "bf16",
                "bf16",
                "f32",
                "f32",
                layout="rrr",
            ),
        ],
        "f16": [
            *benchmark_gemm_tile_candidates(
                "BmgGemmFP16FP16FP32",
                "f16",
                "f16",
                "f32",
                "f32",
            ),
            *benchmark_gemm_tile_candidates(
                "BmgGemmF16F16F16",
                "f16",
                "f16",
                "f16",
                "f16",
                dtype_d="f16",
            ),
        ],
        "tf32": benchmark_gemm_tile_candidates(
            "BmgGemmTF32TF32FP32",
            "tf32",
            "tf32",
            "f32",
            "f32",
        ),
    }

    expanded_streamk = {
        "bf16": benchmark_streamk_tile_candidates(
            "BmgGemmBF16BF16FP32",
            "bf16",
            "bf16",
            "f32",
            "f32",
            tile_shapes=_get_exhaustive_8x4_tiles(),
            source="expanded_streamk_catalog",
            instantiation_level=1,
        ),
        "f16": [
            *benchmark_streamk_tile_candidates(
                "BmgGemmF16F16FP32",
                "f16",
                "f16",
                "f32",
                "f32",
                tile_shapes=_get_exhaustive_8x4_tiles(),
                source="expanded_streamk_catalog",
                instantiation_level=1,
            ),
            *benchmark_streamk_tile_candidates(
                "BmgGemmF16F16F16",
                "f16",
                "f16",
                "f16",
                "f16",
                dtype_d="f16",
                tile_shapes=_get_exhaustive_8x4_tiles(),
                source="expanded_streamk_catalog",
                instantiation_level=1,
            ),
        ],
        "tf32": benchmark_streamk_tile_candidates(
            "BmgGemmTF32TF32FP32",
            "tf32",
            "tf32",
            "f32",
            "f32",
            tile_shapes=EXPANDED_STREAMK_TILE_SHAPES,
            source="expanded_streamk_catalog",
            instantiation_level=1,
        ),
    }
    source_template_gemm = {
        "bf16": [
            *source_template_gemm_tile_candidates(
                "BmgGemmBF16BF16FP32",
                "bf16",
                "bf16",
                "f32",
                "f32",
                layout="rcr",
                source_template_space=source_template_space,
            ),
            *source_template_gemm_tile_candidates(
                "BmgGemmBF16BF16FP32",
                "bf16",
                "bf16",
                "f32",
                "f32",
                layout="rrr",
                source_template_space=source_template_space,
            ),
        ],
        "f16": [
            *source_template_gemm_tile_candidates(
                "BmgGemmFP16FP16FP32",
                "f16",
                "f16",
                "f32",
                "f32",
                source_template_space=source_template_space,
            ),
            *source_template_gemm_tile_candidates(
                "BmgGemmF16F16F16",
                "f16",
                "f16",
                "f16",
                "f16",
                dtype_d="f16",
                source_template_space=source_template_space,
            ),
        ],
        "tf32": source_template_gemm_tile_candidates(
            "BmgGemmTF32TF32FP32",
            "tf32",
            "tf32",
            "f32",
            "f32",
            source_template_space=source_template_space,
        ),
    }
    for entries_by_dtype in (expanded_gemm, source_template_gemm, expanded_streamk):
        for dtype, entries in entries_by_dtype.items():
            existing = {entry["kernel_name"] for entry in expanded.get(dtype, [])}
            new_entries = [entry for entry in entries if entry["kernel_name"] not in existing]
            expanded.setdefault(dtype, []).extend(new_entries)

    catalog = []
    for dtype in sorted(expanded.keys()):
        for seed in expanded.get(dtype, []):
            catalog.append(kernel_catalog_entry(dtype, seed))
    return {
        "schema_version": SCHEMA_VERSION,
        "catalog_version": "expanded-bmg-level1",
        "instantiation_levels": {
            "0": "existing validated benchmark-backed kernels",
            "1": "expanded BMG Gemm/StreamK/DataParallel/SplitK tile shapes with fixed 8x4 subgroup layout",
            "2": "reserved for copy atom, epilogue, stage, and subgroup-layout expansion",
        },
        "generator_arch": "bmg",
        "generator_instantiation_level": 1,
        "search_runtime_schema": SEARCH_RUNTIME_SCHEMA,
        "source_template_space": source_template_space,
        "kernels": catalog,
    }


def generated_layered_bmg_kernel_catalog(constraints=None):
    catalog = generated_expanded_streamk_kernel_catalog()
    expanded = {}
    for entry in catalog["kernels"]:
        expanded.setdefault(entry["dtype_family"], []).append(entry)
    exhaustive_regular_gemm = {
        "bf16": exhaustive_regular_gemm_tile_candidates(
            "BmgGemmBF16BF16FP32",
            "bf16",
            "bf16",
            "f32",
            "f32",
            constraints=constraints,
        ),
        "f16": exhaustive_regular_gemm_tile_candidates(
            "BmgGemmFP16FP16FP32",
            "f16",
            "f16",
            "f32",
            "f32",
            constraints=constraints,
        ),
    }
    for dtype, entries in exhaustive_regular_gemm.items():
        existing = {entry["kernel_name"] for entry in expanded.get(dtype, [])}
        expanded.setdefault(dtype, []).extend(
            kernel_catalog_entry(dtype, entry)
            for entry in entries
            if entry["kernel_name"] not in existing
        )

    # RRR layout: re-run exhaustive search with rrr layout for bf16
    exhaustive_regular_rrr = exhaustive_regular_gemm_tile_candidates(
        "BmgGemmBF16BF16FP32", "bf16", "bf16", "f32", "f32",
        layout="rrr", constraints=constraints,
    )
    existing_rrr = {entry["kernel_name"] for entry in expanded.get("bf16", [])}
    expanded.setdefault("bf16", []).extend(
        kernel_catalog_entry("bf16", entry)
        for entry in exhaustive_regular_rrr
        if entry["kernel_name"] not in existing_rrr
    )

    # Phase 2: StreamK/DataParallel/SplitK for all exhaustive tile shapes (K ≥ 32)
    exhaustive_streamk = {
        "bf16": exhaustive_streamk_tile_candidates(
            "BmgGemmBF16BF16FP32",
            "bf16", "bf16", "f32", "f32",
            constraints=constraints,
        ),
        "f16": exhaustive_streamk_tile_candidates(
            "BmgGemmFP16FP16FP32",
            "f16", "f16", "f32", "f32",
            constraints=constraints,
        ),
    }
    sk_added = {"bf16": 0, "f16": 0}
    for dtype, entries in exhaustive_streamk.items():
        existing = {entry["kernel_name"] for entry in expanded.get(dtype, [])}
        for entry in entries:
            if entry["kernel_name"] not in existing:
                expanded.setdefault(dtype, []).append(kernel_catalog_entry(dtype, entry))
                sk_added[dtype] += 1
    
    # RRR layout: generate StreamK/DP/SplitK for bf16 RRR
    exhaustive_streamk_rrr = exhaustive_streamk_tile_candidates(
        "BmgGemmBF16BF16FP32", "bf16", "bf16", "f32", "f32",
        constraints=constraints, layout="rrr",
    )
    existing_rrr_sk = {entry["kernel_name"] for entry in expanded.get("bf16", [])}
    sk_rrr_added = 0
    for entry in exhaustive_streamk_rrr:
        if entry["kernel_name"] not in existing_rrr_sk:
            expanded.setdefault("bf16", []).append(kernel_catalog_entry("bf16", entry))
            sk_rrr_added += 1
    sk_added["bf16"] += sk_rrr_added
    total_sk_added = sum(sk_added.values())
    kernels = []
    for dtype in sorted(expanded.keys()):
        kernels.extend(expanded[dtype])
    catalog["catalog_version"] = "layered-bmg-regular-gemm-exhaustive"
    catalog["instantiation_levels"]["3"] = (
        "regular GEMM legal tile/subgroup/stage enumeration generated from default constraints"
    )
    catalog["instantiation_levels"]["4"] = (
        "StreamK/DataParallel/SplitK for all exhaustive tile shapes with K ≥ 32 and SG 8x4"
    )
    catalog["regular_gemm_exhaustive_space"] = {
        "stages": list(EXHAUSTIVE_REGULAR_GEMM_STAGES),
        "validity_model": "is_valid_xe2_tile_sg plus selected stage values",
        "bf16_kernel_count": len(exhaustive_regular_gemm["bf16"]),
        "f16_kernel_count": len(exhaustive_regular_gemm["f16"]),
    }
    catalog["exhaustive_streamk_space"] = {
        "modes": ["StreamK", "DataParallel", "SplitK"],
        "sg_layout": [8, 4],
        "min_tile_k": 32,
        "bf16_kernel_count": len(exhaustive_streamk["bf16"]),
        "f16_kernel_count": len(exhaustive_streamk["f16"]),
        "new_kernels_added": sk_added,
    }
    catalog["kernels"] = kernels
    return catalog


def generated_layered_bmg_scheduler_expanded_kernel_catalog(constraints=None):
    catalog = generated_layered_bmg_kernel_catalog(constraints)
    expanded = {}
    replaced_bf16_scheduler_entries = 0
    for entry in catalog["kernels"]:
        if (
            entry["dtype_family"] == "bf16"
            and entry.get("runner") == "benchmark"
            and entry.get("streamk_mode")
        ):
            replaced_bf16_scheduler_entries += 1
            continue
        expanded.setdefault(entry["dtype_family"], []).append(entry)

    exhaustive_streamk = exhaustive_streamk_tile_stage_candidates(
        "BmgGemmBF16BF16FP32",
        "bf16",
        "bf16",
        "f32",
        "f32",
        constraints=constraints,
        layout="rcr",
        source="exhaustive_streamk_catalog",
        instantiation_level=5,
    )
    exhaustive_streamk_rrr = exhaustive_streamk_tile_stage_candidates(
        "BmgGemmBF16BF16FP32",
        "bf16",
        "bf16",
        "f32",
        "f32",
        constraints=constraints,
        layout="rrr",
        source="exhaustive_streamk_catalog",
        instantiation_level=5,
    )
    for dtype, entries in {"bf16": [*exhaustive_streamk, *exhaustive_streamk_rrr]}.items():
        existing = {entry["kernel_name"] for entry in expanded.get(dtype, [])}
        expanded.setdefault(dtype, []).extend(
            kernel_catalog_entry(dtype, entry)
            for entry in entries
            if entry["kernel_name"] not in existing
        )

    kernels = []
    for dtype in sorted(expanded.keys()):
        kernels.extend(expanded[dtype])
    catalog["catalog_version"] = "layered-bmg-scheduler-expanded"
    catalog["instantiation_levels"]["5"] = (
        "scheduler exhaustive tile/subgroup/stage enumeration generated from default constraints"
    )
    catalog["exhaustive_streamk_space"] = {
        "modes": ["StreamK", "DataParallel", "SplitK"],
        "sg_layouts": [
            [sg_m, sg_n]
            for sg_m, sg_n in sorted({(entry["sg_m"], entry["sg_n"]) for entry in exhaustive_streamk}, key=lambda item: (item[0], item[1]))
        ],
        "stages": list(EXHAUSTIVE_REGULAR_GEMM_STAGES),
        "min_tile_k": 32,
        "bf16_kernel_count": len(exhaustive_streamk),
        "bf16_rrr_kernel_count": len(exhaustive_streamk_rrr),
        "replaced_fixed_bf16_scheduler_entries": replaced_bf16_scheduler_entries,
    }
    catalog["kernels"] = kernels
    return catalog


def load_persisted_kernel_catalog(path=DEFAULT_KERNEL_CATALOG_PATH):
    path = path or DEFAULT_KERNEL_CATALOG_PATH
    if path.exists():
        catalog = read_json(path)
        catalog["search_runtime_schema"] = SEARCH_RUNTIME_SCHEMA
        for entry in catalog.get("kernels", []):
            entry.setdefault("dtype_d", entry["dtype_c"])
            entry.setdefault("batch_count", 1)
            entry.setdefault("allowed_runtime_sweeps", ["shape_id", "m", "n", "k", "batch_count"])
            if "batch_count" not in entry["allowed_runtime_sweeps"]:
                entry["allowed_runtime_sweeps"].append("batch_count")
            entry.setdefault("mma_atom", "XE_DPAS_TT")
            entry.setdefault("gmem_copy_atom_a", "auto")
            entry.setdefault("gmem_copy_atom_b", "auto")
            entry.setdefault("epilogue_op", "LinearCombination")
            entry.setdefault("epilogue_tile", "auto")
            entry.setdefault("epilogue_copy_atom_c", "auto")
            entry.setdefault("epilogue_copy_atom_d", "auto")
            entry.setdefault("streamk_dtype_preset", entry["dtype_a"] if entry.get("runner") == "streamk_example" else "")
            entry.setdefault("support_status", "supported")
            entry.setdefault("support_reason", "")
            normalize_benchmark_streamk_splitk(entry)
            apply_epilogue_metadata(entry)
            apply_scheduler_metadata(entry)
        catalog["kernels"] = dedupe_kernel_entries(catalog.get("kernels", []))
        return catalog
    return generated_level0_kernel_catalog()


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


def build_kernel_catalog(
    dtypes=None,
    allowed_runners=("benchmark",),
    catalog_path=DEFAULT_KERNEL_CATALOG_PATH,
    catalog_source="persisted",
    generator_arch="bmg",
    generator_instantiation_level=0,
):
    if catalog_source == "persisted":
        source_catalog = load_persisted_kernel_catalog(catalog_path)
    elif catalog_source == "generator":
        source_catalog = generated_generator_kernel_catalog(
            generator_arch=generator_arch,
            generator_instantiation_level=generator_instantiation_level,
        )
    elif catalog_source in {"expanded_streamk", "expanded_bmg"}:
        source_catalog = generated_expanded_streamk_kernel_catalog()
    elif catalog_source == "layered_bmg":
        from .constraints import default_constraints

        source_catalog = generated_layered_bmg_kernel_catalog(default_constraints())
    elif catalog_source == "layered_bmg_scheduler_expanded":
        from .constraints import default_constraints

        source_catalog = generated_layered_bmg_scheduler_expanded_kernel_catalog(default_constraints())
    else:
        raise ValueError(f"Unsupported kernel catalog source: {catalog_source}")
    selected_dtypes = set(dtypes) if dtypes is not None else None
    catalog = []
    for entry in source_catalog["kernels"]:
        if selected_dtypes is not None and entry["dtype_a"] not in selected_dtypes:
            continue
        if entry["runner"] not in allowed_runners:
            continue
        catalog.append(copy.deepcopy(entry))
    catalog = dedupe_kernel_entries(catalog)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "catalog_version": source_catalog["catalog_version"],
        "instantiation_levels": source_catalog["instantiation_levels"],
        "catalog_source": catalog_source,
        "generator_arch": source_catalog.get("generator_arch", ""),
        "generator_instantiation_level": source_catalog.get("generator_instantiation_level", 0),
        "source_template_space": source_catalog.get("source_template_space", {}),
        "regular_gemm_exhaustive_space": source_catalog.get("regular_gemm_exhaustive_space", {}),
        "search_runtime_schema": source_catalog.get("search_runtime_schema", SEARCH_RUNTIME_SCHEMA),
        "kernels": catalog,
    }
