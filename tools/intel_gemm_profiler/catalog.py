#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import copy
from pathlib import Path

from .catalog_generator import generated_generator_kernel_catalog
from .catalog_layered import (
    generated_expanded_streamk_kernel_catalog,
    generated_layered_bmg_kernel_catalog,
    generated_layered_bmg_scheduler_expanded_kernel_catalog,
)
from .catalog_space import (
    BENCHMARK_STREAMK_TILE_SHAPES,
    EXPANDED_GEMM_TILE_SHAPES,
    EXPANDED_STREAMK_TILE_SHAPES,
    SEED_KERNELS,
    STREAMK_TILE_SHAPES,
    apply_epilogue_metadata,
    apply_scheduler_metadata,
    dedupe_kernel_entries,
    ilp_class,
    kernel_catalog_entry,
    normalize_benchmark_streamk_splitk,
)
from .schemas import SCHEMA_VERSION, SEARCH_RUNTIME_SCHEMA
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
