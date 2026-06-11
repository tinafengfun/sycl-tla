#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import copy

from .candidate_manifest import build_candidate_build_manifest, build_selected_kernel_batches
from .catalog import SEED_KERNELS, build_kernel_catalog
from .candidate_entries import (
    build_compiler_profile_probe_entries,
    build_dpas_probe_entry,
    build_phase_a_probe_entries,
    build_screening_entries,
    choose_candidates_for_shape,
    generate_confirmation_entries,
    select_probe_shape,
    write_config,
)
from .constraints import blocked
from .schemas import RESULT_METADATA_FIELDS, SCHEMA_VERSION, SEARCH_RUNTIME_SCHEMA
from .utils import now_iso


def default_shapes(dtype):
    base = [
        {"m": 1, "n": 4096, "k": 14336, "tags": ["decode"]},
        {"m": 8, "n": 4096, "k": 4096, "tags": ["decode"]},
        {"m": 64, "n": 4096, "k": 4096, "tags": ["prefill"]},
        {"m": 256, "n": 4096, "k": 8192, "tags": ["prefill"]},
    ]
    shapes = []
    for item in base:
        shapes.append(
            {
                "shape_id": f"rcr_{dtype}_{item['m']}_{item['n']}_{item['k']}",
                "layout": "rcr",
                "dtype_a": dtype,
                    "dtype_b": dtype,
                    "dtype_c": "f32",
                    "dtype_d": "f32",
                    "dtype_acc": "f32",
                    "m": item["m"],
                    "n": item["n"],
                    "k": item["k"],
                    "batch_count": 1,
                    "runtime_defaults": {},
                "tags": item["tags"],
            }
        )
    return {"schema_version": SCHEMA_VERSION, "generated_at": now_iso(), "shape_set_id": f"default_{dtype}_decode_prefill", "source": "predefined", "shapes": shapes}


def dry_run_shapes(dtype):
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "shape_set_id": f"dry_run_{dtype}",
        "source": "dry_run",
        "shapes": [
            {
                "shape_id": f"dry_run_rcr_{dtype}_1_64_32",
                "layout": "rcr",
                "dtype_a": dtype,
                    "dtype_b": dtype,
                    "dtype_c": "f32",
                    "dtype_d": "f32",
                    "dtype_acc": "f32",
                    "m": 1,
                    "n": 64,
                    "k": 32,
                    "batch_count": 1,
                    "runtime_defaults": {},
                "tags": ["dry_run"],
            }
        ],
    }


def candidate_class(tile_m, sg_count):
    if tile_m <= 16 and sg_count <= 8:
        return "small_tile"
    if tile_m >= 128 or sg_count >= 16:
        return "large_tile"
    return "medium_tile"


def select_compiler_profile_id(profiles, tile_m, sg_count):
    chosen = None
    available_profiles = [profile for profile in profiles["profiles"] if profile.get("probe_status") not in {"fail", "timeout"}]
    for profile in profiles["profiles"]:
        if profile.get("probe_status") in {"fail", "timeout"}:
            continue
        selector = profile.get("selector", {})
        if "tile_m_min" in selector and tile_m < selector["tile_m_min"]:
            continue
        if "tile_m_max" in selector and tile_m > selector["tile_m_max"]:
            continue
        if "sg_count_min" in selector and sg_count < selector["sg_count_min"]:
            continue
        if "sg_count_max" in selector and sg_count > selector["sg_count_max"]:
            continue
        chosen = profile["compiler_profile_id"]
        break
    if chosen:
        return chosen
    fallback_profiles = available_profiles or profiles["profiles"]
    return fallback_profiles[0]["compiler_profile_id"]


def candidate_id_for(seed):
    candidate_id = f"{seed['layout']}_{seed['dtype_a']}{seed['dtype_b']}{seed['dtype_c']}_tm{seed['tile_m']}_tn{seed['tile_n']}_tk{seed['tile_k']}_sg{seed['sg_m']}x{seed['sg_n']}_st{seed['stages']}_sk{seed['split_k']}"
    if seed.get("dtype_d", seed["dtype_c"]) != seed["dtype_c"]:
        candidate_id = f"{candidate_id}_d{seed['dtype_d']}"
    streamk_mode = seed.get("streamk_mode", "")
    if streamk_mode:
        candidate_id = f"{candidate_id}_{streamk_mode}"
    return candidate_id


def copy_result_metadata(source):
    return {field: source.get(field, "") for field in RESULT_METADATA_FIELDS}


def generate_candidate_space(
    shapes_doc,
    constraints,
    profiles,
    allowed_runners=("benchmark",),
    catalog_source="persisted",
    catalog_path=None,
    generator_arch="bmg",
    generator_instantiation_level=0,
    prefilter_strategy="none",
):
    seen = set()
    candidates = []
    matched_signature_count = 0
    blocked_candidate_count = 0
    dtypes = sorted({shape["dtype_a"] for shape in shapes_doc["shapes"]})
    requested_layouts = {shape["layout"] for shape in shapes_doc["shapes"]}
    requested_signatures = {
        (shape["layout"], shape["dtype_a"], shape["dtype_b"], shape["dtype_c"], shape.get("dtype_d", shape["dtype_c"]), shape["dtype_acc"])
        for shape in shapes_doc["shapes"]
    }
    requested_input_signatures = {
        (shape["layout"], shape["dtype_a"], shape["dtype_b"])
        for shape in shapes_doc["shapes"]
    }
    kernel_catalog = build_kernel_catalog(
        dtypes=dtypes,
        allowed_runners=allowed_runners,
        catalog_path=catalog_path,
        catalog_source=catalog_source,
        generator_arch=generator_arch,
        generator_instantiation_level=generator_instantiation_level,
    )
    available_streamk_signatures = {
        (entry["layout"], entry["dtype_a"], entry["dtype_b"], entry["dtype_c"], entry.get("dtype_d", entry["dtype_c"]), entry["dtype_acc"])
        for entry in kernel_catalog["kernels"]
        if entry.get("runner") == "streamk_example"
    }
    available_layouts = {entry["layout"] for entry in kernel_catalog["kernels"]}
    unsupported_layouts = sorted({shape["layout"] for shape in shapes_doc["shapes"] if shape["layout"] not in available_layouts})
    candidate_exceptions = []
    if unsupported_layouts:
        raise ValueError(
            f"Unsupported layouts in shapes: {', '.join(unsupported_layouts)}. "
            f"Available layouts: {', '.join(sorted(available_layouts))}."
        )
    for seed in kernel_catalog["kernels"]:
        if seed["layout"] not in requested_layouts:
            continue
        if seed.get("support_status") == "unsupported":
            signature = (seed["layout"], seed["dtype_a"], seed["dtype_b"], seed["dtype_c"], seed.get("dtype_d", seed["dtype_c"]), seed["dtype_acc"])
            if signature in requested_signatures:
                candidate_exceptions.append(
                    {
                        "kernel_name": seed["kernel_name"],
                        "reason": seed.get("support_reason", "unsupported_candidate"),
                        "detail": seed.get("support_detail", "Candidate is cataloged but not buildable in the benchmark path."),
                        "future_enable_condition": seed.get("support_future_enable_condition", ""),
                        "layout": seed["layout"],
                        "dtype_a": seed["dtype_a"],
                        "dtype_b": seed["dtype_b"],
                        "dtype_c": seed["dtype_c"],
                        "dtype_d": seed.get("dtype_d", seed["dtype_c"]),
                        "dtype_acc": seed["dtype_acc"],
                        "tile_m": seed["tile_m"],
                        "tile_n": seed["tile_n"],
                        "tile_k": seed["tile_k"],
                        "stages": seed["stages"],
                        "split_k": seed.get("split_k", 1),
                        "streamk_mode": seed.get("streamk_mode", ""),
                        "batch_count": 1,
                        **copy_result_metadata(seed),
                        "mma_atom": seed.get("mma_atom", "XE_DPAS_TT"),
                        "gmem_copy_atom_a": seed.get("gmem_copy_atom_a", "auto"),
                        "gmem_copy_atom_b": seed.get("gmem_copy_atom_b", "auto"),
                        "epilogue_op": seed.get("epilogue_op", "LinearCombination"),
                        "epilogue_tile": seed.get("epilogue_tile", "auto"),
                        "epilogue_copy_atom_c": seed.get("epilogue_copy_atom_c", "auto"),
                        "epilogue_copy_atom_d": seed.get("epilogue_copy_atom_d", "auto"),
                        "mainloop_dispatch_policy": seed.get("mainloop_dispatch_policy", ""),
                        "kernel_schedule": seed.get("kernel_schedule", ""),
                        "tile_scheduler": seed.get("tile_scheduler", ""),
                        "epilogue_dispatch_policy": seed.get("epilogue_dispatch_policy", ""),
                        "example_family": seed.get("example_family", ""),
                        "padding_mode": seed.get("padding_mode", ""),
                        "activation": seed.get("activation", ""),
                        "bias_mode": seed.get("bias_mode", ""),
                        "quant_mode": seed.get("quant_mode", ""),
                        "scale_mode": seed.get("scale_mode", ""),
                    }
                )
            continue
        signature = (seed["layout"], seed["dtype_a"], seed["dtype_b"], seed["dtype_c"], seed.get("dtype_d", seed["dtype_c"]), seed["dtype_acc"])
        if signature not in requested_signatures:
            requested_streamk_signature_available = any(
                requested_signature in available_streamk_signatures
                for requested_signature in requested_signatures
            )
            if (
                seed.get("runner") == "streamk_example"
                and (seed["layout"], seed["dtype_a"], seed["dtype_b"]) in requested_input_signatures
                and not requested_streamk_signature_available
            ):
                candidate_exceptions.append(
                    {
                        "kernel_name": seed["kernel_name"],
                        "reason": "streamk_example_dtype_signature_mismatch",
                        "detail": "StreamK example runner matches layout/input dtype but not requested C/D/accumulator dtype semantics.",
                        "layout": seed["layout"],
                        "dtype_a": seed["dtype_a"],
                        "dtype_b": seed["dtype_b"],
                        "dtype_c": seed["dtype_c"],
                        "dtype_d": seed.get("dtype_d", seed["dtype_c"]),
                        "dtype_acc": seed["dtype_acc"],
                        "requested_signatures": [
                            {
                                "dtype_c": dtype_c,
                                "dtype_d": dtype_d,
                                "dtype_acc": dtype_acc,
                            }
                            for layout, dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc in sorted(requested_signatures)
                            if layout == seed["layout"] and dtype_a == seed["dtype_a"] and dtype_b == seed["dtype_b"]
                        ],
                        "tile_m": seed["tile_m"],
                        "tile_n": seed["tile_n"],
                        "tile_k": seed["tile_k"],
                        "stages": seed["stages"],
                        "batch_count": 1,
                        **copy_result_metadata(seed),
                        "mma_atom": seed.get("mma_atom", "XE_DPAS_TT"),
                        "gmem_copy_atom_a": seed.get("gmem_copy_atom_a", "auto"),
                        "gmem_copy_atom_b": seed.get("gmem_copy_atom_b", "auto"),
                        "epilogue_op": seed.get("epilogue_op", "LinearCombination"),
                        "epilogue_tile": seed.get("epilogue_tile", "auto"),
                        "epilogue_copy_atom_c": seed.get("epilogue_copy_atom_c", "auto"),
                        "epilogue_copy_atom_d": seed.get("epilogue_copy_atom_d", "auto"),
                        "mainloop_dispatch_policy": seed.get("mainloop_dispatch_policy", ""),
                        "kernel_schedule": seed.get("kernel_schedule", ""),
                        "tile_scheduler": seed.get("tile_scheduler", ""),
                        "epilogue_dispatch_policy": seed.get("epilogue_dispatch_policy", ""),
                        "example_family": seed.get("example_family", ""),
                        "padding_mode": seed.get("padding_mode", ""),
                        "activation": seed.get("activation", ""),
                        "bias_mode": seed.get("bias_mode", ""),
                        "quant_mode": seed.get("quant_mode", ""),
                        "scale_mode": seed.get("scale_mode", ""),
                    }
                )
            continue
        matched_signature_count += 1
        if seed.get("source") == "generator_manifest" and seed.get("streamk_mode") == "streamk":
            candidate_exceptions.append(
                {
                    "kernel_name": seed["kernel_name"],
                    "reason": "intel_xe_generated_streamk_tile_scheduler_unsupported",
                    "detail": "Intel Xe generated library kernels currently reject StreamK tile scheduler specialization.",
                    "layout": seed["layout"],
                    "dtype_a": seed["dtype_a"],
                    "dtype_b": seed["dtype_b"],
                    "dtype_c": seed["dtype_c"],
                    "dtype_d": seed.get("dtype_d", seed["dtype_c"]),
                    "dtype_acc": seed["dtype_acc"],
                    "tile_m": seed["tile_m"],
                    "tile_n": seed["tile_n"],
                    "tile_k": seed["tile_k"],
                    "stages": seed["stages"],
                    "batch_count": 1,
                    **copy_result_metadata(seed),
                    "mma_atom": seed.get("mma_atom", "XE_DPAS_TT"),
                    "gmem_copy_atom_a": seed.get("gmem_copy_atom_a", "auto"),
                    "gmem_copy_atom_b": seed.get("gmem_copy_atom_b", "auto"),
                    "epilogue_op": seed.get("epilogue_op", "LinearCombination"),
                    "epilogue_tile": seed.get("epilogue_tile", "auto"),
                    "epilogue_copy_atom_c": seed.get("epilogue_copy_atom_c", "auto"),
                    "epilogue_copy_atom_d": seed.get("epilogue_copy_atom_d", "auto"),
                    "mainloop_dispatch_policy": seed.get("mainloop_dispatch_policy", ""),
                    "kernel_schedule": seed.get("kernel_schedule", ""),
                    "tile_scheduler": seed.get("tile_scheduler", ""),
                    "epilogue_dispatch_policy": seed.get("epilogue_dispatch_policy", ""),
                    "example_family": seed.get("example_family", ""),
                    "padding_mode": seed.get("padding_mode", ""),
                    "activation": seed.get("activation", ""),
                    "bias_mode": seed.get("bias_mode", ""),
                    "quant_mode": seed.get("quant_mode", ""),
                    "scale_mode": seed.get("scale_mode", ""),
                }
            )
            continue
        if blocked(seed, constraints):
            blocked_candidate_count += 1
            continue
        ident = candidate_id_for(seed)
        if ident in seen:
            continue
        seen.add(ident)
        sg_count = seed["sg_m"] * seed["sg_n"]
        candidates.append(
            {
                "candidate_id": ident,
                "kernel_name": seed["kernel_name"],
                "kernel_id": seed["kernel_id"],
                "layout": seed["layout"],
                "dtype_a": seed["dtype_a"],
                "dtype_b": seed["dtype_b"],
                "dtype_c": seed["dtype_c"],
                "dtype_d": seed.get("dtype_d", seed["dtype_c"]),
                "dtype_acc": seed["dtype_acc"],
                "tile_m": seed["tile_m"],
                "tile_n": seed["tile_n"],
                "tile_k": seed["tile_k"],
                "sg_m": seed["sg_m"],
                "sg_n": seed["sg_n"],
                "stages": seed["stages"],
                "split_k": seed["split_k"],
                "streamk_dtype_preset": seed.get("streamk_dtype_preset", ""),
                "batch_count": 1,
                "streamk_mode": seed.get("streamk_mode", ""),
                "runner": seed.get("runner", "benchmark"),
                "benchmark_target": seed["benchmark_target"],
                **copy_result_metadata(seed),
                "grf_mode": seed["grf_mode"],
                "ilp_class": seed["ilp_class"],
                "instantiation_level": seed["instantiation_level"],
                "mma_atom": seed.get("mma_atom", "XE_DPAS_TT"),
                "gmem_copy_atom_a": seed.get("gmem_copy_atom_a", "auto"),
                "gmem_copy_atom_b": seed.get("gmem_copy_atom_b", "auto"),
                "epilogue_op": seed.get("epilogue_op", "LinearCombination"),
                "epilogue_tile": seed.get("epilogue_tile", "auto"),
                "epilogue_copy_atom_c": seed.get("epilogue_copy_atom_c", "auto"),
                "epilogue_copy_atom_d": seed.get("epilogue_copy_atom_d", "auto"),
                "mainloop_dispatch_policy": seed.get("mainloop_dispatch_policy", ""),
                "kernel_schedule": seed.get("kernel_schedule", ""),
                "tile_scheduler": seed.get("tile_scheduler", ""),
                "epilogue_dispatch_policy": seed.get("epilogue_dispatch_policy", ""),
                "example_family": seed.get("example_family", ""),
                "padding_mode": seed.get("padding_mode", ""),
                "activation": seed.get("activation", ""),
                "bias_mode": seed.get("bias_mode", ""),
                "quant_mode": seed.get("quant_mode", ""),
                "scale_mode": seed.get("scale_mode", ""),
                "runtime_defaults": seed["runtime_defaults"],
                "allowed_runtime_sweeps": seed["allowed_runtime_sweeps"],
                "source": seed.get("source", ""),
                "candidate_class": candidate_class(seed["tile_m"], sg_count),
                "compiler_profile_id": select_compiler_profile_id(profiles, seed["tile_m"], sg_count),
                "filters_applied": ["kernel_catalog", constraints["constraint_source"]],
            }
        )
    exception_summary = {}
    for exception in candidate_exceptions:
        reason = exception["reason"]
        summary = exception_summary.setdefault(reason, {"reason": reason, "count": 0, "sample_kernel_names": []})
        summary["count"] += 1
        if len(summary["sample_kernel_names"]) < 5:
            summary["sample_kernel_names"].append(exception["kernel_name"])
    if prefilter_strategy != "none" and prefilter_strategy in {"light", "medium", "aggressive"}:
        from .prefilter import prefilter_candidates as _prefilter
        pre_count = len(candidates)
        candidates = _prefilter(candidates, shapes_doc.get("shapes", []), strategy=prefilter_strategy)
        post_count = len(candidates)
        if post_count < pre_count:
            from .utils import now_iso as _now
            candidate_exceptions.insert(
                0,
                {
                    "kernel_name": f"prefilter_{prefilter_strategy}",
                    "reason": f"prefilter_{prefilter_strategy}",
                    "detail": f"Prefilter ({prefilter_strategy}) removed {pre_count - post_count} of {pre_count} candidates.",
                    "layout": "",
                    "dtype_a": "",
                    "dtype_b": "",
                    "dtype_c": "",
                    "dtype_acc": "",
                    "tile_m": 0,
                    "tile_n": 0,
                    "tile_k": 0,
                    "stages": 0,
                    "batch_count": 0,
                },
            )

    return {
    "schema_version": SCHEMA_VERSION,
    "generated_at": now_iso(),
    "device_arch": constraints["device_arch"],
    "constraint_source": constraints["constraint_source"],
    "search_runtime_schema": SEARCH_RUNTIME_SCHEMA,
    "kernel_catalog": {
        "catalog_version": kernel_catalog["catalog_version"],
        "catalog_source": kernel_catalog.get("catalog_source", "persisted"),
        "generator_arch": kernel_catalog.get("generator_arch", ""),
        "generator_instantiation_level": kernel_catalog.get("generator_instantiation_level", 0),
        "source_template_space": kernel_catalog.get("source_template_space", {}),
        "regular_gemm_exhaustive_space": kernel_catalog.get("regular_gemm_exhaustive_space", {}),
        "kernel_count": len(kernel_catalog["kernels"]),
    },
    "candidate_coverage": {
        "requested_layouts": sorted(requested_layouts),
        "requested_signatures": [
            {
                "layout": layout,
                "dtype_a": dtype_a,
                "dtype_b": dtype_b,
                "dtype_c": dtype_c,
                "dtype_d": dtype_d,
                "dtype_acc": dtype_acc,
            }
            for layout, dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc in sorted(requested_signatures)
        ],
        "catalog_kernel_count": len(kernel_catalog["kernels"]),
        "matched_signature_kernel_count": matched_signature_count,
        "accepted_candidate_count": len(candidates),
        "blocked_candidate_count": blocked_candidate_count,
        "exception_count": len(candidate_exceptions),
    },
    "candidate_exception_summary": sorted(exception_summary.values(), key=lambda item: item["reason"]),
    "candidate_exceptions": candidate_exceptions,
    "candidates": candidates,
    }
