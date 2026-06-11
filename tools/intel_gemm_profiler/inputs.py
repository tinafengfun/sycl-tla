#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import copy
from pathlib import Path

from .ali_dataset import build_ali_gemm_docs
from .candidates import default_shapes, dry_run_shapes
from .utils import read_json


SEARCH_STRATEGY_PRESETS = {
    "manual": {},
    "baseline": {
        "kernel_catalog_source": "persisted",
        "prefilter": "none",
        "run_candidate_build_preflight": False,
        "use_candidate_build_preflight_benchmarks": False,
    },
    "expanded_bmg": {
        "kernel_catalog_source": "expanded_bmg",
        "prefilter": "none",
        "run_candidate_build_preflight": False,
        "use_candidate_build_preflight_benchmarks": False,
    },
    "layered_exhaustive": {
        "kernel_catalog_source": "layered_bmg",
        "prefilter": "none",
        "run_candidate_build_preflight": False,
        "use_candidate_build_preflight_benchmarks": False,
    },
    "bruteforce_scheduler": {
        "kernel_catalog_source": "layered_bmg_scheduler_expanded",
        "prefilter": "none",
        "run_candidate_build_preflight": True,
        "use_candidate_build_preflight_benchmarks": True,
    },
}


def load_compiled_kernel_list(path):
    if not path:
        return None
    kernels = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        if item.startswith("^") and item.endswith("$"):
            item = item[1:-1]
        kernels.append(item)
    return kernels


def filter_candidate_space_by_compiled_kernels(candidate_space, compiled_kernels):
    if compiled_kernels is None:
        return candidate_space
    compiled = set(compiled_kernels)
    filtered = copy.deepcopy(candidate_space)
    filtered["candidates"] = [
        candidate
        for candidate in candidate_space["candidates"]
        if candidate.get("runner", "benchmark") != "benchmark" or candidate["kernel_id"] in compiled
    ]
    filtered["compiled_kernel_filter"] = {
        "source": "compiled_kernel_list",
        "kernel_count": len(compiled),
        "matched_candidate_count": len(filtered["candidates"]),
    }
    if candidate_space["candidates"] and not filtered["candidates"]:
        raise ValueError("Compiled kernel list does not match any generated benchmark candidates.")
    return filtered


def apply_search_strategy_defaults(args):
    strategy = getattr(args, "search_strategy", "manual") or "manual"
    if getattr(args, "bruteforce_scheduler_search", False) and strategy == "manual":
        strategy = "bruteforce_scheduler"
    preset = SEARCH_STRATEGY_PRESETS.get(strategy, {})
    if preset:
        args.kernel_catalog_source = preset["kernel_catalog_source"]
        args.prefilter = preset["prefilter"]
        args.run_candidate_build_preflight = preset["run_candidate_build_preflight"]
        args.use_candidate_build_preflight_benchmarks = preset["use_candidate_build_preflight_benchmarks"]
    if strategy == "bruteforce_scheduler" and getattr(args, "candidate_build_batch_size", 0) <= 0:
        args.candidate_build_batch_size = 1
    if strategy == "bruteforce_scheduler" and (getattr(args, "skip_run", False) or getattr(args, "dry_run", False)):
        args.run_candidate_build_preflight = False
        args.use_candidate_build_preflight_benchmarks = False
    args.search_strategy = strategy
    return args


def apply_bruteforce_scheduler_search_defaults(args):
    args.bruteforce_scheduler_search = True
    return apply_search_strategy_defaults(args)


def load_target_shapes_and_reference(args, dry_run_mode):
    if args.ali_workbook:
        if args.shapes_json:
            raise ValueError("--ali-workbook and --shapes-json are mutually exclusive.")
        if args.reference_json:
            raise ValueError("--ali-workbook and --reference-json are mutually exclusive.")
        shapes_doc, reference_doc = build_ali_gemm_docs(args.ali_workbook)
        return limit_shapes_and_reference(shapes_doc, reference_doc, args.max_shapes)
    shapes_doc = read_json(args.shapes_json) if args.shapes_json else (dry_run_shapes(args.dtype) if dry_run_mode else default_shapes(args.dtype))
    reference_doc = read_json(args.reference_json) if args.reference_json else None
    return limit_shapes_and_reference(shapes_doc, reference_doc, args.max_shapes)


def limit_shapes_and_reference(shapes_doc, reference_doc=None, max_shapes=0):
    if max_shapes is None or max_shapes == 0:
        return shapes_doc, reference_doc
    if max_shapes < 0:
        raise ValueError("--max-shapes must be non-negative.")
    limited_shapes_doc = copy.deepcopy(shapes_doc)
    selected_shapes = limited_shapes_doc.get("shapes", [])[:max_shapes]
    limited_shapes_doc["shapes"] = selected_shapes
    limited_shapes_doc["shape_limit"] = max_shapes
    limited_shapes_doc["unlimited_shape_count"] = len(shapes_doc.get("shapes", []))
    if reference_doc is None:
        return limited_shapes_doc, None
    selected_shape_ids = {shape["shape_id"] for shape in selected_shapes}
    selected_shape_keys = {
        (shape.get("dtype_a"), shape.get("m"), shape.get("n"), shape.get("k"))
        for shape in selected_shapes
    }
    limited_reference_doc = copy.deepcopy(reference_doc)
    limited_reference_doc["entries"] = [
        entry for entry in limited_reference_doc.get("entries", [])
        if entry.get("shape_id") in selected_shape_ids
    ]
    limited_reference_doc["skipped_entries"] = [
        entry
        for entry in limited_reference_doc.get("skipped_entries", [])
        if (entry.get("dtype"), entry.get("m"), entry.get("n"), entry.get("k")) in selected_shape_keys
    ]
    limited_reference_doc["shape_limit"] = max_shapes
    limited_reference_doc["unlimited_reference_entries"] = len(reference_doc.get("entries", []))
    return limited_shapes_doc, limited_reference_doc
