#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import copy


def _copy_result_metadata(source):
    return {
        key: source.get(key)
        for key in (
            "kernel_name",
            "source",
            "layout",
            "dtype_a",
            "dtype_b",
            "dtype_c",
            "dtype_d",
            "dtype_acc",
            "tile_m",
            "tile_n",
            "tile_k",
            "sg_m",
            "sg_n",
            "stages",
            "split_k",
            "streamk_mode",
            "decomposition_mode",
            "reduction_mode",
            "runner",
            "compiler_profile_id",
            "candidate_class",
        )
    }


def choose_candidates_for_shape(shape, candidates):
    matched = []
    for candidate in candidates:
        if candidate["layout"] != shape["layout"]:
            continue
        if candidate["dtype_a"] != shape["dtype_a"] or candidate["dtype_b"] != shape["dtype_b"]:
            continue
        if candidate["dtype_c"] != shape["dtype_c"] or candidate["dtype_acc"] != shape["dtype_acc"]:
            continue
        if candidate.get("dtype_d", candidate["dtype_c"]) != shape.get("dtype_d", shape["dtype_c"]):
            continue
        matched.append(candidate)
    return matched or [
        candidate
        for candidate in candidates
        if candidate["layout"] == shape["layout"]
        and candidate["dtype_a"] == shape["dtype_a"]
        and candidate.get("dtype_d", candidate["dtype_c"]) == shape.get("dtype_d", shape["dtype_c"])
    ]


def select_probe_shape(shapes_doc, dtype, layout, target_m, target_n, target_k, predicate=None):
    pool = [shape for shape in shapes_doc["shapes"] if shape["dtype_a"] == dtype and shape["layout"] == layout]
    if predicate:
        filtered = [shape for shape in pool if predicate(shape)]
        if filtered:
            pool = filtered
    if not pool:
        return None
    return min(
        pool,
        key=lambda shape: (
            abs(shape["m"] - target_m),
            abs(shape["n"] - target_n),
            abs(shape["k"] - target_k),
            shape["m"],
            shape["n"],
            shape["k"],
        ),
    )


def build_phase_a_probe_entries(shapes_doc, candidate_space):
    candidates = candidate_space["candidates"]
    non_splitk = [candidate for candidate in candidates if candidate["split_k"] == 1]
    splitk = [candidate for candidate in candidates if candidate["split_k"] > 1]
    selected = []
    if non_splitk:
        small_candidate = min(non_splitk, key=lambda item: (item["tile_m"], item["sg_m"] * item["sg_n"]))
        selected.append(
            (
                "small",
                small_candidate,
                select_probe_shape(
                    shapes_doc,
                    small_candidate["dtype_a"],
                    small_candidate["layout"],
                    8,
                    4096,
                    4096,
                    predicate=lambda shape: shape["m"] <= 8,
                ),
            )
        )
        medium = [candidate for candidate in non_splitk if 16 <= candidate["tile_m"] <= 64]
        if medium:
            medium_candidate = min(medium, key=lambda item: (item["tile_m"], item["sg_m"] * item["sg_n"]))
            selected.append(
                (
                    "medium",
                    medium_candidate,
                    select_probe_shape(
                        shapes_doc,
                        medium_candidate["dtype_a"],
                        medium_candidate["layout"],
                        64,
                        4096,
                        4096,
                        predicate=lambda shape: 8 < shape["m"] < 128,
                    ),
                )
            )
        large_candidate = max(non_splitk, key=lambda item: (item["tile_m"], item["tile_n"]))
        selected.append(
            (
                "large",
                large_candidate,
                select_probe_shape(
                    shapes_doc,
                    large_candidate["dtype_a"],
                    large_candidate["layout"],
                    256,
                    4096,
                    8192,
                    predicate=lambda shape: shape["m"] >= 128,
                ),
            )
        )
    if splitk:
        splitk_candidate = splitk[0]
        selected.append(
            (
                "splitk",
                splitk_candidate,
                select_probe_shape(
                    shapes_doc,
                    splitk_candidate["dtype_a"],
                    splitk_candidate["layout"],
                    1,
                    4096,
                    14336,
                    predicate=lambda shape: shape["n"] >= 16384 or shape["k"] >= 8192,
                ),
            )
        )
    entries = []
    seen = set()
    for probe_class, candidate, shape in selected:
        if shape is None:
            continue
        key = (candidate["candidate_id"], shape["shape_id"])
        if key in seen:
            continue
        seen.add(key)
        entries.append(
            {
                "bm_name": f"{candidate['candidate_id']}__{shape['shape_id']}__probe__0",
                "stage": "probe",
                "attempt_index": 0,
                "probe_class": probe_class,
                "shape": shape,
                "candidate": candidate,
            }
        )
    return entries


def build_dpas_probe_entry(shapes_doc, candidate_space):
    benchmark_candidates = [
        candidate
        for candidate in candidate_space["candidates"]
        if candidate.get("runner", "benchmark") == "benchmark" and candidate["split_k"] == 1
    ]
    if not benchmark_candidates:
        return None
    baseline_candidate = min(
        benchmark_candidates,
        key=lambda item: (item["tile_m"], item["sg_m"] * item["sg_n"], item["tile_n"], item["tile_k"]),
    )
    dtype_shapes = [
        shape
        for shape in shapes_doc["shapes"]
        if shape["dtype_a"] == baseline_candidate["dtype_a"] and shape["layout"] == baseline_candidate["layout"]
    ]
    if not dtype_shapes:
        return None
    baseline_shape = min(dtype_shapes, key=lambda item: (item["k"], item["m"], item["n"]))
    return {
        "bm_name": f"{baseline_candidate['candidate_id']}__{baseline_shape['shape_id']}__dpas_probe__0",
        "stage": "dpas_probe",
        "attempt_index": 0,
        "probe_class": "dpas_baseline",
        "shape": baseline_shape,
        "candidate": baseline_candidate,
    }


def build_compiler_profile_probe_entries(shapes_doc, candidate_space, profiles):
    probe_entries = build_phase_a_probe_entries(shapes_doc, candidate_space)
    probe_entry_by_class = {
        "small_tile": next(
            (
                entry
                for entry in probe_entries
                if entry["probe_class"] == "small" and entry["candidate"].get("runner", "benchmark") == "benchmark"
            ),
            None,
        ),
        "medium_tile": next(
            (
                entry
                for entry in probe_entries
                if entry["probe_class"] == "medium" and entry["candidate"].get("runner", "benchmark") == "benchmark"
            ),
            None,
        ),
        "large_tile": next(
            (
                entry
                for entry in probe_entries
                if entry["probe_class"] == "large" and entry["candidate"].get("runner", "benchmark") == "benchmark"
            ),
            None,
        ),
    }
    compiler_probe_entries = []
    for profile in profiles["profiles"]:
        base_entry = probe_entry_by_class.get(profile.get("candidate_class"))
        if base_entry is None:
            continue
        entry = copy.deepcopy(base_entry)
        entry["stage"] = "compiler_profile_probe"
        entry["probe_class"] = profile["candidate_class"]
        entry["compiler_profile_probe_id"] = profile["compiler_profile_id"]
        entry["compiler_profile_id"] = profile["compiler_profile_id"]
        entry["bm_name"] = (
            f"{entry['candidate']['candidate_id']}__{entry['shape']['shape_id']}__compiler_probe__"
            f"{profile['compiler_profile_id'].replace('.', '_')}"
        )
        compiler_probe_entries.append(entry)
    return compiler_probe_entries


def build_screening_entries(shapes_doc, candidate_space):
    entries = []
    for shape in shapes_doc["shapes"]:
        for candidate in choose_candidates_for_shape(shape, candidate_space["candidates"]):
            entries.append(
                {
                    "bm_name": f"{candidate['candidate_id']}__{shape['shape_id']}__screening__0",
                    "stage": "screening",
                    "attempt_index": 0,
                    "shape": shape,
                    "candidate": candidate,
                }
            )
    return entries


def generate_confirmation_entries(rows, candidate_space, shapes_doc, top_k, confirm_runs):
    shape_map = {shape["shape_id"]: shape for shape in shapes_doc["shapes"]}
    candidate_map = {candidate["candidate_id"]: candidate for candidate in candidate_space["candidates"]}
    grouped = {}
    for row in rows:
        if row["stage"] == "screening" and row["status"] == "pass":
            grouped.setdefault(row["shape_id"], []).append(row)
    entries = []
    for shape_id, shape_rows in grouped.items():
        ranked = sorted(shape_rows, key=lambda row: float(row["avg_tflops"] or 0.0), reverse=True)[:top_k]
        for attempt_index in range(confirm_runs):
            for row in ranked:
                candidate_id = row["candidate_id"]
                entries.append(
                    {
                        "bm_name": f"{candidate_id}__{shape_id}__confirm__{attempt_index}",
                        "stage": "confirm",
                        "attempt_index": attempt_index,
                        "shape": shape_map[shape_id],
                        "candidate": candidate_map[candidate_id],
                    }
                )
    return entries


def write_config(entries, config_path):
    metadata = {}
    with open(config_path, "w", encoding="utf-8") as handle:
        for entry in entries:
            candidate = entry["candidate"]
            shape = entry["shape"]
            runtime_defaults = dict(candidate.get("runtime_defaults", {}))
            runtime_defaults.update(shape.get("runtime_defaults", {}))
            batch_count = shape.get("batch_count", runtime_defaults.get("batch_count", 1))
            alpha = runtime_defaults.get("alpha", 1.0)
            beta = runtime_defaults.get("beta", 0.0)
            is_generated_library_kernel = (
                candidate.get("source") == "generator_manifest"
                or candidate["kernel_name"].startswith("cutlass3x_")
            )
            benchmark_name = "cutlass_library_gemm" if is_generated_library_kernel else candidate["kernel_name"]
            library_options = ""
            if is_generated_library_kernel:
                library_options = (
                    f" --operation_name={candidate['kernel_name']}"
                    f" --layout={shape['layout']}"
                    f" --dtype_a={shape['dtype_a']}"
                    f" --dtype_b={shape['dtype_b']}"
                    f" --dtype_c={shape['dtype_c']}"
                    f" --dtype_d={shape.get('dtype_d', shape['dtype_c'])}"
                    f" --dtype_acc={shape['dtype_acc']}"
                )
            split_options = f" --split_k_slices={candidate['split_k']}" if candidate.get("split_k", 1) > 1 else ""
            handle.write(
                f"{benchmark_name} --bm_name={entry['bm_name']} --m={shape['m']} --n={shape['n']} --k={shape['k']}"
                f" --l={batch_count} --alpha={alpha} --beta={beta}{split_options}{library_options}\n"
            )
            metadata[entry["bm_name"]] = {
                "shape_id": shape["shape_id"],
                "candidate_id": candidate["candidate_id"],
                "compiler_profile_id": entry.get("compiler_profile_id", candidate["compiler_profile_id"]),
                "stage": entry["stage"],
                "attempt_index": entry["attempt_index"],
                "layout": shape["layout"],
                "dtype_a": shape["dtype_a"],
                "dtype_b": shape["dtype_b"],
                "dtype_c": shape["dtype_c"],
                "dtype_d": shape.get("dtype_d", shape["dtype_c"]),
                "dtype_acc": shape["dtype_acc"],
                "m": shape["m"],
                "n": shape["n"],
                "k": shape["k"],
                "batch_count": batch_count,
                "kernel_name": candidate["kernel_name"],
                "split_k": candidate["split_k"],
            }
            metadata[entry["bm_name"]].update(_copy_result_metadata(candidate))
    return metadata
