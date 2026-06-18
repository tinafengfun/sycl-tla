#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import copy

from .candidate_manifest import build_candidate_build_manifest, build_selected_kernel_batches
from .candidate_space import candidate_class, candidate_id_for, copy_result_metadata, generate_candidate_space, select_compiler_profile_id
from .catalog import SEED_KERNELS
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
from .schemas import SCHEMA_VERSION
from .utils import now_iso


DEFAULT_SHAPE_PRESETS = {
    "bf16": {
        "layout": "rcr",
        "dtype_a": "bf16",
        "dtype_b": "bf16",
        "dtype_c": "f32",
        "dtype_d": "f32",
        "dtype_acc": "f32",
    },
    "f16": {
        "layout": "rcr",
        "dtype_a": "f16",
        "dtype_b": "f16",
        "dtype_c": "f32",
        "dtype_d": "f32",
        "dtype_acc": "f32",
    },
    "tf32": {
        "layout": "rcr",
        "dtype_a": "tf32",
        "dtype_b": "tf32",
        "dtype_c": "f32",
        "dtype_d": "f32",
        "dtype_acc": "f32",
    },
    "bf16_s8": {
        "layout": "rrr",
        "dtype_a": "bf16",
        "dtype_b": "s8",
        "dtype_c": "f32",
        "dtype_d": "f32",
        "dtype_acc": "f32",
        "quant_mode": "weight_only_int8",
        "scale_mode": "groupwise",
    },
    "f16_s8": {
        "layout": "rrr",
        "dtype_a": "f16",
        "dtype_b": "s8",
        "dtype_c": "f32",
        "dtype_d": "f32",
        "dtype_acc": "f32",
        "quant_mode": "weight_only_int8",
        "scale_mode": "tensorwise",
    },
}


def _shape_preset(dtype):
    try:
        return DEFAULT_SHAPE_PRESETS[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported default shape preset: {dtype}") from exc


def default_shapes(dtype):
    base = [
        {"m": 1, "n": 4096, "k": 14336, "tags": ["decode"]},
        {"m": 8, "n": 4096, "k": 4096, "tags": ["decode"]},
        {"m": 64, "n": 4096, "k": 4096, "tags": ["prefill"]},
        {"m": 256, "n": 4096, "k": 8192, "tags": ["prefill"]},
    ]
    preset = _shape_preset(dtype)
    shapes = []
    for item in base:
        shapes.append(
            {
                "shape_id": f"{preset['layout']}_{dtype}_{item['m']}_{item['n']}_{item['k']}",
                "layout": preset["layout"],
                "dtype_a": preset["dtype_a"],
                "dtype_b": preset["dtype_b"],
                "dtype_c": preset["dtype_c"],
                "dtype_d": preset["dtype_d"],
                "dtype_acc": preset["dtype_acc"],
                "m": item["m"],
                "n": item["n"],
                "k": item["k"],
                "batch_count": 1,
                "runtime_defaults": {},
                "quant_mode": preset.get("quant_mode", ""),
                "scale_mode": preset.get("scale_mode", ""),
                "tags": item["tags"],
            }
        )
    return {"schema_version": SCHEMA_VERSION, "generated_at": now_iso(), "shape_set_id": f"default_{dtype}_decode_prefill", "source": "predefined", "shapes": shapes}


def dry_run_shapes(dtype):
    preset = _shape_preset(dtype)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "shape_set_id": f"dry_run_{dtype}",
        "source": "dry_run",
        "shapes": [
            {
                "shape_id": f"dry_run_{preset['layout']}_{dtype}_1_64_32",
                "layout": preset["layout"],
                "dtype_a": preset["dtype_a"],
                "dtype_b": preset["dtype_b"],
                "dtype_c": preset["dtype_c"],
                "dtype_d": preset["dtype_d"],
                "dtype_acc": preset["dtype_acc"],
                "m": 1,
                "n": 64,
                "k": 32,
                "batch_count": 1,
                "runtime_defaults": {},
                "quant_mode": preset.get("quant_mode", ""),
                "scale_mode": preset.get("scale_mode", ""),
                "tags": ["dry_run"],
            }
        ],
    }
