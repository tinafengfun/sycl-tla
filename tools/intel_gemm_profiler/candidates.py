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
