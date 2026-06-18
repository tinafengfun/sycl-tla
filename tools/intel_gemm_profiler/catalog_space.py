#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import copy
import re
from pathlib import Path

from .schemas import infer_epilogue_metadata, infer_scheduler_metadata
from .source_templates import is_valid_xe2_tile_sg, observed_bmg_template_space


BENCHMARK_GEMM_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "gemm"
STREAMK_TILE_DEF_RE = re.compile(r"BMG_STREAMK_TILE\((\d+),\s*(\d+),\s*(\d+)\)")
STREAMK_SEED_TILE_DEF_PATH = BENCHMARK_GEMM_DIR / "bmg_streamk_seed_tile.def"
STREAMK_EXPANDED_TILE_DEF_PATH = BENCHMARK_GEMM_DIR / "bmg_streamk_expanded_tile.def"
STREAMK_EXHAUSTIVE_MISSING_TILE_DEF_PATH = BENCHMARK_GEMM_DIR / "bmg_streamk_exhaustive_missing_tile.def"


def _load_streamk_tile_definitions(path):
    text = path.read_text(encoding="utf-8")
    return [tuple(map(int, match)) for match in STREAMK_TILE_DEF_RE.findall(text)]


STREAMK_TILE_SHAPES = _load_streamk_tile_definitions(STREAMK_SEED_TILE_DEF_PATH)
EXPANDED_STREAMK_TILE_SHAPES = _load_streamk_tile_definitions(STREAMK_EXPANDED_TILE_DEF_PATH)
BENCHMARK_STREAMK_TILE_SHAPES = sorted(
    set(STREAMK_TILE_SHAPES)
    | set(EXPANDED_STREAMK_TILE_SHAPES)
    | set(_load_streamk_tile_definitions(STREAMK_EXHAUSTIVE_MISSING_TILE_DEF_PATH))
)
EXHAUSTIVE_STREAMK_8X4_TILES = None
SOURCE_OBSERVED_SG8X4_GEMM_TILE_SHAPES = [
    (128, 256, 16),
    (128, 512, 32),
    (256, 192, 64),
    (256, 256, 16),
]
EXPANDED_GEMM_TILE_SHAPES = sorted(
    set(EXPANDED_STREAMK_TILE_SHAPES) | set(SOURCE_OBSERVED_SG8X4_GEMM_TILE_SHAPES)
)
STREAMK_SPLIT_SIZES = (2, 3, 4, 6)
EXHAUSTIVE_REGULAR_GEMM_STAGES = (1, 2, 3)

TRUE_BF16_STREAMK_UNSUPPORTED_REASON = "bf16_accumulate_streamk_not_practical_sycl_atomic_unsupported"
TRUE_BF16_STREAMK_UNSUPPORTED_DETAIL = (
    "True BF16 accumulator/output StreamK is not a practical search target: "
    "StreamK/SplitK reductions require atomic add on the accumulator type, "
    "SYCL atomic_ref does not support cutlass::bfloat16_t, and BF16 accumulate "
    "has poor numerical value for the intended GEMM workloads. Keep this as a "
    "disabled placeholder only."
)
TRUE_BF16_STREAMK_FUTURE_ENABLE_CONDITION = (
    "Enable only if a safe BF16 reduction path is implemented, or if the "
    "candidate is changed to use FP32 accumulation with BF16 output."
)


def _get_exhaustive_8x4_tiles():
    from .constraints import default_constraints as _dc

    global EXHAUSTIVE_STREAMK_8X4_TILES
    if EXHAUSTIVE_STREAMK_8X4_TILES is None:
        cons = _dc()["allowed_values"]
        legal_8x4_tiles = {
            (m, n, k) for m in cons["tile_m"] for n in cons["tile_n"] for k in cons["tile_k"]
            if is_valid_xe2_tile_sg((m, n, k), (8, 4, 1))
        }
        registered_8x4_tiles = sorted(
            tile for tile in BENCHMARK_STREAMK_TILE_SHAPES if is_valid_xe2_tile_sg(tile, (8, 4, 1))
        )
        missing_tiles = sorted(legal_8x4_tiles - set(registered_8x4_tiles))
        if missing_tiles:
            raise RuntimeError(
                "benchmarks_sycl.hpp scheduler registry is missing legal SG8x4 tiles: "
                + ", ".join(f"{m}x{n}x{k}" for m, n, k in missing_tiles)
            )
        seed_tiles = set(STREAMK_TILE_SHAPES)
        EXHAUSTIVE_STREAMK_8X4_TILES = [tile for tile in registered_8x4_tiles if tile not in seed_tiles]
    return EXHAUSTIVE_STREAMK_8X4_TILES


def benchmark_streamk_scheduler_variants():
    return (
        ("StreamK", "streamk", 1),
        ("DataParallel", "data_parallel", 1),
        ("SplitK", "splitk", 1),
    )


def normalize_benchmark_streamk_splitk(entry):
    if entry.get("runner", "benchmark") == "benchmark" and entry.get("streamk_mode") == "splitk":
        entry["split_k"] = 1
    return entry


def dedupe_kernel_entries(entries):
    deduped = []
    seen_entries = set()
    for entry in entries:
        dedupe_key = (
            entry.get("runner", ""),
            entry.get("kernel_name", ""),
            entry.get("kernel_id", ""),
            entry.get("layout", ""),
            entry.get("dtype_a", ""),
            entry.get("dtype_b", ""),
            entry.get("dtype_c", ""),
            entry.get("dtype_d", entry.get("dtype_c", "")),
            entry.get("dtype_acc", ""),
            entry.get("tile_m", 0),
            entry.get("tile_n", 0),
            entry.get("tile_k", 0),
            entry.get("sg_m", 0),
            entry.get("sg_n", 0),
            entry.get("stages", 0),
            entry.get("split_k", 1),
            entry.get("streamk_mode", ""),
            entry.get("support_status", ""),
            entry.get("support_reason", ""),
        )
        if dedupe_key in seen_entries:
            continue
        seen_entries.add(dedupe_key)
        deduped.append(entry)
    return deduped


def benchmark_streamk_tile_candidates(
    name_prefix,
    dtype_a,
    dtype_b,
    dtype_c,
    dtype_acc,
    dtype_d=None,
    tile_shapes=STREAMK_TILE_SHAPES,
    source="seed_catalog_level0",
    instantiation_level=0,
):
    entries = []
    for tile_m, tile_n, tile_k in tile_shapes:
        for name_mode, streamk_mode, split_k in benchmark_streamk_scheduler_variants():
            entries.append(
                {
                    "kernel_name": f"{name_prefix}_RCR_{name_mode}_{tile_m}x{tile_n}x{tile_k}",
                    "layout": "rcr",
                    "dtype_a": dtype_a,
                    "dtype_b": dtype_b,
                    "dtype_c": dtype_c,
                    "dtype_d": dtype_d or dtype_c,
                    "dtype_acc": dtype_acc,
                    "tile_m": tile_m,
                    "tile_n": tile_n,
                    "tile_k": tile_k,
                    "sg_m": 8,
                    "sg_n": 4,
                    "stages": 2,
                    "split_k": split_k,
                    "streamk_mode": streamk_mode,
                    "kernel_schedule": "KernelXeCooperative",
                    "tile_scheduler": "StreamKScheduler",
                    "source": source,
                    "instantiation_level": instantiation_level,
                }
            )
    return entries


def exhaustive_streamk_tile_candidates(
    name_prefix,
    dtype_a,
    dtype_b,
    dtype_c,
    dtype_acc,
    dtype_d=None,
    constraints=None,
    source="exhaustive_streamk_catalog",
    instantiation_level=3,
    min_tile_k=32,
    layout="rcr",
):
    allowed = (constraints or {}).get("allowed_values", {})
    limits = (constraints or {}).get("limits", {})
    valid_sg_sizes = limits.get("valid_subgroup_sizes")
    tile_m_values = allowed.get("tile_m", [8, 16, 32, 64, 128, 256, 512])
    tile_n_values = allowed.get("tile_n", [32, 64, 96, 128, 192, 256, 512])
    tile_k_values = [k for k in allowed.get("tile_k", [16, 32, 64]) if k >= min_tile_k]
    entries = []
    for tile_m in tile_m_values:
        for tile_n in tile_n_values:
            for tile_k in tile_k_values:
                if not is_valid_xe2_tile_sg((tile_m, tile_n, tile_k), (8, 4, 1), sg_product_set=valid_sg_sizes):
                    continue
                for name_mode, streamk_mode, split_k in benchmark_streamk_scheduler_variants():
                    entries.append(
                        {
                            "kernel_name": f"{name_prefix}_{layout.upper()}_{name_mode}_{tile_m}x{tile_n}x{tile_k}",
                            "layout": layout,
                            "dtype_a": dtype_a,
                            "dtype_b": dtype_b,
                            "dtype_c": dtype_c,
                            "dtype_d": dtype_d or dtype_c,
                            "dtype_acc": dtype_acc,
                            "tile_m": tile_m,
                            "tile_n": tile_n,
                            "tile_k": tile_k,
                            "sg_m": 8,
                            "sg_n": 4,
                            "stages": 2,
                            "split_k": split_k,
                            "streamk_mode": streamk_mode,
                            "kernel_schedule": "KernelXeCooperative",
                            "tile_scheduler": "StreamKScheduler",
                            "source": source,
                            "instantiation_level": instantiation_level,
                        }
                    )
    return entries


def exhaustive_streamk_tile_stage_candidates(
    name_prefix,
    dtype_a,
    dtype_b,
    dtype_c,
    dtype_acc,
    dtype_d=None,
    constraints=None,
    source="exhaustive_streamk_catalog",
    instantiation_level=5,
    min_tile_k=32,
    layout="rcr",
):
    allowed = (constraints or {}).get("allowed_values", {})
    limits = (constraints or {}).get("limits", {})
    valid_sg_sizes = limits.get("valid_subgroup_sizes")
    tile_m_values = allowed.get("tile_m", [8, 16, 32, 64, 128, 256, 512])
    tile_n_values = allowed.get("tile_n", [32, 64, 96, 128, 192, 256, 512])
    tile_k_values = [k for k in allowed.get("tile_k", [16, 32, 64]) if k >= min_tile_k]
    sg_m_values = allowed.get("sg_m", [1, 2, 4, 8])
    sg_n_values = allowed.get("sg_n", [2, 4, 8])
    stage_values = [stage for stage in allowed.get("stages", list(EXHAUSTIVE_REGULAR_GEMM_STAGES)) if stage in EXHAUSTIVE_REGULAR_GEMM_STAGES]
    entries = []
    for tile_m in tile_m_values:
        for tile_n in tile_n_values:
            for tile_k in tile_k_values:
                for sg_m in sg_m_values:
                    for sg_n in sg_n_values:
                        if not is_valid_xe2_tile_sg((tile_m, tile_n, tile_k), (sg_m, sg_n, 1), sg_product_set=valid_sg_sizes):
                            continue
                        for stages in stage_values:
                            for name_mode, streamk_mode, split_k in benchmark_streamk_scheduler_variants():
                                entries.append(
                                    {
                                        "kernel_name": (
                                            f"{name_prefix}_{layout.upper()}_{name_mode}_"
                                            f"{tile_m}x{tile_n}x{tile_k}_SG{sg_m}x{sg_n}_ST{stages}"
                                        ),
                                        "layout": layout,
                                        "dtype_a": dtype_a,
                                        "dtype_b": dtype_b,
                                        "dtype_c": dtype_c,
                                        "dtype_d": dtype_d or dtype_c,
                                        "dtype_acc": dtype_acc,
                                        "tile_m": tile_m,
                                        "tile_n": tile_n,
                                        "tile_k": tile_k,
                                        "sg_m": sg_m,
                                        "sg_n": sg_n,
                                        "stages": stages,
                                        "split_k": split_k,
                                        "streamk_mode": streamk_mode,
                                        "kernel_schedule": "KernelXeCooperative",
                                        "tile_scheduler": "StreamKScheduler",
                                        "source": source,
                                        "instantiation_level": instantiation_level,
                                    }
                                )
    return entries


def benchmark_gemm_tile_candidates(
    name_prefix,
    dtype_a,
    dtype_b,
    dtype_c,
    dtype_acc,
    dtype_d=None,
    layout="rcr",
    tile_shapes=EXPANDED_GEMM_TILE_SHAPES,
    source="expanded_gemm_catalog",
    instantiation_level=1,
):
    return [
        {
            "kernel_name": f"{name_prefix}_{layout.upper()}_Gemm_{tile_m}x{tile_n}x{tile_k}_SG8x4",
            "layout": layout,
            "dtype_a": dtype_a,
            "dtype_b": dtype_b,
            "dtype_c": dtype_c,
            "dtype_d": dtype_d or dtype_c,
            "dtype_acc": dtype_acc,
            "tile_m": tile_m,
            "tile_n": tile_n,
            "tile_k": tile_k,
            "sg_m": 8,
            "sg_n": 4,
            "stages": 2,
            "split_k": 1,
            "kernel_schedule": "KernelXe",
            "tile_scheduler": "Gemm",
            "source": source,
            "instantiation_level": instantiation_level,
        }
        for tile_m, tile_n, tile_k in tile_shapes
    ]


def source_template_gemm_tile_candidates(
    name_prefix,
    dtype_a,
    dtype_b,
    dtype_c,
    dtype_acc,
    dtype_d=None,
    layout="rcr",
    source_template_space=None,
    source="source_template_gemm_catalog",
    instantiation_level=2,
):
    source_template_space = source_template_space or observed_bmg_template_space()
    entries = []
    for pair in source_template_space["valid_tile_sg_pairs"]:
        tile_m, tile_n, tile_k = pair["tile_shape"]
        sg_m, sg_n, _ = pair["sg_layout"]
        entries.append(
            {
                "kernel_name": f"{name_prefix}_{layout.upper()}_Gemm_{tile_m}x{tile_n}x{tile_k}_SG{sg_m}x{sg_n}",
                "layout": layout,
                "dtype_a": dtype_a,
                "dtype_b": dtype_b,
                "dtype_c": dtype_c,
                "dtype_d": dtype_d or dtype_c,
                "dtype_acc": dtype_acc,
                "tile_m": tile_m,
                "tile_n": tile_n,
                "tile_k": tile_k,
                "sg_m": sg_m,
                "sg_n": sg_n,
                "stages": 2,
                "split_k": 1,
                "kernel_schedule": "KernelXe",
                "tile_scheduler": "Gemm",
                "source": source,
                "instantiation_level": instantiation_level,
            }
        )
    return entries


def exhaustive_regular_gemm_tile_candidates(
    name_prefix,
    dtype_a,
    dtype_b,
    dtype_c,
    dtype_acc,
    dtype_d=None,
    layout="rcr",
    constraints=None,
    stages=EXHAUSTIVE_REGULAR_GEMM_STAGES,
    source="exhaustive_regular_gemm_catalog",
    instantiation_level=3,
):
    allowed = (constraints or {}).get("allowed_values", {})
    tile_m_values = allowed.get("tile_m", [8, 16, 32, 64, 128, 256, 512])
    tile_n_values = allowed.get("tile_n", [32, 64, 96, 128, 192, 256, 512])
    tile_k_values = allowed.get("tile_k", [16, 32, 64])
    sg_m_values = allowed.get("sg_m", [1, 2, 4, 8])
    sg_n_values = allowed.get("sg_n", [2, 4, 8])
    stage_values = [stage for stage in stages if stage in allowed.get("stages", list(stages))]
    limits = (constraints or {}).get("limits", {})
    valid_sg_sizes = limits.get("valid_subgroup_sizes")
    if valid_sg_sizes is None:
        valid_sg_sizes = [16, 32]
    entries = []
    for tile_m in tile_m_values:
        for tile_n in tile_n_values:
            for tile_k in tile_k_values:
                for sg_m in sg_m_values:
                    for sg_n in sg_n_values:
                        if not is_valid_xe2_tile_sg((tile_m, tile_n, tile_k), (sg_m, sg_n, 1), sg_product_set=valid_sg_sizes):
                            continue
                        for stage in stage_values:
                            entries.append(
                                {
                                    "kernel_name": (
                                        f"{name_prefix}_{layout.upper()}_GemmExhaustive_"
                                        f"{tile_m}x{tile_n}x{tile_k}_SG{sg_m}x{sg_n}_ST{stage}"
                                    ),
                                    "layout": layout,
                                    "dtype_a": dtype_a,
                                    "dtype_b": dtype_b,
                                    "dtype_c": dtype_c,
                                    "dtype_d": dtype_d or dtype_c,
                                    "dtype_acc": dtype_acc,
                                    "tile_m": tile_m,
                                    "tile_n": tile_n,
                                    "tile_k": tile_k,
                                    "sg_m": sg_m,
                                    "sg_n": sg_n,
                                    "stages": stage,
                                    "split_k": 1,
                                    "kernel_schedule": "KernelXe",
                                    "tile_scheduler": "Gemm",
                                    "source": source,
                                    "instantiation_level": instantiation_level,
                                }
                            )
    return entries


def unsupported_true_bf16_streamk_example(kernel_suffix, streamk_mode, split_k):
    return {
        "kernel_name": f"03_bmg_gemm_streamk_{kernel_suffix}_bf16_bf16",
        "layout": "rcr",
        "dtype_a": "bf16",
        "dtype_b": "bf16",
        "dtype_c": "bf16",
        "dtype_d": "bf16",
        "dtype_acc": "bf16",
        "tile_m": 256,
        "tile_n": 256,
        "tile_k": 32,
        "sg_m": 8,
        "sg_n": 4,
        "stages": 2,
        "split_k": split_k,
        "runner": "streamk_example",
        "streamk_mode": streamk_mode,
        "streamk_dtype_preset": "bf16_bf16",
        "support_status": "unsupported",
        "support_reason": TRUE_BF16_STREAMK_UNSUPPORTED_REASON,
        "support_detail": TRUE_BF16_STREAMK_UNSUPPORTED_DETAIL,
        "support_future_enable_condition": TRUE_BF16_STREAMK_FUTURE_ENABLE_CONDITION,
    }


def weight_only_mixed_dtype_example_candidates():
    return {
        "bf16_s8": [
            {
                "kernel_name": "02_bmg_gemm_bf16_s8_bf16",
                "layout": "rrr",
                "dtype_a": "bf16",
                "dtype_b": "s8",
                "dtype_c": "f32",
                "dtype_d": "f32",
                "dtype_acc": "f32",
                "tile_m": 256,
                "tile_n": 256,
                "tile_k": 32,
                "sg_m": 8,
                "sg_n": 4,
                "stages": 3,
                "split_k": 1,
                "runner": "mixed_bf16_s8_example",
                "source": "weight_only_mixed_dtype_example",
                "example_family": "02_bmg_gemm_mixed_dtype",
                "quant_mode": "weight_only_int8",
                "scale_mode": "groupwise",
                "benchmark_target": "02_bmg_gemm_bf16_s8_bf16",
            }
        ],
        "f16_s8": [
            {
                "kernel_name": "02_bmg_gemm_f16_s8_f16_tensorwise",
                "layout": "rrr",
                "dtype_a": "f16",
                "dtype_b": "s8",
                "dtype_c": "f32",
                "dtype_d": "f32",
                "dtype_acc": "f32",
                "tile_m": 256,
                "tile_n": 256,
                "tile_k": 32,
                "sg_m": 8,
                "sg_n": 4,
                "stages": 2,
                "split_k": 1,
                "runner": "mixed_f16_s8_example",
                "source": "weight_only_mixed_dtype_example",
                "example_family": "02_bmg_gemm_mixed_dtype",
                "quant_mode": "weight_only_int8",
                "scale_mode": "tensorwise",
                "benchmark_target": "02_bmg_gemm_f16_s8_f16_tensorwise",
            }
        ],
    }


SEED_KERNELS = {
    "bf16": [
        {"kernel_name": "BmgGemmBF16BF16FP32_RRR_TileShape_512_256_32", "layout": "rrr", "dtype_a": "bf16", "dtype_b": "bf16", "dtype_c": "f32", "dtype_acc": "f32", "tile_m": 512, "tile_n": 256, "tile_k": 32, "sg_m": 8, "sg_n": 4, "stages": 2, "split_k": 1},
        {"kernel_name": "BmgGemmBF16BF16FP32_RCR_5", "layout": "rcr", "dtype_a": "bf16", "dtype_b": "bf16", "dtype_c": "f32", "dtype_acc": "f32", "tile_m": 8, "tile_n": 128, "tile_k": 32, "sg_m": 1, "sg_n": 4, "stages": 2, "split_k": 1},
        {"kernel_name": "BmgGemmBF16BF16FP32_RCR_7", "layout": "rcr", "dtype_a": "bf16", "dtype_b": "bf16", "dtype_c": "f32", "dtype_acc": "f32", "tile_m": 8, "tile_n": 128, "tile_k": 32, "sg_m": 1, "sg_n": 8, "stages": 2, "split_k": 1},
        {"kernel_name": "BmgGemmBF16BF16FP32_RCR_9", "layout": "rcr", "dtype_a": "bf16", "dtype_b": "bf16", "dtype_c": "f32", "dtype_acc": "f32", "tile_m": 8, "tile_n": 64, "tile_k": 32, "sg_m": 1, "sg_n": 4, "stages": 2, "split_k": 1},
        {"kernel_name": "BmgGemmBF16BF16FP32_RCR_16", "layout": "rcr", "dtype_a": "bf16", "dtype_b": "bf16", "dtype_c": "f32", "dtype_acc": "f32", "tile_m": 16, "tile_n": 64, "tile_k": 32, "sg_m": 2, "sg_n": 4, "stages": 2, "split_k": 1},
        {"kernel_name": "BmgGemmBF16BF16FP32_RCR_17", "layout": "rcr", "dtype_a": "bf16", "dtype_b": "bf16", "dtype_c": "f32", "dtype_acc": "f32", "tile_m": 64, "tile_n": 128, "tile_k": 32, "sg_m": 4, "sg_n": 4, "stages": 2, "split_k": 1},
        {"kernel_name": "BmgGemmBF16BF16FP32_RCR_18", "layout": "rcr", "dtype_a": "bf16", "dtype_b": "bf16", "dtype_c": "f32", "dtype_acc": "f32", "tile_m": 128, "tile_n": 128, "tile_k": 32, "sg_m": 4, "sg_n": 4, "stages": 2, "split_k": 1},
        {"kernel_name": "BmgGemmBF16BF16FP32_RCR_19", "layout": "rcr", "dtype_a": "bf16", "dtype_b": "bf16", "dtype_c": "f32", "dtype_acc": "f32", "tile_m": 128, "tile_n": 256, "tile_k": 32, "sg_m": 4, "sg_n": 4, "stages": 2, "split_k": 1},
        {"kernel_name": "BmgGemmBF16BF16FP32_RCR_6", "layout": "rcr", "dtype_a": "bf16", "dtype_b": "bf16", "dtype_c": "f32", "dtype_acc": "f32", "tile_m": 256, "tile_n": 256, "tile_k": 32, "sg_m": 8, "sg_n": 4, "stages": 2, "split_k": 1},
        *benchmark_streamk_tile_candidates("BmgGemmBF16BF16FP32", "bf16", "bf16", "f32", "f32"),
        unsupported_true_bf16_streamk_example("streamk", "streamk", 1),
        unsupported_true_bf16_streamk_example("dp", "data_parallel", 1),
        unsupported_true_bf16_streamk_example("splitk", "splitk", 2),
    ],
    "f16": [
        {"kernel_name": "BmgGemmFP16FP16FP32_RCR_5", "layout": "rcr", "dtype_a": "f16", "dtype_b": "f16", "dtype_c": "f32", "dtype_acc": "f32", "tile_m": 8, "tile_n": 128, "tile_k": 32, "sg_m": 1, "sg_n": 4, "stages": 2, "split_k": 1},
        {"kernel_name": "BmgGemmFP16FP16FP32_RCR_7", "layout": "rcr", "dtype_a": "f16", "dtype_b": "f16", "dtype_c": "f32", "dtype_acc": "f32", "tile_m": 8, "tile_n": 128, "tile_k": 32, "sg_m": 1, "sg_n": 8, "stages": 2, "split_k": 1},
        {"kernel_name": "BmgGemmFP16FP16FP32_RCR_9", "layout": "rcr", "dtype_a": "f16", "dtype_b": "f16", "dtype_c": "f32", "dtype_acc": "f32", "tile_m": 8, "tile_n": 64, "tile_k": 32, "sg_m": 1, "sg_n": 4, "stages": 2, "split_k": 1},
        {"kernel_name": "BmgGemmFP16FP16FP32_RCR_16", "layout": "rcr", "dtype_a": "f16", "dtype_b": "f16", "dtype_c": "f32", "dtype_acc": "f32", "tile_m": 16, "tile_n": 64, "tile_k": 32, "sg_m": 2, "sg_n": 4, "stages": 2, "split_k": 1},
        {"kernel_name": "BmgGemmFP16FP16FP32_RCR_17", "layout": "rcr", "dtype_a": "f16", "dtype_b": "f16", "dtype_c": "f32", "dtype_acc": "f32", "tile_m": 64, "tile_n": 128, "tile_k": 32, "sg_m": 4, "sg_n": 4, "stages": 2, "split_k": 1},
        {"kernel_name": "BmgGemmFP16FP16FP32_RCR_18", "layout": "rcr", "dtype_a": "f16", "dtype_b": "f16", "dtype_c": "f32", "dtype_acc": "f32", "tile_m": 128, "tile_n": 128, "tile_k": 32, "sg_m": 4, "sg_n": 4, "stages": 2, "split_k": 1},
        {"kernel_name": "BmgGemmFP16FP16FP32_RCR_19", "layout": "rcr", "dtype_a": "f16", "dtype_b": "f16", "dtype_c": "f32", "dtype_acc": "f32", "tile_m": 128, "tile_n": 256, "tile_k": 32, "sg_m": 4, "sg_n": 4, "stages": 2, "split_k": 1},
        {"kernel_name": "BmgGemmFP16FP16FP32_RCR_6", "layout": "rcr", "dtype_a": "f16", "dtype_b": "f16", "dtype_c": "f32", "dtype_acc": "f32", "tile_m": 256, "tile_n": 256, "tile_k": 32, "sg_m": 8, "sg_n": 4, "stages": 2, "split_k": 1},
        *benchmark_streamk_tile_candidates("BmgGemmF16F16FP32", "f16", "f16", "f32", "f32"),
        {"kernel_name": "03_bmg_gemm_streamk_streamk_f16", "layout": "rcr", "dtype_a": "f16", "dtype_b": "f16", "dtype_c": "f32", "dtype_acc": "f32", "tile_m": 256, "tile_n": 256, "tile_k": 32, "sg_m": 8, "sg_n": 4, "stages": 2, "split_k": 1, "runner": "streamk_example", "streamk_mode": "streamk"},
        {"kernel_name": "03_bmg_gemm_streamk_dp_f16", "layout": "rcr", "dtype_a": "f16", "dtype_b": "f16", "dtype_c": "f32", "dtype_acc": "f32", "tile_m": 256, "tile_n": 256, "tile_k": 32, "sg_m": 8, "sg_n": 4, "stages": 2, "split_k": 1, "runner": "streamk_example", "streamk_mode": "data_parallel"},
        {"kernel_name": "03_bmg_gemm_streamk_splitk_f16", "layout": "rcr", "dtype_a": "f16", "dtype_b": "f16", "dtype_c": "f32", "dtype_acc": "f32", "tile_m": 256, "tile_n": 256, "tile_k": 32, "sg_m": 8, "sg_n": 4, "stages": 2, "split_k": 2, "runner": "streamk_example", "streamk_mode": "splitk"},
        *benchmark_streamk_tile_candidates("BmgGemmF16F16F16", "f16", "f16", "f16", "f16", dtype_d="f16"),
    ],
    "tf32": [
        *benchmark_streamk_tile_candidates("BmgGemmTF32TF32FP32", "tf32", "tf32", "f32", "f32"),
    ],
    **weight_only_mixed_dtype_example_candidates(),
}


def ilp_class(seed):
    ilp = (seed["tile_m"] // max(seed["sg_m"], 1) // 8) * (seed["tile_n"] // max(seed["sg_n"], 1) // 16)
    if ilp >= 16:
        return "ilp16"
    if ilp >= 8:
        return "ilp8"
    return "ilp4"


def apply_scheduler_metadata(entry):
    scheduler_metadata = infer_scheduler_metadata(entry)
    for key, value in scheduler_metadata.items():
        entry.setdefault(key, value)
    return entry


def apply_epilogue_metadata(entry):
    for key, value in infer_epilogue_metadata(entry).items():
        entry.setdefault(key, value)
    return entry


def kernel_catalog_entry(dtype, seed):
    entry = copy.deepcopy(seed)
    entry.setdefault("dtype_d", entry["dtype_c"])
    entry.setdefault("runner", "benchmark")
    entry["kernel_id"] = seed["kernel_name"]
    entry.setdefault("instantiation_level", 0)
    if entry["runner"] == "benchmark":
        entry["benchmark_target"] = "cutlass_benchmarks_gemm_sycl"
    elif entry["runner"] == "streamk_example":
        entry.setdefault("benchmark_target", "03_bmg_gemm_streamk")
    elif entry["runner"] == "mixed_bf16_s8_example":
        entry.setdefault("benchmark_target", "02_bmg_gemm_bf16_s8_bf16")
    elif entry["runner"] == "mixed_f16_s8_example":
        entry.setdefault("benchmark_target", "02_bmg_gemm_f16_s8_f16_tensorwise")
    else:
        entry.setdefault("benchmark_target", entry["runner"])
    entry["grf_mode"] = 256
    entry["ilp_class"] = ilp_class(entry)
    entry["streamk_mode"] = entry.get("streamk_mode", "")
    entry.setdefault("streamk_dtype_preset", entry["dtype_a"] if entry["runner"] == "streamk_example" else "")
    entry.setdefault("support_status", "supported")
    entry.setdefault("support_reason", "")
    entry["batch_count"] = 1
    entry["runtime_defaults"] = {}
    entry["allowed_runtime_sweeps"] = ["shape_id", "m", "n", "k", "batch_count"]
    entry.setdefault("source", "seed_catalog_level0")
    entry["dtype_family"] = dtype
    entry.setdefault("mma_atom", "XE_DPAS_TT")
    entry.setdefault("gmem_copy_atom_a", "auto")
    entry.setdefault("gmem_copy_atom_b", "auto")
    entry.setdefault("epilogue_op", "LinearCombination")
    entry.setdefault("epilogue_tile", "auto")
    entry.setdefault("epilogue_copy_atom_c", "auto")
    entry.setdefault("epilogue_copy_atom_d", "auto")
    normalize_benchmark_streamk_splitk(entry)
    apply_epilogue_metadata(entry)
    apply_scheduler_metadata(entry)
    return entry
