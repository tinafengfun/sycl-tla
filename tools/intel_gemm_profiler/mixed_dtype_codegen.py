#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

from __future__ import annotations

import re
from pathlib import Path

from .catalog_space import kernel_catalog_entry
from .schemas import SCHEMA_VERSION, SEARCH_RUNTIME_SCHEMA
from .source_templates import DEFAULT_SOURCE_ROOT, observed_bmg_template_space
from .utils import ensure_dir, now_iso, write_json


BF16_S8_TEMPLATE = (
    DEFAULT_SOURCE_ROOT
    / "examples"
    / "02_bmg_gemm_mixed_dtype"
    / "02_bmg_gemm_bf16_s8_bf16.cpp"
)
F16_S8_TEMPLATE = (
    DEFAULT_SOURCE_ROOT
    / "examples"
    / "02_bmg_gemm_mixed_dtype"
    / "02_bmg_gemm_f16_s8_f16_tensorwise.cpp"
)

DEFAULT_WEIGHT_ONLY_STAGE_VALUES = {
    "bf16_s8": (2, 3),
    "f16_s8": (2,),
}

DEFAULT_WEIGHT_ONLY_RUNTIME_DEFAULTS = {
    "bf16_s8": {"mode": 2, "g": 128},
    "f16_s8": {},
}

DEFAULT_WEIGHT_ONLY_VALID_TILE_SG_PAIRS = {
    "bf16_s8": (
        {"tile_shape": [128, 128, 32], "sg_layout": [4, 4, 1]},
        {"tile_shape": [128, 128, 64], "sg_layout": [4, 4, 1]},
        {"tile_shape": [128, 256, 32], "sg_layout": [4, 4, 1]},
        {"tile_shape": [256, 64, 32], "sg_layout": [8, 2, 1]},
        {"tile_shape": [256, 128, 32], "sg_layout": [4, 4, 1]},
        {"tile_shape": [256, 128, 32], "sg_layout": [8, 2, 1]},
        {"tile_shape": [256, 128, 32], "sg_layout": [8, 4, 1]},
        {"tile_shape": [256, 256, 32], "sg_layout": [8, 4, 1]},
    ),
    "f16_s8": (
        {"tile_shape": [128, 128, 32], "sg_layout": [4, 4, 1]},
        {"tile_shape": [128, 128, 64], "sg_layout": [4, 4, 1]},
        {"tile_shape": [128, 256, 32], "sg_layout": [4, 4, 1]},
        {"tile_shape": [256, 64, 32], "sg_layout": [8, 2, 1]},
        {"tile_shape": [256, 128, 32], "sg_layout": [4, 4, 1]},
        {"tile_shape": [256, 128, 32], "sg_layout": [8, 2, 1]},
        {"tile_shape": [256, 128, 32], "sg_layout": [8, 4, 1]},
        {"tile_shape": [256, 256, 32], "sg_layout": [8, 4, 1]},
    ),
}

DEFAULT_WEIGHT_ONLY_PAIR_POLICY = "curated_v1"
DEFAULT_WEIGHT_ONLY_BRUTEFORCE_TILE_K_VALUES = (32, 64)
DEFAULT_WEIGHT_ONLY_BRUTEFORCE_SG_LAYOUTS = (
    (4, 4, 1),
    (8, 2, 1),
    (8, 4, 1),
)
DEFAULT_WEIGHT_ONLY_BRUTEFORCE_POLICIES = {
    "template_bruteforce_v1": {"min_tile_m": 64, "min_tile_n": 128},
    "template_bruteforce_v2": {"min_tile_m": 32, "min_tile_n": 32},
}


def _weight_only_family_spec(dtype_family: str) -> dict:
    specs = {
        "bf16_s8": {
            "layout": "rrr",
            "dtype_a": "bf16",
            "dtype_b": "s8",
            "dtype_c": "f32",
            "dtype_d": "f32",
            "dtype_acc": "f32",
            "scale_mode": "groupwise",
            "template_path": BF16_S8_TEMPLATE,
            "benchmark_target_prefix": "weight_only_bf16_s8",
        },
        "f16_s8": {
            "layout": "rrr",
            "dtype_a": "f16",
            "dtype_b": "s8",
            "dtype_c": "f32",
            "dtype_d": "f32",
            "dtype_acc": "f32",
            "scale_mode": "tensorwise",
            "template_path": F16_S8_TEMPLATE,
            "benchmark_target_prefix": "weight_only_f16_s8",
        },
    }
    try:
        return specs[dtype_family]
    except KeyError as exc:
        raise ValueError(f"Unsupported weight-only mixed-dtype family: {dtype_family}") from exc


def _allowed_values(constraints: dict | None, key: str, default_values):
    allowed = (constraints or {}).get("allowed_values", {})
    values = allowed.get(key)
    if values is None:
        return list(default_values)
    return [value for value in default_values if value in values]


def _weight_only_stage_values(dtype_family: str, constraints: dict | None):
    return _allowed_values(constraints, "stages", DEFAULT_WEIGHT_ONLY_STAGE_VALUES[dtype_family])


def _weight_only_pair_policy(constraints: dict | None) -> str:
    return (constraints or {}).get("weight_only_pair_policy", DEFAULT_WEIGHT_ONLY_PAIR_POLICY)


def _template_bruteforce_weight_only_tile_sg_pairs(
    constraints: dict | None,
    source_template_space: dict,
    *,
    min_tile_m: int,
    min_tile_n: int,
) -> list[dict]:
    allowed_sg_layouts = set(DEFAULT_WEIGHT_ONLY_BRUTEFORCE_SG_LAYOUTS)
    allowed_tile_k = set(_allowed_values(constraints, "tile_k", DEFAULT_WEIGHT_ONLY_BRUTEFORCE_TILE_K_VALUES))
    valid_pairs = []
    for pair in source_template_space.get("valid_tile_sg_pairs", []):
        tile_shape = tuple(pair["tile_shape"])
        sg_layout = tuple(pair["sg_layout"])
        if tile_shape[0] < min_tile_m:
            continue
        if tile_shape[1] < min_tile_n:
            continue
        if tile_shape[2] not in allowed_tile_k:
            continue
        if sg_layout not in allowed_sg_layouts:
            continue
        valid_pairs.append({"tile_shape": list(tile_shape), "sg_layout": list(sg_layout)})
    return sorted(
        valid_pairs,
        key=lambda pair: (
            pair["tile_shape"][0],
            pair["tile_shape"][1],
            pair["tile_shape"][2],
            pair["sg_layout"][0],
            pair["sg_layout"][1],
            pair["sg_layout"][2],
        ),
    )


def _base_weight_only_tile_sg_pairs(
    dtype_family: str,
    constraints: dict | None,
    source_template_space: dict,
) -> list[dict]:
    pair_policy = _weight_only_pair_policy(constraints)
    if pair_policy == "curated_v1":
        return list(DEFAULT_WEIGHT_ONLY_VALID_TILE_SG_PAIRS[dtype_family])
    if pair_policy in DEFAULT_WEIGHT_ONLY_BRUTEFORCE_POLICIES:
        return _template_bruteforce_weight_only_tile_sg_pairs(
            constraints,
            source_template_space,
            **DEFAULT_WEIGHT_ONLY_BRUTEFORCE_POLICIES[pair_policy],
        )
    raise ValueError(f"Unsupported weight-only pair policy: {pair_policy}")


def _valid_weight_only_tile_sg_pairs(
    dtype_family: str,
    constraints: dict | None,
    source_template_space: dict,
):
    valid_pairs = _base_weight_only_tile_sg_pairs(dtype_family, constraints, source_template_space)
    allowed_tile_m = set(_allowed_values(constraints, "tile_m", [pair["tile_shape"][0] for pair in valid_pairs]))
    allowed_tile_n = set(_allowed_values(constraints, "tile_n", [pair["tile_shape"][1] for pair in valid_pairs]))
    allowed_tile_k = set(_allowed_values(constraints, "tile_k", [pair["tile_shape"][2] for pair in valid_pairs]))
    allowed_sg_m = set(_allowed_values(constraints, "sg_m", [pair["sg_layout"][0] for pair in valid_pairs]))
    allowed_sg_n = set(_allowed_values(constraints, "sg_n", [pair["sg_layout"][1] for pair in valid_pairs]))
    return [
        pair
        for pair in valid_pairs
        if pair["tile_shape"][0] in allowed_tile_m
        and pair["tile_shape"][1] in allowed_tile_n
        and pair["tile_shape"][2] in allowed_tile_k
        and pair["sg_layout"][0] in allowed_sg_m
        and pair["sg_layout"][1] in allowed_sg_n
    ]


def weight_only_mixed_dtype_candidates(
    *,
    constraints: dict | None = None,
    source_template_space: dict | None = None,
    dtype_families=("bf16_s8", "f16_s8"),
):
    source_template_space = source_template_space or observed_bmg_template_space()
    entries = []
    for dtype_family in dtype_families:
        spec = _weight_only_family_spec(dtype_family)
        pairs = _valid_weight_only_tile_sg_pairs(dtype_family, constraints, source_template_space)
        for pair in pairs:
            tile_m, tile_n, tile_k = pair["tile_shape"]
            sg_m, sg_n, _ = pair["sg_layout"]
            for stage in _weight_only_stage_values(dtype_family, constraints):
                entries.append(
                    {
                        "kernel_name": (
                            f"WtOnly{spec['dtype_a'].upper()}S8F32_{spec['layout'].upper()}_"
                            f"{tile_m}x{tile_n}x{tile_k}_SG{sg_m}x{sg_n}_ST{stage}"
                        ),
                        "layout": spec["layout"],
                        "dtype_a": spec["dtype_a"],
                        "dtype_b": spec["dtype_b"],
                        "dtype_c": spec["dtype_c"],
                        "dtype_d": spec["dtype_d"],
                        "dtype_acc": spec["dtype_acc"],
                        "tile_m": tile_m,
                        "tile_n": tile_n,
                        "tile_k": tile_k,
                        "sg_m": sg_m,
                        "sg_n": sg_n,
                        "stages": stage,
                        "split_k": 1,
                        "runner": "mixed_dtype_codegen",
                        "benchmark_target": f"{spec['benchmark_target_prefix']}_{tile_m}x{tile_n}x{tile_k}_sg{sg_m}x{sg_n}_st{stage}",
                        "source": "weight_only_mixed_dtype_codegen_catalog",
                        "instantiation_level": 20,
                        "quant_mode": "weight_only_int8",
                        "scale_mode": spec["scale_mode"],
                        "example_family": "02_bmg_gemm_mixed_dtype_codegen",
                        "runtime_defaults": dict(DEFAULT_WEIGHT_ONLY_RUNTIME_DEFAULTS[dtype_family]),
                    }
                )
    return entries


def generated_weight_only_mixed_dtype_kernel_catalog(
    constraints: dict | None = None,
    *,
    source_template_space: dict | None = None,
):
    source_template_space = source_template_space or observed_bmg_template_space()
    by_family = {"bf16_s8": [], "f16_s8": []}
    for seed in weight_only_mixed_dtype_candidates(
        constraints=constraints,
        source_template_space=source_template_space,
        dtype_families=tuple(by_family.keys()),
    ):
        dtype_family = "bf16_s8" if seed["dtype_a"] == "bf16" else "f16_s8"
        by_family[dtype_family].append(kernel_catalog_entry(dtype_family, seed))
    return {
        "schema_version": SCHEMA_VERSION,
        "catalog_version": "weight-only-mixed-dtype-codegen-v1",
        "instantiation_levels": {
            "20": "autogenerated mixed-dtype weight-only codegen candidates",
        },
        "generator_arch": "bmg",
        "generator_instantiation_level": 20,
        "catalog_source": "weight_only_codegen",
        "search_runtime_schema": SEARCH_RUNTIME_SCHEMA,
        "source_template_space": source_template_space,
        "kernels": [seed for family in ("bf16_s8", "f16_s8") for seed in by_family[family]],
    }


def _sg_layout_literal(sg_m: int, sg_n: int) -> str:
    return f"Layout<Shape<_{sg_m}, _{sg_n}, _1>, Stride<_{sg_n}, _1, _0>>"


def _rewrite_template_source(template_text: str, candidate: dict) -> str:
    tile_literal = f"using TileShape = Shape<_{candidate['tile_m']}, _{candidate['tile_n']}, _{candidate['tile_k']}>;"
    text = re.sub(
        r"using TileShape = Shape<_\d+,\s*_\d+,\s*_\d+>;",
        tile_literal,
        template_text,
    )
    text = re.sub(
        r"Layout<Shape<_\d+,\s*_\d+,\s*_1>,\s*Stride<_\d+,\s*_1,\s*_0>>",
        _sg_layout_literal(candidate["sg_m"], candidate["sg_n"]),
        text,
    )
    text = re.sub(
        r"constexpr int PipelineStages = \d+;",
        f"constexpr int PipelineStages = {candidate['stages']};",
        text,
    )
    return (
        "// Autogenerated by tools/intel_gemm_profiler/mixed_dtype_codegen.py\n"
        f"// kernel_id={candidate.get('kernel_id', candidate['kernel_name'])}\n"
        f"// benchmark_target={candidate['benchmark_target']}\n"
        f"// tile={candidate['tile_m']}x{candidate['tile_n']}x{candidate['tile_k']} sg={candidate['sg_m']}x{candidate['sg_n']} stages={candidate['stages']}\n\n"
        + text
    )


def _template_path_for_candidate(candidate: dict) -> Path:
    return _weight_only_family_spec("bf16_s8" if candidate["dtype_a"] == "bf16" else "f16_s8")["template_path"]


def emit_weight_only_mixed_dtype_project(
    output_dir,
    candidates,
    *,
    cutlass_source_dir=DEFAULT_SOURCE_ROOT,
):
    output_dir = ensure_dir(Path(output_dir))
    src_dir = ensure_dir(output_dir / "src")
    bin_dir = ensure_dir(output_dir / "bin")
    generated_sources = []
    generated_executables = {}
    for candidate in candidates:
        if candidate.get("runner") != "mixed_dtype_codegen":
            continue
        kernel_id = candidate.get("kernel_id", candidate["kernel_name"])
        template_path = _template_path_for_candidate(candidate)
        rendered = _rewrite_template_source(template_path.read_text(encoding="utf-8"), candidate)
        source_path = src_dir / f"{candidate['benchmark_target']}.cpp"
        source_path.write_text(rendered, encoding="utf-8")
        generated_sources.append(
            {
                "kernel_id": kernel_id,
                "benchmark_target": candidate["benchmark_target"],
                "path": str(source_path),
            }
        )
        generated_executables[kernel_id] = str(bin_dir / candidate["benchmark_target"])

    cmake_lines = [
        "cmake_minimum_required(VERSION 3.22 FATAL_ERROR)",
        "project(weight_only_mixed_dtype_codegen LANGUAGES CXX)",
        'set(CUTLASS_ENABLE_SYCL ON CACHE BOOL "" FORCE)',
        'set(SYCL_INTEL_TARGET ON CACHE BOOL "" FORCE)',
        'set(CUTLASS_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)',
        'set(CUTLASS_ENABLE_BENCHMARKS OFF CACHE BOOL "" FORCE)',
        'set(CUTLASS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)',
        f'include("{(Path(cutlass_source_dir) / "cmake" / "FindDPCPP.cmake").resolve().as_posix()}")',
        f'add_subdirectory("{Path(cutlass_source_dir).resolve().as_posix()}" cutlass)',
        "if(TARGET MKL::MKL)",
        '  set(CUTLASS_USING_SYSTEM_ONEMKL TRUE CACHE BOOL "" FORCE)',
        "endif()",
        "add_custom_target(weight_only_mixed_dtype_codegen_all)",
        "",
        "function(weight_only_codegen_add_executable NAME SOURCE_FILE)",
        "  cutlass_add_executable(${NAME} ${SOURCE_FILE} BATCH_SOURCES OFF)",
        "  target_link_libraries(${NAME} PRIVATE CUTLASS cutlass_tools_util_includes)",
        "  target_compile_definitions(${NAME} PRIVATE CUTLASS_ENABLE_SYCL SYCL_INTEL_TARGET)",
        f'  target_include_directories(${{NAME}} PRIVATE "{(Path(cutlass_source_dir) / "examples" / "common").resolve().as_posix()}" "${{CMAKE_BINARY_DIR}}/cutlass/include")',
        '  set_target_properties(${NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")',
        "  add_onemkl_to_target(TARGET ${NAME})",
        "  add_sycl_to_target(TARGET ${NAME})",
        '  if(NOT DPCPP_SYCL_TARGET STREQUAL "spir64")',
        "    target_link_options(${NAME} PRIVATE -Xs \"-options \\\"-igc_opts 'allowDecompose2DBlockFuncs=0'\\\"\")",
        "  endif()",
        "  add_dependencies(weight_only_mixed_dtype_codegen_all ${NAME})",
        "endfunction()",
        "",
    ]
    for source in generated_sources:
        cmake_lines.append(
            f'weight_only_codegen_add_executable({source["benchmark_target"]} "{Path(source["path"]).resolve().as_posix()}")'
        )
    cmake_path = output_dir / "CMakeLists.txt"
    cmake_path.write_text("\n".join(cmake_lines) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "project_dir": str(output_dir),
        "cmake_lists": str(cmake_path),
        "generated_source_count": len(generated_sources),
        "generated_sources": generated_sources,
        "generated_executables": generated_executables,
    }
    manifest_path = output_dir / "generated_project_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest
