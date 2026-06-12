#!/usr/bin/env python3
"""Artifact writers for exact-shape search reports."""

from __future__ import annotations

import json
from pathlib import Path
import shlex
import subprocess
import sys

TOOLS_DIR = Path(__file__).resolve().parent
GEN_MAIN_SCRIPT = TOOLS_DIR / "gen_main.py"
GEN_MINI_HPP_SCRIPT = TOOLS_DIR / "gen_mini_hpp.py"


def _safe_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def strip_internal_fields(row: dict) -> dict:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def format_numeric_cli_arg(value: int | float) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:g}"


def row_runtime_arguments(row: dict) -> dict[str, int | float]:
    batch_count = _safe_int(row.get("batch_count"))
    alpha = _safe_float(row.get("alpha"))
    beta = _safe_float(row.get("beta"))
    split_k = _safe_int(row.get("split_k"))
    streamk_mode = row.get("streamk_mode", "")

    return {
        "m": _safe_int(row.get("m")) or 0,
        "n": _safe_int(row.get("n")) or 0,
        "k": _safe_int(row.get("k")) or 0,
        "l": batch_count if batch_count is not None else 1,
        "alpha": alpha if alpha is not None else 1.0,
        "beta": beta if beta is not None else 0.0,
        "split_k_slices": split_k if split_k is not None else (1 if streamk_mode == "splitk" else 0),
    }


def make_benchmark_config_line(row: dict, bm_name: str) -> str:
    runtime = row_runtime_arguments(row)
    parts = [
        row.get("kernel", ""),
        f"--bm_name={bm_name}",
        f"--m={runtime['m']}",
        f"--n={runtime['n']}",
        f"--k={runtime['k']}",
        f"--l={runtime['l']}",
        f"--alpha={format_numeric_cli_arg(runtime['alpha'])}",
        f"--beta={format_numeric_cli_arg(runtime['beta'])}",
    ]
    if runtime["split_k_slices"] > 0:
        parts.append(f"--split_k_slices={runtime['split_k_slices']}")
    return " ".join(parts)


def write_repro_filter(path: Path, rows: list[dict]) -> None:
    kernels = [row.get("kernel", "") for row in rows if row.get("kernel")]
    anchored = [f"^{kernel}$" for kernel in kernels]
    path.write_text("\n".join(anchored) + ("\n" if anchored else ""), encoding="utf-8")


def write_repro_config(path: Path, rows: list[dict], shape_tag: str, label: str) -> None:
    lines = [
        make_benchmark_config_line(row, f"repro_{shape_tag}_{label}_{index:02d}")
        for index, row in enumerate(rows)
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def infer_search_limitations(rows: list[dict], run_meta: dict[str, str]) -> list[dict]:
    limitations = []
    has_benchmark_splitk = any(
        row.get("runner") == "benchmark" and row.get("streamk_mode") == "splitk"
        for row in rows
    )
    if has_benchmark_splitk:
        limitations.append(
            {
                "scope": "benchmark-backed SplitK",
                "constraint": "runtime split_k_slices <= 1",
                "reason": "Current Xe benchmark-backed SplitK path rejects split_k_slices>1 to avoid the known hang-prone runtime sweep path.",
            }
        )

    stride_policy = run_meta.get("benchmark_stride_policy")
    if stride_policy:
        limitations.append(
            {
                "scope": "benchmark configuration",
                "constraint": f"stride_policy={stride_policy}",
                "reason": "Exact-shape results are only comparable when replayed with the same benchmark input/stride policy.",
            }
        )

    input_mode = run_meta.get("benchmark_input_mode")
    if input_mode:
        limitations.append(
            {
                "scope": "benchmark configuration",
                "constraint": f"input_mode={input_mode}",
                "reason": "Buffer rotation vs fixed-address replay materially changes cache locality and measured TFLOPS.",
            }
        )

    return limitations


def build_repro_manifest(
    rows: list[dict],
    shape_tag: str,
    label: str,
    run_meta: dict[str, str],
    benchmark_config: dict | list,
    synced_sources_path: Path,
) -> dict:
    return {
        "shape_tag": shape_tag,
        "label": label,
        "kernel_count": len(rows),
        "kernels": [strip_internal_fields(row) for row in rows],
        "run_meta_subset": {
            key: run_meta.get(key, "")
            for key in [
                "repo_root",
                "git_head",
                "kernel_catalog_source",
                "benchmark_input_mode",
                "benchmark_stride_policy",
                "benchmark_input_pool_target_bytes",
                "benchmark_warmup_iters",
                "benchmark_measure_iters",
                "benchmark_fixed_vram_input",
                "perf_env_ONEAPI_DEVICE_SELECTOR",
                "perf_env_SYCL_PROGRAM_COMPILE_OPTIONS",
                "perf_env_IGC_VectorAliasBBThreshold",
                "perf_env_IGC_ExtraOCLOptions",
            ]
        },
        "benchmark_config": benchmark_config,
        "synced_sources_path": str(synced_sources_path),
    }


def write_repro_script(
    path: Path,
    *,
    label: str,
    shape_tag: str,
    rows: list[dict],
    run_meta: dict[str, str],
    benchmark_config: dict | list,
    filter_path: Path,
    config_path: Path,
    manifest_path: Path,
) -> None:
    repo_root = run_meta.get("repo_root", "")
    oneapi_device_selector = run_meta.get("perf_env_ONEAPI_DEVICE_SELECTOR", "")
    sycl_compile_options = run_meta.get("perf_env_SYCL_PROGRAM_COMPILE_OPTIONS", "")
    igc_vector_alias = run_meta.get("perf_env_IGC_VectorAliasBBThreshold", "")
    igc_extra_options = run_meta.get("perf_env_IGC_ExtraOCLOptions", "")
    fixed_vram_input = False
    phase_timing_enabled = False
    if isinstance(benchmark_config, dict):
        fixed_vram_input = bool(benchmark_config.get("fixed_vram_input", False))
        phase_timing_enabled = bool(benchmark_config.get("phase_timing_enabled", False))
    default_gpu = ""
    if rows:
        gpu_value = rows[0].get("gpu", "")
        if gpu_value != "":
            default_gpu = str(gpu_value)
    build_dir_suffix = f"{shape_tag}_{label}"
    icpx_path = "/opt/intel/oneapi/compiler/2025.3/bin/icpx"
    script = f"""#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
RUN_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="${{REPO_ROOT:-{repo_root}}}"
BUILD_DIR="${{BUILD_DIR:-$REPO_ROOT/build_exact_shape_repro_{build_dir_suffix}}}"
SHARED_DEPS_BUILD="${{SHARED_DEPS_BUILD:-$RUN_DIR/workers/gpu{default_gpu or 0}/build}}"
FILTER_FILE="$SCRIPT_DIR/{filter_path.name}"
CONFIG_FILE="$SCRIPT_DIR/{config_path.name}"
MANIFEST_FILE="$SCRIPT_DIR/{manifest_path.name}"

[ -n "$REPO_ROOT" ] || {{ echo "REPO_ROOT is not set" >&2; exit 1; }}
[ -f "$REPO_ROOT/benchmarks/gemm/CMakeLists.txt" ] || {{ echo "REPO_ROOT does not look like a sycl-tla repo: $REPO_ROOT" >&2; exit 1; }}

export PATH="/opt/intel/oneapi/compiler/2025.3/bin:$PATH"
export ONEAPI_DEVICE_SELECTOR={shlex.quote(oneapi_device_selector)}
export SYCL_PROGRAM_COMPILE_OPTIONS={shlex.quote(sycl_compile_options)}
export IGC_VectorAliasBBThreshold={shlex.quote(igc_vector_alias)}
export IGC_ExtraOCLOptions={shlex.quote(igc_extra_options)}
"""
    if fixed_vram_input:
        script += 'export CUTLASS_BENCHMARK_FIXED_VRAM_INPUT=1\n'
    else:
        script += 'unset CUTLASS_BENCHMARK_FIXED_VRAM_INPUT 2>/dev/null || true\n'
    if phase_timing_enabled:
        script += 'export CUTLASS_BENCHMARK_PHASE_TIMING=1\n'
    else:
        script += 'unset CUTLASS_BENCHMARK_PHASE_TIMING 2>/dev/null || true\n'
    if default_gpu:
        script += f'export ZE_AFFINITY_MASK="${{ZE_AFFINITY_MASK:-{default_gpu}}}"\n'
    script += f"""

echo "Using manifest: $MANIFEST_FILE"
cmake -S "$REPO_ROOT" -B "$BUILD_DIR" \\
  -DCMAKE_BUILD_TYPE=Release \\
  -DCMAKE_CXX_COMPILER="${{CMAKE_CXX_COMPILER:-{icpx_path}}}" \\
  -DDPCPP_SYCL_TARGET=intel_gpu_bmg_g31 \\
  -DDPCPP_HOST_COMPILER=g++-13 \\
  -DCUTLASS_ENABLE_SYCL=ON \\
  -DCUTLASS_ENABLE_TESTS=OFF \\
  -DCUTLASS_NVCC_ARCHS= \\
  -DCUTLASS_BENCHMARK_EXPANDED_BMG_STREAMK=ON \\
  -DKERNEL_FILTER_FILE="$FILTER_FILE" \\
  -DCUTLASS_BENCHMARK_EXHAUSTIVE_GEMM=ON \\
  -DCUTLASS_BENCHMARK_EXHAUSTIVE_STREAMK=ON \\
  -DGOOGLETEST_DIR="$REPO_ROOT/_deps/googletest-src" \\
  -DGOOGLEBENCHMARK_DIR="$REPO_ROOT/_deps/googlebenchmark-src"
mkdir -p "$BUILD_DIR/_deps"
if [ -d "$SHARED_DEPS_BUILD/_deps/googlebenchmark-build" ]; then
  ln -sfn "$SHARED_DEPS_BUILD/_deps/googlebenchmark-build" "$BUILD_DIR/_deps/googlebenchmark-build"
fi
if [ -d "$SHARED_DEPS_BUILD/_deps/googletest-build" ]; then
  ln -sfn "$SHARED_DEPS_BUILD/_deps/googletest-build" "$BUILD_DIR/_deps/googletest-build"
fi
cmake --build "$BUILD_DIR" --target cutlass_benchmarks_gemm_sycl -j "${{BUILD_JOBS:-1}}"
"$BUILD_DIR/benchmarks/gemm/cutlass_benchmarks_gemm_sycl" --config_file="$CONFIG_FILE"
"""
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)


def write_kernel_manifest(path: Path, rows: list[dict]) -> None:
    kernels = [row.get("kernel", "") for row in rows if row.get("kernel")]
    path.write_text("\n".join(kernels) + ("\n" if kernels else ""), encoding="utf-8")


def generate_bundle_sources(bundle_dir: Path, kernel_manifest_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(GEN_MINI_HPP_SCRIPT),
            "--manifest",
            str(kernel_manifest_path),
            "--output",
            str(bundle_dir / "benchmarks_sycl.hpp"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(GEN_MAIN_SCRIPT),
            str(kernel_manifest_path),
            str(bundle_dir / "main.cpp"),
        ],
        check=True,
    )


def write_bundle_build_script(
    path: Path,
    *,
    repo_root: str,
    git_head: str,
    oneapi_device_selector: str,
    sycl_compile_options: str,
    igc_vector_alias: str,
    igc_extra_options: str,
) -> None:
    icpx_path = "/opt/intel/oneapi/compiler/2025.3/bin/icpx"
    script = f"""#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
BUNDLE_DIR="$SCRIPT_DIR"
REPO_ROOT="${{REPO_ROOT:-{repo_root}}}"
BUILD_ROOT="${{BUILD_ROOT:-$BUNDLE_DIR/build}}"
OVERLAY_REPO="$BUILD_ROOT/export_repo"
BUILD_DIR="$BUILD_ROOT/cmake-build"
SHARED_DEPS_BUILD="${{SHARED_DEPS_BUILD:-}}"
EXPECTED_GIT_HEAD={shlex.quote(git_head)}

[ -n "$REPO_ROOT" ] || {{ echo "REPO_ROOT is not set" >&2; exit 1; }}
[ -f "$REPO_ROOT/benchmarks/gemm/CMakeLists.txt" ] || {{ echo "REPO_ROOT does not look like a sycl-tla repo: $REPO_ROOT" >&2; exit 1; }}

if command -v git >/dev/null 2>&1; then
  current_head="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || true)"
  if [ -n "$EXPECTED_GIT_HEAD" ] && [ -n "$current_head" ] && [ "$current_head" != "$EXPECTED_GIT_HEAD" ]; then
    echo "WARNING: REPO_ROOT HEAD $current_head does not match exported run git_head $EXPECTED_GIT_HEAD" >&2
  fi
fi

export PATH="/opt/intel/oneapi/compiler/2025.3/bin:$PATH"
export ONEAPI_DEVICE_SELECTOR={shlex.quote(oneapi_device_selector)}
export SYCL_PROGRAM_COMPILE_OPTIONS={shlex.quote(sycl_compile_options)}
export IGC_VectorAliasBBThreshold={shlex.quote(igc_vector_alias)}
export IGC_ExtraOCLOptions={shlex.quote(igc_extra_options)}

rm -rf "$OVERLAY_REPO" "$BUILD_DIR"
mkdir -p "$BUILD_ROOT"
cp -a "$REPO_ROOT/." "$OVERLAY_REPO/"
cp "$BUNDLE_DIR/benchmarks_sycl.hpp" "$OVERLAY_REPO/benchmarks/gemm/benchmarks_sycl.hpp"
cp "$BUNDLE_DIR/main.cpp" "$OVERLAY_REPO/benchmarks/gemm/main.cpp"

cmake -S "$OVERLAY_REPO" -B "$BUILD_DIR" \\
  -DCMAKE_BUILD_TYPE=Release \\
  -DCMAKE_CXX_COMPILER="${{CMAKE_CXX_COMPILER:-{icpx_path}}}" \\
  -DDPCPP_SYCL_TARGET=intel_gpu_bmg_g31 \\
  -DDPCPP_HOST_COMPILER=g++-13 \\
  -DCUTLASS_ENABLE_SYCL=ON \\
  -DCUTLASS_ENABLE_TESTS=OFF \\
  -DCUTLASS_NVCC_ARCHS= \\
  -DCUTLASS_BENCHMARK_EXPANDED_BMG_STREAMK=ON \\
  -DCUTLASS_BENCHMARK_EXHAUSTIVE_GEMM=ON \\
  -DCUTLASS_BENCHMARK_EXHAUSTIVE_STREAMK=ON \\
  -DGOOGLETEST_DIR="$OVERLAY_REPO/_deps/googletest-src" \\
  -DGOOGLEBENCHMARK_DIR="$OVERLAY_REPO/_deps/googlebenchmark-src"

mkdir -p "$BUILD_DIR/_deps"
if [ -n "$SHARED_DEPS_BUILD" ] && [ -d "$SHARED_DEPS_BUILD/_deps/googlebenchmark-build" ]; then
  ln -sfn "$SHARED_DEPS_BUILD/_deps/googlebenchmark-build" "$BUILD_DIR/_deps/googlebenchmark-build"
fi
if [ -n "$SHARED_DEPS_BUILD" ] && [ -d "$SHARED_DEPS_BUILD/_deps/googletest-build" ]; then
  ln -sfn "$SHARED_DEPS_BUILD/_deps/googletest-build" "$BUILD_DIR/_deps/googletest-build"
fi

cmake --build "$BUILD_DIR" --target cutlass_benchmarks_gemm_sycl -j "${{BUILD_JOBS:-1}}"
"""
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)


def write_bundle_run_script(path: Path) -> None:
    script = """#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_ROOT="${BUILD_ROOT:-$SCRIPT_DIR/build}"
BUILD_DIR="${BUILD_DIR:-$BUILD_ROOT/cmake-build}"
"$BUILD_DIR/benchmarks/gemm/cutlass_benchmarks_gemm_sycl" --config_file="$SCRIPT_DIR/repro.cfg"
"""
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)


def write_bundle_makefile(path: Path) -> None:
    text = """BUNDLE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
BUILD_ROOT ?= $(BUNDLE_DIR)/build
REPO_ROOT ?=
SHARED_DEPS_BUILD ?=
BUILD_JOBS ?= 1

.PHONY: all build run clean

all: build

build:
\tREPO_ROOT="$(REPO_ROOT)" BUILD_ROOT="$(BUILD_ROOT)" SHARED_DEPS_BUILD="$(SHARED_DEPS_BUILD)" BUILD_JOBS="$(BUILD_JOBS)" bash "$(BUNDLE_DIR)/build.sh"

run: build
\tBUILD_ROOT="$(BUILD_ROOT)" bash "$(BUNDLE_DIR)/run.sh"

clean:
\trm -rf "$(BUILD_ROOT)"
"""
    path.write_text(text, encoding="utf-8")


def write_bundle_readme(
    path: Path,
    *,
    shape_tag: str,
    label: str,
    rows: list[dict],
    repo_root: str,
) -> None:
    kernels = "\n".join(f"- `{row.get('kernel', '')}`" for row in rows)
    text = f"""# Exported exact-shape kernel bundle

This bundle was exported from an exact-shape report for shape `{shape_tag}` and selection `{label}`.

Selected kernels:

{kernels}

Files:

- `kernel_manifest.txt` — plain kernel ids, one per line
- `kernel_filter.txt` — anchored regex filter matching the selected kernels
- `repro.cfg` — benchmark config entries for the selected kernels
- `repro.json` — run metadata subset and benchmark configuration
- `metadata.json` — exported kernel metadata and source snapshot references
- `kernel_config.json` — shape/runtime configuration for rebuild and replay
- `benchmarks_sycl.hpp` — generated benchmark kernel header for this selection
- `main.cpp` — generated benchmark entrypoint for this selection
- `build.sh` — rebuild the generated source against a sycl-tla checkout
- `run.sh` — execute the rebuilt binary with `repro.cfg`
- `Makefile` — convenience targets (`make build`, `make run`)

Notes:

- This bundle preserves the generated kernel source used for the selected kernels.
- The bundle rebuild currently expects access to a compatible `sycl-tla` checkout via `REPO_ROOT`.
- Default `REPO_ROOT` from the originating run: `{repo_root}`
"""
    path.write_text(text, encoding="utf-8")


def _top_ranked_rows(ranked_desc: list[dict], count: int) -> list[dict]:
    return [strip_internal_fields(row) for row in ranked_desc[:count]]


def write_export_bundle(
    output_dir: Path,
    *,
    shape_tag: str,
    label: str,
    rows: list[dict],
    run_meta: dict[str, str],
    benchmark_config: dict | list,
    synced_sources_path: Path,
) -> dict:
    bundle_dir = output_dir / f"{label}_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    filter_path = bundle_dir / "kernel_filter.txt"
    config_path = bundle_dir / "repro.cfg"
    repro_manifest_path = bundle_dir / "repro.json"
    kernel_manifest_path = bundle_dir / "kernel_manifest.txt"
    metadata_path = bundle_dir / "metadata.json"
    kernel_config_path = bundle_dir / "kernel_config.json"
    build_script_path = bundle_dir / "build.sh"
    run_script_path = bundle_dir / "run.sh"
    makefile_path = bundle_dir / "Makefile"
    readme_path = bundle_dir / "README.md"

    write_repro_filter(filter_path, rows)
    write_repro_config(config_path, rows, shape_tag, label)
    repro_manifest = build_repro_manifest(
        rows,
        shape_tag,
        label,
        run_meta,
        benchmark_config,
        synced_sources_path,
    )
    repro_manifest_path.write_text(json.dumps(repro_manifest, indent=2) + "\n", encoding="utf-8")

    write_kernel_manifest(kernel_manifest_path, rows)
    generate_bundle_sources(bundle_dir, kernel_manifest_path)

    metadata = {
        "shape_tag": shape_tag,
        "label": label,
        "kernel_count": len(rows),
        "kernels": [strip_internal_fields(row) for row in rows],
        "run_meta_subset": repro_manifest["run_meta_subset"],
        "synced_sources_path": str(synced_sources_path),
        "generated_sources": {
            "benchmarks_sycl_hpp": str(bundle_dir / "benchmarks_sycl.hpp"),
            "main_cpp": str(bundle_dir / "main.cpp"),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    kernel_config = {
        "shape_tag": shape_tag,
        "label": label,
        "benchmark_config": benchmark_config,
        "runtime_rows": [
            {
                "kernel": row.get("kernel", ""),
                "runtime_args": row_runtime_arguments(row),
            }
            for row in rows
        ],
    }
    kernel_config_path.write_text(json.dumps(kernel_config, indent=2) + "\n", encoding="utf-8")

    write_bundle_build_script(
        build_script_path,
        repo_root=run_meta.get("repo_root", ""),
        git_head=run_meta.get("git_head", ""),
        oneapi_device_selector=run_meta.get("perf_env_ONEAPI_DEVICE_SELECTOR", ""),
        sycl_compile_options=run_meta.get("perf_env_SYCL_PROGRAM_COMPILE_OPTIONS", ""),
        igc_vector_alias=run_meta.get("perf_env_IGC_VectorAliasBBThreshold", ""),
        igc_extra_options=run_meta.get("perf_env_IGC_ExtraOCLOptions", ""),
    )
    write_bundle_run_script(run_script_path)
    write_bundle_makefile(makefile_path)
    write_bundle_readme(
        readme_path,
        shape_tag=shape_tag,
        label=label,
        rows=rows,
        repo_root=run_meta.get("repo_root", ""),
    )

    return {
        "dir": str(bundle_dir),
        "filter": str(filter_path),
        "config": str(config_path),
        "manifest": str(repro_manifest_path),
        "kernel_manifest": str(kernel_manifest_path),
        "metadata": str(metadata_path),
        "kernel_config": str(kernel_config_path),
        "generated_header": str(bundle_dir / "benchmarks_sycl.hpp"),
        "generated_main": str(bundle_dir / "main.cpp"),
        "build_script": str(build_script_path),
        "run_script": str(run_script_path),
        "makefile": str(makefile_path),
        "readme": str(readme_path),
    }


def write_export_bundles(
    output_dir: Path,
    shape_tag: str,
    ranked_desc: list[dict],
    run_meta: dict[str, str],
    benchmark_config: dict | list,
    synced_sources_path: Path,
) -> dict:
    bundles = {}
    for label, rows in {
        "top1": _top_ranked_rows(ranked_desc, 1),
        "top5": _top_ranked_rows(ranked_desc, 5),
    }.items():
        bundles[label] = write_export_bundle(
            output_dir,
            shape_tag=shape_tag,
            label=label,
            rows=rows,
            run_meta=run_meta,
            benchmark_config=benchmark_config,
            synced_sources_path=synced_sources_path,
        )
    return bundles


def write_repro_artifacts(
    output_dir: Path,
    shape_tag: str,
    ranked_desc: list[dict],
    run_meta: dict[str, str],
    benchmark_config: dict | list,
    synced_sources_path: Path,
) -> dict:
    artifacts = {}
    for label, rows in {
        "top1": _top_ranked_rows(ranked_desc, 1),
        "top5": _top_ranked_rows(ranked_desc, 5),
    }.items():
        filter_path = output_dir / f"{label}_filter.txt"
        config_path = output_dir / f"{label}_repro.cfg"
        manifest_path = output_dir / f"{label}_repro.json"
        script_path = output_dir / f"{label}_repro.sh"
        write_repro_filter(filter_path, rows)
        write_repro_config(config_path, rows, shape_tag, label)
        manifest = build_repro_manifest(
            rows,
            shape_tag,
            label,
            run_meta,
            benchmark_config,
            synced_sources_path,
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        write_repro_script(
            script_path,
            label=label,
            shape_tag=shape_tag,
            rows=rows,
            run_meta=run_meta,
            benchmark_config=benchmark_config,
            filter_path=filter_path,
            config_path=config_path,
            manifest_path=manifest_path,
        )
        artifacts[label] = {
            "filter": str(filter_path),
            "config": str(config_path),
            "manifest": str(manifest_path),
            "script": str(script_path),
        }
    return artifacts
