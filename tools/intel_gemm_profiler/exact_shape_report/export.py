#!/usr/bin/env python3
"""Export bundle writers for exact-shape search reports."""

from __future__ import annotations

import json
from pathlib import Path
import shlex
import subprocess
import sys

from .repro import (
    _top_ranked_rows,
    build_repro_manifest,
    row_runtime_arguments,
    strip_internal_fields,
    write_repro_config,
    write_repro_filter,
)


TOOLS_DIR = Path(__file__).resolve().parents[1]
GEN_MAIN_SCRIPT = TOOLS_DIR / "gen_main.py"
GEN_MINI_HPP_SCRIPT = TOOLS_DIR / "gen_mini_hpp.py"


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


__all__ = [
    "write_export_bundle",
    "write_export_bundles",
]
