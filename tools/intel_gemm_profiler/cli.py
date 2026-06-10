#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import argparse
import json
import sys

from .catalog import SEED_KERNELS
from .dispatch import lookup_dispatch_entry
from .workflow import (
    export_product_bundle_manifest,
    validate_product_bundle_manifest,
    workflow,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Intel GEMM profiler MVP runner for non-legacy registered RCR kernels.")
    parser.add_argument("--workspace", default="", help="Workspace directory for generated files and reports.")
    parser.add_argument("--benchmark-exe", default="./build/benchmarks/gemm/cutlass_benchmarks_gemm_sycl", help="Benchmark executable to run.")
    parser.add_argument("--streamk-example-exe", default="./build/examples/03_bmg_gemm_streamk/03_bmg_gemm_streamk", help="StreamK example executable used for split-k candidates.")
    parser.add_argument("--cwd", default=None, help="Working directory for the benchmark subprocess.")
    parser.add_argument("--shell-init", default="", help="Optional shell snippet executed before the benchmark command, e.g. 'source /home/intel/.bashrc && source /opt/intel/oneapi/setvars.sh'.")
    parser.add_argument("--dtype", choices=sorted(SEED_KERNELS.keys()), default="bf16", help="Default dtype preset.")
    parser.add_argument("--probe-mode", choices=["auto", "off", "static", "run"], default="auto", help="Phase A constraint probe mode. 'auto' runs representative probes unless --skip-run is set.")
    parser.add_argument("--shapes-json", default="", help="Optional path to gemm_target_shapes.json.")
    parser.add_argument("--reference-json", default="", help="Optional path to reference/oracle JSON for dataset comparison.")
    parser.add_argument("--ali-workbook", default="", help="Optional Ali GEMM performance workbook. When set, workflow derives gemm_target_shapes.json and reference comparison input from the workbook.")
    parser.add_argument("--max-shapes", type=int, default=0, help="Limit target shapes to the first N entries after loading --shapes-json, --ali-workbook, or the default shape set. 0 disables the limit.")
    parser.add_argument("--constraints-json", default="", help="Optional path to safe_search_constraints.json.")
    parser.add_argument("--compiler-profiles-json", default="", help="Optional path to compiler_profiles.json.")
    parser.add_argument("--compile-variant", default="", help="Override the selected compile env variant (e.g., perf_default, perf_perfmodel, debug_with_lines). Uses the variant from build_config_bmg_perf.json when empty.")
    parser.add_argument("--runtime-variant", default="", help="Override the selected runtime env variant (e.g., default, ze_affinity_7). Uses the variant from runtime_config_bmg_perf.json when empty.")
    parser.add_argument("--update-compile-variant", default="", help="Persist a new compile variant selection to build_config_bmg_perf.json. Use --list-compile-variants to see available options.")
    parser.add_argument("--update-runtime-variant", default="", help="Persist a new runtime variant selection to runtime_config_bmg_perf.json. Use --list-runtime-variants to see available options.")
    parser.add_argument("--list-compile-variants", action="store_true", help="List available compile env variants and exit.")
    parser.add_argument("--list-runtime-variants", action="store_true", help="List available runtime env variants and exit.")
    parser.add_argument("--kernel-catalog-source", choices=["persisted", "generator", "expanded_streamk", "expanded_bmg", "layered_bmg", "layered_bmg_scheduler_expanded"], default="persisted", help="Catalog source for Phase B candidates. 'expanded_bmg' enables the opt-in BMG Gemm/StreamK/DataParallel/SplitK tile expansion and requires rebuilding the benchmark with the generated build plan. 'layered_bmg' adds regular GEMM legal tile/subgroup/stage enumeration on top of expanded_bmg while preserving the legacy fixed-8x4 scheduler path. 'layered_bmg_scheduler_expanded' keeps the same regular GEMM space but widens BF16 scheduler search across legal subgroup/stage combinations. 'expanded_streamk' is kept as a compatibility alias.")
    parser.add_argument("--kernel-catalog-path", default="", help="Optional path to a persisted kernel catalog JSON. Used when --kernel-catalog-source=persisted.")
    parser.add_argument("--search-strategy", choices=["manual", "baseline", "expanded_bmg", "layered_exhaustive", "bruteforce_scheduler"], default="manual", help="High-level search preset. 'manual' preserves explicit catalog/build flags. 'baseline' keeps the legacy persisted baseline search. 'expanded_bmg' keeps the legacy expanded BMG search. 'layered_exhaustive' keeps the legacy layered exhaustive search. 'bruteforce_scheduler' widens BF16 scheduler search across legal subgroup/stage combinations and routes Phase B through scheduler-focused preflight batches.")
    parser.add_argument("--bruteforce-scheduler-search", action="store_true", help="Enable the no-pruning scheduler search profile: force layered_bmg_scheduler_expanded, disable prefiltering, emit per-kernel preflight build batches, and route Phase B benchmark runs through the preflight executables.")
    parser.add_argument("--prefilter", choices=["none", "light", "medium", "aggressive"], default="none", help="Candidate prefilter strategy: skip configs unlikely to perform well for the target shapes. 'light' removes physically incompatible configs. 'medium' adds ILP-based pruning. 'aggressive' is for fastest search with some risk of missing optimal configs.")
    parser.add_argument("--compiled-kernel-list", default="", help="Optional newline-delimited compiled kernel list or regex filter file. When set, Phase B only runs benchmark candidates present in this list.")
    parser.add_argument("--cmake-source-dir", default="", help="Optional source directory used in the generated candidate benchmark CMake build plan. Defaults to --cwd or the current directory.")
    parser.add_argument("--benchmark-build-dir", default="", help="Optional build directory used in the generated candidate benchmark CMake build plan. Defaults to <workspace>/build/candidate_benchmarks.")
    parser.add_argument("--googlebenchmark-dir", default="", help="Optional local Google Benchmark source directory injected into the generated CMake build plan as GOOGLEBENCHMARK_DIR to avoid FetchContent downloads.")
    parser.add_argument("--googlebenchmark-build-dir", default="", help="Optional prebuilt Google Benchmark build directory injected into the generated CMake build plan as GOOGLEBENCHMARK_BUILD_DIR when isolated workspaces cannot reuse the default build-tree _deps path.")
    parser.add_argument("--cmake-cxx-compiler", default="", help="Optional CMAKE_CXX_COMPILER value injected into the generated candidate benchmark CMake build plan, e.g. 'icpx' for oneAPI SYCL builds.")
    parser.add_argument("--build-candidate-benchmark", action="store_true", help="Execute the generated candidate benchmark CMake configure/build plan before Phase B runs, then use the built benchmark executable for screening and confirmation.")
    parser.add_argument("--candidate-build-batch-size", type=int, default=0, help="Emit additional selected-kernel filter files in batches of N kernels for isolated generated benchmark build preflight/retry. 0 disables batch artifacts.")
    parser.add_argument("--run-candidate-build-preflight", action="store_true", help="Execute per-batch candidate benchmark preflight build plans before the aggregate candidate benchmark build. Requires --candidate-build-batch-size to produce batch plans.")
    parser.add_argument("--use-candidate-build-preflight-benchmarks", action="store_true", help="Route benchmark screening and confirmation entries to per-batch benchmark executables produced by --run-candidate-build-preflight instead of the aggregate benchmark executable.")
    parser.add_argument("--resume-candidate-build-preflight", action="store_true", help="Resume a previous preflight build: skip batches whose log shows successful completion. Uses preflight_progress.json in the reports directory.")
    parser.add_argument("--candidate-build-parallelism", type=int, default=16, help="Number of concurrent batch compilations during preflight. Each batch build uses an auto-sized subset of host CPUs to avoid oversubscribing the machine. 1 runs preflight sequentially.")
    parser.add_argument("--generator-arch", choices=["bmg", "pvc"], default="bmg", help="Intel Xe generator arch used when --kernel-catalog-source=generator.")
    parser.add_argument("--generator-instantiation-level", type=int, default=0, help="Intel Xe generator instantiation level used when --kernel-catalog-source=generator.")
    parser.add_argument("--hw-spec-id", default="", help="Optional hardware reference spec id override, e.g. 'bmg_g21'.")
    parser.add_argument("--skip-run", action="store_true", help="Only emit generated artifacts without invoking the benchmark.")
    parser.add_argument("--dry-run", action="store_true", help="Run a minimal benchmark-backed screening smoke with a tiny shape set and no confirmation.")
    parser.add_argument("--timeout", type=int, default=600, help="Per-subprocess timeout in seconds for benchmark and example runtime execution.")
    parser.add_argument("--build-timeout", type=int, default=0, help="Optional per-subprocess timeout in seconds for candidate benchmark configure/build steps. 0 reuses --timeout.")
    parser.add_argument("--benchmark-entry-chunk-size", type=int, default=0, help="Split screening and confirmation benchmark config execution into chunks of N entries. 0 runs each stage as one subprocess.")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k candidates kept for confirmation.")
    parser.add_argument("--confirm-runs", type=int, default=3, help="Number of confirmation attempts for top-k candidates.")
    parser.add_argument("--close-call-threshold", type=float, default=3.0, help="Gap threshold in percent for close-call labeling.")
    parser.add_argument("--lookup-dispatch-table", default="", help="Lookup mode: path to gemm_dispatch_table.json or optimal_dispatch_table.json. When set, the CLI prints a lookup JSON result instead of running the profiler workflow.")
    parser.add_argument("--validate-product-bundle", default="", help="Validation mode: path to gemm_product_bundle_manifest.json. Prints JSON suitable for release/CI gates and exits nonzero on failure.")
    parser.add_argument("--export-product-bundle", default="", help="Export mode: path to gemm_product_bundle_manifest.json to copy into a standalone product handoff directory.")
    parser.add_argument("--bundle-output-dir", default="", help="Export mode output directory for --export-product-bundle.")
    parser.add_argument("--lookup-layout", default="rcr", help="Lookup mode GEMM layout key, e.g. rcr.")
    parser.add_argument("--lookup-dtype-a", default="bf16", help="Lookup mode A dtype.")
    parser.add_argument("--lookup-dtype-b", default="bf16", help="Lookup mode B dtype.")
    parser.add_argument("--lookup-dtype-c", default="f32", help="Lookup mode C dtype.")
    parser.add_argument("--lookup-dtype-d", default="", help="Lookup mode D/output dtype. Defaults to --lookup-dtype-c.")
    parser.add_argument("--lookup-dtype-acc", default="f32", help="Lookup mode accumulator dtype.")
    parser.add_argument("--lookup-m", type=int, default=0, help="Lookup mode M dimension.")
    parser.add_argument("--lookup-n", type=int, default=0, help="Lookup mode N dimension.")
    parser.add_argument("--lookup-k", type=int, default=0, help="Lookup mode K dimension.")
    parser.add_argument("--lookup-batch-count", type=int, default=1, help="Lookup mode batch/L dimension.")
    parser.add_argument("--fallback-candidate-id", default="", help="Optional fallback candidate id returned when lookup mode misses the exact shape.")
    return parser


def dispatch_lookup_from_args(args):
    missing = [
        flag
        for flag, value in (
            ("--lookup-m", args.lookup_m),
            ("--lookup-n", args.lookup_n),
            ("--lookup-k", args.lookup_k),
            ("--lookup-batch-count", args.lookup_batch_count),
        )
        if value <= 0
    ]
    if missing:
        raise ValueError(f"{', '.join(missing)} must be positive when --lookup-dispatch-table is used.")
    shape = {
        "layout": args.lookup_layout,
        "dtype_a": args.lookup_dtype_a,
        "dtype_b": args.lookup_dtype_b,
        "dtype_c": args.lookup_dtype_c,
        "dtype_d": args.lookup_dtype_d or args.lookup_dtype_c,
        "dtype_acc": args.lookup_dtype_acc,
        "m": args.lookup_m,
        "n": args.lookup_n,
        "k": args.lookup_k,
        "batch_count": args.lookup_batch_count,
    }
    return lookup_dispatch_entry(
        args.lookup_dispatch_table,
        shape,
        fallback_candidate_id=args.fallback_candidate_id,
    )


def main():
    args = build_parser().parse_args()
    selected_modes = sum(
        bool(value)
        for value in (
            args.lookup_dispatch_table,
            args.validate_product_bundle,
            args.export_product_bundle,
        )
    )
    if selected_modes > 1:
        raise ValueError(
            "--lookup-dispatch-table, --validate-product-bundle, and --export-product-bundle are mutually exclusive."
        )
    if args.export_product_bundle:
        if not args.bundle_output_dir:
            raise ValueError("--bundle-output-dir is required with --export-product-bundle.")
        result = export_product_bundle_manifest(args.export_product_bundle, args.bundle_output_dir)
    elif args.validate_product_bundle:
        result = validate_product_bundle_manifest(args.validate_product_bundle)
    elif args.lookup_dispatch_table:
        result = dispatch_lookup_from_args(args)
    else:
        result = workflow(args)
    print(json.dumps(result, indent=2))
    if args.validate_product_bundle and result["status"] != "pass":
        sys.exit(1)
