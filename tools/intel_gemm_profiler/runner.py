#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import os
import platform
import re
import socket
import sys
from pathlib import Path

from .runner_benchmark import (
    parse_benchmark_log,
    row_result_metadata,
    run_benchmark,
    run_entries_with_benchmark,
    timeout_rows,
    with_result_metadata,
)
from .schemas import SCHEMA_VERSION
from .utils import now_iso, resolve_executable, shell_join


def collect_environment_metadata(shell_init, benchmark_exe, streamk_example_exe, cwd=None):
    tracked_env = {}
    for name in ("ONEAPI_DEVICE_SELECTOR", "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS", "ZE_FLAT_DEVICE_HIERARCHY", "SYCL_PROGRAM_COMPILE_OPTIONS", "IGC_ExtraOCLOptions", "IGC_VectorAliasBBThreshold", "IGC_VISAOptions"):
        value = os.environ.get(name)
        if value:
            tracked_env[name] = value
    benchmark_path = resolve_executable(benchmark_exe, cwd=cwd)
    streamk_path = resolve_executable(streamk_example_exe, cwd=cwd)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "hostname": socket.gethostname(),
        "node_id": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "proxy_bootstrap_method": shell_init or "inherited-environment",
        "executables": {
            "benchmark_exe": str(benchmark_path) if benchmark_path else benchmark_exe,
            "benchmark_available": bool(benchmark_path),
            "streamk_example_exe": str(streamk_path) if streamk_path else streamk_example_exe,
            "streamk_example_available": bool(streamk_path),
        },
        "effective_env": tracked_env,
    }


def parse_streamk_example_log(log_path, metadata_by_bm_name, run_id):
    bm_name = next(iter(metadata_by_bm_name))
    metadata = metadata_by_bm_name[bm_name]
    text = Path(log_path).read_text(encoding="utf-8")
    status = "pass" if "Disposition: Passed" in text or "Disposition is skipped." in text else "fail"
    failure_reason = "" if status == "pass" else text.strip().splitlines()[-1] if text.strip() else "missing output"
    perf_match = re.search(r"Cutlass GEMM Performance:\s+\[([0-9.]+)\]TFlop/s\s+\(([0-9.]+)\)ms", text)
    avg_tflops = perf_match.group(1) if perf_match else ""
    avg_runtime_ms = perf_match.group(2) if perf_match else ""
    return [
        with_result_metadata(
            {
                "run_id": run_id,
                "stage": metadata["stage"],
                "attempt_index": metadata["attempt_index"],
                "shape_id": metadata["shape_id"],
                "candidate_id": metadata["candidate_id"],
                "compiler_profile_id": metadata["compiler_profile_id"],
                "status": status,
                "verify_status": status,
                "layout": metadata["layout"],
                "dtype_a": metadata["dtype_a"],
                "dtype_b": metadata["dtype_b"],
                "dtype_c": metadata["dtype_c"],
                "dtype_d": metadata.get("dtype_d", metadata["dtype_c"]),
                "dtype_acc": metadata["dtype_acc"],
                "m": metadata["m"],
                "n": metadata["n"],
                "k": metadata["k"],
                "batch_count": metadata.get("batch_count", 1),
                "split_k": metadata.get("split_k", 1),
                "avg_runtime_ms": avg_runtime_ms,
                "best_runtime_ms": avg_runtime_ms,
                "worst_runtime_ms": avg_runtime_ms,
                "avg_tflops": avg_tflops,
                "avg_throughput": "",
                "max_error": "",
                "close_call_group": "",
                "failure_reason": failure_reason,
                "stdout_log": str(log_path),
            },
            metadata,
        )
    ]


def run_entries_with_streamk_example(entries, logs_dir, exe, cwd=None, shell_init=None, timeout=None):
    rows = []
    commands = []
    for entry in entries:
        candidate = entry["candidate"]
        shape = entry["shape"]
        bm_name = entry["bm_name"]
        metadata = {
            bm_name: {
                "shape_id": shape["shape_id"],
                "candidate_id": candidate["candidate_id"],
                "compiler_profile_id": candidate["compiler_profile_id"],
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
                "batch_count": shape.get("batch_count", 1),
                "kernel_name": candidate["kernel_name"],
                "split_k": candidate["split_k"],
            }
        }
        metadata[bm_name].update(row_result_metadata(candidate))
        log_path = logs_dir / f"{bm_name}.log"
        runtime_defaults = dict(candidate.get("runtime_defaults", {}))
        runtime_defaults.update(shape.get("runtime_defaults", {}))
        batch_count = shape.get("batch_count", runtime_defaults.get("batch_count", 1))
        alpha = runtime_defaults.get("alpha", 1.0)
        beta = runtime_defaults.get("beta", 0.0)
        iterations = runtime_defaults.get("iterations", 20)
        warmup_iterations = runtime_defaults.get("warmup_iterations", 0)
        verify = runtime_defaults.get("verify", 1)
        streamk_dtype = candidate.get("streamk_dtype_preset", candidate["dtype_a"])
        command = [
            exe,
            f"--dtype={streamk_dtype}",
            f"--m={shape['m']}",
            f"--n={shape['n']}",
            f"--k={shape['k']}",
            f"--l={batch_count}",
            f"--alpha={alpha}",
            f"--beta={beta}",
            f"--warmup_iterations={warmup_iterations}",
            f"--iterations={iterations}",
            f"--verify={verify}",
        ]
        streamk_mode = candidate.get("streamk_mode", "")
        if streamk_mode == "splitk":
            command.extend(["--splitk", f"--splits={candidate['split_k']}"])
        elif streamk_mode == "data_parallel":
            command.append("--dp")
        result, timed_out, timeout_reason = run_benchmark(command, log_path, cwd=cwd, shell_init=shell_init, timeout=timeout)
        parsed = timeout_rows([entry], log_path, timeout_reason) if timed_out else parse_streamk_example_log(log_path, metadata, run_id=entry["stage"])
        if result.returncode != 0 and not parsed:
            raise RuntimeError(f"StreamK example subprocess failed with return code {result.returncode}. See {log_path}")
        rows.extend(parsed)
        commands.append(shell_join(command))
    return rows, commands
