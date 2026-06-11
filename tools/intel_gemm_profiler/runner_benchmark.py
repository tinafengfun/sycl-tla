#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import os
import signal
import subprocess

from .candidate_entries import write_config
from .runner_benchmark_parse import parse_benchmark_log, parse_metric, row_result_metadata, timeout_rows, with_result_metadata
from .utils import shell_join, write_json


def run_benchmark(command, log_path, cwd=None, shell_init=None, timeout=None):
    timed_out = False
    timeout_reason = ""

    def output_text(value):
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    if shell_init:
        payload = f"{shell_init} && {shell_join(command)}"
        popen_command = ["bash", "-lc", payload]
    else:
        popen_command = command

    process = subprocess.Popen(
        popen_command,
        cwd=cwd,
        text=True,
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=os.environ.copy(),
    )
    try:
        stdout, _ = process.communicate(timeout=timeout)
        process = subprocess.CompletedProcess(popen_command, process.returncode, output_text(stdout), "")
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        timeout_reason = f"timeout after {timeout}s"
        try:
            os.killpg(process.pid, signal.SIGTERM)
            stdout, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, _ = process.communicate()
        process = subprocess.CompletedProcess(popen_command, 124, output_text(stdout or exc.stdout), output_text(exc.stderr))
        import time

        time.sleep(2.0)

    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write(output_text(process.stdout))
        if timed_out:
            handle.write(f"\nTIMEOUT: {timeout_reason}\n")
    return process, timed_out, timeout_reason


def chunked_path(path, chunk_index):
    return path.with_name(f"{path.stem}_part{chunk_index:03d}{path.suffix}")


def suffixed_path(path, suffix):
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def rows_seen_keys(rows):
    return {
        (row["stage"], int(row["attempt_index"]), row["shape_id"], row["candidate_id"])
        for row in rows
    }


def entries_missing_rows(entries, rows):
    seen = rows_seen_keys(rows)
    return [
        entry
        for entry in entries
        if (
            entry["stage"],
            int(entry["attempt_index"]),
            entry["shape"]["shape_id"],
            entry["candidate"]["candidate_id"],
        )
        not in seen
    ]


def run_entries_with_benchmark_attempt(entries, config_path, manifest_path, log_path, exe, cwd=None, shell_init=None, timeout=None):
    metadata = write_config(entries, config_path)
    write_json(manifest_path, metadata)
    command = [exe, f"--config_file={config_path}"]
    result, timed_out, timeout_reason = run_benchmark(command, log_path, cwd=cwd, shell_init=shell_init, timeout=timeout)
    rows = parse_benchmark_log(log_path, metadata, run_id=entries[0]["stage"]) if entries else []
    return rows, command, result, timed_out, timeout_reason


def run_entries_with_benchmark_once(entries, config_path, manifest_path, log_path, exe, cwd=None, shell_init=None, timeout=None):
    rows, command, result, timed_out, timeout_reason = run_entries_with_benchmark_attempt(
        entries,
        config_path,
        manifest_path,
        log_path,
        exe,
        cwd=cwd,
        shell_init=shell_init,
        timeout=timeout,
    )
    if timed_out:
        rows.extend(timeout_rows(entries_missing_rows(entries, rows), log_path, timeout_reason))
    if result.returncode != 0 and not rows:
        raise RuntimeError(f"Benchmark subprocess failed with return code {result.returncode}. See {log_path}")
    return rows, command


def run_entries_with_benchmark_retrying_timeouts(entries, config_path, manifest_path, log_path, exe, cwd=None, shell_init=None, timeout=None, depth=0):
    rows, command, result, timed_out, timeout_reason = run_entries_with_benchmark_attempt(
        entries,
        config_path,
        manifest_path,
        log_path,
        exe,
        cwd=cwd,
        shell_init=shell_init,
        timeout=timeout,
    )
    commands = [command]
    if timed_out:
        missing = entries_missing_rows(entries, rows)
        if missing and len(missing) < len(entries):
            split_size = max(1, (len(missing) + 1) // 2)
            for retry_index, start in enumerate(range(0, len(missing), split_size)):
                retry_entries = missing[start:start + split_size]
                suffix = f"retry{depth:02d}_{retry_index:03d}"
                retry_rows, retry_commands = run_entries_with_benchmark_retrying_timeouts(
                    retry_entries,
                    suffixed_path(config_path, suffix),
                    suffixed_path(manifest_path, suffix),
                    suffixed_path(log_path, suffix),
                    exe,
                    cwd=cwd,
                    shell_init=shell_init,
                    timeout=timeout,
                    depth=depth + 1,
                )
                rows.extend(retry_rows)
                commands.extend(retry_commands)
        elif missing:
            rows.extend(timeout_rows(missing, log_path, timeout_reason))
    if result.returncode != 0 and not rows:
        raise RuntimeError(f"Benchmark subprocess failed with return code {result.returncode}. See {log_path}")
    return rows, commands


def run_entries_with_benchmark(entries, config_path, manifest_path, log_path, exe, cwd=None, shell_init=None, timeout=None, chunk_size=0, retry_timeouts=True):
    if not entries or chunk_size <= 0 or len(entries) <= chunk_size:
        if retry_timeouts:
            rows, commands = run_entries_with_benchmark_retrying_timeouts(entries, config_path, manifest_path, log_path, exe, cwd=cwd, shell_init=shell_init, timeout=timeout)
            return rows, commands[0] if len(commands) == 1 else commands
        return run_entries_with_benchmark_once(entries, config_path, manifest_path, log_path, exe, cwd=cwd, shell_init=shell_init, timeout=timeout)

    rows = []
    commands = []
    for chunk_index, start in enumerate(range(0, len(entries), chunk_size)):
        chunk = entries[start:start + chunk_size]
        chunk_config_path = chunked_path(config_path, chunk_index)
        chunk_manifest_path = chunked_path(manifest_path, chunk_index)
        chunk_log_path = chunked_path(log_path, chunk_index)
        if retry_timeouts:
            chunk_rows, command = run_entries_with_benchmark_retrying_timeouts(
                chunk,
                chunk_config_path,
                chunk_manifest_path,
                chunk_log_path,
                exe,
                cwd=cwd,
                shell_init=shell_init,
                timeout=timeout,
            )
        else:
            chunk_rows, command = run_entries_with_benchmark_once(
                chunk,
                chunk_config_path,
                chunk_manifest_path,
                chunk_log_path,
                exe,
                cwd=cwd,
                shell_init=shell_init,
                timeout=timeout,
            )
            command = [command]
        rows.extend(chunk_rows)
        commands.extend(command)
    return rows, commands
