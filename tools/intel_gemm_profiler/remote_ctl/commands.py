#!/usr/bin/env python3
"""Command builders and command execution helpers for remote exact-shape control."""

from __future__ import annotations

import json
import posixpath
import shlex
import sys
from datetime import datetime

from .common import RemoteSession, SYNC_FILES, sync_files


def print_sync_result() -> None:
    print(json.dumps({"status": "synced", "files": SYNC_FILES}, indent=2))


def build_launch_command(remote_repo: str, runs_dir: str, args) -> tuple[str, str, str]:
    skip_remote_repo_sync = args.skip_remote_repo_sync
    run_id = args.run_id or f"shape_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = posixpath.join(runs_dir, run_id)
    log_path = posixpath.join(run_dir, "launcher.log")
    env_parts = [
        ("RUN_ID", run_id),
        ("RUN_DIR", run_dir),
        ("GPU_IDS", args.gpu_ids),
        ("SHAPES", args.shapes),
        ("LAYOUTS", args.layouts),
        ("KERNEL_CATALOG_SOURCE", args.kernel_catalog_source),
        ("BATCH_SIZE", str(args.batch_size)),
        ("STOP_EXISTING", "1" if args.stop_existing else "0"),
        ("RESUME_RUN", "1" if args.resume_run else "0"),
        ("SKIP_SYNC", "1" if skip_remote_repo_sync else "0"),
    ]
    payload = json.dumps(
        {
            "remote_repo": remote_repo,
            "run_dir": run_dir,
            "log_path": log_path,
            "env": dict(env_parts),
        }
    )
    launch_cmd = f"""python3 - <<'PY'
import json
import os
import subprocess

cfg = json.loads({shlex.quote(payload)})
os.makedirs(cfg["run_dir"], exist_ok=True)
env = os.environ.copy()
env.update(cfg["env"])
with open(cfg["log_path"], "ab", buffering=0) as log_file:
    proc = subprocess.Popen(
        ["bash", "tools/intel_gemm_profiler/remote_exact_shape_search.sh"],
        cwd=cfg["remote_repo"],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
print(proc.pid)
PY"""
    return launch_cmd, run_id, log_path


def build_status_command(remote_repo: str, runs_dir: str, run_dir: str) -> str:
    cmd = f"cd {shlex.quote(remote_repo)} && RUNS_DIR={shlex.quote(runs_dir)} "
    if run_dir:
        cmd += f"RUN_DIR={shlex.quote(run_dir)} "
    return cmd + "bash tools/intel_gemm_profiler/remote_exact_shape_search_status.sh"


def build_stop_command(remote_repo: str, runs_dir: str, run_dir: str) -> str:
    cmd = f"cd {shlex.quote(remote_repo)} && RUNS_DIR={shlex.quote(runs_dir)} "
    if run_dir:
        cmd += f"RUN_DIR={shlex.quote(run_dir)} "
    return cmd + "bash tools/intel_gemm_profiler/remote_exact_shape_search_stop.sh"


def build_report_command(remote_repo: str, run_dir: str, output_dir: str, shape_tag: str) -> str:
    cmd = (
        f"cd {shlex.quote(remote_repo)} && "
        f"python3 tools/intel_gemm_profiler/exact_shape_search_report.py "
        f"--run-dir {shlex.quote(run_dir)} "
        f"--output-dir {shlex.quote(output_dir)} "
    )
    if shape_tag:
        cmd += f"--shape-tag {shlex.quote(shape_tag)} "
    return cmd


def emit_command_output(code: int, out: str, err: str) -> None:
    sys.stdout.write(out)
    if err:
        sys.stderr.write(err)
    raise SystemExit(code)


def command_sync(session: RemoteSession) -> None:
    sync_files(session)
    print_sync_result()
