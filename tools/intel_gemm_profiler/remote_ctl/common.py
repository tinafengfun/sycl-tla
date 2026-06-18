#!/usr/bin/env python3
"""Shared SSH/session helpers for remote exact-shape control."""

from __future__ import annotations

import argparse
import os
import posixpath
from dataclasses import dataclass
from pathlib import Path

try:
    import paramiko
except ImportError as exc:  # pragma: no cover - runtime environment dependent
    raise SystemExit("paramiko is required for remote_exact_shape_search_ctl.py") from exc


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REMOTE_ROOT = "/root/cutlass_profile_device7_b70_2500mhz"
DEFAULT_REMOTE_REPO = f"{DEFAULT_REMOTE_ROOT}/sycl-tla"
DEFAULT_RUNS_DIR = f"{DEFAULT_REMOTE_ROOT}/screen_runs"
SYNC_FILES = [
    "benchmarks/common.hpp",
    "benchmarks/gemm/CMakeLists.txt",
    "benchmarks/gemm/benchmark_runner.hpp",
    "benchmarks/gemm/benchmarks_sycl.hpp",
    "tools/util/include/cutlass/util/sycl_event_manager.hpp",
    "benchmarks/gemm/bmg_streamk_seed_tile.def",
    "benchmarks/gemm/bmg_streamk_expanded_tile.def",
    "benchmarks/gemm/bmg_streamk_exhaustive_missing_tile.def",
    "test/benchmarks/intel_gemm_profiler.py",
    "test/benchmarks/intel_gemm_profiler/__init__.py",
    "tools/intel_gemm_profiler/__init__.py",
    "tools/intel_gemm_profiler/ali_dataset.py",
    "tools/intel_gemm_profiler/candidate_entries.py",
    "tools/intel_gemm_profiler/candidate_manifest.py",
    "tools/intel_gemm_profiler/candidate_space.py",
    "tools/intel_gemm_profiler/candidates.py",
    "tools/intel_gemm_profiler/config.py",
    "tools/intel_gemm_profiler/catalog.py",
    "tools/intel_gemm_profiler/catalog_generator.py",
    "tools/intel_gemm_profiler/catalog_layered.py",
    "tools/intel_gemm_profiler/catalog_space.py",
    "tools/intel_gemm_profiler/constraints.py",
    "tools/intel_gemm_profiler/constraints_probe.py",
    "tools/intel_gemm_profiler/device_target.py",
    "tools/intel_gemm_profiler/dispatch.py",
    "tools/intel_gemm_profiler/hw_specs.py",
    "tools/intel_gemm_profiler/mixed_dtype_codegen.py",
    "tools/intel_gemm_profiler/prefilter.py",
    "tools/intel_gemm_profiler/phase_a.py",
    "tools/intel_gemm_profiler/runner.py",
    "tools/intel_gemm_profiler/runner_benchmark.py",
    "tools/intel_gemm_profiler/runner_benchmark_parse.py",
    "tools/intel_gemm_profiler/schemas.py",
    "tools/intel_gemm_profiler/selector.py",
    "tools/intel_gemm_profiler/selector_summary.py",
    "tools/intel_gemm_profiler/source_templates.py",
    "tools/intel_gemm_profiler/utils.py",
    "tools/intel_gemm_profiler/workflow.py",
    "tools/intel_gemm_profiler/cli.py",
    "tools/intel_gemm_profiler/build_config_bmg_perf.json",
    "tools/intel_gemm_profiler/runtime_config_bmg_perf.json",
    "tools/intel_gemm_profiler/intel_gemm_kernel_catalog_level0.json",
    "tools/intel_gemm_profiler/intel_gemm_hw_reference_specs.json",
    "tools/intel_gemm_profiler/analysis.py",
    "tools/intel_gemm_profiler/analysis_gap.py",
    "tools/intel_gemm_profiler/artifacts.py",
    "tools/intel_gemm_profiler/bundle.py",
    "tools/intel_gemm_profiler/build_plan.py",
    "tools/intel_gemm_profiler/build_exec.py",
    "tools/intel_gemm_profiler/inputs.py",
    "tools/intel_gemm_profiler/exact_shape_priority.py",
    "tools/intel_gemm_profiler/README.md",
    "tools/intel_gemm_profiler/OPERATION_MANUAL.md",
    "tools/intel_gemm_profiler/phase_b.py",
    "tools/intel_gemm_profiler/phase_b_outputs.py",
    "tools/intel_gemm_profiler/gen_main.py",
    "tools/intel_gemm_profiler/gen_mini_hpp.py",
    "tools/intel_gemm_profiler/exact_shape_search_report.py",
    "tools/intel_gemm_profiler/exact_shape_search_report_artifacts.py",
    "tools/intel_gemm_profiler/exact_shape_search_report_export.py",
    "tools/intel_gemm_profiler/exact_shape_search_report_repro.py",
    "tools/intel_gemm_profiler/exact_shape_search_report_rows.py",
    "tools/intel_gemm_profiler/exact_shape_report/__init__.py",
    "tools/intel_gemm_profiler/exact_shape_report/artifacts.py",
    "tools/intel_gemm_profiler/exact_shape_report/export.py",
    "tools/intel_gemm_profiler/exact_shape_report/repro.py",
    "tools/intel_gemm_profiler/exact_shape_report/rows.py",
    "tools/intel_gemm_profiler/remote_exact_shape_search.sh",
    "tools/intel_gemm_profiler/remote_exact_shape_search_ctl.py",
    "tools/intel_gemm_profiler/remote_exact_shape_search_ctl_common.py",
    "tools/intel_gemm_profiler/remote_exact_shape_search_ctl_commands.py",
    "tools/intel_gemm_profiler/remote_ctl/__init__.py",
    "tools/intel_gemm_profiler/remote_ctl/common.py",
    "tools/intel_gemm_profiler/remote_ctl/commands.py",
    "tools/intel_gemm_profiler/remote_exact_shape_search_status.sh",
    "tools/intel_gemm_profiler/remote_exact_shape_search_stop.sh",
    "tools/gen_main.py",
    "tools/gen_mini_hpp.py",
    "tools/remote_exact_shape_search.sh",
    "tools/remote_exact_shape_search_status.sh",
    "tools/remote_exact_shape_search_stop.sh",
    "tools/exact_shape_search_report.py",
    "media/docs/cpp/intel_b70_exact_shape_search_runbook.md",
]


@dataclass
class RemoteConfig:
    host: str
    user: str
    password: str
    remote_repo: str
    runs_dir: str
    accept_new_host_key: bool


class RemoteSession:
    def __init__(self, config: RemoteConfig):
        self.config = config
        self.client = paramiko.SSHClient()
        self.client.load_system_host_keys()
        if config.accept_new_host_key:
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        else:
            self.client.set_missing_host_key_policy(paramiko.RejectPolicy())
        self.client.connect(
            config.host,
            username=config.user,
            password=config.password or None,
            allow_agent=True,
            look_for_keys=True,
            timeout=20,
        )

    def close(self) -> None:
        self.client.close()

    def run(self, command: str, timeout: int = 120) -> tuple[int, str, str]:
        stdin, stdout, stderr = self.client.exec_command(command, timeout=timeout)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        code = stdout.channel.recv_exit_status()
        return code, out, err

    def upload(self, local_path: Path, remote_path: str, executable: bool = False) -> None:
        sftp = self.client.open_sftp()
        try:
            self._ensure_remote_dir(sftp, posixpath.dirname(remote_path))
            sftp.put(str(local_path), remote_path)
            if executable:
                sftp.chmod(remote_path, 0o755)
        finally:
            sftp.close()

    def _ensure_remote_dir(self, sftp, remote_dir: str) -> None:
        if not remote_dir or remote_dir == "/":
            return
        parts = []
        current = remote_dir
        while current and current != "/":
            parts.append(current)
            current = posixpath.dirname(current)
        for path in reversed(parts):
            try:
                sftp.stat(path)
            except FileNotFoundError:
                sftp.mkdir(path)


def build_remote_config(args: argparse.Namespace) -> RemoteConfig:
    return RemoteConfig(
        host=args.host,
        user=args.user,
        password=args.password,
        remote_repo=args.remote_repo,
        runs_dir=args.runs_dir,
        accept_new_host_key=args.accept_new_host_key,
    )


def sync_files(session: RemoteSession) -> None:
    for rel in SYNC_FILES:
        local_path = REPO_ROOT / rel
        remote_path = posixpath.join(session.config.remote_repo, rel)
        session.upload(local_path, remote_path, executable=remote_path.endswith(".sh"))


def default_password() -> str:
    return os.environ.get("EXACT_SHAPE_REMOTE_PASSWORD", "")
