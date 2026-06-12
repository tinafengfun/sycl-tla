#!/usr/bin/env python3
"""Local controller for remote exact-shape B70 search workflows."""

from __future__ import annotations

import argparse
import json
import posixpath
import sys

if __package__ in (None, ""):
    from pathlib import Path

    PACKAGE_ROOT = Path(__file__).resolve().parents[1]
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    __package__ = "intel_gemm_profiler"

from .remote_exact_shape_search_ctl_commands import (
    build_launch_command,
    build_report_command,
    build_status_command,
    build_stop_command,
    command_sync as run_sync,
    emit_command_output,
)
from .remote_exact_shape_search_ctl_common import (
    DEFAULT_REMOTE_REPO,
    DEFAULT_RUNS_DIR,
    RemoteSession,
    build_remote_config,
    default_password,
    sync_files,
)


def command_sync(args: argparse.Namespace) -> None:
    session = RemoteSession(build_remote_config(args))
    try:
        run_sync(session)
    finally:
        session.close()


def command_launch(args: argparse.Namespace) -> None:
    session = RemoteSession(build_remote_config(args))
    try:
        skip_remote_repo_sync = args.skip_remote_repo_sync
        if not args.no_sync_files:
            sync_files(session)
            skip_remote_repo_sync = True

        args.skip_remote_repo_sync = skip_remote_repo_sync
        launch_cmd, run_id, log_path = build_launch_command(
            session.config.remote_repo,
            session.config.runs_dir,
            args,
        )
        code, out, err = session.run(launch_cmd, timeout=60)
        if code != 0 or not out.strip():
            raise SystemExit(json.dumps({"status": "launch_failed", "stdout": out, "stderr": err}, indent=2))
        pid = out.strip().splitlines()[-1].strip()
        run_dir = posixpath.join(session.config.runs_dir, run_id)
        print(json.dumps({"status": "launched", "run_id": run_id, "run_dir": run_dir, "pid": pid, "log_file": log_path}, indent=2))
    finally:
        session.close()


def command_status(args: argparse.Namespace) -> None:
    session = RemoteSession(build_remote_config(args))
    try:
        cmd = build_status_command(
            session.config.remote_repo,
            session.config.runs_dir,
            args.run_dir or "",
        )
        code, out, err = session.run(cmd, timeout=args.timeout)
        emit_command_output(code, out, err)
    finally:
        session.close()


def command_stop(args: argparse.Namespace) -> None:
    session = RemoteSession(build_remote_config(args))
    try:
        if not args.no_sync_files:
            sync_files(session)
        cmd = build_stop_command(
            session.config.remote_repo,
            session.config.runs_dir,
            args.run_dir or "",
        )
        code, out, err = session.run(cmd, timeout=120)
        emit_command_output(code, out, err)
    finally:
        session.close()


def command_report(args: argparse.Namespace) -> None:
    session = RemoteSession(build_remote_config(args))
    try:
        if not args.no_sync_files:
            sync_files(session)
        run_dir = args.run_dir or ""
        if not run_dir:
            raise SystemExit("--run-dir is required for report")
        output_dir = args.output_dir or posixpath.join(run_dir, "reports")
        cmd = build_report_command(
            session.config.remote_repo,
            run_dir,
            output_dir,
            args.shape_tag,
        )
        code, out, err = session.run(cmd, timeout=args.timeout)
        emit_command_output(code, out, err)
    finally:
        session.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local controller for remote exact-shape B70 search workflows.")
    parser.add_argument("--host", default="10.239.11.149")
    parser.add_argument("--user", default="root")
    parser.add_argument("--password", default=default_password())
    parser.add_argument(
        "--accept-new-host-key",
        action="store_true",
        help="Trust and add a new host key for the remote host if it is not already present in known_hosts.",
    )
    parser.add_argument("--remote-repo", default=DEFAULT_REMOTE_REPO)
    parser.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)

    subparsers = parser.add_subparsers(dest="command", required=True)

    sync_parser = subparsers.add_parser("sync", help="Upload remote exact-shape scripts/docs to the remote repo.")
    sync_parser.set_defaults(func=command_sync)

    launch_parser = subparsers.add_parser("launch", help="Launch a remote exact-shape search run.")
    launch_parser.add_argument("--run-id", default="")
    launch_parser.add_argument("--gpu-ids", default="0,1,2,3,4")
    launch_parser.add_argument("--shapes", default="2048x384x3584;8192x384x3584")
    launch_parser.add_argument("--layouts", default="rcr,rrr")
    launch_parser.add_argument("--kernel-catalog-source", default="layered_bmg_scheduler_expanded")
    launch_parser.add_argument("--batch-size", type=int, default=1)
    launch_parser.add_argument("--skip-remote-repo-sync", action="store_true", help="Pass SKIP_SYNC=1 to the remote launcher.")
    launch_parser.add_argument("--stop-existing", action="store_true", default=True, help="Stop existing exact-shape runs before launch.")
    launch_parser.add_argument("--no-stop-existing", action="store_false", dest="stop_existing")
    launch_parser.add_argument("--resume-run", action="store_true", help="Reuse an existing run dir and skip completed batches.")
    launch_parser.add_argument("--no-sync-files", action="store_true", help="Do not upload local scripts/docs before launch.")
    launch_parser.set_defaults(func=command_launch)

    status_parser = subparsers.add_parser("status", help="Query remote exact-shape search status.")
    status_parser.add_argument("--run-dir", default="")
    status_parser.add_argument("--timeout", type=int, default=180)
    status_parser.set_defaults(func=command_status)

    stop_parser = subparsers.add_parser("stop", help="Stop a remote exact-shape search run.")
    stop_parser.add_argument("--run-dir", default="")
    stop_parser.add_argument("--no-sync-files", action="store_true", help="Do not upload local scripts/docs before stopping.")
    stop_parser.set_defaults(func=command_stop)

    report_parser = subparsers.add_parser("report", help="Generate exact-shape summary reports on the remote run output.")
    report_parser.add_argument("--run-dir", required=True)
    report_parser.add_argument("--shape-tag", default="")
    report_parser.add_argument("--output-dir", default="")
    report_parser.add_argument("--timeout", type=int, default=180)
    report_parser.add_argument("--no-sync-files", action="store_true", help="Do not upload local scripts/docs before reporting.")
    report_parser.set_defaults(func=command_report)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
