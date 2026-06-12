#!/usr/bin/env python3
"""Compatibility wrapper for remote exact-shape control commands."""

from __future__ import annotations

try:
    from .remote_ctl.commands import (
        build_launch_command,
        build_report_command,
        build_status_command,
        build_stop_command,
        command_sync,
        emit_command_output,
        print_sync_result,
    )
except ImportError:
    from remote_ctl.commands import (  # type: ignore
        build_launch_command,
        build_report_command,
        build_status_command,
        build_stop_command,
        command_sync,
        emit_command_output,
        print_sync_result,
    )
