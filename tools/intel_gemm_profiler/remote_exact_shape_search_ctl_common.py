#!/usr/bin/env python3
"""Compatibility wrapper for remote exact-shape control helpers."""

from __future__ import annotations

try:
    from .remote_ctl.common import DEFAULT_REMOTE_REPO, DEFAULT_RUNS_DIR, RemoteConfig, RemoteSession, SYNC_FILES, build_remote_config, default_password, sync_files
except ImportError:
    from remote_ctl.common import DEFAULT_REMOTE_REPO, DEFAULT_RUNS_DIR, RemoteConfig, RemoteSession, SYNC_FILES, build_remote_config, default_password, sync_files  # type: ignore
