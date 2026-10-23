"""CLI argument registration for the snapshot preflight command."""
from __future__ import annotations

from pathlib import Path


def add_preflight_subcommand(snapshot_sub):
    """Register the inputs required by the post-import preflight command."""
    preflight = snapshot_sub.add_parser(
        "preflight", help="check imported legacy generation and feature tables before capture")
    preflight.add_argument("--source-root", required=True, type=Path)
    preflight.add_argument("--materialization-root", required=True, type=Path)
    preflight.add_argument("--receipt-id", required=True)
