#!/usr/bin/env python3
"""EXP-182 registered wrapper for the D0/D-1 gate-parity runner."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.v2.ops.errors import fail

raise fail(
    "INVALID_REQUEST",
    "EXP-182 disabled: reads EXP-179 parquet directly (no v2_snapshot_id), bypassing holdouts",
)
