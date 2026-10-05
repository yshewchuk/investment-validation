#!/usr/bin/env python3
"""EXP-184 registered wrapper for the STR-THRU gate promotion confirmatory validation runner."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

pinned_root = os.environ.get("INVESTING_PLAN_ROOT")
pinned_source = os.environ.get("INVESTING_PLAN_PINNED_SOURCE")
if pinned_root:
    ROOT = Path(pinned_root)
    HERE = ROOT / "experiments" / "EXP-184_str_thru_gate_promotion_confirmatory_val_registered"
    if pinned_source:
        SOURCE = Path(pinned_source)
    else:
        SOURCE = ROOT / "experiments" / "EXP-147_str_thru_gate_promotion_confirmatory_val" / "run.py"
else:
    ROOT = Path(__file__).resolve().parents[2]
    HERE = Path(__file__).resolve().parent
    SOURCE = ROOT / "experiments" / "EXP-147_str_thru_gate_promotion_confirmatory_val" / "run.py"
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("exp184_runner", SOURCE)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
module.HERE = HERE

if __name__ == "__main__":
    module.main()