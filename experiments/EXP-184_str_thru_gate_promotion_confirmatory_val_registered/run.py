#!/usr/bin/env python3
"""EXP-184 registered wrapper for the STR-THRU gate promotion confirmatory validation runner."""
from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path

STAGING_ROOT = os.environ.get("INVESTING_PLAN_ROOT")
ROOT = Path(STAGING_ROOT) if STAGING_ROOT else Path(__file__).resolve().parents[2]
HERE = ROOT if STAGING_ROOT else Path(__file__).resolve().parent
SOURCE = Path(os.environ.get("INVESTING_PLAN_PINNED_SOURCE") or
              ROOT / "experiments" / "EXP-147_str_thru_gate_promotion_confirmatory_val" / "run.py")
sys.path.insert(0, str(ROOT))

loader = importlib.machinery.SourceFileLoader("exp184_runner", str(SOURCE))
spec = importlib.util.spec_from_loader("exp184_runner", loader)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
module.HERE = HERE

if __name__ == "__main__":
    module.main()