#!/usr/bin/env python3
"""EXP-182 registered wrapper for the D0/D-1 gate-parity runner."""
from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("INVESTING_PLAN_ROOT") or Path(__file__).parents[2])
HERE = ROOT / "experiments" / "EXP-182_d_1_gated_execution_parity_registered"
SOURCE = Path(os.environ.get("INVESTING_PLAN_PINNED_SOURCE")
              or ROOT / "experiments" / "EXP-181_d_1_gated_execution_parity" / "run.py")
sys.path.insert(0, str(ROOT))

loader = importlib.machinery.SourceFileLoader("exp182_runner", str(SOURCE))
spec = importlib.util.spec_from_loader("exp182_runner", loader)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
module.HERE = HERE
module.RESULTS = HERE / "results"

if __name__ == "__main__":
    module.main()
