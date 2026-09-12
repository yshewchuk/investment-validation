#!/usr/bin/env python3
"""EXP-183 registered frozen-D0 cross-clock gate experiment."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "experiments" / "EXP-181_d_1_gated_execution_parity" / "run.py"
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("exp183_runner", SOURCE)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
module.HERE = HERE
module.RESULTS = HERE / "results"

if __name__ == "__main__":
    sys.argv.append("--cross-clock")
    module.main()
