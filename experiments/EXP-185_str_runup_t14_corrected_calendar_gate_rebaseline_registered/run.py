#!/usr/bin/env python3
"""EXP-185 registered wrapper for the STR-RUNUP T14 corrected-calendar gate rebaseline runner."""
from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path

pinned_root = os.environ.get("INVESTING_PLAN_ROOT")
pinned_source = os.environ.get("INVESTING_PLAN_PINNED_SOURCE")
if pinned_root:
    ROOT = Path(pinned_root)
    HERE = ROOT / "experiments" / "EXP-185_str_runup_t14_corrected_calendar_gate_rebaseline_registered"
    if pinned_source:
        SOURCE = Path(pinned_source)
    else:
        SOURCE = ROOT / "experiments" / "EXP-144_str_runup_t14_corrected_calendar_gate_rebaseline" / "run.py"
else:
    ROOT = Path(__file__).resolve().parents[2]
    HERE = Path(__file__).resolve().parent
    SOURCE = ROOT / "experiments" / "EXP-144_str_runup_t14_corrected_calendar_gate_rebaseline" / "run.py"
sys.path.insert(0, str(ROOT))

loader = importlib.machinery.SourceFileLoader("exp185_runner", str(SOURCE))
spec = importlib.util.spec_from_loader("exp185_runner", loader)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
module.HERE = HERE
module.RESULTS = HERE / "results"
# EXP-144's run.py has exactly one top-level name derived from HERE besides
# HERE itself (RESULTS); SIM_DIR and the V2_* constants come from ROOT, so they
# are already correct for this wrapper.

if __name__ == "__main__":
    module.main()
