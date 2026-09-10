#!/usr/bin/env python3
"""The DYN-SV chooser's training set.

The chooser ranks the structures offered on an event by event-demeaned
realized PnL. Its training rows are candidate-level: one row per offered
structure per event, with the EXP-161 registered inputs, the menu one-hots,
the thirteen analytic payoff-schematic features, and the Tier-4 forecast
columns, against the dev demeaned target.

The pipeline lives in EXP-169's runner (it is the definition of the promoted
configuration — the registry entry cites the experiment). This module is the
engine-side handle the model-evidence builder dispatches to, so the
dashboard rebuilds the exact rows the champion was trained on rather than a
reimplementation that could drift."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_EXPERIMENT = ROOT / "experiments/EXP-169_menu7prime_confirmation/run.py"

TARGET = "dev_target"


def _experiment():
    spec = importlib.util.spec_from_file_location("evidence_chooser_exp169", _EXPERIMENT)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("evidence_chooser_exp169", module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_dataset():
    """(rows, target, features) for the evidence table — the chooser's frame."""
    exp = _experiment()
    mid, _raw = exp.load_data_menu(exp.MENU)
    dataset = exp.base.add_causal_analogs(mid)
    dataset = exp.build_schematics(dataset)
    dataset = exp.join_tier4(dataset)
    event_mean = dataset.groupby("event_id")["pnl"].transform("mean")
    dataset = dataset.assign(dev_target=dataset["pnl"] - event_mean)
    features = list(exp.menu_features(exp.MENU))
    return dataset, TARGET, features
