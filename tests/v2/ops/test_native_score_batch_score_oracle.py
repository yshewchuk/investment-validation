"""Independent oracle for chooser fold selection and native score output."""
from __future__ import annotations

from dataclasses import replace
from datetime import date
from types import SimpleNamespace

import pandas as pd

from engine.v2.models.admissible_table import legacy_n_admissible_table
from engine.v2.models.chooser_cutoff import canonical_day
from engine.v2.models.lineage import DataDependency, Lineage
from engine.v2.models.training.chooser_pool import build_chooser_analog_pool_artifact
from engine.v2.ops import native_score_batch as nsb
from engine.v2.scoring import application, release_bindings
from tests.v2.ops.test_native_score_batch_driver_alias import (
    SIZE,
    _assemble,
    _event_inputs,
    _stage,
)

_AS_OF = "2026-01-10"
_EVENT_DATE = "2026-02-15"
_IV_CRUSH = ("iv_crush", "*", ("x", "iv"))
_CHOOSER_PRIMITIVES = (
    "mean_prior_abs_move", "ema12r_abs", "signed_streak",
    "mean_prior_or_implied", "mcap_log", "or_implied", "or_rvol30",
    "spy_ret21", "spy_ret63", "spy_ret252", "spy_dd252", "spy_vol20",
    "spy_vol5", "spy_vol60", "spy_vol252", "spy_vol20_rel252",
)
_FOLDS = (
    ("2025-12-01", "fold-2025-12", "legacy_signal"),
    ("2026-01-01", "fold-2026-01", "signal"),
    ("2026-02-01", "fold-2026-02", "future_signal"),
)


def _tiny_analog_pool():
    lineage = Lineage(data=(DataDependency(
        table="synthetic.chooser_oracle", end_exclusive="2026-01-11"),))
    rows = [
        {"strategy": "TWIN-P", "entry_date": entry, "exit_date": exit,
         "pnl": float(i),
         "exp_pnl_sim": float(i) / 10.0, "width_over_forecast": 0.8,
         "n_legs": 7.0, "anchor_over_spot": 1.0, "rel_spread": 0.1}
        for i, entry, exit in ((9, "2025-09-01", "2025-09-02"),
                               (10, "2025-10-01", "2025-10-02"),
                               (11, "2025-11-01", "2025-11-02"))
    ]
    return build_chooser_analog_pool_artifact(
        rows, pool_id="tiny-chooser-oracle", cutoff=_AS_OF, lineage=lineage)


def _event(ticker: str, event_id: str, signal: float):
    panel = {name: 0.1 for name in _CHOOSER_PRIMITIVES}
    panel.update({
        "date": _EVENT_DATE, "mcap_log": 20.0, "iv30": 55.0,
        "signal": signal, "legacy_signal": 7.0, "future_signal": 9.0,
        "x": 0.5, "iv": 0.02, "zz": 0.03,
    })
    calendar = {
        "event_id": event_id, "ticker": ticker, "event_date": _EVENT_DATE,
        "entry_date": _EVENT_DATE, "exit_date": "2026-02-16",
        "expiry": "2026-02-16", "spot": 100.0,
        "calendar_observed_through": "2026-01-09",
    }
    quotes = [
        {"right": "P", "strike": float(strike), "expiry": "2026-02-16",
         "bid": max(100.0 - strike, 0.0) + 0.4,
         "ask": max(100.0 - strike, 0.0) + 0.5,
         "observed_at": "2026-01-09"}
        for strike in range(70, 131)
    ]
    return _event_inputs(
        strategy="TWIN-P", ticker=ticker,
        event_date=pd.Timestamp(_EVENT_DATE),
        calendar_row=calendar, panel_row=panel, panel_anchor="2026-01-09",
        tier4_row={"pred_abs_move": 0.05,
                   "pred_abs_move_fold_start": "2026-01-01"},
        quote_rows=quotes,
    )


def test_native_score_batch_matches_hand_computed_fold_and_score(
        tmp_path, monkeypatch):
    specs = [
        SIZE,
        _IV_CRUSH,
        ("chooser", "DYN-SV", ("pred_im_t1_d14",)),
        ("implied_t1", "fold-2025-12", ("legacy_signal",)),
        ("implied_t1", "fold-2026-01", ("signal",)),
        ("implied_t1", "fold-2026-02", ("future_signal",)),
        ("runup_move", "DYN-SV", ("signal",)),
    ]
    binding = _stage(tmp_path, specs)
    by_id = {item.binding_id: item for item in binding.model_release.bindings}
    fold_bindings = {
        fold_start: by_id[f"b-implied_t1-{strategy_id}"]
        for fold_start, strategy_id, _feature in _FOLDS
    }
    runup_binding = by_id["b-runup_move-DYN-SV"]
    binding = replace(
        binding,
        chooser_analog_artifacts=(_tiny_analog_pool(),),
        chooser_admissible_artifacts=(legacy_n_admissible_table(),),
    )

    # Independent oracle: use the event/as-of month, then choose the latest
    # hand-written fold at or before that month. Jan 10 therefore picks Jan 1.
    decision_day = min(date.fromisoformat(_AS_OF), date.fromisoformat(_EVENT_DATE))
    cutoff_month = decision_day.replace(day=1).isoformat()
    eligible = [fold for fold, _strategy, _feature in _FOLDS
                if fold <= cutoff_month]
    expected_fold = max(eligible)
    assert expected_fold == "2026-01-01"

    observed_folds = []

    def load_selected_fold(_binding, event_date, as_of, fold_start, decision_cutoff):
        assert canonical_day(event_date) == _EVENT_DATE
        assert canonical_day(as_of) == _AS_OF
        assert canonical_day(decision_cutoff) == _AS_OF
        observed_folds.append((fold_start, decision_cutoff))
        producer = fold_bindings[fold_start]
        pools = {
            "pred_abs_move": {"predictions": (1.0, 2.0),
                              "residuals": (0.0, 0.1), "interval_floor": 0.1},
            "pred_im_t1_d14": {"predictions": (1.0, 2.0),
                               "residuals": (0.0, 0.1), "interval_floor": 0.0},
            "pred_runup_abs_move_d14": {"predictions": (1.0, 2.0),
                                         "residuals": (0.0, 0.1),
                                         "interval_floor": 0.0},
        }
        producers = {
            "pred_im_t1_d14": producer,
            "pred_runup_abs_move_d14": runup_binding,
        }
        provenance = {
            output: {"training_cutoff": "2025-10-01", "fold_start": fold_start,
                     "descriptor_hash": f"synthetic-{fold_start}"}
            for output in producers
        }
        refs = tuple(member.content_hash
                     for item in (producer, runup_binding)
                     for member in item.members)
        return pools, producers, provenance, refs

    monkeypatch.setattr(release_bindings, "_load_chooser_fold_state", load_selected_fold)

    def entry_rule_declarations(_binding, **kwargs):
        return {
            "paired_residual_artifact": None,
            "paired_residual_recipe": {
                "move_model_id": "size", "crush_model_id": "iv-crush",
                "cutoff": kwargs["cutoff"], "content_hash": "synthetic-entry-rule",
            },
            "trailing_cutoff_artifact": None,
            "gate_recipe": {"mode": "entry_rule", "mcap_usd": kwargs["mcap_usd"]},
        }

    monkeypatch.setattr(nsb, "entry_rule_declarations", entry_rule_declarations)
    events = (_event("AAA", "evt-a", 2.0), _event("BBB", "evt-b", 3.0))
    assembled, refusals = _assemble(binding, events, as_of=_AS_OF)
    assert refusals == ()
    assert set(assembled) == {event.key for event in events}
    assert observed_folds
    assert {fold for fold, _cutoff in observed_folds} == {expected_fold}
    assert {cutoff for _fold, cutoff in observed_folds} == {_AS_OF}

    records = {
        event.key: application.score_one(*assembled[event.key])
        for event in events
    }
    # The staged JSON-linear producer is 1 + signal; the staged chooser is
    # 1 + pred_im_t1_d14. Thus signal 2 -> score 4, and signal 3 -> score 5.
    assert records[events[0].key].forecasts["chooser_score"] == 4.0
    assert records[events[1].key].forecasts["chooser_score"] == 5.0