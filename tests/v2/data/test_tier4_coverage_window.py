"""#1047: the Tier-4 coverage requirement spans every fold the score window can reach, down to the month before the event."""
from __future__ import annotations

from engine.v2.data import tier4_coverage as tc

PANEL_SHA = "a" * 64
MODELS = ({"model_id": "opf_implied_t1_gbm", "features": ("f",), "produces": "x"},
          {"model_id": "runup_move_d14_v1_gbm", "features": ("f",), "produces": "y"})


def test_next_month_event_requires_both_current_and_next_fold():
    # session in October, event in November: decision date may land in either month
    required = tc.required_serving_triples([("2026-11-05", "2026-10-19")], MODELS, PANEL_SHA)
    assert required == frozenset(
        (m["model_id"], fold, PANEL_SHA) for m in MODELS for fold in ("202610", "202611"))


def test_same_month_event_requires_previous_and_event_fold():
    required = tc.required_serving_triples([("2026-10-28", "2026-10-19")], MODELS[:1], PANEL_SHA)
    assert required == frozenset(("opf_implied_t1_gbm", f, PANEL_SHA) for f in ("202609", "202610"))


def test_event_before_as_of_requires_month_before_and_event_fold():
    required = tc.required_serving_triples([("2026-09-20", "2026-10-19")], MODELS[:1], PANEL_SHA)
    assert required == frozenset(("opf_implied_t1_gbm", f, PANEL_SHA) for f in ("202608", "202609"))


def test_early_entry_strategy_event_early_in_session_month_requires_prior_fold():
    # STR-RUNUP decides ~14 trading days before the event, before the session
    required = tc.required_serving_triples([("2026-10-05", "2026-10-01")], MODELS[:1], PANEL_SHA)
    assert required == frozenset(("opf_implied_t1_gbm", f, PANEL_SHA) for f in ("202609", "202610"))


def test_event_two_months_out_requires_every_intermediate_fold():
    required = tc.required_serving_triples([("2026-12-02", "2026-10-19")], MODELS[:1], PANEL_SHA)
    assert required == frozenset(("opf_implied_t1_gbm", f, PANEL_SHA) for f in ("202610", "202611", "202612"))


def test_missing_next_month_fold_is_reported_absent():
    required = tc.required_serving_triples([("2026-11-05", "2026-10-19")], MODELS[:1], PANEL_SHA)
    missing = tc.missing_triples(required, {}, None, registry_models=MODELS[:1])
    assert missing == [
        {"model_id": "opf_implied_t1_gbm", "fold": "202610", "panel_sha12": "a" * 12, "reason": "absent"},
        {"model_id": "opf_implied_t1_gbm", "fold": "202611", "panel_sha12": "a" * 12, "reason": "absent"},
    ]
