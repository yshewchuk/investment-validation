"""Focused coverage for the opt-in analog matching evidence hook."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import engine.analogs as analogs_module
from engine.analogs import AnalogMatcher


def _row(
    event_id: str,
    *,
    ret: float,
    moneyness_pct: float,
    exit_date: str,
    strategy: str = "STR-THRU",
) -> dict:
    spot = 100.0
    return {
        "event_id": event_id,
        "strategy": strategy,
        "fill_alpha": 0.5,
        "ret": ret,
        "mcap_usd": 5e9,
        "dte_entry": 5,
        "spot_entry": spot,
        "strike": spot * (1.0 + moneyness_pct / 100.0),
        "or_implied": 5.0,
        "mean_prior_or_implied": 5.0,
        "event_date": pd.Timestamp(exit_date) - pd.Timedelta(days=1),
        "exit_date": pd.Timestamp(exit_date),
        "metadata": {"source": event_id},
    }


def _matcher() -> AnalogMatcher:
    rows = [
        _row("exact-1", ret=0.10, moneyness_pct=0.0, exit_date="2020-01-02"),
        _row("exact-2", ret=0.20, moneyness_pct=0.0, exit_date="2020-01-03"),
        _row("wide-1", ret=-0.10, moneyness_pct=4.0, exit_date="2020-01-04"),
        _row("wide-2", ret=0.30, moneyness_pct=4.0, exit_date="2020-01-05"),
        _row("wide-null", ret=np.nan, moneyness_pct=4.0, exit_date="2020-01-06"),
        _row("future", ret=9.99, moneyness_pct=0.0, exit_date="2025-01-02"),
        _row(
            "other-strategy",
            ret=8.88,
            moneyness_pct=0.0,
            exit_date="2020-01-02",
            strategy="STR-RUNUP",
        ),
    ]
    return AnalogMatcher(pd.DataFrame(rows), snapshot="snapshot-123")


def _buckets(matcher: AnalogMatcher) -> dict:
    return matcher.buckets_for(
        mcap_usd=5e9,
        dte=5,
        moneyness_pct=0.0,
        implied_ratio=1.0,
    )


def _event_ids(block: dict) -> list[str]:
    return [row["values"]["event_id"] for row in block["rows"]]


def test_evidence_hook_preserves_result_and_reports_exact_matching_rows():
    default_matcher = _matcher()
    default = default_matcher.match(
        "STR-THRU",
        _buckets(default_matcher),
        alpha=0.5,
        as_of="2021-01-01",
        min_analogs=5,
        bootstrap=0,
        request_key="request-7",
    )

    hooked_matcher = _matcher()
    captured = []
    hooked = hooked_matcher.match(
        "STR-THRU",
        _buckets(hooked_matcher),
        alpha=0.5,
        as_of="2021-01-01",
        min_analogs=5,
        bootstrap=0,
        request_key="request-7",
        evidence_hook=captured.append,
    )

    assert hooked == default
    assert hooked.as_dict() == default.as_dict()
    assert len(captured) == 1

    evidence = captured[0]
    assert evidence["schema_version"] == "analog_match_evidence.v1"
    assert evidence["snapshot"] == "snapshot-123"
    assert evidence["cutoff"] == "2021-01-01T00:00:00"
    assert evidence["request_key"] == "request-7"
    assert evidence["bucket_query"]["moneyness_band"] == "ATM"
    assert evidence["effective_bucket_query"]["implied_tercile"] == "mid"

    assert set(_event_ids(evidence["population"])) == {
        "exact-1", "exact-2", "wide-1", "wide-2", "wide-null", "future",
    }
    assert set(_event_ids(evidence["causal"])) == {
        "exact-1", "exact-2", "wide-1", "wide-2", "wide-null",
    }
    assert set(_event_ids(evidence["selected"])) == {
        "exact-1", "exact-2", "wide-1", "wide-2", "wide-null",
    }
    assert set(_event_ids(evidence["contributing"])) == {
        "exact-1", "exact-2", "wide-1", "wide-2",
    }
    assert evidence["selected"]["row_ids"] == evidence["widening_steps"][-1]["row_ids"]
    assert [step["count"] for step in evidence["widening_steps"]] == [2, 5]
    assert evidence["widening_steps"][0]["accepted"] is False
    assert evidence["widening_steps"][1]["accepted"] is True
    assert evidence["widening_steps"][1]["dropped_dimensions"] == [
        "moneyness_band"
    ]
    assert hooked.n == 4
    json.dumps(evidence)


def test_default_match_does_not_build_or_emit_evidence(monkeypatch):
    matcher = _matcher()

    def fail_if_called(_frame):
        raise AssertionError("default matching must not build evidence")

    monkeypatch.setattr(analogs_module, "_evidence_rows", fail_if_called)
    result = matcher.match(
        "STR-THRU",
        _buckets(matcher),
        alpha=0.5,
        as_of="2021-01-01",
        min_analogs=5,
        bootstrap=0,
    )
    assert result.n == 4


def test_evidence_rows_are_defensive_copies():
    matcher = _matcher()
    captured = []
    kwargs = {
        "alpha": 0.5,
        "as_of": "2021-01-01",
        "min_analogs": 5,
        "bootstrap": 0,
        "request_key": "request-7",
    }
    matcher.match(
        "STR-THRU",
        _buckets(matcher),
        evidence_hook=captured.append,
        **kwargs,
    )
    first = captured[0]
    first["selected"]["rows"][0]["values"]["ret"] = 999.0
    first["selected"]["rows"][0]["values"]["metadata"]["source"] = "changed"
    first["bucket_query"]["moneyness_band"] = "changed"

    second_capture = []
    repeated = matcher.match(
        "STR-THRU",
        _buckets(matcher),
        evidence_hook=second_capture.append,
        **kwargs,
    )
    second = second_capture[0]

    assert repeated.mean == 0.125
    assert second["selected"]["rows"][0]["values"]["ret"] == 0.10
    assert second["selected"]["rows"][0]["values"]["metadata"]["source"] == "exact-1"
    assert second["bucket_query"]["moneyness_band"] == "ATM"
    assert matcher.trades.loc[0, "ret"] == 0.10
    assert matcher.trades.loc[0, "metadata"]["source"] == "exact-1"
