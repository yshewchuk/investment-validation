"""Price version identity digests, verification and bounded per-ticker memory."""
import tracemalloc
from copy import deepcopy
from datetime import date, timedelta

import numpy as np
import pytest

from engine.v2.data.errors import DataError
from engine.v2.foundation import content_hash
from engine.v2.ops import native_registration
from engine.v2.research import _magnitude_panel_sources as sources
from engine.v2.research import magnitude_panel_qualified as builder


def _rows(n, ticker, bump=None):
    rows = [{"date": (date(2000, 1, 3) + timedelta(days=i)).isoformat(),
             "retrieved_at": "2026-10-01T00:00:00+00:00", "deleted": False,
             "source_hash": f"sha256:{ticker}{i}", "capture_id": f"cap{i}"}
            for i in range(n)]
    if isinstance(bump, int):
        rows[bump]["source_hash"] = "sha256:revised"
    return rows


def _patch(monkeypatch, rows_by_ticker):
    monkeypatch.setattr(sources, "price_dates",
        lambda repo, snap, ticker, limits: rows_by_ticker[ticker])


def test_digests_equal_for_equal_inputs(monkeypatch):
    rows = {"A": _rows(5, "A"), "B": _rows(3, "B"), "^GSPC": _rows(7, "^GSPC")}
    _patch(monkeypatch, rows)
    first = builder._price_inputs(None, None, ["A", "B"], {})
    second = builder._price_inputs(None, None, ["A", "B"], {})
    assert first[0] == second[0]
    digests, dates = first
    assert set(digests) == {"A", "B", "^GSPC"}
    for ticker in ("A", "B", "^GSPC"):
        assert digests[ticker] == {"rows": len(rows[ticker]),
                                   "digest": content_hash(rows[ticker])}
    assert isinstance(dates["A"], np.ndarray)
    assert dates["A"].dtype == np.dtype("datetime64[ns]")
    assert len(dates["A"]) == 5


def test_digest_changes_when_one_version_row_changes(monkeypatch):
    rows = {"A": _rows(5, "A"), "B": _rows(4, "B"), "^GSPC": _rows(6, "^GSPC")}
    _patch(monkeypatch, rows)
    baseline, _ = builder._price_inputs(None, None, ["A", "B"], {})
    revised = {"A": _rows(5, "A", bump=3), "B": rows["B"], "^GSPC": rows["^GSPC"]}
    _patch(monkeypatch, revised)
    changed, _ = builder._price_inputs(None, None, ["A", "B"], {})
    assert changed["A"]["digest"] != baseline["A"]["digest"]
    assert changed["A"]["rows"] == baseline["A"]["rows"] == 5
    assert changed["B"] == baseline["B"]
    assert changed["^GSPC"] == baseline["^GSPC"]


def test_versions_verify_digest(monkeypatch):
    rows = {"A": _rows(4, "A"), "^GSPC": _rows(6, "^GSPC")}
    _patch(monkeypatch, rows)
    digests, _ = builder._price_inputs(None, None, ["A"], {})
    context = {"request": {"limits": {}}, "price_row_digests": digests}
    versions = builder._versions(None, None, context, "A")
    assert set(versions) == {row["date"] for row in rows["A"]}
    assert all(versions[row["date"]] == row for row in rows["A"])
    _patch(monkeypatch, {"A": _rows(4, "A", bump=1), "^GSPC": rows["^GSPC"]})
    with pytest.raises(DataError) as raised:
        builder._versions(None, None, context, "A")
    assert raised.value.code == "IDENTITY_CONFLICT"
    assert "IDENTITY_CONFLICT" in str(raised.value)
    _patch(monkeypatch, {"A": _rows(3, "A"), "^GSPC": rows["^GSPC"]})
    with pytest.raises(DataError) as raised:
        builder._versions(None, None, context, "A")
    assert raised.value.code == "IDENTITY_CONFLICT"


def test_identity_peak_memory_independent_of_total_rows(monkeypatch):
    ROWS = 1500

    def price_dates(repository, snapshot, ticker, limits):
        return _rows(ROWS, ticker)

    monkeypatch.setattr(sources, "price_dates", price_dates)

    def peak(tickers):
        tracemalloc.start()
        try:
            builder._price_inputs(None, None, [f"T{i}" for i in range(tickers)], {})
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    peak_10 = peak(10)
    peak_120 = peak(120)
    # The v3 inline metadata held ~1.3 KB of Python objects per row, i.e.
    # >200 MB at this scale; identity must stay bounded per ticker.
    assert peak_120 < 3 * peak_10
    assert peak_120 < 8 * 1024 * 1024

def test_registration_accepts_v3_v4_and_v5_contexts():
    assert native_registration._SIZE_PANEL_CONTEXTS == (
        "magnitude_panel_context.v3", "magnitude_panel_context.v4",
        "magnitude_panel_context.v5")


def _panel_document_v17(context):
    return {"snapshot": "snap",
        "size_panel": {"context": context, "feature_snapshot": {"snapshot_id": "feat"}},
        "size_request": {"schema_version": "magnitude_search_request.v2"},
        "size_comparison": {"schema_version": "size_causal_comparator.v1"},
        "size_panel_request": {}}


def _season_receipt(**fields):
    """Receipt shaped as season_population._receipt builds it: the id is the digest of the body."""
    receipt = {"schema_version": "season_heat_population.v1", **fields}
    receipt["receipt_id"] = content_hash({k: v for k, v in receipt.items() if k != "receipt_id"})
    return receipt


def test_size_panel_v17_v5_requires_season_heat_population_receipt_id():
    context = {"schema_version": "magnitude_panel_context.v5", "context_id": "ctx",
        "source_snapshot": "snap", "target_events": {"$size_artifact": "events"},
        "request": {"event_ids": {"$size_artifact": "ids"}}}
    assert not native_registration._size_panel_v17(_panel_document_v17(context))
    context["season_heat_population"] = {"window": {"latest_decision": "2024-01-01"}}
    assert not native_registration._size_panel_v17(_panel_document_v17(context))
    context["season_heat_population"] = {**context["season_heat_population"],
        "receipt_id": "sha256:test"}  # an id that is not the digest of its own body
    assert not native_registration._size_panel_v17(_panel_document_v17(context))
    context["season_heat_population"] = _season_receipt(
        window={"latest_decision": "2024-01-01"})
    assert native_registration._size_panel_v17(_panel_document_v17(context))
    v4 = {**context, "schema_version": "magnitude_panel_context.v4"}
    assert native_registration._size_panel_v17(_panel_document_v17(v4))
    v4.pop("season_heat_population")
    assert native_registration._size_panel_v17(_panel_document_v17(v4))


_TARGET_RECIPE = {"schema_version": "earnings_magnitude.v1", "units": "percentage_points",
    "target": "absolute_earnings_close_to_close_return",
    "price_basis": "split_neutral_non_dividend_adjusted", "estimand": "conditional_mean",
    "primary_loss": "pooled_rmse", "secondary_diagnostics": ["mae", "quantile_calibration"]}
_FOLD_POLICY = {"kind": "expanding_event_year", "group_key": "event_id",
    "fit_boundary": "min_year_start_and_evaluation_forecast_ceilings",
    "training_label_availability": "strictly_before_fit_boundary_required_later",
    "issuer_overlap": "allowed_across_time", "inner_search": "not_defined"}
_AVAILABILITY = {"basis": "reconstructed_date_bounds", "event_schedule_as_known": "unverified",
    "source_publication": "unverified"}
_EVENT_COLUMNS = ["event_id", "ticker", "event_date", "session", "date_conflict", "event_cluster_id"]
_CALENDAR_ID = content_hash("synthetic-calendar")


def _plan_event(event_id, day, pre_close):
    event = {"event_id": event_id, "ticker": "AAAA", "event_date": day, "session": "AMC",
             "population_use": "post-release selection"}
    event["forecasts"] = []
    for horizon in (5, 3, 1):
        clock = {"schema_version": "earnings_forecast_cutoff.v1", "event_date": day,
            "session": "AMC", "horizon_sessions": horizon, "pre_announcement_close": pre_close,
            "source_session_ceiling": pre_close, "calendar_id": _CALENDAR_ID}
        event["forecasts"].append({**clock, "cutoff_id": content_hash(clock)})
    return event


def _v16_size_document():
    """Synthetic v1.6 SIZE registration whose v1.3 baseline passes the sealed magnitude binding."""
    events = [_plan_event("early", "2022-06-02", "2022-05-20"),
              _plan_event("late", "2024-06-02", "2024-05-20")]
    snapshot = {"snapshot_id": "snap-v16", "calendar_version": "cal-v1",
        "table_versions": {"earnings_events": {"dataset_version_id": "dv-events"},
                           "daily_market": {"dataset_version_id": "dv-daily"}}}
    plan = {"schema_version": "magnitude_temporal_plan.v1", "snapshot_id": snapshot["snapshot_id"],
        "holdout_as_of_month": "2025-01", "random_membership_version": "canonical-event-sha256.v1",
        "rolling_membership_version": "calendar-months.v1", "training_start": "2021-01-04",
        "evaluation_years": ["2024"], "horizons": [5, 3, 1], "target_recipe": _TARGET_RECIPE,
        "fold_policy": _FOLD_POLICY, "events": events, "availability": _AVAILABILITY,
        "population_input": {"table": "earnings_events", "columns": _EVENT_COLUMNS,
            "dataset_version": "dv-events",
            "objects": [{"fragment_id": "f1", "object_id": "o1", "content_hash": content_hash("obj-1")}]},
        "calendar_input": {"table": "daily_market", "columns": ["date"], "dataset_version": "dv-daily",
            "snapshot_id": snapshot["snapshot_id"], "calendar_version": snapshot["calendar_version"],
            "objects": [{"fragment_id": "f2", "object_id": "o2", "content_hash": content_hash("obj-2")}]},
        "folds": [{"year": "2024", "training_labels_before": "2023-12-29",
                   "training_candidate_event_ids": ["early"], "evaluation_event_ids": ["late"]}]}
    plan["plan_id"] = content_hash({k: v for k, v in plan.items() if k != "plan_id"})
    request = {"as_of_month": "2024-06", "policy": {}, "event_ids": ["early", "late"],
        "horizons": [1, 3, 5], "history_start": "2020-01-02", "limits": {},
        "source_start": "2021-01-04", "metadata_limits": {}}
    context = {"schema_version": "magnitude_panel_context.v5", "source_snapshot": snapshot,
        "history_snapshot": {"snapshot_id": "hist-v16"},
        "price_history_snapshot": {"snapshot_id": "price-v16"},
        "season_heat_population": _season_receipt(
            window={"start": "2024-01-02", "end": "2024-06-20",
                "earliest_decision": "2024-06-10", "latest_decision": "2024-06-20"},
            counts={"population": 2, "in_window": 2, "retained": 2, "excluded": 0,
                "not_yet_available": 0}),
        "target_events": [{"event_id": event["event_id"]} for event in events], "request": request}
    context["context_id"] = content_hash({k: v for k, v in context.items() if k != "context_id"})
    feature = {"snapshot_id": "feat-v16"}
    history_ref = context["history_snapshot"]["snapshot_id"]
    price_ref = context["price_history_snapshot"]["snapshot_id"]
    return {"schema_version": "native_experiment_registration.v1.6",
        "spec_hash": content_hash("spec"), "execution_plan": {"folds": ["2024"],
            "price_source": "computed_moves", "seed": 0, "economic_params": {}},
        "snapshot": snapshot, "scope": "shadow",
        "source_closure": {"experiments/size_search.py": content_hash("source")},
        "environment": {"python": "3.14"}, "event_ids": ["early", "late"],
        "holdouts": {name: plan[name] for name in ("holdout_as_of_month", "random_membership_version",
            "rolling_membership_version")},
        "magnitude_request": {"training_start": plan["training_start"],
            "evaluation_years": plan["evaluation_years"]},
        "magnitude_plan": plan,
        "magnitude_execution": native_registration.magnitude_execution_document("2025-01-02"),
        "size_request": {"schema_version": "magnitude_search_request.v2",
            "feature_snapshot_id": feature["snapshot_id"], "history_snapshot_id": history_ref},
        "size_panel_request": {**request, "feature_snapshot_id": feature["snapshot_id"],
            "history_snapshot": history_ref, "price_history_snapshot": price_ref},
        "size_panel": {"context": context, "feature_snapshot": feature},
        "size_comparison": {"schema_version": "size_causal_comparator.v1"}}


def test_size_binding_v5_context_requires_season_heat_population_receipt():
    document = _v16_size_document()
    assert native_registration._size_binding(document)
    tampered = deepcopy(document)
    context = tampered["size_panel"]["context"]
    context.pop("season_heat_population")
    context["context_id"] = content_hash({k: v for k, v in context.items() if k != "context_id"})
    assert not native_registration._size_binding(tampered)


def test_size_binding_v5_context_refuses_an_arbitrary_season_receipt_id():
    """A made-up receipt_id passes only if the context_id is rehashed over it; the digest must match."""
    document = _v16_size_document()
    tampered = deepcopy(document)
    context = tampered["size_panel"]["context"]
    receipt = context["season_heat_population"]
    context["season_heat_population"] = {**receipt, "receipt_id": "sha256:" + "a" * 64}
    context["context_id"] = content_hash({k: v for k, v in context.items() if k != "context_id"})
    assert not native_registration._size_binding(tampered)
    context["season_heat_population"] = receipt  # the derived id of the same body
    context["context_id"] = content_hash({k: v for k, v in context.items() if k != "context_id"})
    assert native_registration._size_binding(tampered)
