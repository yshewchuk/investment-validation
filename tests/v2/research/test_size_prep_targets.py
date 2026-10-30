"""Workspace targets derivation filters the population window and plans non-strictly over a bounds record."""
import json
from copy import deepcopy

import pandas as pd
import pytest

from engine.v2.data import errors as data_errors
from engine.v2.data.repository import Repository
from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError
from engine.v2.research.eligibility_policy import eligibility_policy
from experiments import size_prep_history as hist
from experiments import size_prep_inputs as prep
from experiments import size_prep_targets as tgt
from tests.v2.research.test_size_prep_history import _pin_names
from tests.v2.research.test_size_prep_inputs import (  # noqa: F401
    _AS_OF, bounds_ws, source_root)

_ROWS = [
    ("ev-start", "AAAA", "2024-01-02"),
    ("ev-x", "AAAA", "2024-01-15"),
    ("ev-u", "AAAA", "2024-01-20"),
    ("ev-end", "AAAA", "2024-02-01"),
    ("ev-other", "BBBB", "2024-01-10"),
    ("ev-before", "AAAA", "2023-12-31"),
]
_UNIVERSE = ["ev-start", "ev-u", "ev-x"]
_DEFAULT_PLAN = {"plan_id": "plan_1", "retained": ["ev-start"], "unavailable": ["ev-u"],
                 "exclusions": [{"event_id": "ev-x", "reasons": ["embargo_ticker"]}], "before_start": []}


def _bounds(bounds_ws, source_root):
    ws, clock, _ = bounds_ws
    manifest = prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["AAAA"])
    record = prep.generate_workspace_bounds(ws, "shadow", manifest, source_root=source_root,
                                            as_of=_AS_OF, clock=clock)
    return ws, clock, record


def _prepared(bounds_ws, source_root, monkeypatch):
    ws, clock, record = _bounds(bounds_ws, source_root)
    repository = Repository(ws.conn, ws.store, pin_snapshots=True)
    snapshot = repository.resolve(record["bounds_snapshot_id"])
    monkeypatch.setattr(hist, "HISTORY_TABLES",
                        tuple(t for t in hist.HISTORY_TABLES if t in snapshot.table_versions))
    return ws, clock, record, snapshot


def _fakes(monkeypatch, rows=_ROWS, plan=None):
    calls = {"load": [], "plan": [], "bounds": []}

    def load(repository, snapshot, **kwargs):
        calls["load"].append(kwargs)
        return pd.DataFrame([{"event_id": event_id, "ticker": ticker,
                              "event_date": pd.Timestamp(day)} for event_id, ticker, day in rows])

    def plan_targets(repository, snapshot, **kwargs):
        calls["plan"].append(kwargs)
        if isinstance(plan, Exception):
            raise plan
        return deepcopy(plan if plan is not None else _DEFAULT_PLAN)

    real_bounds = hist.checked_bounds

    def checked_bounds(workspace, scope, record, metadata_limits):
        calls["bounds"].append(scope)
        return real_bounds(workspace, scope, record, metadata_limits)

    monkeypatch.setattr(tgt, "load_population", load)
    monkeypatch.setattr(tgt.eligible_history, "plan_targets", plan_targets)
    monkeypatch.setattr(hist, "checked_bounds", checked_bounds)
    return calls


def _args(clock, **over):
    kwargs = dict(tickers=["AAAA"], event_start="2024-01-01", event_end="2024-02-01",
                  as_of_month="2024-01", history_start="2020-01-01",
                  metadata_limits={k: 10**6 for k in hist.size_panel._METADATA_LIMITS}, clock=clock)
    kwargs.update(over)
    return kwargs


def _refusal(call, code, reason=None):
    with pytest.raises((OpsError, data_errors.DataError)) as raised:
        call()
    assert raised.value.code == code
    if reason is not None:
        assert raised.value.problem.details["reason"] == reason


def _pin_body(ws, name):
    bodies = []
    for path in (ws.paths.store / "objects").rglob("*"):
        if not path.is_file() or path.stat().st_size > (1 << 20):
            continue
        try:
            stored = json.loads(path.open("rb").read(1 << 20))
        except ValueError:
            continue
        if isinstance(stored, dict) and stored.get("name") == name:
            bodies.append(stored)
    assert len(bodies) == 1
    return bodies[0]["pin"]


def test_happy_path_filters_window_and_records_targets_pin(bounds_ws, source_root, monkeypatch):
    ws, clock, record, snapshot = _prepared(bounds_ws, source_root, monkeypatch)
    rows = _ROWS + [("ev-onstart", "AAAA", "2024-01-01")]
    universe = ["ev-onstart"] + _UNIVERSE
    plan = {**_DEFAULT_PLAN, "retained": ["ev-onstart", "ev-start"]}
    calls = _fakes(monkeypatch, rows=rows, plan=plan)
    doc = tgt.derive_workspace_targets(ws, "shadow", record, **_args(clock))
    assert calls["load"] == [{"as_of_month": "2024-01"}]
    kwargs = calls["plan"][0]
    assert "ev-onstart" in universe and kwargs["event_ids"] == universe
    assert kwargs["horizons"] == [1, 3, 5]
    assert kwargs["policy"] == eligibility_policy(version="grouped_ticker_embargo.v1", embargo_days=30)
    assert kwargs["as_of_month"] == "2024-01" and kwargs["history_start"] == "2020-01-01"
    assert kwargs["limits"] == hist._derive_history_limits(
        Repository(ws.conn, ws.store, pin_snapshots=True), snapshot, len(universe))
    assert doc["event_ids"] == ["ev-onstart", "ev-start"]
    assert doc["dropped"] == [{"event_id": "ev-u", "reasons": ["label_bounds_unavailable"]},
                              {"event_id": "ev-x", "reasons": ["embargo_ticker"]}]
    assert doc["schema_version"] == tgt.TARGETS_VERSION
    assert doc["scope"] == "shadow" and doc["record_id"] == record["record_id"]
    assert doc["bounds_snapshot_id"] == snapshot.snapshot_id
    assert doc["event_start"] == "2024-01-01" and doc["event_end"] == "2024-02-01"
    assert doc["tickers_sha"] == content_hash(["AAAA"])
    assert doc["universe_count"] == 4 and doc["universe_sha"] == content_hash(universe)
    assert doc["target_count"] == 2 and doc["targets_sha"] == content_hash(["ev-onstart", "ev-start"])
    assert doc["dropped_count"] == 2 and doc["dropped_sha"] == content_hash(doc["dropped"])
    assert doc["eligibility_plan_id"] == "plan_1"
    assert doc["targets_id"] == content_hash(
        {k: v for k, v in doc.items()
         if k not in ("targets_id", "event_ids", "dropped", "read_pin_ref")})
    pin = _pin_body(ws, "size-prep-targets:" + doc["targets_id"].split(":", 1)[1])
    assert pin == {k: v for k, v in doc.items() if k not in ("event_ids", "dropped", "read_pin_ref")}
    assert doc["read_pin_ref"] is not None


def test_invalid_requests_refuse_before_any_read_or_pin(bounds_ws, source_root, monkeypatch):
    ws, clock, record, _ = _prepared(bounds_ws, source_root, monkeypatch)
    calls = _fakes(monkeypatch)
    pins = _pin_names(ws)

    def call(**over):
        scope = over.pop("scope", "shadow")
        return lambda: tgt.derive_workspace_targets(ws, scope, record, **_args(clock, **over))

    for over in ({"tickers": ["BBBB", "AAAA"]}, {"tickers": ["AAAA", "AAAA"]}, {"tickers": []},
                 {"tickers": [""]}, {"tickers": [" AAAA"]}, {"tickers": ["AAAA "]}, {"tickers": "AAAA"},
                 {"tickers": [1]},
                 {"scope": 123}, {"as_of_month": None}, {"history_start": 5},
                 {"metadata_limits": []}, {"metadata_limits": None},
                 {"event_start": "2024-1-1"}, {"event_end": "2024-02-30"},
                 {"event_start": "2024-02-01"}, {"event_start": "2024-02-02"}):
        _refusal(call(**over), "INVALID_REQUEST")
    assert calls == {"load": [], "plan": [], "bounds": []}
    assert _pin_names(ws) == pins


def test_every_request_field_is_validated_before_any_read(bounds_ws, source_root, monkeypatch):
    ws, clock, record, _ = _prepared(bounds_ws, source_root, monkeypatch)
    calls = _fakes(monkeypatch)
    pins = _pin_names(ws)

    def call(**over):
        return lambda: tgt.derive_workspace_targets(ws, "shadow", record, **_args(clock, **over))

    for over, code in (({"tickers": [[]]}, "INVALID_REQUEST"),
                       ({"tickers": [1, "A"]}, "INVALID_REQUEST"),
                       ({"metadata_limits": {}}, "CONTRACT_MISMATCH"),
                       ({"as_of_month": "bad"}, "INVALID_REQUEST"),
                       ({"history_start": "bad"}, "INVALID_REQUEST")):
        _refusal(call(**over), code)
    assert calls == {"load": [], "plan": [], "bounds": []}
    assert _pin_names(ws) == pins


def test_empty_universe_refuses_no_targets_without_a_pin(bounds_ws, source_root, monkeypatch):
    ws, clock, record, _ = _prepared(bounds_ws, source_root, monkeypatch)
    calls = _fakes(monkeypatch, rows=[("ev-end", "AAAA", "2024-02-01"), ("ev-other", "BBBB", "2024-01-10")])
    pins = _pin_names(ws)
    _refusal(lambda: tgt.derive_workspace_targets(ws, "shadow", record, **_args(clock)),
             "PUBLICATION_REFUSED", "no_targets")
    assert calls == {"load": [{"as_of_month": "2024-01"}], "plan": [], "bounds": ["shadow"]}
    assert _pin_names(ws) == pins


def test_all_dropped_refuses_no_targets_without_a_pin(bounds_ws, source_root, monkeypatch):
    ws, clock, record, _ = _prepared(bounds_ws, source_root, monkeypatch)
    calls = _fakes(monkeypatch, plan={"plan_id": "plan_1", "retained": [],
                                      "unavailable": list(_UNIVERSE), "exclusions": [],
                                      "before_start": []})
    pins = _pin_names(ws)
    _refusal(lambda: tgt.derive_workspace_targets(ws, "shadow", record, **_args(clock)),
             "PUBLICATION_REFUSED", "no_targets")
    assert len(calls["plan"]) == 1
    assert _pin_names(ws) == pins


def test_unaccounted_universe_event_refuses_plan_incomplete(bounds_ws, source_root, monkeypatch):
    ws, clock, record, _ = _prepared(bounds_ws, source_root, monkeypatch)
    _fakes(monkeypatch, plan={"plan_id": "plan_1", "retained": ["ev-start"],
                              "unavailable": [], "exclusions": [], "before_start": []})
    pins = _pin_names(ws)
    _refusal(lambda: tgt.derive_workspace_targets(ws, "shadow", record, **_args(clock)),
             "PUBLICATION_REFUSED", "plan_incomplete")
    assert _pin_names(ws) == pins


def test_moved_head_and_foreign_record_refuse_before_reads(bounds_ws, source_root, monkeypatch):
    ws, clock, record, _ = _prepared(bounds_ws, source_root, monkeypatch)
    calls = _fakes(monkeypatch)
    pins = _pin_names(ws)
    with ws.conn:
        ws.conn.execute("UPDATE data_snapshot_heads SET snapshot_id = ?, generation = generation + 1 "
                        "WHERE scope = 'shadow'", (ws.snapshot_id,))
    _refusal(lambda: tgt.derive_workspace_targets(ws, "shadow", record, **_args(clock)),
             "WORKSPACE_REFUSED", "head_not_bounds")
    tampered = deepcopy(record)
    tampered["record_id"] = "sha256:" + "0" * 64
    _refusal(lambda: tgt.derive_workspace_targets(ws, "shadow", tampered, **_args(clock)),
             "PUBLICATION_REFUSED", "record_invalid")
    assert calls == {"load": [], "plan": [], "bounds": ["shadow", "shadow"]}
    assert _pin_names(ws) == pins


def test_before_start_universe_event_is_dropped_and_call_succeeds(bounds_ws, source_root, monkeypatch):
    ws, clock, record, _ = _prepared(bounds_ws, source_root, monkeypatch)
    calls = _fakes(monkeypatch, plan={"plan_id": "plan_1", "retained": ["ev-start"],
                                      "unavailable": ["ev-u"], "exclusions": [],
                                      "before_start": ["ev-x"]})
    doc = tgt.derive_workspace_targets(ws, "shadow", record, **_args(clock))
    assert doc["event_ids"] == ["ev-start"]
    assert doc["dropped"] == [{"event_id": "ev-u", "reasons": ["label_bounds_unavailable"]},
                              {"event_id": "ev-x", "reasons": ["before_history_start"]}]
    assert doc["universe_count"] == 3 and doc["target_count"] == 1 and doc["dropped_count"] == 2
    assert calls["plan"] and _pin_body(ws, "size-prep-targets:" + doc["targets_id"].split(":", 1)[1])


def test_before_start_id_also_excluded_drops_as_before_history_start(bounds_ws, source_root, monkeypatch):
    ws, clock, record, _ = _prepared(bounds_ws, source_root, monkeypatch)
    _fakes(monkeypatch, plan={"plan_id": "plan_1", "retained": ["ev-start"],
                              "unavailable": ["ev-u"],
                              "exclusions": [{"event_id": "ev-x", "reasons": ["embargo_ticker"]}],
                              "before_start": ["ev-x"]})
    doc = tgt.derive_workspace_targets(ws, "shadow", record, **_args(clock))
    assert doc["event_ids"] == ["ev-start"]
    assert doc["dropped"] == [{"event_id": "ev-u", "reasons": ["label_bounds_unavailable"]},
                              {"event_id": "ev-x", "reasons": ["before_history_start"]}]


def test_all_early_plan_refuses_no_targets_without_a_pin(bounds_ws, source_root, monkeypatch):
    ws, clock, record, _ = _prepared(bounds_ws, source_root, monkeypatch)
    calls = _fakes(monkeypatch, plan={"plan_id": None, "retained": [], "unavailable": [],
                                      "exclusions": [], "before_start": list(_UNIVERSE)})
    pins = _pin_names(ws)
    _refusal(lambda: tgt.derive_workspace_targets(ws, "shadow", record, **_args(clock)),
             "PUBLICATION_REFUSED", "no_targets")
    assert len(calls["plan"]) == 1
    assert _pin_names(ws) == pins


def test_plan_producer_refusal_propagates_without_a_pin(bounds_ws, source_root, monkeypatch):
    ws, clock, record, _ = _prepared(bounds_ws, source_root, monkeypatch)
    calls = _fakes(monkeypatch, plan=data_errors.fail("HOLDOUT_ACCESS_DENIED", "x"))
    pins = _pin_names(ws)
    _refusal(lambda: tgt.derive_workspace_targets(ws, "shadow", record, **_args(clock)),
             "HOLDOUT_ACCESS_DENIED")
    assert len(calls["plan"]) == 1
    assert _pin_names(ws) == pins
