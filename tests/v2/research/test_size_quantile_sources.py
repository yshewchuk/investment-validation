"""EXP-190 Q2a: sealed panel, dictionary and label reader. Synthetic fixtures only (no store, no network, no data/)."""
import copy
import json
import math
import types
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from engine.v2.data.errors import DataError
from engine.v2.data.errors import fail as data_fail
from engine.v2.foundation import content_hash, to_document
from engine.v2.ops.errors import OpsError
from engine.v2.research.derived_features import LOG_COLUMNS, PRODUCT_PAIRS, derived_names
from engine.v2.research.magnitude_panel_qualified_table import recipe
from experiments import size_panel
from experiments import size_quantile_sources as qsrc
from experiments.size_pnl_readers import admit_events as REAL_ADMIT_EVENTS
from experiments.size_search_binding import SCHEMA
from tests.v2.research.test_magnitude_panel_qualified import panel_data  # noqa: F401  (fixture)

HMOSX = recipe("HMOSX")["model_columns"]
DERIVED = derived_names("HMOSXLP")
COMPACT = {"synthetic": "size-panel-binding"}
PANEL_ID = content_hash(COMPACT)
SNAP = {"snapshot_id": "snap-1"}
FEAT = {"snapshot_id": "feat-1"}
TARGETS = [
    {"event_id": "e1", "event_date": "2025-01-02", "forecasts": []},
    {"event_id": "e2", "event_date": "2025-01-03", "forecasts": []},
    {"event_id": "e3", "event_date": "2024-12-30", "forecasts": []},  # not admitted
    {"event_id": "e4", "event_date": "2025-01-01", "forecasts": []},
]
CONN, STORE, SPEC, PLAN_REF = object(), object(), object(), object()
CUTOFF = "2025-01-01"


def _refusal(exc):
    return exc.value.problem.code, exc.value.problem.details.get("reason")


def _assert_message(exc, reason):
    message = exc.value.problem.message
    if reason in ("plan_malformed", "snapshot_mismatch"):  # raised by size_pnl_sources: non-empty only
        assert message
    else:
        assert message.startswith(f"{reason}: ")


def _plan():
    dictionary = recipe("HMOSXLP")
    candidate = {"candidate_id": "c1", "source_binding_id": PANEL_ID,
                 "feature_recipe": {"subset": "HMOSXLP", "dictionary_hash": dictionary["dictionary_hash"],
                                    "model_columns": list(dictionary["model_columns"])}}
    return {"plan_id": "plan-1", "candidates": [candidate]}


def _rows():
    return [{**{name: 1.0 for name in HMOSX}, "event_id": event, "horizon_sessions": horizon}
            for event in ("e1", "e2", "e3", "e4") for horizon in (1, 3)]


@pytest.fixture
def world(monkeypatch):
    st = types.SimpleNamespace(
        calls=[], panel_args=[], admit_args=[], binding_doc=COMPACT, plan=_plan(), rows=_rows(),
        admitted=["e1", "e2", "e4"], panel_error=None, resolved={"snap-1": SNAP, "feat-1": FEAT},
        document={"schema_version": SCHEMA, "snapshot": SNAP, "holdouts": {"holdout_as_of_month": "2025-01-01"},
                  "size_panel": {"context": {"context_id": "ctx-1", "target_events": copy.deepcopy(TARGETS)},
                                 "feature_snapshot": FEAT}})

    def registration(conn, store, spec):
        st.calls.append("registration")
        return types.SimpleNamespace(document=copy.deepcopy(st.document),
                                     binding_json=json.dumps({"size_panel": st.binding_doc}))

    def plan(store, reg, ref):
        st.calls.append("plan")
        return st.plan

    def snapshot(repository, snapshot_id):
        st.calls.append(("resolve", snapshot_id))
        return copy.deepcopy(st.resolved[snapshot_id])

    def admit(repository, snap, month, *, max_object_bytes):
        st.calls.append("admit")
        st.admit_args.append((repository, snap, month, max_object_bytes))
        return {"events": {"event_id": list(st.admitted)}}

    def read(repository, feature, context, columns, lineage=True):
        st.calls.append("panel")
        st.panel_args.append((tuple(columns), lineage))
        if st.panel_error is not None:
            raise st.panel_error
        return copy.deepcopy(st.rows)

    monkeypatch.setattr(qsrc, "read_native_registration", registration)
    monkeypatch.setattr(qsrc, "load_plan", plan)
    monkeypatch.setattr(qsrc, "Repository", lambda conn, store, pin_snapshots: "repo")
    monkeypatch.setattr(qsrc, "_bound_snapshot", lambda conn, snapshot_id: st.calls.append(("bound", snapshot_id)))
    monkeypatch.setattr(qsrc, "_snapshot", snapshot)
    monkeypatch.setattr(qsrc, "to_document", lambda snap: snap)
    monkeypatch.setattr(qsrc, "admit_events", admit)
    monkeypatch.setattr(qsrc._magnitude_panel_store, "read", read)
    return st


def _load():
    return qsrc.load_sealed_panel(CONN, STORE, SPEC, PLAN_REF)


def test_happy_path_filters_sorts_and_keeps_only_lead_one_admitted_rows(world):
    sealed = _load()
    assert [e["event_id"] for e in sealed["events"]] == ["e4", "e1", "e2"]
    assert sealed["excluded_events"] == 1
    assert [(r["event_id"], r["horizon_sessions"]) for r in sealed["rows"]] == [("e1", 1), ("e2", 1), ("e4", 1)]
    assert set(DERIVED) <= set(sealed["rows"][0])
    assert sealed["dictionary_hash"] == recipe("HMOSXLP")["dictionary_hash"]
    assert sealed["model_columns"] == recipe("HMOSXLP")["model_columns"]
    assert world.panel_args == [(tuple(HMOSX), False)]
    assert ("bound", "snap-1") in world.calls and ("bound", "feat-1") in world.calls
    assert world.calls.index(("bound", "snap-1")) < world.calls.index(("resolve", "snap-1"))
    assert world.calls.index(("resolve", "snap-1")) < world.calls.index("admit")
    assert world.calls.index(("bound", "feat-1")) < world.calls.index(("resolve", "feat-1"))
    assert world.calls.index(("resolve", "feat-1")) < world.calls.index("panel")


def test_none_values_stay_missing_through_the_derived_columns(world):
    log_name = f"log1p_{LOG_COLUMNS[0]}"
    assert log_name in DERIVED
    world.rows[0][LOG_COLUMNS[0]] = None  # e1 at horizon 1
    row = next(r for r in _load()["rows"] if r["event_id"] == "e1")
    assert row[LOG_COLUMNS[0]] is None and row[log_name] is None


def test_admission_runs_before_the_panel_read(world):
    _load()
    assert world.calls.index("admit") < world.calls.index("panel")


def test_binding_is_stable_and_follows_the_admitted_event_set(world):
    first = _load()["binding"]
    assert _load()["binding"] == first
    world.admitted = ["e1", "e2"]
    assert _load()["binding"] != first


def test_binding_names_plan_panel_snapshot_dictionary_and_sorted_events(world):
    expected = content_hash({"plan_id": "plan-1", "panel_id": PANEL_ID, "snapshot_id": "snap-1",
                             "dictionary_hash": recipe("HMOSXLP")["dictionary_hash"],
                             "event_ids": ["e4", "e1", "e2"]})
    assert _load()["binding"] == expected


def test_derived_columns_are_computed_from_the_source_values(world):
    values = {name: 2.0 + index for index, name in enumerate(LOG_COLUMNS)}
    world.rows[0].update(values)  # e1 at horizon 1
    row = next(r for r in _load()["rows"] if r["event_id"] == "e1")
    assert row[f"log1p_{LOG_COLUMNS[0]}"] == pytest.approx(math.log1p(values[LOG_COLUMNS[0]]))
    left, right = PRODUCT_PAIRS[0]
    assert row[f"{left}_x_{right}"] == pytest.approx(values[left] * values[right])


def test_admit_receives_repository_snapshot_month_and_default_cap(world):
    _load()
    assert world.admit_args == [("repo", SNAP, "2025-01-01", qsrc.MAX_OBJECT_BYTES)]


def test_explicit_byte_cap_is_forwarded_to_admission(world):
    qsrc.load_sealed_panel(CONN, STORE, SPEC, PLAN_REF, max_object_bytes=123)
    assert world.admit_args[0][3] == 123


def test_real_admission_refuses_a_zero_byte_cap_before_the_panel_read(world, monkeypatch):
    monkeypatch.setattr(qsrc, "admit_events", REAL_ADMIT_EVENTS)
    with pytest.raises(OpsError) as exc:
        qsrc.load_sealed_panel(CONN, STORE, SPEC, PLAN_REF, max_object_bytes=0)
    assert _refusal(exc) == ("INVALID_EXPERIMENT_SPEC", "max_object_bytes_invalid")
    assert "panel" not in world.calls


def _no_hgb_subset(st):
    st.plan["candidates"][0]["feature_recipe"]["subset"] = "HMO"


def _wrong_hash(st):
    st.plan["candidates"][0]["feature_recipe"]["dictionary_hash"] = "sha256:bad"


def _wrong_columns(st):
    st.plan["candidates"][0]["feature_recipe"]["model_columns"] = list(reversed(recipe("HMOSXLP")["model_columns"]))


def _second_candidate_wrong_hash(st):
    second = copy.deepcopy(st.plan["candidates"][0])
    second["candidate_id"] = "c2"
    second["feature_recipe"]["dictionary_hash"] = "sha256:bad"
    st.plan["candidates"].append(second)


def _snapshot_extra(st):
    st.resolved["snap-1"] = {"snapshot_id": "snap-1", "extra": True}


def _feature_extra(st):
    st.resolved["feat-1"] = {"snapshot_id": "feat-1", "extra": True}


def _no_admitted(st):
    st.admitted = []


def _no_holdouts(st):
    del st.document["holdouts"]


def _no_feature_recipe(st):
    del st.plan["candidates"][0]["feature_recipe"]


def _candidates_not_iterable(st):
    st.plan["candidates"] = None


REFUSALS = [
    pytest.param(lambda st: st.document.__setitem__("schema_version", "native_experiment_registration.v1.6"),
                 "registration_schema_unknown", id="schema_unknown"),
    pytest.param(lambda st: setattr(st, "binding_doc", {"synthetic": "other"}), "panel_binding_mismatch",
                 id="panel_binding_mismatch"),
    pytest.param(_no_hgb_subset, "dictionary_mismatch", id="dictionary_no_candidate"),
    pytest.param(_wrong_hash, "dictionary_mismatch", id="dictionary_wrong_hash"),
    pytest.param(_wrong_columns, "dictionary_mismatch", id="dictionary_wrong_columns"),
    pytest.param(_second_candidate_wrong_hash, "dictionary_mismatch", id="dictionary_second_candidate_wrong_hash"),
    pytest.param(_snapshot_extra, "snapshot_mismatch", id="snapshot_mismatch"),
    pytest.param(_feature_extra, "panel_mismatch", id="panel_feature_differs"),
    pytest.param(_no_admitted, "admission_empty", id="admission_empty"),
    pytest.param(_no_holdouts, "sealed_malformed", id="malformed_holdouts"),
    pytest.param(_no_feature_recipe, "sealed_malformed", id="malformed_candidate"),
    pytest.param(_candidates_not_iterable, "plan_malformed", id="malformed_candidates_type"),
]


@pytest.mark.parametrize("mutate, reason", REFUSALS)
def test_refusals_stop_before_the_panel_read(world, mutate, reason):
    mutate(world)
    with pytest.raises(OpsError) as exc:
        _load()
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", reason)
    _assert_message(exc, reason)
    assert "panel" not in world.calls


@pytest.mark.parametrize("code", ["HOLDOUT_ACCESS_DENIED", "CONTRACT_MISMATCH", "IDENTITY_CONFLICT"])
def test_panel_read_refusal_codes_become_panel_mismatch(world, code):
    world.panel_error = data_fail(code, "synthetic")
    with pytest.raises(OpsError) as exc:
        _load()
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "panel_mismatch")
    assert exc.value.problem.message.startswith("panel_mismatch: ")
    assert exc.value.__cause__ is None


def test_resource_data_error_propagates_unchanged(world):
    world.panel_error = data_fail("RESULT_LIMIT_EXCEEDED", "synthetic")
    with pytest.raises(DataError) as exc:
        _load()
    assert exc.value is world.panel_error


def test_refusal_classes_are_not_treated_as_malformed_documents():
    assert not issubclass(OpsError, qsrc._MALFORMED)
    assert not issubclass(DataError, qsrc._MALFORMED)


@pytest.fixture
def frame():
    out = pd.DataFrame({"event_id": ["e1"], "absolute_move_pct": [3.0]})
    out.attrs["omitted_labels"] = []
    return out


@pytest.fixture
def labels(monkeypatch, frame):
    calls = []

    def fake(repository, snapshot, *, as_of_month, labels_before, event_ids):
        calls.append({"repository": repository, "snapshot": snapshot, "as_of_month": as_of_month,
                      "labels_before": labels_before, "event_ids": event_ids})
        return frame

    monkeypatch.setattr(qsrc, "qualify_magnitude_training_targets", fake)
    return calls


SEALED = {"repository": "repo", "snapshot": "snap", "as_of_month": "2025-01-01",
          "events": [{"event_id": "e1"}, {"event_id": "e2"}]}


def test_unadmitted_label_request_refuses_before_the_reader(labels):
    with pytest.raises(OpsError) as exc:
        qsrc.training_labels(SEALED, ["e1", "e9"], CUTOFF)
    assert _refusal(exc) == ("HOLDOUT_ACCESS_DENIED", "key_not_admitted")
    assert exc.value.problem.message == "requested event ids are not admitted"
    assert "e9" not in exc.value.problem.message
    assert labels == []


@pytest.mark.parametrize("event_ids", [[], ("e1", "e1"), ["e1", 3], "e1", None])
def test_invalid_event_ids_refuse_before_the_reader(labels, event_ids):
    with pytest.raises(OpsError) as exc:
        qsrc.training_labels(SEALED, event_ids, CUTOFF)
    assert _refusal(exc) == ("INVALID_EXPERIMENT_SPEC", "event_ids_invalid")
    assert exc.value.problem.message == "event_ids must be a non-empty list of unique strings"
    assert labels == []


def test_admitted_ids_reach_the_reader_with_snapshot_month_and_cutoff(labels, frame, capsys):
    returned = qsrc.training_labels(SEALED, ("e2", "e1"), CUTOFF)
    assert labels == [{"repository": "repo", "snapshot": "snap", "as_of_month": "2025-01-01",
                       "labels_before": CUTOFF, "event_ids": ["e2", "e1"]}]
    assert returned is frame
    assert list(returned["event_id"]) == ["e1"]
    assert list(returned["absolute_move_pct"]) == [3.0]
    err = capsys.readouterr().err
    assert "[quantile training labels] start" in err
    assert "[quantile_labels] requested=2 qualified=1 omitted=0" in err


def test_module_source_never_calls_the_rederiving_readers():
    text = Path(qsrc.__file__).read_text()
    for name in ("read_search_plan", "prepare_plan", "verify_installation", "read_stored_panel",
                 "size_panel.read", "size_panel.materialize"):
        assert name not in text


def _patch_real(monkeypatch, data, panel, event_ids):
    document = {"schema_version": SCHEMA, "snapshot": to_document(data.panel_source),
                "holdouts": {"holdout_as_of_month": "2024-07-01"}, "size_panel": panel}
    monkeypatch.setattr(qsrc, "read_native_registration", lambda conn, store, spec: types.SimpleNamespace(
        document=copy.deepcopy(document), binding_json=json.dumps({"size_panel": COMPACT})))
    monkeypatch.setattr(qsrc, "load_plan", lambda store, reg, ref: _plan())
    monkeypatch.setattr(qsrc, "admit_events", lambda repository, snap, month, *, max_object_bytes: {
        "events": {"event_id": list(event_ids)}})
    monkeypatch.setattr(qsrc, "Repository", lambda conn, store, pin_snapshots: data.repository)


def test_real_stored_panel_returns_lead_one_rows_with_derived_columns(panel_data, monkeypatch):
    data = panel_data
    artifact = size_panel.materialize(data.repository, data.panel_source, **data.panel_request)
    context = artifact["context"]
    ids = [e["event_id"] for e in context["target_events"]]
    _patch_real(monkeypatch, data, {"context": context, "feature_snapshot": to_document(artifact["feature_snapshot"])},
                ids)
    sealed = qsrc.load_sealed_panel(data.conn, data.repository._store, SPEC, PLAN_REF)
    rows = sealed["rows"]
    assert len(rows) == len(context["target_events"])
    counts = Counter(r["event_id"] for r in rows)
    assert all(counts[event_id] == 1 for event_id in ids)
    assert {r["horizon_sessions"] for r in rows} == {1}
    assert set(DERIVED) <= set(rows[0])
    # Independent oracle: the public reader over the same sealed artifact, lead-1 rows only.
    stored = size_panel.read(data.repository, data.panel_source, artifact["feature_snapshot"],
                             columns=list(HMOSX), **data.panel_request)["rows"]
    oracle = {r["event_id"]: r for r in stored if r["horizon_sessions"] == 1}
    assert set(oracle) == set(ids)
    compared = 0
    for row in rows:
        for name in HMOSX:
            assert row[name] == oracle[row["event_id"]][name]
            compared += row[name] is not None
    assert compared > 0
    log_columns = [c for c in LOG_COLUMNS if c in HMOSX]
    picked = [(row, c) for row in rows for c in log_columns if row[c] is not None]
    assert picked
    row, column = picked[0]
    assert row[f"log1p_{column}"] == pytest.approx(math.log1p(row[column]))


def test_panel_read_value_error_is_sealed_malformed_without_its_message(world):
    world.panel_error = ValueError("synthetic-row-secret-42")
    with pytest.raises(OpsError) as exc:
        _load()
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "sealed_malformed")
    assert "ValueError" in exc.value.problem.message
    assert "synthetic-row-secret-42" not in exc.value.problem.message
    assert "synthetic-row-secret-42" not in str(exc.value)
    assert exc.value.__cause__ is None


def test_real_stored_panel_refuses_a_tampered_context_id(panel_data, monkeypatch):
    data = panel_data
    artifact = size_panel.materialize(data.repository, data.panel_source, **data.panel_request)
    context = {**artifact["context"], "context_id": "sha256:" + "b" * 64}
    ids = [e["event_id"] for e in context["target_events"]]
    _patch_real(monkeypatch, data, {"context": context, "feature_snapshot": to_document(artifact["feature_snapshot"])},
                ids)
    with pytest.raises(OpsError) as exc:
        qsrc.load_sealed_panel(data.conn, data.repository._store, SPEC, PLAN_REF)
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "panel_mismatch")
