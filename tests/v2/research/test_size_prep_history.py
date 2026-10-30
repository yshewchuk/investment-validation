"""Workspace panel preparation composes eligibility and panel producers over a bounds record."""
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from engine.v2.data import errors as data_errors
from engine.v2.data.repository import Repository
from engine.v2.foundation import canonical_json, content_hash
from engine.v2.ops.errors import OpsError, fail
from experiments import eligible_history, size_panel
from experiments import size_prep_history as hist
from experiments import size_prep_inputs as prep
from tests.v2.research.test_size_prep_inputs import (  # noqa: F401
    _AS_OF, _pin_names, _small_bar, bounds_ws, source_root)


def _bounds(bounds_ws, source_root):
    ws, clock, _ = bounds_ws
    manifest = prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["AAAA"])
    record = prep.generate_workspace_bounds(ws, "shadow", manifest, source_root=source_root,
                                            as_of=_AS_OF, clock=clock)
    return ws, clock, record


def _fake(monkeypatch):
    calls = {"history": [], "panel": []}

    def history(repository, snapshot, **kwargs):
        calls["history"].append(kwargs)
        return {"feature_snapshot": SimpleNamespace(snapshot_id="snap_hist")}

    def panel(repository, snapshot, **kwargs):
        calls["panel"].append(kwargs)
        return {"feature_snapshot": SimpleNamespace(snapshot_id="snap_feat"),
                "context": {"eligibility_plan": {"plan_id": "plan_x"}}}

    monkeypatch.setattr(hist.eligible_history, "materialize", history)
    monkeypatch.setattr(hist.size_panel, "materialize", panel)
    return calls


def _args(clock, **over):
    kwargs = dict(event_ids=["event-b", "event-a"], as_of_month="2024-01",
                  history_start="2020-01-01", source_start="2021-01-01",
                  metadata_limits={k: 1 for k in hist.size_panel._METADATA_LIMITS}, clock=clock)
    kwargs.update(over)
    return kwargs


def _refusal(call, code, reason=None):
    with pytest.raises(OpsError) as raised:
        call()
    assert raised.value.code == code
    if reason is not None:
        assert raised.value.problem.details["reason"] == reason


def _repo_record(bounds_ws, source_root):
    ws, clock, record = _bounds(bounds_ws, source_root)
    repository = Repository(ws.conn, ws.store, pin_snapshots=True)
    return ws, clock, record, repository, repository.resolve(record["bounds_snapshot_id"])


def _expected_limits(ws, snapshot_id, tables, event_count, margin):
    placeholders = ",".join("?" for _ in tables)
    row = ws.conn.execute(
        "SELECT COALESCE(SUM(f.row_count),0) AS rows, COUNT(*) AS frags, "
        "COALESCE(SUM(o.byte_size),0) AS total, COALESCE(MAX(o.byte_size),0) AS biggest "
        "FROM data_snapshot_tables t "
        "JOIN data_version_fragments v ON v.dataset_version_id = t.dataset_version_id "
        "JOIN data_fragments f ON f.fragment_id = v.fragment_id "
        "JOIN data_objects o ON o.object_id = f.object_id "
        "WHERE t.snapshot_id = ? AND t.table_name IN (" + placeholders + ")",
        (snapshot_id, *tables)).fetchone()
    limits = {"max_source_rows": row["rows"] * margin, "max_fragments": row["frags"] * margin,
              "max_object_bytes": row["biggest"] * margin, "max_total_bytes": row["total"] * margin,
              "max_output_rows": event_count * 3 * margin}
    assert sum(limits.values()) > 0
    return limits


def test_happy_path_binds_history_panel_limits_and_read_pin(bounds_ws, source_root, monkeypatch):
    ws, clock, record, repository, snapshot = _repo_record(bounds_ws, source_root)
    calls = _fake(monkeypatch)
    monkeypatch.setattr(hist, "HISTORY_TABLES",
                        tuple(t for t in hist.HISTORY_TABLES if t in snapshot.table_versions))
    ids = ["event-b", "event-a"]
    doc = hist.prepare_workspace_panel(ws, "shadow", record, **_args(clock, event_ids=ids))
    assert doc["read_pin_ref"] is not None
    assert doc["bounds_snapshot_id"] == record["bounds_snapshot_id"]
    assert doc["history_snapshot_id"] == "snap_hist"
    assert doc["feature_snapshot_id"] == "snap_feat"
    assert doc["eligibility_plan_id"] == "plan_x"
    assert doc["events_sha"] == content_hash(sorted(ids))
    history = calls["history"][0]
    assert history["horizons"] == [1, 3, 5] and history["policy"]["embargo_days"] == 30
    panel = calls["panel"][0]
    assert panel["history_snapshot"].snapshot_id == "snap_hist"
    assert panel["price_history_snapshot"].snapshot_id == record["bounds_snapshot_id"]
    assert panel["source_start"] == "2021-01-01" and panel["metadata_limits"] == {
        k: 1 for k in hist.size_panel._METADATA_LIMITS}
    expected = _expected_limits(ws, snapshot.snapshot_id, hist.HISTORY_TABLES, len(ids), 2)
    assert panel["limits"] == expected
    name = "size-prep-panel:" + doc["panel_id"].split(":", 1)[1]
    bodies = []
    for path in (ws.paths.store / "objects").rglob("*"):
        if not path.is_file():
            continue
        try:
            stored = json.loads(path.read_bytes())
        except ValueError:
            continue
        if isinstance(stored, dict) and stored.get("name") == name:
            bodies.append(stored)
    assert len(bodies) == 1
    pin = bodies[0]["pin"]
    assert pin == {k: v for k, v in doc.items() if k != "read_pin_ref"}
    assert pin["limits"] == expected
    assert pin["record_id"] == record["record_id"]


def test_limits_scale_with_margin_and_event_count(bounds_ws, source_root, monkeypatch):
    ws, _, _, repository, snapshot = _repo_record(bounds_ws, source_root)
    present = tuple(t for t in hist.HISTORY_TABLES if t in snapshot.table_versions)
    monkeypatch.setattr(hist, "HISTORY_TABLES", present)
    two = hist.derive_history_limits(repository, snapshot, 10, margin=2)
    one = hist.derive_history_limits(repository, snapshot, 10, margin=1)
    assert two == _expected_limits(ws, snapshot.snapshot_id, present, 10, 2)
    assert one == _expected_limits(ws, snapshot.snapshot_id, present, 10, 1)
    assert two["max_output_rows"] == 60
    assert one["max_source_rows"] * 2 == two["max_source_rows"]
    for kwargs in ({"event_count": 0}, {"event_count": 10, "margin": 0}, {"event_count": True}):
        _refusal(lambda: hist.derive_history_limits(repository, snapshot, **kwargs),
                 "PUBLICATION_REFUSED", "limits_unavailable")
    monkeypatch.undo()
    stripped = SimpleNamespace(table_versions={
        name: version for name, version in snapshot.table_versions.items() if name != "price_history"})
    _refusal(lambda: hist.derive_history_limits(repository, stripped, 1),
             "PUBLICATION_REFUSED", "limits_unavailable")


def test_record_refusals_happen_before_producers_or_writes(bounds_ws, source_root, monkeypatch):
    ws, clock, record, _, _ = _repo_record(bounds_ws, source_root)
    calls = _fake(monkeypatch)
    pins = _pin_names(ws)
    path = (ws.paths.root / "size_prep"
            / ("provenance-" + record["record_id"].split(":", 1)[1] + ".json"))
    tampered_id = deepcopy(record)
    tampered_id["record_id"] = "sha256:" + "0" * 64
    changed = deepcopy(record)
    changed["entry_count"] += 1
    changed["record_id"] = content_hash({k: v for k, v in changed.items() if k != "record_id"})
    wrong_label = deepcopy(record)
    wrong_label["label"] = "other"
    other_source = deepcopy(record)
    other_source["source_snapshot_id"] = "sha256:" + "1" * 64
    other_source["record_id"] = content_hash({k: v for k, v in other_source.items() if k != "record_id"})
    for bad in (tampered_id, changed, wrong_label, other_source, "not a record"):
        _refusal(lambda: hist.prepare_workspace_panel(ws, "shadow", bad, **_args(clock)),
                 "PUBLICATION_REFUSED", "record_invalid")
    path.unlink()
    _refusal(lambda: hist.prepare_workspace_panel(ws, "shadow", record, **_args(clock)),
             "PUBLICATION_REFUSED", "record_invalid")
    assert calls == {"history": [], "panel": []}
    assert _pin_names(ws) == pins


def test_head_must_be_the_bounds_snapshot(bounds_ws, source_root, monkeypatch):
    ws, clock, record, _, _ = _repo_record(bounds_ws, source_root)
    calls = _fake(monkeypatch)
    _refusal(lambda: hist.prepare_workspace_panel(ws, "smoke", record, **_args(clock)),
             "WORKSPACE_REFUSED", "head_not_bounds")
    assert calls == {"history": [], "panel": []}


def test_moved_head_is_refused(bounds_ws, source_root, monkeypatch):
    ws, clock, record, _, snapshot = _repo_record(bounds_ws, source_root)
    calls = _fake(monkeypatch)
    monkeypatch.setattr(hist, "HISTORY_TABLES",
                        tuple(t for t in hist.HISTORY_TABLES if t in snapshot.table_versions))
    with ws.conn:
        ws.conn.execute("UPDATE data_snapshot_heads SET snapshot_id = ?, generation = generation + 1 "
                        "WHERE scope = 'shadow'", (ws.snapshot_id,))
    _refusal(lambda: hist.prepare_workspace_panel(ws, "shadow", record, **_args(clock)),
             "WORKSPACE_REFUSED", "head_not_bounds")
    assert calls["history"] == []


def test_producer_error_propagates_unchanged_without_a_panel_pin(bounds_ws, source_root, monkeypatch):
    ws, clock, record, _, snapshot = _repo_record(bounds_ws, source_root)
    calls = _fake(monkeypatch)
    monkeypatch.setattr(hist, "HISTORY_TABLES",
                        tuple(t for t in hist.HISTORY_TABLES if t in snapshot.table_versions))

    def explode(repository, snapshot, **kwargs):
        calls["panel"].append(kwargs)
        raise fail("HOLDOUT_ACCESS_DENIED", "x")

    monkeypatch.setattr(hist.size_panel, "materialize", explode)
    _refusal(lambda: hist.prepare_workspace_panel(ws, "shadow", record, **_args(clock)),
             "HOLDOUT_ACCESS_DENIED")
    assert len(calls["history"]) == 1
    assert not any(name.startswith("size-prep-panel:") for name in _pin_names(ws))


def test_malformed_inputs_are_refused_before_producers(bounds_ws, source_root, monkeypatch):
    ws, clock, record, _, _ = _repo_record(bounds_ws, source_root)
    calls = _fake(monkeypatch)

    def call(**over):
        scope = over.pop("scope", "shadow")
        return hist.prepare_workspace_panel(ws, scope, record, **_args(clock, **over))

    for over in ({"event_ids": []}, {"event_ids": ["a", "a"]}, {"event_ids": [1]},
                 {"metadata_limits": []}, {"scope": 123}):
        _refusal(lambda over=over: call(**over), "INVALID_REQUEST")
    assert calls == {"history": [], "panel": []}


def test_panel_only_inputs_are_validated_before_history(bounds_ws, source_root, monkeypatch):
    ws, clock, record, _, snapshot = _repo_record(bounds_ws, source_root)
    calls = _fake(monkeypatch)
    monkeypatch.setattr(hist, "HISTORY_TABLES",
                        tuple(t for t in hist.HISTORY_TABLES if t in snapshot.table_versions))
    for over in ({"metadata_limits": {"max_x": 1}}, {"source_start": "2021-1-1"},
                 {"source_start": "not-a-date"}):
        with pytest.raises(data_errors.DataError) as raised:
            hist.prepare_workspace_panel(ws, "shadow", record, **_args(clock, **over))
        assert raised.value.code == "CONTRACT_MISMATCH"
        assert calls["history"] == []


def test_scope_admission_is_complete_before_any_read(bounds_ws, source_root, monkeypatch):
    ws, clock, record, _, _ = _repo_record(bounds_ws, source_root)
    calls = _fake(monkeypatch)
    for scope in ("\ud800", "", " shadow", "a\x00b"):
        _refusal(lambda scope=scope: hist.prepare_workspace_panel(ws, scope, record, **_args(clock)),
                 "WORKSPACE_REFUSED", "scope_invalid")
    assert calls["history"] == []


def test_provenance_file_that_is_oversized_or_a_symlink_is_refused(bounds_ws, source_root, monkeypatch):
    ws, clock, record, _, snapshot = _repo_record(bounds_ws, source_root)
    calls = _fake(monkeypatch)
    monkeypatch.setattr(hist, "HISTORY_TABLES",
                        tuple(t for t in hist.HISTORY_TABLES if t in snapshot.table_versions))
    real_read = hist._read_regular
    recorded = []

    def recording_read(path, limit, *args, **kwargs):
        recorded.append(limit)
        return real_read(path, limit, *args, **kwargs)

    monkeypatch.setattr(hist, "_read_regular", recording_read)
    path = (ws.paths.root / "size_prep"
            / ("provenance-" + record["record_id"].split(":", 1)[1] + ".json"))
    with path.open("a") as handle:
        handle.write(" " * (1 << 20))
    _refusal(lambda: hist.prepare_workspace_panel(ws, "shadow", record, **_args(clock)),
             "PUBLICATION_REFUSED", "record_invalid")
    assert recorded == [len(canonical_json(record).encode("utf-8"))]
    assert calls["history"] == []
    path.unlink()
    elsewhere = ws.paths.root / "elsewhere.json"
    elsewhere.write_text(canonical_json(record))
    path.symlink_to(elsewhere)
    _refusal(lambda: hist.prepare_workspace_panel(ws, "shadow", record, **_args(clock)),
             "PUBLICATION_REFUSED", "record_invalid")
    assert calls["history"] == []


def test_derived_limits_and_request_shape_are_accepted_by_the_real_producer_validators(
        bounds_ws, source_root, monkeypatch):
    ws, clock, record, _, snapshot = _repo_record(bounds_ws, source_root)
    calls = _fake(monkeypatch)
    monkeypatch.setattr(hist, "HISTORY_TABLES",
                        tuple(t for t in hist.HISTORY_TABLES if t in snapshot.table_versions))
    hist.prepare_workspace_panel(ws, "shadow", record, **_args(clock))
    kw = calls["history"][0]
    pk = calls["panel"][0]
    eligible_history.validate_request(kw)
    assert set(pk) == size_panel._HISTORY_KEYS | size_panel._PANEL_KEYS
    assert {k: pk[k] for k in size_panel._HISTORY_KEYS} == kw
    bad = deepcopy(kw)
    bad["limits"] = dict(bad["limits"])
    bad["limits"]["max_source_rows"] = 0
    with pytest.raises(Exception) as raised:
        eligible_history.validate_request(bad)
    assert raised.value.code == "CONTRACT_MISMATCH"
