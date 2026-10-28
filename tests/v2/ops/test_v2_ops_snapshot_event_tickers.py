"""``snapshot_event_tickers`` (issue #471): the scheduled run's own event-ticker universe.

Every seam the function touches is monkeypatched -- no catalog, no artifact store, no
network. The requests the fake scan returns carry only the ``.ticker`` attribute the
function reads. ``_resolve_scope_snapshot`` is patched for the scan, empty-window and
``DataError``-mapping cases; only the moved-head and unreadable-artifact refusals reach
the real helper, with its head resolve (and, for the moved-head case, its ``SnapshotRef``
load) patched so that refusal fires before ``Repository`` or the scan is ever built.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from engine.v2.data.errors import fail as data_fail  # noqa: E402
from engine.v2.foundation import ArtifactError  # noqa: E402
from engine.v2.ops import snapshot_planning  # noqa: E402
from engine.v2.ops.errors import OpsError  # noqa: E402

_AS_OF = "2026-09-25"
_HORIZON = snapshot_planning.GENERATED_HORIZON_DAYS
_SNAPSHOT = SimpleNamespace(snapshot_id="snap_pinned", manifest_hash="m1",
                            calendar_version="cal.v1")
_CLOCK = object()


def _request(ticker):
    return SimpleNamespace(ticker=ticker)


class _DummyConn:
    def close(self):
        return None


def _explode(*args, **kwargs):
    raise AssertionError("must not be reached on this branch")


def _pin_head(monkeypatch, resolve=None):
    monkeypatch.setattr(snapshot_planning, "_resolve_scope_snapshot",
                        resolve or (lambda *args, **kwargs: _SNAPSHOT))
    repo_calls = []
    sentinel_repo = object()
    monkeypatch.setattr(
        snapshot_planning, "Repository",
        lambda *a, **k: repo_calls.append((a, k)) or sentinel_repo)


@pytest.mark.parametrize("request_tickers, expected", [
    (["BBB", "AAA", "BBB"], ("AAA", "BBB")),  # sorted, de-duplicated
    (["ZZZ", "CCC"], ("CCC", "ZZZ")),
])
def test_snapshot_event_tickers_scans_once_and_returns_the_sorted_distinct_tickers(
        monkeypatch, request_tickers, expected):
    # One head resolve (clock + expected id), EXACTLY one scan with tickers=None (never a
    # restricted watchlist), the fixed 35-day horizon, and the result de-duplicated and sorted.
    resolved, resolve_args = [], []
    conn, store = object(), object()  # sentinels: identity-checked against the helper's own args

    def fake_resolve(*args, **kwargs):
        resolve_args.append(args)
        resolved.append(kwargs)
        return _SNAPSHOT

    monkeypatch.setattr(snapshot_planning, "_resolve_scope_snapshot", fake_resolve)
    repo_calls, scan_repo = [], []
    sentinel_repo = object()
    monkeypatch.setattr(
        snapshot_planning, "Repository",
        lambda *a, **k: repo_calls.append((a, k)) or sentinel_repo)
    scan_calls = []

    def fake_scan(repository, snapshot, *, as_of, horizon_days, tickers):
        scan_repo.append(repository)
        scan_calls.append({"snapshot": snapshot.snapshot_id, "as_of": as_of,
                           "horizon_days": horizon_days, "tickers": tickers})
        return [_request(ticker) for ticker in request_tickers]

    monkeypatch.setattr(snapshot_planning, "scan_forward_board_requests", fake_scan)

    result = snapshot_planning.snapshot_event_tickers(
        conn, store, "shadow", as_of=_AS_OF, clock=_CLOCK,
        expected_snapshot_id="snap_pinned")

    assert result == expected
    assert scan_calls == [{"snapshot": "snap_pinned", "as_of": _AS_OF,
                           "horizon_days": 35, "tickers": None}]  # exactly one scan
    assert scan_calls[0]["horizon_days"] == 35  # the fixed 35-day horizon, literally
    assert scan_calls[0]["horizon_days"] == _HORIZON
    assert scan_calls[0]["as_of"] == _AS_OF  # the scan's own window start
    assert resolved == [{"clock": _CLOCK, "expected_snapshot_id": "snap_pinned"}]
    assert resolve_args[0][0] is conn  # the caller's own connection, unchanged
    assert resolve_args[0][1] is store  # the caller's own store, unchanged
    assert resolve_args[0][2] == "shadow"  # the head was resolved for this scope
    assert repo_calls == [((conn, store), {})]
    assert scan_repo[0] is sentinel_repo


def test_an_empty_window_is_a_typed_invalid_request_with_as_of_and_horizon(monkeypatch):
    _pin_head(monkeypatch)
    monkeypatch.setattr(snapshot_planning, "scan_forward_board_requests",
                        lambda *args, **kwargs: ())

    with pytest.raises(OpsError) as raised:
        snapshot_planning.snapshot_event_tickers(
            _DummyConn(), object(), "shadow", as_of=_AS_OF, clock=_CLOCK)

    assert raised.value.code == "INVALID_REQUEST"
    assert raised.value.problem.details["as_of"] == _AS_OF
    assert raised.value.problem.details["horizon_days"] == _HORIZON


def test_a_data_error_from_the_scan_is_input_changed_with_its_data_code(monkeypatch):
    # The reader's DataError surfaces as INPUT_CHANGED with details.data_code --
    # the same mapping generated_population already makes for its own scan.
    _pin_head(monkeypatch)

    def explode(*args, **kwargs):
        raise data_fail("CONTRACT_MISMATCH", "synthetic reader refusal")

    monkeypatch.setattr(snapshot_planning, "scan_forward_board_requests", explode)

    with pytest.raises(OpsError) as raised:
        snapshot_planning.snapshot_event_tickers(
            _DummyConn(), object(), "shadow", as_of=_AS_OF, clock=_CLOCK)

    assert raised.value.code == "INPUT_CHANGED"
    assert raised.value.problem.details["data_code"] == "CONTRACT_MISMATCH"


def test_a_data_error_from_the_head_resolve_is_the_same_input_changed_mapping(monkeypatch):
    # A head resolve that already refuses (no committed head, an unreadable events
    # table, ...) maps to INPUT_CHANGED in each caller's own except-DataError clause,
    # and neither reaches the scan.
    def missing_head(*args, **kwargs):
        raise data_fail("SNAPSHOT_NOT_READY", "this scope has no committed head")

    _pin_head(monkeypatch, resolve=missing_head)
    monkeypatch.setattr(snapshot_planning, "scan_forward_board_requests", _explode)

    with pytest.raises(OpsError) as raised:
        snapshot_planning.snapshot_event_tickers(
            _DummyConn(), object(), "shadow", as_of=_AS_OF, clock=_CLOCK)

    assert raised.value.code == "INPUT_CHANGED"
    assert raised.value.problem.details["data_code"] == "SNAPSHOT_NOT_READY"


def test_a_head_that_moves_vs_the_expected_snapshot_id_is_refused_before_the_scan(monkeypatch):
    # The shared helper's own refusal fires before Repository/scan are even built --
    # generated_population reaches this same refusal through the same helper.
    head_ref = SimpleNamespace(byte_size=1)
    monkeypatch.setattr(snapshot_planning, "resolve_snapshot_head",
                        lambda *args, **kwargs: head_ref)
    monkeypatch.setattr(snapshot_planning, "from_document",
                        lambda cls, document: SimpleNamespace(snapshot_id="snap_other"))
    monkeypatch.setattr(snapshot_planning, "Repository", _explode)
    monkeypatch.setattr(snapshot_planning, "scan_forward_board_requests", _explode)
    store = SimpleNamespace(read_verified=lambda head: b"{}")

    with pytest.raises(OpsError) as raised:
        snapshot_planning.snapshot_event_tickers(
            _DummyConn(), store, "shadow", as_of=_AS_OF, clock=_CLOCK,
            expected_snapshot_id="snap_pinned")

    assert raised.value.code == "INPUT_CHANGED"
    assert "moved" in raised.value.problem.message
    assert not raised.value.problem.details


@pytest.mark.parametrize("payload, artifact_error, artifact_code", [
    (ArtifactError("MISSING", "gone"), "ArtifactError", "MISSING"),
    (b"not json", "JSONDecodeError", None),
    (b"{}", "DocumentError", "MISSING_FIELD"),
])
def test_an_unreadable_head_artifact_is_input_changed_with_the_artifact_error(
        monkeypatch, payload, artifact_error, artifact_code):
    # The REAL _resolve_scope_snapshot path through snapshot_event_tickers: only the head
    # resolve is patched, the store's own read/decode refuses, and the scan never runs.
    head_ref = SimpleNamespace(byte_size=1)
    monkeypatch.setattr(snapshot_planning, "resolve_snapshot_head",
                        lambda *args, **kwargs: head_ref)
    monkeypatch.setattr(snapshot_planning, "scan_forward_board_requests", _explode)

    def read_verified(head):
        assert head is head_ref
        if isinstance(payload, Exception):
            raise payload
        return payload

    store = SimpleNamespace(read_verified=read_verified)

    with pytest.raises(OpsError) as raised:
        snapshot_planning.snapshot_event_tickers(
            _DummyConn(), store, "shadow", as_of=_AS_OF, clock=_CLOCK)

    assert raised.value.code == "INPUT_CHANGED"
    assert raised.value.problem.details["artifact_error"] == artifact_error
    assert raised.value.problem.details["artifact_code"] == artifact_code


def test_success_path_does_not_consult_the_carried_set(monkeypatch):
    # Tripwire: snapshot_event_tickers builds its universe from a single scan; if a
    # change makes it fall back to resolve_carried_set, this fails loudly.
    def consulted(*args, **kwargs):
        raise AssertionError("carried set must not be consulted")

    monkeypatch.setattr(snapshot_planning, "resolve_carried_set", consulted)
    _pin_head(monkeypatch)
    monkeypatch.setattr(snapshot_planning, "scan_forward_board_requests",
                        lambda *args, **kwargs: [_request("AAA")])

    result = snapshot_planning.snapshot_event_tickers(
        _DummyConn(), object(), "shadow", as_of=_AS_OF, clock=_CLOCK)

    assert result == ("AAA",)
