"""Direct tests for ``computed_moves_store.py``'s helpers, and an end-to-end
test of ``run_computed_moves_refresh`` itself: a complete unit, a
same-session cached rerun that re-fetches nothing, and a provider failure
mapped to its typed code. The store is still unwired to the nightly job graph
(see ``engine/v2/ops/ARCHITECTURE.md``'s "not yet wired" note) -- these tests
call it directly, the same way its eventual worker will."""
from __future__ import annotations

import pandas as pd
import pytest

from engine.v2.contracts import KeyPredicate
from engine.v2.data.errors import DataError, fail as data_fail
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactStore, canonical_json
from engine.v2.ops import computed_moves_store
from engine.v2.ops.catalog import transaction
from engine.v2.ops.computed_moves_store import _capture_id_for, _fence_check_for
from engine.v2.ops.errors import OpsError
from engine.v2.ops.incremental_data import RefreshParameters, RefreshUnit
from engine.v2.ops.pinned_partition_reader import RetainedRowCount
from tests.data_scan_support import (
    commit_tables, contract_for, contract_ref_for, hand_built_record, publish_and_inspect,
    table_from_rows)
from tests.ops_support import FakeClock, catalog, enqueue_claim


def _unit(ticker: str, day: str) -> RefreshUnit:
    return RefreshUnit(request_id=f"computed_moves:{ticker}:{day}",
                       table_name="computed_moves", partition_key=ticker,
                       expected_keys=(ticker,))


def test_capture_id_is_stable_for_the_same_unit_and_differs_for_a_different_one():
    """Same (ticker, as_of) -> same capture_id, called twice -- independent of
    wall-clock time. This is the bug the ``request_id``-derived id fixes: the
    old code folded in the run's own ``created_at``, so a same-day retry never
    matched the already-logged capture and ``_insert_captures`` would have
    double-logged it."""
    unit = _unit("ABCD", "2026-09-24")
    first = _capture_id_for(unit)
    second = _capture_id_for(unit)
    assert first == second
    assert first.startswith("capture_")
    other_ticker = _capture_id_for(_unit("WXYZ", "2026-09-24"))
    other_day = _capture_id_for(_unit("ABCD", "2026-09-25"))
    assert other_ticker != first
    assert other_day != first


def test_fence_check_for_matches_the_real_verify_fence_and_keeps_the_lease_check(tmp_path):
    """``_fence_check_for`` must build a callable ``verify_fence`` accepts with
    its REAL signature (``conn, attempt_id, fence, now`` -- no
    ``check_lease_time`` keyword; ``engine.v2.ops.lifecycle.verify_fence`` has
    never had one). It must also still enforce the production wall-clock
    lease-expiry check -- never a skipped check, no matter who calls this
    store."""
    conn, clock, supervisor = catalog(tmp_path)
    claim = enqueue_claim(conn, clock, supervisor)
    check = _fence_check_for(claim.attempt_id, claim.fence, clock)

    with transaction(conn):
        job, attempt = check(conn)
        assert job["fence"] == claim.fence
        assert attempt["fence"] == claim.fence

    clock.advance(10 ** 6)  # long past any lease_expires_at
    with transaction(conn):
        with pytest.raises(OpsError) as err:
            check(conn)
        assert err.value.code == "LEASE_LOST"


def test_fence_check_for_with_no_staged_attempt_is_a_noop():
    """No staged attempt (e.g. a manual/ad-hoc invocation with nothing to
    fence against): the returned callable does nothing and returns ``None``,
    matching ``_commit_generation``'s previous inline ternary."""
    check = _fence_check_for(None, None, clock=None)
    assert check(object()) is None


# --------------------------------------------------------------------------
# capture_id must not collide two attempts with genuinely different content
# (CodeRabbit round 2: request_id alone is not a sufficient identity)
# --------------------------------------------------------------------------


def test_capture_id_folds_in_source_hash_so_different_bytes_do_not_collide():
    """Same (ticker, as_of) request_id, but the upstream data genuinely
    differs between two attempts (a same-day yfinance correction/backfill) --
    the identity must differ too, or ``_insert_captures``'s dedup would
    silently keep only the FIRST attempt's outcome even though the second
    attempt's actual captured content was different."""
    unit = _unit("ABCD", "2026-09-24")
    same_a = _capture_id_for(unit, source_hash="hash-one")
    same_b = _capture_id_for(unit, source_hash="hash-one")
    different = _capture_id_for(unit, source_hash="hash-two")
    no_source = _capture_id_for(unit)
    assert same_a == same_b
    assert different != same_a
    assert no_source != same_a
    assert no_source != different


def test_capture_id_folds_in_the_contract_definition_hash():
    """A contract/schema change between two captures of the same unit must
    not silently reuse the old capture identity."""
    unit = _unit("ABCD", "2026-09-24")
    original = _capture_id_for(unit, source_hash="hash-one")
    other_schema = _capture_id_for(unit, source_hash="hash-one",
                                   definition_hash="sha256:" + "0" * 64)
    assert other_schema != original


# --------------------------------------------------------------------------
# run_computed_moves_refresh end to end (CodeRabbit round 2: no direct test
# of the callback existed): a real snapshot/catalog/store fixture, no network.
# --------------------------------------------------------------------------

_EVENTS = contract_for("earnings_events")
_EVENTS_REF = contract_ref_for(_EVENTS)
_DAILY = contract_for("daily_market")
_DAILY_REF = contract_ref_for(_DAILY)

_BDAYS = pd.bdate_range("2024-01-02", "2024-01-31")  # 22 business days
_EVENT_DAYS = [_BDAYS[2], _BDAYS[6], _BDAYS[10], _BDAYS[14], _BDAYS[18]]  # 5 spread-out prints
_AS_OF = "2024-02-05"


def _event_row(ticker: str, event_date) -> dict:
    d = event_date.to_pydatetime()
    return dict(
        event_id=f"{ticker}_{d.date()}", ticker=ticker, event_date=d, year=d.year,
        session="AMC", session_src="orats", annc_tod=None, src_orats=True,
        src_oquants=False, src_nasdaq=False, src_yfinance=False, date_agree=True,
        date_conflict=False, updated_at=None, event_cluster_id=None, claim_count=None,
        reconciliation=None)


def _closes_csv() -> bytes:
    lines = ["Date,Close"] + [f"{d.date()},{100.0 + i}" for i, d in enumerate(_BDAYS)]
    return "\n".join(lines).encode()


def _head_row(conn, scope="shadow"):
    return conn.execute(
        "SELECT snapshot_id, generation FROM data_snapshot_heads WHERE scope = ?",
        (scope,)).fetchone()


def _build_parent(conn, clock, store, *, events_rows=()):
    tables = {"daily_market": []}
    if events_rows:
        record = publish_and_inspect(store, _EVENTS, _EVENTS_REF, list(events_rows), "2024")
        tables["earnings_events"] = [record]
    else:
        tables["earnings_events"] = []
    contracts = {"earnings_events": _EVENTS, "daily_market": _DAILY}
    commit_tables(conn, clock, tables, contracts, store=store)
    head = _head_row(conn)
    from engine.v2.data import reference_catalog
    from engine.v2.data.reference_catalog import ReferenceInput

    receipt_id = reference_catalog.committed_receipt_for_snapshot(
        conn, scope="shadow", snapshot_id=head["snapshot_id"])
    assert receipt_id is not None
    pin = ReferenceInput(
        kind="calendar", legacy_path="calendar/trading_days.csv",
        object_id="object-reference-calendar", content_hash="sha256:" + "a" * 64,
        byte_size=1)
    with transaction(conn):
        reference_catalog.insert_reference_inputs(conn, receipt_id, [pin])
    return head


class _CountingFetcher:
    """Records every ticker it is called for; returns the same complete CSV."""

    def __init__(self, csv_bytes: bytes):
        self._csv = csv_bytes
        self.calls: list[str] = []

    def __call__(self, ticker: str):
        self.calls.append(ticker)
        return self._csv, "complete", {}, None


def _refused_fetcher(_ticker):
    return b"not-a-real-response", "refused", {}, None


def test_tier1_yfinance_history_fetcher_uses_newest_success_and_empty_for_missing(
        tmp_path, monkeypatch):
    from types import SimpleNamespace

    def entry(endpoint, period, ticker, status, fetched_at, path, body):
        return SimpleNamespace(
            endpoint=endpoint, params={"period": period, "ticker": ticker},
            meta={"status": status, "fetched_at": fetched_at}, path=path,
            body=lambda: body)

    rows = [
        entry("history", "max", "AAAA", 200, "2026-01-01", "cache/old", b"old"),
        entry("history", "max", "AAAA", 200, "2026-01-02", "cache/new", b"new"),
        entry("quote", "max", "AAAA", 200, "2026-01-09", "cache/wrong-endpoint", b"bad"),
        entry("history", "1y", "AAAA", 200, "2026-01-10", "cache/wrong-period", b"bad"),
        entry("history", "max", "AAAA", 500, "2026-01-11", "cache/failed", b"bad"),
    ]
    seen = {}

    def iter_cache(source_root, source):
        seen.update(source_root=source_root, source=source)
        return rows

    monkeypatch.setattr(computed_moves_store, "iter_raw_fetch_cache", iter_cache)
    cache, fetcher = computed_moves_store.tier1_yfinance_history_fetcher(tmp_path)

    assert seen == {"source_root": tmp_path, "source": "yfinance"}
    assert cache["AAAA"] is rows[1]
    assert fetcher("AAAA") == (b"new", "complete", {}, [])
    assert fetcher("MISSING") == (b"", "legitimate_empty", {}, [])


def _parameters(head, *, expected_ids, catalog_path, objects_root,
                overrides=None) -> RefreshParameters:
    kwargs = dict(
        expected_ids=expected_ids, parent_snapshot_id=head["snapshot_id"],
        refresh_plan_hash="sha256:" + "d" * 64, provider_calls=1,
        catalog_path=str(catalog_path), objects_root=str(objects_root), scope="shadow",
        expected_head_generation=head["generation"], expected_head_snapshot_id=head["snapshot_id"])
    if overrides:
        kwargs.update(overrides)
    return RefreshParameters(**kwargs)


def _write_input(root, *, catalog_path, objects_root, head, as_of=_AS_OF, overrides=None):
    root.mkdir(exist_ok=True)
    document = {
        "catalog_path": str(catalog_path), "objects_root": str(objects_root),
        "scope": "shadow", "expected_head_generation": head["generation"],
        "expected_head_snapshot_id": head["snapshot_id"], "as_of": as_of,
    }
    if overrides:
        document.update(overrides)
    (root / computed_moves_store.INPUT_PATH).write_text(canonical_json(document))


def test_computed_moves_command_rejects_invalid_as_of_before_planning(tmp_path):
    from types import SimpleNamespace
    from engine.v2.ops.cli import computed_moves_command

    conn, clock, _ = catalog(tmp_path)
    source_root = tmp_path / "legacy"
    source_root.mkdir()
    args = SimpleNamespace(source_root=source_root, as_of="not-a-date",
                           scope="shadow", dry_run=False)
    with pytest.raises(OpsError) as err:
        computed_moves_command(args, tmp_path, conn, clock)
    assert err.value.code == "INVALID_REQUEST"


def test_computed_moves_command_refuses_missing_scoped_head(tmp_path):
    from types import SimpleNamespace
    from engine.v2.ops.cli import computed_moves_command

    conn, clock, _ = catalog(tmp_path)
    source_root = tmp_path / "legacy"
    source_root.mkdir()
    args = SimpleNamespace(source_root=source_root, as_of=_AS_OF,
                           scope="shadow", dry_run=False)
    with pytest.raises(OpsError) as err:
        computed_moves_command(args, tmp_path, conn, clock)
    assert err.value.code == "SNAPSHOT_NOT_READY"


def test_missing_tier1_history_is_logged_no_history_without_failing(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(computed_moves_store, "target_tickers_from_snapshot",
                        lambda *a, **k: (["AAAA", "BBBB"], {}))
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    events_rows = ([_event_row("AAAA", d) for d in _EVENT_DAYS]
                   + [_event_row("BBBB", d) for d in _EVENT_DAYS])
    head = _build_parent(conn, clock, store, events_rows=events_rows)
    entry = SimpleNamespace(
        endpoint="history", params={"period": "max", "ticker": "AAAA"},
        meta={"status": 200, "fetched_at": "2026-01-01"}, path="cache/AAAA",
        body=_closes_csv)
    monkeypatch.setattr(computed_moves_store, "iter_raw_fetch_cache",
                        lambda root, source: [entry])
    _, fetcher = computed_moves_store.tier1_yfinance_history_fetcher(tmp_path)
    root = tmp_path / "attempt"
    _write_input(root, catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path, head=head)
    parameters = _parameters(head, expected_ids=("AAAA", "BBBB"),
                             catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path)

    result = computed_moves_store.run_computed_moves_refresh(
        parameters, root, as_of=_AS_OF, fetcher=fetcher)

    assert result.status == "complete"
    row = conn.execute(
        "SELECT outcome FROM data_computed_moves_captures WHERE ticker = ?", ("BBBB",)
    ).fetchone()
    assert row["outcome"] == "no_history"


def test_computed_moves_command_refuses_missing_source_root(tmp_path):
    from types import SimpleNamespace
    from engine.v2.ops.cli import computed_moves_command

    conn, clock, _ = catalog(tmp_path)
    args = SimpleNamespace(source_root=tmp_path / "missing", as_of=_AS_OF,
                           scope="shadow", dry_run=False)
    with pytest.raises(OpsError) as err:
        computed_moves_command(args, tmp_path, conn, clock)
    assert err.value.code == "INVALID_REQUEST"


def test_computed_moves_command_refuses_held_supervisor_lock(tmp_path):
    from types import SimpleNamespace
    from engine.v2.ops.cli import computed_moves_command
    from engine.v2.ops.recovery import SupervisorLock

    conn, clock, _ = catalog(tmp_path)
    source_root = tmp_path / "legacy"
    source_root.mkdir()
    held = SupervisorLock(tmp_path / "supervisor.lock")
    assert held.acquire()
    args = SimpleNamespace(source_root=source_root, as_of=_AS_OF,
                           scope="shadow", dry_run=False)
    try:
        with pytest.raises(OpsError) as err:
            computed_moves_command(args, tmp_path, conn, clock)
        assert err.value.code == "RESOURCE_UNAVAILABLE"
    finally:
        held.release()


def test_computed_moves_command_dry_run_reports_coverage_without_data_writes(
        tmp_path, monkeypatch):
    from types import SimpleNamespace
    from engine.v2.ops.cli import computed_moves_command

    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    events_rows = [_event_row("AAAA", d) for d in _EVENT_DAYS]
    head = _build_parent(conn, clock, store, events_rows=events_rows)
    source_root = tmp_path / "legacy"
    source_root.mkdir()
    monkeypatch.setattr(computed_moves_store, "target_tickers_from_snapshot",
                        lambda *a, **k: (["AAAA"], {}))
    cached = SimpleNamespace(
        endpoint="history", params={"period": "max", "ticker": "AAAA"},
        meta={"status": 200, "fetched_at": "2026-01-01"}, path="cache/AAAA",
        body=lambda: pytest.fail("dry-run must not read a history body"))
    cache_entries = [cached]
    monkeypatch.setattr(computed_moves_store, "iter_raw_fetch_cache",
                        lambda root, source: cache_entries)
    monkeypatch.setattr(
        computed_moves_store, "run_computed_moves_refresh",
        lambda *a, **k: pytest.fail("dry-run must not invoke the capture runner"))
    before_head = _head_row(conn)["snapshot_id"]
    before_receipts = conn.execute("SELECT COUNT(*) FROM data_raw_receipts").fetchone()[0]
    before_captures = conn.execute(
        "SELECT COUNT(*) FROM data_computed_moves_captures").fetchone()[0]
    args = SimpleNamespace(source_root=source_root, as_of=_AS_OF,
                           scope="shadow", dry_run=True)

    report = computed_moves_command(args, tmp_path, conn, clock)
    assert report["target_count"] == 1
    assert report["with_tier1_entry"] == 1
    assert report["without_tier1_entry"] == 0
    assert report["dry_run"] is True
    cache_entries.clear()
    missing_report = computed_moves_command(args, tmp_path, conn, clock)
    assert missing_report["with_tier1_entry"] == 0
    assert missing_report["without_tier1_entry"] == 1
    assert _head_row(conn)["snapshot_id"] == before_head
    assert conn.execute("SELECT COUNT(*) FROM data_raw_receipts").fetchone()[0] == before_receipts
    assert conn.execute(
        "SELECT COUNT(*) FROM data_computed_moves_captures").fetchone()[0] == before_captures
    assert not (tmp_path / computed_moves_store.INPUT_PATH).exists()


def test_run_computed_moves_refresh_captures_one_complete_unit(tmp_path, monkeypatch):
    monkeypatch.setattr(computed_moves_store, "target_tickers_from_snapshot",
                        lambda *a, **k: (["AAAA"], {}))
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    events_rows = [_event_row("AAAA", d) for d in _EVENT_DAYS]
    head = _build_parent(conn, clock, store, events_rows=events_rows)

    root = tmp_path / "attempt"
    _write_input(root, catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path, head=head)
    fetcher = _CountingFetcher(_closes_csv())
    parameters = _parameters(head, expected_ids=("AAAA",),
                             catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path)

    result = computed_moves_store.run_computed_moves_refresh(
        parameters, root, as_of=_AS_OF, fetcher=fetcher)

    assert result.status == "complete"
    assert result.coverage_advanced is True
    assert result.completed_ids == ("AAAA",)
    assert fetcher.calls == ["AAAA"]
    assert conn.execute("SELECT COUNT(*) FROM data_raw_receipts").fetchone()[0] == 1
    row = conn.execute(
        "SELECT outcome FROM data_computed_moves_captures WHERE ticker = ?",
        ("AAAA",)).fetchone()
    assert row["outcome"] == "added"


def test_run_computed_moves_refresh_carries_reference_pins_and_lineage(tmp_path, monkeypatch):
    """Exercise the real runner, row producer and snapshot commit on tiny inputs.

    Only target selection is stubbed to keep the fixture focused on one ticker.
    The parent catalog receipt is seeded with one reference pin; runner,
    ``build_rows``, ``commit_snapshot`` and its reference callback are real.
    """
    from engine.v2.data import reference_catalog

    monkeypatch.setattr(computed_moves_store, "target_tickers_from_snapshot",
                        lambda *a, **k: (["AAAA"], {}))
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    events_rows = [_event_row("AAAA", d) for d in _EVENT_DAYS]
    head = _build_parent(conn, clock, store, events_rows=events_rows)
    base_receipt_id = reference_catalog.committed_receipt_for_snapshot(
        conn, scope="shadow", snapshot_id=head["snapshot_id"])
    assert base_receipt_id is not None
    reference = reference_catalog.reference_inputs_for_receipt(
        conn, receipt_id=base_receipt_id)[0]

    root = tmp_path / "attempt"
    _write_input(root, catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path, head=head)
    parameters = _parameters(head, expected_ids=("AAAA",),
                             catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path)
    result = computed_moves_store.run_computed_moves_refresh(
        parameters, root, as_of=_AS_OF, fetcher=_CountingFetcher(_closes_csv()))

    assert result.status == "complete"
    child_receipt_id = reference_catalog.committed_receipt_for_snapshot(
        conn, scope="shadow", snapshot_id=result.candidate_snapshot_id)
    assert child_receipt_id is not None
    assert reference_catalog.reference_inputs_for_receipt(
        conn, receipt_id=child_receipt_id) == (reference,)
    lineage = conn.execute(
        "SELECT kind, base_receipt_id FROM data_receipt_lineage WHERE receipt_id = ?",
        (child_receipt_id,)).fetchone()
    assert lineage["kind"] == "price_history_capture"
    assert lineage["base_receipt_id"] == base_receipt_id


def test_run_computed_moves_refresh_never_commits_a_row_for_an_event_or_exit_after_as_of(
        tmp_path, monkeypatch):
    """Issue #99: a re-run of a past session fetches the ticker's FULL
    history at the run's real wall-clock time, which extends past ``as_of``.
    Nothing may reach ``build_rows`` beyond ``as_of``: not the closes series,
    and not an event dated on/after it (which would be scored against real
    future closes and committed stamped ``computed_at = as_of``, leaking
    realized data).

    Row-level read-back: ``_scan_rows`` -- the store's only generic scanner --
    cannot read this table back (it assumes year-partitioned fragments;
    ``computed_moves`` fragments are partitioned by ticker, so
    ``int(partition_key)`` raises before any scan). So this test proves
    committed content with a direct ``DataQuery``/``KeyPredicate`` scan below
    -- the same mechanism ``tests/test_v2_ops_price_history.py`` uses for its
    own ticker-partitioned table -- plus a ``build_rows`` spy for what was
    proposed to the row builder: no event at/after ``as_of``, a series
    truncated at ``as_of``, and the ordinary pre-``as_of`` event still
    computing a real (non-skipped) move. The capture log's
    ``outcome == "added"`` additionally shows a fragment WAS written (not a
    skip).
    """
    monkeypatch.setattr(computed_moves_store, "target_tickers_from_snapshot",
                        lambda *a, **k: (["AAAA"], {}))
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    extended = pd.bdate_range("2024-01-02", "2024-02-29")  # runs well past _AS_OF
    csv_bytes = "\n".join(["Date,Close"] + [f"{d.date()},{100.0 + i}"
                                            for i, d in enumerate(extended)]).encode()
    events_rows = ([_event_row("AAAA", d) for d in _EVENT_DAYS]  # all before _AS_OF
                   + [_event_row("AAAA", pd.Timestamp("2024-02-05")),  # exactly on _AS_OF
                      _event_row("AAAA", pd.Timestamp("2024-02-08"))])  # after _AS_OF
    head = _build_parent(conn, clock, store, events_rows=events_rows)

    captured: dict = {}
    real_build_rows = computed_moves_store.build_rows

    def _spy_build_rows(ticker, events, sd, sc, daily, **kwargs):
        captured["event_dates"] = [str(ts.date()) for ts in events["event_date"]]
        captured["series_last"] = str(pd.Timestamp(sd[-1]).date())
        captured["rows"] = real_build_rows(ticker, events, sd, sc, daily, **kwargs)
        return captured["rows"]

    monkeypatch.setattr(computed_moves_store, "build_rows", _spy_build_rows)

    root = tmp_path / "attempt"
    _write_input(root, catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path, head=head)
    fetcher = _CountingFetcher(csv_bytes)
    parameters = _parameters(head, expected_ids=("AAAA",),
                             catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path)

    result = computed_moves_store.run_computed_moves_refresh(
        parameters, root, as_of=_AS_OF, fetcher=fetcher)

    assert result.status == "complete"
    row = conn.execute(
        "SELECT outcome FROM data_computed_moves_captures WHERE ticker = ?",
        ("AAAA",)).fetchone()
    assert row["outcome"] == "added"  # a fragment WAS written -- the ticker wasn't skipped

    assert captured["event_dates"]  # the spy really fired
    assert captured["series_last"] <= _AS_OF  # series truncated at as_of_day
    assert all(d < _AS_OF for d in captured["event_dates"])
    assert "2024-02-08" not in captured["event_dates"]  # the leaked event never got here
    assert "2024-02-05" not in captured["event_dates"]  # the on-as_of event is excluded too

    rows = captured["rows"]
    assert rows
    # Every committed row is dated strictly before _AS_OF -- not even a
    # skipped=True placeholder exists for the post-as_of event.
    assert all(r["event_date"] < _AS_OF for r in rows)
    pre = [r for r in rows if r["event_date"] == str(_EVENT_DAYS[0].date())]
    assert len(pre) == 1
    assert pre[0]["skipped"] is False
    assert pre[0]["realized_move_pct"] is not None  # ordinary case still computes

    from engine.v2.contracts import DataQuery, KeyPredicate
    from engine.v2.data.computed_moves_table import COMPUTED_MOVES_TABLE_NAME

    repository = Repository(conn, store)
    snapshot = repository.resolve(result.candidate_snapshot_id)
    dvr = snapshot.table_versions[COMPUTED_MOVES_TABLE_NAME]
    key_filter = (KeyPredicate(column="ticker", operator="eq", values=("AAAA",)),)
    bound = repository.scan_population_bound(
        snapshot.snapshot_id, table_name=COMPUTED_MOVES_TABLE_NAME,
        table_contract_ref=dvr.table_contract_ref, key_filter=key_filter,
        time_interval=None)
    query = DataQuery(
        snapshot_id=snapshot.snapshot_id, table_contract_ref=dvr.table_contract_ref,
        columns=("event_date", "realized_move_pct", "skipped"),
        key_filter=key_filter,
        order_by=("ticker", "event_date"),
        max_batch_rows=min(100, bound), max_result_rows=min(100, bound))
    committed_rows = [r for batch in repository.scan(query, table_name=COMPUTED_MOVES_TABLE_NAME)
                      for r in batch.to_pylist()]

    assert committed_rows
    assert all(str(r["event_date"]) < _AS_OF for r in committed_rows)  # no post-as_of row at all
    assert "2024-02-05" not in [str(r["event_date"]) for r in committed_rows]  # on-as_of too
    committed_pre = [r for r in committed_rows
                     if str(r["event_date"]) == str(_EVENT_DAYS[0].date())]
    assert len(committed_pre) == 1
    assert committed_pre[0]["skipped"] is False
    assert committed_pre[0]["realized_move_pct"] is not None


def test_run_computed_moves_refresh_series_entirely_after_as_of_is_too_few(
        tmp_path, monkeypatch):
    """The issue #99 truncation branch of ``_capture_targets``: every fetched
    close is dated AFTER ``as_of``, so the series truncates to empty before
    hashing and the unit logs ``too_few`` -- a legitimate business finding
    that does not fail the job, exactly like the "BBBB has no events"
    ``too_few`` outcome two functions below.

    AAAA keeps real pre-``as_of`` events, so the branch under test is
    specifically the series-truncates-to-empty one (``sd.size == 0``), not
    the earlier ``events is None`` branch. A second ticker, BBBB, carrying an
    ordinary complete series, is what lets this run commit at all: a
    zero-fragment run short-circuits to a true noop BEFORE
    ``_insert_captures`` (reached only through the commit's
    ``record_references``) logs anything, so a ``too_few`` outcome is only
    observable in a run that writes at least one fragment -- the same shape
    as the BBBB test below, with the roles mirrored.
    """
    monkeypatch.setattr(computed_moves_store, "target_tickers_from_snapshot",
                        lambda *a, **k: (["AAAA", "BBBB"], {}))
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    events_rows = [_event_row(ticker, d) for ticker in ("AAAA", "BBBB")
                   for d in _EVENT_DAYS]  # every event strictly before _AS_OF
    head = _build_parent(conn, clock, store, events_rows=events_rows)

    after_as_of = pd.bdate_range("2024-02-06", "2024-02-29")  # entirely after _AS_OF
    truncated = "\n".join(["Date,Close"] + [f"{d.date()},{100.0 + i}"
                                            for i, d in enumerate(after_as_of)]).encode()
    series_by_ticker = {"AAAA": truncated, "BBBB": _closes_csv()}
    calls: list[str] = []

    def fetcher(ticker):  # same return shape as _CountingFetcher, per-ticker bytes
        calls.append(ticker)
        return series_by_ticker[ticker], "complete", {}, None

    root = tmp_path / "attempt"
    _write_input(root, catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path, head=head)
    parameters = _parameters(head, expected_ids=("AAAA", "BBBB"),
                             catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path)

    result = computed_moves_store.run_computed_moves_refresh(
        parameters, root, as_of=_AS_OF, fetcher=fetcher)

    assert result.status == "complete"  # a too_few outcome never fails the job
    row = conn.execute(
        "SELECT outcome FROM data_computed_moves_captures WHERE ticker = ?",
        ("AAAA",)).fetchone()
    assert row["outcome"] == "too_few"
    other = conn.execute(
        "SELECT outcome FROM data_computed_moves_captures WHERE ticker = ?",
        ("BBBB",)).fetchone()
    assert other["outcome"] == "added"  # the run really committed, not a noop


def test_run_computed_moves_refresh_capture_id_is_stable_across_different_post_as_of_tails(
        tmp_path_factory, monkeypatch):
    """Issue #99: capture identity must be a pure function of the series
    TRUNCATED to ``as_of``, never of the raw fetch. Two runs whose fetches
    agree on every date ``<= as_of`` but disagree on what comes after it --
    the same ticker pulled on two different wall-clock days, with different
    post-``as_of`` closes -- must log the SAME ``capture_id``: ``source_hash``
    (and so ``_capture_id_for``) is computed from the truncated closes, and
    the post-``as_of`` tail is invisible to capture identity.

    The two runs are fully independent -- separate sqlite catalogs and
    separate ``ArtifactStore`` roots via ``tmp_path_factory`` -- so this
    compares hashes across DIFFERENT raw fetches, not the same-catalog
    rerun no-op/cache behavior already covered by
    ``test_run_computed_moves_refresh_cached_rerun_refetches_nothing``.
    """
    monkeypatch.setattr(computed_moves_store, "target_tickers_from_snapshot",
                        lambda *a, **k: (["AAAA"], {}))

    through_as_of = pd.bdate_range("2024-01-02", "2024-02-05")  # identical, incl. _AS_OF

    def _series(tail):  # tail: (date, close) pairs strictly AFTER _AS_OF
        lines = [f"{d.date()},{100.0 + i}" for i, d in enumerate(through_as_of)]
        lines += [f"{d.date()},{close}" for d, close in tail]
        return "\n".join(["Date,Close"] + lines).encode()

    tail_a = [(d, 900.0 + i) for i, d in enumerate(pd.bdate_range("2024-02-06", "2024-02-08"))]
    tail_b = [(d, 800.0 + 7 * i)
              for i, d in enumerate(pd.bdate_range("2024-02-06", "2024-02-15"))]

    def _run(directory, csv_bytes):
        conn, clock, _ = catalog(directory)
        store = ArtifactStore(directory)
        events_rows = [_event_row("AAAA", d) for d in _EVENT_DAYS]  # all before _AS_OF
        head = _build_parent(conn, clock, store, events_rows=events_rows)
        root = directory / "attempt"
        _write_input(root, catalog_path=directory / "ops.sqlite",
                     objects_root=directory, head=head)
        parameters = _parameters(head, expected_ids=("AAAA",),
                                 catalog_path=directory / "ops.sqlite",
                                 objects_root=directory)
        result = computed_moves_store.run_computed_moves_refresh(
            parameters, root, as_of=_AS_OF, fetcher=_CountingFetcher(csv_bytes))
        assert result.status == "complete"
        logged = conn.execute(
            "SELECT outcome, capture_id FROM data_computed_moves_captures WHERE ticker = ?",
            ("AAAA",)).fetchone()
        assert logged["outcome"] == "added"  # a real capture both times, not two no-ops
        return logged["capture_id"]

    capture_a = _run(tmp_path_factory.mktemp("a"), _series(tail_a))
    capture_b = _run(tmp_path_factory.mktemp("b"), _series(tail_b))
    assert capture_a == capture_b  # identity ignores the differing post-as_of tail


def test_run_computed_moves_refresh_completed_ids_cover_every_target_even_when_one_has_no_committable_rows(
        tmp_path, monkeypatch):
    """Round 3 fix (Opus finding 2): completed_ids must report the full
    whole-market universe target_tickers_from_snapshot derives, not only the
    tickers that happened to get a written fragment. BBBB has no earnings
    events committed at all, so _capture_targets logs its outcome as
    "too_few" and writes it no fragment -- a legitimate business finding, not
    a failure -- yet it is still part of the run's coverage."""
    monkeypatch.setattr(computed_moves_store, "target_tickers_from_snapshot",
                        lambda *a, **k: (["AAAA", "BBBB"], {}))
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    events_rows = [_event_row("AAAA", d) for d in _EVENT_DAYS]  # BBBB gets none
    head = _build_parent(conn, clock, store, events_rows=events_rows)

    root = tmp_path / "attempt"
    _write_input(root, catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path, head=head)
    fetcher = _CountingFetcher(_closes_csv())
    parameters = _parameters(head, expected_ids=("AAAA", "BBBB"),
                             catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path)

    result = computed_moves_store.run_computed_moves_refresh(
        parameters, root, as_of=_AS_OF, fetcher=fetcher)

    assert result.status == "complete"
    assert result.completed_ids == ("AAAA", "BBBB")
    assert fetcher.calls == ["AAAA", "BBBB"]
    row = conn.execute(
        "SELECT outcome FROM data_computed_moves_captures WHERE ticker = ?",
        ("BBBB",)).fetchone()
    assert row["outcome"] == "too_few"


def test_run_computed_moves_refresh_cached_rerun_refetches_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(computed_moves_store, "target_tickers_from_snapshot",
                        lambda *a, **k: (["AAAA"], {}))
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    events_rows = [_event_row("AAAA", d) for d in _EVENT_DAYS]
    head = _build_parent(conn, clock, store, events_rows=events_rows)

    root = tmp_path / "attempt"
    _write_input(root, catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path, head=head)
    fetcher = _CountingFetcher(_closes_csv())
    parameters = _parameters(head, expected_ids=("AAAA",),
                             catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path)
    first = computed_moves_store.run_computed_moves_refresh(
        parameters, root, as_of=_AS_OF, fetcher=fetcher)
    assert first.status == "complete"
    assert fetcher.calls == ["AAAA"]

    new_head = _head_row(conn)
    root2 = tmp_path / "attempt2"
    _write_input(root2, catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path, head=new_head)
    parameters2 = _parameters(new_head, expected_ids=("AAAA",),
                              catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path)

    second = computed_moves_store.run_computed_moves_refresh(
        parameters2, root2, as_of=_AS_OF, fetcher=fetcher)

    # The rerun is now a TRUE no-op (fixed; was tracked as #41): every
    # committed row's computed_at is derived from as_of, not the run's wall
    # clock, so identical inputs (same as_of, same cached bytes) produce
    # byte-identical fragment content -- the commit resolves back to the
    # parent snapshot instead of a new generation, and the fetcher is never
    # called again.
    assert second.status == "noop"
    assert second.coverage_advanced is False
    assert fetcher.calls == ["AAAA"]  # unchanged: the second run never re-fetched
    assert conn.execute("SELECT COUNT(*) FROM data_raw_receipts").fetchone()[0] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM data_computed_moves_captures WHERE ticker = ?",
        ("AAAA",)).fetchone()[0] == 1  # stable capture_id (round 2 fix) dedups the rerun
    assert dict(_head_row(conn)) == dict(new_head)  # the head never actually moved


def test_run_computed_moves_refresh_provider_failure_maps_to_its_code(tmp_path, monkeypatch):
    monkeypatch.setattr(computed_moves_store, "target_tickers_from_snapshot",
                        lambda *a, **k: (["CCCC"], {}))
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    head = _build_parent(conn, clock, store, events_rows=())

    root = tmp_path / "attempt"
    _write_input(root, catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path, head=head)
    parameters = _parameters(head, expected_ids=("CCCC",),
                             catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path)

    with pytest.raises(OpsError) as err:
        computed_moves_store.run_computed_moves_refresh(
            parameters, root, as_of=_AS_OF, fetcher=_refused_fetcher)

    assert err.value.code == "SOURCE_INVALID"


# --------------------------------------------------------------------------
# as_of must be validated before any I/O (Opus review finding on a sibling
# PR, #40, applies here too: main's RefreshParameters has no as_of field, so
# this function takes it as an explicit keyword and validates it itself)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad_as_of", [
    123,
    123.5,
    True,
    pd.NaT,
    pd.Timestamp("2024-01-01", tz="UTC"),
    "not-a-date",
])
def test_run_computed_moves_refresh_refuses_a_bad_as_of_before_any_io(bad_as_of):
    """``parameters=object()`` and ``root=None`` both raise on the first real
    touch (attribute access / path use) -- so an ``OpsError`` instead of an
    ``AttributeError``/``TypeError`` proves the as_of check runs BEFORE
    ``parameters``, ``root``, or the filesystem are touched at all, exactly
    as ``run_computed_moves_refresh``'s own docstring now claims."""
    with pytest.raises(OpsError) as err:
        computed_moves_store.run_computed_moves_refresh(
            object(), None, as_of=bad_as_of, fetcher=None)
    assert err.value.code == "INVALID_REQUEST"


# --------------------------------------------------------------------------
# every other runner input validated before any I/O too (Opus gate BLOCK on
# f7e4dfb, PR #39 round 4): scope, expected_head_generation, all_scoreable,
# since, unknown document keys, document/parameters agreement, and a missing
# document -- each refused with a typed INVALID_REQUEST before any fetch call
# or receipt write.
# --------------------------------------------------------------------------


def _refusal_fixture(tmp_path):
    """A valid parent snapshot + parameters + fetcher, ready to write a bad
    input document against. Shared by every before-any-I/O refusal test."""
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    head = _build_parent(conn, clock, store, events_rows=())
    parameters = _parameters(head, expected_ids=("CCCC",),
                             catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path)
    fetcher = _CountingFetcher(_closes_csv())
    return conn, head, parameters, fetcher


def _assert_refused_before_any_io(conn, fetcher, *, err):
    assert err.value.code == "INVALID_REQUEST"
    assert fetcher.calls == []
    assert conn.execute("SELECT COUNT(*) FROM data_raw_receipts").fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM data_computed_moves_captures").fetchone()[0] == 0


def test_run_computed_moves_refresh_refuses_an_unknown_document_key(tmp_path):
    conn, head, parameters, fetcher = _refusal_fixture(tmp_path)
    root = tmp_path / "attempt"
    _write_input(root, catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path, head=head,
                overrides={"bogus_field": "nope"})

    with pytest.raises(OpsError) as err:
        computed_moves_store.run_computed_moves_refresh(
            parameters, root, as_of=_AS_OF, fetcher=fetcher)
    _assert_refused_before_any_io(conn, fetcher, err=err)


@pytest.mark.parametrize("bad_scope", [None, "", "production", 7, True])
def test_run_computed_moves_refresh_refuses_a_bad_scope(tmp_path, bad_scope):
    conn, head, parameters, fetcher = _refusal_fixture(tmp_path)
    root = tmp_path / "attempt"
    _write_input(root, catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path, head=head,
                overrides={"scope": bad_scope})

    with pytest.raises(OpsError) as err:
        computed_moves_store.run_computed_moves_refresh(
            parameters, root, as_of=_AS_OF, fetcher=fetcher)
    _assert_refused_before_any_io(conn, fetcher, err=err)


@pytest.mark.parametrize("bad_generation", [None, True, 1.9, "3", -1])
def test_run_computed_moves_refresh_refuses_a_bad_expected_head_generation(tmp_path,
                                                                           bad_generation):
    conn, head, parameters, fetcher = _refusal_fixture(tmp_path)
    root = tmp_path / "attempt"
    _write_input(root, catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path, head=head,
                overrides={"expected_head_generation": bad_generation})

    with pytest.raises(OpsError) as err:
        computed_moves_store.run_computed_moves_refresh(
            parameters, root, as_of=_AS_OF, fetcher=fetcher)
    _assert_refused_before_any_io(conn, fetcher, err=err)


def test_run_computed_moves_refresh_refuses_a_non_bool_all_scoreable(tmp_path):
    """The old call site did ``bool(document.get("all_scoreable", True))`` --
    ``bool("false")`` is ``True`` in Python, silently inverting the string
    value instead of refusing it. This must now be a typed refusal, not a
    silently-flipped selection mode."""
    conn, head, parameters, fetcher = _refusal_fixture(tmp_path)
    root = tmp_path / "attempt"
    _write_input(root, catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path, head=head,
                overrides={"all_scoreable": "false"})

    with pytest.raises(OpsError) as err:
        computed_moves_store.run_computed_moves_refresh(
            parameters, root, as_of=_AS_OF, fetcher=fetcher)
    _assert_refused_before_any_io(conn, fetcher, err=err)


@pytest.mark.parametrize("bad_since", [123, True, "not-a-date", "2024-01-01T00:00:00+00:00"])
def test_run_computed_moves_refresh_refuses_a_bad_since(tmp_path, bad_since):
    conn, head, parameters, fetcher = _refusal_fixture(tmp_path)
    root = tmp_path / "attempt"
    _write_input(root, catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path, head=head,
                overrides={"since": bad_since})

    with pytest.raises(OpsError) as err:
        computed_moves_store.run_computed_moves_refresh(
            parameters, root, as_of=_AS_OF, fetcher=fetcher)
    _assert_refused_before_any_io(conn, fetcher, err=err)


def test_target_tickers_from_snapshot_refuses_a_bad_since(tmp_path):
    """:201's since handling must validate through the same ``_as_of_day``
    helper as_of gets, not a bare ``pd.Timestamp(since).normalize()`` that
    silently reads a number as a UNIX timestamp or accepts a tz-aware value.
    """
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    repository = Repository(conn, store)
    head = _build_parent(conn, clock, store, events_rows=())

    with pytest.raises(OpsError) as err:
        computed_moves_store.target_tickers_from_snapshot(
            repository, head["snapshot_id"], since="not-a-date", as_of=_AS_OF,
            events=pd.DataFrame(), daily=pd.DataFrame())
    assert err.value.code == "INVALID_REQUEST"


def test_run_computed_moves_refresh_refuses_as_of_disagreeing_with_document(tmp_path):
    conn, head, parameters, fetcher = _refusal_fixture(tmp_path)
    root = tmp_path / "attempt"
    _write_input(root, catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path, head=head,
                as_of="2024-02-06")  # disagrees with the as_of keyword below

    with pytest.raises(OpsError) as err:
        computed_moves_store.run_computed_moves_refresh(
            parameters, root, as_of=_AS_OF, fetcher=fetcher)
    _assert_refused_before_any_io(conn, fetcher, err=err)


@pytest.mark.parametrize("field,bad_value", [
    ("catalog_path", "/somewhere/else.sqlite"),
    ("scope", "smoke"),
    ("expected_head_generation", 999),
])
def test_run_computed_moves_refresh_refuses_document_disagreeing_with_parameters(
        tmp_path, field, bad_value):
    conn, head, parameters, fetcher = _refusal_fixture(tmp_path)
    root = tmp_path / "attempt"
    _write_input(root, catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path, head=head,
                overrides={field: bad_value})

    with pytest.raises(OpsError) as err:
        computed_moves_store.run_computed_moves_refresh(
            parameters, root, as_of=_AS_OF, fetcher=fetcher)
    _assert_refused_before_any_io(conn, fetcher, err=err)


def test_run_computed_moves_refresh_refuses_a_missing_input_document(tmp_path):
    """A missing staged document must be a typed refusal with an error code
    -- not ``status="failed"`` with no code at all (Opus review, PR #39)."""
    conn, head, parameters, fetcher = _refusal_fixture(tmp_path)
    root = tmp_path / "attempt_never_written"

    with pytest.raises(OpsError) as err:
        computed_moves_store.run_computed_moves_refresh(
            parameters, root, as_of=_AS_OF, fetcher=fetcher)
    _assert_refused_before_any_io(conn, fetcher, err=err)


# --------------------------------------------------------------------------
# #41 closed by #39 (not by touching the shared computed_moves.py): a rerun
# at a genuinely different wall-clock time must still commit byte-identical
# fragment content and resolve to a true noop, because computed_at is
# derived from as_of, never SystemClock().now().
# --------------------------------------------------------------------------


def test_run_computed_moves_refresh_rerun_at_a_different_clock_time_is_a_true_noop(
        tmp_path, monkeypatch):
    """Two runs on the SAME as_of/inputs, at two DIFFERENT injected clock
    times (three hours apart), must commit byte-identical fragment content
    (same fragment_id, same object content hash) and the second run must
    resolve to a true noop -- never a fresh generation."""
    monkeypatch.setattr(computed_moves_store, "target_tickers_from_snapshot",
                        lambda *a, **k: (["AAAA"], {}))
    conn, real_clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    events_rows = [_event_row("AAAA", d) for d in _EVENT_DAYS]
    head = _build_parent(conn, real_clock, store, events_rows=events_rows)

    fake_clock = FakeClock()
    monkeypatch.setattr(computed_moves_store, "SystemClock", lambda: fake_clock)

    root = tmp_path / "attempt"
    _write_input(root, catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path, head=head)
    fetcher = _CountingFetcher(_closes_csv())
    parameters = _parameters(head, expected_ids=("AAAA",),
                             catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path)

    first = computed_moves_store.run_computed_moves_refresh(
        parameters, root, as_of=_AS_OF, fetcher=fetcher)
    assert first.status == "complete"

    repository = Repository(conn, store)

    def _fragment(snapshot_id):
        resolved = repository.resolve_full(snapshot_id)
        return next(record for record in resolved.records
                   if record.table_contract_ref.contract_id
                   == computed_moves_store.COMPUTED_MOVES_CONTRACT.contract_id
                   and record.partition_key == "AAAA")

    fragment1 = _fragment(first.candidate_snapshot_id)

    fake_clock.advance(3 * 3600)  # a genuinely different wall-clock time

    new_head = _head_row(conn)
    root2 = tmp_path / "attempt2"
    _write_input(root2, catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path, head=new_head)
    parameters2 = _parameters(new_head, expected_ids=("AAAA",),
                              catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path)

    second = computed_moves_store.run_computed_moves_refresh(
        parameters2, root2, as_of=_AS_OF, fetcher=fetcher)

    assert second.status == "noop"
    assert second.coverage_advanced is False
    assert fetcher.calls == ["AAAA"]  # never re-fetched
    assert dict(_head_row(conn)) == dict(new_head)  # the head never actually moved

    fragment2 = _fragment(new_head["snapshot_id"])
    assert fragment1.fragment_id == fragment2.fragment_id
    assert fragment1.object_ref.content_hash == fragment2.object_ref.content_hash


# --------------------------------------------------------------------------
# round 5 (Opus re-gate BLOCK on 05fd8f7): catalog_path/objects_root
# existence, parameters' own parent_snapshot_id/refresh_plan_hash, and
# expected_head_snapshot_id/fence format -- all validated before the sqlite
# connection opens or any fetch/receipt happens. No pre-fetch head check: a
# stale head is still only caught at commit time as SNAPSHOT_CONFLICT.
# --------------------------------------------------------------------------


def _connect_spy(monkeypatch):
    """Proves ``sqlite3.connect`` is never reached by a before-any-I/O
    refusal: a real connect, wrapped to also record every call."""
    calls = []
    real_connect = computed_moves_store.sqlite3.connect

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(computed_moves_store.sqlite3, "connect", spy)
    return calls


def test_run_computed_moves_refresh_refuses_a_catalog_path_that_is_not_a_file(
        tmp_path, monkeypatch):
    conn, head, parameters, fetcher = _refusal_fixture(tmp_path)
    connect_calls = _connect_spy(monkeypatch)
    missing_catalog = tmp_path / "missing.sqlite"
    root = tmp_path / "attempt"
    _write_input(root, catalog_path=missing_catalog, objects_root=tmp_path, head=head,
                overrides={"catalog_path": str(missing_catalog)})
    parameters = _parameters(head, expected_ids=("CCCC",), catalog_path=missing_catalog,
                             objects_root=tmp_path)

    with pytest.raises(OpsError) as err:
        computed_moves_store.run_computed_moves_refresh(
            parameters, root, as_of=_AS_OF, fetcher=fetcher)
    _assert_refused_before_any_io(conn, fetcher, err=err)
    assert connect_calls == []


def test_run_computed_moves_refresh_refuses_an_objects_root_that_is_not_a_directory(
        tmp_path, monkeypatch):
    conn, head, parameters, fetcher = _refusal_fixture(tmp_path)
    connect_calls = _connect_spy(monkeypatch)
    not_a_dir = tmp_path / "ops.sqlite"  # a real file, not a directory
    root = tmp_path / "attempt"
    _write_input(root, catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path, head=head,
                overrides={"objects_root": str(not_a_dir)})

    with pytest.raises(OpsError) as err:
        computed_moves_store.run_computed_moves_refresh(
            parameters, root, as_of=_AS_OF, fetcher=fetcher)
    _assert_refused_before_any_io(conn, fetcher, err=err)
    assert connect_calls == []


def test_run_computed_moves_refresh_refuses_an_objects_root_disagreeing_with_parameters(
        tmp_path, monkeypatch):
    conn, head, parameters, fetcher = _refusal_fixture(tmp_path)
    connect_calls = _connect_spy(monkeypatch)
    other_dir = tmp_path / "other_objects"
    other_dir.mkdir()
    root = tmp_path / "attempt"
    _write_input(root, catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path, head=head,
                overrides={"objects_root": str(other_dir)})

    with pytest.raises(OpsError) as err:
        computed_moves_store.run_computed_moves_refresh(
            parameters, root, as_of=_AS_OF, fetcher=fetcher)
    _assert_refused_before_any_io(conn, fetcher, err=err)
    assert connect_calls == []


@pytest.mark.parametrize("bad_head", ["", "x" * 129, 7, True])
def test_run_computed_moves_refresh_refuses_a_bad_expected_head_snapshot_id(
        tmp_path, monkeypatch, bad_head):
    conn, head, parameters, fetcher = _refusal_fixture(tmp_path)
    connect_calls = _connect_spy(monkeypatch)
    root = tmp_path / "attempt"
    _write_input(root, catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path, head=head,
                overrides={"expected_head_snapshot_id": bad_head})

    with pytest.raises(OpsError) as err:
        computed_moves_store.run_computed_moves_refresh(
            parameters, root, as_of=_AS_OF, fetcher=fetcher)
    _assert_refused_before_any_io(conn, fetcher, err=err)
    assert connect_calls == []


@pytest.mark.parametrize("bad_fence", [True, 0, -1, 1.5, "1"])
def test_run_computed_moves_refresh_refuses_a_bad_fence(tmp_path, monkeypatch, bad_fence):
    conn, head, parameters, fetcher = _refusal_fixture(tmp_path)
    connect_calls = _connect_spy(monkeypatch)
    root = tmp_path / "attempt"
    _write_input(root, catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path, head=head,
                overrides={"fence": bad_fence})

    with pytest.raises(OpsError) as err:
        computed_moves_store.run_computed_moves_refresh(
            parameters, root, as_of=_AS_OF, fetcher=fetcher)
    _assert_refused_before_any_io(conn, fetcher, err=err)
    assert connect_calls == []


def test_run_computed_moves_refresh_refuses_fence_set_without_attempt_id(
        tmp_path, monkeypatch):
    """Issue #58: a bare ``fence`` with no ``attempt_id`` would make
    ``_fence_check_for`` a no-op (fail-open -- the commit goes through
    unfenced). Refused up front instead of silently no-op'ing."""
    conn, head, parameters, fetcher = _refusal_fixture(tmp_path)
    connect_calls = _connect_spy(monkeypatch)
    root = tmp_path / "attempt"
    _write_input(root, catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path, head=head,
                overrides={"fence": 3})

    with pytest.raises(OpsError) as err:
        computed_moves_store.run_computed_moves_refresh(
            parameters, root, as_of=_AS_OF, fetcher=fetcher)
    _assert_refused_before_any_io(conn, fetcher, err=err)
    assert connect_calls == []


def test_run_computed_moves_refresh_refuses_attempt_id_set_without_fence(
        tmp_path, monkeypatch):
    """Issue #58: a bare ``attempt_id`` with no ``fence`` would previously
    only fail later, inside ``verify_fence``, after the sqlite connection
    opens and fetches happen. Refused up front instead."""
    conn, head, parameters, fetcher = _refusal_fixture(tmp_path)
    connect_calls = _connect_spy(monkeypatch)
    root = tmp_path / "attempt"
    _write_input(root, catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path, head=head,
                overrides={"attempt_id": "attempt-1"})

    with pytest.raises(OpsError) as err:
        computed_moves_store.run_computed_moves_refresh(
            parameters, root, as_of=_AS_OF, fetcher=fetcher)
    _assert_refused_before_any_io(conn, fetcher, err=err)
    assert connect_calls == []


@pytest.mark.parametrize("bad_parent_snapshot_id", ["", "x" * 129, 7, None])
def test_run_computed_moves_refresh_refuses_a_bad_parent_snapshot_id(
        tmp_path, monkeypatch, bad_parent_snapshot_id):
    """``parameters.parent_snapshot_id`` is validated before ANY I/O -- even
    though a REAL, otherwise-valid input document exists here (proving the
    check fires before ``sqlite3.connect``, not because the document was
    missing: on 05fd8f7, this same setup let a bad parent_snapshot_id reach
    ``repository.resolve_full`` and raise an unrelated ``DataError``
    (``SNAPSHOT_NOT_FOUND``), not this typed ``OpsError``)."""
    conn, head, _base_parameters, fetcher = _refusal_fixture(tmp_path)
    connect_calls = _connect_spy(monkeypatch)
    root = tmp_path / "attempt"
    _write_input(root, catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path, head=head)
    parameters = _parameters(
        head, expected_ids=("CCCC",), catalog_path=tmp_path / "ops.sqlite",
        objects_root=tmp_path, overrides={"parent_snapshot_id": bad_parent_snapshot_id})

    with pytest.raises(OpsError) as err:
        computed_moves_store.run_computed_moves_refresh(
            parameters, root, as_of=_AS_OF, fetcher=fetcher)
    _assert_refused_before_any_io(conn, fetcher, err=err)
    assert connect_calls == []


@pytest.mark.parametrize("bad_refresh_plan_hash", [
    "", "not-a-hash", "sha256:" + "g" * 64, "sha256:" + "d" * 63, 7,
])
def test_run_computed_moves_refresh_refuses_a_bad_refresh_plan_hash(
        tmp_path, monkeypatch, bad_refresh_plan_hash):
    """``parameters.refresh_plan_hash`` is validated before ANY I/O -- even
    though a REAL, otherwise-valid input document exists here (proving the
    check fires before ``sqlite3.connect``, not because the document was
    missing: on 05fd8f7, this same setup did not raise at all -- the run
    completed with ``status="noop"`` and the garbage hash baked straight
    into the ``RefreshCallbackResult``)."""
    conn, head, _base_parameters, fetcher = _refusal_fixture(tmp_path)
    connect_calls = _connect_spy(monkeypatch)
    root = tmp_path / "attempt"
    _write_input(root, catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path, head=head)
    parameters = _parameters(
        head, expected_ids=("CCCC",), catalog_path=tmp_path / "ops.sqlite",
        objects_root=tmp_path, overrides={"refresh_plan_hash": bad_refresh_plan_hash})

    with pytest.raises(OpsError) as err:
        computed_moves_store.run_computed_moves_refresh(
            parameters, root, as_of=_AS_OF, fetcher=fetcher)
    _assert_refused_before_any_io(conn, fetcher, err=err)
    assert connect_calls == []


def _capture_population_scan(monkeypatch, repository, population_bound):
    captured = []
    original_scan = repository.scan
    monkeypatch.setattr(
        repository, "scan_population_bound", lambda *_a, **_k: population_bound)

    def _scan(query, **kwargs):
        captured.append(query)
        return original_scan(query, **kwargs)

    monkeypatch.setattr(repository, "scan", _scan)
    return captured


def _capture_scans(monkeypatch, repository):
    """Spy on ``repository.scan`` only -- the bound comes from the REAL
    ``scan_population_bound`` (spec PR 360: the positive lowered-limit case
    must not mock the bound)."""
    captured = []
    original_scan = repository.scan

    def _scan(query, **kwargs):
        captured.append(query)
        return original_scan(query, **kwargs)

    monkeypatch.setattr(repository, "scan", _scan)
    return captured


def _actual_population_bound(repository, snapshot, table_name: str) -> int:
    """The real ``scan_population_bound`` for the exact snapshot, table, and
    year predicate ``_scan_rows`` builds -- same contract ref, same key
    filter, no mocking."""
    contract_ref = snapshot.table_versions[table_name].table_contract_ref
    years = tuple(sorted({int(record.partition_key)
                          for record in repository.fragment_records(snapshot, table_name)}))
    key_filter = (KeyPredicate(column="year", operator="in", values=years),)
    return repository.scan_population_bound(
        snapshot.snapshot_id, table_name=table_name, table_contract_ref=contract_ref,
        key_filter=key_filter, time_interval=None)


def _scan_fixture(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    head = _build_parent(
        conn, clock, store,
        events_rows=[_event_row("AAAA", day) for day in _EVENT_DAYS])
    repository = Repository(conn, store)
    return repository, repository.resolve(head["snapshot_id"])


def _read_scan_rows(repository, snapshot, table_name: str, columns) -> list[dict]:
    """Materialize the leased stream ``_scan_rows`` yields: enter each
    ``RetainedBatch`` with ``with lease as batch``, copy its rows out while the
    lease is live (a released lease clears its list), then advance so the lease
    releases. A typed repository refusal lands mid-iteration and propagates
    unchanged out of this helper -- never caught, never retried."""
    rows: list[dict] = []
    for lease in computed_moves_store._scan_rows(repository, snapshot, table_name, columns):
        with lease as batch:
            rows.extend(batch)
    return rows


def test__scan_rows_keeps_limit_when_population_bound_is_at_or_above_current(tmp_path,
                                                                             monkeypatch):
    repository, snapshot = _scan_fixture(tmp_path)
    columns = ("event_id", "ticker", "event_date", "year", "src_orats")
    bound = _actual_population_bound(repository, snapshot, "earnings_events")
    assert bound > 0
    monkeypatch.setattr(computed_moves_store, "MAX_SCAN_ROWS", bound)
    baseline = _read_scan_rows(repository, snapshot, "earnings_events", columns)
    expected = computed_moves_store.MAX_SCAN_ROWS
    assert expected == bound
    captured = _capture_population_scan(monkeypatch, repository, expected)

    result = _read_scan_rows(repository, snapshot, "earnings_events", columns)

    assert result == baseline
    assert len(result) == len(_EVENT_DAYS)
    assert all(r["ticker"] == "AAAA" for r in result)
    assert {r["event_id"] for r in result} == {f"AAAA_{day.date()}" for day in _EVENT_DAYS}
    assert captured[0].max_result_rows == expected
    assert captured[0].max_batch_rows == min(_EVENTS.maximum_batch_rows, 50_000,
                                             computed_moves_store.MAX_SCAN_ROWS)
    assert captured[0].max_batch_rows <= captured[0].max_result_rows


def test__scan_rows_uses_smaller_selected_population_bound(tmp_path, monkeypatch):
    """PR 360's cap removal, proven against the REAL bound: the prepared
    query's ``max_result_rows`` is exactly ``min(existing_limit, bound)`` for
    the same snapshot, table, and year predicate, the batch limit is lowered
    only as needed to keep ``max_batch_rows <= max_result_rows``, and the
    returned rows are unchanged."""
    repository, snapshot = _scan_fixture(tmp_path)
    columns = ("event_id", "ticker", "event_date", "year", "src_orats")
    baseline = _read_scan_rows(repository, snapshot, "earnings_events", columns)
    bound = _actual_population_bound(repository, snapshot, "earnings_events")
    existing_limit = computed_moves_store.MAX_SCAN_ROWS
    assert 0 < bound < existing_limit
    captured = _capture_scans(monkeypatch, repository)

    result = _read_scan_rows(repository, snapshot, "earnings_events", columns)

    assert result == baseline
    assert len(result) == len(_EVENT_DAYS)
    assert all(r["ticker"] == "AAAA" for r in result)
    assert {r["event_id"] for r in result} == {f"AAAA_{day.date()}" for day in _EVENT_DAYS}
    assert captured[0].max_result_rows == min(existing_limit, bound)
    assert captured[0].max_batch_rows == min(_EVENTS.maximum_batch_rows, 50_000, bound)
    assert captured[0].max_batch_rows <= captured[0].max_result_rows


def test__scan_rows_committed_zero_row_year_fragment_is_a_valid_empty_result(tmp_path,
                                                                              monkeypatch):
    """Gate finding (#407): a partition that exists but holds zero rows is a
    valid EMPTY selected population, not a refusal and not a zero batch
    ceiling. A real committed ``earnings_events`` fragment with
    ``row_count == 0`` (the established hand-built shape of
    ``tests/test_v2_data_query.py::test_zero_row_fragment_metadata_has_zero_bound_and_scans_empty``
    -- ``publish_and_inspect`` itself refuses an empty partition, since a
    ``fragment_record`` needs non-empty primary-key bounds) keeps the year
    partition present, so ``_scan_rows`` does NOT take its no-fragments
    short-circuit: the real ``scan_population_bound`` for its exact year
    selection is 0, ``max_result_rows`` correctly lowers to 0, and the batch
    ceiling must stay positive. No refusal is swallowed: the real scan runs
    and simply yields no rows."""
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    zero_record = hand_built_record(
        store, _EVENTS, _EVENTS_REF, table_from_rows(_EVENTS, []),
        partition_key="2024", row_count=0,
        primary_key_min=("AAAA_2024-01-01",), primary_key_max=("ZZZZ_2024-12-31",))
    commit_tables(conn, clock, {"earnings_events": [zero_record]}, {"earnings_events": _EVENTS})

    repository = Repository(conn, store)
    snapshot = repository.resolve(_head_row(conn)["snapshot_id"])

    records = repository.fragment_records(snapshot, "earnings_events")
    assert len(records) == 1
    assert records[0].row_count == 0
    assert int(records[0].partition_key) == 2024  # the year partition really exists
    assert _actual_population_bound(repository, snapshot, "earnings_events") == 0

    captured = _capture_scans(monkeypatch, repository)
    columns = ("event_id", "ticker", "event_date", "year", "src_orats")

    result = _read_scan_rows(repository, snapshot, "earnings_events", columns)

    assert result == []
    assert len(captured) == 1
    assert captured[0].max_result_rows == 0
    assert captured[0].max_batch_rows == min(_EVENTS.maximum_batch_rows, 50_000) > 0


# --------------------------------------------------------------------------
# slice 2 (the wired scanner): ``_scan_rows`` now streams one pinned selection
# as ``ops.pinned_partition_reader`` ``RetainedBatch`` leases, so a selected
# population LARGER than ``MAX_SCAN_ROWS`` still arrives whole and in
# primary-key order while only one batch is ever retained. The shared lease
# cleanup contract that keeps this streaming safe -- a caller that edits a
# leased list still gets its creation-time charge discharged and the very
# list it mutated cleared -- is exercised by the already-merged
# tests/test_v2_ops_partition_consumer.py::test_release_discharges_the_creation_charge_after_the_list_is_edited,
# and, through the wired ``_scan_rows`` consumer itself, by
# test__scan_rows_discharges_the_creation_charge_of_an_edited_leased_batch below.
# --------------------------------------------------------------------------

_SCAN_COLUMNS = ("event_id", "ticker", "event_date", "year", "src_orats")


def _legacy_scan_rows(repository, snapshot, table_name: str, columns) -> list[dict]:
    """Test-local oracle of the DELETED pre-slice-2 ``_scan_rows``: the exact
    old full-materialization path -- pinned contract ref and table contract,
    sorted represented year partitions (empty means ``[]`` without scanning),
    the ``year in`` key filter, the manifest population bound,
    ``max_result_rows = min(MAX_SCAN_ROWS, population_bound)``, a batch ceiling
    of ``min(contract.maximum_batch_rows, 50_000)`` further clamped by the
    positive bound and the result limit, the primary-key ``DataQuery``, and
    flattened scan batches. It reads ``computed_moves_store.MAX_SCAN_ROWS``
    live, so a test can materialize the old path's expected rows BEFORE
    lowering the cap."""
    from engine.v2.contracts import DataQuery

    contract_ref = snapshot.table_versions[table_name].table_contract_ref
    contract = repository.table_contract(snapshot, table_name)
    years = tuple(sorted({int(record.partition_key)
                          for record in repository.fragment_records(snapshot, table_name)}))
    if not years:
        return []
    key_filter = (KeyPredicate(column="year", operator="in", values=years),)
    population_bound = repository.scan_population_bound(
        snapshot.snapshot_id, table_name=table_name, table_contract_ref=contract_ref,
        key_filter=key_filter, time_interval=None)
    max_result_rows = min(computed_moves_store.MAX_SCAN_ROWS, population_bound)
    max_batch_rows = min(contract.maximum_batch_rows, 50_000)
    if population_bound > 0:
        max_batch_rows = min(max_batch_rows, population_bound, max_result_rows)
    query = DataQuery(
        snapshot_id=snapshot.snapshot_id, table_contract_ref=contract_ref,
        columns=tuple(columns),
        key_filter=key_filter,
        order_by=tuple(contract.primary_key),
        max_batch_rows=max_batch_rows,
        max_result_rows=max_result_rows)
    rows: list[dict] = []
    for batch in repository.scan(query, table_name=table_name):
        rows.extend(batch.to_pylist())
    return rows


def test__scan_rows_streams_a_population_larger_than_max_scan_rows(tmp_path, monkeypatch):
    """One injected ``RetainedRowCount`` over the fixture's ``MAX_SCAN_ROWS + 1``
    unique ``earnings_events`` rows: every row comes back, in ``event_id``
    (contract primary-key) order, the result limit is the manifest
    selected-population bound -- never the retained cap -- the batch limit
    stays at or under the cap, and the account ends empty with its peak at one
    in-flight batch. A second complete read of the same pin returns an
    identical list. The streamed rows are also checked against the deleted
    pre-slice-2 scanner's own output (the ``_legacy_scan_rows`` oracle, run
    while the cap still exceeds the population) -- the new path is verified
    against the OLD behavior, never against itself."""
    repository, snapshot = _scan_fixture(tmp_path)  # len(_EVENT_DAYS) unique event_ids
    cap = len(_EVENT_DAYS) - 1  # the fixture holds exactly cap + 1 rows
    bound = _actual_population_bound(repository, snapshot, "earnings_events")
    assert bound == len(_EVENT_DAYS) > cap  # the population exceeds the retained cap
    assert computed_moves_store.MAX_SCAN_ROWS > bound  # the old cap still covers every row
    legacy_expected = _legacy_scan_rows(repository, snapshot, "earnings_events",
                                        _SCAN_COLUMNS)
    assert len(legacy_expected) == cap + 1  # the oracle really materialized the whole pin
    monkeypatch.setattr(computed_moves_store, "MAX_SCAN_ROWS", cap)
    captured = _capture_scans(monkeypatch, repository)
    account = RetainedRowCount()
    monkeypatch.setattr(computed_moves_store, "RetainedRowCount", lambda: account)

    rows = _read_scan_rows(repository, snapshot, "earnings_events", _SCAN_COLUMNS)
    second = _read_scan_rows(repository, snapshot, "earnings_events", _SCAN_COLUMNS)
    assert second == rows  # the same pin re-read complete and identical

    expected = sorted((_event_row("AAAA", day) for day in _EVENT_DAYS),
                      key=lambda row: row["event_id"])
    ids = [row["event_id"] for row in rows]
    assert len(set(ids)) == len(ids) == cap + 1  # unique keys, nothing dropped or added
    assert ids == sorted(ids) == [row["event_id"] for row in expected]  # primary-key order
    assert rows == [{column: row[column] for column in _SCAN_COLUMNS} for row in expected]
    assert rows == legacy_expected  # byte-for-byte what the deleted old path materialized
    assert account.live_rows == 0  # every lease discharged as the reader advanced
    assert account.peak_rows <= computed_moves_store.MAX_SCAN_ROWS  # one batch, never more
    assert account.peak_rows < bound  # retention stayed under the whole population
    assert account.peak_rows == captured[0].max_batch_rows  # exactly one lease live at a time
    assert len(captured) == 2  # both complete reads, each one scan
    assert captured[0].max_result_rows == bound  # the manifest bound, not the cap
    assert captured[0].max_batch_rows == min(_EVENTS.maximum_batch_rows, 50_000, cap)


def test__scan_rows_propagates_one_typed_limit_guard_and_retains_nothing(tmp_path,
                                                                         monkeypatch):
    """A pinned population bound below what the fragment actually holds: the
    repository's own ``RESULT_LIMIT_EXCEEDED`` is terminal -- one scan call, no
    retry, no fallback, no partial success -- and because it lands mid-stream
    (one lease was already handed out) the reader's cleanup must leave the
    injected account back at zero. ``MAX_SCAN_ROWS`` is lowered to two so the
    stream must hand out more than one leased batch before the guard fires: an
    exact peak of two -- never four -- proves the earlier batch was consumed
    and released on the way to the refusal, and ``rows`` staying ``None``
    proves nothing was ever materialized out of the failed scan."""
    repository, snapshot = _scan_fixture(tmp_path)
    real_bound = _actual_population_bound(repository, snapshot, "earnings_events")
    pinned_bound = real_bound - 1  # the manifest now under-reports the rows behind it
    assert pinned_bound > 0
    monkeypatch.setattr(computed_moves_store, "MAX_SCAN_ROWS", 2)
    assert real_bound > computed_moves_store.MAX_SCAN_ROWS  # the population exceeds the cap
    account = RetainedRowCount()
    monkeypatch.setattr(computed_moves_store, "RetainedRowCount", lambda: account)
    captured = _capture_population_scan(monkeypatch, repository, pinned_bound)

    rows = None
    with pytest.raises(DataError) as err:
        rows = _read_scan_rows(repository, snapshot, "earnings_events", _SCAN_COLUMNS)

    assert err.value.code == "RESULT_LIMIT_EXCEEDED"
    assert err.value.problem.retryable is False
    assert rows is None  # the refusal propagated out of the helper: nothing materialized
    assert len(captured) == 1  # exactly one scan, never a retry
    assert captured[0].max_result_rows == pinned_bound < real_bound
    assert captured[0].max_batch_rows == computed_moves_store.MAX_SCAN_ROWS == 2
    assert account.peak_rows == 2  # one in-flight batch: earlier ones were released first
    assert account.live_rows == 0  # ... and the live lease was discharged on the way out


def test__scan_rows_propagates_a_missing_pinned_contract_without_reaching_scan(tmp_path,
                                                                               monkeypatch):
    """The wired store's own R1 case: the reader's first move is the pinned
    contract lookup, so a typed ``CONTRACT_MISMATCH`` injected at
    ``repository.table_contract`` propagates out of the ``_read_scan_rows``
    helper unchanged and exactly once -- no scan is ever reached, no row result is
    ever assigned, and the fresh lease account stays empty."""
    repository, snapshot = _scan_fixture(tmp_path)
    problem = data_fail("CONTRACT_MISMATCH", "synthetic missing pinned contract",
                        details={"table_name": "earnings_events"})

    def _refuse(*_args, **_kwargs):
        raise problem

    monkeypatch.setattr(repository, "table_contract", _refuse)
    account = RetainedRowCount()
    monkeypatch.setattr(computed_moves_store, "RetainedRowCount", lambda: account)
    captured = _capture_scans(monkeypatch, repository)

    rows = None
    with pytest.raises(DataError) as err:
        rows = _read_scan_rows(repository, snapshot, "earnings_events", _SCAN_COLUMNS)

    assert err.value is problem  # the very same typed refusal, propagated once, unwrapped
    assert err.value.code == "CONTRACT_MISMATCH"
    assert rows is None  # no row result was ever assigned
    assert captured == []  # the refusal lands before any scan call -- never retried
    assert account.live_rows == 0


def test__scan_rows_discards_a_partial_read_on_a_mid_scan_integrity_failure(tmp_path,
                                                                            monkeypatch):
    """The wired store's own R2/R3/R5 case: the first REAL leased batch lands,
    then a pre-created ``MANIFEST_CORRUPT`` refusal interrupts the scan. The
    same exception object propagates -- exactly one scan call, never a retry --
    and because a partial read is provisional the result never becomes a row
    list, while the account is back at zero with its peak at the one in-flight
    two-row batch. Synthetic fixture only: nothing is published or mutated."""
    repository, snapshot = _scan_fixture(tmp_path)
    problem = data_fail("MANIFEST_CORRUPT", "synthetic mid-scan integrity failure")
    monkeypatch.setattr(computed_moves_store, "MAX_SCAN_ROWS", 2)
    assert problem.code == "MANIFEST_CORRUPT"
    assert problem.problem.retryable is False
    account = RetainedRowCount()
    monkeypatch.setattr(computed_moves_store, "RetainedRowCount", lambda: account)
    real_scan = repository.scan
    captured = []

    def _scan_once_then_fail(query, **kwargs):
        captured.append(query)
        batches = real_scan(query, **kwargs)
        yield next(batches)  # the first real batch, then the terminal refusal
        raise problem

    monkeypatch.setattr(repository, "scan", _scan_once_then_fail)

    result = None
    with pytest.raises(DataError) as err:
        result = _read_scan_rows(repository, snapshot, "earnings_events", _SCAN_COLUMNS)

    assert err.value is problem  # the repository's refusal, unchanged
    assert len(captured) == 1  # exactly one scan call, no retry
    assert captured[0].max_batch_rows == computed_moves_store.MAX_SCAN_ROWS == 2
    assert result is None  # the partial read never materialized into a list
    assert account.peak_rows == 2  # the one in-flight leased batch
    assert account.live_rows == 0  # discharged on the way out


def test__scan_rows_discharges_the_creation_charge_of_an_edited_leased_batch(tmp_path,
                                                                             monkeypatch):
    """Integration regression for caller mutation of a yielded lease list
    through the WIRED ``_scan_rows`` path -- not merely a direct
    ``RetainedBatch`` unit test (that one lives in
    ``tests/test_v2_ops_partition_consumer.py``). A generator wrapper
    monkeypatched over ``iter_pinned_scan_batches`` enters each real lease,
    copies the rows still to come, pops one row WHILE the lease is live --
    a deliberate violation of the no-resize rule -- and yields the same
    lease object to the production ``_scan_rows``. The fixed creation charge
    never moves with the edit; as the scanner advances, every saved lease
    list comes back cleared, its charge discharged, and the stream that
    lands is exactly the expected rows minus one popped row per batch (the
    cap lowered to two makes three batches)."""
    repository, snapshot = _scan_fixture(tmp_path)
    monkeypatch.setattr(computed_moves_store, "MAX_SCAN_ROWS", 2)
    account = RetainedRowCount()
    monkeypatch.setattr(computed_moves_store, "RetainedRowCount", lambda: account)
    expected = [{column: row[column] for column in _SCAN_COLUMNS}
                for row in sorted((_event_row("AAAA", day) for day in _EVENT_DAYS),
                                  key=lambda row: row["event_id"])]

    real_iter = computed_moves_store.iter_pinned_scan_batches
    saved_lists: list[list] = []
    charges: list[int] = []
    post_pop: list[dict] = []
    consumed = 0

    def _mutating_wrapper(*args, **kwargs):
        nonlocal consumed
        for lease in real_iter(*args, **kwargs):
            rows = lease.__enter__()  # the lease's actual mutable list
            remaining = expected[consumed:]  # a copy: the rows not yet handed out
            charge = len(rows)  # the original creation charge, fixed until release
            rows.pop()  # resize WHILE the lease is live -- the documented violation
            assert account.live_rows == charge  # the edit moved neither the charge...
            saved_lists.append(rows)  # the very list object _scan_rows is handed
            charges.append(charge)
            post_pop.extend(remaining[:charge - 1])  # ... nor what the scanner yields
            consumed += charge
            yield lease

    monkeypatch.setattr(computed_moves_store, "iter_pinned_scan_batches", _mutating_wrapper)

    rows = _read_scan_rows(repository, snapshot, "earnings_events", _SCAN_COLUMNS)

    assert charges == [2, 2, 1]  # three leased batches, one row popped in each
    assert rows == post_pop == [expected[0], expected[2]]  # exactly the post-pop stream
    assert saved_lists and all(saved == [] for saved in saved_lists)  # every list cleared
    assert account.live_rows == 0  # every fixed creation charge discharged
    assert account.peak_rows == 2  # the exact expected peak (max(charges) == 2)
    assert account.peak_rows <= computed_moves_store.MAX_SCAN_ROWS


def test__scan_once_keeps_only_orats_confirmed_events_that_carry_a_session(tmp_path,
                                                                            monkeypatch):
    """Slice 2 changed HOW rows arrive, not which rows are usable: the wired
    ``_scan_once`` must still apply the existing availability filter
    (``src_orats`` confirmed AND ``session`` present). One pin, three
    synthetic events: (a) a confirmed ORATS row with a session, (b) a row with
    ``src_orats=False``, and (c) a confirmed row with ``session=None`` -- only
    (a) may appear in the returned events frame. The committed-empty
    ``daily_market`` table is the established fixture shape.

    Cap wiring on the SAME production path: ``MAX_SCAN_ROWS`` is lowered to
    two, BELOW the three-row population, and every ``RetainedRowCount`` the
    two ``_scan_frame`` calls create plus every ``repository.scan`` query is
    captured. The events scan's batch limit is exactly two, its account peaks
    at exactly two (the population leased as two batches, one in flight at a
    time) and returns to zero, and the empty daily scan's account never
    leaves zero.

    Conversion-while-leased is proven DIRECTLY, never left to ``pd.DataFrame``
    consuming a generator: ``computed_moves_store.pd`` is replaced by a proxy
    (``DataFrame`` / ``concat`` / ``to_datetime``) whose ``DataFrame`` must be
    handed the CURRENT batch ``list`` -- its ticker rows exactly the next
    slice of the primary-key stream -- while the events account still carries
    the live charge ``live_rows == len(batch) <= MAX_SCAN_ROWS``, then
    delegates to the real constructor. The recorded batch lengths are the two
    events leases, ``[2, 1]``; the empty daily scan contributes none. A
    regression handing ``_scan_rows``' generator (or any list once its lease
    has discharged) to ``pd.DataFrame`` trips the proxy instead of passing
    silently."""
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    confirmed = _event_row("AAAA", _EVENT_DAYS[0])
    unconfirmed = dict(_event_row("BBBB", _EVENT_DAYS[2]), src_orats=False)
    sessionless = dict(_event_row("CCCC", _EVENT_DAYS[4]), session=None)
    head = _build_parent(conn, clock, store,
                         events_rows=[confirmed, unconfirmed, sessionless])
    repository = Repository(conn, store)
    snapshot = repository.resolve(head["snapshot_id"])

    monkeypatch.setattr(computed_moves_store, "MAX_SCAN_ROWS", 2)
    accounts: list[RetainedRowCount] = []

    def _capturing_account() -> RetainedRowCount:
        account = RetainedRowCount()
        accounts.append(account)
        return account

    monkeypatch.setattr(computed_moves_store, "RetainedRowCount", _capturing_account)
    captured = _capture_scans(monkeypatch, repository)

    from types import SimpleNamespace

    real_pd = computed_moves_store.pd
    converted_batches: list[int] = []
    batched_tickers = ["AAAA", "BBBB", "CCCC"]  # the pin in event_id (primary-key) order

    def _lease_live_frame(data=None, *args, **kwargs):
        if data is None:  # _scan_frame's no-chunks empty-frame call carries no batch
            return real_pd.DataFrame(*args, **kwargs)
        assert isinstance(data, list)  # a batch list, never a generator handed to pandas
        assert len(accounts) == 1  # only the events scan reaches a batch here
        assert (accounts[0].live_rows == len(data)
                <= computed_moves_store.MAX_SCAN_ROWS)  # conversion under the LIVE lease
        start = sum(converted_batches)
        assert [row["ticker"] for row in data] == batched_tickers[start:start + len(data)]
        converted_batches.append(len(data))
        return real_pd.DataFrame(data, *args, **kwargs)

    monkeypatch.setattr(computed_moves_store, "pd", SimpleNamespace(
        DataFrame=_lease_live_frame,
        concat=real_pd.concat,
        to_datetime=real_pd.to_datetime))

    events, daily = computed_moves_store._scan_once(repository, snapshot)

    assert list(events["ticker"]) == ["AAAA"]  # (b) and (c) never survive the filter
    assert list(events["session"]) == ["AMC"]
    assert list(events["src_orats"]) == [True]
    assert list(events["event_date"]) == [pd.Timestamp(_EVENT_DAYS[0])]
    assert daily.empty  # the empty daily table scans through unchanged

    assert converted_batches == [2, 1]  # the two events leases; the daily scan adds no chunk
    assert len(accounts) == 2  # one fresh account per _scan_frame: events, then daily
    assert len(captured) == 1  # the empty daily selection never reaches repository.scan
    assert captured[0].max_batch_rows == 2  # the events scan is held to the lowered cap
    assert accounts[0].peak_rows == 2  # three rows leased as two batches (2 + 1), never 3
    assert accounts[0].live_rows == 0  # every events lease discharged as its chunk built
    assert accounts[1].peak_rows == 0  # the empty daily scan's account ...
    assert accounts[1].live_rows == 0  # ... never left zero


def test__scan_frame_discards_its_chunks_on_a_mid_partition_failure(tmp_path, monkeypatch):
    """The batch-aware ``_scan_frame`` consumer, not just ``_scan_rows``: the
    first REAL leased batch lands (its frame chunk built while the lease is
    live), then the same pre-created ``MANIFEST_CORRUPT`` refusal interrupts
    the scan before a second batch. The exact typed error propagates --
    exactly one scan, no retry, and no frame result ever assigned -- while the
    local provisional chunk list is discarded on the unwind and the injected
    account ends discharged with its peak at the one in-flight two-row batch
    (``MAX_SCAN_ROWS`` lowered to two under the fixture's five-row pin).
    Synthetic ``tmp_path`` catalog data only; nothing is published."""
    repository, snapshot = _scan_fixture(tmp_path)
    problem = data_fail("MANIFEST_CORRUPT", "synthetic mid-partition failure")
    monkeypatch.setattr(computed_moves_store, "MAX_SCAN_ROWS", 2)
    assert problem.code == "MANIFEST_CORRUPT"
    account = RetainedRowCount()
    monkeypatch.setattr(computed_moves_store, "RetainedRowCount", lambda: account)
    real_scan = repository.scan
    captured = []

    def _scan_once_then_fail(query, **kwargs):
        captured.append(query)
        batches = real_scan(query, **kwargs)
        yield next(batches)  # exactly the first real batch, then the terminal refusal
        raise problem

    monkeypatch.setattr(repository, "scan", _scan_once_then_fail)

    frame = None
    with pytest.raises(DataError) as err:
        frame = computed_moves_store._scan_frame(repository, snapshot, "earnings_events",
                                                 _SCAN_COLUMNS)

    assert err.value is problem  # the very same pre-created refusal, propagated unwrapped
    assert len(captured) == 1  # exactly one scan, never a retry
    assert frame is None  # the partial read never became a frame result
    assert account.peak_rows == 2  # the one in-flight leased batch, cap-sized
    assert account.live_rows == 0  # discharged as the reader unwound


# --------------------------------------------------------------------------
# issue #179 (contract: engine/v2/ops/ARCHITECTURE.md, "capture and inherited
# fragments respect as_of"): a run at an EARLIER as_of that would inherit a
# committed fragment reaching on/after that as_of must be refused -- the
# capture-time truncation only bounds rows THIS run writes.
# --------------------------------------------------------------------------


def test_earlier_as_of_refuses_inherited_future_dated_fragment(tmp_path):
    """Run A captures ``BBBB`` at the later ``_AS_OF`` and commits a fragment
    whose ``event_date`` rows extend past the earlier ``as_of`` run B will
    request. Run B pins run A's committed snapshot as its parent and targets
    only ``AAAA``, so only carry-forward could preserve that fragment: the
    contract refuses the whole generation with a non-retryable
    ``VALIDATION_FAILED``, leaving the head at run A's snapshot and its
    planted row intact."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        targets = ["BBBB"]
        monkeypatch.setattr(computed_moves_store, "target_tickers_from_snapshot",
                            lambda *a, **k: (list(targets), {}))
        conn, clock, _ = catalog(tmp_path)
        store = ArtifactStore(tmp_path)
        events_rows = ([_event_row("AAAA", d) for d in _BDAYS[:5]]  # 5 events before 2024-01-25
                       + [_event_row("BBBB", d) for d in _EVENT_DAYS])  # planted 2024-01-26 tail
        head = _build_parent(conn, clock, store, events_rows=events_rows)

        fetcher = _CountingFetcher(_closes_csv())
        root_a = tmp_path / "attempt_a"
        _write_input(root_a, catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path,
                     head=head)
        parameters_a = _parameters(head, expected_ids=("BBBB",),
                                   catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path)
        first = computed_moves_store.run_computed_moves_refresh(
            parameters_a, root_a, as_of=_AS_OF, fetcher=fetcher)
        assert first.status == "complete"  # run A really committed the tail

        # Confirm the regressed inheritance from the committed parent, via
        # the same DataQuery reader the issue #99 test uses: at least one
        # BBBB row already reaches on/after the earlier as_of run B asks for.
        committed_parent = dict(_head_row(conn))
        earlier_as_of = "2024-01-25"  # < _AS_OF, < the planted tail date
        planted = str(_EVENT_DAYS[-1].date())

        from engine.v2.contracts import DataQuery
        from engine.v2.data.computed_moves_table import COMPUTED_MOVES_TABLE_NAME

        repository = Repository(conn, store)
        snapshot = repository.resolve(committed_parent["snapshot_id"])
        dvr = snapshot.table_versions[COMPUTED_MOVES_TABLE_NAME]
        key_filter = (KeyPredicate(column="ticker", operator="eq", values=("BBBB",)),)
        bound = repository.scan_population_bound(
            snapshot.snapshot_id, table_name=COMPUTED_MOVES_TABLE_NAME,
            table_contract_ref=dvr.table_contract_ref, key_filter=key_filter)
        result_limit = min(100, bound)
        batch_limit = min(100, result_limit) if result_limit > 0 else 100
        query = DataQuery(
            snapshot_id=snapshot.snapshot_id, table_contract_ref=dvr.table_contract_ref,
            columns=("event_date",),
            key_filter=key_filter,
            order_by=("ticker", "event_date"),
            max_batch_rows=batch_limit, max_result_rows=result_limit)
        bbbb_dates = [str(r["event_date"])
                      for batch in repository.scan(query, table_name=COMPUTED_MOVES_TABLE_NAME)
                      for r in batch.to_pylist()]
        assert planted in bbbb_dates
        assert any(d >= earlier_as_of for d in bbbb_dates)

        targets[:] = ["AAAA"]  # BBBB survives run B only by carry-forward
        root_b = tmp_path / "attempt_b"
        _write_input(root_b, catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path,
                     head=committed_parent, as_of=earlier_as_of)
        parameters_b = _parameters(committed_parent, expected_ids=("AAAA",),
                                   catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path)

        with pytest.raises(OpsError) as err:
            computed_moves_store.run_computed_moves_refresh(
                parameters_b, root_b, as_of=earlier_as_of, fetcher=fetcher)
        assert err.value.code == "VALIDATION_FAILED"
        assert err.value.problem.retryable is False

        # The refusal rewrote nothing: head is still run A's committed
        # snapshot, and that snapshot's BBBB rows still carry the planted date.
        assert dict(_head_row(conn)) == committed_parent
        still = [str(r["event_date"])
                 for batch in repository.scan(query, table_name=COMPUTED_MOVES_TABLE_NAME)
                 for r in batch.to_pylist()]
        assert planted in still
