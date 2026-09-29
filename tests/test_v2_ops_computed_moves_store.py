"""Direct tests for ``computed_moves_store.py``'s helpers, and an end-to-end
test of ``run_computed_moves_refresh`` itself: a complete unit, a
same-session cached rerun that re-fetches nothing, and a provider failure
mapped to its typed code. The store is still unwired to the nightly job graph
(see ``engine/v2/ops/ARCHITECTURE.md``'s "not yet wired" note) -- these tests
call it directly, the same way its eventual worker will."""
from __future__ import annotations

import pandas as pd
import pytest

from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactStore, canonical_json
from engine.v2.ops import computed_moves_store
from engine.v2.ops.catalog import transaction
from engine.v2.ops.computed_moves_store import _capture_id_for, _fence_check_for
from engine.v2.ops.errors import OpsError
from engine.v2.ops.incremental_data import RefreshParameters, RefreshUnit
from tests.data_scan_support import commit_tables, contract_for, contract_ref_for, publish_and_inspect
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
    return _head_row(conn)


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
    query = DataQuery(
        snapshot_id=snapshot.snapshot_id, table_contract_ref=dvr.table_contract_ref,
        columns=("event_date", "realized_move_pct", "skipped"),
        key_filter=(KeyPredicate(column="ticker", operator="eq", values=("AAAA",)),),
        order_by=("ticker", "event_date"), max_batch_rows=100, max_result_rows=100)
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
