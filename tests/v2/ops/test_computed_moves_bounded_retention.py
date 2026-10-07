"""Bounded retained source rows in the ``computed_moves`` reader (issue #362).

Synthetic data only. A small multi-ticker history with nulls, corrected rows
and date-boundary cases is run end to end; the per-ticker committed-row digests
and the selected targets are pinned to what the PRE-change whole-history reader
produced for the same fixture, so any drift in the bounded path fails here.
The same fixture then runs under a guard far below its size, and
instrumentation (the shared live-row account) shows the rows retained at once
stay within ``MAX_SCAN_ROWS`` however much history sits behind them.
"""
from __future__ import annotations

import gc
import hashlib

import pandas as pd
import pytest

from engine.v2.contracts import DataQuery, KeyPredicate
from engine.v2.data.computed_moves_table import COMPUTED_MOVES_TABLE_NAME
from engine.v2.data.errors import DataError, fail as data_fail
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactStore, canonical_json
from engine.v2.ops import computed_moves_store
from engine.v2.ops.errors import OpsError
from engine.v2.ops.pinned_partition_reader import RetainedRowCount
from tests.data_scan_support import commit_tables, publish_and_inspect
from tests.ops_support import catalog
from tests.test_v2_ops_computed_moves_store import (
    _DAILY, _DAILY_REF, _EVENTS, _EVENTS_REF, _event_row, _head_row, _parameters, _write_input)

_AS_OF = "2024-09-16"
_GRID = pd.bdate_range("2024-01-02", "2024-09-30")
_AS_OF_IDX = int(_GRID.get_loc(pd.Timestamp(_AS_OF)))

#: ticker -> (qualifying ORATS events before as_of, has daily_market rows, has history).
_PLAN = {
    "A": (14, True, True),    # ordinary; prefix of "AA"/"AB" (event_id order interleaves tickers)
    "AA": (12, True, True),   # exactly MIN_SCOREABLE, plus events ON and AFTER as_of
    "AB": (11, True, True),   # one short; a non-ORATS and a null-session event never count
    "B": (13, False, True),   # scoreable but no daily_market rows
    "C": (13, True, True),    # history starts ON its first event date; implied_move all null
    "D": (14, True, False),   # provider has no history: no_history, no fragment
    "E": (12, True, True),    # corrected events (date_conflict / updated_at / reconciliation)
}
_EXPECTED_TARGETS = ["A", "AA", "C", "D", "E"]
_FRAGMENT_TICKERS = ["A", "AA", "C", "E"]


def _events_for(ticker: str, qualifying: int) -> list[dict]:
    rows = []
    for k in range(qualifying):
        row = _event_row(ticker, _GRID[10 + 12 * k])
        row["session"] = "AMC" if k % 2 else "BMO"
        rows.append(row)
    if ticker == "AA":
        rows += [_event_row(ticker, _GRID[_AS_OF_IDX]),           # on as_of: never "past"
                 _event_row(ticker, _GRID[_AS_OF_IDX + 3])]       # after as_of
    if ticker == "AB":
        rows += [{**_event_row(ticker, _GRID[10 + 12 * 11]), "src_orats": False},
                 {**_event_row(ticker, _GRID[10 + 12 * 11 + 2]), "session": None}]
    if ticker == "E":
        for row in rows[::3]:
            row.update(date_conflict=True, updated_at="2024-08-01T00:00:00",
                       reconciliation="corrected")
    return rows


def _daily_for(ticker: str) -> list[dict]:
    template = {column.name: None for column in _DAILY.columns if column.nullable}
    offset = sum(map(ord, ticker))
    rows = []
    for i, day in enumerate(_GRID):
        implied = (None if ticker == "C" or i % 5 == 0
                   else round(2 + ((i * 7 + offset) % 11) * 0.25, 2))
        rows.append({**template, "ticker": ticker, "date": day.to_pydatetime(),
                     "year": 2024, "implied_move": implied})
    return rows


def _history_csv(ticker: str) -> bytes:
    start = _GRID[10] if ticker == "C" else pd.Timestamp("2023-12-01")
    offset = sum(map(ord, ticker)) % 7
    days = pd.bdate_range(start, "2024-12-31")  # runs past as_of: the fetch must be truncated
    lines = ["Date,Close"] + [f"{d.date()},{50 + (i * (3 + offset)) % 17 + i * 0.1:.4f}"
                              for i, d in enumerate(days)]
    return "\n".join(lines).encode()


def _fetcher(ticker: str):
    if not _PLAN.get(ticker, (0, 0, True))[2]:
        return b"", "legitimate_empty", {}, None
    return _history_csv(ticker), "complete", {}, None


def _bulk_tickers(count: int) -> list[str]:
    return [f"S{index:03d}" for index in range(count)]


def _build_parent(conn, clock, store, *, bulk: int = 0) -> dict:
    """The golden history, plus ``bulk`` extra scoreable tickers of the same shape."""
    plan = dict(_PLAN, **{ticker: (12, True, True) for ticker in _bulk_tickers(bulk)})
    events = sorted((row for ticker, (n, _, _) in plan.items() for row in _events_for(ticker, n)),
                    key=lambda row: row["event_id"])
    daily = sorted((row for ticker, (_, has, _) in plan.items() if has
                    for row in _daily_for(ticker)), key=lambda row: (row["ticker"], row["date"]))
    tables = {"earnings_events": [publish_and_inspect(store, _EVENTS, _EVENTS_REF, events, "2024")],
              "daily_market": [publish_and_inspect(store, _DAILY, _DAILY_REF, daily, "2024")]}
    commit_tables(conn, clock, tables, {"earnings_events": _EVENTS, "daily_market": _DAILY},
                  store=store)
    head = _head_row(conn)
    from engine.v2.data import reference_catalog
    from engine.v2.ops.catalog import transaction

    receipt_id = reference_catalog.committed_receipt_for_snapshot(
        conn, scope="shadow", snapshot_id=head["snapshot_id"])
    pin = reference_catalog.ReferenceInput(
        kind="calendar", legacy_path="calendar/trading_days.csv",
        object_id="object-reference-calendar", content_hash="sha256:" + "a" * 64, byte_size=1)
    with transaction(conn):
        reference_catalog.insert_reference_inputs(conn, receipt_id, [pin])
    return dict(head)


def _setup(tmp_path, *, bulk: int = 0):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    head = _build_parent(conn, clock, store, bulk=bulk)
    return conn, store, head, Repository(conn, store)


def _refresh(tmp_path, head, repository, fetcher=_fetcher):
    targets, report = computed_moves_store.target_tickers_from_snapshot(
        repository, head["snapshot_id"], as_of=_AS_OF)
    root = tmp_path / "attempt"
    _write_input(root, catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path, head=head,
                 as_of=_AS_OF)
    parameters = _parameters(head, expected_ids=tuple(targets),
                             catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path)
    result = computed_moves_store.run_computed_moves_refresh(
        parameters, root, as_of=_AS_OF, fetcher=fetcher)
    return targets, report, result


def _run(tmp_path, *, bulk: int = 0):
    conn, store, head, repository = _setup(tmp_path, bulk=bulk)
    return (conn, store, head, repository, *_refresh(tmp_path, head, repository))


def _digest(rows) -> str:
    return hashlib.sha256(canonical_json(rows).encode()).hexdigest()


def _committed(conn, store, snapshot_id) -> dict[str, tuple[int, str]]:
    """``{ticker: (row count, digest of its every committed column)}`` read back from the new head."""
    repository = Repository(conn, store)
    snapshot = repository.resolve(snapshot_id)
    version = snapshot.table_versions[COMPUTED_MOVES_TABLE_NAME]
    out = {}
    for record in repository.fragment_records(snapshot, COMPUTED_MOVES_TABLE_NAME):
        key_filter = (KeyPredicate(column="ticker", operator="eq", values=(record.partition_key,)),)
        bound = repository.scan_population_bound(
            snapshot.snapshot_id, table_name=COMPUTED_MOVES_TABLE_NAME,
            table_contract_ref=version.table_contract_ref, key_filter=key_filter,
            time_interval=None)
        query = DataQuery(
            snapshot_id=snapshot.snapshot_id, table_contract_ref=version.table_contract_ref,
            columns=computed_moves_store.FRAGMENT_COLUMNS, key_filter=key_filter,
            order_by=("ticker", "event_date"), max_batch_rows=min(100, bound),
            max_result_rows=bound)
        rows = [row for batch in repository.scan(query, table_name=COMPUTED_MOVES_TABLE_NAME)
                for row in batch.to_pylist()]
        out[record.partition_key] = (len(rows), _digest(rows))
    return out


def _outcomes(conn) -> dict[str, str]:
    return {row["ticker"]: row["outcome"] for row in conn.execute(
        "SELECT ticker, outcome FROM data_computed_moves_captures")}


#: Captured from the pre-change whole-history reader over this exact fixture.
_GOLDEN_REPORT = {"mode": "all_scoreable", "scoreable_on_orats_calendar": 6, "also_in_oquants": 0,
                  "no_daily_market_rows": 1, "targets": 5}
_GOLDEN_ROWS = {
    "A": (14, "50cade2d51ca63f6e8514ea1945cb140b5b5e8a84317f949770f0873d290f2c1"),
    "AA": (12, "df833c5995b1c599dce4184f7be37cc72a65cce82119eeda9f2267dde07921aa"),
    "C": (13, "4f2d835c98a1945957ea33d101798907ea46c929c00ffe51c77b3932d69776d4"),
    "E": (12, "ca14eb66f19135530963117c5f7cb258abcfbbe3e7828ff5e7baf4af31d30c08"),
}
_GOLDEN_OUTCOMES = {"A": "added", "AA": "added", "C": "added", "D": "no_history", "E": "added"}
_GOLDEN_ROWS_TOTAL = sum(count for count, _ in _GOLDEN_ROWS.values())


def _assert_golden(conn, store, targets, report, result, extra=()):
    assert targets == sorted(_EXPECTED_TARGETS + list(extra))
    assert result.status == "complete"
    committed = _committed(conn, store, result.candidate_snapshot_id)
    assert {t: committed[t] for t in _FRAGMENT_TICKERS} == _GOLDEN_ROWS
    assert sorted(committed) == sorted(_FRAGMENT_TICKERS + list(extra))
    assert {t: o for t, o in _outcomes(conn).items() if t not in extra} == _GOLDEN_OUTCOMES
    return report


def test_golden_history_outputs_match_the_pre_change_reader(tmp_path):
    """(a) Nulls (null session, null ``implied_move``), corrected events, events on and
    after ``as_of``, a history starting on an event date, a one-short ticker and a
    ticker with no daily rows: targets, report, every committed column and every
    capture outcome equal the whole-history reader's."""
    conn, store, _head, _repo, targets, report, result = _run(tmp_path)

    assert _assert_golden(conn, store, targets, report, result) == _GOLDEN_REPORT


def test_selection_from_streamed_counters_matches_whole_frame_selection(tmp_path):
    """The counter path and the caller-frames path agree, including at the ``since`` boundary."""
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    head = _build_parent(conn, clock, store)
    repository = Repository(conn, store)
    snapshot = repository.resolve(head["snapshot_id"])
    events, daily = computed_moves_store._scan_once(repository, snapshot)
    boundary = str(_GRID[10 + 12 * 11].date())  # AA's/E's/A's 12th event: >= since includes it

    for since in (None, boundary, str(_GRID[10 + 12 * 11 + 1].date())):
        for all_scoreable in (True, False):
            kwargs = dict(all_scoreable=all_scoreable, since=since, oquants_tickers=("A",),
                          as_of=_AS_OF)
            streamed = computed_moves_store.target_tickers_from_snapshot(
                repository, head["snapshot_id"], **kwargs)
            framed = computed_moves_store.target_tickers_from_snapshot(
                repository, head["snapshot_id"], events=events, daily=daily, **kwargs)
            assert streamed == framed
    assert computed_moves_store.target_tickers_from_snapshot(
        repository, head["snapshot_id"], since=boundary, as_of=_AS_OF)[0] == ["A", "AA", "C", "D", "E"]


def test_history_larger_than_the_guard_completes_with_identical_outputs(tmp_path, monkeypatch):
    """(b) The fixture holds ~1,100 source rows; the guard is 600. Bounded processing
    completes and reproduces the golden outputs exactly."""
    monkeypatch.setattr(computed_moves_store, "MAX_SCAN_ROWS", 600)
    conn, store, head, repository, targets, report, result = _run(tmp_path)
    population = _population(repository, head)

    assert population > computed_moves_store.MAX_SCAN_ROWS
    assert _assert_golden(conn, store, targets, report, result) == _GOLDEN_REPORT


def _population(repository, head) -> int:
    snapshot = repository.resolve(head["snapshot_id"])
    return sum(record.row_count for name in ("earnings_events", "daily_market")
               for record in repository.fragment_records(snapshot, name))


def _frames_now() -> list:
    """Strong references (so ids stay unique) to every DataFrame alive right now."""
    gc.collect()
    return [obj for obj in gc.get_objects() if isinstance(obj, pd.DataFrame)]


def _frame_rows_since(baseline) -> int:
    """Rows alive in DataFrames that did not exist when ``baseline`` was taken, so frames
    other tests leave alive in a shared process never count."""
    known = {id(frame) for frame in baseline}
    return sum(len(obj) for obj in gc.get_objects()
               if isinstance(obj, pd.DataFrame) and id(obj) not in known)


def _spy_live_frame_rows(monkeypatch) -> list[int]:
    """Rows alive in ANY run-created DataFrame each time a ticker's rows are built --
    measured with ``gc``, independent of the store's own live-row accounting."""
    peaks: list[int] = []
    baseline = _frames_now()
    real = computed_moves_store.build_rows

    def spy(*args, **kwargs):
        peaks.append(_frame_rows_since(baseline))
        return real(*args, **kwargs)

    monkeypatch.setattr(computed_moves_store, "build_rows", spy)
    return peaks


def test_no_frame_is_alive_while_the_next_chunk_loads(tmp_path, monkeypatch):
    """The previous chunk's frames, including the capture loop's own reference to a
    ticker's events, are released before the next chunk's scan starts, so nothing escapes
    the retained-row account (CodeRabbit round 2)."""
    monkeypatch.setattr(computed_moves_store, "MAX_SCAN_ROWS", 600)  # one ticker per chunk
    alive_at_load: list[int] = []
    baseline = _frames_now()
    real = computed_moves_store._scan_once

    def spy(*args, **kwargs):
        gc.collect()
        alive_at_load.append(_frame_rows_since(baseline))
        return real(*args, **kwargs)

    monkeypatch.setattr(computed_moves_store, "_scan_once", spy)
    _run(tmp_path)

    assert len(alive_at_load) > 1  # several chunks really loaded
    assert alive_at_load == [0] * len(alive_at_load)


@pytest.mark.parametrize("bulk", [0, 10, 30])
def test_retained_input_rows_do_not_grow_with_total_history(tmp_path, monkeypatch, bulk):
    """(c) Total history grows ~7x across the cases; the rows retained at once never
    exceed the guard, and past the guard they are a small fraction of the history.
    Two instruments: ONE shared live-row account (its peak is the chunk frames plus
    the in-flight lease) and a ``gc`` count of rows alive in DataFrames at every
    per-ticker build, which the pre-change whole-table frames cannot satisfy."""
    guard = 600
    monkeypatch.setattr(computed_moves_store, "MAX_SCAN_ROWS", guard)
    account = RetainedRowCount()
    monkeypatch.setattr(computed_moves_store, "RetainedRowCount", lambda: account)
    frame_rows = _spy_live_frame_rows(monkeypatch)
    conn, store, head, repository, targets, _report, result = _run(tmp_path, bulk=bulk)
    population = _population(repository, head)

    assert frame_rows  # the spy really fired, once per fragment-producing ticker
    assert account.peak_rows <= guard
    assert account.live_rows <= guard
    assert max(frame_rows) <= guard
    if bulk:
        assert population > 4 * guard
        assert account.peak_rows * 4 < population
        assert max(frame_rows) * 4 < population
    committed = _committed(conn, store, result.candidate_snapshot_id)
    assert {t: committed[t] for t in _FRAGMENT_TICKERS} == _GOLDEN_ROWS  # golden unaffected
    assert targets == sorted(_EXPECTED_TARGETS + _bulk_tickers(bulk))


def test_one_ticker_above_the_guard_is_refused_before_any_fetch_or_commit(tmp_path, monkeypatch):
    monkeypatch.setattr(computed_moves_store, "MAX_SCAN_ROWS", 300)  # a ticker weighs ~210 rows
    conn, _store, head, repository = _setup(tmp_path)
    calls = []

    with pytest.raises(OpsError) as err:
        _refresh(tmp_path, head, repository, fetcher=lambda t: calls.append(t) or _fetcher(t))

    assert err.value.code == "RESOURCE_LIMIT_EXCEEDED"
    assert calls == []  # refused before any provider call
    assert conn.execute("SELECT COUNT(*) AS n FROM data_computed_moves_captures"
                        ).fetchone()["n"] == 0
    assert _head_row(conn)["snapshot_id"] == head["snapshot_id"]  # head unmoved


def test_source_refusal_outranks_a_bad_as_of_and_a_bad_as_of_is_still_refused(monkeypatch):
    sentinel = data_fail("CONTRACT_MISMATCH", "synthetic missing pinned table")

    def _refuse(*_args, **_kwargs):
        raise sentinel

    class _Repo:
        resolve = staticmethod(lambda snapshot_id: object())

    monkeypatch.setattr(computed_moves_store, "_scan_stats", _refuse)
    with pytest.raises(DataError) as err:
        computed_moves_store.target_tickers_from_snapshot(_Repo(), "snap", as_of="not-a-date")
    assert err.value is sentinel  # the scan refusal, exactly as before

    with pytest.raises(OpsError) as bad:
        computed_moves_store.target_tickers_from_snapshot(
            _Repo(), "snap", events=pd.DataFrame(), daily=pd.DataFrame(), as_of="not-a-date")
    assert bad.value.code == "INVALID_REQUEST"
