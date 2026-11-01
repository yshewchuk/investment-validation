"""Synthetic pinned-source preparation, faithful metadata and isolated writes."""
import weakref
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from engine.v2.contracts.data import KeyPredicate
from engine.v2.data import objects
from engine.v2.data.errors import DataError, fail
from engine.v2.data.repository import Repository
from engine.v2.ops.workspace.overlay import create_workspace
from engine.v2.research.iv_crush_inputs import load_iv_crush_targets
from engine.v2.research.snapshot_rows import iter_snapshot_rows, read_snapshot_rows
from experiments import iv_crush_prepare
from experiments.iv_crush_prepare import prepare_iv_crush_targets
from tests.data_scan_support import (
    catalog_and_store,
    commit_tables,
    contract_for,
    contract_ref_for,
    publish_and_inspect,
)
from tests.v2.research.magnitude_trial_support import history_event

CAPS = dict(max_rows=100, max_decoded_bytes=1000000, max_object_bytes=1000000, max_total_bytes=1000000)


@contextmanager
def workspace(tmp_path, *, events=None, daily=None, fragment_rows=None):
    root = tmp_path / "source"
    root.mkdir()
    conn, clock, store = catalog_and_store(root)
    events = events if events is not None else [history_event("2024-01-04"),
        history_event("2024-02-06", session="BMO"), history_event("2024-03-05", protected=True)]
    daily = daily if daily is not None else [dict(ticker="SYNTH", date=datetime.fromisoformat(day),
        year=2024, iv30=iv) for day, iv in (("2024-01-04", 50), ("2024-01-05", 30),
        ("2024-02-05", 40), ("2024-02-06", 60), ("2024-03-05", 90), ("2024-03-06", 1))]
    contracts = {name: contract_for(name) for name in ("earnings_events", "daily_market")}
    tables, hashes = {}, {}
    for name, supplied in (("earnings_events", events), ("daily_market", daily)):
        contract = contracts[name]
        ordered = sorted(supplied, key=lambda r: tuple(r[k] for k in contract.primary_key))
        size = fragment_rows or len(ordered)
        tables[name] = [publish_and_inspect(store, contract, contract_ref_for(contract),
            ordered[start:start + size], "2024") for start in range(0, len(ordered), size)]
        hashes[name] = {"2024": objects.partition_logical_hash(store,
            [r.object_ref for r in tables[name]], contract, contract_ref_for(contract), "2024")}
    source = commit_tables(conn, clock, tables, contracts, partition_logical_hashes=hashes, store=store)
    conn.close()
    with create_workspace(tmp_path / "work", production_catalog=root / "catalog.sqlite",
            production_store=root / "store", snapshot_id=source.snapshot_id,
            manifest_hash=source.manifest_hash, clock=clock) as work:
        yield work, root


def run(work, **changes):
    return prepare_iv_crush_targets(work, as_of_month="2025-01", **{**CAPS, **changes})


def rows(repository, snapshot, table):
    contract = repository.table_contract(snapshot, table)
    column = contract.partition_columns[0]
    values = tuple(sorted({int(r.partition_key) if column == "year" else r.partition_key
        for r in repository.fragment_records(snapshot, table)}))
    return read_snapshot_rows(repository, snapshot, table, [c.name for c in contract.columns],
        key_filter=(KeyPredicate(column=column, operator="in", values=values),),
        **{k: v for k, v in CAPS.items() if k != "max_total_bytes"})


def state(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in Path(root).rglob("*") if p.is_file()}


def test_amc_bmo_signs_holdout_exclusion_source_unchanged_and_reader_cutoff(tmp_path):
    with workspace(tmp_path) as (work, root):
        before = state(root)
        result = run(work)
        assert state(root) == before
        repository = Repository(work.conn, work.store)
        target = result["target_snapshot"]
        assert result["source_snapshot"] == work.source_snapshot()
        assert target.snapshot_id != work.snapshot_id
        assert set(target.table_versions) == {"earnings_events", "iv_crush_targets"}
        assert result["coverage"] == dict(selected_events=2, available=2, skipped_by_reason={})
        assert len(rows(repository, target, "earnings_events")) == 3
        values = rows(repository, target, "iv_crush_targets")
        assert [r["crush_pct_iv30"] for r in values] == pytest.approx([-40, 50])
        assert [(r["pre_date"], r["post_date"]) for r in values] == [
            ("2024-01-04", "2024-01-05"), ("2024-02-05", "2024-02-06")]
        assert [r["pre_iv30"] for r in values] == [50, 40]
        assert run(work) == result
        loaded = load_iv_crush_targets(repository, target, as_of_month="2025-01", labels_before="2024-07-01")
        assert len(loaded) == 2
        with pytest.raises(DataError) as error:
            load_iv_crush_targets(repository, target, as_of_month="2025-01", labels_before="2024-01-06",
                                 event_ids=[history_event("2024-02-06", session="BMO")["event_id"]])
        assert error.value.code == "CONTRACT_MISMATCH"


def test_destination_repository_pins_exact_snapshot_without_changing_publication(tmp_path, monkeypatch):
    with workspace(tmp_path) as (work, root):
        before = state(root)
        with monkeypatch.context() as control:
            control.setattr(iv_crush_prepare, "Repository",
                lambda conn, store, **kwargs: Repository(conn, store, pin_snapshots=False))
            expected = run(work)
        opened = []
        def destination(conn, store, **kwargs):
            assert conn is work.conn and store is work.store
            repository = Repository(conn, store, **kwargs)
            opened.append(repository)
            return repository
        monkeypatch.setattr(iv_crush_prepare, "Repository", destination)
        actual = run(work)
        assert len(opened) == 1 and opened[0].pins_snapshots
        assert actual == expected
        assert actual["source_snapshot"] == work.source_snapshot()
        assert actual["target_snapshot"] != actual["source_snapshot"]
        assert opened[0].resolve(actual["target_snapshot"].snapshot_id) == actual["target_snapshot"]
        assert state(root) == before


def test_aliases_and_nullable_large_integer_metadata_are_preserved(tmp_path):
    good = history_event("2024-01-04", claim_count=9007199254740993)
    duplicate = history_event("2024-02-06")
    alias = {**duplicate, "event_id": "separate-alias", "claim_count": None}
    with workspace(tmp_path, events=[good, duplicate, alias]) as (work, _):
        result = run(work)
        repository = Repository(work.conn, work.store)
        metadata = rows(repository, result["target_snapshot"], "earnings_events")
        assert metadata == rows(work.source, result["source_snapshot"], "earnings_events")
        assert next(r for r in metadata if r["event_id"] == good["event_id"])["claim_count"] == 9007199254740993
        assert result["coverage"]["selected_events"] == 1


@pytest.mark.parametrize("quote,expected,reason", [(None, None, "invalid_post_iv30"),
    (0, -100, None), (-20, -140, None)])
def test_adjacent_invalid_and_nonpositive_quotes_are_not_skipped(tmp_path, quote, expected, reason):
    daily = [dict(ticker="SYNTH", date=datetime.fromisoformat(day), year=2024, iv30=value)
        for day, value in (("2024-01-04", 50), ("2024-01-05", quote), ("2024-01-08", 99))]
    with workspace(tmp_path, events=[history_event("2024-01-04")], daily=daily) as (work, _):
        result = run(work)
        target = rows(Repository(work.conn, work.store), result["target_snapshot"], "iv_crush_targets")[0]
        assert target["post_date"] == "2024-01-05"
        assert target["crush_pct_iv30"] == expected
        assert target["skip_reason"] == reason


@pytest.mark.parametrize("override", [dict(max_rows=1), dict(max_decoded_bytes=1),
    dict(max_object_bytes=1), dict(max_total_bytes=1)])
def test_caps_refuse_without_destination_snapshot(tmp_path, override):
    with workspace(tmp_path) as (work, _):
        with pytest.raises(DataError) as error:
            run(work, **override)
        assert error.value.code in {"RESULT_LIMIT_EXCEEDED", "OBJECT_TOO_LARGE"}
        assert work.conn.execute("SELECT COUNT(*) FROM data_snapshots").fetchone()[0] == 0


def test_explicit_protected_event_refuses_before_daily_scan(tmp_path, monkeypatch):
    with workspace(tmp_path) as (work, _):
        protected = history_event("2024-03-05", protected=True)["event_id"]
        original = work.source.scan
        def scan(query, *, table_name, **kwargs):
            assert table_name != "daily_market"
            yield from original(query, table_name=table_name, **kwargs)
        monkeypatch.setattr(work.source, "scan", scan)
        with pytest.raises(DataError) as error:
            run(work, event_ids=[protected])
        assert error.value.code == "HOLDOUT_ACCESS_DENIED"


def test_nonmidnight_daily_date_refuses_without_truncation(tmp_path):
    daily = [dict(ticker="SYNTH", date=datetime(2024, 1, 4) + timedelta(hours=1), year=2024, iv30=50)]
    with workspace(tmp_path, daily=daily) as (work, _):
        with pytest.raises(DataError) as error:
            run(work)
        assert error.value.code == "CONTRACT_MISMATCH"


@pytest.mark.parametrize("cap", list(CAPS))
@pytest.mark.parametrize("value", [0, True])
def test_invalid_caps_refuse_before_source_payload_reads(tmp_path, monkeypatch, cap, value):
    with workspace(tmp_path) as (work, _):
        def refused(*args, **kwargs):
            pytest.fail("invalid caps opened source payload")
        monkeypatch.setattr(work.source, "scan", refused)
        with pytest.raises(DataError) as error:
            run(work, **{cap: value})
        assert error.value.code == "CONTRACT_MISMATCH"


def test_multifragment_tickers_and_missing_quotes_preserve_exact_coverage(tmp_path):
    events = [history_event("2024-01-04", ticker=ticker) for ticker in ("AAA", "BBB", "DDD")]
    daily = [dict(ticker=ticker, date=datetime.fromisoformat(day), year=2024, iv30=value)
        for ticker, values in (("AAA", (50, 30)), ("BBB", (40, 60)), ("ZZZ", (1, 99)))
        for day, value in zip(("2024-01-04", "2024-01-05"), values)]
    with workspace(tmp_path, events=events, daily=daily, fragment_rows=2) as (work, _):
        result = run(work)
        assert result["coverage"] == dict(selected_events=3, available=2, skipped_by_reason={"missing_pre": 1})
        values = rows(Repository(work.conn, work.store), result["target_snapshot"], "iv_crush_targets")
        assert [(r["ticker"], r["crush_pct_iv30"], r["skip_reason"]) for r in values] == [
            ("AAA", -40., None), ("BBB", 50., None), ("DDD", None, "missing_pre")]


def test_first_ticker_build_starts_before_all_daily_batches_are_decoded(tmp_path, monkeypatch):
    tickers = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")
    events = [history_event("2024-01-04", ticker=ticker) for ticker in tickers]
    daily = [dict(ticker=ticker, date=datetime(2024, 1, 1) + timedelta(days=i), year=2024, iv30=50.)
             for ticker in tickers for i in range(300)]
    with workspace(tmp_path, events=events, daily=daily, fragment_rows=300) as (work, _):
        scanned, started = [], []
        original_scan, original_build = work.source.scan, iv_crush_prepare.build_rows
        def scan(query, *, table_name, **kwargs):
            for batch in original_scan(query, table_name=table_name, **kwargs):
                if table_name == "daily_market":
                    assert batch.num_rows <= 1024
                    scanned.append(batch.num_rows)
                yield batch
        def build(ticker, events, quotes, **kwargs):
            started.append((ticker, sum(scanned)))
            return original_build(ticker, events, quotes, **kwargs)
        monkeypatch.setattr(work.source, "scan", scan)
        monkeypatch.setattr(iv_crush_prepare, "build_rows", build)
        result = run(work, max_rows=2000)
        assert started[0] == ("AAA", 1024)
        assert sum(scanned) == 1800
        assert [ticker for ticker, _ in started] == list(tickers)
        assert result["coverage"] == dict(selected_events=6, available=6, skipped_by_reason={})


def test_late_daily_scan_failure_never_publishes_partial_targets(tmp_path, monkeypatch):
    with workspace(tmp_path, fragment_rows=2) as (work, _):
        original = work.source.scan
        def scan(query, *, table_name, **kwargs):
            yield from original(query, table_name=table_name, **kwargs)
            if table_name == "daily_market":
                raise fail("MANIFEST_CORRUPT", "synthetic late failure")
        monkeypatch.setattr(work.source, "scan", scan)
        before = state(work.paths.root)
        with pytest.raises(DataError) as error:
            run(work)
        assert error.value.code == "MANIFEST_CORRUPT"
        assert work.conn.execute("SELECT COUNT(*) FROM data_snapshots").fetchone()[0] == 0
        assert state(work.paths.root) == before


def test_stream_releases_decoded_rows_before_next_batch(tmp_path, monkeypatch):
    live, peak = [0], [0]
    class Row(dict):
        pass
    def released():
        live[0] -= 1
    class Batch:
        nbytes = 8
        def to_pylist(self):
            result = []
            for _ in range(2):
                row = Row(iv30=50.)
                live[0] += 1
                peak[0] = max(peak[0], live[0])
                weakref.finalize(row, released)
                result.append(row)
            return result
    with workspace(tmp_path) as (work, _):
        calls = []
        def scan(query, **kwargs):
            calls.append(query)
            for _ in range(3):
                yield Batch()
        monkeypatch.setattr(work.source, "scan", scan)
        stream = iter_snapshot_rows(work.source, work.source_snapshot(), "daily_market", ("iv30",),
            key_filter=(KeyPredicate(column="ticker", operator="eq", values=("SYNTH",)),),
            max_rows=100, max_decoded_bytes=24, max_object_bytes=1000000, max_batch_rows=2)
        assert sum(row["iv30"] for row in stream) == 300.
        assert calls[0].max_batch_rows == 2
        assert peak[0] <= 3  # Two current rows plus at most the consumer's previous row.
        assert live[0] == 0


def test_collecting_reader_discards_late_cumulative_byte_failure(tmp_path, monkeypatch):
    with workspace(tmp_path) as (work, _):
        original = work.source.scan
        seen = []
        def scan(query, **kwargs):
            for batch in original(query, **kwargs):
                seen.append(batch.nbytes)
                yield batch
        monkeypatch.setattr(work.source, "scan", scan)
        with pytest.raises(DataError) as error:
            read_snapshot_rows(work.source, work.source_snapshot(), "daily_market", ("iv30",),
                key_filter=(KeyPredicate(column="ticker", operator="eq", values=("SYNTH",)),),
                max_rows=100, max_decoded_bytes=16, max_object_bytes=1000000, max_batch_rows=1)
        assert error.value.code == "RESULT_LIMIT_EXCEEDED"
        assert len(seen) >= 2 and seen[0] <= 16 and sum(seen) > 16


def test_abandoned_stream_and_failed_ticker_close_underlying_scan(tmp_path, monkeypatch):
    with workspace(tmp_path) as (work, _):
        original, closed = work.source.scan, []
        def scan(query, *, table_name, **kwargs):
            try:
                yield from original(query, table_name=table_name, **kwargs)
            finally:
                closed.append(table_name)
        monkeypatch.setattr(work.source, "scan", scan)
        stream = iter_snapshot_rows(work.source, work.source_snapshot(), "daily_market", ("iv30",),
            key_filter=(KeyPredicate(column="ticker", operator="eq", values=("SYNTH",)),),
            max_rows=100, max_decoded_bytes=1000000, max_object_bytes=1000000, max_batch_rows=1)
        next(stream)
        assert not closed
        stream.close()
        assert closed == ["daily_market"]
        def broken(*args, **kwargs):
            raise fail("CONTRACT_MISMATCH", "synthetic target refusal")
        monkeypatch.setattr(iv_crush_prepare, "build_rows", broken)
        with pytest.raises(DataError):
            run(work)
        assert closed[-1] == "daily_market"
        assert work.conn.execute("SELECT COUNT(*) FROM data_snapshots").fetchone()[0] == 0
