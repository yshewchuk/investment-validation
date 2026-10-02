"""Direct tests for ``engine.v2.features.daily_state_inputs``.

The bounded design's four explicit CodeRabbit acceptance items are driven
directly here:

* the resolved ``SnapshotRef`` must equal the supplied ref before any read;
* the whole bounded scan is consumed before any return, so a late repository
  ``DataError`` can never yield partial success;
* an eligible row whose numeric fields are all null yields empty ``values``
  with a non-null ``source_session``;
* ``DailyStateInputs.values`` is deeply immutable.

A small fake repository inspects the exact emitted ``DataQuery`` and drives
the row-validation refusals without any disk/Parquet cost. One real
catalog/ArtifactStore fixture proves the same contract against a real
published Parquet ``daily_market`` fragment; every expected number there is
an independently written literal, not a call into ``panel_math``.
"""
from __future__ import annotations

import copy
import datetime as dt
from dataclasses import FrozenInstanceError, replace
from types import MappingProxyType

import pandas as pd
import pytest

from engine.v2.contracts.data import DatasetVersionRef, KeyPredicate, SnapshotRef, TimeInterval
from engine.v2.data.errors import DataError, fail as data_fail
from engine.v2.data.repository import Repository
from engine.v2.features import panel_math
from engine.v2.features.daily_state_inputs import DailyStateInputs, scan_daily_state_inputs
from tests.data_scan_support import (
    catalog_and_store,
    commit_tables,
    contract_for,
    contract_ref_for,
    fake_hash,
    publish_and_inspect,
)

_DM = contract_for("daily_market")
_DM_REF = contract_ref_for(_DM)
_SEC = contract_for("securities")
_SEC_REF = contract_ref_for(_SEC)
_TABLE = "daily_market"

# Literal expectations for the 11 eligible AAA rows below, i=0..10, at
# decision 2024-01-16 (current i=10) and 2024-01-12 (current i=9).
_I10_VALUES = {
    "im": 6.0, "im_d1": 0.5, "im_d5": 2.5, "im_d10": 5.0,
    "iv10": 20.0, "iv10_d1": 1.0, "iv10_d5": 5.0, "iv10_d10": 10.0,
    "iv30": 30.0, "iv30_d1": 1.0, "iv30_d5": 5.0, "iv30_d10": 10.0,
    "exern_iv10": 25.0, "exern_iv30": 50.0, "exern_iv30_d1": 2.0,
    "exern_iv30_d5": 10.0, "exern_iv30_d10": 20.0,
    "iee": 0.75, "skew": 2.0, "contango": 1.5, "fwd90_30": 3.0,
    "fexern90_30": 4.0, "rvol30": 50.0, "spot": 110.0, "mcap_log": 6.0,
}
_I9_VALUES = {
    "im": 5.5, "im_d1": 0.5, "im_d5": 2.5,
    "iv10": 19.0, "iv10_d1": 1.0, "iv10_d5": 5.0,
    "iv30": 29.0, "iv30_d1": 1.0, "iv30_d5": 5.0,
    "exern_iv10": 24.0, "exern_iv30": 48.0, "exern_iv30_d1": 2.0, "exern_iv30_d5": 10.0,
    "iee": 0.7, "skew": 1.9, "contango": 1.4, "fwd90_30": 2.9, "fexern90_30": 3.9,
    "rvol30": 49.0, "spot": 109.0, "mcap_log": 5.9,
}


# --------------------------------------------------------------------------
# fixtures: synthetic rows, a fake repository, and a real catalog snapshot
# --------------------------------------------------------------------------


def _row(ticker: str, day: str, *, src_iv: str | None = "orats", **fields) -> dict:
    stamp = pd.Timestamp(day)
    return {"ticker": ticker, "date": stamp.to_pydatetime(), "year": stamp.year,
            "src_iv": src_iv, **fields}


def _eligible_rows(ticker: str = "AAA") -> list[dict]:
    days = ("2024-01-02", "2024-01-03", "2024-01-05", "2024-01-06", "2024-01-07",
            "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
            "2024-01-15")
    return [
        _row(ticker, day, implied_move=1.0 + 0.5 * i, iv10=10.0 + i, iv30=20.0 + i,
             exern_iv10=15.0 + i, exern_iv30=30.0 + 2.0 * i, iee=0.25 + 0.05 * i,
             skew=1.0 + 0.1 * i, contango=0.5 + 0.1 * i, fwd90_30=2.0 + 0.1 * i,
             fexern90_30=3.0 + 0.1 * i, rvol30=40.0 + i, spot=100.0 + i,
             mcap_log=5.0 + 0.1 * i)
        for i, day in enumerate(days)
    ]


def _mcap_only_row(ticker: str, day: str) -> dict:
    stamp = pd.Timestamp(day)
    return {"ticker": ticker, "date": stamp.to_pydatetime(), "year": stamp.year,
            "src_iv": None, "mcap_usd": 1e9, "mcap_log": 6.5,
            "mcap_asof": stamp.to_pydatetime(), "mcap_age_days": 0.0, "src_mcap": "orats"}


def _wild_rows(ticker: str, days: tuple[str, ...]) -> list[dict]:
    fields = {name: 777.0 for name in panel_math.DAILY_STATE_FIELDS}
    return [_row(ticker, day, **fields) for day in days]


def _fake_scan_rows() -> list[dict]:
    rows = [*_eligible_rows(), _mcap_only_row("AAA", "2024-01-04"),
            _mcap_only_row("AAA", "2024-01-16")]
    return sorted(rows, key=lambda row: row["date"])


def _real_rows() -> list[dict]:
    rows = [*_eligible_rows(),
            *_wild_rows("AAA", ("2024-01-01", "2024-01-17")),
            _mcap_only_row("AAA", "2024-01-04"),
            _mcap_only_row("AAA", "2024-01-16"),
            *_wild_rows("BBB", ("2024-01-03", "2024-01-15"))]
    return sorted(rows, key=lambda row: (row["ticker"], row["date"]))


class _Batch:
    """A batch whose rows are handed back exactly as built (no copy)."""

    def __init__(self, rows) -> None:
        self._rows = rows

    def to_pylist(self):
        return self._rows


class _FakeRepository:
    """The ``Repository`` surface this module needs, with call counters."""

    def __init__(self, snapshot: SnapshotRef, *, contract=_DM, batches=(), failure=None) -> None:
        self._snapshot = snapshot
        self._contract = contract
        self._batches = list(batches)
        self._failure = failure
        self.resolve_calls = 0
        self.head_lookups = 0
        self.table_contract_calls = 0
        self.scan_calls = 0
        self.queries = []

    def resolve(self, snapshot_id: str) -> SnapshotRef:
        self.resolve_calls += 1
        assert snapshot_id == self._snapshot.snapshot_id
        return self._snapshot

    def resolve_pinned(self, scope: str) -> SnapshotRef:
        self.head_lookups += 1
        raise AssertionError("scan_daily_state_inputs must never consult a head")

    def resolve_full_pinned(self, scope: str):
        self.head_lookups += 1
        raise AssertionError("scan_daily_state_inputs must never consult a head")

    def table_contract(self, snapshot_ref: SnapshotRef, table_name: str):
        self.table_contract_calls += 1
        if table_name not in snapshot_ref.table_versions:
            raise data_fail("CONTRACT_MISMATCH", "table is not part of this snapshot",
                            details={"table_name": table_name})
        return self._contract

    def scan(self, query, *, table_name: str):
        self.scan_calls += 1
        self.queries.append((query, table_name))
        for batch in self._batches:
            yield batch
        if self._failure is not None:
            raise self._failure


def _snapshot(*, dataset_version_id: str = "dsv-daily-1") -> SnapshotRef:
    return SnapshotRef(
        snapshot_id="snap-raw-inputs",
        manifest_hash=fake_hash("snapshot-manifest"),
        table_versions={"daily_market": DatasetVersionRef(
            dataset_version_id=dataset_version_id,
            table_contract_ref=_DM_REF,
            manifest_hash=fake_hash("daily-market-manifest"))},
        calendar_version="cal.v1",
        source_priority_version="prio.v1",
        finality_receipt_refs=(),
        knowledge_mode_by_table={"daily_market": "reconstructed"},
    )


def _scan(repository, snapshot, *, ticker="AAA", history_start="2024-01-02",
          decision_session="2024-01-16") -> DailyStateInputs:
    return scan_daily_state_inputs(repository, snapshot, ticker=ticker,
                                   history_start=history_start,
                                   decision_session=decision_session)


def _real_snapshot(tmp_path) -> tuple[Repository, SnapshotRef, list[dict]]:
    conn, clock, store = catalog_and_store(tmp_path)
    rows = _real_rows()
    record = publish_and_inspect(store, _DM, _DM_REF, rows, "2024")
    snap = commit_tables(conn, clock, {"daily_market": [record]}, {"daily_market": _DM},
                         store=store)
    repository = Repository(conn, store)
    return repository, repository.resolve(snap.snapshot_id), rows


def _securities_row(ticker: str) -> dict:
    return {"ticker": ticker, "year": 2024, "mcap_usd": 1.5e9, "mcap_log": 21.1,
            "mcap_raw": 1.5, "mcap_unit_era": "billions", "mcap_quantized": False,
            "n_obs": 250, "src": "orats"}


# --------------------------------------------------------------------------
# the exact query, the expected mapping, parity, and immutability
# --------------------------------------------------------------------------


def test_exact_query_mapping_parity_and_immutability():
    snapshot = _snapshot()
    rows = _fake_scan_rows()
    originals = copy.deepcopy(rows)
    repo = _FakeRepository(snapshot, batches=[_Batch(rows[:5]), _Batch(rows[5:])])

    result = _scan(repo, snapshot)

    assert result.values == _I10_VALUES
    assert result.source_session == "2024-01-15"
    assert result.snapshot_id == "snap-raw-inputs"
    assert result.dataset_version_id == "dsv-daily-1"

    assert repo.resolve_calls == 1 and repo.head_lookups == 0
    assert repo.table_contract_calls == 1 and repo.scan_calls == 1
    ((query, table),) = repo.queries
    assert table == _TABLE
    assert query.snapshot_id == snapshot.snapshot_id
    assert query.table_contract_ref == snapshot.table_versions[_TABLE].table_contract_ref
    assert query.columns == ("ticker", "date", "src_iv", *panel_math.DAILY_STATE_FIELDS)
    assert query.key_filter == (KeyPredicate(column="ticker", operator="eq", values=("AAA",)),)
    assert query.time_interval == TimeInterval(column="date", start_inclusive="2024-01-02",
                                               end_exclusive="2024-01-17")
    assert query.order_by == _DM.primary_key
    assert query.max_batch_rows == min(_DM.maximum_batch_rows, 1000)
    assert query.max_result_rows == min(_DM.maximum_result_rows, 10000)

    assert rows == originals
    assert dict(result.values) == panel_math.daily_state_lookup(rows, pd.Timestamp("2024-01-16"))
    assert dict(result.values) == panel_math.daily_state_lookup(
        list(reversed(rows)), pd.Timestamp("2024-01-16"))

    assert isinstance(result.values, MappingProxyType)
    with pytest.raises(TypeError):
        result.values["im"] = 0.0


def test_inclusive_cutoff_and_insufficient_history():
    snapshot = _snapshot()
    rows = [row for row in _fake_scan_rows() if row["date"] <= pd.Timestamp("2024-01-12")]
    repo = _FakeRepository(snapshot, batches=[_Batch(rows)])

    result = _scan(repo, snapshot, decision_session="2024-01-12")

    assert result.values == _I9_VALUES
    assert result.source_session == "2024-01-12"
    for key in ("im_d10", "iv10_d10", "iv30_d10", "exern_iv30_d10"):
        assert key not in result.values


def test_shuffled_scan_batches_match_sorted_expectations():
    snapshot = _snapshot()
    rows = list(reversed(_fake_scan_rows()))
    repo = _FakeRepository(snapshot, batches=[_Batch(rows[:4]), _Batch(rows[4:])])

    result = _scan(repo, snapshot)

    assert result.values == _I10_VALUES
    assert result.source_session == "2024-01-15"


def test_plain_date_objects_are_accepted():
    snapshot = _snapshot()
    repo = _FakeRepository(snapshot, batches=[_Batch(_fake_scan_rows())])

    result = _scan(repo, snapshot, history_start=dt.date(2024, 1, 2),
                   decision_session=dt.date(2024, 1, 16))

    assert result.values == _I10_VALUES


def test_values_are_deeply_immutable_and_not_aliased():
    source = {"im": 1.5}
    result = DailyStateInputs(values=source, source_session="2024-01-02",
                              snapshot_id="snap", dataset_version_id="dsv")
    source["im"] = 99.0
    source["other"] = 1.0

    assert dict(result.values) == {"im": 1.5}
    assert isinstance(result.values, MappingProxyType)
    with pytest.raises(TypeError):
        result.values["im"] = 2.0
    with pytest.raises(FrozenInstanceError):
        result.snapshot_id = "other"


# --------------------------------------------------------------------------
# absent values: no rows, no IV surface, all-null eligible rows, null lags
# --------------------------------------------------------------------------


def test_no_rows_or_no_eligible_rows_return_empty_without_session():
    snapshot = _snapshot()
    empty = _FakeRepository(snapshot, batches=[_Batch([])])
    result = _scan(empty, snapshot, decision_session="2024-01-05")
    assert dict(result.values) == {} and result.source_session is None

    mcap_only = _FakeRepository(snapshot, batches=[_Batch([_mcap_only_row("AAA", "2024-01-03")])])
    result = _scan(mcap_only, snapshot, decision_session="2024-01-05")
    assert dict(result.values) == {} and result.source_session is None


def test_eligible_all_null_row_keeps_source_session():
    snapshot = _snapshot()
    rows = [_row("AAA", "2024-01-02"), _row("AAA", "2024-01-03")]
    repo = _FakeRepository(snapshot, batches=[_Batch(rows)])

    result = _scan(repo, snapshot, decision_session="2024-01-03")

    assert dict(result.values) == {}
    assert result.source_session == "2024-01-03"


def test_null_lag_operand_drops_only_that_lag():
    snapshot = _snapshot()
    rows = _eligible_rows()
    rows[5]["implied_move"] = None
    repo = _FakeRepository(snapshot, batches=[_Batch(rows)])

    result = _scan(repo, snapshot, decision_session="2024-01-15")

    assert result.values["im"] == 6.0
    assert result.values["im_d1"] == 0.5
    assert "im_d5" not in result.values
    assert result.values["im_d10"] == 5.0
    assert result.values["iv10_d5"] == 5.0


# --------------------------------------------------------------------------
# malformed arguments refuse before any repository read
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [None, "", "   ", 7, True])
def test_malformed_ticker_refuses_before_reads(bad):
    snapshot = _snapshot()
    repo = _FakeRepository(snapshot)
    with pytest.raises(DataError) as exc:
        _scan(repo, snapshot, ticker=bad)
    assert exc.value.code == "CONTRACT_MISMATCH"
    assert repo.resolve_calls == 0 and repo.table_contract_calls == 0 and repo.scan_calls == 0


@pytest.mark.parametrize("field", ["history_start", "decision_session"])
@pytest.mark.parametrize("bad", [None, True, 20240102, pd.NaT, "not-a-date",
                                 "2024-01-02T01:00:00", "2024-01-02T00:00:00Z",
                                 pd.Timestamp("2024-01-02", tz="UTC")])
def test_malformed_dates_refuse_before_reads(field, bad):
    snapshot = _snapshot()
    repo = _FakeRepository(snapshot)
    kwargs = {"history_start": "2024-01-02", "decision_session": "2024-01-16", field: bad}
    with pytest.raises(DataError) as exc:
        _scan(repo, snapshot, **kwargs)
    assert exc.value.code == "CONTRACT_MISMATCH"
    assert "not-a-date" not in str(exc.value)
    assert repo.resolve_calls == 0 and repo.table_contract_calls == 0 and repo.scan_calls == 0


def test_reversed_bounds_refuse_before_reads():
    snapshot = _snapshot()
    repo = _FakeRepository(snapshot)
    with pytest.raises(DataError) as exc:
        _scan(repo, snapshot, history_start="2024-01-16", decision_session="2024-01-02")
    assert exc.value.code == "CONTRACT_MISMATCH"
    assert repo.resolve_calls == 0


def test_unbounded_decision_day_refuses():
    snapshot = _snapshot()
    repo = _FakeRepository(snapshot)
    last = pd.Timestamp.max.normalize()
    with pytest.raises(DataError) as exc:
        _scan(repo, snapshot, history_start=last, decision_session=last)
    assert exc.value.code == "CONTRACT_MISMATCH"
    assert repo.resolve_calls == 0


def test_blank_supplied_snapshot_id_refuses():
    snapshot = replace(_snapshot(), snapshot_id=" ")
    repo = _FakeRepository(snapshot)
    with pytest.raises(DataError) as exc:
        _scan(repo, snapshot)
    assert exc.value.code == "CONTRACT_MISMATCH"
    assert repo.resolve_calls == 0


def test_non_snapshot_ref_refuses():
    repo = _FakeRepository(_snapshot())
    with pytest.raises(DataError) as exc:
        _scan(repo, {"snapshot_id": "snap-raw-inputs"})
    assert exc.value.code == "CONTRACT_MISMATCH"
    assert repo.resolve_calls == 0


def test_altered_version_with_same_snapshot_id_refuses_before_reads():
    snapshot = _snapshot()
    altered = replace(snapshot, table_versions={"daily_market": replace(
        snapshot.table_versions["daily_market"], dataset_version_id="dsv-other")})
    repo = _FakeRepository(snapshot, batches=[_Batch(_fake_scan_rows())])
    with pytest.raises(DataError) as exc:
        _scan(repo, altered)
    assert exc.value.code == "CONTRACT_MISMATCH"
    assert repo.resolve_calls == 1 and repo.table_contract_calls == 0 and repo.scan_calls == 0


def test_table_not_pinned_under_supplied_snapshot_refuses():
    snapshot = replace(_snapshot(), table_versions={}, knowledge_mode_by_table={})
    repo = _FakeRepository(snapshot, batches=[_Batch(_fake_scan_rows())])
    with pytest.raises(DataError) as exc:
        _scan(repo, snapshot)
    assert exc.value.code == "CONTRACT_MISMATCH"
    assert exc.value.problem.details == {"table_name": "daily_market"}
    assert repo.scan_calls == 0


def test_blank_dataset_version_refuses_before_scan():
    base = _snapshot()
    version = replace(base.table_versions["daily_market"], dataset_version_id=" ")
    snapshot = replace(base, table_versions={"daily_market": version})
    repo = _FakeRepository(snapshot, batches=[_Batch(_fake_scan_rows())])
    with pytest.raises(DataError) as exc:
        _scan(repo, snapshot)
    assert exc.value.code == "CONTRACT_MISMATCH"
    assert repo.scan_calls == 0


# --------------------------------------------------------------------------
# malformed scanned rows refuse without leaking submitted values
# --------------------------------------------------------------------------


@pytest.mark.parametrize("row", [
    [],
    _row("BBB", "2024-01-02"),
    _row("AAA", "2024-01-01"),
    _row("AAA", "2024-01-17"),
    _row("AAA", "2024-01-10T01:00:00"),
    {"ticker": "AAA", "date": "not-a-date", "year": 2024, "src_iv": "orats"},
    {"ticker": "AAA", "year": 2024, "src_iv": "orats"},
    {"ticker": "AAA", "date": pd.NaT, "year": 2024, "src_iv": "orats"},
])
def test_bad_scanned_rows_refuse_without_leaking(row):
    snapshot = _snapshot()
    repo = _FakeRepository(snapshot, batches=[_Batch([row])])
    with pytest.raises(DataError) as exc:
        _scan(repo, snapshot)
    assert exc.value.code == "CONTRACT_MISMATCH"
    assert "not-a-date" not in str(exc.value)
    assert "BBB" not in str(exc.value)


def test_duplicate_dates_refuse_as_identity_conflict():
    snapshot = _snapshot()
    row = _eligible_rows()[0]
    repo = _FakeRepository(snapshot, batches=[_Batch([row]), _Batch([dict(row)])])
    with pytest.raises(DataError) as exc:
        _scan(repo, snapshot)
    assert exc.value.code == "IDENTITY_CONFLICT"


def test_malformed_numeric_conversion_refuses_without_leaking():
    snapshot = _snapshot()
    rows = _eligible_rows()
    rows[10]["iv30"] = "not-a-number"
    repo = _FakeRepository(snapshot, batches=[_Batch(rows)])
    with pytest.raises(DataError) as exc:
        _scan(repo, snapshot, decision_session="2024-01-15")
    assert exc.value.code == "CONTRACT_MISMATCH"
    assert "not-a-number" not in str(exc.value)


def test_own_result_cap_refuses_an_injected_overrun():
    contract = replace(_DM, maximum_result_rows=3)
    snapshot = _snapshot()
    repo = _FakeRepository(snapshot, contract=contract, batches=[_Batch(_eligible_rows()[:4])])
    with pytest.raises(DataError) as exc:
        _scan(repo, snapshot, decision_session="2024-01-15")
    assert exc.value.code == "RESULT_LIMIT_EXCEEDED"
    ((query, _),) = repo.queries
    assert query.max_result_rows == 3


def test_late_repository_failure_never_yields_partial_success():
    problem = data_fail("OBJECT_CORRUPT", "synthetic late corruption")
    snapshot = _snapshot()
    repo = _FakeRepository(snapshot, batches=[_Batch(_eligible_rows()[:3])], failure=problem)
    with pytest.raises(DataError) as exc:
        _scan(repo, snapshot)
    assert exc.value is problem


# --------------------------------------------------------------------------
# the real catalog/ArtifactStore fixture
# --------------------------------------------------------------------------


def test_real_published_fragment_matches_independent_expectations(tmp_path, monkeypatch):
    repository, snapshot, rows = _real_snapshot(tmp_path)
    originals = copy.deepcopy(rows)

    def forbidden_head(scope):
        raise AssertionError("scan_daily_state_inputs must never consult a scope head")

    monkeypatch.setattr(repository, "resolve_pinned", forbidden_head)
    monkeypatch.setattr(repository, "resolve_full_pinned", forbidden_head)

    result = _scan(repository, snapshot)

    assert result.values == _I10_VALUES
    assert result.source_session == "2024-01-15"
    assert result.snapshot_id == snapshot.snapshot_id
    assert result.dataset_version_id == snapshot.table_versions[_TABLE].dataset_version_id

    again = _scan(repository, snapshot)
    assert again == result

    cutoff = _scan(repository, snapshot, decision_session="2024-01-12")
    assert cutoff.values == _I9_VALUES
    assert cutoff.source_session == "2024-01-12"

    assert rows == originals


def test_snapshot_without_daily_market_refuses(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    record = publish_and_inspect(store, _SEC, _SEC_REF, [_securities_row("AAA")], "2024")
    snap = commit_tables(conn, clock, {"securities": [record]}, {"securities": _SEC}, store=store)
    repository = Repository(conn, store)
    snapshot = repository.resolve(snap.snapshot_id)

    with pytest.raises(DataError) as exc:
        _scan(repository, snapshot)

    assert exc.value.code == "CONTRACT_MISMATCH"
    assert exc.value.problem.details == {"table_name": "daily_market"}


def test_real_repository_failure_propagates_unchanged(tmp_path, monkeypatch):
    repository, snapshot, _ = _real_snapshot(tmp_path)
    problem = data_fail("OBJECT_CORRUPT", "synthetic corrupt fragment")

    def broken(*args, **kwargs):
        raise problem

    monkeypatch.setattr(repository, "scan", broken)
    with pytest.raises(DataError) as exc:
        _scan(repository, snapshot)
    assert exc.value is problem
