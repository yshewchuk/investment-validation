"""Phase 6 slice 6: the v2 research tools, on pinned snapshots (UD-4).

``engine/v2/research`` moved the pure cores of ``tools/signal_screen.py`` and
``tools/fill_quality.py`` and the read path of
``engine/data/pulls/polygon_fills.py`` onto bounded ``Repository.scan`` reads
against one explicitly resolved snapshot. Every fixture here is a real
synthetic snapshot: the same ``build_legacy_mapping`` contracts, real Parquet
fragments published through ``ArtifactStore`` and inspected through
``objects.inspect_fragment``, committed with the same ``commit_snapshot``
``tests/test_v2_data_commit.py`` already drives.

Four cases per tool:

* the moved function is equal to its legacy counterpart on the same frame;
* the tool's output carries the pinned snapshot id;
* an explicit ``--snapshot-id`` (``snapshot_id=``) still reads the first
  snapshot after the scope's head has moved;
* a manifest whose fragment membership has been deleted is refused
  (``MANIFEST_CORRUPT``) rather than read.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.data.pulls import polygon_fills as legacy_polygon_fills  # noqa: E402
from engine.v2.contracts.data import TableContractRef  # noqa: E402
from engine.v2.data import manifests  # noqa: E402
from engine.v2.data.catalog import commit_snapshot  # noqa: E402
from engine.v2.data.errors import DataError  # noqa: E402
from engine.v2.data.objects import inspect_fragment  # noqa: E402
from engine.v2.data.repository import Repository  # noqa: E402
from engine.v2.foundation import ArtifactStore, content_hash  # noqa: E402
from engine.v2.ops.bootstrap import open_catalog  # noqa: E402
from engine.v2.ops.catalog import connect as ops_connect  # noqa: E402
from engine.v2.research import fill_quality as v2_fill_quality  # noqa: E402
from engine.v2.research import polygon_fills as v2_polygon_fills  # noqa: E402
from engine.v2.research import signal_screen as v2_signal_screen  # noqa: E402
from tests.ops_support import FakeClock  # noqa: E402
from tests.test_v2_data_manifests import (  # noqa: E402
    IRH_A,
    RECEIPT_A,
    _contract,
    _publish_bytes,
    _table_from_rows,
    _to_bytes,
)
from tools import fill_quality as legacy_fill_quality  # noqa: E402
from tools import signal_screen as legacy_signal_screen  # noqa: E402


# --------------------------------------------------------------------------
# synthetic pinned snapshots (real fragments, real commit machinery)
# --------------------------------------------------------------------------


def _noop_fence(conn) -> None:
    return None


def _catalog(tmp_path):
    clock = FakeClock()
    conn = open_catalog(tmp_path / "catalog.sqlite", clock=clock)
    store = ArtifactStore(tmp_path / "objects")
    return conn, clock, store


def _contract_ref(table_name: str) -> TableContractRef:
    contract = _contract(table_name)
    return TableContractRef(contract_id=contract.contract_id,
                            definition_hash=contract.definition_hash)


def _record(store: ArtifactStore, table_name: str, rows: list[dict], year) -> object:
    contract = _contract(table_name)
    ref = _contract_ref(table_name)
    table = _table_from_rows(contract, rows)
    obj = _publish_bytes(store, _to_bytes(table))
    inspection = inspect_fragment(store, obj, contract, ref, str(year))
    return manifests.fragment_record(inspection, ref, input_receipt_refs=(RECEIPT_A,),
                                     import_request_hash=IRH_A)


def _commit(conn, clock, store, frames, *, receipt_id, scope="shadow",
            expected_head=None, generation=0):
    """One committed snapshot: ``frames`` is ``{table: {year: rows}}``."""
    contracts, objects, records, table_manifests = [], [], [], {}
    for table_name, per_year in frames.items():
        table_records = [_record(store, table_name, rows, year)
                         for year, rows in per_year.items()]
        table_manifests[table_name] = manifests.dataset_manifest(
            _contract_ref(table_name), table_records, knowledge_mode="reconstructed",
            coverage_receipt_refs=(RECEIPT_A,), availability_evidence_refs=())
        contracts.append(_contract(table_name))
        objects.extend(record.object_ref for record in table_records)
        records.extend(table_records)
    snapshot = manifests.snapshot_ref(table_manifests, calendar_version="cal.v1",
                                      source_priority_version="prio.v1",
                                      finality_receipt_refs=(RECEIPT_A,))
    receipt = commit_snapshot(
        conn, scope=scope, request_hash=content_hash({"label": receipt_id}),
        contracts=contracts, objects=objects, records=records,
        manifests=list(table_manifests.values()), snapshot=snapshot,
        expected_head_snapshot_id=expected_head, expected_head_generation=generation,
        receipt_id=receipt_id, attempt_id=f"att-{receipt_id}", fence=1,
        fence_check=_noop_fence, clock=clock)
    return receipt, snapshot


def _consistent_copy(src_path: Path, dst_path: Path) -> None:
    src = sqlite3.connect(str(src_path))
    dst = sqlite3.connect(str(dst_path))
    try:
        src.backup(dst)
    finally:
        src.close()
        dst.close()


def _corrupt_membership(tmp_path, table_name):
    """A consistent backup copy with ``table_name``'s fragment membership deleted."""
    copy_path = tmp_path / f"corrupt_{table_name}.sqlite"
    _consistent_copy(tmp_path / "catalog.sqlite", copy_path)
    corrupt = ops_connect(copy_path)
    corrupt.execute("DROP TRIGGER data_version_fragments_no_delete")
    corrupt.execute("DROP TRIGGER data_version_fragments_no_update")
    dsv_id = corrupt.execute(
        "SELECT dataset_version_id FROM data_snapshot_tables WHERE table_name = ?",
        (table_name,)).fetchone()[0]
    corrupt.execute("DELETE FROM data_version_fragments WHERE dataset_version_id = ?", (dsv_id,))
    return corrupt


# --------------------------------------------------------------------------
# signal_screen
# --------------------------------------------------------------------------


def _daily_market_rows(year: int, ticker: str, count: int = 320) -> list[dict]:
    rows = []
    for i in range(count):
        day = datetime(year, 1, 1) + pd.Timedelta(days=i)
        rows.append(dict(
            ticker=ticker, date=day, year=year,
            spot=100.0 + i * 0.5, iv10=30.0, iv30=32.0 + (i % 7) * 0.1,
            exern_iv10=29.0, exern_iv30=31.0, implied_move=5.0,
            implied_reconstructed=False, rvol30=28.0, skew=1.1, contango=0.5,
            fwd90_30=33.0, fexern90_30=34.0, iee=0.2, mcap_usd=1e9, mcap_log=20.7,
            mcap_asof=day, mcap_age_days=0.0,
            src_spot="orats", src_iv="orats", src_mcap="orats"))
    return rows


def _daily_market_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)[list(v2_signal_screen.FEATURE_COLUMNS)].copy()


class _DailyMarketStore:
    """The legacy screen's store surface, over one frame."""

    def __init__(self, daily_market: pd.DataFrame) -> None:
        self._daily_market = daily_market

    def iter_table(self, table_name, columns=None):
        assert table_name == "daily_market"
        return iter([("all", self._daily_market)])


def test_signal_screen_move_matches_legacy_on_same_frame(monkeypatch):
    dm = _daily_market_frame(_daily_market_rows(2024, "AAA"))
    monkeypatch.setattr(legacy_signal_screen, "store", _DailyMarketStore(dm.copy()))

    expected_features = legacy_signal_screen.build_features()
    actual_features = v2_signal_screen.build_features(dm.copy())
    pd.testing.assert_frame_equal(actual_features, expected_features)
    pd.testing.assert_frame_equal(v2_signal_screen.screen(actual_features),
                                  legacy_signal_screen.screen(expected_features))
    pd.testing.assert_frame_equal(
        v2_signal_screen.mcap_slice(actual_features, v2_signal_screen.SLICE_SIGNALS),
        legacy_signal_screen.mcap_slice(expected_features, v2_signal_screen.SLICE_SIGNALS))


def test_signal_screen_output_carries_pinned_snapshot_id(tmp_path):
    conn, clock, store = _catalog(tmp_path)
    frames = {"daily_market": {2024: _daily_market_rows(2024, "AAA")}}
    _receipt, snapshot = _commit(conn, clock, store, frames, receipt_id="ss-a")

    result = v2_signal_screen.run(Repository(conn, store=store),
                                  reports_dir=tmp_path / "reports", scope="shadow")
    assert result["snapshot_id"] == snapshot.snapshot_id
    frame = pd.read_parquet(result["paths"]["parquet"])
    assert set(frame["snapshot_id"]) == {snapshot.snapshot_id}
    assert snapshot.snapshot_id in Path(result["paths"]["md"]).read_text()
    conn.close()


def test_signal_screen_explicit_snapshot_id_ignores_a_moved_head(tmp_path):
    conn, clock, store = _catalog(tmp_path)
    frames_a = {"daily_market": {2024: _daily_market_rows(2024, "AAA")}}
    _receipt, snap_a = _commit(conn, clock, store, frames_a, receipt_id="ss-a")
    frames_b = {"daily_market": {2024: _daily_market_rows(2024, "AAA"),
                                 2025: _daily_market_rows(2025, "BBB")}}
    _receipt, snap_b = _commit(conn, clock, store, frames_b, receipt_id="ss-b",
                               expected_head=snap_a.snapshot_id, generation=1)

    repository = Repository(conn, store=store)
    pinned = v2_signal_screen.run(repository, reports_dir=tmp_path / "pinned", scope="shadow")
    assert pinned["snapshot_id"] == snap_b.snapshot_id
    assert pinned["rows"] == 640  # AAA 2024 + BBB 2025

    first = v2_signal_screen.run(repository, reports_dir=tmp_path / "first", scope="shadow",
                                 snapshot_id=snap_a.snapshot_id)
    assert first["snapshot_id"] == snap_a.snapshot_id
    assert first["rows"] == 320  # AAA 2024 only
    conn.close()


def test_signal_screen_refuses_a_corrupt_manifest(tmp_path):
    conn, clock, store = _catalog(tmp_path)
    frames = {"daily_market": {2024: _daily_market_rows(2024, "AAA")}}
    _commit(conn, clock, store, frames, receipt_id="ss-a")
    conn.close()

    corrupt = _corrupt_membership(tmp_path, "daily_market")
    with pytest.raises(DataError) as err:
        v2_signal_screen.run(Repository(corrupt, store=store),
                             reports_dir=tmp_path / "reports", scope="shadow")
    assert err.value.code == "MANIFEST_CORRUPT"
    corrupt.close()


# --------------------------------------------------------------------------
# fill_quality
# --------------------------------------------------------------------------


def _option_daily_rows(close: float = 3.5) -> list[dict]:
    return [dict(contract_ticker="O:AAA240906C00100000", ticker="AAA",
                 obs_date=datetime(2024, 8, 20), year=2024,
                 expiry=datetime(2024, 9, 6), strike=100.0, right="C",
                 open=3.4, high=3.7, low=3.3, close=close, vwap=3.4,
                 volume=120.0, n_trades=5, src="polygon", src_file="f1")]


def _option_chains_rows() -> list[dict]:
    return [dict(ticker="AAA", obs_date=datetime(2024, 8, 20), year=2024,
                 expiry=datetime(2024, 9, 6), dte=17, strike=100.0, right="C",
                 bid=3.0, ask=4.0, mid=3.5, iv=0.5, delta=0.5, spot=100.0,
                 src="orats", src_file="f2", chain_kind="entry",
                 volume=None, open_interest=None, bid_size=None, ask_size=None,
                 quote_repaired=False)]


def _option_daily_frame(close: float = 3.5) -> pd.DataFrame:
    return pd.DataFrame(_option_daily_rows(close))


def _option_chains_frame() -> pd.DataFrame:
    return pd.DataFrame(_option_chains_rows())


class _FillStore:
    """The legacy join's store surface, over two frames."""

    def __init__(self, trades: pd.DataFrame, quotes: pd.DataFrame) -> None:
        self._trades = trades
        self._quotes = quotes

    def read_table(self, table_name):
        assert table_name == "option_daily"
        return self._trades

    def iter_table(self, table_name, *, years=None, columns=None):
        assert table_name == "option_chains"
        return iter([(years[0] if years else None, self._quotes)])


def test_fill_quality_move_matches_legacy_on_same_frames(monkeypatch):
    trades = _option_daily_frame()
    quotes = _option_chains_frame()
    monkeypatch.setattr(legacy_fill_quality, "store", _FillStore(trades, quotes))

    expected = legacy_fill_quality.join_quotes_and_trades()
    actual = v2_fill_quality.join_quotes_and_trades(trades, quotes)
    pd.testing.assert_frame_equal(actual, expected)
    keys = v2_fill_quality.contract_ticker_column(quotes).tolist()
    assert keys == legacy_fill_quality.contract_ticker_column(quotes).tolist()
    assert keys == ["O:AAA240906C00100000"]


def test_fill_quality_output_carries_pinned_snapshot_id(tmp_path):
    conn, clock, store = _catalog(tmp_path)
    frames = {"option_daily": {2024: _option_daily_rows()},
              "option_chains": {2024: _option_chains_rows()}}
    _receipt, snapshot = _commit(conn, clock, store, frames, receipt_id="fq-a")

    result = v2_fill_quality.run(Repository(conn, store=store),
                                 reports_dir=tmp_path / "reports", scope="shadow")
    assert result["snapshot_id"] == snapshot.snapshot_id
    assert result["rows"] == 1
    frame = pd.read_parquet(result["paths"]["parquet"])
    assert set(frame["snapshot_id"]) == {snapshot.snapshot_id}
    assert snapshot.snapshot_id in Path(result["paths"]["md"]).read_text()
    conn.close()


def test_fill_quality_explicit_snapshot_id_ignores_a_moved_head(tmp_path):
    conn, clock, store = _catalog(tmp_path)
    frames_a = {"option_daily": {2024: _option_daily_rows(close=3.5)},
                "option_chains": {2024: _option_chains_rows()}}
    _receipt, snap_a = _commit(conn, clock, store, frames_a, receipt_id="fq-a")
    frames_b = {"option_daily": {2024: _option_daily_rows(close=9.9)},
                "option_chains": {2024: _option_chains_rows()}}
    _receipt, snap_b = _commit(conn, clock, store, frames_b, receipt_id="fq-b",
                               expected_head=snap_a.snapshot_id, generation=1)

    repository = Repository(conn, store=store)
    pinned = v2_fill_quality.run(repository, reports_dir=tmp_path / "pinned", scope="shadow")
    assert pinned["snapshot_id"] == snap_b.snapshot_id
    assert set(pd.read_parquet(pinned["paths"]["parquet"])["close"]) == {9.9}

    first = v2_fill_quality.run(repository, reports_dir=tmp_path / "first", scope="shadow",
                                snapshot_id=snap_a.snapshot_id)
    assert first["snapshot_id"] == snap_a.snapshot_id
    assert set(pd.read_parquet(first["paths"]["parquet"])["close"]) == {3.5}
    conn.close()


def test_fill_quality_refuses_a_corrupt_manifest(tmp_path):
    conn, clock, store = _catalog(tmp_path)
    frames = {"option_daily": {2024: _option_daily_rows()},
              "option_chains": {2024: _option_chains_rows()}}
    _commit(conn, clock, store, frames, receipt_id="fq-a")
    conn.close()

    corrupt = _corrupt_membership(tmp_path, "option_chains")
    with pytest.raises(DataError) as err:
        v2_fill_quality.run(Repository(corrupt, store=store),
                            reports_dir=tmp_path / "reports", scope="shadow")
    assert err.value.code == "MANIFEST_CORRUPT"
    corrupt.close()


# --------------------------------------------------------------------------
# polygon_fills
# --------------------------------------------------------------------------


def _trades_rows(strikes: tuple[float, ...] = (100.0,)) -> list[dict]:
    rows = []
    for i, strike in enumerate(strikes):
        legs = json.dumps({
            "entry": [{"name": "leg", "right": "C", "side": "buy", "qty": 1.0,
                       "expiry": "2024-09-06", "strike": strike, "dte": 10}],
            "exit": [{"name": "leg", "right": "C", "side": "sell", "qty": 1.0,
                      "expiry": "2024-09-06", "strike": strike, "dte": 10}],
        })
        rows.append(dict(
            trade_id=f"t{i}", kind="sim", strategy="CAL-P", variant=None, ticker="AAA",
            event_id=None, event_date=datetime(2024, 9, 1), year=2024, legs=legs,
            entry_date=datetime(2024, 8, 20), exit_date=datetime(2024, 9, 6),
            strike=strike, expiry=datetime(2024, 9, 6), fill_alpha=0.5, entry_cost=1.0,
            exit_value=1.5, ret=0.5, provenance="sim"))
    return rows


def test_polygon_fills_move_matches_legacy_on_same_frame():
    trades = pd.DataFrame(_trades_rows((100.0, 110.0)))
    assert v2_polygon_fills.collect_contracts(trades) == \
        legacy_polygon_fills.collect_contracts(trades)
    assert v2_polygon_fills.option_ticker("TSLA", "2024-09-06", "C", 210.0) == \
        legacy_polygon_fills.option_ticker("TSLA", "2024-09-06", "C", 210.0) == \
        "O:TSLA240906C00210000"


def test_polygon_fills_plan_carries_pinned_snapshot_id(tmp_path):
    conn, clock, store = _catalog(tmp_path)
    frames = {"trades": {2024: _trades_rows((100.0,))}}
    _receipt, snapshot = _commit(conn, clock, store, frames, receipt_id="pf-a")

    result = v2_polygon_fills.run(Repository(conn, store=store),
                                  out_dir=tmp_path / "reports", scope="shadow")
    assert result["snapshot_id"] == snapshot.snapshot_id
    assert result["contracts_in_trades"] == 1
    assert result["contract_days"] == 2  # entry and exit both observed
    written = json.loads(Path(result["path"]).read_text())
    assert written["snapshot_id"] == snapshot.snapshot_id
    assert written["contracts"][0]["contract_ticker"] == "O:AAA240906C00100000"
    conn.close()


def test_polygon_fills_explicit_snapshot_id_ignores_a_moved_head(tmp_path):
    conn, clock, store = _catalog(tmp_path)
    frames_a = {"trades": {2024: _trades_rows((100.0,))}}
    _receipt, snap_a = _commit(conn, clock, store, frames_a, receipt_id="pf-a")
    frames_b = {"trades": {2024: _trades_rows((100.0, 110.0))}}
    _receipt, snap_b = _commit(conn, clock, store, frames_b, receipt_id="pf-b",
                               expected_head=snap_a.snapshot_id, generation=1)

    repository = Repository(conn, store=store)
    pinned = v2_polygon_fills.run(repository, out_dir=tmp_path / "pinned", scope="shadow")
    assert pinned["snapshot_id"] == snap_b.snapshot_id
    assert pinned["contracts_in_trades"] == 2

    first = v2_polygon_fills.run(repository, out_dir=tmp_path / "first", scope="shadow",
                                 snapshot_id=snap_a.snapshot_id)
    assert first["snapshot_id"] == snap_a.snapshot_id
    assert first["contracts_in_trades"] == 1
    conn.close()


def test_polygon_fills_refuses_a_corrupt_manifest(tmp_path):
    conn, clock, store = _catalog(tmp_path)
    frames = {"trades": {2024: _trades_rows((100.0,))}}
    _commit(conn, clock, store, frames, receipt_id="pf-a")
    conn.close()

    corrupt = _corrupt_membership(tmp_path, "trades")
    with pytest.raises(DataError) as err:
        v2_polygon_fills.run(Repository(corrupt, store=store),
                             out_dir=tmp_path / "reports", scope="shadow")
    assert err.value.code == "MANIFEST_CORRUPT"
    corrupt.close()
