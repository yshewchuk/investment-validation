"""Slice-7 replay move: pricing parity, snapshot pinning, and refusals.

The core acceptance test is ``test_replay_move_matches_legacy_on_same_synthetic_input``:
``engine.v2.research.replay`` and legacy ``engine.replay`` must produce
byte-identical frames from the same input rows. The v2 pricing primitives it
imports (``engine.v2.research._pricing``) are asserted identical to the legacy
functions they were moved from, and the store-reaching edge is exercised
against a real committed snapshot (``tests/data_scan_support.py``).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import replay as legacy  # noqa: E402
from engine.v2.contracts.data import DatasetManifest  # noqa: E402
from engine.v2.data import catalog, manifests  # noqa: E402
from engine.v2.data.errors import DataError  # noqa: E402
from engine.v2.data.repository import Repository  # noqa: E402
from engine.v2.research import _pricing, replay  # noqa: E402
from tests.data_scan_support import (  # noqa: E402
    catalog_and_store,
    contract_for,
    contract_ref_for,
    fake_hash,
    publish_and_inspect,
)

import engine.structures as legacy_structures  # noqa: E402


def _calendar() -> _pricing.TradingCalendar:
    """Consecutive weekdays around the test event (mirrors tests/test_replay.py)."""
    return _pricing.TradingCalendar(pd.bdate_range("2024-03-01", periods=60))


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"event_id": "TEST_2024-05-02", "ticker": "TEST",
             "event_date": pd.Timestamp("2024-05-02"), "session": "AMC"},
        ]
    )


def _chain(ticker, obs_date, *, call=(2.0, 2.4), put=(1.0, 1.4), spot=100.0):
    """Two expiries × three strikes, round numbers so P&L is checkable by hand."""
    rows = []
    obs = pd.Timestamp(obs_date)
    for expiry, dte in ((pd.Timestamp("2024-05-03"), 2), (pd.Timestamp("2024-05-24"), 23)):
        for strike in (95.0, 100.0, 105.0):
            for right, (bid, ask) in (("C", call), ("P", put)):
                scale = 1.0 if dte < 10 else 2.0
                rows.append(
                    {
                        "ticker": ticker, "obs_date": obs, "expiry": expiry, "dte": dte,
                        "strike": strike, "right": right,
                        "bid": bid * scale, "ask": ask * scale,
                        "spot": spot, "quote_repaired": False,
                    }
                )
    return pd.DataFrame(rows)


def _index() -> replay.ChainIndex:
    return replay.ChainIndex(
        {
            ("TEST", pd.Timestamp("2024-05-02")): _chain("TEST", "2024-05-02"),
            ("TEST", pd.Timestamp("2024-05-03")): _chain("TEST", "2024-05-03"),
        }
    )


def _plan_row() -> dict:
    return {
        "event_id": "TEST_2024-05-02", "ticker": "TEST",
        "event_date": pd.Timestamp("2024-05-02"), "session": "AMC",
        "decision_date": pd.Timestamp("2024-05-02"),
        "entry_date": pd.Timestamp("2024-05-02"),
        "exit_date": pd.Timestamp("2024-05-03"),
    }


# --------------------------------------------------------------------------
# the moved pricing primitives are identical
# --------------------------------------------------------------------------


def test_moved_price_structure_matches_legacy_on_synthetic_input():
    rows = _chain("TEST", "2024-05-02")
    session = "AMC"
    event_date = pd.Timestamp("2024-05-02")

    old = legacy_structures.price_structure(
        legacy_structures.straddle_through(),
        legacy_structures.ChainSnapshot(
            ticker="TEST", obs_date=pd.Timestamp("2024-05-02"),
            event_date=event_date, rows=rows, session=session,
        ),
        legacy_structures.FillModel(0.5),
    )
    new = _pricing.price_structure(
        _pricing.straddle_through(),
        _pricing.ChainSnapshot(
            ticker="TEST", obs_date=pd.Timestamp("2024-05-02"),
            event_date=event_date, rows=rows, session=session,
        ),
        _pricing.FillModel(0.5),
    )
    assert new.to_dict() == old.to_dict()


def test_moved_structure_return_matches_legacy_on_synthetic_input():
    def both(module):
        structure = module.straddle_through()
        entry = module.price_structure(
            structure,
            module.ChainSnapshot(ticker="TEST", obs_date=pd.Timestamp("2024-05-02"),
                                 event_date=pd.Timestamp("2024-05-02"),
                                 rows=_chain("TEST", "2024-05-02"), session="AMC"),
            module.FillModel(0.25),
        )
        exit_ = module.price_structure(
            structure,
            module.ChainSnapshot(ticker="TEST", obs_date=pd.Timestamp("2024-05-03"),
                                 event_date=pd.Timestamp("2024-05-02"),
                                 rows=_chain("TEST", "2024-05-03"), session="AMC"),
            module.FillModel(0.25), pin=entry.legs, closing=True,
        )
        return module.structure_return(entry, exit_)

    assert both(_pricing) == both(legacy_structures)


# --------------------------------------------------------------------------
# replay_one and replay are byte-identical to legacy
# --------------------------------------------------------------------------


def test_replay_one_matches_legacy_on_same_synthetic_input():
    plan_row = _plan_row()
    index = _index()
    old_rows, old_reason = legacy.replay_one(
        legacy_structures.straddle_through(), plan_row, index
    )
    new_rows, new_reason = replay.replay_one(
        _pricing.straddle_through(), plan_row, index
    )
    assert new_reason == old_reason is None
    pd.testing.assert_frame_equal(pd.DataFrame(new_rows), pd.DataFrame(old_rows))


def test_replay_move_matches_legacy_on_same_synthetic_input():
    events = _events()
    cal = _calendar()
    old = legacy.replay(
        "STR-THRU", events, calendar=cal, index=_index()
    )
    new = replay.replay(
        None, None, "STR-THRU", events, calendar=cal, index=_index()
    )

    pd.testing.assert_frame_equal(new.trades, old.trades)
    for key in ("planned", "replayable", "priced", "rows", "coverage",
                "fill_rate", "skipped", "strategy", "variant"):
        assert new.as_dict()[key] == old.as_dict()[key], key


# --------------------------------------------------------------------------
# the store-reaching edge: one real committed snapshot
# --------------------------------------------------------------------------


def _chain_rows(exit_call=(3.0, 3.4), exit_put=(2.0, 2.4)) -> list[dict]:
    rows: list[dict] = []
    for obs_date in ("2024-05-02", "2024-05-03"):
        call = (2.0, 2.4) if obs_date == "2024-05-02" else exit_call
        put = (1.0, 1.4) if obs_date == "2024-05-02" else exit_put
        frame = _chain("TEST", obs_date, call=call, put=put)
        frame["year"] = 2024
        rows.extend(frame.to_dict("records"))
    rows.sort(key=lambda row: (row["ticker"], row["obs_date"], row["expiry"],
                               row["strike"], row["right"]))
    return rows


def _event_rows() -> list[dict]:
    return [
        {
            "event_id": "TEST_2024-05-02", "ticker": "TEST",
            "event_date": pd.Timestamp("2024-05-02"), "year": 2024,
            "session": "AMC", "session_src": "test", "annc_tod": None,
            "src_orats": True, "src_oquants": True, "src_nasdaq": False,
            "src_yfinance": False, "date_agree": True, "date_conflict": False,
            "updated_at": None, "event_cluster_id": None, "claim_count": 1,
            "reconciliation": "test",
        }
    ]


def _published(store, name, rows, partition_key):
    contract = contract_for(name)
    ref = contract_ref_for(contract)
    return contract, ref, publish_and_inspect(store, contract, ref, rows, partition_key)


def _commit(conn, clock, store, *, chain_rows, event_rows, receipt_id,
            expected_head=None, generation=0):
    """A real multi-table snapshot with an explicit expected head.

    ``tests/data_scan_support.commit_tables`` is hardcoded to an empty scope;
    the moved-head case needs the same commit with the prior head pinned.
    """
    oc_contract, oc_ref, oc_record = _published(store, "option_chains", chain_rows, "2024")
    ee_contract, ee_ref, ee_record = _published(store, "earnings_events", event_rows, "2024")
    contracts = {"option_chains": oc_contract, "earnings_events": ee_contract}
    tables = {"option_chains": [oc_record], "earnings_events": [ee_record]}
    table_manifests: dict[str, DatasetManifest] = {}
    all_records, all_objects = [], []
    for table_name, records in tables.items():
        table_manifests[table_name] = manifests.dataset_manifest(
            contract_ref_for(contracts[table_name]), records,
            knowledge_mode="reconstructed", coverage_receipt_refs=(fake_hash("cov"),),
            availability_evidence_refs=())
        all_records.extend(records)
        all_objects.extend(r.object_ref for r in records)
    snap = manifests.snapshot_ref(
        table_manifests, calendar_version="cal.v1",
        source_priority_version="prio.v1", finality_receipt_refs=(fake_hash("fin"),),
    )
    receipt = catalog.commit_snapshot(
        conn, scope="shadow", request_hash=fake_hash(f"{receipt_id}-request"),
        contracts=list(contracts.values()), objects=all_objects, records=all_records,
        manifests=list(table_manifests.values()), snapshot=snap,
        expected_head_snapshot_id=expected_head, expected_head_generation=generation,
        receipt_id=receipt_id, attempt_id="att-1", fence=1,
        fence_check=lambda _conn: None, clock=clock, store=store,
    )
    return receipt.snapshot_ref


def test_replay_output_carries_pinned_snapshot_id(tmp_path, monkeypatch):
    conn, clock, store = catalog_and_store(tmp_path)
    snap = _commit(conn, clock, store, chain_rows=_chain_rows(),
                   event_rows=_event_rows(), receipt_id="r1")
    monkeypatch.setattr(replay, "trading_calendar", lambda: _calendar())
    repository = Repository(conn, store)
    events = replay._events_frame(repository, snap)
    outcome = replay.run(
        repository, strategies=["STR-THRU"], events=events,
        reports_dir=tmp_path / "reports", snapshot_id=snap.snapshot_id,
        stamp="slice7",
    )
    assert outcome["snapshot_id"] == snap.snapshot_id
    assert len(outcome["trades"]) == 5
    assert set(outcome["trades"]["provenance"]) == {"engine.v2.research.replay"}
    assert set(outcome["trades"]["snapshot_id"]) == {snap.snapshot_id}
    report = json.loads(Path(outcome["path"]).read_text())
    assert report["snapshot_id"] == snap.snapshot_id
    assert report["results"][0]["strategy"] == "STR-THRU"
    conn.close()


def test_replay_explicit_snapshot_id_ignores_a_moved_head(tmp_path, monkeypatch):
    conn, clock, store = catalog_and_store(tmp_path)
    first = _commit(conn, clock, store, chain_rows=_chain_rows(exit_call=(3.0, 3.4)),
                    event_rows=_event_rows(), receipt_id="r1")
    second = _commit(conn, clock, store, chain_rows=_chain_rows(exit_call=(5.0, 5.4)),
                     event_rows=_event_rows(), receipt_id="r2",
                     expected_head=first.snapshot_id, generation=1)
    assert second.snapshot_id != first.snapshot_id
    monkeypatch.setattr(replay, "trading_calendar", lambda: _calendar())
    repository = Repository(conn, store)
    events = replay._events_frame(repository, first)

    pinned = replay.run(repository, strategies=["STR-THRU"], events=events,
                        reports_dir=tmp_path / "pinned", snapshot_id=first.snapshot_id,
                        stamp="pinned")
    head = replay.run(repository, strategies=["STR-THRU"], events=events,
                      reports_dir=tmp_path / "head", scope="shadow", stamp="head")

    # alpha=0 exit value sells the exit bid: FIRST's 3.0+2.0, not SECOND's 5.0+2.0.
    pinned_exit = pinned["trades"].loc[pinned["trades"]["fill_alpha"] == 0.0, "exit_value"]
    head_exit = head["trades"].loc[head["trades"]["fill_alpha"] == 0.0, "exit_value"]
    assert list(pinned_exit) == [5.0]
    assert list(head_exit) == [7.0]
    conn.close()


def test_replay_refuses_a_corrupt_manifest(tmp_path, monkeypatch):
    conn, clock, store = catalog_and_store(tmp_path)
    snap = _commit(conn, clock, store, chain_rows=_chain_rows(),
                   event_rows=_event_rows(), receipt_id="r1")
    conn.execute("DROP TRIGGER data_version_fragments_no_delete")
    conn.execute("DROP TRIGGER data_version_fragments_no_update")
    dsv_id = conn.execute(
        "SELECT dataset_version_id FROM data_snapshot_tables "
        "WHERE snapshot_id = ? AND table_name = 'option_chains'",
        (snap.snapshot_id,)).fetchone()[0]
    conn.execute("DELETE FROM data_version_fragments WHERE dataset_version_id = ?",
                 (dsv_id,))
    monkeypatch.setattr(replay, "trading_calendar", lambda: _calendar())
    repository = Repository(conn, store)
    with pytest.raises(DataError) as err:
        replay.run(repository, strategies=["STR-THRU"], events=_events(),
                   reports_dir=tmp_path / "reports", snapshot_id=snap.snapshot_id)
    assert err.value.code == "MANIFEST_CORRUPT"
    conn.close()


# --------------------------------------------------------------------------
# negative control: a missing chain index entry refuses the event, no partial row
# --------------------------------------------------------------------------


def test_replay_missing_chain_index_entry_refuses_not_partial():
    plan_row = _plan_row()
    bad_chain = _chain("TEST", "2024-05-03")
    bad_chain["bid"] = np.nan
    bad_chain["ask"] = np.nan
    index = replay.ChainIndex(
        {
            ("TEST", pd.Timestamp("2024-05-02")): _chain("TEST", "2024-05-02"),
            ("TEST", pd.Timestamp("2024-05-03")): bad_chain,
        }
    )
    rows, reason = replay.replay_one(_pricing.straddle_through(), plan_row, index)
    assert reason == "bad_quote"
    assert rows == []

    result = replay.replay(None, None, "STR-THRU", _events(),
                           calendar=_calendar(), index=index)
    assert len(result.trades) == 0
    assert result.skipped["bad_quote"] == 1
    assert result.planned == 1
