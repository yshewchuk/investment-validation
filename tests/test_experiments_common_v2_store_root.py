"""Regression: ``common_v2``'s ``store_root`` is the OPS root, not
``<ops_root>/objects`` — ``ArtifactStore(root)`` appends ``objects/`` (and
``attempts/``) itself (``engine/v2/foundation/artifacts.py``).

EXP-147's ``run.py`` used to set ``V2_STORE_ROOT = <ops root>/objects``, which
made the real store resolve one ``objects/`` too deep: the catalog verified
fine, then every published fragment's object was unreachable. This test
proves the convention end-to-end through the full
``common_v2.load_v2_trades`` call: the correct root finds the row, the old
``<ops root>/objects`` root does not. It uses the full loader (rather than a
direct ``ArtifactStore``/``Repository`` round-trip) because the real fixture
is small — ``tests/data_scan_support`` plus the ``_trade_row``/``_event_rows``
helpers — following ``tests/test_v2_research_experiment_trades.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.data.errors import DataError  # noqa: E402
from engine.v2.research import experiment_trades  # noqa: E402
from experiments import common_v2  # noqa: E402
from tests.data_scan_support import (  # noqa: E402
    catalog_and_store,
    commit_tables,
    contract_for,
    contract_ref_for,
    publish_and_inspect,
)
from tests.test_v2_research_build_trades import _trade_row  # noqa: E402
from tests.test_v2_research_replay import _event_rows  # noqa: E402

PROVENANCE = experiment_trades.PROVENANCE


def test_load_v2_trades_store_root_is_the_ops_root_not_root_objects(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    trades_contract = contract_for("trades")
    events_contract = contract_for("earnings_events")
    trades_record = publish_and_inspect(
        store, trades_contract, contract_ref_for(trades_contract),
        [_trade_row("T-THRU-A", "STR-THRU", PROVENANCE)], "2024")
    events_record = publish_and_inspect(
        store, events_contract, contract_ref_for(events_contract),
        _event_rows(), "2024")
    snapshot = commit_tables(
        conn, clock,
        {"trades": [trades_record], "earnings_events": [events_record]},
        {"trades": trades_contract, "earnings_events": events_contract},
        store=store)
    conn.close()

    # ``catalog_and_store``'s ArtifactStore root -- the value ``V2_STORE_ROOT``
    # must hold. The published fragments live under ``<ops_root>/objects/...``.
    ops_root = tmp_path / "store"
    assert (ops_root / "objects").is_dir()

    trades = common_v2.load_v2_trades(
        "STR-THRU", catalog=tmp_path / "catalog.sqlite", store_root=ops_root,
        snapshot_id=snapshot.snapshot_id)
    assert list(trades["trade_id"]) == ["T-THRU-A"]
    assert list(trades["session"]) == ["AMC"]

    # The old, buggy convention: ``run.py`` pointed V2_STORE_ROOT at
    # ``<ops_root>/objects``, so ``ArtifactStore`` appended a second
    # ``objects/`` and the fragment was unreachable. The observed behavior of
    # the real read path (not a guessed exception type) is the scan's object
    # verification refusing with OBJECT_CORRUPT, wrapping ArtifactStore's
    # MISSING.
    with pytest.raises(DataError) as excinfo:
        common_v2.load_v2_trades(
            "STR-THRU", catalog=tmp_path / "catalog.sqlite",
            store_root=ops_root / "objects", snapshot_id=snapshot.snapshot_id)
    assert excinfo.value.code == "OBJECT_CORRUPT"
