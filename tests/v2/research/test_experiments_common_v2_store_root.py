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

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
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

_RUN_PY = ROOT / "experiments" / "EXP-147_str_thru_gate_promotion_confirmatory_val" / "run.py"
_spec = importlib.util.spec_from_file_location("exp147_run_for_test", _RUN_PY)
_exp147_run = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_exp147_run)


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
    assert _exp147_run.V2_STORE_ROOT.name != "objects", (
        "run.py's V2_STORE_ROOT regressed to the ops_root/\"objects\" bug"
    )

    trades = common_v2.load_v2_trades(
        "STR-THRU", catalog=tmp_path / "catalog.sqlite", store_root=ops_root,
        snapshot_id=snapshot.snapshot_id, as_of_month="2025-01")
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
            store_root=ops_root / "objects", snapshot_id=snapshot.snapshot_id, as_of_month="2025-01")
    assert excinfo.value.code == "OBJECT_CORRUPT"


def test_pinned_runner_builds_the_native_v2_provenance():
    """The EXP-147 runner must hand its analog population the native v2 tag, not
    the legacy default, and must obtain that tag from the public v2 module rather
    than an inline literal or a private/legacy source.

    The runner's ``main()`` loads the spec, the pinned v2 snapshot, and the
    trained gate — none of which a test may execute here. So the wiring is
    verified at the call site: the module bound the constant, and the
    ``ga.build_dataset`` call passes it. The Scorer actually selecting on that
    tag is covered in ``test_score.py``; the forwarding from ``build_dataset``
    to ``Scorer`` in ``test_training.py``.

    The call-site check is structural (AST-based) rather than a text-search.
    """
    # Imported from the public v2 module, and the same constant the loader uses.
    assert getattr(_exp147_run, "experiment_trades", None) is experiment_trades
    assert _exp147_run.experiment_trades.PROVENANCE == PROVENANCE

    # The build_dataset call site passes that constant through the selector.
    tree = ast.parse(_RUN_PY.read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "build_dataset"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "ga"
    ]
    assert len(calls) == 1, (
        f"expected exactly one ga.build_dataset(...) call in run.py, found {len(calls)}"
    )

    keywords = {kw.arg: kw.value for kw in calls[0].keywords}
    assert "trade_provenance" in keywords, (
        "the ga.build_dataset(...) call in run.py is missing the trade_provenance keyword"
    )

    # It never inlines a provenance string as the selector (only the constant):
    # only the exact experiment_trades.PROVENANCE attribute access satisfies this.
    value = keywords["trade_provenance"]
    assert (
        isinstance(value, ast.Attribute)
        and value.attr == "PROVENANCE"
        and isinstance(value.value, ast.Name)
        and value.value.id == "experiment_trades"
    ), (
        "trade_provenance must be passed as experiment_trades.PROVENANCE, "
        "not an inline literal or a different name"
    )
