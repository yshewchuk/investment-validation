"""Native per-arm consumption over real committed synthetic Parquet inputs."""
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from engine.v2.data.errors import DataError
from engine.v2.data.repository import Repository
from engine.v2.ops.errors import OpsError
from engine.v2.ops.experiments import experiment_spec_from_document
from experiments import v2_gate_variant as gate
from tests.data_scan_support import (
    catalog_and_store,
    commit_tables,
    contract_for,
    contract_ref_for,
    publish_and_inspect,
)
from tests.test_v2_research_replay import _chain_rows, _commit, _event_rows


def _document(alpha=0.0):
    """Build the smallest supported single-arm synthetic specification."""
    return {"experiment_id": "synthetic-gate", "hypothesis": "synthetic fill check",
            "primary_arm_id": "candidate", "arms": ["candidate"], "folds": [],
            "seed": 0, "economic_params": {"fill": alpha},
            "price_source": "option_chains", "runner": "native-gate-replay"}


@pytest.fixture
def source(tmp_path):
    """Commit real option, calendar and event fragments into a temporary catalog."""
    conn, clock, store = catalog_and_store(tmp_path)
    snapshot = _commit(conn, clock, store, chain_rows=_chain_rows(),
                       event_rows=_event_rows(), receipt_id="gate-source")
    yield conn, clock, store, Repository(conn, store), snapshot
    conn.close()


def _price(source, alpha=0.0, **kwargs):
    """Execute one independently resolved arm on the fixture's exact pin."""
    return gate.price_variant(source[3], source[4],
                              experiment_spec_from_document(_document(alpha)),
                              strategy="STR-THRU", as_of_month="2025-01", **kwargs)


def test_changed_fill_is_consumed_by_real_native_replay(source):
    """Both alpha arms must match independently calculated cash flows."""
    worst, best = _price(source), _price(source, 1.0)
    # Two long legs: entry bid total 3, ask total 3.8; exit bid 5, ask 5.8.
    for result, alpha, cost, exit_value, pnl in ((worst, 0.0, 3.8, 5.0, 1.2),
                                               (best, 1.0, 3.0, 5.8, 2.8)):
        row, = result.replay.trades.to_dict("records")
        assert row["entry_cost"] == pytest.approx(cost)
        assert row["exit_value"] == pytest.approx(exit_value)
        assert row["pnl"] == pytest.approx(pnl)
        assert row["fill_alpha"] == alpha
        assert result.stage_inputs.fill_alpha == alpha
        assert result.stage_inputs.snapshot_id == source[4].snapshot_id
        assert result.stage_inputs.event_ids == ("TEST_2024-05-02",)
        assert result.stage_inputs.holdout_as_of_month == "2025-01"
    assert worst.execution_plan is not best.execution_plan
    assert worst.execution_plan.json_bytes() != best.execution_plan.json_bytes()
    assert worst.replay.trades is not best.replay.trades
    with pytest.raises(TypeError):
        worst.execution_plan.economic_params["fill"] = 1.0
    with pytest.raises(FrozenInstanceError):
        worst.stage_inputs.fill_alpha = 1.0


def test_replay_cannot_consume_prior_trade_outputs(source):
    """A rerun rebuilds prices rather than consuming the prior result frame."""
    assert "trades" not in source[4].table_versions
    result = _price(source)
    result.replay.trades.loc[:, "entry_cost"] = 999.0
    replayed = _price(source)
    assert replayed.replay.trades.iloc[0]["entry_cost"] == pytest.approx(3.8)


@pytest.mark.parametrize("economics", [
    {}, {"fill": True}, {"fill": "0.5"}, {"fill": -0.1}, {"fill": 1.1},
    {"fill": float("nan")}, {"fill": 10**1000},
    {"fill": 0.0, "strike_offset": 1},
    {"fill": 0.0, "exit": {"kind": "fixed_day", "trading_days": 1}},
])
def test_unused_or_invalid_economics_refuse_before_reads(economics):
    """Bad declarations fail before the deliberately absent repository is touched."""
    spec = experiment_spec_from_document({**_document(), "economic_params": economics})
    with pytest.raises(OpsError) as caught:
        gate.price_variant(None, None, spec, strategy="STR-THRU", as_of_month="2025-01")
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"


@pytest.mark.parametrize("changed", [
    {"runner": "synthetic"}, {"price_source": "old-experiment-output"},
    {"arms": ["candidate", "other"]},
    {"input_files": ["experiments/previous/results.parquet"]},
])
def test_unsupported_plan_refuses_before_reads(changed):
    """Unsupported runner, source, arm shape and file inputs cannot reach scans."""
    spec = experiment_spec_from_document({**_document(), **changed})
    with pytest.raises(OpsError) as caught:
        gate.price_variant(None, None, spec, strategy="STR-THRU", as_of_month="2025-01")
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"


def test_unsupported_strategy_refuses_before_reads():
    """An unknown strategy cannot reach population or quote access."""
    with pytest.raises(OpsError) as caught:
        gate.price_variant(None, None, experiment_spec_from_document(_document()),
                           strategy="unknown", as_of_month="2025-01")
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"


def test_missing_one_events_quotes_refuses_whole_variant(source):
    """A successfully priced sibling must not hide missing event quotes."""
    conn, clock, store, repository, old = source
    events = _event_rows() + [{**_event_rows()[0], "event_id": "MISSING_2024-05-02",
                              "ticker": "MISSING"}]
    snapshot = _commit(conn, clock, store, chain_rows=_chain_rows(),
                       event_rows=sorted(events, key=lambda row: row["event_id"]),
                       receipt_id="missing-quotes",
                       expected_head=old.snapshot_id, generation=1)
    with pytest.raises(OpsError) as caught:
        gate.price_variant(repository, snapshot, experiment_spec_from_document(_document()),
                           strategy="STR-THRU", as_of_month="2025-01")
    assert caught.value.code == "EXPERIMENT_VARIANT_FAILED"


def test_supplied_pin_does_not_follow_a_moving_head(source):
    """Publishing changed quotes cannot retarget a caller's retained pin."""
    conn, clock, store, _, old = source
    _commit(conn, clock, store, chain_rows=_chain_rows(exit_call=(8.0, 8.4)),
            event_rows=_event_rows(), receipt_id="new-head",
            expected_head=old.snapshot_id, generation=1)
    assert _price(source).replay.trades.iloc[0]["pnl"] == pytest.approx(1.2)


def test_holdout_denied_before_any_pricing_population_is_returned(source, tmp_path):
    """An explicit rolling-holdout request refuses without creating any output."""
    spec = experiment_spec_from_document(_document())
    before = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    with pytest.raises(DataError) as caught:
        gate.price_variant(source[3], source[4], spec, strategy="STR-THRU",
                           as_of_month="2024-05", event_ids=["TEST_2024-05-02"])
    assert caught.value.code == "HOLDOUT_ACCESS_DENIED"
    assert before == sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))


def test_missing_requested_event_refuses(source):
    """Unresolved requested membership is denied rather than silently dropped."""
    with pytest.raises(DataError) as caught:
        _price(source, event_ids=["UNKNOWN"])
    assert caught.value.code == "HOLDOUT_ACCESS_DENIED"


def test_holdout_refusal_precedes_even_missing_quote_tables(tmp_path):
    """Holdout admission fails before a quote-free snapshot can reach pricing."""
    conn, clock, store = catalog_and_store(tmp_path)
    try:
        contract = contract_for("earnings_events")
        fragment = publish_and_inspect(store, contract, contract_ref_for(contract),
                                       _event_rows(), "2024")
        snapshot = commit_tables(conn, clock, {"earnings_events": [fragment]},
                                 {"earnings_events": contract}, store=store)
        with pytest.raises(DataError) as caught:
            gate.price_variant(Repository(conn, store), snapshot,
                               experiment_spec_from_document(_document()),
                               strategy="STR-THRU", as_of_month="2024-05",
                               event_ids=["TEST_2024-05-02"])
        assert caught.value.code == "HOLDOUT_ACCESS_DENIED"
    finally:
        conn.close()


def _cli_args(tmp_path):
    """Prepare explicit temporary CLI paths without creating a catalog."""
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(_document()))
    return ["--catalog", str(tmp_path / "catalog.sqlite"),
            "--store-root", str(tmp_path / "store"), "--spec", str(spec_path),
            "--strategy", "STR-THRU", "--as-of-month", "2025-01"]


def test_cli_no_ledger_smoke_writes_no_rows_or_reports(source, tmp_path, capsys):
    """The real CLI preserves all catalog counts and ledger bytes."""
    args = _cli_args(tmp_path)
    ledger = tmp_path / "experiments" / "LEDGER.csv"
    ledger.parent.mkdir()
    ledger.write_text("id,spec_hash,stage\nexisting,unchanged,planned\n")
    before = ledger.read_bytes()
    conn = source[0]
    tables = [row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    counts = {name: conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
              for name in tables}
    assert gate.main([*args, "--no-ledger"]) == 0
    evidence = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert evidence["recording_mode"] == "no-ledger"
    assert evidence["stage_inputs"]["snapshot_id"] == source[4].snapshot_id
    assert evidence["priced_events"] == 1
    assert ledger.read_bytes() == before
    assert counts == {name: conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
                      for name in tables}
    assert not list(tmp_path.rglob("REPORT.md"))


def test_cli_requires_no_ledger_before_opening_catalog(tmp_path):
    """Missing smoke authorization refuses before a catalog can be created."""
    with pytest.raises(SystemExit) as caught:
        gate.main(_cli_args(tmp_path))
    assert caught.value.code == 2
    assert not (tmp_path / "catalog.sqlite").exists()


def test_cli_missing_catalog_has_redacted_refusal(tmp_path, capsys):
    """Storage failures expose a stable code instead of the local path."""
    assert gate.main([*_cli_args(tmp_path), "--no-ledger"]) == 2
    assert json.loads(capsys.readouterr().out) == {"refused": "INPUT_CHANGED"}
    assert not (tmp_path / "catalog.sqlite").exists()


@pytest.mark.parametrize("text", ["not-json", json.dumps({**_document(), "input_files": None})])
def test_cli_malformed_spec_has_redacted_refusal(tmp_path, capsys, text):
    """Malformed JSON and field types expose only the specification refusal."""
    args = _cli_args(tmp_path)
    (tmp_path / "spec.json").write_text(text)
    assert gate.main([*args, "--no-ledger"]) == 2
    assert json.loads(capsys.readouterr().out) == {"refused": "INVALID_EXPERIMENT_SPEC"}
    assert not (tmp_path / "catalog.sqlite").exists()


def test_documented_pricing_boundary():
    """The documented authority and interface declarations match this consumer."""
    root = Path(__file__).resolve().parents[3]
    doc = (root / "experiments" / "ARCHITECTURE.md").read_text()
    assert "No prepriced trade frame or previous experiment output is accepted" in doc
    assert "no gate fitting or sweep lifecycle" in doc
    assert "`EXPERIMENT_VARIANT_FAILED`; no partial result is returned" in doc
    readme = (root / "engine/v2/research/README.md").read_text()
    interface = readme.split("<!-- public-interface:", 1)[1].split("-->", 1)[0]
    consumers = readme.split("## Consumers", 1)[1].split("## Usage", 1)[0]
    assert "replay.ReplayResult" in interface
    assert "experiments/v2_gate_variant.py" in consumers
