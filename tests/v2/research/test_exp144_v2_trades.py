"""Tier-0 tests for the v2 routing helpers EXP-144's runner now exposes.

EXP-144's full ``main()`` needs the feature panel, SPY daily history and the
EXP-142 factor-simulation module, none of which are test fixtures, so these
tests import its ``run.py`` as a module -- the same
``importlib.util.spec_from_file_location`` technique
``experiments/EXP-185_str_runup_t14_corrected_calendar_gate_rebaseline_registered/run.py``
uses -- and call the standalone helpers directly. ``main()`` is never run.
"""
from __future__ import annotations

import ast
import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from engine.v2.research import experiment_trades  # noqa: E402
from tests.data_scan_support import catalog_and_store  # noqa: E402
from tests.test_v2_research_build_trades import _commit_all  # noqa: E402
from tests.test_v2_research_experiment_trades import (  # noqa: E402
    _RANDOM,
    _ROLLING,
    _SAFE,
    _holdout_snapshot,
)

PROVENANCE = experiment_trades.PROVENANCE
LEGACY_PROVENANCE = "engine.replay"
RUNNER_DIR = "EXP-144_str_runup_t14_corrected_calendar_gate_rebaseline"

#: gate_midfill_str_runup's registered threshold, as EXP-144's spec.yaml
#: incumbent block pins it.
STORED_THRESHOLD = 0.0725137593996064

#: The columns EXP-144's ``load_trades`` projects to, in its own order.
PROJECTED_COLUMNS = [
    "trade_id", "kind", "strategy", "variant", "ticker", "event_id",
    "event_date", "legs", "entry_date", "exit_date", "strike", "expiry",
    "fill_alpha", "entry_cost", "exit_value", "ret", "provenance",
    "snapshot_id", "holdout_as_of_month", "random_membership_version",
    "rolling_membership_version", "population_use",
]
DATE_COLUMNS = ("event_date", "entry_date", "exit_date", "expiry")


def _exp144():
    """A fresh module object per call, so a test can point ``V2_CATALOG`` /
    ``V2_STORE_ROOT`` at its own ``tmp_path`` without touching another test's
    (or the real ``private/ops`` root's) view of them."""
    source = ROOT / "experiments" / RUNNER_DIR / "run.py"
    module_spec = importlib.util.spec_from_file_location(
        "exp144_helpers_under_test", source
    )
    module = importlib.util.module_from_spec(module_spec)
    assert module_spec.loader is not None
    module_spec.loader.exec_module(module)
    return module


def _runup_row(trade_id, variant, provenance, *, fill_alpha=0.5):
    """One STR-RUNUP trade row, shaped like the v2 ``trades`` contract.

    ``event_id`` is the ``tests/test_v2_research_replay._event_rows`` event, so
    ``experiment_trades.load_trades`` finds a session to join.
    """
    event_date = pd.Timestamp("2024-05-02")
    return {
        "trade_id": trade_id, "kind": "sim", "strategy": "STR-RUNUP",
        "variant": variant, "ticker": "TEST", "event_id": "TEST_2024-05-02",
        "event_date": event_date, "year": 2024, "legs": "{}",
        "entry_date": pd.Timestamp("2024-04-30"),
        "exit_date": event_date, "strike": 100.0,
        "expiry": pd.Timestamp("2024-06-21"), "fill_alpha": fill_alpha,
        "entry_cost": 3.8, "exit_value": 4.2, "ret": (4.2 - 3.8) / 3.8,
        "provenance": provenance,
    }


def test_require_v2_snapshot_id_refuses_a_missing_or_null_pin():
    module = _exp144()

    for spec in ({"id": "EXP-144"}, {"id": "EXP-144", "v2_snapshot_id": None}):
        with pytest.raises(SystemExit) as excinfo:
            module.require_v2_snapshot_id(spec)
        assert "v2_snapshot_id" in str(excinfo.value)

    assert module.require_v2_snapshot_id(
        {"id": "EXP-185", "v2_snapshot_id": "snap_abc"}
    ) == "snap_abc"


def test_load_trades_reads_the_pinned_v2_snapshot(tmp_path):
    module = _exp144()
    module.V2_CATALOG = tmp_path / "catalog.sqlite"
    module.V2_STORE_ROOT = tmp_path / "store"
    conn, clock, store = catalog_and_store(tmp_path)
    wanted = module.VARIANT
    snapshot = _commit_all(conn, clock, store, receipt_id="r1", trades_rows=[
        _runup_row("T-RUNUP-MID", wanted, PROVENANCE, fill_alpha=0.5),
        _runup_row("T-RUNUP-WORST", wanted, PROVENANCE, fill_alpha=0.0),
        # planted decoys: a different variant, and a legacy-provenance row for
        # the wanted variant. Neither may survive the v2 read plus the filter.
        _runup_row("T-OTHER-VARIANT", "e+0_x+1", PROVENANCE),
        _runup_row("T-LEGACY-PROVENANCE", wanted, LEGACY_PROVENANCE),
    ])

    frame = module.load_trades(snapshot.snapshot_id, as_of_month="2025-01")
    conn.close()

    assert sorted(frame["trade_id"].astype(str)) == ["T-RUNUP-MID", "T-RUNUP-WORST"]
    assert set(frame["event_id"].astype(str)) == {"TEST_2024-05-02"}
    assert set(frame["variant"].astype(str)) == {wanted}
    assert list(frame.columns) == PROJECTED_COLUMNS
    assert set(frame["snapshot_id"]) == {snapshot.snapshot_id}
    assert set(frame["holdout_as_of_month"]) == {"2025-01"}
    for column in DATE_COLUMNS:
        assert pd.api.types.is_datetime64_any_dtype(frame[column]), column


def test_add_champion_decisions_selects_on_the_stored_threshold():
    module = _exp144()
    scores = pd.DataFrame({
        "event_id": ["E-LOW", "E-AT", "E-HIGH", "E-ZERO"],
        "incumbent_complete_case": [
            0.05, STORED_THRESHOLD, STORED_THRESHOLD + 0.02, 0.0,
        ],
        "incumbent_complete_case_pwin": [0.10, 0.40, 0.90, 0.05],
    })

    out = module.add_champion_decisions(scores, STORED_THRESHOLD)

    assert list(out["selected_champion"]) == [False, True, True, False]
    assert list(out["champion_pwin"]) == list(scores["incumbent_complete_case_pwin"])
    # the candidate's own columns are untouched by the copy
    assert "selected_champion" not in scores.columns


def test_add_champion_decisions_treats_a_nan_score_as_not_selected():
    module = _exp144()
    scores = pd.DataFrame({
        "event_id": ["E-NAN"],
        "incumbent_complete_case": [float("nan")],
        "incumbent_complete_case_pwin": [float("nan")],
    })

    out = module.add_champion_decisions(scores, STORED_THRESHOLD)

    assert list(out["selected_champion"]) == [False]
    assert pd.isna(out["champion_pwin"]).all()


def test_main_forces_both_membership_unbound_caches_to_recompute():
    module = _exp144()
    tree = ast.parse(Path(module.__file__).read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    for name in ("build_dataset", "generate_scores"):
        calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == name]
        assert len(calls) == 1
        force = next(kw.value for kw in calls[0].keywords if kw.arg == "force")
        assert isinstance(force, ast.Constant) and force.value is True
    comparisons = [node for node in ast.walk(main) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Name)
                   and node.func.id in ("eligible_incumbent_comparison", "incumbent_reproduction")]
    assert len(comparisons) == 1
    assert comparisons[0].func.id == "eligible_incumbent_comparison"
    assert [arg.id for arg in comparisons[0].args] == ["scores", "spec", "trades"]
    writes = [node for node in ast.walk(main) if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Name) and node.func.id == "write_eligible_scores"]
    assert len(writes) == 1 and writes[0].args[0].id == "oos"
    evaluations = [node for node in ast.walk(main) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Name) and node.func.id == "evaluate"]
    assert len(evaluations) == 4
    for call in evaluations:
        files = next(kw.value for kw in call.keywords if kw.arg == "input_files")
        assert any(isinstance(item, ast.Name) and item.id == "eligible_scores_path" for item in files.elts)
    literals = [node.value for node in ast.walk(main) if isinstance(node, ast.Constant)
                and isinstance(node.value, str)]
    assert literals.count("eligible_incumbent_comparison.json") == 2
    assert "incumbent_reproduction.json" not in literals
    assert "incumbent_reproduction" not in literals


def test_dataset_recompute_does_not_return_a_stale_holdout_cache(tmp_path, monkeypatch):
    module = _exp144()
    module.RESULTS = tmp_path
    stale = pd.DataFrame([_runup_row("OLD", module.VARIANT, PROVENANCE)])
    stale["event_id"] = "EVENT-5"
    cache = tmp_path / "factor_dataset.parquet"
    stale.to_parquet(cache, index=False)
    eligible = stale.copy()
    eligible["event_id"] = "EVENT-0"
    assert module.build_dataset(eligible, force=False)["event_id"].tolist() == ["EVENT-5"]

    class RecomputeReached(Exception):
        pass

    def source(**kwargs):
        raise RecomputeReached

    # Only the expensive feature source is replaced; the actual cache guard
    # reads a real stale Parquet artifact for the negative control above.
    monkeypatch.setattr(module.FeatureContext, "load", source)
    before = cache.read_bytes()
    with pytest.raises(RecomputeReached):
        module.build_dataset(eligible, force=True)
    assert cache.read_bytes() == before


def test_score_recompute_replaces_stale_holdout_rows(tmp_path, monkeypatch):
    module = _exp144()
    module.RESULTS = tmp_path
    score_dir = tmp_path / "score_folds"
    score_dir.mkdir()
    score_path = score_dir / "scores_2019.parquet"
    pd.DataFrame({"event_id": ["EVENT-5"], "event_date": [pd.Timestamp("2019-01-15")],
                  "year": [2019]}).to_parquet(score_path, index=False)
    (score_dir / "diagnostics_2019.json").write_text("{}")
    base = {"ticker": "TEST", "event_date": pd.Timestamp("2018-01-15"), "year": 2018,
            "ret": 0.0, "quote_present": True, "mcap_log": 0.0, "relative_spread": 0.0}
    dataset = pd.DataFrame([{**base, "event_id": f"TRAIN-{i}"} for i in range(500)] + [
        {**base, "event_id": "EVENT-0", "year": 2019, "event_date": pd.Timestamp("2019-01-15")}])
    sim_data = pd.DataFrame({"year": []})
    assert module.generate_scores(dataset, sim_data, 1, force=False)[0]["event_id"].tolist() == ["EVENT-5"]
    monkeypatch.setattr(module, "fit_direct", lambda train, test, features, complete_case:
                        (module.np.zeros(len(test)), module.np.zeros(len(test)), len(train)))
    scores, _ = module.generate_scores(dataset, sim_data, 1, force=True)
    assert scores["event_id"].tolist() == ["EVENT-0"]
    assert pd.read_parquet(score_path)["event_id"].tolist() == ["EVENT-0"]


@pytest.mark.parametrize("registered", [False, True])
@pytest.mark.parametrize("passes", [False, True])
def test_eligible_comparison_preserves_history_without_claiming_reproduction(tmp_path, registered, passes):
    module = _exp144()
    conn, repository, snapshot = _holdout_snapshot(tmp_path, [_SAFE, _RANDOM, _ROLLING])
    try:
        trades = experiment_trades.load_trades(repository, snapshot, "STR-THRU", as_of_month="2024-10")
        scores = pd.DataFrame({"event_id": trades["event_id"], "year": [2024],
                               "incumbent_complete_case": [0.25]})
        spec = {"incumbent": {"stored_threshold": 0.5, "expected_oos_rows": 3,
                              "expected_selected_at_stored_threshold": 2}}
        if registered:
            ids = ["EVENT-0", "EVENT-5", "EVENT-1"]
            module.REGISTERED_POPULATION_PATH = tmp_path / "registered.parquet"
            pd.DataFrame({"event_id": ids}).to_parquet(module.REGISTERED_POPULATION_PATH)
            spec["incumbent"].update(population_event_id_count=len(ids),
                                     population_event_id_sha256=module._population_digest(ids))
        before = copy.deepcopy(spec)
        with pytest.raises(RuntimeError):
            module.incumbent_reproduction(scores, spec)
        result = module.eligible_incumbent_comparison(scores, spec, trades)
        assert result["historical_reproduction"] is False
        assert result["population_use"] == "post-release selection"
        assert result["oos_rows"] == 1
        assert result["selected_at_stored_threshold"] == 0
        assert result["historical_expectations"] == before["incumbent"]
        assert spec == before
        assert result["holdout_context"] == {key: trades.iloc[0][key] for key in (
            "snapshot_id", "holdout_as_of_month", "random_membership_version", "rolling_membership_version")}
        assert result["holdout_exclusions"] == trades.attrs["holdout_exclusions"]
        _assert_eligible_report_label(module, result, passes)
        module.write_json(tmp_path / "eligible_incumbent_comparison.json", result)
        assert '"historical_reproduction": false' in (tmp_path / "eligible_incumbent_comparison.json").read_text()
        result["historical_expectations"]["expected_oos_rows"] = 999
        assert spec == before
    finally:
        conn.close()


def _assert_eligible_report_label(module, comparison, passes):
    metrics = {"n": 1, "mean": 0, "dollar_weighted": 0, "cagr": 0, "sharpe_trade": 0,
               "years_positive": 0, "years_evaluated": 1, "breakeven_alpha": None}
    result = SimpleNamespace(results={"headline": metrics})
    primary_metrics = dict(metrics, mean=1, dollar_weighted=1, cagr=1, sharpe_trade=1,
                           years_positive=1, breakeven_alpha=0.5) if passes else metrics
    primary = SimpleNamespace(results={"headline": primary_metrics})
    if passes:
        metrics["breakeven_alpha"] = 1
    ranks = {arm: {"n": 1, "spearman": 0, "top_bottom_decile": 0} for arm in module.ALL_ARMS}
    if passes:
        ranks[module.PRIMARY].update(spearman=1, top_bottom_decile=1)
    counts = dict.fromkeys(["priced", "oos", "incumbent_scoreable", "native_scoreable",
                           "simulation_scoreable", "quote_present", "quote_absent"], 1)
    sections = module.report_sections(primary, dict.fromkeys(module.ALL_ARMS, result), ranks, [],
        {"ci90": [0.1 if passes else -1, 1], "observed": 0, "p_gt_zero": 0.5}, [], comparison, counts)
    text = " ".join(sections[0]["body"])
    assert "Post-release selection incumbent comparison" in text
    assert "historical reproduction not claimed" in text
    assert "no final holdout conclusion" in text
    assert "Incumbent reproduction:" not in text
    assert "PROMOTION CRITERIA MET" not in text
    assert ("SELECTION COMPARISON CRITERIA MET" if passes else "SELECTION COMPARISON CRITERIA NOT MET") in text
    assert sections[2]["title"] == "Selection comparison checks"


@pytest.mark.parametrize("corruption", ["excluded_score", "mixed_context"])
def test_eligible_comparison_refuses_population_or_context_drift_before_output(tmp_path, corruption):
    module = _exp144()
    conn, repository, snapshot = _holdout_snapshot(tmp_path, [_SAFE, _RANDOM])
    try:
        trades = experiment_trades.load_trades(repository, snapshot, "STR-THRU", as_of_month="2024-10")
        scores = pd.DataFrame({"event_id": ["EVENT-0"], "year": [2024], "incumbent_complete_case": [0.25]})
        if corruption == "excluded_score":
            scores.loc[0, "event_id"] = "EVENT-5"
        else:
            other = trades.copy()
            other["snapshot_id"] = "another-snapshot"
            trades = pd.concat([trades, other], ignore_index=True)
        with pytest.raises(RuntimeError, match="pinned eligible"):
            result = module.eligible_incumbent_comparison(scores, {"incumbent": {"stored_threshold": 0.5}}, trades)
            module.write_json(tmp_path / "comparison.json", result)
        assert not (tmp_path / "comparison.json").exists()
    finally:
        conn.close()


def test_eligible_score_writer_preserves_the_registered_historical_artifact(tmp_path):
    module = _exp144()
    module.RESULTS = tmp_path
    module.REGISTERED_POPULATION_PATH = tmp_path / "oos_scores.parquet"
    ids = ["EVENT-0", "EVENT-5", "EVENT-1"]
    pd.DataFrame({"event_id": ids}).to_parquet(module.REGISTERED_POPULATION_PATH, index=False)
    before = module.REGISTERED_POPULATION_PATH.read_bytes()
    digest = module._population_digest(ids)
    conn, repository, snapshot = _holdout_snapshot(tmp_path, [_SAFE, _RANDOM, _ROLLING])
    try:
        eligible = experiment_trades.load_trades(repository, snapshot, "STR-THRU", as_of_month="2024-10")
        path = module.write_eligible_scores(eligible[["event_id"]])
        assert path == tmp_path / "eligible_oos_scores.parquet"
        assert pd.read_parquet(path)["event_id"].tolist() == ["EVENT-0"]
        assert module.REGISTERED_POPULATION_PATH.read_bytes() == before
        assert module._load_registered_population(module.REGISTERED_POPULATION_PATH, len(ids), digest) == set(ids)
    finally:
        conn.close()


@pytest.mark.parametrize("argument,code", [("--force", 2), ("--help", 0)])
def test_cli_removes_ineffective_force_without_running_the_experiment(tmp_path, monkeypatch, capsys, argument, code):
    module = _exp144()
    module.RESULTS = tmp_path / "never-created"
    monkeypatch.setattr(sys, "argv", ["run.py", "--holdout-as-of-month", "2024-10", argument])
    with pytest.raises(SystemExit) as caught:
        module.main()
    assert caught.value.code == code
    captured = capsys.readouterr()
    if code:
        assert "unrecognized arguments: --force" in captured.err
    else:
        assert "--force" not in captured.out
        assert "--holdout-as-of-month" in captured.out
    assert not module.RESULTS.exists()
