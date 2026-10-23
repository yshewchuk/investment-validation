"""Real pinned synthetic history, option replay, fold fitting and arm recording."""
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from engine.v2.contracts import ArtifactRef
from engine.v2.data import objects
from engine.v2.data.computed_moves_table import COMPUTED_MOVES_CONTRACT
from engine.v2.data.repository import Repository
from engine.v2.foundation import from_document
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.experiments import ExperimentSpec
from experiments import gate_report as gate
from experiments import native_outcomes, native_registration
from experiments import v2_gate as cli
from experiments.record_export import export_native_run
from tests.data_scan_support import (
    catalog_and_store,
    commit_tables,
    contract_for,
    contract_ref_for,
    publish_and_inspect,
)
from tests.test_v2_research_replay import _chain
from tests.v2.research.magnitude_trial_support import (
    history_event,
    outcome_move,
)
from tests.v2.research.scan_progress_support import assert_scan_boundaries, observe_scan_outcomes

REPO = Path(__file__).resolve().parents[3]
MONTH = "2025-01"


def build_gate_spec(alpha=0.0, arm="candidate", **changes):
    return ExperimentSpec(**dict(experiment_id="synthetic-reconstructed-gate", hypothesis="Synthetic gate check",
        primary_arm_id=arm, arms=(arm,), seed=0, folds=("2023", "2024"),
        economic_params={"fill": alpha, "strategy": "STR-THRU"}, price_source="option_chains",
        runner=gate.RUNNER, **changes))


def build_gate_source(tmp_path, *, delayed=None, missing_quotes=False, calendar_start="2020-12-01",
            runup=False, test_quote_shift=0.0, rejected=False, early_january=False,
            feature_driver=None, break_even=False):
    conn, clock, store = catalog_and_store(tmp_path)
    days = ["2021-02-02", "2021-05-04", "2022-02-02", "2022-05-04",
            "2023-02-02", "2023-05-04", "2024-02-02", "2024-05-02"]
    if runup:
        days[3] = "2023-01-03"
        days[5] = "2024-01-01"
    if early_january:
        days[3:6] = ["2022-12-29", "2022-12-30", "2023-01-02"]
    eligible = [history_event(day) for day in days]
    if early_january:
        # A separate ticker keeps its Jan 2 exit quote distinct from the test entry.
        eligible[4] = history_event(days[4], ticker="AFTER")
    if rejected:
        eligible.append(history_event("2023-06-01", ticker="NEW"))
    history = []
    if feature_driver:
        eligible = [history_event((pd.Timestamp(f"{2022 + i // 4}-02-02") + pd.offsets.BDay(i % 4)).date().isoformat(),
                           ticker=f"T{i}") for i in range(8)]
        # Independent quadrants: equal counts with different means, and vice versa.
        counts, means = [1, 1, 2, 2] * 2, [2.0, 8.0, 2.0, 8.0] * 2
        history = [(history_event(day, ticker=event["ticker"]), mean)
                   for event, count, mean in zip(eligible, counts, means, strict=True)
                   for day in ("2021-02-02", "2021-05-04")[:count]]
    protected = [history_event("2022-03-02", protected=True), history_event("2024-12-02")]
    events = eligible + [event for event, _ in history] + protected
    moves = [outcome_move(event, 2.0 * (index + 1)) for index, event in enumerate(eligible)]
    moves += [outcome_move(event, mean) for event, mean in history]
    moves += [outcome_move(event, None) for event in protected]
    if delayed:
        moves[delayed[0]]["available_as_of_date"] = delayed[1]
    chains = []
    for index, event in enumerate(eligible):
        if missing_quotes and index == 6:
            continue
        day = event["event_date"]
        entry_day = day - pd.offsets.BDay(14) if runup else day
        exit_day = day if runup else day + pd.offsets.BDay(1)
        exit_call = (0.2, 0.6) if index in (0, 3) else (2.0, 2.4) if index == 1 else (3.0, 3.4)
        if rejected and index == 8:
            exit_call = (9.0, 9.4)
        exit_put = (0.2, 0.6) if index in (0, 3) else (1.4, 1.8) if index == 1 else (2.0, 2.4)
        if feature_driver:
            positive = counts[index] == 2 if feature_driver == "n_prior" else means[index] == 8.0
            exit_call, exit_put = ((3.0, 3.4), (2.0, 2.4)) if positive else ((0.2, 0.6), (0.2, 0.6))
        if break_even and index == 1:
            exit_call, exit_put = (2.4, 2.8), (1.4, 1.8)
        if index >= 4:
            exit_call = tuple(value + test_quote_shift for value in exit_call)
        for stamp, call, put in ((entry_day, (2.0, 2.4), (1.0, 1.4)), (exit_day, exit_call, exit_put)):
            frame = _chain(event["ticker"], stamp, call=call, put=put)
            frame["expiry"] += day - pd.Timestamp("2024-05-02")
            if runup:
                frame["expiry"] += pd.Timedelta(days=60)
                frame["dte"] += 60
            frame["year"] = stamp.year
            chains.extend(frame.to_dict("records"))
    daily = [{"ticker": "MKT", "date": day, "year": day.year}
             for day in pd.bdate_range(calendar_start, "2024-12-04")]
    contracts = {name: contract_for(name) for name in ("earnings_events", "daily_market", "option_chains")}
    contracts["computed_moves"] = COMPUTED_MOVES_CONTRACT
    tables, hashes = {}, {}
    for name, rows in (("earnings_events", events), ("computed_moves", moves),
                       ("option_chains", chains), ("daily_market", daily)):
        contract, groups = contracts[name], {}
        ref = contract_ref_for(contract)
        for row in rows:
            partition = str(row[contract.partition_columns[0]])
            piece = (row["ticker"], row["event_date"]) if name == "computed_moves" else "all"
            groups.setdefault((partition, piece), []).append(row)
        records = [publish_and_inspect(store, contract, ref,
            sorted(group, key=lambda row: tuple(row[key] for key in contract.primary_key)), partition)
            for (partition, _), group in groups.items()]
        tables[name] = sorted(records, key=lambda row: (row.partition_key, row.primary_key_min))
        partitions = {}
        for record in tables[name]:
            partitions.setdefault(record.partition_key, []).append(record)
        hashes[name] = {key: objects.partition_logical_hash(store, [r.object_ref for r in group], contract, ref, key)
                       for key, group in partitions.items() if len(group) > 1}
    snapshot = commit_tables(conn, clock, tables, contracts, partition_logical_hashes=hashes, store=store)
    return SimpleNamespace(conn=conn, store=store, repository=Repository(conn, store), snapshot=snapshot,
                           events=eligible, protected=protected, root=tmp_path)


@pytest.fixture
def source(tmp_path):
    value = build_gate_source(tmp_path)
    yield value
    value.conn.close()


def _register(source, spec=None, **kwargs):
    spec = spec or build_gate_spec()
    registered = native_registration.register_native(source.conn, source.store, source.repository,
        spec, code_root=REPO, as_of_month=MONTH, **kwargs)
    return spec, registered


def _invoke(source, capsys, action, spec=None, *extra, status=0, required_tables=None,
            exp_number="101"):
    path = source.root / "spec.json"
    path.write_text(json.dumps(asdict(spec or build_gate_spec())))
    arguments = [action, "--spec", str(path), "--catalog", str(source.root / "catalog.sqlite"),
                 "--store-root", str(source.store.root), "--as-of-month", MONTH, *map(str, extra)]
    if action == "run":
        arguments += ["--experiments-root", str(source.root / "experiments")]
        if exp_number is not None:
            arguments += ["--exp-number", str(exp_number)]
    with observe_scan_outcomes({"earnings_events", "daily_market", "option_chains"}) as scans:
        assert cli.main(arguments) == status
    captured = capsys.readouterr()
    stderr = captured.err
    if action == "run" and "--no-record-export" in extra:
        warning = "WARNING: no EXP number; run record NOT exported\n"
        lines = stderr.splitlines(keepends=True)
        assert lines.count(warning) == 1
        stderr = "".join(line for line in lines if line != warning)
    # Replays still admit events, but only fresh evaluation needs calendar/quotes.
    required = {"earnings_events"} if required_tables is None else required_tables
    assert_scan_boundaries(stderr, tables={"earnings_events", "daily_market", "option_chains"},
                           expected_scans=scans, required_tables=required, allow_failures=status != 0)
    return json.loads(captured.out)


def _result(output):
    report = Path(output["report"]).read_text()
    return json.loads(report.split("```json\n", 1)[1].split("\n```", 1)[0])


def _forbid(*args, **kwargs):
    pytest.fail("refusal/replay reached features, prices, fitting or ledger")


def test_real_history_fold_features_scores_and_provenance(source):
    spec, registered = _register(source)
    result = gate.gate_result(source.repository, registered, spec)
    assert result == gate.gate_result(source.repository, registered, spec)
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert result["provenance"] == registered.document
    assert result["availability"] == {"basis": "reconstructed_prior_completed_session",
        "source_publication": "unverified", "event_schedule_as_known": "unverified"}
    assert result["features"] == ["n_prior", "mean_prior_abs_move"]
    assert result["attempted_variants"] == 1
    assert result["all_candidates"] == {"count": 4, "mean_mark_pnl": pytest.approx(1.2)}
    assert result["selected"]["count"] == sum(row["selected"] for row in result["predictions"])
    assert result["holdout_results"] == {"random": "excluded; not evaluated", "rolling": "excluded; not evaluated"}
    for fold, train_count, year in zip(result["folds"], (4, 6), ("2023", "2024"), strict=True):
        assert fold["training_exits_before"] == year + "-01-01"
        assert len(fold["train_history"]) == train_count
        assert len(fold["test_history"]) == 2
        train_x = [row["model_values"] for row in fold["train_history"]]
        test_x = [row["model_values"] for row in fold["test_history"]]
        assert train_x == [[0.0, 0.0]] + [[float(i), float(i + 1)] for i in range(1, train_count)]
        assert test_x == [[float(i), float(i + 1)] for i in range(train_count, train_count + 2)]
        fitted = DecisionTreeClassifier(max_depth=1, random_state=0).fit(train_x, fold["training_labels"])
        threshold = np.quantile(fitted.predict_proba(train_x)[:, 1], 0.5)
        assert fold["threshold"] == threshold
        predicted = [row for row in result["predictions"] if row["fold"] == year]
        assert [row["score"] for row in predicted] == pytest.approx(fitted.predict_proba(test_x)[:, 1])
        assert [row["selected"] for row in predicted] == [row["score"] >= threshold for row in predicted]
        for row in fold["train_history"] + fold["test_history"]:
            assert row["feature_ceiling"] < row["gate_session"]
            history = row["history"]
            assert history["snapshot_id"] == source.snapshot.snapshot_id
            assert history["decision_session"] == row["feature_ceiling"]
            assert "holdout_exclusions" not in history
            assert all(item["target_available_on"] <= row["feature_ceiling"] for item in history["history"])
            assert all(item["object_id"] and item["content_hash"].startswith("sha256:") for item in history["read_inputs"])


@pytest.mark.parametrize("feature,expected", [
    ("n_prior", [0, 0, 1, 1]), ("mean_prior_abs_move", [0, 1, 0, 1]),
])
def test_real_multiticker_histories_require_each_fitted_feature(tmp_path, feature, expected):
    source = build_gate_source(tmp_path, feature_driver=feature)
    try:
        # Historical events supply features but are not enrolled training trades.
        spec, registered = _register(source, replace(build_gate_spec(), folds=("2023",)),
            event_ids=[event["event_id"] for event in source.events])
        result = gate.gate_result(source.repository, registered, spec)
        fold = result["folds"][0]
        for key in ("train_history", "test_history"):
            assert [row["model_values"] for row in fold[key]] == [[1.0, 2.0], [1.0, 8.0], [2.0, 2.0], [2.0, 8.0]]
        assert fold["training_labels"] == expected
        # Only the named feature can make pure leaves. Dropping it makes every score 1/2.
        assert fold["threshold"] == 0.5
        assert [row["event_id"] for row in result["predictions"]] == [event["event_id"] for event in source.events[4:]]
        assert [row["score"] for row in result["predictions"]] == expected
        assert [row["selected"] for row in result["predictions"]] == [bool(value) for value in expected]
        assert result["selected"] == {"count": 2, "mean_mark_pnl": pytest.approx(1.2)}
    finally:
        source.conn.close()


def test_break_even_training_trade_is_negative_and_keeps_fold_probabilities(tmp_path):
    source = build_gate_source(tmp_path, break_even=True)
    try:
        spec, registered = _register(source, replace(build_gate_spec(), folds=("2023",)))
        result = gate.gate_result(source.repository, registered, spec)
        fold = result["folds"][0]
        assert fold["training_cashflows"][1] == {"event_id": source.events[1]["event_id"],
            "entry_cost": pytest.approx(3.8), "exit_value": pytest.approx(3.8), "pnl": 0.0}
        assert fold["training_labels"] == [0, 0, 1, 0]
        # The first two histories form a zero-success leaf; the other leaf has 1/2 successes.
        assert fold["threshold"] == 0.25
        assert [row["score"] for row in result["predictions"]] == [0.5, 0.5]
        assert [row["selected"] for row in result["predictions"]] == [True, True]
    finally:
        source.conn.close()


def test_two_real_cli_arms_consume_distinct_fill_record_and_replay_once(source, capsys, monkeypatch):
    ledger, results, outputs = source.root / "ledger.csv", [], []
    for spec in (build_gate_spec(), build_gate_spec(1.0, "control")):
        _invoke(source, capsys, "register", spec)
        output = _invoke(source, capsys, "run", spec, "--ledger", ledger,
                         required_tables={"earnings_events", "daily_market", "option_chains"})
        outputs.append(output)
        result = _result(output)
        results.append(result)
        for fold in result["folds"]:
            assert fold["train_inputs"]["fill_alpha"] == fold["test_inputs"]["fill_alpha"] == spec.economic_params["fill"]
            assert fold["test_inputs"]["snapshot_id"] == source.snapshot.snapshot_id
        assert "exploratory" in Path(output["report"]).read_text().lower()
    assert results[0]["variant_id"] != results[1]["variant_id"]
    assert [r["all_candidates"]["mean_mark_pnl"] for r in results] == pytest.approx([1.2, 2.8])
    assert results[0]["folds"][0]["training_labels"] == [0, 0, 1, 0]
    assert results[1]["folds"][0]["training_labels"] == [0, 1, 1, 0]
    rows = list(csv.DictReader(ledger.open()))
    assert len(rows) == 2 and {row["stage"] for row in rows} == {"ran"}
    assert {row["spec_hash"] for row in rows} == {result["variant_id"] for result in results}
    export_folder = next((source.root / "experiments").glob("EXP-101_*"))
    assert outputs[1]["record_export"]["status"] == "written"
    run_dirs = list((export_folder / "results" / "runs").iterdir())
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "request.json").is_file()
    assert (run_dirs[0] / "outcome.json").is_file()
    before = ledger.read_bytes()
    first_output = outputs[0]
    assert Path(outputs[0]["report"]).read_text().strip() in (export_folder / "REPORT.md").read_text()
    assert set(first_output["record_export"]) == {"status", "trimmed"}
    assert "dir" not in first_output["record_export"]
    assert json.loads((export_folder / "results" / "request.json").read_text()) == {
        "action": "run",
        "as_of_month": MONTH,
        "scope": "shadow",
        "event_ids": None,
        "purpose": "sweep",
        "no_ledger": False,
    }
    monkeypatch.setattr(gate, "gate_result", _forbid)
    for index, (spec, output) in enumerate(zip((build_gate_spec(), build_gate_spec(1.0, "control")), outputs, strict=True)):
        replayed = _invoke(source, capsys, "run", spec, "--ledger", ledger)
        assert replayed["record_export"]["status"] == "unchanged"
        assert output["record_export"]["status"] == "written"
        assert replayed["recording_mode"] == "recorded"
        assert replayed["variant_id"] == results[index]["variant_id"]
        assert {key: value for key, value in replayed.items() if key != "record_export"} == {
            key: value for key, value in output.items() if key != "record_export"}
    assert ledger.read_bytes() == before


def test_smoke_never_accesses_ledger_and_changed_identity_refuses_before_evaluation(source, capsys, monkeypatch):
    _invoke(source, capsys, "register")
    poison = source.root / "never-open" / "ledger.csv"
    monkeypatch.setattr(native_outcomes, "ledger_append", _forbid)
    output = _invoke(source, capsys, "run", None, "--no-ledger", "--ledger", poison)
    assert output["recording_mode"] == "smoke"
    assert "no-ledger smoke; not recorded" in Path(output["report"]).read_text()
    assert not poison.parent.exists() and not list(source.root.rglob("*.csv"))
    monkeypatch.setattr(gate, "gate_result", _forbid)
    changed = _invoke(source, capsys, "run", build_gate_spec(1.0), "--no-ledger", status=2, required_tables=set())
    assert changed["refused"] == "EXPERIMENT_IDENTITY_CONFLICT"


@pytest.mark.parametrize("changes", [
    {"seed": 2}, {"folds": ()}, {"folds": ("2024", "2023")}, {"folds": ("2023", "2023")},
    {"folds": ("x",)}, {"runner": "native-gate-replay"}, {"economic_params": {"fill": 0.0}},
    {"economic_params": {"fill": 0.0, "strategy": "STR-THRU", "exit": {"kind": "fixed_day", "trading_days": 1}}},
])
def test_unused_or_invalid_declarations_refuse_before_registration(source, capsys, monkeypatch, changes):
    monkeypatch.setattr(Repository, "resolve_pinned", _forbid)
    document = asdict(build_gate_spec())
    document.update(changes)
    spec = ExperimentSpec(**document)
    response = _invoke(source, capsys, "register", spec, status=2, required_tables=set())
    assert response["refused"] == "INVALID_EXPERIMENT_SPEC"
    assert source.conn.execute("SELECT COUNT(*) FROM experiment_runs").fetchone()[0] == 0


@pytest.mark.parametrize("kind", ["random", "rolling", "unknown", "final"])
def test_explicit_holdout_refuses_before_any_feature_quote_metric_or_report(source, capsys, monkeypatch, kind):
    real_scan, scanned = Repository.scan, []
    def scan(repository, query, *, table_name):
        assert table_name == "earnings_events"
        scanned.append(table_name)
        yield from real_scan(repository, query, table_name=table_name)
    monkeypatch.setattr(Repository, "scan", scan)
    monkeypatch.setattr(gate, "gate_result", _forbid)
    extra = ("--purpose", "final") if kind == "final" else ("--event-id",
        "absent" if kind == "unknown" else source.protected[0 if kind == "random" else 1]["event_id"])
    response = _invoke(source, capsys, "register", None, *extra, status=2,
                       required_tables=set() if kind == "final" else {"earnings_events"})
    assert response["refused"] == "HOLDOUT_ACCESS_DENIED"
    assert not list(source.root.rglob("REPORT.md"))
    assert source.conn.execute("SELECT COUNT(*) FROM experiment_runs").fetchone()[0] == 0
    assert bool(scanned) == (kind != "final")


def test_fit_freezes_scores_before_test_quotes_and_protected_history_is_never_opened(source, monkeypatch):
    spec, registered = _register(source)
    protected_days = {event["event_date"].date().isoformat() for event in source.protected}
    blocked = {record.object_ref.object_id for record in source.repository.fragment_records(source.snapshot, "computed_moves")
               if record.time_min in protected_days}
    real_verify = objects.verify_object_path
    def verify(store, ref):
        assert ref.object_id not in blocked
        return real_verify(store, ref)
    monkeypatch.setattr(objects, "verify_object_path", verify)
    original_fit, original_price, stages = gate.fit_walk_forward_fold, gate.price_variant, []
    def fit(estimator, train_x, train_y, test_x, rule):
        assert stages[-1][0] == "train"
        result = original_fit(estimator, train_x, train_y, test_x, rule)
        stages.append(("fit", result.test_scores.copy()))
        return result
    def price(repository, snapshot, supplied, **kwargs):
        ids = kwargs["event_ids"]
        # Each fold starts with a fresh train replay, followed by fit, then test replay.
        stage = "test" if stages and stages[-1][0] == "fit" else "train"
        fold_index = len(stages) // 3
        train_count = (4, 6)[fold_index]
        expected = source.events[:train_count] if stage == "train" else source.events[train_count:train_count + 2]
        assert ids == [event["event_id"] for event in expected]
        assert snapshot == source.snapshot and supplied is spec
        stages.append((stage, ids))
        return original_price(repository, snapshot, supplied, **kwargs)
    monkeypatch.setattr(gate, "fit_walk_forward_fold", fit)
    monkeypatch.setattr(gate, "price_variant", price)
    gate.gate_result(source.repository, registered, spec)
    assert [stage for stage, _ in stages] == ["train", "fit", "test"] * 2


def test_late_history_label_is_omitted_not_backdated(tmp_path):
    source = build_gate_source(tmp_path, delayed=(3, "2023-02-02"))
    try:
        spec, registered = _register(source)
        result = gate.gate_result(source.repository, registered, spec)
        first = result["folds"][0]["test_history"][0]
        assert first["feature_ceiling"] == "2023-02-01"
        assert first["model_values"] == [3.0, 4.0]
        assert first["history"]["omitted_history"] == [{"event_id": source.events[3]["event_id"], "reason": "unavailable"}]
    finally:
        source.conn.close()


def test_missing_evaluation_quotes_records_one_failed_attempt_and_replay_does_not_refit(tmp_path, capsys, monkeypatch):
    source = build_gate_source(tmp_path, missing_quotes=True)
    try:
        ledger = source.root / "ledger.csv"
        _invoke(source, capsys, "register")
        response = _invoke(source, capsys, "run", None, "--ledger", ledger, status=2)
        assert response["refused"] == "EXPERIMENT_VARIANT_FAILED"
        rows = list(csv.DictReader(ledger.open()))
        assert len(rows) == 1 and rows[0]["stage"] == "failed"
        evidence = json.loads(source.conn.execute("SELECT evidence_json FROM experiment_runs").fetchone()[0])
        outcome = json.loads(source.store.read_verified(from_document(ArtifactRef, evidence["native_outcome"])))
        assert outcome["attempted_variants"] == 1 and outcome["report_ref"] is None
        assert not list(source.root.rglob("REPORT.md"))
        monkeypatch.setattr(gate, "gate_result", _forbid)
        before = ledger.read_bytes()
        replayed = _invoke(source, capsys, "run", None, "--ledger", ledger, status=2)
        assert replayed["refused"] == "EXPERIMENT_VARIANT_FAILED"
        assert replayed["replayed"] is True
        assert replayed["record_export"]["status"] == "written"
        assert ledger.read_bytes() == before
    finally:
        source.conn.close()


def test_january_event_with_december_entry_is_not_a_closed_training_label(tmp_path):
    source = build_gate_source(tmp_path, runup=True)
    try:
        spec = replace(build_gate_spec(), economic_params={"fill": 0.0, "strategy": "STR-RUNUP"})
        spec, registered = _register(source, spec)
        result = gate.gate_result(source.repository, registered, spec)
        first, second = result["folds"]
        boundary = source.events[3]["event_id"]
        assert boundary not in first["train_inputs"]["event_ids"]
        assert boundary not in first["test_inputs"]["event_ids"]
        assert boundary in second["train_inputs"]["event_ids"]
        row = next(row for row in second["train_history"] if row["event_id"] == boundary)
        assert row["gate_session"] == "2022-12-14"
        assert row["feature_ceiling"] == "2022-12-13"
        assert first["training_exits_before"] == "2023-01-01"
        exact = source.events[5]["event_id"]
        assert exact in first["test_inputs"]["event_ids"]
        assert exact not in second["train_inputs"]["event_ids"]
        assert second["training_exits_before"] == "2024-01-01"
    finally:
        source.conn.close()


def test_changing_evaluation_quotes_cannot_change_that_folds_fitted_scores(tmp_path):
    sources = []
    try:
        for index in range(2):
            path = tmp_path / str(index)
            path.mkdir()
            sources.append(build_gate_source(path, test_quote_shift=-1.5 * index))
        results = [gate.gate_result(source.repository, registered, spec)
                   for source in sources for spec, registered in [_register(source)]]
        first = [[row for row in result["predictions"] if row["fold"] == "2023"] for result in results]
        assert [row["score"] for row in first[0]] == [row["score"] for row in first[1]]
        assert [row["selected"] for row in first[0]] == [row["selected"] for row in first[1]]
        assert [row["pnl"] for row in first[1]] == pytest.approx([row["pnl"] - 1.5 for row in first[0]])
        assert all(row["pnl"] > 0 for row in first[0]) and all(row["pnl"] < 0 for row in first[1])
        assert results[0]["folds"][0]["threshold"] == results[1]["folds"][0]["threshold"]
        assert results[0]["folds"][0]["training_labels"] == results[1]["folds"][0]["training_labels"]
    finally:
        for source in sources:
            source.conn.close()


def test_missing_prior_calendar_session_refuses_before_features_or_quotes(tmp_path, monkeypatch):
    source = build_gate_source(tmp_path, calendar_start="2021-02-02")
    try:
        spec, registered = _register(source)
        monkeypatch.setattr(gate, "load_event_history", _forbid)
        monkeypatch.setattr(gate, "price_variant", _forbid)
        with pytest.raises(OpsError, match="EXPERIMENT_VARIANT_FAILED"):
            gate.gate_result(source.repository, registered, spec)
    finally:
        source.conn.close()


def test_current_source_admission_precedes_saved_replay(source, capsys, monkeypatch):
    _invoke(source, capsys, "register")
    output = _invoke(source, capsys, "run", None, "--no-ledger")
    report = Path(output["report"]).read_bytes()
    original = native_registration.source_closure
    def changed(root, entries):
        return {**original(root, entries), gate.RUNNER: "sha256:" + "0" * 64}
    monkeypatch.setattr(native_registration, "source_closure", changed)
    monkeypatch.setattr(gate, "gate_result", _forbid)
    monkeypatch.setattr(native_outcomes, "replay_native_outcome", _forbid)
    response = _invoke(source, capsys, "run", None, "--no-ledger", status=2, required_tables=set())
    assert response["refused"] == "EXPERIMENT_IDENTITY_CONFLICT"
    assert Path(output["report"]).read_bytes() == report


def test_report_rejects_scalar_type_provenance_drift(source):
    spec, registered = _register(source)
    result = gate.gate_result(source.repository, registered, spec)
    result["provenance"]["execution_plan"]["economic_params"]["fill"] = False
    with pytest.raises(OpsError, match="EXPERIMENT_IDENTITY_CONFLICT"):
        gate.render_gate_report(registered, result, spec=spec, no_ledger=True)


@pytest.mark.parametrize("field", ["variant_id", "run_id", "spec_hash"])
def test_report_rejects_each_identity_mismatch_with_valid_control(source, field):
    spec, registered = _register(source)
    result = gate.gate_result(source.repository, registered, spec)
    assert result["variant_id"] == registered.variant_id
    assert result["run_id"] == registered.run_id
    assert spec.spec_hash == registered.document["spec_hash"]
    assert result["provenance"] == registered.document
    valid = gate.render_gate_report(registered, result, spec=spec, no_ledger=True)
    assert json.loads(valid.split("```json\n", 1)[1].split("\n```", 1)[0]) == result
    changed_result, changed_spec = dict(result), spec
    if field == "spec_hash":
        changed_spec = replace(spec, hypothesis="Different synthetic hypothesis")
        assert changed_spec.spec_hash != registered.document["spec_hash"]
    else:
        changed_result[field] = "different-" + result[field]
    with pytest.raises(OpsError) as failure:
        gate.render_gate_report(registered, changed_result, spec=changed_spec, no_ledger=True)
    assert failure.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert gate.render_gate_report(registered, result, spec=spec, no_ledger=True) == valid


@pytest.mark.parametrize("alpha,selected_mean,all_mean", [(0.0, 1.2, 3.2), (1.0, 2.8, 4.8)])
def test_real_gate_rejects_low_score_candidate_and_counts_only_selected_cashflows(
        tmp_path, alpha, selected_mean, all_mean):
    source = build_gate_source(tmp_path, rejected=True)
    try:
        spec, registered = _register(source, build_gate_spec(alpha))
        result = gate.gate_result(source.repository, registered, spec)
        fold = result["folds"][0]
        predicted = [row for row in result["predictions"] if row["fold"] == "2023"]
        assert {row["event_id"] for row in predicted if row["selected"]} == {
            event["event_id"] for event in source.events[4:6]}
        rejected = next(row for row in predicted if row["event_id"] == source.events[-1]["event_id"])
        assert rejected["score"] == 0.0 < fold["threshold"]
        assert rejected["pnl"] == pytest.approx(selected_mean + 6.0)
        assert fold["selected"] == {"count": 2, "mean_mark_pnl": pytest.approx(selected_mean)}
        assert fold["all_candidates"] == {"count": 3, "mean_mark_pnl": pytest.approx(all_mean)}
        # Four ordinary selected trades and one rejected trade worth six more.
        assert result["selected"] == {"count": 4, "mean_mark_pnl": pytest.approx(selected_mean)}
        assert result["all_candidates"] == {"count": 5, "mean_mark_pnl": pytest.approx(selected_mean + 6 / 5)}
        # Calculate from the fixture's two leg quotes, independently of replay/report helpers.
        entry = 3.8 - 0.8 * alpha
        exits = [0.4, 3.4, 5.0, 0.4, 5.0, 5.0, 5.0, 5.0, 11.0]
        cashflows = {event["event_id"]: {"event_id": event["event_id"],
            "entry_cost": pytest.approx(entry), "exit_value": pytest.approx(exit_bid + 0.8 * alpha),
            "pnl": pytest.approx(exit_bid + 0.8 * alpha - entry)}
            for event, exit_bid in zip(source.events, exits, strict=True)}
        for fold, indexes in zip(result["folds"], ([0, 1, 2, 3], [8, 0, 1, 2, 3, 4, 5]), strict=True):
            assert fold["training_cashflows"] == [cashflows[source.events[i]["event_id"]] for i in indexes]
        for row in result["predictions"]:
            assert {key: row[key] for key in ("event_id", "entry_cost", "exit_value", "pnl")} == cashflows[row["event_id"]]
    finally:
        source.conn.close()


def test_real_gate_empty_selected_group_has_no_mean(tmp_path):
    source = build_gate_source(tmp_path, rejected=True)
    try:
        spec = replace(build_gate_spec(), folds=("2023",))
        # Train on four earlier SYNTH events, evaluate only the unseen NEW ticker.
        spec, registered = _register(source, spec,
            event_ids=[event["event_id"] for event in source.events[:4] + source.events[-1:]])
        result = gate.gate_result(source.repository, registered, spec)
        assert result["predictions"] == [{"event_id": source.events[-1]["event_id"], "fold": "2023",
            "score": 0.0, "selected": False, "entry_cost": pytest.approx(3.8),
            "exit_value": pytest.approx(11.0), "pnl": pytest.approx(7.2),
            "population_use": "post-release selection"}]
        for summary in (result, result["folds"][0]):
            assert summary["selected"] == {"count": 0, "mean_mark_pnl": None}
            assert summary["all_candidates"] == {"count": 1, "mean_mark_pnl": pytest.approx(7.2)}
    finally:
        source.conn.close()


def test_early_january_training_exits_must_precede_prior_year_feature_ceiling(tmp_path):
    source = build_gate_source(tmp_path, early_january=True)
    try:
        spec, registered = _register(source, replace(build_gate_spec(), folds=("2023",)))
        rows, _ = gate._metadata(source.repository, source.snapshot, registered.document, "STR-THRU")
        planned = rows.set_index("event_id")
        at_ceiling, after_ceiling, evaluation = [event["event_id"] for event in source.events[3:6]]
        assert planned.loc[at_ceiling, "exit_date"] == pd.Timestamp("2022-12-30")
        assert planned.loc[after_ceiling, "exit_date"] == pd.Timestamp("2023-01-02")
        assert planned.loc[evaluation, "gate_session"] == pd.Timestamp("2023-01-02")
        assert planned.loc[evaluation, "feature_ceiling"] == pd.Timestamp("2022-12-30")
        result = gate.gate_result(source.repository, registered, spec)
        fold = result["folds"][0]
        assert fold["training_exits_before"] == "2022-12-30"
        assert fold["train_inputs"]["event_ids"] == [event["event_id"] for event in source.events[:3]]
        assert fold["training_exits"] == ["2021-02-03", "2021-05-05", "2022-02-03"]
        assert fold["test_inputs"]["event_ids"] == [evaluation]
    finally:
        source.conn.close()


@pytest.mark.parametrize("kind,message", [
    ("incomplete_plan", "gate population needs complete calendar plans"),
    ("empty_evaluation", "gate fold has no evaluation events"),
    ("empty_training", "gate fold has no closed training events"),
])
def test_plan_and_fold_failures_record_one_failed_attempt_without_partial_report(
        tmp_path, capsys, monkeypatch, kind, message):
    source = build_gate_source(tmp_path, calendar_start="2021-03-01" if kind == "incomplete_plan" else "2020-12-01")
    try:
        # The second empty evaluation fold fails after the first fold has really completed.
        years = ("2023", "2025") if kind == "empty_evaluation" else ("2021", "2023") if kind == "empty_training" else ("2023", "2024")
        spec, ledger = replace(build_gate_spec(), folds=years), source.root / "ledger.csv"
        _invoke(source, capsys, "register", spec)
        real_result, failures = gate.gate_result, []
        def result(*args, **kwargs):
            with pytest.raises(OpsError) as failure:
                real_result(*args, **kwargs)
            assert failure.value.code == "EXPERIMENT_VARIANT_FAILED"
            assert failure.value.problem.message == message
            failures.append(failure.value.code)
            raise failure.value
        monkeypatch.setattr(gate, "gate_result", result)
        response = _invoke(source, capsys, "run", spec, "--ledger", ledger, status=2)
        assert response["refused"] == "EXPERIMENT_VARIANT_FAILED"
        assert failures == ["EXPERIMENT_VARIANT_FAILED"]
        rows = list(csv.DictReader(ledger.open()))
        assert len(rows) == 1 and rows[0]["stage"] == "failed"
        evidence = json.loads(source.conn.execute("SELECT evidence_json FROM experiment_runs").fetchone()[0])
        outcome = json.loads(source.store.read_verified(from_document(ArtifactRef, evidence["native_outcome"])))
        assert rows[0]["spec_hash"] == outcome["variant_id"] == evidence["variant_id"]
        assert outcome["failure_code"] == "EXPERIMENT_VARIANT_FAILED"
        assert outcome["attempted_variants"] == 1 and outcome["report_ref"] is None
        assert not list(source.root.rglob("REPORT.md"))
        before = ledger.read_bytes()
        monkeypatch.setattr(gate, "gate_result", _forbid)
        replayed = _invoke(source, capsys, "run", spec, "--ledger", ledger, status=2)
        assert replayed["refused"] == "EXPERIMENT_VARIANT_FAILED"
        assert replayed["replayed"] is True
        assert replayed["record_export"]["status"] == "written"
        assert ledger.read_bytes() == before
    finally:
        source.conn.close()


def test_record_export_failure_refuses_run_and_second_run_succeeds(source, capsys, monkeypatch):
    _invoke(source, capsys, "register")
    def boom(*args, **kwargs):
        raise fail("RESOURCE_UNAVAILABLE", "x")
    monkeypatch.setattr(cli.record_export, "export_native_run", boom)
    response = _invoke(source, capsys, "run", None, "--no-ledger", status=2)
    assert response == {"refused": "record_export_failed", "cause": "RESOURCE_UNAVAILABLE"}
    monkeypatch.setattr(cli.record_export, "export_native_run", export_native_run)
    output = _invoke(source, capsys, "run", None, "--no-ledger")
    assert output["record_export"]["status"] == "written"


def test_run_without_exp_number_refuses_and_creates_no_experiments_dir(source, capsys):
    _invoke(source, capsys, "register")
    response = _invoke(source, capsys, "run", None, "--no-ledger", status=2, exp_number=None)
    assert response["refused"] == "INVALID_EXPERIMENT_SPEC"
    assert not (source.root / "experiments").exists()


def test_run_no_record_export_skips_and_warns_no_exp_number(source, capsys):
    _invoke(source, capsys, "register")
    output = _invoke(source, capsys, "run", None, "--no-ledger", "--no-record-export",
                     exp_number=None, required_tables={"earnings_events", "daily_market", "option_chains"})
    assert output["record_export"] == "skipped"
    assert not (source.root / "experiments").exists()


def test_run_no_record_export_still_validates_exp_number(source, capsys):
    _invoke(source, capsys, "register")
    response = _invoke(source, capsys, "run", None, "--no-ledger", "--no-record-export",
                       status=2, exp_number="auto")
    assert response["refused"] == "INVALID_EXPERIMENT_SPEC"
    assert not (source.root / "experiments").exists()


def test_registered_holdout_request_records_refusal_without_attempt_or_partial_report(source, capsys, monkeypatch):
    _invoke(source, capsys, "register")
    ledger = source.root / "ledger.csv"
    monkeypatch.setattr(gate, "gate_result", _forbid)
    real_scan = Repository.scan
    def scan(repository, query, *, table_name):
        assert table_name == "earnings_events"
        yield from real_scan(repository, query, table_name=table_name)
    monkeypatch.setattr(Repository, "scan", scan)
    response = _invoke(source, capsys, "run", None, "--ledger", ledger,
        "--event-id", source.protected[0]["event_id"], status=2)
    assert response["refused"] == "HOLDOUT_ACCESS_DENIED"
    rows = list(csv.DictReader(ledger.open()))
    assert len(rows) == 1 and rows[0]["stage"] == "refused"
    evidence = json.loads(source.conn.execute("SELECT evidence_json FROM experiment_runs").fetchone()[0])
    outcome = json.loads(source.store.read_verified(from_document(ArtifactRef, evidence["native_outcome"])))
    assert outcome["attempted_variants"] == 0 and outcome["report_ref"] is None
    assert not list(source.root.rglob("REPORT.md"))
