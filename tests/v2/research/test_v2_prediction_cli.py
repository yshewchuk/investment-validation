"""Native CLI over real synthetic pinned inputs, registration, fits and outcomes."""
import json
import math
import os
import sqlite3
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pyarrow as pa
import pytest
from sklearn.dummy import DummyClassifier

from engine.v2.contracts import ArtifactRef
from engine.v2.data.errors import fail as fail_data
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactStore, from_document
from engine.v2.ops.errors import fail
from experiments import native_outcomes as outcomes
from experiments import native_registration as native
from experiments import prediction_report as prediction
from experiments import v2_prediction as cli
from tests.v2.research.test_native_outcomes import _clock, _rows
from tests.v2.research.test_native_registration import (
    _advance_head,
    _catalog,
    _new_snapshot,
    _objects,
)
from tests.v2.research.test_prediction_inputs import MONTH, _commit
from tests.v2.research.test_prediction_report import _dataset, _spec


@dataclass
class Source:
    conn: object
    store: ArtifactStore
    root: Path
    events: list
    snapshot: object
    spec_path: Path


def _source(tmp_path, *, spec=None, single_class=False):
    events, moves = _dataset()
    if single_class:
        for move in moves[:3]:
            move["realized_move_pct"] = 1
    conn, _, snapshot = _commit(tmp_path, events, moves)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(asdict(spec or _spec())))
    return Source(conn, ArtifactStore(tmp_path / "store"), tmp_path, events,
                  snapshot, spec_path)


@pytest.fixture
def source(tmp_path):
    value = _source(tmp_path)
    yield value
    value.conn.close()


def _args(source, action, *extra):
    return [action, "--spec", str(source.spec_path), "--catalog", str(source.root / "catalog.sqlite"),
            "--store-root", str(source.store.root),
            "--as-of-month", MONTH, *map(str, extra)]


def _invoke(source, capsys, action, *extra, status=0):
    assert cli.main(_args(source, action, *extra)) == status
    captured = capsys.readouterr()
    assert not captured.err
    assert "Traceback" not in captured.out
    return json.loads(captured.out)


def _evidence(source):
    row = source.conn.execute("SELECT evidence_json FROM experiment_runs").fetchone()
    return json.loads(row[0]) if row else {}


def _document(source, ref):
    return json.loads(source.store.read_verified(from_document(ArtifactRef, ref)))


def _assert_runtime_sources(source):
    root = Path(__file__).resolve().parents[3]
    registered = _document(source, _evidence(source)["native_registration"])
    assert registered["source_closure"] == native.source_closure(root, [prediction.RUNNER])
    assert {"experiments/v2_prediction.py", "experiments/prediction_report.py"} <= set(
        registered["source_closure"])


def _changed_source_fingerprint(monkeypatch):
    real_closure = native.source_closure

    def changed(root, entries):
        assert root == Path(__file__).resolve().parents[3]
        assert entries == [prediction.RUNNER]
        return {**real_closure(root, entries), prediction.RUNNER: "sha256:" + "0" * 64}

    monkeypatch.setattr(native, "source_closure", changed)


def _forbid(*_args, **_kwargs):
    raise AssertionError("forbidden target read, fit, report or ledger access")


def _metadata_only(monkeypatch):
    real_scan = Repository.scan

    def scan(repository, query, *, table_name):
        assert table_name == "earnings_events", "refusal must precede target reads"
        yield from real_scan(repository, query, table_name=table_name)

    monkeypatch.setattr(Repository, "scan", scan)
    monkeypatch.setattr(DummyClassifier, "fit", _forbid)
    monkeypatch.setattr(prediction, "render_prediction_report", _forbid)


def _changed_metadata(source, monkeypatch):
    real_scan = Repository.scan

    def scan(repository, query, *, table_name):
        assert table_name == "earnings_events", "metadata drift must precede target reads"
        for batch in real_scan(repository, query, table_name=table_name):
            frame = batch.to_pandas()
            frame.loc[frame.event_id == source.events[0]["event_id"], "date_conflict"] = True
            yield pa.RecordBatch.from_pandas(frame, schema=batch.schema, preserve_index=False)

    monkeypatch.setattr(Repository, "scan", scan)
    monkeypatch.setattr(DummyClassifier, "fit", _forbid)
    monkeypatch.setattr(prediction, "render_prediction_report", _forbid)


def _private_refusal(source, response, code):
    assert response["refused"] == code
    receipt = _document(source, response["refusal_ref"])
    assert receipt["failure_code"] == code
    assert receipt["schema_version"] == "native_experiment_refusal.v1.0"


def _preserved(source, catalog, objects):
    assert _catalog(source.conn) == catalog
    after = _objects(source.store)
    assert {key: after[key] for key in objects} == objects


def test_register_then_smoke_has_real_fold_metrics_and_never_touches_poison_ledger(
        source, monkeypatch, capsys):
    """Registration is metadata-only; smoke fits real priors and ignores its ledger."""
    with monkeypatch.context() as patch:
        _metadata_only(patch)
        registered = _invoke(source, capsys, "register")
    initial = _evidence(source)
    assert initial["planned_variants"] == 1
    assert not {"native_smoke", "native_outcome", "attempted_variants"} & initial.keys()
    assert registered["registered"] == initial["variant_id"]
    _assert_runtime_sources(source)
    assert not list(source.root.rglob("REPORT.md"))
    poison = source.root / "do-not-touch" / "ledger.csv"
    def protect(operation):
        def guarded(path, *args, **kwargs):
            assert "do-not-touch" not in path.parts, "smoke accessed its ledger"
            return operation(path, *args, **kwargs)
        return guarded

    real_fit, fits = DummyClassifier.fit, []

    def fit(estimator, features, labels, *args, **kwargs):
        assert estimator.strategy == "prior"
        fits.append(len(labels))
        return real_fit(estimator, features, labels, *args, **kwargs)

    with monkeypatch.context() as patch:
        for method in ("resolve", "open", "stat", "lstat", "mkdir", "touch"):
            patch.setattr(Path, method, protect(getattr(Path, method)))
        patch.setattr(outcomes, "ledger_append", _forbid)
        patch.setattr(DummyClassifier, "fit", fit)
        output = _invoke(source, capsys, "run", "--no-ledger", "--ledger", poison)
    assert fits == [3, 7]
    assert output["recording_mode"] == "smoke"
    report = Path(output["report"]).read_text()
    for text in ("Brier", "log-loss", "weighted ECE", "## OOF reliability bins",
                 "## OOF rank deciles", "Random holdout: excluded", "Rolling holdout: excluded",
                 "Attempted variants: 1", "no-ledger smoke; not recorded"):
        assert text in report
    result = json.loads(report.split("```json\n", 1)[1].split("\n```", 1)[0])
    labels, scores = [0, 1, 1, 1, 0, 1], [2 / 3] * 4 + [5 / 7] * 2
    assert result["metrics"]["brier"] == pytest.approx(
        sum((p - y) ** 2 for p, y in zip(scores, labels)) / 6)
    assert result["metrics"]["log_loss"] == pytest.approx(
        -sum(math.log(p if y else 1 - p) for p, y in zip(scores, labels)) / 6)
    assert result["metrics"]["ece"] == pytest.approx(
        (4 * abs(2 / 3 - 3 / 4) + 2 * abs(5 / 7 - 1 / 2)) / 6)
    assert len(result["metrics"]["reliability"]) == len(result["metrics"]["rank_deciles"]) == 10
    assert result["provenance"]["snapshot"]["snapshot_id"] == source.snapshot.snapshot_id
    evidence = _evidence(source)
    assert "native_smoke_receipt" in evidence and "native_outcome" not in evidence
    assert _document(source, evidence["native_smoke"])["ledger_destination"] is None
    assert not poison.parent.exists()
    assert not list(source.root.rglob("*.csv"))
    assert not list(source.root.rglob("*.append.lock"))


def test_recorded_run_after_smoke_replays_original_date_and_exactly_one_row(source, monkeypatch, capsys):
    _invoke(source, capsys, "register")
    smoke = _invoke(source, capsys, "run", "--no-ledger")
    ledger = source.root / "recorded" / "ledger.csv"
    _clock(monkeypatch, "2026-01-02")
    first = _invoke(source, capsys, "run", "--ledger", ledger)
    assert first["recording_mode"] == "recorded"
    assert first["report"] != smoke["report"]
    assert "recorded evaluation" in Path(first["report"]).read_text()
    before = _catalog(source.conn), _objects(source.store), ledger.read_bytes()
    _clock(monkeypatch, "2026-02-10")
    _metadata_only(monkeypatch)
    repeated = _invoke(source, capsys, "run", "--ledger", ledger)
    assert repeated == first
    _preserved(source, before[0], before[1])
    assert ledger.read_bytes() == before[2]
    assert len(_rows(ledger)) == 1
    assert _rows(ledger)[0]["date"] == "2026-01-02"
    assert _rows(ledger)[0]["stage"] == "ran"


def test_interrupted_recording_replays_saved_intent_without_refit(source, monkeypatch, capsys):
    """A failed append retains the original report/date; explicit replay finishes once."""
    _invoke(source, capsys, "register")
    ledger = source.root / "ledger.csv"
    _clock(monkeypatch, "2026-01-02")

    def interrupted(*args, **kwargs):
        raise OSError("synthetic interrupted append")

    with monkeypatch.context() as patch:
        patch.setattr(outcomes, "ledger_append", interrupted)
        response = _invoke(source, capsys, "run", "--ledger", ledger, status=2)
    _private_refusal(source, response, "RESOURCE_UNAVAILABLE")
    evidence = _evidence(source)
    assert "native_outcome" in evidence and "native_outcome_receipt" not in evidence
    original = _document(source, evidence["native_outcome"])
    assert original["ledger_row"]["date"] == "2026-01-02"
    assert not ledger.exists()
    assert not list(source.root.rglob("REPORT.md"))
    _clock(monkeypatch, "2026-02-10")
    _metadata_only(monkeypatch)
    result = _invoke(source, capsys, "run", "--ledger", ledger)
    assert result["recording_mode"] == "recorded"
    assert _rows(ledger) == [original["ledger_row"]]
    assert _document(source, _evidence(source)["native_outcome"]) == original
    assert Path(result["report"]).read_bytes() == source.store.read_verified(
        from_document(ArtifactRef, original["report_ref"]))


@pytest.mark.parametrize("kind", ["unregistered", "unsupported-runner", "missing-ledger", "missing-scope"])
def test_pre_admission_refusals_do_not_evaluate_or_record(source, monkeypatch, capsys, kind):
    if kind == "unsupported-runner":
        document = json.loads(source.spec_path.read_text())
        document["runner"] = "experiments/unsupported.py"
        source.spec_path.write_text(json.dumps(document))
    extra = [] if kind == "missing-ledger" else ["--no-ledger"]
    if kind == "missing-scope":
        _invoke(source, capsys, "register")
        extra += ["--scope", "does-not-exist"]
    _metadata_only(monkeypatch)
    before, objects = _catalog(source.conn), _objects(source.store)
    response = _invoke(source, capsys, "run", *extra, status=2)
    _private_refusal(source, response,
                     "SNAPSHOT_UNRESOLVED" if kind == "missing-scope" else "INVALID_EXPERIMENT_SPEC")
    _preserved(source, before, objects)
    assert not list(source.root.rglob("REPORT.md"))
    assert not list(source.root.rglob("*.csv"))


@pytest.mark.parametrize("kind", ["empty-train", "empty-test", "single-class"])
@pytest.mark.parametrize("smoke", [False, True])
def test_invalid_folds_publish_typed_r4_without_partial_report(tmp_path, monkeypatch, capsys, kind, smoke):
    folds = {"empty-train": ("2022",), "empty-test": ("2023", "2025"),
             "single-class": ("2023",)}[kind]
    source = _source(tmp_path, spec=_spec(folds=folds), single_class=kind == "single-class")
    try:
        _invoke(source, capsys, "register")
        monkeypatch.setattr(prediction, "render_prediction_report", _forbid)
        ledger = tmp_path / "ledger.csv"
        args = ["--ledger", ledger] + (["--no-ledger"] if smoke else [])
        response = _invoke(source, capsys, "run", *args, status=2)
        _private_refusal(source, response, "EXPERIMENT_VARIANT_FAILED")
        evidence = _evidence(source)
        outcome = _document(source, evidence["native_smoke" if smoke else "native_outcome"])
        assert outcome["failure_code"] == "EXPERIMENT_VARIANT_FAILED"
        assert outcome["ledger_row"]["stage"] == "failed" and outcome["attempted_variants"] == 1
        assert outcome["report_ref"] is None
        assert not list(tmp_path.rglob("REPORT.md"))
        if smoke:
            assert not ledger.exists()
        else:
            assert [row["stage"] for row in _rows(ledger)] == ["failed"]
    finally:
        source.conn.close()


@pytest.mark.parametrize("smoke", [False, True])
@pytest.mark.parametrize("code", ["FEATURE_LOOKAHEAD", "CONTRACT_MISMATCH"])
def test_injected_r3_and_r4_at_prediction_input_boundary_are_recorded_and_replayed(
        source, monkeypatch, capsys, smoke, code):
    """Inject shared R3/R4 input failures; the prior has no predictive feature path."""
    _invoke(source, capsys, "register")
    _metadata_only(monkeypatch)

    def input_failure(*_args, **_kwargs):
        raise (fail if code == "FEATURE_LOOKAHEAD" else fail_data)(
            code, "Synthetic prediction input boundary refusal")

    monkeypatch.setattr(prediction, "load_prediction_targets", input_failure)
    ledger = source.root / "recorded" / "ledger.csv"
    recording = ["--ledger", ledger] + (["--no-ledger"] if smoke else [])
    expected = "FEATURE_LOOKAHEAD" if code == "FEATURE_LOOKAHEAD" else "EXPERIMENT_VARIANT_FAILED"
    response = _invoke(source, capsys, "run", *recording, status=2)
    _private_refusal(source, response, expected)
    evidence = _evidence(source)
    key = "native_smoke" if smoke else "native_outcome"
    outcome = _document(source, evidence[key])
    assert outcome["failure_code"] == expected
    assert outcome["ledger_row"]["stage"] == ("refused" if code == "FEATURE_LOOKAHEAD" else "failed")
    assert outcome["attempted_variants"] == 1
    assert outcome["report_ref"] is None
    assert not list(source.root.rglob("REPORT.md"))
    if smoke:
        assert not ledger.parent.exists()
    else:
        assert _rows(ledger) == [outcome["ledger_row"]]
    catalog, objects = _catalog(source.conn), _objects(source.store)
    ledger_bytes = None if smoke else ledger.read_bytes()
    monkeypatch.setattr(prediction, "load_prediction_targets", _forbid)
    replayed = _invoke(source, capsys, "run", *recording, status=2)
    assert replayed == {"refused": expected, "receipt_ref": evidence[key + "_receipt"]}
    assert _catalog(source.conn) == catalog and _objects(source.store) == objects
    assert not list(source.root.rglob("REPORT.md"))
    if smoke:
        assert not ledger.parent.exists()
    else:
        assert ledger.read_bytes() == ledger_bytes


@pytest.mark.parametrize("smoke", [False, True])
def test_transient_prediction_input_failure_stays_private_and_can_retry(
        source, monkeypatch, capsys, smoke):
    _invoke(source, capsys, "register")
    catalog, objects, evidence = _catalog(source.conn), _objects(source.store), _evidence(source)
    ledger = source.root / "recorded" / "ledger.csv"
    recording = ["--ledger", ledger] + (["--no-ledger"] if smoke else [])

    def unavailable(*_args, **_kwargs):
        raise fail_data("RESOURCE_UNAVAILABLE", "Synthetic transient input failure")

    with monkeypatch.context() as patch:
        _metadata_only(patch)
        patch.setattr(prediction, "load_prediction_targets", unavailable)
        response = _invoke(source, capsys, "run", *recording, status=2)
    _private_refusal(source, response, "RESOURCE_UNAVAILABLE")
    _preserved(source, catalog, objects)
    assert _evidence(source) == evidence
    added = _objects(source.store).keys() - objects.keys()
    assert len(added) == 1
    assert all(json.loads(_objects(source.store)[key])["schema_version"] ==
               "native_experiment_refusal.v1.0" for key in added)
    assert not ledger.parent.exists()
    assert not list(source.root.rglob("REPORT.md"))
    completed = _invoke(source, capsys, "run", *recording)
    assert completed["recording_mode"] == ("smoke" if smoke else "recorded")
    assert Path(completed["report"]).is_file()
    if smoke:
        assert not ledger.parent.exists()
    else:
        assert [row["stage"] for row in _rows(ledger)] == ["ran"]


@pytest.mark.parametrize("kind", ["random", "rolling", "metadata-drift"])
@pytest.mark.parametrize("smoke", [False, True])
def test_r5_precedes_targets_fit_and_report_for_explicit_or_changed_metadata(
        source, monkeypatch, capsys, kind, smoke):
    requested = source.events[:10] if kind == "metadata-drift" else []
    selected = [part for event in requested for part in ("--event-id", event["event_id"])]
    _invoke(source, capsys, "register", *selected)
    if kind == "metadata-drift":
        _changed_metadata(source, monkeypatch)
    else:
        _metadata_only(monkeypatch)
        selected = ["--event-id", source.events[-2 if kind == "random" else -1]["event_id"]]
    ledger = source.root / "ledger.csv"
    args = ["--ledger", ledger, *selected] + (["--no-ledger"] if smoke else [])
    response = _invoke(source, capsys, "run", *args, status=2)
    _private_refusal(source, response, "HOLDOUT_ACCESS_DENIED")
    evidence = _evidence(source)
    outcome = _document(source, evidence["native_smoke" if smoke else "native_outcome"])
    assert outcome["failure_code"] == "HOLDOUT_ACCESS_DENIED"
    assert outcome["ledger_row"]["stage"] == "refused" and outcome["attempted_variants"] == 0
    assert outcome["report_ref"] is None
    assert not list(source.root.rglob("REPORT.md"))
    if smoke:
        assert not ledger.exists()
    else:
        assert [row["stage"] for row in _rows(ledger)] == ["refused"]


@pytest.mark.parametrize("kind", ["explicit-holdout", "metadata-drift"])
@pytest.mark.parametrize("smoke", [False, True])
def test_current_r5_blocks_saved_success_replay_without_changing_prior_bytes(
        source, monkeypatch, capsys, kind, smoke):
    selected = [part for event in source.events[:10] for part in ("--event-id", event["event_id"])]
    _invoke(source, capsys, "register", *selected)
    ledger = source.root / "ledger.csv"
    recording = ["--ledger", ledger] + (["--no-ledger"] if smoke else [])
    completed = _invoke(source, capsys, "run", *recording, *selected)
    report = Path(completed["report"])
    catalog, objects, report_bytes = _catalog(source.conn), _objects(source.store), report.read_bytes()
    ledger_bytes = None if smoke else ledger.read_bytes()
    if kind == "metadata-drift":
        _changed_metadata(source, monkeypatch)
    else:
        _metadata_only(monkeypatch)
        selected = ["--event-id", source.events[-1]["event_id"]]
    monkeypatch.setattr(outcomes, "replay_native_outcome", _forbid)
    response = _invoke(source, capsys, "run", *recording, *selected, status=2)
    _private_refusal(source, response, "HOLDOUT_ACCESS_DENIED")
    _preserved(source, catalog, objects)
    assert report.read_bytes() == report_bytes
    if smoke:
        assert not ledger.exists()
    else:
        assert ledger.read_bytes() == ledger_bytes
    added = _objects(source.store).keys() - objects.keys()
    assert added
    assert all(json.loads(_objects(source.store)[key])["schema_version"] ==
               "native_experiment_refusal.v1.0" for key in added)


@pytest.mark.parametrize("dimension", ["hypothesis", "spec", "code", "snapshot", "scope", "environment"])
@pytest.mark.parametrize("holdout", ["random", "rolling"])
@pytest.mark.parametrize("smoke", [False, True])
def test_changed_binding_precedes_holdout_and_cannot_consume_original_outcome(
        source, monkeypatch, capsys, dimension, holdout, smoke):
    """R6 stays private, then restored inputs can still publish their first success."""
    if dimension == "snapshot":
        # The shared builder commits a real, distinct 2024-only metadata snapshot.
        alternate = _new_snapshot(replace(source, events=[
            event for event in source.events if event["year"] == 2024]))
    elif dimension == "scope":
        source.conn.execute("INSERT INTO data_snapshot_heads SELECT 'same-snapshot', snapshot_id, 1, "
                            "updated_at, update_receipt_ref FROM data_snapshot_heads WHERE scope='shadow'")
    registered = _invoke(source, capsys, "register")
    original_evidence, original_catalog = _evidence(source), _catalog(source.conn)
    original_registration = _document(source, original_evidence["native_registration"])
    spec_bytes = source.spec_path.read_bytes()
    ledger = source.root / "recorded" / "ledger.csv"
    recording = ["--ledger", ledger] + (["--no-ledger"] if smoke else [])
    selected = ["--event-id", source.events[-2 if holdout == "random" else -1]["event_id"]]
    try:
        with monkeypatch.context() as patch:
            if dimension in {"hypothesis", "spec"}:
                document = json.loads(spec_bytes)
                if dimension == "hypothesis":
                    document["hypothesis"] = "A changed hypothesis under the same registered arm."
                else:
                    document["seed"] += 1
                source.spec_path.write_text(json.dumps(document))
            elif dimension == "code":
                _changed_source_fingerprint(patch)
            elif dimension == "snapshot":
                _advance_head(source, alternate)
            elif dimension == "scope":
                selected += ["--scope", "same-snapshot"]
            else:
                environment = native.environment_identity(1)
                environment["libraries"]["numpy"] = "changed-numerical-version"
                patch.setattr(native, "environment_identity", lambda _threads: environment)
            # A conflicting identity must be rejected even before metadata admission.
            patch.setattr(native, "load_population", _forbid)
            patch.setattr(Repository, "scan", _forbid)
            patch.setattr(DummyClassifier, "fit", _forbid)
            patch.setattr(prediction, "render_prediction_report", _forbid)
            patch.setattr(outcomes, "publish_native_outcome", _forbid)
            patch.setattr(outcomes, "replay_native_outcome", _forbid)
            patch.setattr(outcomes, "ledger_append", _forbid)
            catalog, objects = _catalog(source.conn), _objects(source.store)
            response = _invoke(source, capsys, "run", *recording, *selected, status=2)
            _private_refusal(source, response, "EXPERIMENT_IDENTITY_CONFLICT")
            _preserved(source, catalog, objects)
            assert _evidence(source) == original_evidence
            added = _objects(source.store).keys() - objects.keys()
            assert len(added) == 1
            assert all(json.loads(_objects(source.store)[key])["schema_version"] ==
                       "native_experiment_refusal.v1.0" for key in added)
            assert not ledger.parent.exists()
            assert not list(source.root.rglob("REPORT.md"))
            assert not list(source.root.rglob("*.csv"))
            assert not list(source.root.rglob("*.append.lock"))
    finally:
        source.spec_path.write_bytes(spec_bytes)
        if dimension == "snapshot":
            # Repinning the original bytes still advances the head generation.
            _advance_head(source, source.snapshot)
    restored_catalog = _catalog(source.conn)
    assert {key: value for key, value in restored_catalog.items() if key != "data_snapshot_heads"} == {
        key: value for key, value in original_catalog.items() if key != "data_snapshot_heads"}
    assert Repository(source.conn, source.store).resolve_pinned("shadow") == source.snapshot
    completed = _invoke(source, capsys, "run", *recording)
    assert completed["variant_id"] == registered["registered"]
    assert completed["recording_mode"] == ("smoke" if smoke else "recorded")
    assert Path(completed["report"]).is_file()
    evidence = _evidence(source)
    assert evidence["native_registration"] == original_evidence["native_registration"]
    assert _document(source, evidence["native_registration"]) == original_registration
    outcome = _document(source, evidence["native_smoke" if smoke else "native_outcome"])
    assert outcome["failure_code"] is None
    assert outcome["ledger_row"]["stage"] == "ran" and outcome["attempted_variants"] == 1
    if smoke:
        assert not ledger.parent.exists()
    else:
        assert [row["stage"] for row in _rows(ledger)] == ["ran"]


@pytest.mark.parametrize("kind", ["code", "spec", "registration-bytes", "catalog-evidence", "outcome-bytes", "destination"])
def test_r6_preserves_prior_report_catalog_and_ledger(source, monkeypatch, capsys, kind):
    _invoke(source, capsys, "register")
    ledger = source.root / "ledger.csv"
    completed = _invoke(source, capsys, "run", "--ledger", ledger)
    report = Path(completed["report"])
    target = ledger
    if kind == "code":
        _changed_source_fingerprint(monkeypatch)
    elif kind == "spec":
        spec = json.loads(source.spec_path.read_text())
        spec["seed"] += 1
        source.spec_path.write_text(json.dumps(spec))
    elif kind in {"registration-bytes", "outcome-bytes"}:
        key = "native_registration" if kind == "registration-bytes" else "native_outcome"
        ref = from_document(ArtifactRef, _evidence(source)[key])
        path = source.store.root / ref.storage_key
        path.chmod(0o644)
        path.write_bytes(b"!" * ref.byte_size)
    elif kind == "catalog-evidence":
        source.conn.execute("UPDATE experiment_runs SET evidence_json='null'")
    else:
        target = source.root / "different" / "ledger.csv"
    before, objects, csv_bytes, report_bytes = (
        _catalog(source.conn), _objects(source.store), ledger.read_bytes(), report.read_bytes())
    _metadata_only(monkeypatch)
    response = _invoke(source, capsys, "run", "--ledger", target, status=2)
    _private_refusal(source, response, "EXPERIMENT_IDENTITY_CONFLICT")
    _preserved(source, before, objects)
    assert ledger.read_bytes() == csv_bytes and report.read_bytes() == report_bytes
    assert not (source.root / "different").exists()


@pytest.mark.parametrize("fault", ["registration", "catalog", "refusal", "admitted-refusal", "export"])
def test_storage_faults_are_typed_without_raw_tracebacks(source, monkeypatch, capsys, fault):
    action, extra = "run", ["--no-ledger"]
    if fault in {"admitted-refusal", "export"}:
        _invoke(source, capsys, "register")
    if fault == "admitted-refusal":
        extra += ["--event-id", source.events[-1]["event_id"]]
    if fault in {"registration", "refusal", "admitted-refusal"}:
        real_publish = ArtifactStore.publish_bytes
        schema = {"registration": "native_experiment_registration.v1.0",
                  "refusal": "native_experiment_refusal.v1.0",
                  "admitted-refusal": outcomes.SCHEMA}[fault]

        def publish(store, data, *, schema_ref):
            if schema_ref == schema:
                raise OSError("SECRET synthetic storage failure")
            return real_publish(store, data, schema_ref=schema_ref)

        monkeypatch.setattr(ArtifactStore, "publish_bytes", publish)
        if fault == "registration":
            action = "register"
    elif fault == "catalog":
        def unavailable(*args, **kwargs):
            raise sqlite3.OperationalError("SECRET synthetic catalog failure")
        monkeypatch.setattr(cli, "connect", unavailable)
    else:
        def unavailable(*args, **kwargs):
            raise OSError("SECRET synthetic export failure")
        monkeypatch.setattr(outcomes, "export_native_report", unavailable)
    response = _invoke(source, capsys, action, *extra, status=2)
    assert response["refused"] == "RESOURCE_UNAVAILABLE"
    assert "SECRET" not in json.dumps(response)
    assert not list(source.root.rglob("REPORT.md"))
    assert not list(source.root.rglob("*.csv"))


def test_module_subprocess_register_and_smoke(source):
    """Exercise the actual module entrypoint, not just its Python main function."""
    for action in ("register", "run"):
        completed = subprocess.run([sys.executable, "-m", "experiments.v2_prediction",
            *_args(source, action, "--no-ledger")], cwd=Path(__file__).resolve().parents[3],
            env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                 "MKL_NUM_THREADS": "1"}, capture_output=True, text=True, timeout=60)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert not completed.stderr
        result = json.loads(completed.stdout)
    assert result["recording_mode"] == "smoke"
    _assert_runtime_sources(source)
    assert Path(result["report"]).is_file()
    assert not list(source.root.rglob("*.csv"))


def test_cli_rejects_alternative_fingerprint_root(source, capsys):
    """An unexecuted source tree cannot be substituted into registered provenance."""
    fake_root = source.root / "unexecuted-code"
    runner = fake_root / prediction.RUNNER
    runner.parent.mkdir(parents=True)
    runner.write_text("# Synthetic source that is never executed.\n")
    catalog, objects = _catalog(source.conn), _objects(source.store)
    with pytest.raises(SystemExit) as error:
        cli.main(_args(source, "register", "--code-root", fake_root))
    assert error.value.code == 2
    assert "unrecognized arguments: --code-root" in capsys.readouterr().err
    assert _catalog(source.conn) == catalog
    assert _objects(source.store) == objects


@pytest.mark.parametrize("seed", [-1, 2**32])
@pytest.mark.parametrize("action", ["register", "run"])
@pytest.mark.parametrize("smoke", [False, True])
def test_invalid_sklearn_seed_refuses_before_admission_or_reads(tmp_path, monkeypatch, capsys, seed, action, smoke):
    spec = _spec(seed=seed)
    source = _source(tmp_path, spec=spec)
    try:
        if action == "run":
            native.register_native(source.conn, source.store, Repository(source.conn, source.store), spec,
                code_root=Path(__file__).resolve().parents[3], as_of_month=MONTH)
        before, objects = _catalog(source.conn), _objects(source.store)
        monkeypatch.setattr(Repository, "scan", _forbid)
        monkeypatch.setattr(DummyClassifier, "fit", _forbid)
        monkeypatch.setattr(prediction, "render_prediction_report", _forbid)
        monkeypatch.setattr(outcomes, "ledger_append", _forbid)
        ledger = tmp_path / "must-not-exist" / "ledger.csv"
        args = ["--ledger", ledger] + (["--no-ledger"] if smoke else [])
        response = _invoke(source, capsys, action, *args, status=2)
        _private_refusal(source, response, "INVALID_EXPERIMENT_SPEC")
        _preserved(source, before, objects)
        assert not ledger.parent.exists()
        assert not list(source.root.rglob("REPORT.md"))
    finally:
        source.conn.close()


@pytest.mark.parametrize("seed", [0, 2**32 - 1])
def test_sklearn_seed_boundaries_complete_real_cli_smoke(tmp_path, capsys, seed):
    source = _source(tmp_path, spec=_spec(seed=seed))
    try:
        _invoke(source, capsys, "register")
        result = _invoke(source, capsys, "run", "--no-ledger")
        assert result["recording_mode"] == "smoke"
        assert Path(result["report"]).is_file()
    finally:
        source.conn.close()


@pytest.mark.parametrize("phase", ["fresh", "pending", "completed"])
@pytest.mark.parametrize("smoke", [False, True])
@pytest.mark.parametrize("obstruction", ["foreign", "symlink", "directory"])
def test_report_destination_conflict_precedes_cli_publication_effects(
        source, monkeypatch, capsys, phase, smoke, obstruction):
    _invoke(source, capsys, "register")
    registration = native.read_native_registration(source.conn, source.store, _spec())
    ledger = source.root / "recorded" / "ledger.csv"
    args = ["--ledger", ledger] + (["--no-ledger"] if smoke else [])
    if phase == "pending":
        real_reserve = outcomes._reserve

        def interrupt_completion(conn, registration, key, *args, **kwargs):
            if key.endswith("_receipt"):
                raise OSError("synthetic completion interruption")
            return real_reserve(conn, registration, key, *args, **kwargs)

        def interrupt_append(*args, **kwargs):
            raise OSError("synthetic append interruption")

        with monkeypatch.context() as patch:
            patch.setattr(outcomes, "_reserve", interrupt_completion)
            patch.setattr(outcomes, "ledger_append", interrupt_append)
            response = _invoke(source, capsys, "run", *args, status=2)
        _private_refusal(source, response, "RESOURCE_UNAVAILABLE")
    elif phase == "completed":
        _invoke(source, capsys, "run", *args)
    path = source.store.root / "native_reports" / registration.run_id / (
        "smoke" if smoke else "recorded") / "REPORT.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    if obstruction == "foreign":
        path.write_bytes(b"unrelated report")
    elif obstruction == "directory":
        path.mkdir()
    else:
        path.symlink_to(source.root / "missing-foreign-report")
    before, objects = _catalog(source.conn), _objects(source.store)
    ledger_before = ledger.read_bytes() if ledger.exists() else None
    with monkeypatch.context() as patch:
        if phase != "fresh":
            _metadata_only(patch)
        patch.setattr(outcomes, "ledger_append", _forbid)
        response = _invoke(source, capsys, "run", *args, status=2)
    _private_refusal(source, response, "EXPERIMENT_IDENTITY_CONFLICT")
    _preserved(source, before, objects)
    assert (ledger.read_bytes() if ledger.exists() else None) == ledger_before
    after = _objects(source.store)
    for key in set(after) - set(objects):
        assert json.loads(after[key])["schema_version"] == "native_experiment_refusal.v1.0"
    assert path.is_symlink() if obstruction == "symlink" else (
        path.is_dir() if obstruction == "directory" else path.read_bytes() == b"unrelated report")
    path.rmdir() if obstruction == "directory" else path.unlink()
    result = _invoke(source, capsys, "run", *args)
    assert Path(result["report"]) == path
    assert "# Native prediction report" in path.read_text()
    assert not ledger.exists() if smoke else len(_rows(ledger)) == 1
