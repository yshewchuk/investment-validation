"""Native admission binds synthetic canonical metadata before any outcome access."""
import ast
import builtins
import io
import json
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, dataclass, replace
from pathlib import Path
from threading import Barrier

import pytest

from checks.package_readmes import directive
from engine.v2.contracts import ArtifactRef
from engine.v2.data.errors import DataError
from engine.v2.data.repository import Repository
from engine.v2.foundation import (
    ArtifactStore,
    canonical_json,
    content_hash,
    from_document,
    to_document,
)
from engine.v2.foundation.experiment_holdouts import ExperimentHoldouts
from engine.v2.ops.catalog import connect
from engine.v2.ops.errors import OpsError
from engine.v2.ops.experiments import (
    experiment_spec_from_document,
    register_hypothesis,
    resolve_experiment_plan,
)
from experiments import native_registration as native
from tests.data_scan_support import (
    commit_tables,
    contract_for,
    contract_ref_for,
    publish_and_inspect,
)
from tests.ops_support import FakeClock
from tests.v2.research.test_prediction_inputs import MONTH, TrackingRepository, _commit, _event


def _spec(**changes):
    """Resolve a minimal synthetic single-arm specification with explicit overrides."""
    return experiment_spec_from_document({
        "experiment_id": "synthetic-native", "hypothesis": "synthetic preregistration",
        "primary_arm_id": "candidate", "arms": ["candidate"], "folds": ["2024"],
        "seed": 7, "economic_params": {"fill": 0.5}, "price_source": "option_chains",
        "runner": "experiment/run.py", **changes,
    })


class MetadataRepository(TrackingRepository):
    """Track the permitted one-time head pin and reject all outcome scans."""

    def __init__(self, conn, store):
        super().__init__(conn, store)
        self.pins = []

    def resolve_pinned(self, scope):
        self.pins.append(scope)
        return Repository.resolve_pinned(self, scope)

    def scan(self, query, *, table_name):
        assert table_name == "earnings_events", "preregistration touched outcomes"
        yield from super().scan(query, table_name=table_name)


@dataclass
class Source:
    conn: object
    store: ArtifactStore
    repository: MetadataRepository
    snapshot: object
    code_root: Path
    catalog_path: Path
    events: list


def _source(tmp_path, *, include_targets=True):
    """Commit real synthetic Parquet and a source-only runner with a local import."""
    events = [_event("2024-05-01"), _event("2024-05-02"),
              _event("2024-05-03", random=True), _event("2024-12-03")]
    conn, _, snapshot = _commit(tmp_path, events, include_targets=include_targets)
    store = ArtifactStore(tmp_path / "store")
    root = tmp_path / "code"
    package = root / "experiment"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# Synthetic local package.\n")
    (package / "helper.py").write_text("SYNTHETIC_CONSTANT = 7\n")
    (package / "run.py").write_text(
        "from experiment.helper import SYNTHETIC_CONSTANT\n"
        "raise AssertionError('registration must never execute the runner')\n")
    return Source(conn, store, MetadataRepository(conn, store), snapshot, root,
                  tmp_path / "catalog.sqlite", events)


@pytest.fixture
def source(tmp_path):
    """Keep one independently committed synthetic source alive for each test."""
    value = _source(tmp_path)
    yield value
    value.conn.close()


def _call(source, *, admit=False, spec=None, **kwargs):
    """Invoke either public entrypoint with the same explicit source context."""
    method = native.require_native_registration if admit else native.register_native
    return method(source.conn, source.store, source.repository, spec or _spec(),
                  **{"code_root": source.code_root, "as_of_month": MONTH, **kwargs})


def _catalog(conn):
    """Snapshot every catalog row so refusals cannot silently change other tables."""
    tables = [row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    return {table: sorted(tuple(row) for row in conn.execute(f'SELECT * FROM "{table}"'))
            for table in tables}


def _objects(store):
    """Capture durable object bytes without including transient staging files."""
    return {str(path.relative_to(store.root)): path.read_bytes()
            for path in (store.root / "objects").rglob("*") if path.is_file()}


def _evidence(source, run_id):
    """Read the native reference from the existing experiment-run evidence slot."""
    return json.loads(source.conn.execute(
        "SELECT evidence_json FROM experiment_runs WHERE run_id=?", (run_id,)).fetchone()[0])


def _ref(source, registration):
    """Decode the registered reference through the shared typed contract."""
    return from_document(ArtifactRef, _evidence(source, registration.run_id)["native_registration"])


def _new_snapshot(source, scope="alternate"):
    """Commit another exact snapshot with identical events and no target table."""
    contract = contract_for("earnings_events")
    records = [publish_and_inspect(source.store, contract, contract_ref_for(contract),
                                  sorted(source.events, key=lambda row: row["event_id"]), "2024")]
    return commit_tables(source.conn, FakeClock(), {"earnings_events": records},
                         {"earnings_events": contract}, scope=scope,
                         receipt_id=scope, attempt_id=scope, store=source.store)


def _advance_head(source, snapshot):
    """Advance only the current scope head, leaving all pinned snapshots intact."""
    source.conn.execute("UPDATE data_snapshot_heads SET snapshot_id=?, generation=generation+1 "
                        "WHERE scope='shadow'", (snapshot.snapshot_id,))


def _forbid(*_args, **_kwargs):
    """Fail immediately if a read-only path attempts a forbidden effect."""
    raise AssertionError("unexpected write or outcome execution")


def test_register_binds_complete_plan_and_only_eligible_metadata(source):
    """Bind the complete declared identity while excluding both holdout populations."""
    before = _catalog(source.conn)
    result = _call(source)
    document = result.document
    assert result.run_id.startswith("native_")
    assert result.binding_json == canonical_json(document).encode("utf-8")
    assert result.variant_id == content_hash(document)
    assert document["schema_version"] == "native_experiment_registration.v1.0"
    assert document["execution_plan"] == resolve_experiment_plan(_spec()).as_document()
    assert document["spec_hash"] == _spec().spec_hash
    assert document["snapshot"] == to_document(source.snapshot)
    assert document["scope"] == "shadow"
    assert document["event_ids"] == sorted(event["event_id"] for event in source.events[:2])
    assert document["holdouts"] == {
        "holdout_as_of_month": MONTH,
        "random_membership_version": ExperimentHoldouts.random_version,
        "rolling_membership_version": ExperimentHoldouts.rolling_version,
    }
    assert set(document["source_closure"]) == {
        "experiment/run.py", "experiment/helper.py", "experiment/__init__.py"}
    assert document["environment"] == native.environment_identity(1)
    assert source.repository.pins == ["shadow"]
    assert source.repository.scans
    assert all(query.snapshot_id == source.snapshot.snapshot_id
               for _, query in source.repository.scans)
    run = source.conn.execute("SELECT * FROM experiment_runs").fetchone()
    hypothesis = source.conn.execute("SELECT * FROM hypotheses").fetchone()
    assert run["run_id"] == hypothesis["run_id"] == result.run_id
    assert run["spec_hash"] == hypothesis["spec_hash"] == _spec().spec_hash
    assert run["input_hash"] == hypothesis["input_hash"] == result.variant_id
    assert run["mode"] == "primary"
    assert hypothesis["payload_hash"] == content_hash({"spec": _spec().spec_hash,
                                                        "input": result.variant_id})
    evidence = _evidence(source, result.run_id)
    assert evidence["variant_id"] == result.variant_id
    assert evidence["planned_variants"] == 1
    assert "variants_tried" not in evidence
    ref = _ref(source, result)
    assert ref.schema_ref == document["schema_version"]
    assert source.store.read_verified(ref) == result.binding_json
    assert source.store.verify(ref).stat().st_mode & 0o222 == 0
    after = _catalog(source.conn)
    assert {key: value for key, value in after.items() if key not in {"hypotheses", "experiment_runs"}} == {
        key: value for key, value in before.items() if key not in {"hypotheses", "experiment_runs"}}


def test_result_is_frozen_and_document_copies_cannot_change_identity(source):
    """Neither dataclass assignment nor nested document edits may alter admitted bytes."""
    result = _call(source)
    original = result.binding_json
    changed = result.document
    changed["execution_plan"]["economic_params"]["fill"] = 0
    changed["event_ids"].clear()
    assert result.binding_json == original
    assert result.document != changed
    with pytest.raises(FrozenInstanceError):
        result.run_id = "changed"
    with pytest.raises(FrozenInstanceError):
        result.binding_json = b"{}"
    assert _call(source, admit=True) == result


def test_repeated_registration_and_admission_are_read_only(source, monkeypatch):
    """Canonical population order permits identical retries with all writes disabled."""
    result = _call(source, event_ids=list(reversed([e["event_id"] for e in source.events[:2]])))
    before, objects, changes = _catalog(source.conn), _objects(source.store), source.conn.total_changes
    monkeypatch.setattr(source.store, "publish_bytes", _forbid)
    source.conn.execute("PRAGMA query_only=ON")
    assert _call(source) == result
    assert _call(source, admit=True) == result
    assert _catalog(source.conn) == before
    assert _objects(source.store) == objects
    assert source.conn.total_changes == changes
    assert source.repository.pins == ["shadow"] * 3


def test_registration_and_admission_never_read_old_files_or_fit(source, monkeypatch):
    """Poison prior ledgers, reports, model files and outputs against all file opens."""
    forbidden = [source.code_root / path for path in (
        "experiments/LEDGER.csv", "experiments/prior/REPORT.md",
        "experiments/prior/results.parquet", "data/models/model.joblib")]
    for path in forbidden:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic existing output; must stay unread and unchanged")
    bytes_before = {path: path.read_bytes() for path in forbidden}
    originals = [builtins.open, io.open, os.open]

    def guarded(original):
        def call(path, *args, **kwargs):
            if isinstance(path, (str, bytes, os.PathLike)):
                assert Path(os.fsdecode(path)).absolute() not in forbidden, "old output accessed"
            return original(path, *args, **kwargs)
        return call

    with monkeypatch.context() as patch:
        for owner, original in zip((builtins, io, os), originals):
            patch.setattr(owner, "open", guarded(original))
        from sklearn.linear_model import LogisticRegression
        patch.setattr(LogisticRegression, "fit", _forbid)
        result = _call(source)
        assert _call(source, admit=True) == result
    assert {path: path.read_bytes() for path in forbidden} == bytes_before
    assert list(source.code_root.rglob("REPORT.md")) == [forbidden[1]]


def test_target_table_is_not_a_preregistration_prerequisite(tmp_path):
    """The metadata-only boundary must work without any committed outcomes."""
    source = _source(tmp_path, include_targets=False)
    try:
        assert "computed_moves" not in source.snapshot.table_versions
        registered = _call(source)
        assert _call(source, admit=True) == registered
        assert {name for name, _ in source.repository.scans} == {"earnings_events"}
    finally:
        source.conn.close()


@pytest.mark.parametrize("changes", [{"experiment_id": "another-experiment"},
    {"primary_arm_id": "another-arm", "arms": ["another-arm"]}])
def test_distinct_experiment_or_primary_arm_has_a_distinct_registration(source, changes):
    """The stable key separates experiment IDs and primary-arm identities."""
    first = _call(source)
    second = _call(source, spec=_spec(**changes))
    assert first.run_id != second.run_id
    assert first.variant_id != second.variant_id
    assert source.conn.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0] == 2
    assert _call(source, admit=True) == first
    assert _call(source, admit=True, spec=_spec(**changes)) == second


def test_missing_registration_is_read_only_refusal(source, monkeypatch):
    """Admission cannot turn an absent registration into an implicit reservation."""
    before, objects = _catalog(source.conn), _objects(source.store)
    source.conn.execute("PRAGMA query_only=ON")
    monkeypatch.setattr(source.store, "publish_bytes", _forbid)
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        _call(source, admit=True)
    assert _catalog(source.conn) == before
    assert _objects(source.store) == objects


@pytest.mark.parametrize("changes", [
    {"input_files": ["old-results.csv"]}, {"runner": "run.sh"},
    {"runner": "../run.py"}, {"runner": "<absolute-runner>"}, {"runner": "run\\bad.py"},
    {"arms": ["candidate", "other"]}, {"primary_arm_id": "other"},
    {"seed": True}, {"economic_params": {"unused": 1}},
])
def test_invalid_spec_refuses_before_head_or_files(source, changes):
    """Invalid plans and external inputs cannot reach snapshot pinning or scans."""
    if changes.get("runner") == "<absolute-runner>":
        changes = {"runner": str(source.code_root / "run.py")}
    before, objects = _catalog(source.conn), _objects(source.store)
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        _call(source, spec=_spec(**changes))
    assert source.repository.pins == source.repository.scans == []
    assert _catalog(source.conn) == before
    assert _objects(source.store) == objects


def test_missing_python_source_has_no_registration_effect(source):
    """Missing runner bytes refuse before reading canonical metadata or publishing."""
    (source.code_root / "experiment/run.py").unlink()
    before, objects = _catalog(source.conn), _objects(source.store)
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        _call(source)
    assert source.repository.pins == source.repository.scans == []
    assert _catalog(source.conn) == before
    assert _objects(source.store) == objects


def test_missing_scope_head_refuses_without_fallback_or_scan(source):
    """An unresolved scope must never fall back to another committed head."""
    before, objects = _catalog(source.conn), _objects(source.store)
    with pytest.raises(OpsError, match="SNAPSHOT_UNRESOLVED"):
        _call(source, scope="never-committed")
    assert source.repository.pins == ["never-committed"]
    assert source.repository.scans == []
    assert _catalog(source.conn) == before
    assert _objects(source.store) == objects


@pytest.mark.parametrize("kind", ["random", "rolling", "unknown", "empty", "invalid-month"])
@pytest.mark.parametrize("admit", [False, True])
def test_holdout_refuses_before_any_target_access_or_write(source, kind, admit):
    """Both public entrypoints preserve the shared holdout refusal and no-write boundary."""
    kwargs = {"event_ids": {
        "random": [source.events[2]["event_id"]], "rolling": [source.events[3]["event_id"]],
        "unknown": ["UNKNOWN"], "empty": [], "invalid-month": None,
    }[kind]}
    if kind == "invalid-month":
        kwargs["as_of_month"] = "2025-13"
    before, objects = _catalog(source.conn), _objects(source.store)
    with pytest.raises(DataError, match="HOLDOUT_ACCESS_DENIED"):
        _call(source, admit=admit, **kwargs)
    assert all(table == "earnings_events" for table, _ in source.repository.scans)
    if kind == "invalid-month":
        assert source.repository.scans == []
    assert _catalog(source.conn) == before
    assert _objects(source.store) == objects


@pytest.mark.parametrize("dimension", [
    "hypothesis", "seed", "folds", "economics", "exit", "price_source", "runner",
    "runner-bytes", "imported-helper", "environment", "snapshot", "scope",
    "population", "month", "random-version", "rolling-version",
])
def test_every_binding_dimension_conflicts_without_changing_registered_evidence(source, monkeypatch, dimension):
    """Every identity-bearing dimension is immutable under the same registration key."""
    original = _call(source)
    spec, kwargs = _spec(), {}
    changes = {"hypothesis": {"hypothesis": "changed hypothesis"}, "seed": {"seed": 8},
               "folds": {"folds": ["2023", "2024"]},
               "economics": {"economic_params": {"fill": 0.7}},
               "exit": {"economic_params": {"fill": 0.5, "exit": {"kind": "fixed_day", "trading_days": 2}}},
               "price_source": {"price_source": "another-canonical-source"}}
    if dimension in changes:
        spec = _spec(**changes[dimension])
    elif dimension == "runner":
        (source.code_root / "experiment/other.py").write_text("# Another runner.\n")
        spec = _spec(runner="experiment/other.py")
    elif dimension in {"runner-bytes", "imported-helper"}:
        leaf = "run.py" if dimension == "runner-bytes" else "helper.py"
        path = source.code_root / "experiment" / leaf
        path.write_text(path.read_text() + "# Changed implementation.\n")
    elif dimension == "environment":
        environment = native.environment_identity(1)
        environment["libraries"]["numpy"] = "different-version"
        monkeypatch.setattr(native, "environment_identity", lambda _threads: environment)
    elif dimension == "snapshot":
        _advance_head(source, _new_snapshot(source))
    elif dimension == "scope":
        source.conn.execute("INSERT INTO data_snapshot_heads SELECT 'other', snapshot_id, 1, "
                            "updated_at, update_receipt_ref FROM data_snapshot_heads WHERE scope='shadow'")
        kwargs["scope"] = "other"
    elif dimension == "population":
        kwargs["event_ids"] = [source.events[0]["event_id"]]
    elif dimension == "month":
        kwargs["as_of_month"] = "2025-02"
    else:
        monkeypatch.setattr(ExperimentHoldouts, "random_version" if dimension == "random-version"
                            else "rolling_version", "changed-membership.v2")
    before, objects = _catalog(source.conn), _objects(source.store)
    for admit in (False, True):
        with pytest.raises(OpsError, match="EXPERIMENT_IDENTITY_CONFLICT") as caught:
            _call(source, admit=admit, spec=spec, **kwargs)
        assert not caught.value.problem.retryable
        assert _catalog(source.conn) == before
        assert _objects(source.store) == objects
        assert source.store.read_verified(_ref(source, original)) == original.binding_json


def test_current_head_is_pinned_once_even_if_head_moves_during_metadata_read(source, monkeypatch):
    """A concurrent head move cannot retarget an already pinned metadata scan."""
    newer = _new_snapshot(source)
    original_scan = source.repository.scan
    moved = False

    def move_then_scan(query, *, table_name):
        nonlocal moved
        if not moved:
            moved = True
            _advance_head(source, newer)
        yield from original_scan(query, table_name=table_name)

    monkeypatch.setattr(source.repository, "scan", move_then_scan)
    result = _call(source)
    assert source.repository.pins == ["shadow"]
    assert result.document["snapshot"] == to_document(source.snapshot)
    assert all(query.snapshot_id == source.snapshot.snapshot_id for _, query in source.repository.scans)
    with pytest.raises(OpsError, match="EXPERIMENT_IDENTITY_CONFLICT"):
        _call(source, admit=True)


@pytest.mark.parametrize("kind", [
    "run-spec", "run-input", "run-mode", "hypothesis-input", "hypothesis-payload",
    "hypothesis-document", "missing-hypothesis",
    "invalid-json", "list-evidence", "null-evidence", "missing-ref", "variant", "count", "bool-count",
    "ref-id", "ref-schema", "ref-size", "ref-hash", "ref-path", "ref-unknown", "other-object",
    "missing-object", "changed-bytes",
])
def test_tampered_catalog_reference_or_artifact_is_never_admitted_or_repaired(source, kind):
    """Corrupt evidence refuses non-retryably and remains untouched for inspection."""
    result = _call(source)
    evidence, ref = _evidence(source, result.run_id), _ref(source, result)
    if kind.startswith("run-"):
        field, value = {"run-spec": ("spec_hash", content_hash("other")),
                        "run-input": ("input_hash", content_hash("other")),
                        "run-mode": ("mode", "smoke")}[kind]
        source.conn.execute(f"UPDATE experiment_runs SET {field}=?", (value,))
    elif kind in {"hypothesis-input", "hypothesis-payload"}:
        field = "input_hash" if kind == "hypothesis-input" else "payload_hash"
        source.conn.execute(f"UPDATE hypotheses SET {field}=?", (content_hash("other"),))
    elif kind == "hypothesis-document":
        source.conn.execute("UPDATE hypotheses SET payload_json=?", ('{"changed": true}',))
    elif kind == "missing-hypothesis":
        source.conn.execute("DELETE FROM hypotheses")
    elif kind in {"invalid-json", "list-evidence", "null-evidence"}:
        text = {"invalid-json": "not-json", "list-evidence": "[]", "null-evidence": "null"}[kind]
        source.conn.execute("UPDATE experiment_runs SET evidence_json=?", (text,))
    elif kind in {"missing-object", "changed-bytes"}:
        path = source.store.root / ref.storage_key
        if kind == "missing-object":
            path.unlink()
        else:
            path.chmod(0o644)
            path.write_bytes(b"!" * ref.byte_size)
    else:
        if kind == "missing-ref":
            evidence.pop("native_registration")
        elif kind == "variant":
            evidence["variant_id"] = content_hash("other")
        elif kind in {"count", "bool-count"}:
            evidence["planned_variants"] = 2 if kind == "count" else True
        elif kind == "other-object":
            evidence["native_registration"] = to_document(source.store.publish_bytes(
                b"{}", schema_ref="native_experiment_registration.v1.0"))
        else:
            field, value = {"ref-id": ("artifact_id", "art_wrong"),
                            "ref-schema": ("schema_ref", "unrelated.v1.0"),
                            "ref-size": ("byte_size", ref.byte_size + 1),
                            "ref-hash": ("content_hash", content_hash("other")),
                            "ref-path": ("storage_key", "../untrusted"),
                            "ref-unknown": ("unknown_field", "unexpected")}[kind]
            evidence["native_registration"][field] = value
        source.conn.execute("UPDATE experiment_runs SET evidence_json=?", (json.dumps(evidence),))
    before, objects = _catalog(source.conn), _objects(source.store)
    for admit in (True, False):
        with pytest.raises(OpsError, match="EXPERIMENT_IDENTITY_CONFLICT") as caught:
            _call(source, admit=admit)
        assert not caught.value.problem.retryable
        assert _catalog(source.conn) == before
        assert _objects(source.store) == objects


def test_existing_nonnative_primary_reservation_cannot_be_adopted(source):
    """A legacy primary reservation cannot silently become native preregistration."""
    register_hypothesis(source.conn, _spec(), content_hash("other-input"), mode="primary")
    before = _catalog(source.conn)
    with pytest.raises(OpsError, match="EXPERIMENT_IDENTITY_CONFLICT"):
        _call(source)
    assert _catalog(source.conn) == before
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        _call(source, admit=True)


@pytest.mark.parametrize("point", ["copied", "linked"])
def test_publication_failure_leaves_no_partial_registration_or_implicit_retry(source, point):
    """Artifact crash points cannot leave either reservation row partially admitted."""
    calls = []

    def fault(at):
        calls.append(at)
        if at == point:
            raise OSError("synthetic publication failure")

    source.store = ArtifactStore(source.store.root, fault=fault)
    before = _catalog(source.conn)
    with pytest.raises(OSError, match="synthetic publication failure"):
        _call(source)
    assert calls.count(point) == 1
    assert _catalog(source.conn) == before
    assert not source.conn.in_transaction
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        _call(source, admit=True)


def test_commit_failure_rolls_back_both_rows_and_leaves_only_complete_orphan(source):
    """A failed COMMIT rolls back both rows while allowing a complete durable orphan."""
    class FailCommit:
        def __getattr__(self, name):
            return getattr(source.conn, name)

        def execute(self, statement, *args):
            if statement == "COMMIT":
                raise sqlite3.OperationalError("synthetic commit failure")
            return source.conn.execute(statement, *args)

    before, objects = _catalog(source.conn), _objects(source.store)
    with pytest.raises(sqlite3.OperationalError, match="synthetic commit failure"):
        native.register_native(FailCommit(), source.store, source.repository, _spec(),
                               code_root=source.code_root, as_of_month=MONTH)
    assert _catalog(source.conn) == before
    assert not source.conn.in_transaction
    orphans = {key: value for key, value in _objects(source.store).items() if key not in objects}
    assert len(orphans) == 1
    assert json.loads(next(iter(orphans.values())))["spec_hash"] == _spec().spec_hash
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        _call(source, admit=True)
    result = _call(source)
    assert result.binding_json == next(iter(orphans.values()))
    assert _call(source, admit=True) == result


def test_concurrent_conflicting_registrations_admit_exactly_one_whole_binding(source):
    """Two real SQLite connections race after publication and reserve only one identity."""
    ready = Barrier(2)

    class RacingStore(ArtifactStore):
        def publish_bytes(self, data, *, schema_ref):
            result = super().publish_bytes(data, schema_ref=schema_ref)
            ready.wait(timeout=20)
            return result

    def register(seed):
        conn = connect(source.catalog_path, must_exist=True)
        store = RacingStore(source.store.root)
        try:
            result = native.register_native(conn, store, MetadataRepository(conn, store),
                                            _spec(seed=seed), code_root=source.code_root,
                                            as_of_month=MONTH)
            return seed, result
        except OpsError as exc:
            return seed, exc
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(register, seed) for seed in (7, 8)]
        results = [future.result(timeout=30) for future in futures]
    winners = [(seed, result) for seed, result in results if isinstance(result, native.NativeRegistration)]
    losers = [(seed, result) for seed, result in results if isinstance(result, OpsError)]
    assert len(winners) == len(losers) == 1
    seed, result = winners[0]
    assert losers[0][1].code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert not losers[0][1].problem.retryable
    assert source.conn.execute("SELECT COUNT(*) FROM experiment_runs").fetchone()[0] == 1
    assert source.conn.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0] == 1
    assert _call(source, admit=True, spec=_spec(seed=seed)) == result
    assert source.store.read_verified(_ref(source, result)) == result.binding_json
    with pytest.raises(OpsError, match="EXPERIMENT_IDENTITY_CONFLICT"):
        _call(source, admit=True, spec=_spec(seed=losers[0][0]))


@pytest.mark.parametrize("seam", ["population", "publication"])
def test_mutating_callers_economics_cannot_rewrite_captured_spec(source, monkeypatch, seam):
    """Caller-owned nested economics cannot drift while scans or publication execute."""
    economics = {"fill": 0.5, "exit": {"kind": "fixed_day", "trading_days": 2}}
    pristine = _spec(economic_params=json.loads(json.dumps(economics)))
    supplied = _spec(economic_params=economics)

    def mutate():
        supplied.economic_params["fill"] = 0.9
        supplied.economic_params["exit"]["trading_days"] = 5

    if seam == "population":
        original = source.repository.scan

        def scan(query, *, table_name):
            mutate()
            yield from original(query, table_name=table_name)

        monkeypatch.setattr(source.repository, "scan", scan)
    else:
        original = source.store.publish_bytes

        def publish(data, *, schema_ref):
            mutate()
            return original(data, schema_ref=schema_ref)

        monkeypatch.setattr(source.store, "publish_bytes", publish)
    result = _call(source, spec=supplied)
    assert supplied.spec_hash != pristine.spec_hash
    assert result.document["spec_hash"] == pristine.spec_hash
    assert result.document["execution_plan"] == resolve_experiment_plan(pristine).as_document()
    assert source.conn.execute("SELECT spec_hash FROM experiment_runs").fetchone()[0] == pristine.spec_hash
    assert source.conn.execute("SELECT spec_hash FROM hypotheses").fetchone()[0] == pristine.spec_hash
    assert _call(source, admit=True, spec=pristine) == result
    with pytest.raises(OpsError, match="EXPERIMENT_IDENTITY_CONFLICT"):
        _call(source, admit=True, spec=supplied)


@pytest.mark.parametrize("leaf", ["run.py", "helper.py"])
def test_symlinked_source_closure_is_refused_before_metadata_or_registration(source, leaf):
    """Runner and imported-helper symlinks cannot extend the supplied source boundary."""
    target = source.code_root.parent / "outside.py"
    target.write_text("# This source lies outside the supplied checkout.\n")
    path = source.code_root / "experiment" / leaf
    path.unlink()
    path.symlink_to(target)
    before, objects = _catalog(source.conn), _objects(source.store)
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        _call(source)
    assert source.repository.pins == source.repository.scans == []
    assert _catalog(source.conn) == before
    assert _objects(source.store) == objects


@pytest.mark.parametrize("change", [{"input_files": None}, {"input_files": []},
                                    {"hypothesis": " "}, {"experiment_id": " "},
                                    {"arms": (" ",), "primary_arm_id": " "}])
def test_malformed_direct_spec_is_typed_and_has_no_reads(source, change):
    """Direct dataclass callers receive the same typed validation as parsed specs."""
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        _call(source, spec=replace(_spec(), **change))
    assert source.repository.pins == source.repository.scans == []
    assert source.conn.execute("SELECT COUNT(*) FROM experiment_runs").fetchone()[0] == 0


@pytest.mark.parametrize("scope", [[], {}, None, "", " "], ids=[
    "list", "mapping", "none", "empty", "whitespace"])
@pytest.mark.parametrize("admit", [False, True])
def test_invalid_scope_shape_refuses_before_any_pin_scan_or_write(source, scope, admit):
    """Malformed scopes cannot escape as SQLite binding errors or trigger fallback."""
    before, objects = _catalog(source.conn), _objects(source.store)
    with pytest.raises(OpsError, match="SNAPSHOT_UNRESOLVED"):
        _call(source, admit=admit, scope=scope)
    assert source.repository.pins == source.repository.scans == []
    assert _catalog(source.conn) == before
    assert _objects(source.store) == objects


def test_native_ops_imports_are_explicitly_public_and_have_a_documented_consumer():
    """Cover this outside-engine consumer, which the engine import graph omits."""
    root = Path(__file__).resolve().parents[3]
    readme = (root / "engine/v2/ops/README.md").read_text()
    public = directive(readme, "public-interface")
    imports = [node for node in ast.walk(ast.parse(Path(native.__file__).read_text()))
               if isinstance(node, ast.ImportFrom) and node.module.startswith("engine.v2.ops.")]
    assert imports
    for node in imports:
        for alias in node.names:
            assert alias.name in public
            qualified = node.module.removeprefix("engine.v2.ops.") + "." + alias.name
            assert f"`{qualified}`" in readme
    assert "`experiments/native_registration.py`" in readme
    architecture = (root / "engine/v2/ops/ARCHITECTURE.md").read_text()
    assert "`experiments/native_registration.py`" in architecture
    assert "`catalog.transaction` around `register_hypothesis_in_transaction`" in architecture


def test_native_registration_tests_use_the_supported_engine_mutation_matrix():
    """Top-level consumers run in the matrix; the pilot mutates engine paths only."""
    from tools import mutation_pilot

    config = mutation_pilot.load_config()
    assert "tests/v2/research/test_native_registration.py" in mutation_pilot.test_files(
        config, "ops_cli")
    assert all(path.startswith("engine/") for path in mutation_pilot.mutate_files(
        config, "ops_cli"))
