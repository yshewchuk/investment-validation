"""Public fixtures and preservation oracles for bounded native evidence tests."""
import json

from engine.v2.contracts import ArtifactRef
from engine.v2.foundation import from_document
from engine.v2.ops.experiments import experiment_spec_from_document
from experiments import native_registration as native
from tests.v2.research.test_prediction_inputs import MONTH


def register_native_case(source):
    """Register this suite's minimal arm through the public admission API."""
    spec = experiment_spec_from_document({
        "experiment_id": "synthetic-native-bounds", "hypothesis": "synthetic evidence boundaries",
        "primary_arm_id": "candidate", "arms": ["candidate"], "folds": ["2024"],
        "seed": 7, "economic_params": {"fill": 0.5}, "price_source": "option_chains",
        "runner": "experiment/run.py",
    })
    return native.register_native(source.conn, source.store, source.repository, spec,
                                  code_root=source.code_root, as_of_month=MONTH)


def native_evidence(source, run_id):
    return json.loads(source.conn.execute(
        "SELECT evidence_json FROM experiment_runs WHERE run_id=?", (run_id,)).fetchone()[0])


def registration_ref(source, registration):
    return from_document(ArtifactRef, native_evidence(source, registration.run_id)["native_registration"])


def catalog_rows(conn):
    """Snapshot every catalog table, including unrelated rows protected by refusal."""
    tables = [row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    return {table: sorted(tuple(row) for row in conn.execute(f'SELECT * FROM "{table}"'))
            for table in tables}


def object_bytes(store):
    """Capture durable objects only, without transient report staging files."""
    return {str(path.relative_to(store.root)): path.read_bytes()
            for path in (store.root / "objects").rglob("*") if path.is_file()}


def assert_identity_refusal(source, captured, registration):
    """Check the exact nonretryable envelope and its readable private receipt."""
    error = captured.value
    assert error.code == "EXPERIMENT_IDENTITY_CONFLICT" and error.problem.retryable is False
    ref = from_document(ArtifactRef, error.problem.details["refusal_ref"])
    receipt = json.loads(source.store.read_verified(ref))
    assert receipt == {"schema_version": "native_experiment_refusal.v1.0",
                       "failure_code": "EXPERIMENT_IDENTITY_CONFLICT",
                       "request": {"run_id": registration.run_id}, "variant_id": None}


def assert_objects_preserved(source, before):
    after = object_bytes(source.store)
    assert {key: after[key] for key in before} == before
