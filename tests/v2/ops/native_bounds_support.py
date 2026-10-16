"""Public fixtures and preservation oracles for bounded native evidence tests."""
import json

from engine.v2.contracts import ArtifactRef
from engine.v2.foundation import from_document
from engine.v2.ops.experiments import experiment_spec_from_document
from experiments import native_registration as native
from tests.v2.research.test_prediction_inputs import MONTH


class BoundedSqlRows:
    """Observe selected SQL columns before callers can decode transferred rows.

    ``max_bytes`` is the SQL encoding's byte allowance, including any UTF-16
    multiplier. Record lengths, not payloads, and return the original rows.
    """

    def __init__(self, conn, *, max_bytes, column, sql_fragment):
        self.conn, self.max_bytes, self.column = conn, max_bytes, column
        self.sql_fragment = sql_fragment.casefold()
        self.encoding = conn.execute("PRAGMA encoding").fetchone()[0]
        self.sizes = []

    def __getattr__(self, name):
        return getattr(self.conn, name)

    def execute(self, sql, *args, **kwargs):
        cursor = self.conn.execute(sql, *args, **kwargs)
        if self.sql_fragment in sql.casefold():
            return BoundedSqlCursor(cursor, self)
        return cursor


class BoundedSqlCursor:
    """Preserve cursor behavior while checking every fetched evidence value."""

    def __init__(self, cursor, rows):
        self.cursor, self.rows = cursor, rows
        self.index = (rows.column if isinstance(rows.column, int) else
                      [item[0] for item in cursor.description].index(rows.column))

    def __getattr__(self, name):
        return getattr(self.cursor, name)

    def checked(self, row):
        if row is not None:
            value = row[self.index]
            assert value is None or isinstance(value, (bytes, str))
            size = None if value is None else len(value.encode(self.rows.encoding) if isinstance(value, str) else value)
            assert size is None or size <= self.rows.max_bytes, "oversized evidence crossed the SQL row-transfer bound"
            self.rows.sizes.append(size)
        return row

    def fetchone(self):
        return self.checked(self.cursor.fetchone())

    def fetchmany(self, *args, **kwargs):
        return [self.checked(row) for row in self.cursor.fetchmany(*args, **kwargs)]

    def fetchall(self):
        return [self.checked(row) for row in self.cursor.fetchall()]

    def __iter__(self):
        return self

    def __next__(self):
        return self.checked(next(self.cursor))


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
