"""Mock terminals and isolated metadata catalogs only; never final data or live authority."""
import ast
import builtins
import io
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import pytest

from engine.v2.contracts import DATA_FAILURE_CODES, FAILURE_CODES
from engine.v2.foundation import artifact_reference, canonical_json, content_hash, to_document
from engine.v2.ops import final_holdout as authority, schema
from engine.v2.ops.catalog import connect
from engine.v2.ops.errors import OpsError
from engine.v2.ops.migrations import applied_versions, migrate
from engine.v2.ops.native_registration import verified_registration_bytes
from experiments import native_registration as native
from tests.v2.research.test_final_holdout_requests import _prepared, source  # noqa: F401


class Terminal(io.StringIO):
    def isatty(self):
        return True


@pytest.fixture
def context(source, monkeypatch):
    registration, request = _prepared(source)
    source.registration, source.request = registration, request
    source.clock = SimpleNamespace(now=lambda: datetime(2026, 10, 10, tzinfo=timezone.utc))
    monkeypatch.setattr(authority.os, "getuid", lambda: 1234)
    monkeypatch.setattr(authority.pwd, "getpwuid", lambda uid: SimpleNamespace(pw_name="synthetic-user"))
    monkeypatch.setattr(authority, "sys", SimpleNamespace(stdin=Terminal(), stdout=Terminal()))
    def input_line(prompt):
        line = authority.sys.stdin.readline()
        if not line:
            raise EOFError
        return line.removesuffix("\n")
    monkeypatch.setattr(builtins, "input", input_line)
    _assent(monkeypatch, request)
    return source


def _phrase(request):
    return f'AUTHORIZE {json.dumps(request.experiment_id)} {content_hash(to_document(request))} AS "1234:synthetic-user"'


def _assent(monkeypatch, request, text=None):
    monkeypatch.setattr(authority.sys, "stdin", Terminal((_phrase(request) if text is None else text) + "\n"))


def _authorize(context, **kwargs):
    return authority.authorize_final_holdout(
        context.conn, context.store, context.request, clock=context.clock, **kwargs)


def _rows(context):
    return [dict(row) for row in context.conn.execute("SELECT * FROM final_holdout_authorizations")]


def test_exact_user_act_is_the_single_durable_spend_without_data_reads(context, monkeypatch):
    read = context.store.read_verified
    seen = []

    def only_registration(ref):
        assert ref == context.request.registration_ref
        seen.append(ref)
        return read(ref)

    def forbidden(*args, **kwargs):
        pytest.fail("authority attempted publication, pinning or population access")

    monkeypatch.setattr(context.store, "read_verified", only_registration)
    monkeypatch.setattr(context.store, "publish_bytes", forbidden)
    monkeypatch.setattr(context.repository, "resolve_pinned", forbidden)
    monkeypatch.setattr(native, "load_population", forbidden)
    result = _authorize(context)
    assert result == _rows(context)[0]
    assert result == {"experiment_id": context.request.experiment_id,
                      "schema_version": "final_holdout_authorization.v1.0",
                      "registration_run_id": context.registration.run_id,
                      "request_digest": content_hash(to_document(context.request)),
                      "request_json": canonical_json(to_document(context.request)),
                      "actor": "1234:synthetic-user", "authorized_at": "2026-10-10T00:00:00.000000Z",
                      "confirmation": _phrase(context.request)}
    assert seen == [context.request.registration_ref] * 2
    assert result["request_json"] in authority.sys.stdout.getvalue()
    assert not {"grant", "expires_at", "result", "read_token"} & result.keys()
    context.conn.close()
    context.conn = connect(context.root / "catalog.sqlite", must_exist=True)
    assert _rows(context) == [result]


@pytest.mark.parametrize("mode", ["stdin", "stdout", "empty", "yes", "digest-only", "eof", "interrupt"])
def test_terminal_and_fresh_explicit_assent_are_required(context, monkeypatch, mode):
    if mode in ("stdin", "stdout"):
        monkeypatch.setattr(authority.sys, mode, io.StringIO())
        monkeypatch.setattr(context.store, "read_verified", lambda *a: pytest.fail("non-TTY read"))
    elif mode == "eof":
        monkeypatch.setattr(authority.sys, "stdin", Terminal())
    elif mode == "interrupt":
        def interrupted(*args):
            raise KeyboardInterrupt
        monkeypatch.setattr(builtins, "input", interrupted)
    else:
        _assent(monkeypatch, context.request, {"empty": "", "yes": "yes",
                                              "digest-only": content_hash(to_document(context.request))}[mode])
    with pytest.raises(OpsError, match="HOLDOUT_ACCESS_DENIED") as caught:
        _authorize(context)
    assert not caught.value.problem.retryable
    assert _rows(context) == []


def test_repeated_act_needs_confirmation_and_returns_original_without_expiry(context, monkeypatch):
    first = _authorize(context)
    context.clock = SimpleNamespace(now=lambda: datetime(2099, 1, 1, tzinfo=timezone.utc))
    with pytest.raises(OpsError, match="HOLDOUT_ACCESS_DENIED"):
        _authorize(context)
    _assent(monkeypatch, context.request)
    assert _authorize(context) == first
    assert _rows(context) == [first]
    monkeypatch.setattr(authority.pwd, "getpwuid", lambda uid: SimpleNamespace(pw_name="another-user"))
    _assent(monkeypatch, context.request, _phrase(context.request).replace("synthetic-user", "another-user"))
    with pytest.raises(OpsError, match="HOLDOUT_ACCESS_DENIED"):
        _authorize(context)
    assert _rows(context) == [first]


@pytest.mark.parametrize("field", ["snapshot_id", "execution_plan_hash", "source_closure_hash", "final_read"])
def test_caller_metadata_cannot_substitute_for_original_registration(context, monkeypatch, field):
    value = (replace(context.request.final_read, frozen_model_refs=()) if field == "final_read" else
             content_hash("forged") if field.endswith("hash") else "other-snapshot")
    context.request = replace(context.request, **{field: value})
    monkeypatch.setattr(builtins, "input", lambda *a: pytest.fail("forged request reached confirmation"))
    with pytest.raises(OpsError, match="EXPERIMENT_IDENTITY_CONFLICT"):
        _authorize(context)
    assert _rows(context) == []


@pytest.mark.parametrize("kind", ["missing", "reference", "schema", "hypothesis", "variant", "bytes"])
def test_untrusted_catalog_or_registration_refuses_without_spend(context, monkeypatch, kind):
    row = context.conn.execute("SELECT evidence_json FROM experiment_runs WHERE run_id=?",
                               (context.registration.run_id,)).fetchone()
    evidence = json.loads(row[0])
    if kind == "missing":
        context.request = replace(context.request, registration_run_id="absent")
    elif kind == "reference":
        evidence["native_registration"]["content_hash"] = content_hash("another-object")
    elif kind == "schema":
        evidence["native_registration"]["schema_ref"] = "unrelated.v1.0"
    elif kind == "hypothesis":
        context.conn.execute("UPDATE hypotheses SET payload_json='[]'")
    elif kind == "variant":
        evidence["variant_id"] = content_hash("other")
    else:
        monkeypatch.setattr(context.store, "read_verified", lambda *a: b"{}")
    context.conn.execute("UPDATE experiment_runs SET evidence_json=? WHERE run_id=?",
                         (json.dumps(evidence), context.registration.run_id))
    if kind in ("missing", "reference", "schema"):
        monkeypatch.setattr(context.store, "read_verified", lambda *a: pytest.fail("unbound object read"))
    with pytest.raises(OpsError, match="EXPERIMENT_IDENTITY_CONFLICT"):
        _authorize(context)
    assert _rows(context) == []


def test_registration_is_rechecked_after_confirmation_before_commit(context, monkeypatch):
    def changed(prompt):
        assert not context.conn.in_transaction
        context.conn.execute("UPDATE hypotheses SET input_hash=?", (content_hash("changed"),))
        return _phrase(context.request)
    monkeypatch.setattr(builtins, "input", changed)
    with pytest.raises(OpsError, match="EXPERIMENT_IDENTITY_CONFLICT"):
        _authorize(context)
    assert _rows(context) == []


def test_caller_hash_cannot_make_an_arbitrary_object_preregistration(context, monkeypatch):
    ref = artifact_reference(b'{"synthetic_metadata":"unregistered"}', "native_experiment_registration.v1.1")
    context.request = replace(context.request, registration_ref=ref, variant_id=ref.content_hash)
    monkeypatch.setattr(context.store, "read_verified", lambda *a: pytest.fail("caller-selected object was read"))
    with pytest.raises(OpsError, match="EXPERIMENT_IDENTITY_CONFLICT"):
        _authorize(context)
    assert _rows(context) == []


@pytest.mark.parametrize("after_commit", [False, True])
@pytest.mark.parametrize("error", [sqlite3.OperationalError, KeyboardInterrupt])
def test_commit_failure_never_acknowledges_and_identical_retry_reconciles(context, monkeypatch, after_commit, error):
    real = context.conn

    class FailedCommit:
        def __getattr__(self, name):
            return getattr(real, name)

        def execute(self, statement, *args):
            if statement == "COMMIT":
                if after_commit:
                    real.execute(statement)
                raise error("synthetic storage failure")
            return real.execute(statement, *args)

    context.conn = FailedCommit()
    with pytest.raises(OpsError, match="RESOURCE_UNAVAILABLE"):
        _authorize(context)
    assert len(_rows(context)) == int(after_commit)
    assert not real.in_transaction
    context.conn = real
    prior = _rows(context)
    _assent(monkeypatch, context.request)
    acknowledgement = _authorize(context)
    assert _rows(context) == [acknowledgement]
    if after_commit:
        assert prior == [acknowledgement]


@pytest.mark.parametrize("same_request", [True, False])
def test_concurrent_acts_serialize_to_one_experiment_spend(context, monkeypatch, same_request):
    requests = [context.request, context.request]
    if not same_request:
        context.spec = replace(context.spec, primary_arm_id="alternate", arms=("alternate",))
        _, requests[1] = _prepared(context)
    barrier = Barrier(2)

    def assent(prompt):
        barrier.wait(timeout=10)
        return prompt.removeprefix("Type exactly: ").rstrip("\n")

    monkeypatch.setattr(builtins, "input", assent)

    def invoke(request):
        conn = connect(context.root / "catalog.sqlite", must_exist=True)
        try:
            return authority.authorize_final_holdout(conn, context.store, request, clock=context.clock)
        except OpsError as exc:
            return exc.code
        finally:
            conn.close()

    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(invoke, requests))
    rows = _rows(context)
    assert len(rows) == 1
    assert sum(result == rows[0] for result in results) == (2 if same_request else 1)
    if not same_request:
        assert "IDEMPOTENCY_CONFLICT" in results


def test_spend_is_append_only_and_confirmation_overrides_are_not_supported(context):
    original = _authorize(context)
    for sql in ("DELETE FROM final_holdout_authorizations", "UPDATE final_holdout_authorizations SET actor='other'",
                "INSERT OR REPLACE INTO final_holdout_authorizations SELECT * FROM final_holdout_authorizations"):
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            context.conn.execute(sql)
    for kwargs in ({"confirmed": True}, {"actor": "user"}, {"interactive": True}, {"expires_at": None}):
        with pytest.raises(TypeError):
            _authorize(context, **kwargs)
    assert _rows(context) == [original]


def test_migration_is_atomic_and_does_not_rewrite_prior_versions(tmp_path):
    clock = SimpleNamespace(now=lambda: datetime(2026, 10, 10, tzinfo=timezone.utc))
    conn = connect(tmp_path / "old.sqlite")
    try:
        migrate(conn, schema.OWNER, schema.MIGRATIONS[:-1], clock=clock)
        prior = applied_versions(conn, schema.OWNER)
        def crash(point):
            raise RuntimeError("synthetic migration interruption")
        with pytest.raises(RuntimeError):
            migrate(conn, schema.OWNER, schema.MIGRATIONS, clock=clock, fault=crash)
        assert applied_versions(conn, schema.OWNER) == prior
        assert not conn.execute("SELECT 1 FROM sqlite_master WHERE name='final_holdout_authorizations'").fetchone()
        assert migrate(conn, schema.OWNER, schema.MIGRATIONS, clock=clock) == [10]
        assert migrate(conn, schema.OWNER, schema.MIGRATIONS, clock=clock) == []
    finally:
        conn.close()


def test_private_authority_is_not_reachable_from_production_callers_and_code_matches_data():
    root = Path(__file__).resolve().parents[3]
    target = "engine.v2.ops.final_holdout"
    for directory in ("engine", "experiments", "tools"):
        for path in (root / directory).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    assert node.module != target, path
                    assert not (node.module == "engine.v2.ops" and any(
                        alias.name == "final_holdout" for alias in node.names)), path
                elif isinstance(node, ast.Import):
                    assert target not in [alias.name for alias in node.names], path
    assert FAILURE_CODES["HOLDOUT_ACCESS_DENIED"] == DATA_FAILURE_CODES["HOLDOUT_ACCESS_DENIED"] == ("validation", False)
    assert verified_registration_bytes.__module__ == "engine.v2.ops.native_registration"


def test_mutation_owners_include_authority_and_shared_registration_verification():
    from tools import mutation_pilot
    if not (mutation_pilot.REPO / ".git").exists():
        pytest.skip("mutation work copy has no git index")
    config = mutation_pilot.load_config()
    for name in ("ops_catalog_store", "ops_catalog_state", "ops_cli"):
        assert "tests/v2/ops/test_final_holdout_authority.py" in mutation_pilot.test_files(config, name)
