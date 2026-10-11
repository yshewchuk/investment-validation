"""Mock terminals and isolated metadata catalogs only; never final data or live authority."""
import ast
import builtins
import io
import json
import multiprocessing
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import pytest

from engine.v2.contracts import DATA_FAILURE_CODES, FAILURE_CODES
from engine.v2.foundation import ArtifactError, artifact_reference, canonical_json, content_hash, to_document
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
    monkeypatch.setattr(authority.pwd, "getpwuid", lambda uid: SimpleNamespace(pw_name="renamed-user"))
    _assent(monkeypatch, context.request, _phrase(context.request).replace("synthetic-user", "renamed-user"))
    assert _authorize(context) == first
    assert _rows(context) == [first]
    monkeypatch.setattr(authority.os, "getuid", lambda: 5678)
    _assent(monkeypatch, context.request,
            _phrase(context.request).replace("1234:synthetic-user", "5678:renamed-user"))
    with pytest.raises(OpsError, match="HOLDOUT_ACCESS_DENIED"):
        _authorize(context)
    assert _rows(context) == [first]


def test_missing_os_user_record_is_resource_unavailable(context, monkeypatch):
    def missing_user(uid):
        raise KeyError(uid)

    monkeypatch.setattr(authority.pwd, "getpwuid", missing_user)
    with pytest.raises(OpsError, match="RESOURCE_UNAVAILABLE"):
        _authorize(context)
    assert _rows(context) == []


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
        assert not conn.execute("SELECT 1 FROM sqlite_master WHERE name='final_holdout_state_events'").fetchone()
        assert migrate(conn, schema.OWNER, schema.MIGRATIONS, clock=clock) == [11]
        assert migrate(conn, schema.OWNER, schema.MIGRATIONS, clock=clock) == []
    finally:
        conn.close()


def test_native_admission_maps_artifact_reference_errors_to_identity_conflict(context, monkeypatch):
    def invalid_reference(*args, **kwargs):
        raise ArtifactError("UNSAFE_PATH", "synthetic invalid reference")

    monkeypatch.setattr(native, "artifact_reference", invalid_reference)
    with pytest.raises(OpsError, match="EXPERIMENT_IDENTITY_CONFLICT"):
        native._existing(context.conn, context.store, context.registration)


def test_private_authority_is_not_reachable_from_production_callers_and_code_matches_data():
    root = Path(__file__).resolve().parents[3]
    target = "engine.v2.ops.final_holdout"
    for directory in ("engine", "experiments", "tools"):
        for path in (root / directory).rglob("*.py"):
            package = path.relative_to(root).with_suffix("").parts[:-1]
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if node.level:
                        base = package[:len(package) - node.level + 1]
                        parts = (*base, *(module.split(".") if module else ()))
                        module = ".".join(parts)
                    imported = {alias.name for alias in node.names}
                    assert module != target, path
                    assert not (module == "engine.v2.ops" and "final_holdout" in imported), path
                elif isinstance(node, ast.Import):
                    assert target not in [alias.name for alias in node.names], path
                elif isinstance(node, ast.Call):
                    dynamic = (isinstance(node.func, ast.Name) and node.func.id == "__import__"
                               or isinstance(node.func, ast.Attribute) and node.func.attr == "import_module")
                    if dynamic and node.args and isinstance(node.args[0], ast.Constant):
                        assert node.args[0].value != target, path
    assert FAILURE_CODES["HOLDOUT_ACCESS_DENIED"] == DATA_FAILURE_CODES["HOLDOUT_ACCESS_DENIED"] == ("validation", False)
    assert verified_registration_bytes.__module__ == "engine.v2.ops.native_registration"


def test_mutation_owners_include_authority_and_shared_registration_verification():
    from tools import mutation_pilot
    if not (mutation_pilot.REPO / ".git").exists():
        pytest.skip("mutation work copy has no git index")
    config = mutation_pilot.load_config()
    for name in ("ops_final_holdout", "ops_catalog_store", "ops_catalog_state", "ops_cli"):
        assert "tests/v2/ops/test_final_holdout_authority.py" in mutation_pilot.test_files(config, name)
    owned = mutation_pilot.mutate_files(config, "ops_final_holdout")
    assert set(owned) == {"engine/v2/ops/final_holdout.py", "engine/v2/ops/native_registration.py"}
    for name in config["modules"]:
        if name != "ops_final_holdout":
            assert set(owned).isdisjoint(mutation_pilot.mutate_files(config, name))


def _state_phrase(context, action):
    digest = content_hash(to_document(context.request))
    return f'{action} {json.dumps(context.request.experiment_id)} {digest} AS "1234:synthetic-user"'


def _state_assent(monkeypatch, context, action):
    phrase = _state_phrase(context, action)
    monkeypatch.setattr(builtins, "input", lambda prompt: phrase)


def _start(context):
    return authority.start_final_holdout_read(
        context.conn, context.store, context.request, clock=context.clock)


def _revoke(context):
    return authority.revoke_final_holdout(
        context.conn, context.store, context.request, clock=context.clock)


@pytest.mark.parametrize("action", ["START", "REVOKE"])
def test_state_action_displays_exact_request_and_prompt(context, monkeypatch, action):
    _authorize(context)
    output = Terminal()
    monkeypatch.setattr(authority.sys, "stdout", output)
    expected_phrase = _state_phrase(context, action)
    prompts = []

    def confirm(prompt):
        prompts.append(prompt)
        return expected_phrase

    monkeypatch.setattr(builtins, "input", confirm)
    (_start if action == "START" else _revoke)(context)
    assert output.getvalue() == canonical_json(to_document(context.request)) + "\n"
    assert prompts == [f"Type exactly: {expected_phrase}\n"]
    assert _state_events(context)[0]["confirmation"] == expected_phrase


@pytest.mark.parametrize(("action", "failure", "expected_code", "retryable"), [
    ("START", "storage", "RESOURCE_UNAVAILABLE", True),
    ("START", "process", "RESOURCE_UNAVAILABLE", True),
    ("START", "user", "RESOURCE_UNAVAILABLE", True),
    ("START", "eof", "HOLDOUT_ACCESS_DENIED", False),
    ("START", "interrupt", "HOLDOUT_ACCESS_DENIED", False),
    ("REVOKE", "storage", "RESOURCE_UNAVAILABLE", True),
    ("REVOKE", "user", "RESOURCE_UNAVAILABLE", True),
    ("REVOKE", "eof", "HOLDOUT_ACCESS_DENIED", False),
    ("REVOKE", "interrupt", "HOLDOUT_ACCESS_DENIED", False),
    ("RECOVERY", "storage", "RESOURCE_UNAVAILABLE", True),
    ("RECOVERY", "process", "RESOURCE_UNAVAILABLE", True),
])
def test_lifecycle_faults_return_typed_retryability_and_preserve_state(
        context, monkeypatch, action, failure, expected_code, retryable):
    authorization = _authorize(context)
    if action == "RECOVERY":
        _state_assent(monkeypatch, context, "START")
        _start(context)
    def process_unavailable(*args):
        raise OSError("process unavailable")

    if failure in ("eof", "interrupt"):
        def interrupted(prompt):
            if failure == "eof":
                raise EOFError
            raise KeyboardInterrupt
        monkeypatch.setattr(builtins, "input", interrupted)
    elif failure == "user":
        def missing_user(uid):
            raise KeyError(uid)
        monkeypatch.setattr(authority.pwd, "getpwuid", missing_user)
    elif failure == "process" and action == "START":
        monkeypatch.setattr(authority, "process_info", process_unavailable)
    elif failure == "process":
        monkeypatch.setattr(authority, "is_alive", process_unavailable)
    elif failure == "storage":
        if action == "RECOVERY":
            monkeypatch.setattr(authority, "read_boot_id", lambda: "synthetic-reboot")
        def deny_event_insert(sql_action, table, column, database, trigger):
            if sql_action == sqlite3.SQLITE_INSERT and table == "final_holdout_state_events":
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        context.conn.set_authorizer(deny_event_insert)
    if action in ("START", "REVOKE") and failure not in ("eof", "interrupt", "user"):
        _state_assent(monkeypatch, context, action)
    try:
        with pytest.raises(OpsError) as caught:
            if action == "START":
                _start(context)
            elif action == "REVOKE":
                _revoke(context)
            else:
                authority.recover_interrupted_final_holdout_read(
                    context.conn, context.request.experiment_id, clock=context.clock)
        assert caught.value.code == expected_code
        assert caught.value.problem.retryable is retryable
        assert _rows(context) == [authorization]
        expected_states = ["read_started"] if action == "RECOVERY" else []
        assert [row["state"] for row in _state_events(context)] == expected_states
    finally:
        context.conn.set_authorizer(None)


def _state_events(context):
    return [dict(row) for row in context.conn.execute(
        "SELECT * FROM final_holdout_state_events ORDER BY event_id")]


@pytest.mark.parametrize("action", ["START", "REVOKE"])
def test_state_action_rechecks_registration_after_assent(context, monkeypatch, action):
    _authorize(context)

    def changed_during_assent(prompt):
        assert not context.conn.in_transaction
        context.conn.execute("UPDATE hypotheses SET input_hash=?", (content_hash("changed"),))
        return _state_phrase(context, action)

    monkeypatch.setattr(builtins, "input", changed_during_assent)
    state_action = _start if action == "START" else _revoke
    with pytest.raises(OpsError, match="EXPERIMENT_IDENTITY_CONFLICT"):
        state_action(context)
    assert _state_events(context) == []


def test_fenced_start_spends_once_and_revocation_invalidates_publication(context, monkeypatch):
    authorization = _authorize(context)
    assert authorization["authorized_at"] == "2026-10-10T00:00:00.000000Z"
    monkeypatch.setattr(context.clock, "now", lambda: datetime(2026, 10, 10, 0, 1, tzinfo=timezone.utc))
    _state_assent(monkeypatch, context, "START")
    started = _start(context)
    assert started["state"] == "read_started"
    assert started["fence"] == 1
    assert started["owner_pid"] == os.getpid()
    assert started["owner_process_group"] == os.getpgid(os.getpid())
    assert authority.final_holdout_fence_current(
        context.conn, context.request.experiment_id, started["request_digest"], started["fence"])
    assert not authority.final_holdout_fence_current(
        context.conn, context.request.experiment_id, content_hash("wrong request"), started["fence"])
    assert not authority.final_holdout_fence_current(
        context.conn, context.request.experiment_id, started["request_digest"], started["fence"] + 1)
    with pytest.raises(OpsError, match="HOLDOUT_ACCESS_DENIED"):
        _start(context)
    assert len(_state_events(context)) == 1
    monkeypatch.setattr(context.clock, "now", lambda: datetime(2026, 10, 10, 0, 2, tzinfo=timezone.utc))
    _state_assent(monkeypatch, context, "REVOKE")
    revoked = _revoke(context)
    assert revoked["state"] == "revoked"
    assert revoked["fence"] == 2
    events = _state_events(context)
    assert [(row["actor"], row["occurred_at"], row["confirmation"]) for row in events] == [
        ("1234:synthetic-user", "2026-10-10T00:01:00.000000Z", _state_phrase(context, "START")),
        ("1234:synthetic-user", "2026-10-10T00:02:00.000000Z", _state_phrase(context, "REVOKE")),
    ]
    assert not authority.final_holdout_fence_current(
        context.conn, context.request.experiment_id, started["request_digest"], started["fence"])
    with pytest.raises(OpsError, match="HOLDOUT_ACCESS_DENIED"):
        _start(context)
    assert [row["state"] for row in _state_events(context)] == ["read_started", "revoked"]
    assert len(_rows(context)) == 1


def test_revocation_before_start_and_state_history_are_permanent(context, monkeypatch):
    _authorize(context)
    _state_assent(monkeypatch, context, "REVOKE")
    revoked = _revoke(context)
    assert revoked["state"] == "revoked"
    _state_assent(monkeypatch, context, "START")
    with pytest.raises(OpsError, match="HOLDOUT_ACCESS_DENIED"):
        _start(context)
    for sql in (
        "DELETE FROM final_holdout_state_events",
        "UPDATE final_holdout_state_events SET state='read_started'",
    ):
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            context.conn.execute(sql)
    row = _state_events(context)[0]
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        context.conn.execute(
            "INSERT OR REPLACE INTO final_holdout_state_events "
            "(event_id,experiment_id,request_digest,state,fence,actor,occurred_at,confirmation) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (row["event_id"], row["experiment_id"], row["request_digest"], row["state"],
             row["fence"], row["actor"], row["occurred_at"], row["confirmation"]))
    assert [row["state"] for row in _state_events(context)] == ["revoked"]


def test_revoke_refuses_maximum_started_fence_without_mutation(context, monkeypatch):
    authorization = _authorize(context)
    identity, *_ = authority.process_info(os.getpid(), authority.read_boot_id())
    context.conn.execute(
        "INSERT INTO final_holdout_state_events "
        "(experiment_id,request_digest,state,fence,owner_boot_id,owner_pid,owner_start_ticks,"
        "owner_process_group,actor,occurred_at,confirmation) "
        "VALUES (?,?, 'read_started',?,?,?,?,?,?,?,?)",
        (context.request.experiment_id, authorization["request_digest"], 2**63 - 1,
         identity.boot_id, identity.pid, identity.start_ticks, identity.process_group,
         "1234:synthetic-user", "2026-10-10T00:00:00.000000Z", _state_phrase(context, "START")),
    )
    authorization_before = _rows(context)
    events_before = [dict(row) for row in _state_events(context)]
    _state_assent(monkeypatch, context, "REVOKE")
    with pytest.raises(OpsError) as exc_info:
        _revoke(context)
    assert exc_info.value.code == "VALIDATION_FAILED"
    assert _rows(context) == authorization_before
    assert [dict(row) for row in _state_events(context)] == events_before


def test_recovery_refuses_while_owner_is_alive(context, monkeypatch):
    _authorize(context)
    _state_assent(monkeypatch, context, "START")
    started = _start(context)
    with pytest.raises(OpsError, match="HOLDOUT_ACCESS_DENIED"):
        authority.recover_interrupted_final_holdout_read(
            context.conn, context.request.experiment_id, clock=context.clock)
    assert [row["state"] for row in _state_events(context)] == ["read_started"]


def test_recovery_rejects_revocation_between_death_check_and_commit(context, monkeypatch):
    _authorize(context)
    _state_assent(monkeypatch, context, "START")
    _start(context)
    monkeypatch.setattr(builtins, "input", lambda prompt: _state_phrase(context, "REVOKE"))

    def revoke_after_death_check(identity, boot_id):
        conn = connect(context.root / "catalog.sqlite", must_exist=True)
        try:
            revoked = authority.revoke_final_holdout(
                conn, context.store, context.request, clock=context.clock)
            assert revoked["state"] == "revoked"
        finally:
            conn.close()
        return False

    monkeypatch.setattr(authority, "is_alive", revoke_after_death_check)
    with pytest.raises(OpsError) as exc_info:
        authority.recover_interrupted_final_holdout_read(
            context.conn, context.request.experiment_id, clock=context.clock)
    assert exc_info.value.code == "STALE_EXPECTATION"
    events = _state_events(context)
    assert [row["state"] for row in events] == ["read_started", "revoked"]
    assert all(row["state"] != "indeterminate" for row in events)


def test_real_owner_exit_is_proven_dead_without_a_timeout(context, monkeypatch):
    authorization = _authorize(context)
    monkeypatch.setattr(context.clock, "now", lambda: datetime(2026, 10, 10, 0, 1, tzinfo=timezone.utc))
    _state_assent(monkeypatch, context, "START")
    ctx = multiprocessing.get_context("fork")

    def start_in_child():
        conn = connect(context.root / "catalog.sqlite", must_exist=True)
        try:
            authority.start_final_holdout_read(conn, context.store, context.request, clock=context.clock)
        finally:
            conn.close()

    child = ctx.Process(target=start_in_child)
    child.start()
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            waited = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            if waited is not None and waited.si_pid == child.pid:
                break
            time.sleep(0.01)
        else:
            pytest.fail("child did not exit before the deadline")
        assert authority.process_info(child.pid, authority.read_boot_id())[2] == "Z"
        started = _state_events(context)[0]
        assert started["owner_pid"] == child.pid
        assert started["occurred_at"] == "2026-10-10T00:01:00.000000Z"
        monkeypatch.setattr(context.clock, "now", lambda: datetime(2026, 10, 10, 0, 2, tzinfo=timezone.utc))
        recovered = authority.recover_interrupted_final_holdout_read(
            context.conn, context.request.experiment_id, clock=context.clock)
        assert recovered["state"] == "indeterminate"
        assert recovered["fence"] == started["fence"]
        assert recovered["request_digest"] == authorization["request_digest"]
        assert recovered["occurred_at"] == "2026-10-10T00:02:00.000000Z"
        assert not authority.final_holdout_fence_current(
            context.conn, context.request.experiment_id, started["request_digest"], started["fence"])
        _state_assent(monkeypatch, context, "START")
        with pytest.raises(OpsError, match="HOLDOUT_ACCESS_DENIED"):
            _start(context)
        _state_assent(monkeypatch, context, "REVOKE")
        with pytest.raises(OpsError, match="HOLDOUT_ACCESS_DENIED"):
            _revoke(context)
        with pytest.raises(OpsError, match="STALE_EXPECTATION"):
            authority.recover_interrupted_final_holdout_read(
                context.conn, context.request.experiment_id, clock=context.clock)
        assert [row["state"] for row in _state_events(context)] == ["read_started", "indeterminate"]
    finally:
        child.join(20)
        if child.is_alive():
            child.terminate()
            child.join(5)
        if child.is_alive():
            child.kill()
            child.join(5)
    assert not child.is_alive()
    assert child.exitcode == 0


def test_concurrent_starts_serialize_to_one_fence(context, monkeypatch):
    _authorize(context)
    phrase = _state_phrase(context, "START")
    barrier = Barrier(2)

    def assent(prompt):
        barrier.wait(timeout=10)
        return phrase

    monkeypatch.setattr(builtins, "input", assent)

    def invoke():
        conn = connect(context.root / "catalog.sqlite", must_exist=True)
        try:
            return authority.start_final_holdout_read(
                conn, context.store, context.request, clock=context.clock)
        except OpsError as exc:
            return exc.code
        finally:
            conn.close()

    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: invoke(), range(2)))
    assert sum(isinstance(result, dict) and result["state"] == "read_started"
               for result in results) == 1
    assert results.count("HOLDOUT_ACCESS_DENIED") == 1
    assert len(_state_events(context)) == 1


def test_start_and_revoke_serialize_to_a_revoked_fence(context, monkeypatch):
    _authorize(context)
    barrier = Barrier(2)

    def assent(prompt):
        barrier.wait(timeout=10)
        action = "START" if "START" in prompt else "REVOKE"
        return _state_phrase(context, action)

    monkeypatch.setattr(builtins, "input", assent)

    def invoke(action):
        conn = connect(context.root / "catalog.sqlite", must_exist=True)
        try:
            if action == "START":
                return authority.start_final_holdout_read(
                    conn, context.store, context.request, clock=context.clock)
            return authority.revoke_final_holdout(
                conn, context.store, context.request, clock=context.clock)
        except OpsError as exc:
            return exc.code
        finally:
            conn.close()

    with ThreadPoolExecutor(2) as pool:
        start_future = pool.submit(invoke, "START")
        revoke_future = pool.submit(invoke, "REVOKE")
        started, revoked = start_future.result(), revoke_future.result()
    states = [row["state"] for row in _state_events(context)]
    assert states in (["revoked"], ["read_started", "revoked"])
    assert isinstance(revoked, dict) and revoked["state"] == "revoked"
    assert revoked["fence"] == (2 if isinstance(started, dict) else 1)
    if isinstance(started, dict):
        assert started["state"] == "read_started"
        assert not authority.final_holdout_fence_current(
            context.conn, context.request.experiment_id, started["request_digest"], started["fence"])
    else:
        assert started == "HOLDOUT_ACCESS_DENIED"


def test_boot_change_proves_the_old_process_dead(context, monkeypatch):
    authorization = _authorize(context)
    identity, *_ = authority.process_info(os.getpid(), "synthetic-old-boot")
    context.conn.execute(
        "INSERT INTO final_holdout_state_events "
        "(experiment_id,request_digest,state,fence,owner_boot_id,owner_pid,owner_start_ticks,"
        "owner_process_group,actor,occurred_at,confirmation) "
        "VALUES (?,?, 'read_started',?,?,?,?,?,?,?,?)",
        (context.request.experiment_id, authorization["request_digest"], 37,
         identity.boot_id, identity.pid, identity.start_ticks, identity.process_group,
         "1234:synthetic-user", "2026-10-10T00:01:00.000000Z", _state_phrase(context, "START")),
    )
    monkeypatch.setattr(context.clock, "now", lambda: datetime(2026, 10, 10, 0, 2, tzinfo=timezone.utc))
    monkeypatch.setattr(authority, "read_boot_id", lambda: "synthetic-new-boot")
    monkeypatch.setattr(authority, "is_alive", lambda *args: pytest.fail("old-boot identity checked in new boot"))
    recovered = authority.recover_interrupted_final_holdout_read(
        context.conn, context.request.experiment_id, clock=context.clock)
    assert recovered["state"] == "indeterminate"
    assert recovered["request_digest"] == authorization["request_digest"]
    assert recovered["occurred_at"] == "2026-10-10T00:02:00.000000Z"
    events = _state_events(context)
    assert events[0]["fence"] == 37
    assert recovered["fence"] == 37
    assert [row["state"] for row in events] == ["read_started", "indeterminate"]
    assert [row["fence"] for row in events] == [37, 37]


@pytest.mark.parametrize(("field", "invalid_value"), [
    ("boot_id", ""),
    ("pid", 0), ("pid", -1), ("pid", True), ("pid", 1.5), ("pid", "1"),
    ("start_ticks", 0), ("start_ticks", -1), ("start_ticks", True),
    ("start_ticks", 1.5), ("start_ticks", "1"),
    ("process_group", 0), ("process_group", -1), ("process_group", True),
    ("process_group", 1.5), ("process_group", "1"),
])
def test_start_refuses_malformed_owner_identity_before_spending(context, monkeypatch, field, invalid_value):
    authorization = _authorize(context)
    _state_assent(monkeypatch, context, "START")
    identity = {"boot_id": "synthetic-boot", "pid": os.getpid(),
                "start_ticks": 1, "process_group": os.getpgrp()}
    identity[field] = invalid_value
    monkeypatch.setattr(
        authority, "process_info",
        lambda pid, boot_id: (SimpleNamespace(**identity), None),
    )
    with pytest.raises(OpsError) as exc_info:
        _start(context)
    assert exc_info.value.code == "VALIDATION_FAILED"
    assert _state_events(context) == []
    assert _rows(context) == [authorization]


@pytest.mark.parametrize(("experiment_id", "request_digest", "fence"), [
    (1 << 100, content_hash("request"), 1),
    ("  ", content_hash("request"), 1),
    (" padded ", content_hash("request"), 1),
    ("experiment", 1 << 100, 1),
    ("experiment", "malformed", 1),
    ("experiment", "a" * 64, 1),
    ("experiment", "sha256:" + "a" * 63, 1),
    ("experiment", "sha256:" + "a" * 65, 1),
    ("experiment", "sha256:" + "A" * 64, 1),
    ("experiment", content_hash("request"), 0),
    ("experiment", content_hash("request"), -1),
    ("experiment", content_hash("request"), True),
    ("experiment", content_hash("request"), 1.0),
    ("experiment", content_hash("request"), 2**63),
])
def test_publication_fence_rejects_malformed_arguments_before_lookup(
        context, monkeypatch, experiment_id, request_digest, fence):
    monkeypatch.setattr(authority, "_latest_event", lambda *args: pytest.fail("invalid input reached SQLite lookup"))
    with pytest.raises(OpsError) as exc_info:
        authority.final_holdout_fence_current(context.conn, experiment_id, request_digest, fence)
    assert exc_info.value.code == "VALIDATION_FAILED"


def test_publication_fence_accepts_maximum_valid_fence(context):
    authorization = _authorize(context)
    identity, *_ = authority.process_info(os.getpid(), authority.read_boot_id())
    context.conn.execute(
        "INSERT INTO final_holdout_state_events "
        "(experiment_id,request_digest,state,fence,owner_boot_id,owner_pid,owner_start_ticks,"
        "owner_process_group,actor,occurred_at,confirmation) "
        "VALUES (?,?, 'read_started',?,?,?,?,?,?,?,?)",
        (context.request.experiment_id, authorization["request_digest"], 2**63 - 1,
         identity.boot_id, identity.pid, identity.start_ticks, identity.process_group,
         "1234:synthetic-user", "2026-10-10T00:00:00.000000Z", "synthetic confirmation"),
    )
    assert authority.final_holdout_fence_current(
        context.conn, context.request.experiment_id, authorization["request_digest"], 2**63 - 1) is True


@pytest.mark.parametrize("experiment_id", [None, "", "  ", " padded ", 1 << 100])
def test_recovery_rejects_malformed_experiment_id_before_lookup(context, monkeypatch, experiment_id):
    monkeypatch.setattr(authority, "_latest_event", lambda *args: pytest.fail("invalid input reached SQLite lookup"))
    with pytest.raises(OpsError) as exc_info:
        authority.recover_interrupted_final_holdout_read(
            context.conn, experiment_id, clock=context.clock)
    assert exc_info.value.code == "VALIDATION_FAILED"


@pytest.mark.parametrize(("action", "experiment_id"), [
    ("START", "  "), ("START", " padded "),
    ("REVOKE", "  "), ("REVOKE", " padded "),
])
def test_state_actions_reject_noncanonical_experiment_ids(context, action, experiment_id):
    authorization = _authorize(context)
    request = replace(context.request, experiment_id=experiment_id)
    with pytest.raises(OpsError) as exc_info:
        if action == "START":
            authority.start_final_holdout_read(context.conn, context.store, request, clock=context.clock)
        else:
            authority.revoke_final_holdout(context.conn, context.store, request, clock=context.clock)
    assert exc_info.value.code == "VALIDATION_FAILED"
    assert _state_events(context) == []
    assert _rows(context) == [authorization]


def test_publication_fence_storage_failure_is_typed(monkeypatch, context):
    def unavailable(*args):
        raise sqlite3.OperationalError("synthetic storage failure")

    monkeypatch.setattr(authority, "_latest_event", unavailable)
    with pytest.raises(OpsError, match="RESOURCE_UNAVAILABLE"):
        authority.final_holdout_fence_current(
            context.conn, context.request.experiment_id, content_hash("request"), 1)


@pytest.mark.parametrize(("column", "value"), [
    ("confirmation", "tampered confirmation"), ("authorized_at", "not a timestamp"),
])
def test_start_rejects_corrupt_authorization_evidence(context, monkeypatch, column, value):
    _authorize(context)
    context.conn.execute("DROP TRIGGER final_holdout_authorizations_no_update")
    context.conn.execute(f"UPDATE final_holdout_authorizations SET {column}=?", (value,))
    _state_assent(monkeypatch, context, "START")
    with pytest.raises(OpsError, match="VALIDATION_FAILED"):
        _start(context)
    assert _state_events(context) == []


@pytest.mark.parametrize(("state", "missing_field"), [
    ("read_started", "owner_boot_id"),
    ("read_started", "owner_pid"),
    ("read_started", "owner_start_ticks"),
    ("read_started", "owner_process_group"),
    ("read_started", "actor"),
    ("read_started", "confirmation"),
    ("revoked", "actor"),
    ("revoked", "confirmation"),
])
def test_state_schema_requires_each_owner_and_assent_field(context, state, missing_field):
    authorization = _authorize(context)
    identity, *_ = authority.process_info(os.getpid(), authority.read_boot_id())
    values = {
        "experiment_id": context.request.experiment_id,
        "request_digest": authorization["request_digest"],
        "state": state,
        "fence": 1,
        "owner_boot_id": identity.boot_id,
        "owner_pid": identity.pid,
        "owner_start_ticks": identity.start_ticks,
        "owner_process_group": identity.process_group,
        "actor": "1234:synthetic-user",
        "occurred_at": "2026-10-10T00:00:00.000000Z",
        "confirmation": "synthetic confirmation",
    }
    values.pop(missing_field)
    columns = tuple(values)
    with pytest.raises(sqlite3.IntegrityError, match=missing_field):
        context.conn.execute(
            f"INSERT INTO final_holdout_state_events ({', '.join(columns)}) "
            f"VALUES ({', '.join('?' for _ in columns)})", tuple(values.values()))
    assert _state_events(context) == []


@pytest.mark.parametrize(("action", "mode"), [
    ("START", "noninteractive"), ("REVOKE", "noninteractive"),
    ("START", "wrong_uid"), ("REVOKE", "wrong_uid"),
    ("START", "wrong_phrase"), ("REVOKE", "wrong_phrase"),
])
def test_state_assent_denials_do_not_append(context, monkeypatch, action, mode):
    _authorize(context)
    if mode == "noninteractive":
        monkeypatch.setattr(authority.sys, "stdin", io.StringIO())
    elif mode == "wrong_uid":
        monkeypatch.setattr(authority.os, "getuid", lambda: 9876)
    else:
        monkeypatch.setattr(builtins, "input", lambda prompt: "wrong")
    with pytest.raises(OpsError, match="HOLDOUT_ACCESS_DENIED"):
        if action == "START":
            _start(context)
        else:
            _revoke(context)
    assert _state_events(context) == []


def test_start_rejects_foreign_digest_in_persisted_history(context, monkeypatch):
    authorization = _authorize(context)
    identity, *_ = authority.process_info(os.getpid(), authority.read_boot_id())
    context.conn.execute(
        "INSERT INTO final_holdout_state_events "
        "(experiment_id,request_digest,state,fence,owner_boot_id,owner_pid,owner_start_ticks,"
        "owner_process_group,actor,occurred_at,confirmation) "
        "VALUES (?,?, 'read_started',1,?,?,?,?,?,?,?)",
        (context.request.experiment_id, content_hash("another request"), identity.boot_id,
         identity.pid, identity.start_ticks, identity.process_group, "1234:synthetic-user",
         "2026-10-10T00:00:00.000000Z", "synthetic confirmation"),
    )
    _state_assent(monkeypatch, context, "START")
    with pytest.raises(OpsError, match="VALIDATION_FAILED"):
        _start(context)
    assert len(_state_events(context)) == 1
    assert _state_events(context)[0]["request_digest"] != authorization["request_digest"]


@pytest.mark.parametrize(("column", "value"), [
    ("confirmation", "tampered confirmation"), ("authorized_at", "not a timestamp"),
])
def test_recovery_rejects_corrupt_authorization_evidence(context, monkeypatch, column, value):
    _authorize(context)
    _state_assent(monkeypatch, context, "START")
    _start(context)
    context.conn.execute("DROP TRIGGER final_holdout_authorizations_no_update")
    context.conn.execute(f"UPDATE final_holdout_authorizations SET {column}=?", (value,))
    monkeypatch.setattr(authority, "is_alive", lambda *args: pytest.fail("corrupt authorization reached owner check"))
    with pytest.raises(OpsError, match="VALIDATION_FAILED"):
        authority.recover_interrupted_final_holdout_read(
            context.conn, context.request.experiment_id, clock=context.clock)
    assert [row["state"] for row in _state_events(context)] == ["read_started"]


@pytest.mark.parametrize(("field", "invalid_value"), [
    ("owner_boot_id", ""),
    ("owner_pid", 0), ("owner_pid", -1),
    ("owner_start_ticks", 0), ("owner_start_ticks", -1),
    ("owner_process_group", 0), ("owner_process_group", -1),
])
def test_recovery_rejects_malformed_persisted_owner_identity(context, field, invalid_value):
    authorization = _authorize(context)
    identity, *_ = authority.process_info(os.getpid(), authority.read_boot_id())
    values = {
        "owner_boot_id": identity.boot_id,
        "owner_pid": identity.pid,
        "owner_start_ticks": identity.start_ticks,
        "owner_process_group": identity.process_group,
    }
    values[field] = invalid_value
    context.conn.execute(
        "INSERT INTO final_holdout_state_events "
        "(experiment_id,request_digest,state,fence,owner_boot_id,owner_pid,owner_start_ticks,"
        "owner_process_group,actor,occurred_at,confirmation) "
        "VALUES (?,?, 'read_started',1,?,?,?,?,?,?,?)",
        (context.request.experiment_id, authorization["request_digest"], values["owner_boot_id"],
         values["owner_pid"], values["owner_start_ticks"], values["owner_process_group"],
         "1234:synthetic-user", "2026-10-10T00:00:00.000000Z", "synthetic confirmation"),
    )
    original = dict(_state_events(context)[0])
    with pytest.raises(OpsError) as fence_error:
        authority.final_holdout_fence_current(
            context.conn, context.request.experiment_id, authorization["request_digest"], 1)
    assert fence_error.value.code == "VALIDATION_FAILED"
    with pytest.raises(OpsError) as recovery_error:
        authority.recover_interrupted_final_holdout_read(
            context.conn, context.request.experiment_id, clock=context.clock)
    assert recovery_error.value.code == "VALIDATION_FAILED"
    assert dict(_state_events(context)[0]) == original


@pytest.mark.parametrize(("field", "invalid_value"), [
    ("owner_boot_id", None),
    ("owner_pid", True),
    ("owner_start_ticks", 1.5),
    ("owner_process_group", "1"),
])
def test_reader_refuses_mistyped_owner_identity_without_changing_state(
        context, monkeypatch, field, invalid_value):
    authorization = _authorize(context)
    identity, *_ = authority.process_info(os.getpid(), authority.read_boot_id())
    context.conn.execute(
        "INSERT INTO final_holdout_state_events "
        "(experiment_id,request_digest,state,fence,owner_boot_id,owner_pid,owner_start_ticks,"
        "owner_process_group,actor,occurred_at,confirmation) "
        "VALUES (?,?, 'read_started',1,?,?,?,?,?,?,?)",
        (context.request.experiment_id, authorization["request_digest"], identity.boot_id,
         identity.pid, identity.start_ticks, identity.process_group, "1234:synthetic-user",
         "2026-10-10T00:00:00.000000Z", "synthetic confirmation"),
    )
    original = dict(_state_events(context)[0])
    malformed = dict(original)
    malformed[field] = invalid_value
    monkeypatch.setattr(authority, "_latest_event", lambda *args: malformed)
    with pytest.raises(OpsError) as fence_error:
        authority.final_holdout_fence_current(
            context.conn, context.request.experiment_id, authorization["request_digest"], 1)
    assert fence_error.value.code == "VALIDATION_FAILED"
    with pytest.raises(OpsError) as recovery_error:
        authority.recover_interrupted_final_holdout_read(
            context.conn, context.request.experiment_id, clock=context.clock)
    assert recovery_error.value.code == "VALIDATION_FAILED"
    assert dict(_state_events(context)[0]) == original
