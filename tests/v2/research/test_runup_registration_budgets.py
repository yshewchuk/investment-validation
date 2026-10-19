"""RUNUP registration propagates the shared finite SQL/artifact evidence bound."""
import json

import pytest

from engine.v2.foundation import canonical_json
from engine.v2.ops.catalog import connect
from experiments import native_registration as native
from experiments import registered_runup as study
from experiments import v2_runup_study as cli
from tests.data_scan_support import catalog_and_store
from tests.v2.ops.native_bounds_support import BoundedSqlRows, catalog_rows, object_bytes
from tests.v2.ops.test_native_registration_read_bounds import SPEC, pad, refuses, registered_catalog
from tests.v2.research import test_registered_runup as support
from tests.v2.research.magnitude_trial_support import TrackingRepository
from tests.v2.research.test_runup_exit_report import EVENT_ID, commit


@pytest.fixture
def source(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    snapshot = commit((conn, clock, store))
    yield conn, clock, store, TrackingRepository(conn, store), snapshot
    conn.close()


def prepare(source, **changes):
    return native.prepare_native_registration(source[3], support.specification(), code_root=study.ROOT,
        **{"snapshot": source[4], "scope": "shadow", "as_of_month": "2025-01",
           "event_ids": [EVENT_ID], **changes})


INVALID = [None, 0, -1, True, False, 1.0, "1", float("nan"), float("inf"), 2**62]


@pytest.mark.parametrize("budget", INVALID)
@pytest.mark.parametrize("action", [study.register_runup, study.run_runup])
def test_public_budget_rejected_before_storage_or_preparation(monkeypatch, budget, action):
    monkeypatch.setattr(study, "_prepare", support.forbid)
    with refuses("registration read requires a positive SQLite-safe byte budget", "INVALID_EXPERIMENT_SPEC"):
        action(None, None, None, support.specification(), snapshot=None, scope="shadow",
               as_of_month="2025-01", event_ids=[EVENT_ID], max_evidence_bytes=budget)


@pytest.mark.parametrize("budget", INVALID[1:])
def test_reservation_validates_optional_budget_before_io(budget):
    with refuses("registration read requires a positive SQLite-safe byte budget", "INVALID_EXPERIMENT_SPEC"):
        native.reserve_native_registration(None, None, None, None, max_evidence_bytes=budget)


def test_new_binding_exact_utf8_limit_and_minus_one_before_effects(source, monkeypatch):
    scope = "déclaré"
    registration, _ = prepare(source, scope=scope)
    size = len(registration.binding_json)
    assert size > len(registration.binding_json.decode())
    before_rows, before_objects = catalog_rows(source[0]), object_bytes(source[2])
    with monkeypatch.context() as patch:
        patch.setattr(source[2], "publish_bytes", support.forbid)
        with refuses("native registration exceeds its UTF-8 byte budget"):
            support.invoke(source, study.register_runup, scope=scope, max_evidence_bytes=size - 1)
    assert catalog_rows(source[0]) == before_rows
    assert object_bytes(source[2]) == before_objects
    assert support.invoke(source, study.register_runup, scope=scope, max_evidence_bytes=size) == registration
    before_rows, before_objects = catalog_rows(source[0]), object_bytes(source[2])
    monkeypatch.setattr(source[2], "publish_bytes", support.forbid)
    assert support.invoke(source, study.register_runup, scope=scope, max_evidence_bytes=size) == registration
    monkeypatch.setattr(source[2], "read_verified", support.forbid)
    with refuses("native registration exceeds its UTF-8 byte budget"):
        support.invoke(source, study.register_runup, scope=scope, max_evidence_bytes=size - 1)
    assert catalog_rows(source[0]) == before_rows
    assert object_bytes(source[2]) == before_objects


def test_binding_and_catalog_preflight_precede_all_io():
    class NoDecode(native.NativeRegistration):
        @property
        def document(self):
            pytest.fail("oversized prepared bytes were decoded")
    with refuses("native registration exceeds its UTF-8 byte budget"):
        native.reserve_native_registration(None, None, NoDecode("x", b"x" * 2), None, max_evidence_bytes=1)
    binding = canonical_json({"schema_version": "native_experiment_registration.v1.0"}).encode()
    with refuses("native evidence exceeds its canonical object contract"):
        native.reserve_native_registration(None, None, native.NativeRegistration("x", binding), None,
                                            max_evidence_bytes=len(binding))


@pytest.mark.parametrize("encoding", ["UTF-8", "UTF-16le", "UTF-16be"])
@pytest.mark.parametrize("extra", [0, 1])
def test_existing_reservation_bounds_sql_before_transfer_decode_or_artifact(tmp_path, monkeypatch, encoding, extra):
    cap = 2048
    with registered_catalog(tmp_path, encoding=encoding) as (conn, store, registration, _, evidence):
        payload = canonical_json(pad(evidence, cap + extra, "漢"))
        conn.execute("UPDATE experiment_runs SET evidence_json=?", (payload,))
        before = catalog_rows(conn)
        sql_cap = cap * (2 if encoding.startswith("UTF-16") else 1)
        bounded = BoundedSqlRows(conn, max_bytes=sql_cap, column="evidence_json", sql_fragment="FROM experiment_runs")
        monkeypatch.setattr(store, "publish_bytes", support.forbid)
        if extra:
            monkeypatch.setattr(store, "read_verified", support.forbid)
            with refuses("native evidence requires bounded JSON object text"):
                native.reserve_native_registration(bounded, store, registration, SPEC, max_evidence_bytes=cap)
        else:
            assert native.reserve_native_registration(bounded, store, registration, SPEC,
                                                       max_evidence_bytes=cap) == registration
        size = len(payload.encode(encoding))
        assert bounded.sizes == [None if size > sql_cap else size]
        assert catalog_rows(conn) == before


def test_re_registration_after_terminal_enforces_budget_without_publication(source, monkeypatch):
    registration = support.invoke(source, study.register_runup)
    support.invoke(source, study.run_runup)
    cap = len(registration.binding_json) + 1024
    evidence = support.evidence(source)[0]
    source[0].execute("UPDATE experiment_runs SET evidence_json=?", (canonical_json(pad(evidence, cap, "漢")),))
    before = catalog_rows(source[0])
    monkeypatch.setattr(source[2], "publish_bytes", support.forbid)
    assert support.invoke(source, study.register_runup, max_evidence_bytes=cap) == registration
    monkeypatch.setattr(source[2], "read_verified", support.forbid)
    with refuses("native evidence requires bounded JSON object text"):
        support.invoke(source, study.register_runup, max_evidence_bytes=cap - 1)
    assert catalog_rows(source[0]) == before


def test_transactional_recheck_caps_concurrently_created_evidence(source, monkeypatch):
    registration, spec = prepare(source)
    cap = len(registration.binding_json)
    catalog = source[0].execute("PRAGMA database_list").fetchone()[2]
    competitor = connect(catalog, must_exist=True)
    original_publish, competing = source[2].publish_bytes, False
    before_competitor = None
    def publish(payload, *, schema_ref):
        nonlocal competing, before_competitor
        ref = original_publish(payload, schema_ref=schema_ref)
        if not competing:
            competing = True
            native.reserve_native_registration(competitor, source[2], registration, spec)
            evidence = json.loads(competitor.execute("SELECT evidence_json FROM experiment_runs").fetchone()[0])
            competitor.execute("UPDATE experiment_runs SET evidence_json=?",
                               (canonical_json(pad(evidence, cap + 1, "x")),))
            before_competitor = catalog_rows(competitor)
        return ref
    monkeypatch.setattr(source[2], "publish_bytes", publish)
    bounded = BoundedSqlRows(source[0], max_bytes=cap, column="evidence_json", sql_fragment="FROM experiment_runs")
    try:
        with refuses("native evidence requires bounded JSON object text"):
            native.reserve_native_registration(bounded, source[2], registration, spec, max_evidence_bytes=cap)
        assert bounded.sizes == [None]  # First lookup absent; transactional row withheld.
        assert catalog_rows(source[0]) == before_competitor
    finally:
        competitor.close()


@pytest.mark.parametrize("budget", [None, native.MAX_NATIVE_EVIDENCE_BYTES])
def test_default_runup_registration_always_forwards_finite_budget(source, monkeypatch, budget):
    reserve, seen = native.reserve_native_registration, []
    def observe(*args, **kwargs):
        seen.append(kwargs["max_evidence_bytes"])
        return reserve(*args, **kwargs)
    monkeypatch.setattr(native, "reserve_native_registration", observe)
    options = {} if budget is None else {"max_evidence_bytes": budget}
    support.invoke(source, study.register_runup, **options)
    assert seen == [16 * 1024 * 1024]


def test_cli_register_budget_and_no_ledger_for_both_actions(source, tmp_path, capsys):
    registration, _ = prepare(source)
    cap = len(registration.binding_json)
    path = tmp_path / "spec.json"
    path.write_text(canonical_json(vars(support.specification())))
    catalog = source[0].execute("PRAGMA database_list").fetchone()[2]
    args = ["--spec", str(path), "--catalog", catalog, "--store-root", str(source[2].root),
            "--snapshot-id", source[4].snapshot_id, "--scope", "shadow", "--as-of-month", "2025-01",
            "--event-id", EVENT_ID, "--no-ledger"]
    assert cli.main(["register", *args, "--max-evidence-bytes", str(cap - 1)]) == 2
    assert json.loads(capsys.readouterr().out) == {"refused": "EXPERIMENT_IDENTITY_CONFLICT", "details": {}}
    assert support.evidence(source) == []
    assert cli.main(["register", *args, "--max-evidence-bytes", str(cap)]) == 0
    assert json.loads(capsys.readouterr().out) == {"registered": registration.variant_id, "run_id": registration.run_id}
    assert cli.main(["run", *args]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["receipt"]["recording_mode"] == "smoke"
    assert result["receipt"]["ledger_destination"] is None


@pytest.mark.parametrize("budget", ["0", "-1", "nan", "inf", str(2**62)])
def test_cli_invalid_budget_is_parser_refusal_before_spec_or_storage(monkeypatch, capsys, budget):
    monkeypatch.setattr(cli, "_spec", support.forbid)
    args = ["register", "--spec", "missing", "--catalog", "missing", "--store-root", "missing",
            "--snapshot-id", "missing", "--scope", "shadow", "--as-of-month", "2025-01",
            "--event-id", EVENT_ID, "--max-evidence-bytes", budget]
    with pytest.raises(SystemExit) as caught:
        cli.main(args)
    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "--max-evidence-bytes" in captured.err
