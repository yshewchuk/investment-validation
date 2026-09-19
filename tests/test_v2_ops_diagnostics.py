"""P2-C02 review fix: ``ops doctor`` crashed against any initialized catalog
because ``diagnostics.unmanaged_processes`` unpacked ``executor_watchdog.
process_info``'s rows as 4 values after commit e5053ac widened them to 5
(``(identity, ppid, state, rss_bytes, session)``). Also covers the CLI
catch-all's redacted exception-type detail.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.foundation import SystemClock  # noqa: E402
from engine.v2.ops import cli  # noqa: E402
from engine.v2.ops.bootstrap import open_catalog  # noqa: E402
from engine.v2.ops.diagnostics import unmanaged_processes  # noqa: E402
from engine.v2.ops.executor_watchdog import process_table  # noqa: E402


def test_process_table_rows_are_five_wide():
    """Pins the shape ``unmanaged_processes`` must unpack -- this is the
    fixture-free half of the regression: catches a re-widened/re-narrowed
    row before it ever reaches a catalog-backed test."""
    (row,) = list(process_table("boot").values())[:1] or [None]
    if row is not None:
        assert len(row) == 5


def test_unmanaged_processes_does_not_crash_on_a_real_process_table(tmp_path):
    """The exact call ``doctor`` makes: a real ``process_table`` against a
    real (empty) ``process_members`` table must not raise."""
    conn = open_catalog(tmp_path / "catalog.sqlite", clock=SystemClock())
    try:
        rows = unmanaged_processes(conn, boot_id="boot")
    finally:
        conn.close()
    assert isinstance(rows, list)
    # This process itself is alive right now, so the table is never empty --
    # the crash this fix closes only showed up once at least one row existed.
    assert any(row["pid"] for row in rows) or rows == []


def test_doctor_reports_unmanaged_processes_against_an_initialized_catalog(tmp_path):
    conn = open_catalog(tmp_path / "catalog.sqlite", clock=SystemClock())
    conn.close()
    result = cli.doctor(tmp_path, SystemClock())
    assert result["catalog_exists"] is True
    assert isinstance(result["diagnostics"]["unmanaged_processes"], list)


def test_doctor_json_cli_succeeds_against_an_initialized_catalog(tmp_path, capsys):
    conn = open_catalog(tmp_path / "catalog.sqlite", clock=SystemClock())
    conn.close()
    code = cli.main(["--root", str(tmp_path), "doctor", "--json"])
    assert code == 0
    document = json.loads(capsys.readouterr().out)
    assert document["catalog_exists"] is True
    assert isinstance(document["diagnostics"]["unmanaged_processes"], list)


def test_catchall_keeps_the_exception_class_name_without_its_text(tmp_path, monkeypatch, capsys):
    """The CLI's ``(OSError, ValueError, TypeError)`` catch-all must name
    WHICH kind of bug it caught (so a future regression like this one is
    diagnosable from the CLI's own output) while never printing the
    exception's own text -- it may carry a value or a path (§5.2)."""
    def _boom(root, clock):
        raise ValueError("some/sensitive/path or value that must never print")

    monkeypatch.setattr(cli, "doctor", _boom)
    code = cli.main(["--root", str(tmp_path), "doctor", "--json"])
    assert code == 2
    out = capsys.readouterr().out
    document = json.loads(out)
    assert document["code"] == "INVALID_REQUEST"
    assert document["details"]["exception_type"] == "ValueError"
    assert "sensitive" not in out


def test_serve_rejects_a_store_root_that_is_not_a_directory(tmp_path, capsys):
    conn = open_catalog(tmp_path / "catalog.sqlite", clock=SystemClock())
    conn.close()
    missing = tmp_path / "does-not-exist"
    code = cli.main(["--root", str(tmp_path), "serve", "--once", "--store-root", str(missing)])
    assert code == 2
    document = json.loads(capsys.readouterr().out)
    assert document["code"] == "INVALID_REQUEST"


# --------------------------------------------------------------------------
# --store-root: the code checkout is not always the legacy checkout
# --------------------------------------------------------------------------
#
# engine/v2/ops/cli.py's ``serve`` command built ``Service(..., code_source=
# Path(__file__).resolve().parents[3])`` with no ``store_root=`` -- Service
# defaults it to ``code_source``, so a snapshot-backed launch resolved its
# pinned legacy read set against the CODE checkout. From a frozen git
# worktree (no ``data/``), the read pin refuses INPUT_CHANGED before the
# job ever starts. legacy_decisions is the cheapest real legacy action
# (needs no market data — see tests/test_v2_ops_supervised_legacy.py's own
# docstring); DEFAULT_POLICY is patched down to TEST_POLICY's tiny memory
# footprint so this runs safely alongside other work on a shared host.


def _decisions_fixture(root, legacy_root):
    from engine.v2.contracts import LegacyFileRef
    from engine.v2.foundation import ArtifactStore, content_hash
    from engine.v2.ops.catalog import transaction
    from engine.v2.ops.fingerprints import file_hash
    from engine.v2.ledger.decisions import set_authority
    from tests.test_v2_ops_supervised_legacy import (
        SESSION, _decision_evidence, _manifest_ref, _publish, _submit_decisions,
    )

    clock = SystemClock()
    conn = open_catalog(root / "catalog.sqlite", clock=clock)
    store = ArtifactStore(root)
    fixture = legacy_root / "unused.txt"
    fixture.write_bytes(b"a legacy read-set member decisions never opens")
    manifest_ref = _manifest_ref(store, conn, clock, (LegacyFileRef(
        path="unused.txt", content_hash=file_hash(fixture), byte_size=fixture.stat().st_size),))
    score = {"ticker": "FAKE", "event_id": "event-1", "event_date": SESSION,
            "as_of": SESSION, "entry_date": SESSION, "evidence_cutoff": SESSION,
            "strategy": "TWIN-P", "strike": 100.0, "expiry": "2026-10-16",
            "session": "AMC", "snapshot_hash": "sha256:" + "a" * 64}
    score_ref = _publish(store, conn, clock, {"rows": [score]}, "legacy_action.v1.0")
    finality = {"date": SESSION, "is_final": True, "market_wide": True,
               "daily_share": 1.0, "chain_share": 1.0, "covered": 1}
    finality_ref = _publish(store, conn, clock, finality, "legacy_action.v1.0")
    plan = {"schema_version": "decision_plan.v1.0", "session": SESSION,
           "deployment": "shadow-deployment", "decision_clock": SESSION + "T21:00:00+00:00",
           "expected_population": ["FAKE|TWIN-P|" + SESSION]}
    plan_ref = _publish(store, conn, clock, plan, "decision_plan.v1.0")
    evidence = _decision_evidence(score_ref, finality_ref, plan_ref, score, finality, plan)
    evidence_ref = _publish(store, conn, clock, evidence, "decision_evidence.v1.0")
    with transaction(conn):
        set_authority(conn, None, "catalog", SESSION + "T20:00:00.000000Z")
    receipt = _submit_decisions(conn, manifest_ref=manifest_ref, score_ref=score_ref,
                                finality_ref=finality_ref, plan_ref=plan_ref,
                                evidence_ref=evidence_ref)
    conn.close()
    return receipt.job_id


def _job_state(root, job_id):
    """State and failure after ``serve --once``. A job the host could not
    admit fails here as ``RESOURCE WAIT`` with the queue reason's numbers,
    not as a bare ``'queued' == ...`` assertion further down."""
    from tests.ops_support import AdmissionWatch

    conn = open_catalog(root / "catalog.sqlite", clock=SystemClock())
    try:
        AdmissionWatch(conn, job_id).check(final=True)
        row = conn.execute("SELECT state, failure_json FROM jobs WHERE job_id=?",
                           (job_id,)).fetchone()
        return row["state"], row["failure_json"]
    finally:
        conn.close()


def test_serve_store_root_pins_a_legacy_root_outside_the_code_checkout(tmp_path, monkeypatch):
    from tests.ops_support import TEST_POLICY

    monkeypatch.setattr(cli, "DEFAULT_POLICY", TEST_POLICY)
    root = tmp_path / "ops"
    root.mkdir()
    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    job_id = _decisions_fixture(root, legacy_root)

    code = cli.main(["--root", str(root), "serve", "--once", "--store-root", str(legacy_root)])
    assert code == 0
    state, failure = _job_state(root, job_id)
    assert state == "succeeded", failure


def test_serve_without_store_root_still_refuses_the_read_set(tmp_path, monkeypatch):
    """The regression control: omitting ``--store-root`` keeps today's
    default (``code_source``, this repo checkout) -- ``unused.txt`` lives
    only under a tmp legacy root the code checkout knows nothing about, so
    the pinned read set still refuses exactly as before this fix."""
    from tests.ops_support import TEST_POLICY

    monkeypatch.setattr(cli, "DEFAULT_POLICY", TEST_POLICY)
    root = tmp_path / "ops"
    root.mkdir()
    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    job_id = _decisions_fixture(root, legacy_root)

    code = cli.main(["--root", str(root), "serve", "--once"])
    assert code == 0  # `serve` itself always exits clean; the JOB is what refuses
    state, failure = _job_state(root, job_id)
    assert state == "failed"
    assert "INPUT_CHANGED" in failure
