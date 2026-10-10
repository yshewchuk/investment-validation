"""Attempt staging reclaim (#609, #467) on a real catalog and real attempt dirs."""
from __future__ import annotations

import types
from argparse import Namespace

import pytest

import engine.v2.ops.reclaim as reclaim_module
import engine.v2.ops.supervisor as supervisor_module
from engine.v2.ops import cli
from engine.v2.ops.errors import make_problem
from engine.v2.ops.legacy_adapter import overlay_read_set
from engine.v2.ops.lifecycle import Outcome, commit_attempt
from engine.v2.ops.reclaim import reclaim
from engine.v2.ops.snapshot_roots import default_materialization_base
from engine.v2.ops.supervisor import Service
from tests.ops_support import catalog
from tools.v2_ops_fixtures import enqueue_claim

HEX = "a" * 64


def _stage(root, attempt_id):
    staging = root / "attempts" / attempt_id / "staging"
    (staging / "legacy" / "data").mkdir(parents=True)
    (staging / "legacy" / "data" / "px.csv").write_bytes(b"x" * 100)
    (staging / "legacy" / "data" / "px.csv").chmod(0o444)
    (staging / "legacy" / "data").chmod(0o555)
    (staging / "diagnostics").mkdir()
    (staging / "diagnostics" / "failure_details.json").write_text("{}")
    (staging / "out.json").write_text("{}")
    return staging


def _end(conn, clock, claim, ok):
    failure = None if ok else make_problem("VALIDATION_FAILED", "x")
    commit_attempt(conn, claim.attempt_id, claim.fence,
                   Outcome(ok, "verified_dead", 0 if ok else 1, failure), clock=clock)


@pytest.fixture
def world(tmp_path):
    root = tmp_path / "ops"
    root.mkdir()
    conn, clock, supervisor = catalog(root)
    ids = {}
    for name, ok in (("succeeded", True), ("failed", False)):
        claim = enqueue_claim(conn, clock, supervisor, key=name)
        _end(conn, clock, claim, ok)
        ids[name] = claim.attempt_id
    claim = enqueue_claim(conn, clock, supervisor, key="quarantined")
    _end(conn, clock, claim, False)
    conn.execute("UPDATE attempts SET process_state = 'quarantined' WHERE attempt_id = ?",
                 (claim.attempt_id,))
    ids["quarantined"] = claim.attempt_id
    claim = enqueue_claim(conn, clock, supervisor, key="live")
    ids["starting"] = claim.attempt_id
    ids["unknown"] = "att_not_in_catalog"
    for attempt_id in ids.values():
        _stage(root, attempt_id)
    yield types.SimpleNamespace(root=root, conn=conn, clock=clock, ids=ids)
    conn.close()


def _legacy(world, name):
    return world.root / "attempts" / world.ids[name] / "staging" / "legacy"


def test_dry_run_lists_only_settled_terminal_attempts_and_removes_nothing(world):
    entries = reclaim(world.conn, world.root)
    assert sorted(entry["attempt_id"] for entry in entries) == sorted(
        [world.ids["succeeded"], world.ids["failed"]])
    assert all(entry["bytes"] == 100 and entry["removed"] is False for entry in entries)
    assert all(_legacy(world, name).is_dir() for name in world.ids)


def test_apply_removes_legacy_only_and_is_idempotent(world):
    entries = reclaim(world.conn, world.root, apply=True)
    assert all(entry["removed"] for entry in entries) and len(entries) == 2
    for name in ("succeeded", "failed"):
        staging = _legacy(world, name).parent
        assert not _legacy(world, name).exists()
        assert (staging / "diagnostics" / "failure_details.json").is_file()
        assert (staging / "out.json").is_file()
    for name in ("quarantined", "starting", "unknown"):
        assert (_legacy(world, name) / "data" / "px.csv").is_file()
    assert reclaim(world.conn, world.root, apply=True) == []
    assert world.conn.execute("SELECT state FROM attempts WHERE attempt_id = ?",
                              (world.ids["starting"],)).fetchone()[0] == "starting"


def test_a_half_removed_tree_is_finished_by_the_next_pass(world):
    (_legacy(world, "succeeded") / "data").chmod(0o755)
    (_legacy(world, "succeeded") / "data" / "px.csv").unlink()
    entries = reclaim(world.conn, world.root, apply=True)
    assert len(entries) == 2 and not _legacy(world, "succeeded").exists()


def test_overlay_symlinks_are_removed_but_never_followed(world, tmp_path):
    source = tmp_path / "materialized"
    (source / "data").mkdir(parents=True)
    (source / "data" / "px.csv").write_bytes(b"pinned")
    (source / "data").chmod(0o555)
    source.chmod(0o555)
    try:
        reclaim(world.conn, world.root, apply=True)
        overlay_read_set(source, _legacy(world, "succeeded"))
        assert (_legacy(world, "succeeded") / "data" / "px.csv").is_symlink()
        reclaim(world.conn, world.root, apply=True)
        assert not _legacy(world, "succeeded").exists()
        assert (source / "data" / "px.csv").read_bytes() == b"pinned"
    finally:
        source.chmod(0o755)
        (source / "data").chmod(0o755)


def test_orphan_partial_roots_follow_the_same_rule(world):
    base = default_materialization_base(world.root)
    base.mkdir()
    done, live = world.ids["failed"], world.ids["starting"]
    for name in (f".{HEX}.partial-{done}", f".{HEX}.partial-{live}", HEX):
        (base / name / "tables").mkdir(parents=True)
        (base / name / "tables" / "t.csv").write_bytes(b"1")
        (base / name / "tables" / "t.csv").chmod(0o444)
        (base / name / "tables").chmod(0o555)
        (base / name).chmod(0o555)
    listed = [entry for entry in reclaim(world.conn, world.root)
              if entry["kind"] == "partial_materialization"]
    assert [entry["attempt_id"] for entry in listed] == [done]
    reclaim(world.conn, world.root, apply=True)
    assert not (base / f".{HEX}.partial-{done}").exists()
    assert (base / f".{HEX}.partial-{live}").is_dir() and (base / HEX).is_dir()


def test_a_failed_removal_is_reported_and_the_rest_still_run(world, monkeypatch):
    real = reclaim_module._remove

    def flaky(path):
        if world.ids["succeeded"] in str(path):
            raise PermissionError("denied")
        real(path)

    monkeypatch.setattr(reclaim_module, "_remove", flaky)
    entries = reclaim(world.conn, world.root, apply=True)
    by_id = {entry["attempt_id"]: entry for entry in entries}
    assert by_id[world.ids["succeeded"]]["error"] == "PermissionError"
    assert by_id[world.ids["failed"]]["removed"] is True
    assert _legacy(world, "succeeded").is_dir()


def _service(world):
    return types.SimpleNamespace(
        conn=world.conn, root=world.root, clock=world.clock,
        materialization_base=default_materialization_base(world.root),
        _reclaim_next_at=0.0, _reclaim_reported=set(),
        _RECLAIM_IDLE_SECONDS=Service._RECLAIM_IDLE_SECONDS)


def test_tick_pass_removes_one_entry_then_idles_and_never_touches_jobs(world):
    service = _service(world)
    before = [tuple(row) for row in world.conn.execute(
        "SELECT job_id, state FROM jobs ORDER BY job_id")]
    Service._reclaim_terminal_staging(service)
    assert sum(_legacy(world, name).exists() for name in ("succeeded", "failed")) == 1
    Service._reclaim_terminal_staging(service)
    assert not any(_legacy(world, name).exists() for name in ("succeeded", "failed"))
    Service._reclaim_terminal_staging(service)
    assert service._reclaim_next_at == world.clock.monotonic() + 30.0
    assert [tuple(row) for row in world.conn.execute(
        "SELECT job_id, state FROM jobs ORDER BY job_id")] == before


def test_tick_pass_survives_a_reclaim_failure_and_reports_it_once(world, monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise OSError("disk")

    monkeypatch.setattr(supervisor_module, "reclaim", boom)
    service = _service(world)
    Service._reclaim_terminal_staging(service)
    service._reclaim_next_at = 0.0
    Service._reclaim_terminal_staging(service)
    assert capsys.readouterr().out.count("reclaim_failed") == 1
    assert _legacy(world, "succeeded").is_dir()


def test_cli_is_a_dry_run_unless_apply_is_given(world):
    args = cli.parser().parse_args(["--root", str(world.root), "reclaim"])
    assert args.apply is False
    document = cli.reclaim_command(args, world.root, world.conn, world.clock)
    assert (document["dry_run"], document["count"], document["bytes"], document["removed"]) == (
        True, 2, 200, 0)
    applied = cli.reclaim_command(Namespace(apply=True), world.root, world.conn, world.clock)
    assert (applied["dry_run"], applied["removed"]) == (False, 2)