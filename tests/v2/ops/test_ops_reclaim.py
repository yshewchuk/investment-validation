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
from engine.v2.ops.lifecycle import Outcome, commit_attempt, heartbeat
from engine.v2.ops.recovery import expire_leases
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
    real = reclaim_module.remove_tree

    def flaky(path, keepalive, on_progress=None):
        if world.ids["succeeded"] in str(path):
            raise PermissionError("denied")
        real(path, keepalive, on_progress=on_progress)

    monkeypatch.setattr(reclaim_module, "remove_tree", flaky)
    entries = reclaim(world.conn, world.root, apply=True)
    by_id = {entry["attempt_id"]: entry for entry in entries}
    assert by_id[world.ids["succeeded"]]["error"] == "PermissionError"
    assert by_id[world.ids["failed"]]["removed"] is True
    assert _legacy(world, "succeeded").is_dir()


def _service(world):
    renewals = []
    return types.SimpleNamespace(
        renewals=renewals, _renew_other_leases=lambda: renewals.append(1),
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


def test_an_idle_pass_wakes_when_another_attempt_ends(world):
    service = _service(world)
    for _ in range(3):
        Service._reclaim_terminal_staging(service)
    assert service._reclaim_next_at > world.clock.monotonic()
    fence = world.conn.execute("SELECT fence FROM attempts WHERE attempt_id = ?",
                               (world.ids["starting"],)).fetchone()[0]
    commit_attempt(world.conn, world.ids["starting"], fence,
                   Outcome(True, "verified_dead", 0), clock=world.clock)
    Service._reclaim_terminal_staging(service)
    assert not _legacy(world, "starting").exists()


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


def test_removal_renews_leases_per_file_so_slow_cleanup_cannot_starve_them(world):
    (_legacy(world, "succeeded") / "data").chmod(0o755)
    for index in range(20):
        (_legacy(world, "succeeded") / "data" / f"extra{index}.csv").write_bytes(b"1")
    calls = []
    reclaim(world.conn, world.root, apply=True, keepalive=lambda: calls.append(1))
    assert len(calls) >= 21


def test_remove_tree_reports_progress_after_each_removal(tmp_path):
    root = tmp_path / "root"
    (root / "dirs" / "nested").mkdir(parents=True)
    (root / "a.bin").write_bytes(b"a")
    (root / "dirs" / "b.bin").write_bytes(b"b")
    (root / "dirs" / "nested" / "c.bin").write_bytes(b"c")
    progress, renewals = [], []

    reclaim_module.remove_tree(root, lambda: renewals.append(1),
                               on_progress=lambda done, total: progress.append((done, total)))

    assert progress == [(1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6)]
    assert len(renewals) >= 12
    assert not root.exists()


def test_the_tick_pass_passes_the_lease_renewal_into_reclaim(world):
    service = _service(world)
    Service._reclaim_terminal_staging(service)
    assert len(service.renewals) >= 2


def test_symlinked_roots_are_unlinked_never_traversed_or_sized(world, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "big.bin").write_bytes(b"y" * 500)
    reclaim(world.conn, world.root, apply=True)
    legacy = _legacy(world, "succeeded")
    legacy.symlink_to(outside, target_is_directory=True)
    base = default_materialization_base(world.root)
    base.mkdir()
    partial = base / f".{HEX}.partial-{world.ids['failed']}"
    partial.symlink_to(outside, target_is_directory=True)
    listed = reclaim(world.conn, world.root)
    assert len(listed) == 2 and all(entry["bytes"] == 0 for entry in listed)
    reclaim(world.conn, world.root, apply=True)
    assert not legacy.is_symlink() and not partial.is_symlink()
    assert (outside / "big.bin").read_bytes() == b"y" * 500


def _slow_reclaim(world, renew):
    (_legacy(world, "succeeded") / "data").chmod(0o755)
    for index in range(10):
        (_legacy(world, "succeeded") / "data" / f"extra{index}.csv").write_bytes(b"1")
    live = world.ids["starting"]
    fence = world.conn.execute("SELECT fence FROM attempts WHERE attempt_id = ?",
                               (live,)).fetchone()[0]
    world.clock.advance(60)
    heartbeat(world.conn, live, fence, clock=world.clock, lease_seconds=120)

    def keepalive():
        world.clock.advance(30)
        if renew:
            heartbeat(world.conn, live, fence, clock=world.clock, lease_seconds=120)

    reclaim(world.conn, world.root, apply=True, keepalive=keepalive)
    return expire_leases(world.conn, clock=world.clock), live


def test_a_slow_reclaim_with_renewal_keeps_a_running_attempts_lease_alive(world):
    expired, live = _slow_reclaim(world, renew=True)
    assert live not in expired


def test_the_same_slow_reclaim_without_renewal_would_expire_that_lease(world):
    expired, live = _slow_reclaim(world, renew=False)
    assert live in expired


def test_an_inaccessible_materialization_base_raises_but_a_missing_one_is_absent(
        world, monkeypatch):
    base = default_materialization_base(world.root)
    assert all(entry["kind"] == "staging" for entry in reclaim(world.conn, world.root))
    base.mkdir()
    real = type(base).iterdir

    def denied(path):
        if path == base:
            raise PermissionError("denied")
        return real(path)

    monkeypatch.setattr(type(base), "iterdir", denied)
    with pytest.raises(PermissionError):
        reclaim(world.conn, world.root)


def test_a_terminal_attempt_whose_staging_is_a_symlink_to_a_live_one_is_skipped(world):
    failed = world.root / "attempts" / world.ids["failed"]
    (failed / "staging").rename(failed / "moved")
    live_staging = world.root / "attempts" / world.ids["starting"] / "staging"
    (failed / "staging").symlink_to(live_staging, target_is_directory=True)
    dry = [e for e in reclaim(world.conn, world.root) if e["attempt_id"] == world.ids["failed"]]
    assert [(e["error"], e["bytes"]) for e in dry] == [("SymlinkedAncestor", 0)]
    entries = reclaim(world.conn, world.root, apply=True)
    refused = [e for e in entries if e["attempt_id"] == world.ids["failed"]]
    assert [e["error"] for e in refused] == ["SymlinkedAncestor"]
    assert (live_staging / "legacy" / "data" / "px.csv").read_bytes() == b"x" * 100
    assert (live_staging / "legacy" / "data").stat().st_mode & 0o777 == 0o555
    assert not _legacy(world, "succeeded").exists()


def test_a_symlinked_materialization_base_is_refused_not_followed(world, tmp_path):
    target = tmp_path / "elsewhere"
    partial = target / f".{HEX}.partial-{world.ids['failed']}"
    (partial / "tables").mkdir(parents=True)
    (partial / "tables" / "t.csv").write_bytes(b"1")
    base = default_materialization_base(world.root)
    base.symlink_to(target, target_is_directory=True)
    entries = [e for e in reclaim(world.conn, world.root, apply=True)
               if e["kind"] == "partial_materialization"]
    assert [e["error"] for e in entries] == ["SymlinkedAncestor"]
    assert (partial / "tables" / "t.csv").is_file()


def test_lease_renewal_immediately_before_unlink(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    (root / "file.txt").touch()

    events = []
    real_unlink = reclaim_module.os.unlink

    def mock_unlink(path, *args, **kwargs):
        events.append("unlink")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(reclaim_module.os, "unlink", mock_unlink)
    reclaim_module.remove_tree(root, lambda: events.append("renew"))

    assert events == ["renew", "renew", "unlink", "renew", "renew", "renew"]