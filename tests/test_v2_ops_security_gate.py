"""The publication security gate's real wiring: ``run_security_scan``
(``engine/v2/ops/legacy_adapter.py``) against a real subprocess and a real
tar bundle.

Covers what ``tests/test_repo_hygiene.py`` cannot, because ``check_bundle``
there is exercised in-process on a pure ``{path: bytes}`` mapping:

* the declared-file list actually comes from the render contract
  (``engine.dashboard.render.RENDERED_DATA_STEMS``), not a hand-copied name;
* ``repo_root`` (the code checkout the subprocess imports from) and
  ``env_root`` (the checkout the real ``.env`` is read from) are genuinely
  two different directories -- the whole point of the fix, since a
  snapshot-backed/frozen worktree run carries no ``.env``;
* the subprocess's JSON result carries ``secrets_loaded``.

No secret value is ever written, printed or asserted on here -- only a
made-up test needle and counts.
"""
from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

from engine.v2.ops.legacy_adapter import run_security_scan

REPO = Path(__file__).resolve().parents[1]


def _bundle_tar(tmp_path, members: dict) -> Path:
    """A real tar file (not in-memory) so ``run_security_scan``'s own
    ``tarfile.open`` subprocess call has a real path to open."""
    path = tmp_path / "bundle.tar"
    with tarfile.open(path, "w") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return path


def _fake_env_root(tmp_path) -> Path:
    root = tmp_path / "store"
    root.mkdir()
    (root / ".env").write_text("TEST_FAKE_SECRET=made-up-test-secret-value-0000\n")
    return root


def test_a_declared_render_file_over_1mb_passes(tmp_path):
    env_root = _fake_env_root(tmp_path)
    bundle = _bundle_tar(tmp_path, {"bundle/data/models.json": b"x" * 2_000_000})
    result = run_security_scan(bundle, REPO, env_root)
    assert result["ok"] is True, result["violations"]
    assert result["checked"] == 1
    assert result["secrets_loaded"] >= 1


def test_an_undeclared_file_over_1mb_refuses(tmp_path):
    env_root = _fake_env_root(tmp_path)
    bundle = _bundle_tar(tmp_path, {"bundle/data/unexpected.json": b"x" * 2_000_000})
    result = run_security_scan(bundle, REPO, env_root)
    assert result["ok"] is False
    assert any(v[1] == "oversize" for v in result["violations"])


def test_env_is_read_from_env_root_not_repo_root(tmp_path):
    # REPO (this worktree) has no .env; env_root does. A clean small bundle
    # must still see the env_root's needles, proving repo_root is only the
    # importable code checkout, never where secrets come from.
    assert not (REPO / ".env").exists()
    env_root = _fake_env_root(tmp_path)
    bundle = _bundle_tar(tmp_path, {"bundle/index.html": b"<html>clean</html>"})
    result = run_security_scan(bundle, REPO, env_root)
    assert result["ok"] is True, result["violations"]
    assert result["secrets_loaded"] >= 1


def test_missing_env_at_env_root_fails_closed(tmp_path):
    empty_root = tmp_path / "no_env"
    empty_root.mkdir()
    bundle = _bundle_tar(tmp_path, {"bundle/index.html": b"<html>clean</html>"})
    result = run_security_scan(bundle, REPO, empty_root)
    assert result["ok"] is False
    assert result["secrets_loaded"] == 0
    assert any(v[1] == "no-secrets-loaded" for v in result["violations"])


def test_env_root_defaults_to_repo_root_when_omitted(tmp_path):
    # No env_root passed at all -- falls back to repo_root, which for this
    # worktree has no .env, so this still fails closed rather than silently
    # scanning with zero needles.
    bundle = _bundle_tar(tmp_path, {"bundle/index.html": b"<html>clean</html>"})
    result = run_security_scan(bundle, REPO)
    assert result["ok"] is False
    assert result["secrets_loaded"] == 0


def test_a_planted_needle_inside_a_declared_file_still_refuses(tmp_path):
    env_root = _fake_env_root(tmp_path)
    bundle = _bundle_tar(tmp_path, {"bundle/data/models.json": b"TEST_FAKE_SECRET=made-up-test-secret-value-0000"})
    result = run_security_scan(bundle, REPO, env_root)
    assert result["ok"] is False
    assert any(v[1] == "secret" for v in result["violations"])
