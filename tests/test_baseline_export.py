"""The baseline exporter: immutable publication, reproducibility, tree identity.

§6 acceptance: a second export from the same tree state and snapshot is
byte-identical. The 2026-09-12 review found no evidence the exporter had ever
been checked against itself, a package recorded against a dirty tree with no way
to say what the dirt was, and writes made in place. Each is planted here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import baseline_export as be  # noqa: E402
from engine import score  # noqa: E402
from engine.v2.diagnosis import content_hash  # noqa: E402


def files(value: str = "1") -> dict[str, str]:
    parts = {"definitions/a.json": json.dumps({"a": value}) + "\n",
             "requirements.txt": "numpy==2.5.1\n"}
    manifest = {"parts": {n: content_hash(t) for n, t in sorted(parts.items())},
                "package_hash": content_hash({n: content_hash(t) for n, t in sorted(parts.items())})}
    return parts | {"MANIFEST.json": json.dumps(manifest) + "\n"}


# --------------------------------------------------------------------------
# publication
# --------------------------------------------------------------------------


def test_publish_writes_the_version_and_points_current_at_it(tmp_path):
    out = be.publish(tmp_path, "v1", files())
    assert (out / "definitions" / "a.json").is_file()
    assert json.loads((tmp_path / "CURRENT").read_text()) == {"version": "v1"}
    assert not list(tmp_path.glob("*.tmp"))


def test_a_frozen_version_is_never_overwritten_without_force(tmp_path):
    be.publish(tmp_path, "v1", files("1"))
    with pytest.raises(SystemExit):
        be.publish(tmp_path, "v1", files("2"))
    assert '"1"' in (tmp_path / "v1" / "definitions" / "a.json").read_text()
    be.publish(tmp_path, "v1", files("2"), force=True)
    assert '"2"' in (tmp_path / "v1" / "definitions" / "a.json").read_text()


def test_a_forced_replacement_leaves_no_stale_file_behind(tmp_path):
    be.publish(tmp_path, "v1", files() | {"old/extra.json": "{}\n"})
    be.publish(tmp_path, "v1", files(), force=True)
    assert not (tmp_path / "v1" / "old" / "extra.json").exists()


# --------------------------------------------------------------------------
# reproducibility
# --------------------------------------------------------------------------


def test_verify_accepts_an_identical_rebuild_and_names_what_differs(tmp_path, monkeypatch):
    monkeypatch.setattr(be, "tree_state", lambda: {"sha": "abc", "dirty": False})
    out = be.publish(tmp_path, "v1", files("1"))
    assert be.verify(out, files("1"))["byte_identical"]
    changed = be.verify(out, files("2"))
    assert not changed["byte_identical"]
    assert "definitions/a.json" in changed["differing"]
    (out / "stray.json").write_text("{}")
    assert be.verify(out, files("1"))["extra"] == ["stray.json"]


def test_the_verify_receipt_names_the_package_it_verified(tmp_path, monkeypatch):
    monkeypatch.setattr(be, "tree_state", lambda: {"sha": "abc", "dirty": False})
    out = be.publish(tmp_path, "v1", files())
    receipt = be.write_verify_receipt(tmp_path, be.verify(out, files()))
    payload = json.loads(receipt.read_text())["payload"]
    manifest = json.loads((out / "MANIFEST.json").read_text())
    assert payload["package_hash"] == manifest["package_hash"]
    assert payload["byte_identical"] is True


# --------------------------------------------------------------------------
# tree identity
# --------------------------------------------------------------------------


def _fake_git(status: str):
    def _git(*args):
        if args[:1] == ("rev-parse",):
            return "abc123"
        if args[:1] == ("status",):
            return status
        return ""
    return _git


def test_the_tree_state_names_the_uncommitted_diff(monkeypatch):
    """Same sha, different uncommitted work: different states."""
    monkeypatch.setattr(be, "_git", _fake_git(" M engine/score.py"))
    monkeypatch.setattr(be, "_git_bytes", lambda *a: b"diff one")
    one = be.tree_state()
    monkeypatch.setattr(be, "_git_bytes", lambda *a: b"diff two")
    two = be.tree_state()
    assert one["dirty"] and one["sha"] == two["sha"]
    assert one["worktree_diff_hash"] != two["worktree_diff_hash"]


def test_a_clean_tree_has_no_diff_hash(monkeypatch):
    monkeypatch.setattr(be, "_git", _fake_git(""))
    monkeypatch.setattr(be, "_git_bytes", lambda *a: b"")
    assert be.tree_state()["worktree_diff_hash"] is None


# --------------------------------------------------------------------------
# exported definitions
# --------------------------------------------------------------------------


def test_the_bad_quote_threshold_is_exported_as_a_constant():
    constants = be.resolved_gates()["payload"]["constants"]
    assert constants["bad_quote_cost_pct"] == 30.0


@pytest.mark.parametrize("key", ["chooser_score", "exp_pnl_sim"])
@pytest.mark.parametrize("n", [2, 7])
def test_the_exported_tie_rule_is_the_measured_behaviour(key, n):
    """A DYN-SV tie is broken by input-row order on both ranking paths.

    `definitions/dyn_sv.json` states this; the statement is only as good as the
    measurement behind it, so the measurement is a test.
    """
    rows = [{"ticker": "X", "event_date": "2026-09-14", "strategy": strategy,
             "exp_pnl_sim": 0.01,
             "chooser_score": 0.5 if key == "chooser_score" else np.nan}
            for strategy in score.DYNAMIC_MENU[:n]]
    forward = score.dynamic_short_vol(pd.DataFrame(rows))["chosen_strategy"].iloc[0]
    backward = score.dynamic_short_vol(pd.DataFrame(rows[::-1]))["chosen_strategy"].iloc[0]
    assert forward == rows[0]["strategy"]
    assert backward == rows[-1]["strategy"]
    assert "INPUT-ROW ORDER" in be.resolved_dyn_sv()["payload"]["tie_behaviour"]
