"""tools.finality_probe: capture redacts keys, content hashes are order-stable,
and compare classifies false-final / true-not-final pairs.

# packages: foundation
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools import finality_probe  # noqa: E402

SENTINEL = "SENTINELKEY123"
ROWS = [
    {"updatedAt": 1700000000, "strike": 100},
    {"updatedAt": 1700000100, "strike": 101},
]


def _ns(out, label="L1", session="2024-01-02", endpoints="hist/strikes"):
    return argparse.Namespace(
        session=session,
        tickers="SPY",
        label=label,
        out=str(out),
        orats_endpoints=endpoints,
        massive=False,
        ignore_keys="",
    )


def _body(rows=None):
    return json.dumps({"data": rows if rows is not None else ROWS}).encode("utf-8")


def _fake_get(payload=None):
    def http_get(req):
        return 200, {"content-type": "application/json"}, payload if payload is not None else _body()

    return http_get


@pytest.fixture
def orats_key(monkeypatch):
    monkeypatch.setenv("ORATS_API_KEY", SENTINEL)
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    return SENTINEL


def test_capture_writes_body_meta_redacts_token_and_fills_meta_fields(tmp_path, orats_key):
    fixed_now = lambda: __import__("datetime").datetime(2024, 1, 3, tzinfo=__import__("datetime").timezone.utc)
    rc = finality_probe.capture(_ns(tmp_path), http_get=_fake_get(), now=fixed_now)
    assert rc == 0

    unit = tmp_path / "2024-01-02" / "L1"
    body_f = unit / "orats__hist.strikes__SPY.body"
    meta_f = unit / "orats__hist.strikes__SPY.meta.json"
    assert body_f.is_file() and meta_f.is_file()
    assert body_f.read_bytes() == _body()

    meta = json.loads(meta_f.read_text())
    assert meta["row_count"] == 2
    assert meta["max_updated_at"] == 1700000100
    assert isinstance(meta["content_sha256"], str) and len(meta["content_sha256"]) == 64
    assert meta["status"] == 200

    url = meta["url_redacted"]
    assert "REDACTED" in url
    assert SENTINEL not in url


def test_sentinel_key_never_appears_in_any_written_file(tmp_path, orats_key):
    finality_probe.capture(_ns(tmp_path), http_get=_fake_get())
    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert files, "capture must write at least one file"
    needle = SENTINEL.encode("utf-8")
    for p in files:
        assert needle not in p.read_bytes(), f"API key leaked into {p}"


def test_content_sha256_ignores_row_order_but_not_value_changes():
    rows_a = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    rows_b = [{"a": 3, "b": 4}, {"a": 1, "b": 2}]
    rows_c = [{"a": 1, "b": 99}, {"a": 3, "b": 4}]
    sha_a = finality_probe._content_sha256(rows_a, [])
    sha_b = finality_probe._content_sha256(rows_b, [])
    sha_c = finality_probe._content_sha256(rows_c, [])
    assert sha_a == sha_b
    assert sha_a != sha_c


def test_compare_counts_false_final_and_true_not_final(tmp_path, orats_key, capsys):
    from datetime import datetime, timezone

    same = _body()
    changed = _body([{"updatedAt": 1700000000, "strike": 999}, {"updatedAt": 1700000100, "strike": 101}])
    times = {
        "L1": lambda: datetime(2024, 1, 3, tzinfo=timezone.utc),
        "L2": lambda: datetime(2024, 1, 4, tzinfo=timezone.utc),
        "L3": lambda: datetime(2024, 1, 5, tzinfo=timezone.utc),
    }
    payloads = {"L1": same, "L2": same, "L3": changed}
    for label in ("L1", "L2", "L3"):
        finality_probe.capture(
            _ns(tmp_path, label=label),
            http_get=_fake_get(payloads[label]),
            now=times[label],
        )

    capsys.readouterr()
    args = argparse.Namespace(out=str(tmp_path), session=None)
    finality_probe.compare(args)
    out = capsys.readouterr().out

    assert "pair L1->L2" in out
    assert "pair L2->L3" in out
    for line in out.splitlines():
        if line.startswith("- pair L1->L2:"):
            assert "false_final=True" in line
            assert "true_not_final=False" in line
        if line.startswith("- pair L2->L3:"):
            assert "false_final=False" in line
            assert "true_not_final=True" in line
    assert "- false_final: 1" in out
    assert "- true_not_final: 1" in out


def test_missing_env_key_exits_2_without_network_call(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ORATS_API_KEY", raising=False)
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    called = []

    def boom(req):
        called.append(req)
        raise AssertionError("network must not be called")

    with pytest.raises(SystemExit) as exc:
        finality_probe.capture(_ns(tmp_path), http_get=boom)
    assert exc.value.code == 2
    assert called == []
    assert "missing ORATS_API_KEY" in capsys.readouterr().err


def test_refuses_out_dir_inside_repo(tmp_path, monkeypatch):
    monkeypatch.setenv("ORATS_API_KEY", SENTINEL)
    repo_root = Path(__file__).resolve().parents[3]
    assert (repo_root / ".git").exists() or (repo_root / ".git").is_symlink()
    with pytest.raises(SystemExit) as exc:
        finality_probe.capture(_ns(repo_root / "tests"), http_get=_fake_get())
    assert exc.value.code == 2
