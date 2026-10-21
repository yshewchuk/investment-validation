"""Capture redacts keys; compare classifies false-final / true-not-final pairs.

# packages: foundation
"""
from __future__ import annotations

import argparse
import hashlib
import json
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools import finality_probe  # noqa: E402

SENTINEL = "SENTINELKEY123"
ROWS = [
    {"updatedAt": "1700000000", "strike": 100},
    {"updatedAt": "1700000100", "strike": 101},
]
MROWS = [{"day": {"last_updated": 1700000100}, "strike": 100}]
NEXT1 = "https://api.massive.com/v3/snapshot/options/SPY?cursor=1"


def _ns(out, label="L1", session="2024-01-02", endpoints="hist/strikes", tickers="SPY", massive=False):
    return argparse.Namespace(session=session, tickers=tickers, label=label, out=str(out),
                              orats_endpoints=endpoints, massive=massive, ignore_keys="")


def _body(rows=None):
    return json.dumps({"data": rows if rows is not None else ROWS}).encode("utf-8")


def _mbody(rows, next_url=None):
    d = {"results": rows}
    if next_url: d["next_url"] = next_url
    return json.dumps(d).encode("utf-8")


def _mk_get(page0, page1):
    def http_get(req):
        return 200, {"content-type": "application/json"}, page1 if "cursor=1" in req["url"] else page0
    return http_get


def _fake_get(payload=None, headers=None):
    def http_get(req):
        return 200, headers or {"content-type": "application/json"}, payload if payload is not None else _body()

    return http_get


def _at(day):
    return lambda: datetime(2024, 1, day, tzinfo=timezone.utc)


@pytest.fixture
def orats_key(monkeypatch):
    monkeypatch.setenv("ORATS_API_KEY", SENTINEL)
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    return SENTINEL


@pytest.fixture
def massive_key(orats_key, monkeypatch):
    monkeypatch.setenv("MASSIVE_API_KEY", SENTINEL)
    return SENTINEL


def test_capture_writes_body_meta_redacts_token_and_fills_meta_fields(tmp_path, orats_key):
    rc = finality_probe.capture(_ns(tmp_path), http_get=_fake_get(), now=_at(3))
    assert rc == 0

    unit = tmp_path / "2024-01-02" / "L1"
    body_f, meta_f = unit / "orats__hist.strikes__SPY.body", unit / "orats__hist.strikes__SPY.meta.json"
    assert body_f.is_file() and meta_f.is_file() and body_f.read_bytes() == _body()

    meta = json.loads(meta_f.read_text())
    body_bytes = body_f.read_bytes()
    assert meta["body_sha256"] == hashlib.sha256(body_bytes).hexdigest()
    assert meta["body_bytes"] == len(body_bytes)
    assert meta["status"] == 200
    assert meta["ok"] is True
    assert meta["row_count"] == 2
    assert meta["max_updated_at"] == "1700000100"
    assert isinstance(meta["content_sha256"], str) and len(meta["content_sha256"]) == 64

    url = meta["url_redacted"]
    assert "token=REDACTED" in url
    assert SENTINEL not in url


def test_sentinel_leak_guard_refuses_raw_and_urlencoded_key(tmp_path, orats_key, monkeypatch):
    leaky = json.dumps({"data": ROWS, "note": SENTINEL}).encode("utf-8")
    with pytest.raises(SystemExit) as exc:
        finality_probe.capture(_ns(tmp_path), http_get=_fake_get(leaky))
    assert exc.value.code == 2
    unit = tmp_path / "2024-01-02" / "L1"
    assert not list(unit.glob("*.body")) and not list(unit.glob("*.meta.json"))

    key = "ab+cd/ef=="
    monkeypatch.setenv("ORATS_API_KEY", key)
    body = json.dumps({"data": ROWS, "note": quote(key, safe="")}).encode("utf-8")
    assert key.encode("utf-8") not in body
    with pytest.raises(SystemExit) as exc:
        finality_probe.capture(_ns(tmp_path, label="L2"), http_get=_fake_get(body))
    assert exc.value.code == 2
    unit = tmp_path / "2024-01-02" / "L2"
    assert not list(unit.glob("*.body")) and not list(unit.glob("*.meta.json"))


def test_content_sha256_order_invariance_via_captures(tmp_path, orats_key):
    rows_a = [{"updatedAt": 1700000000, "x": 1}, {"updatedAt": 1700000001, "x": 2}]
    rows_b = list(reversed(rows_a))
    rows_c = [{"updatedAt": 1700000000, "x": 99}, {"updatedAt": 1700000001, "x": 2}]
    shas = []
    for i, (label, rows) in enumerate((("L1", rows_a), ("L2", rows_b), ("L3", rows_c))):
        finality_probe.capture(_ns(tmp_path, label=label), http_get=_fake_get(_body(rows)), now=_at(3 + i))
        meta_f = tmp_path / "2024-01-02" / label / "orats__hist.strikes__SPY.meta.json"
        shas.append(json.loads(meta_f.read_text())["content_sha256"])
    assert shas[0] == shas[1]
    assert shas[0] != shas[2]


def test_compare_counts_false_final_and_true_not_final(tmp_path, orats_key, capsys):
    same = _body()
    changed = _body([{"updatedAt": "1700000000", "strike": 999}, {"updatedAt": "1700000100", "strike": 101}])
    payloads = {"L1": same, "L2": same, "L3": changed}
    for i, label in enumerate(("L1", "L2", "L3")):
        finality_probe.capture(_ns(tmp_path, label=label), http_get=_fake_get(payloads[label]), now=_at(3 + i))

    capsys.readouterr()
    finality_probe.compare(argparse.Namespace(out=str(tmp_path), session=None))
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


def test_compare_keeps_two_sessions_in_separate_groups(tmp_path, orats_key, capsys):
    for i, session in enumerate(("2024-01-02", "2024-01-03")):
        finality_probe.capture(_ns(tmp_path, session=session), http_get=_fake_get(), now=_at(3 + i))
    capsys.readouterr()
    finality_probe.compare(argparse.Namespace(out=str(tmp_path), session=None))
    out = capsys.readouterr().out
    assert "## 2024-01-02 orats hist/strikes SPY page=None" in out
    assert "## 2024-01-03 orats hist/strikes SPY page=None" in out
    assert "- groups: 2" in out


def test_status_500_excluded_from_stability_and_capture_exits_1(tmp_path, orats_key, capsys):
    finality_probe.capture(_ns(tmp_path, label="L1"), http_get=_fake_get(), now=_at(3))
    finality_probe.capture(_ns(tmp_path, label="L2"), http_get=_fake_get(), now=_at(4))
    rc = finality_probe.capture(_ns(tmp_path, label="L3"),
                                http_get=lambda req: (500, {"content-type": "application/json"}, _body()), now=_at(5))
    assert rc == 1
    capsys.readouterr()
    finality_probe.compare(argparse.Namespace(out=str(tmp_path), session=None))
    out = capsys.readouterr().out
    assert "- excluded_not_ok: [L3: 500]" in out
    assert "pair L1->L2" in out
    assert "pair L1->L3" not in out
    assert "pair L2->L3" not in out


def test_header_tracking_requires_every_group(tmp_path, orats_key, capsys):
    a = _body()
    b = _body([{"updatedAt": "1700000000", "strike": 999}, {"updatedAt": "1700000100", "strike": 101}])
    finality_probe.capture(_ns(tmp_path, label="L1", tickers="SPY"), http_get=_fake_get(a, {"etag": "e1"}), now=_at(3))
    finality_probe.capture(_ns(tmp_path, label="L2", tickers="SPY"), http_get=_fake_get(b, {"etag": "e2"}), now=_at(4))
    finality_probe.capture(_ns(tmp_path, label="L1", tickers="QQQ"), http_get=_fake_get(a, {"etag": "e1"}), now=_at(3))
    finality_probe.capture(_ns(tmp_path, label="L2", tickers="QQQ"), http_get=_fake_get(b, {"etag": "e1"}), now=_at(4))
    capsys.readouterr()
    finality_probe.compare(argparse.Namespace(out=str(tmp_path), session=None))
    out = capsys.readouterr().out
    assert "etag=yes" in out
    assert "etag=no" in out
    assert "- header etag: header does not track content" in out


def test_invalid_label_and_session_exit_2(tmp_path, orats_key):
    for kwargs in ({"label": "../x"}, {"session": "2026-1-1"}):
        with pytest.raises(SystemExit) as exc:
            finality_probe.capture(_ns(tmp_path, **kwargs), http_get=_fake_get())
        assert exc.value.code == 2


def test_invalid_calendar_session_exits_2_without_network_call(tmp_path, orats_key):
    called = []

    def boom(req):
        called.append(req)
        raise AssertionError("network must not be called")

    with pytest.raises(SystemExit) as exc:
        finality_probe.capture(_ns(tmp_path, session="2026-02-30"), http_get=boom)
    assert exc.value.code == 2
    assert called == []
    assert not (tmp_path / "2026-02-30").exists()


def test_compare_session_traversal_exits_2(tmp_path):
    with pytest.raises(SystemExit) as exc:
        finality_probe.compare(argparse.Namespace(out=str(tmp_path), session="../x"))
    assert exc.value.code == 2


def test_duplicate_tickers_exit_2_without_network_call(tmp_path, orats_key):
    called = []

    def boom(req):
        called.append(req)
        raise AssertionError("network must not be called")

    with pytest.raises(SystemExit) as exc:
        finality_probe.capture(_ns(tmp_path, tickers="SPY,SPY"), http_get=boom)
    assert exc.value.code == 2
    assert called == []
    assert not any(tmp_path.iterdir())


def test_compare_empty_dir_prints_inconclusive(tmp_path, capsys):
    finality_probe.compare(argparse.Namespace(out=str(tmp_path), session=None))
    out = capsys.readouterr().out
    assert "- header: inconclusive (no captures)" in out
    assert "header tracks content" not in out


def test_compare_skips_symlink_and_oversized_meta(tmp_path, orats_key, capsys):
    finality_probe.capture(_ns(tmp_path), http_get=_fake_get(), now=_at(3))
    unit = tmp_path / "2024-01-02" / "L1"
    (unit / "orats__hist.strikes__QQQ.meta.json").symlink_to(tmp_path / "missing.json")
    (unit / "orats__hist.strikes__ZZZ.meta.json").write_bytes(b"x" * (1024 * 1024 + 1))
    capsys.readouterr()
    finality_probe.compare(argparse.Namespace(out=str(tmp_path), session=None))
    captured = capsys.readouterr()
    assert "skipped" in captured.err and "symlink" in captured.err
    assert "larger than 1 MiB" in captured.err
    assert "- groups: 1" in captured.out


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
    assert capsys.readouterr().err == "missing ORATS_API_KEY\n"


def test_refuses_out_dir_inside_repo(tmp_path, monkeypatch):
    monkeypatch.setenv("ORATS_API_KEY", SENTINEL)
    repo_root = Path(__file__).resolve().parents[3]
    assert (repo_root / ".git").exists() or (repo_root / ".git").is_symlink()
    with pytest.raises(SystemExit) as exc:
        finality_probe.capture(_ns(repo_root / "tests"), http_get=_fake_get())
    assert exc.value.code == 2


def test_massive_composite_capture_compare_oracle(tmp_path, massive_key, capsys):
    p0 = _mbody(MROWS, NEXT1)
    p1 = _mbody([{"day": {"last_updated": 1700000300}, "strike": 102}])
    p1c = _mbody([{"day": {"last_updated": 1700000300}, "strike": 999}])
    for i, (lab, pg1) in enumerate((("A", p1), ("B", p1), ("C", p1c))):
        rc = finality_probe.capture(_ns(tmp_path, label=lab, endpoints="", massive=True),
                                    http_get=_mk_get(p0, pg1), now=_at(3 + i))
        assert rc == 0
    capsys.readouterr()
    finality_probe.compare(argparse.Namespace(out=str(tmp_path), session=None))
    out = capsys.readouterr().out
    assert out.count("## 2024-01-02 massive v3/snapshot/options SPY page=None") == 1
    assert "page=0" not in out and "page=1" not in out
    assert "pages=2" in out and "- groups: 1" in out
    for line in out.splitlines():
        if line.startswith("- pair A->B:"):
            assert "false_final=True" in line and "true_not_final=False" in line
        if line.startswith("- pair B->C:"):
            assert "false_final=False" in line and "true_not_final=True" in line
    assert "- false_final: 1" in out and "- true_not_final: 1" in out


def test_massive_next_url_beyond_page_cap_marks_truncated_and_exits_1(tmp_path, massive_key):
    def looping(req):
        return 200, {"content-type": "application/json"}, _mbody(MROWS, NEXT1)
    rc = finality_probe.capture(_ns(tmp_path, endpoints="", massive=True), http_get=looping, now=_at(3))
    assert rc == 1
    m = json.loads((tmp_path / "2024-01-02" / "L1" / "massive__v3.snapshot.options__SPY__p19.meta.json").read_text())
    assert m.get("truncated") is True and m["ok"] is False


def test_compare_session_selection_and_stable_from(tmp_path, orats_key, capsys):
    same = _body()
    other = _body([{"updatedAt": "1700000001", "strike": 999}, {"updatedAt": "1700000100", "strike": 101}])
    for i, lab in enumerate(("X1", "X2", "X3")):
        finality_probe.capture(_ns(tmp_path, session="2024-01-02", label=lab), http_get=_fake_get(same), now=_at(1 + i))
    for i, (lab, body) in enumerate((("Y1", same), ("Y2", other), ("Y3", other))):
        finality_probe.capture(_ns(tmp_path, session="2024-01-06", label=lab), http_get=_fake_get(body), now=_at(10 + i))
    capsys.readouterr()
    finality_probe.compare(argparse.Namespace(out=str(tmp_path), session="2024-01-02"))
    out = capsys.readouterr().out
    assert "2024-01-06" not in out and "- groups: 1" in out
    assert "- stable_from: X1" in out
    finality_probe.compare(argparse.Namespace(out=str(tmp_path), session="2024-01-06"))
    out = capsys.readouterr().out
    assert "2024-01-02" not in out
    assert "- stable_from: Y2" in out


def test_empty_tickers_oversized_body_and_file_modes(tmp_path, orats_key):
    with pytest.raises(SystemExit) as exc:
        finality_probe.capture(_ns(tmp_path, tickers=" , "), http_get=_fake_get())
    assert exc.value.code == 2 and not any(tmp_path.iterdir())

    def too_big(req):
        raise ValueError("BodyTooLarge")
    assert finality_probe.capture(_ns(tmp_path), http_get=too_big, now=_at(3)) == 1
    unit = tmp_path / "2024-01-02" / "L1"
    meta = json.loads((unit / "orats__hist.strikes__SPY.meta.json").read_text())
    assert meta["status"] == 0 and meta["error"] == "ValueError" and meta["ok"] is False
    assert stat.S_IMODE((tmp_path / "2024-01-02").stat().st_mode) == 0o700
    assert stat.S_IMODE(unit.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(f.stat().st_mode) == 0o600 for f in unit.iterdir())
