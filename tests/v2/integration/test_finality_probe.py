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
from urllib.parse import quote, quote_plus, urlparse

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
NEXT2 = "https://api.massive.com/v3/snapshot/options/SPY?cursor=2"
MP1_NAME = "massive__v3.snapshot.options__SPY__p1"


def _ns(out, label="L1", session="2024-01-02", endpoints="hist/strikes", tickers="SPY", massive=False, replace=False):
    return argparse.Namespace(session=session, tickers=tickers, label=label, out=str(out),
                              orats_endpoints=endpoints, massive=massive, replace=replace)


def _body(rows=None):
    return json.dumps({"data": rows if rows is not None else ROWS}).encode("utf-8")


def _mbody(rows, next_url=None):
    d = {"results": rows}
    if next_url:
        d["next_url"] = next_url
    return json.dumps(d).encode("utf-8")


def _mk_get(page0, page1):
    def http_get(req):
        http_get.calls.append(req)
        u = req["url"]
        if "cursor=" in u:
            return 200, {"content-type": "application/json"}, page1
        ticker = urlparse(u).path.rstrip("/").rsplit("/", 1)[-1]
        d = json.loads(page0)
        if isinstance(d.get("next_url"), str):
            p = urlparse(d["next_url"])
            d["next_url"] = p._replace(path=f"/v3/snapshot/options/{ticker}").geturl()
        return 200, {"content-type": "application/json"}, json.dumps(d).encode("utf-8")
    http_get.calls = []
    return http_get


def _fake_get(payload=None, headers=None):
    def http_get(req):
        http_get.calls.append(req)
        return 200, headers or {"content-type": "application/json"}, payload if payload is not None else _body()
    http_get.calls = []
    return http_get


def _at(day):
    return lambda: datetime(2024, 1, day, tzinfo=timezone.utc)


def _group_block(out, title):
    parts = out.split(f"## {title}\n")
    assert len(parts) == 2, title
    return parts[1].split("\n## ", 1)[0]


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
    get = _fake_get()
    rc = finality_probe.capture(_ns(tmp_path), http_get=get, now=_at(3))
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
    assert meta["fetched_at_utc"] == datetime(2024, 1, 3, tzinfo=timezone.utc).isoformat()
    assert isinstance(meta["run_id"], str) and len(meta["run_id"]) == 32
    rows_sorted = sorted(ROWS, key=lambda r: json.dumps(r, sort_keys=True, separators=(",", ":")))
    expected_sha = hashlib.sha256(json.dumps(rows_sorted, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert meta["content_sha256"] == expected_sha

    url = meta["url_redacted"]
    assert "token=REDACTED" in url
    assert SENTINEL not in url

    assert len(get.calls) == 1
    u = urlparse(get.calls[0]["url"])
    assert u.scheme == "https" and u.netloc == "api.orats.io"
    assert u.path == "/datav2/hist/strikes"
    assert get.calls[0]["params"]["tradeDate"] == "2024-01-02"
    assert get.calls[0]["params"]["ticker"] == "SPY"


def test_sentinel_leak_guard_refuses_raw_and_urlencoded_key(tmp_path, orats_key, monkeypatch, capsys):
    leaky = json.dumps({"data": ROWS, "note": SENTINEL}).encode("utf-8")
    with pytest.raises(SystemExit) as exc:
        finality_probe.capture(_ns(tmp_path), http_get=_fake_get(leaky))
    assert exc.value.code == 2
    assert "refusing to write provider record containing API key" in capsys.readouterr().err
    unit = tmp_path / "2024-01-02" / "L1"
    assert not list(unit.glob("*.body")) and not list(unit.glob("*.meta.json"))

    key = "ab+cd/ef=="
    monkeypatch.setenv("ORATS_API_KEY", key)
    body = json.dumps({"data": ROWS, "note": quote(key, safe="")}).encode("utf-8")
    assert key.encode("utf-8") not in body
    with pytest.raises(SystemExit) as exc:
        finality_probe.capture(_ns(tmp_path, label="L2"), http_get=_fake_get(body))
    assert exc.value.code == 2
    assert "refusing to write provider record containing API key" in capsys.readouterr().err
    unit = tmp_path / "2024-01-02" / "L2"
    assert not list(unit.glob("*.body")) and not list(unit.glob("*.meta.json"))

    key = "ab cd"
    monkeypatch.setenv("ORATS_API_KEY", key)
    body = json.dumps({"data": ROWS, "note": quote_plus(key)}).encode("utf-8")
    assert key.encode("utf-8") not in body
    assert quote_plus(key).encode("utf-8") in body
    with pytest.raises(SystemExit) as exc:
        finality_probe.capture(_ns(tmp_path, label="L3"), http_get=_fake_get(body))
    assert exc.value.code == 2
    assert "refusing to write provider record containing API key" in capsys.readouterr().err
    unit = tmp_path / "2024-01-02" / "L3"
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

    assert "- pair L1->L2: R_declares_final_at_i=True false_final=True true_not_final=False" in out
    assert "- pair L2->L3: R_declares_final_at_i=False false_final=False true_not_final=True" in out
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
    calls = []

    def http_get(req):
        calls.append(req)
        return 500, {"content-type": "application/json"}, _body()

    rc = finality_probe.capture(_ns(tmp_path, label="L3"), http_get=http_get, now=_at(5))
    assert rc == 1
    meta = json.loads((tmp_path / "2024-01-02" / "L3" / "orats__hist.strikes__SPY.meta.json").read_text())
    assert meta["status"] == 500 and meta["ok"] is False
    assert calls and calls[0]["params"]["tradeDate"] == "2024-01-02" and calls[0]["params"]["ticker"] == "SPY"
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
    finality_probe.capture(_ns(tmp_path, label="L3", tickers="QQQ"), http_get=_fake_get(a, {"etag": "e1"}), now=_at(3))
    finality_probe.capture(_ns(tmp_path, label="L4", tickers="QQQ"), http_get=_fake_get(b, {"etag": "e1"}), now=_at(4))
    capsys.readouterr()
    finality_probe.compare(argparse.Namespace(out=str(tmp_path), session=None))
    out = capsys.readouterr().out
    spy = _group_block(out, "2024-01-02 orats hist/strikes SPY page=None")
    qqq = _group_block(out, "2024-01-02 orats hist/strikes QQQ page=None")
    assert "- header_tracking: etag=yes" in spy
    assert "- header_tracking: etag=no" in qqq
    assert "- header etag: header does not track content" in out


def test_invalid_label_and_session_exit_2(tmp_path, orats_key, capsys):
    with pytest.raises(SystemExit) as exc:
        finality_probe.capture(_ns(tmp_path, label="../x"), http_get=_fake_get())
    assert exc.value.code == 2
    assert "invalid label" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        finality_probe.capture(_ns(tmp_path, session="2026-1-1"), http_get=_fake_get())
    assert exc.value.code == 2
    assert "invalid session" in capsys.readouterr().err


def test_invalid_calendar_session_exits_2_without_network_call(tmp_path, orats_key, capsys):
    called = []

    def boom(req):
        called.append(req)
        raise AssertionError("network must not be called")

    with pytest.raises(SystemExit) as exc:
        finality_probe.capture(_ns(tmp_path, session="2026-02-30"), http_get=boom)
    assert exc.value.code == 2
    assert "invalid session" in capsys.readouterr().err
    assert called == []
    assert not (tmp_path / "2026-02-30").exists()


def test_compare_session_traversal_exits_2(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        finality_probe.compare(argparse.Namespace(out=str(tmp_path), session="../x"))
    assert exc.value.code == 2
    assert "invalid session" in capsys.readouterr().err


def test_duplicate_tickers_exit_2_without_network_call(tmp_path, orats_key, capsys):
    called = []

    def boom(req):
        called.append(req)
        raise AssertionError("network must not be called")

    with pytest.raises(SystemExit) as exc:
        finality_probe.capture(_ns(tmp_path, tickers="SPY,SPY"), http_get=boom)
    assert exc.value.code == 2
    assert "duplicate tickers" in capsys.readouterr().err
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


def test_refuses_out_dir_inside_repo(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ORATS_API_KEY", SENTINEL)
    repo_root = Path(__file__).resolve().parents[3]
    assert (repo_root / ".git").exists() or (repo_root / ".git").is_symlink()
    with pytest.raises(SystemExit) as exc:
        finality_probe.capture(_ns(repo_root / "tests"), http_get=_fake_get())
    assert exc.value.code == 2
    assert "refusing to write provider data inside a repository" in capsys.readouterr().err


def test_massive_composite_capture_compare_oracle(tmp_path, massive_key, capsys):
    p0 = _mbody(MROWS, NEXT1)
    p1 = _mbody([{"day": {"last_updated": 1700000300}, "last_quote": {"last_updated": 1700000400}, "strike": 102}])
    p1c = _mbody([{"day": {"last_updated": 1700000300}, "last_quote": {"last_updated": 1700000400}, "strike": 999}])
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
    assert "| 2 | 1700000400 |" in out
    assert "etag=n/a" in out and "last-modified=n/a" in out
    assert "- pair A->B: R_declares_final_at_i=True false_final=True true_not_final=False" in out
    assert "- pair B->C: R_declares_final_at_i=False false_final=False true_not_final=True" in out
    assert "- excluded_not_ok: []" in out
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


def test_empty_tickers_oversized_body_and_file_modes(tmp_path, orats_key, capsys):
    with pytest.raises(SystemExit) as exc:
        finality_probe.capture(_ns(tmp_path, tickers=" , "), http_get=_fake_get())
    assert exc.value.code == 2
    assert "empty ticker list" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())

    def too_big(req):
        raise ValueError("BodyTooLarge")
    assert finality_probe.capture(_ns(tmp_path), http_get=too_big, now=_at(3)) == 1
    unit = tmp_path / "2024-01-02" / "L1"
    meta = json.loads((unit / "orats__hist.strikes__SPY.meta.json").read_text())
    assert meta["status"] == 0 and meta["error"] == "ValueError" and meta["ok"] is False
    assert stat.S_IMODE((tmp_path / "2024-01-02").stat().st_mode) == 0o700
    assert stat.S_IMODE(unit.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(f.stat().st_mode) == 0o600 for f in unit.iterdir())


def test_header_tracking_na_when_content_never_changes(tmp_path, orats_key, capsys):
    finality_probe.capture(_ns(tmp_path, label="L1"), http_get=_fake_get(_body(), {"etag": "e1"}), now=_at(3))
    finality_probe.capture(_ns(tmp_path, label="L2"), http_get=_fake_get(_body(), {"etag": "e1"}), now=_at(4))
    capsys.readouterr()
    finality_probe.compare(argparse.Namespace(out=str(tmp_path), session=None))
    out = capsys.readouterr().out
    assert "etag=n/a" in out
    assert "last-modified=n/a" in out


def test_massive_retry_removes_stale_pages(tmp_path, massive_key, capsys):
    p0 = _mbody(MROWS, NEXT1)
    p1 = _mbody([{"day": {"last_updated": 1700000300}, "strike": 102}])
    rc = finality_probe.capture(_ns(tmp_path, endpoints="", massive=True), http_get=_mk_get(p0, p1), now=_at(3))
    assert rc == 0
    unit = tmp_path / "2024-01-02" / "L1"
    assert (unit / "massive__v3.snapshot.options__SPY__p1.body").is_file()
    rc = finality_probe.capture(_ns(tmp_path, endpoints="", massive=True, replace=True), http_get=_fake_get(_mbody(MROWS)), now=_at(4))
    assert rc == 0
    assert (unit / "massive__v3.snapshot.options__SPY__p0.body").is_file()
    assert not (unit / "massive__v3.snapshot.options__SPY__p1.body").exists()
    assert not (unit / "massive__v3.snapshot.options__SPY__p1.meta.json").exists()
    capsys.readouterr()
    finality_probe.compare(argparse.Namespace(out=str(tmp_path), session=None))
    out = capsys.readouterr().out
    assert "pages=1" in out and "pages=2" not in out


def test_compare_skips_tampered_body(tmp_path, orats_key, capsys):
    finality_probe.capture(_ns(tmp_path), http_get=_fake_get(), now=_at(3))
    (tmp_path / "2024-01-02" / "L1" / "orats__hist.strikes__SPY.body").write_bytes(b"tampered")
    capsys.readouterr()
    finality_probe.compare(argparse.Namespace(out=str(tmp_path), session=None))
    captured = capsys.readouterr()
    assert "body mismatch" in captured.err
    assert "- groups: 0" in captured.out


def test_error_envelope_marked_not_ok(tmp_path, orats_key):
    payload = json.dumps({"error": "x"}).encode("utf-8")
    rc = finality_probe.capture(_ns(tmp_path), http_get=_fake_get(payload), now=_at(3))
    assert rc == 1
    meta = json.loads((tmp_path / "2024-01-02" / "L1" / "orats__hist.strikes__SPY.meta.json").read_text())
    assert meta["status"] == 200 and meta["ok"] is False


def test_invalid_page_meta_skipped_compare_succeeds(tmp_path, orats_key, capsys):
    finality_probe.capture(_ns(tmp_path, label="L1"), http_get=_fake_get(), now=_at(3))
    finality_probe.capture(_ns(tmp_path, label="L2"), http_get=_fake_get(), now=_at(4))
    meta_f = tmp_path / "2024-01-02" / "L2" / "orats__hist.strikes__SPY.meta.json"
    m = json.loads(meta_f.read_text())
    m["page"] = "x"
    meta_f.write_text(json.dumps(m))
    capsys.readouterr()
    finality_probe.compare(argparse.Namespace(out=str(tmp_path), session=None))
    captured = capsys.readouterr()
    assert "skipped" in captured.err and "invalid page" in captured.err
    assert "- groups: 1" in captured.out


def test_default_http_get_enforces_body_cap(monkeypatch):
    import requests
    monkeypatch.setattr(finality_probe, "MAX_BODY_BYTES", 100)

    class FakeResp:
        status_code = 200
        headers = {"content-type": "application/json"}

        def iter_content(self, chunk_size):
            yield b"a" * 60
            yield b"b" * 60

        def close(self):
            return None

    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResp())
    with pytest.raises(ValueError, match="BodyTooLarge"):
        finality_probe._default_http_get({"url": "https://example.com/x"})


def test_massive_next_url_off_origin_stops_paging(tmp_path, massive_key):
    def http_get(req):
        return 200, {"content-type": "application/json"}, _mbody(MROWS, "https://evil.example/x")

    rc = finality_probe.capture(_ns(tmp_path, endpoints="", massive=True), http_get=http_get, now=_at(3))
    assert rc == 1
    unit = tmp_path / "2024-01-02" / "L1"
    assert not list(unit.glob("*__p1.*"))
    m = json.loads((unit / "massive__v3.snapshot.options__SPY__p0.meta.json").read_text())
    assert m.get("truncated") is True and m["ok"] is False


def test_compare_skips_tampered_content_digest(tmp_path, orats_key, capsys):
    finality_probe.capture(_ns(tmp_path), http_get=_fake_get(), now=_at(3))
    meta_f = tmp_path / "2024-01-02" / "L1" / "orats__hist.strikes__SPY.meta.json"
    m = json.loads(meta_f.read_text())
    m["content_sha256"] = "0" * 64
    meta_f.write_text(json.dumps(m))
    capsys.readouterr()
    finality_probe.compare(argparse.Namespace(out=str(tmp_path), session=None))
    captured = capsys.readouterr()
    assert "content digest mismatch" in captured.err
    assert "- groups: 0" in captured.out


def test_compare_skips_mismatched_hash_policy(tmp_path, orats_key, capsys):
    finality_probe.capture(_ns(tmp_path), http_get=_fake_get(), now=_at(3))
    meta_f = tmp_path / "2024-01-02" / "L1" / "orats__hist.strikes__SPY.meta.json"
    m = json.loads(meta_f.read_text())
    assert m["hash_policy"] == "rows-sorted-canonical-json-v1"
    m["hash_policy"] = "rows-sorted-canonical-json-v0"
    meta_f.write_text(json.dumps(m))
    capsys.readouterr()
    finality_probe.compare(argparse.Namespace(out=str(tmp_path), session=None))
    captured = capsys.readouterr()
    assert "hash policy mismatch" in captured.err
    assert "- groups: 0" in captured.out


def test_compare_rejects_bool_page_and_string_status(tmp_path, orats_key, capsys):
    finality_probe.capture(_ns(tmp_path, label="L1"), http_get=_fake_get(), now=_at(3))
    finality_probe.capture(_ns(tmp_path, label="L2"), http_get=_fake_get(), now=_at(4))
    unit = tmp_path / "2024-01-02"
    meta_f = unit / "L1" / "orats__hist.strikes__SPY.meta.json"
    m = json.loads(meta_f.read_text())
    m["page"] = True
    meta_f.write_text(json.dumps(m))
    meta_f = unit / "L2" / "orats__hist.strikes__SPY.meta.json"
    m = json.loads(meta_f.read_text())
    m["status"] = "200"
    meta_f.write_text(json.dumps(m))
    capsys.readouterr()
    finality_probe.compare(argparse.Namespace(out=str(tmp_path), session=None))
    captured = capsys.readouterr()
    assert "invalid page" in captured.err
    assert "invalid status" in captured.err
    assert "- groups: 0" in captured.out


def test_massive_failed_replace_rerun_invalidates_prior_pages(tmp_path, massive_key):
    p0 = _mbody(MROWS, NEXT1)
    p1 = _mbody([{"day": {"last_updated": 1700000300}, "strike": 102}])
    rc = finality_probe.capture(_ns(tmp_path, endpoints="", massive=True), http_get=_mk_get(p0, p1), now=_at(3))
    assert rc == 0
    unit = tmp_path / "2024-01-02" / "L1"
    p1_body = unit / "massive__v3.snapshot.options__SPY__p1.body"
    p1_meta = unit / "massive__v3.snapshot.options__SPY__p1.meta.json"
    assert p1_body.is_file() and p1_meta.is_file()

    def boom(req):
        raise RuntimeError("fetch failed")

    rc = finality_probe.capture(_ns(tmp_path, endpoints="", massive=True, replace=True), http_get=boom, now=_at(4))
    assert rc == 1
    assert not p1_body.exists() and not p1_meta.exists()
    m0 = json.loads((unit / "massive__v3.snapshot.options__SPY__p0.meta.json").read_text())
    assert m0.get("error") == "RuntimeError" and m0["ok"] is False


def test_massive_next_url_bad_path_or_api_key_marks_truncated(tmp_path, massive_key):
    cases = ("https://api.massive.com/other?cursor=1",
             "https://api.massive.com/v3/snapshot/options/SPY?apiKey=x")
    for i, nu in enumerate(cases):
        label = f"L{i}"

        def http_get(req, nu=nu):
            return 200, {"content-type": "application/json"}, _mbody(MROWS, nu)

        rc = finality_probe.capture(_ns(tmp_path, label=label, endpoints="", massive=True),
                                    http_get=http_get, now=_at(3 + i))
        assert rc == 1
        unit = tmp_path / "2024-01-02" / label
        assert not list(unit.glob("*__p1.*"))
        m = json.loads((unit / "massive__v3.snapshot.options__SPY__p0.meta.json").read_text())
        assert m.get("truncated") is True and m["ok"] is False


def test_interrupted_massive_capture_excluded(tmp_path, massive_key, capsys):
    p0 = _mbody(MROWS, NEXT1)

    def interrupted(req):
        if "cursor=1" in req["url"]:
            raise SystemExit(130)
        return 200, {"content-type": "application/json"}, p0

    with pytest.raises(SystemExit) as exc:
        finality_probe.capture(_ns(tmp_path, endpoints="", massive=True), http_get=interrupted, now=_at(3))
    assert exc.value.code == 130
    unit = tmp_path / "2024-01-02" / "L1"
    m0 = json.loads((unit / "massive__v3.snapshot.options__SPY__p0.meta.json").read_text())
    assert m0.get("has_next") is True and m0["ok"] is True
    assert not list(unit.glob("*__p1.*"))
    capsys.readouterr()
    finality_probe.compare(argparse.Namespace(out=str(tmp_path), session=None))
    out = capsys.readouterr().out
    assert "- excluded_not_ok: [L1: 200]" in out
    assert "- pair " not in out


def test_two_run_ids_composite_excluded(tmp_path, massive_key, capsys):
    p0 = _mbody(MROWS, NEXT1)
    p1 = _mbody([{"day": {"last_updated": 1700000300}, "strike": 102}])
    rc = finality_probe.capture(_ns(tmp_path, endpoints="", massive=True), http_get=_mk_get(p0, p1), now=_at(3))
    assert rc == 0
    meta_f = tmp_path / "2024-01-02" / "L1" / "massive__v3.snapshot.options__SPY__p1.meta.json"
    m = json.loads(meta_f.read_text())
    m["run_id"] = "0" * 32
    meta_f.write_text(json.dumps(m, sort_keys=True, indent=2) + "\n")
    capsys.readouterr()
    finality_probe.compare(argparse.Namespace(out=str(tmp_path), session=None))
    out = capsys.readouterr().out
    assert "- excluded_not_ok: [L1: 200]" in out
    assert "- pair " not in out


def test_one_page_massive_etag_tracks_content(tmp_path, massive_key, capsys):
    a = _mbody(MROWS)
    b = _mbody([{"day": {"last_updated": 1700000300}, "strike": 102}])
    get = _fake_get(a, {"etag": "e1"})
    rc = finality_probe.capture(_ns(tmp_path, label="A", endpoints="", massive=True), http_get=get, now=_at(3))
    assert rc == 0
    assert get.calls[0]["url"].startswith("https://api.massive.com/v3/snapshot/options/")
    assert get.calls[0]["headers"]["Authorization"] == f"Bearer {SENTINEL}"
    m = json.loads((tmp_path / "2024-01-02" / "A" / "massive__v3.snapshot.options__SPY__p0.meta.json").read_text())
    assert m["status"] == 200 and m["ok"] is True
    rc = finality_probe.capture(_ns(tmp_path, label="B", endpoints="", massive=True),
                                 http_get=_fake_get(b, {"etag": "e2"}), now=_at(4))
    assert rc == 0
    capsys.readouterr()
    finality_probe.compare(argparse.Namespace(out=str(tmp_path), session=None))
    out = capsys.readouterr().out
    block = _group_block(out, "2024-01-02 massive v3/snapshot/options SPY page=None")
    assert "pages=1" in block
    assert "- header_tracking: etag=yes" in block


def test_tampered_max_updated_at_skipped(tmp_path, orats_key, capsys):
    finality_probe.capture(_ns(tmp_path), http_get=_fake_get(), now=_at(3))
    meta_f = tmp_path / "2024-01-02" / "L1" / "orats__hist.strikes__SPY.meta.json"
    m = json.loads(meta_f.read_text())
    m["max_updated_at"] = "1699999999"
    meta_f.write_text(json.dumps(m, sort_keys=True, indent=2) + "\n")
    capsys.readouterr()
    finality_probe.compare(argparse.Namespace(out=str(tmp_path), session=None))
    captured = capsys.readouterr()
    assert "max_updated_at mismatch" in captured.err
    assert "- groups: 0" in captured.out


def test_existing_label_refused_byte_identical_then_replace_succeeds(tmp_path, orats_key, capsys):
    finality_probe.capture(_ns(tmp_path), http_get=_fake_get(), now=_at(3))
    unit = tmp_path / "2024-01-02" / "L1"
    before = {f.name: f.read_bytes() for f in unit.iterdir()}
    assert before
    with pytest.raises(SystemExit) as exc:
        finality_probe.capture(_ns(tmp_path), http_get=_fake_get(), now=_at(4))
    assert exc.value.code == 2
    assert "label already captured; pass --replace to invalidate the prior capture" in capsys.readouterr().err
    after = {f.name: f.read_bytes() for f in unit.iterdir()}
    assert after == before
    rc = finality_probe.capture(_ns(tmp_path, replace=True), http_get=_fake_get(), now=_at(4))
    assert rc == 0
    meta = json.loads((unit / "orats__hist.strikes__SPY.meta.json").read_text())
    assert meta["fetched_at_utc"] == datetime(2024, 1, 4, tzinfo=timezone.utc).isoformat()


def test_continuation_metadata_tamper_skipped(tmp_path, massive_key, capsys):
    p0 = _mbody(MROWS, NEXT1)
    p1 = _mbody([{"day": {"last_updated": 1700000300}, "strike": 102}])
    rc = finality_probe.capture(_ns(tmp_path, endpoints="", massive=True), http_get=_mk_get(p0, p1), now=_at(3))
    assert rc == 0
    meta_f = tmp_path / "2024-01-02" / "L1" / "massive__v3.snapshot.options__SPY__p0.meta.json"
    m = json.loads(meta_f.read_text())
    assert m["has_next"] is True
    m["has_next"] = False
    meta_f.write_text(json.dumps(m, sort_keys=True, indent=2) + "\n")
    capsys.readouterr()
    finality_probe.compare(argparse.Namespace(out=str(tmp_path), session=None))
    captured = capsys.readouterr()
    assert "continuation mismatch" in captured.err
    assert "- excluded_not_ok: [L1: 200]" in captured.out
    assert "- pair " not in captured.out


def test_identity_tamper_skipped_compare_succeeds(tmp_path, orats_key, capsys):
    for lab in ("A", "B", "C", "D"):
        finality_probe.capture(_ns(tmp_path, label=lab), http_get=_fake_get(), now=_at(3))
    unit = tmp_path / "2024-01-02"
    meta_f = unit / "A" / "orats__hist.strikes__SPY.meta.json"
    m = json.loads(meta_f.read_text())
    m["ticker"] = ["SPY"]
    meta_f.write_text(json.dumps(m, sort_keys=True, indent=2) + "\n")
    meta_f = unit / "B" / "orats__hist.strikes__SPY.meta.json"
    m = json.loads(meta_f.read_text())
    m["run_id"] = ["not-a-str"]
    meta_f.write_text(json.dumps(m, sort_keys=True, indent=2) + "\n")
    meta_f = unit / "C" / "orats__hist.strikes__SPY.meta.json"
    m = json.loads(meta_f.read_text())
    m["session"] = "2024-01-03"
    meta_f.write_text(json.dumps(m, sort_keys=True, indent=2) + "\n")
    capsys.readouterr()
    finality_probe.compare(argparse.Namespace(out=str(tmp_path), session=None))
    captured = capsys.readouterr()
    assert "invalid ticker" in captured.err
    assert "invalid run_id" in captured.err
    assert "session mismatch" in captured.err
    assert "- groups: 1" in captured.out


def test_identity_fields_must_match_filename_stem(tmp_path, orats_key, capsys):
    for lab in ("A", "B", "C"):
        finality_probe.capture(_ns(tmp_path, label=lab), http_get=_fake_get(), now=_at(3))
    unit = tmp_path / "2024-01-02"
    meta_f = unit / "A" / "orats__hist.strikes__SPY.meta.json"
    m = json.loads(meta_f.read_text())
    m["ticker"] = "QQQ"
    meta_f.write_text(json.dumps(m, sort_keys=True, indent=2) + "\n")
    meta_f = unit / "B" / "orats__hist.strikes__SPY.meta.json"
    m = json.loads(meta_f.read_text())
    m["provider"] = "massive"
    meta_f.write_text(json.dumps(m, sort_keys=True, indent=2) + "\n")
    meta_f = unit / "C" / "orats__hist.strikes__SPY.meta.json"
    m = json.loads(meta_f.read_text())
    m["page"] = 0
    meta_f.write_text(json.dumps(m, sort_keys=True, indent=2) + "\n")
    capsys.readouterr()
    finality_probe.compare(argparse.Namespace(out=str(tmp_path), session=None))
    captured = capsys.readouterr()
    assert captured.err.count("identity does not match filename") == 3
    assert "- groups: 0" in captured.out


def test_truncated_tamper_on_complete_last_page_skipped(tmp_path, massive_key, capsys):
    p0 = _mbody(MROWS, NEXT1)
    p1 = _mbody([{"day": {"last_updated": 1700000300}, "strike": 102}])
    rc = finality_probe.capture(_ns(tmp_path, endpoints="", massive=True), http_get=_mk_get(p0, p1), now=_at(3))
    assert rc == 0
    meta_f = tmp_path / "2024-01-02" / "L1" / "massive__v3.snapshot.options__SPY__p1.meta.json"
    m = json.loads(meta_f.read_text())
    assert "truncated" not in m and m["has_next"] is False
    m["truncated"] = True
    meta_f.write_text(json.dumps(m, sort_keys=True, indent=2) + "\n")
    capsys.readouterr()
    finality_probe.compare(argparse.Namespace(out=str(tmp_path), session=None))
    captured = capsys.readouterr()
    assert "continuation mismatch" in captured.err
    after = json.loads(meta_f.read_text())
    assert after["has_next"] is False
    assert "- excluded_not_ok: [L1: 200]" in captured.out


def test_truncated_tamper_on_foreign_next_url_skipped(tmp_path, massive_key, capsys):
    def http_get(req):
        return 200, {"content-type": "application/json"}, _mbody(MROWS, "https://evil.example/x")

    rc = finality_probe.capture(_ns(tmp_path, endpoints="", massive=True), http_get=http_get, now=_at(3))
    assert rc == 1
    meta_f = tmp_path / "2024-01-02" / "L1" / "massive__v3.snapshot.options__SPY__p0.meta.json"
    m = json.loads(meta_f.read_text())
    assert m.get("truncated") is True and m["has_next"] is False
    m["truncated"] = False
    meta_f.write_text(json.dumps(m, sort_keys=True, indent=2) + "\n")
    capsys.readouterr()
    finality_probe.compare(argparse.Namespace(out=str(tmp_path), session=None))
    captured = capsys.readouterr()
    assert "continuation mismatch" in captured.err
    after = json.loads(meta_f.read_text())
    assert after["has_next"] is False
    assert "- groups: 0" in captured.out


def test_unusable_output_path_exits_2_without_traceback(tmp_path, orats_key, capsys):
    bad = tmp_path / "regular-file"
    bad.write_text("not a directory")
    with pytest.raises(SystemExit) as exc:
        finality_probe.capture(_ns(bad), http_get=_fake_get())
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "cannot use output path:" in err
    assert "Traceback" not in err
    with pytest.raises(SystemExit) as exc:
        finality_probe.compare(argparse.Namespace(out=str(bad), session=None))
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "cannot use output path:" in err
    assert "Traceback" not in err


def test_massive_missing_middle_page_excluded(tmp_path, massive_key, capsys):
    p0 = _mbody(MROWS, NEXT1)
    p1 = _mbody([{"day": {"last_updated": 1700000300}, "strike": 102}], NEXT2)
    p2 = _mbody([{"day": {"last_updated": 1700000500}, "strike": 103}])

    def get(req):
        u = req["url"]
        body = p2 if "cursor=2" in u else p1 if "cursor=1" in u else p0
        return 200, {"content-type": "application/json"}, body

    rc = finality_probe.capture(_ns(tmp_path, endpoints="", massive=True), http_get=get, now=_at(3))
    assert rc == 0
    unit = tmp_path / "2024-01-02" / "L1"
    for ext in ("body", "meta.json"):
        (unit / f"massive__v3.snapshot.options__SPY__p1.{ext}").unlink()
    capsys.readouterr()
    finality_probe.compare(argparse.Namespace(out=str(tmp_path), session=None))
    out = capsys.readouterr().out
    assert "- excluded_not_ok: [L1: 200]" in out
    assert "- pair " not in out


def test_ts_units_and_garbage():
    expected = 1700000000.0
    for v in (1700000000, 1700000000000, 1700000000000000000,
              "1700000000", "1700000000000", "1700000000000000000",
              "2023-11-14T22:13:20Z"):
        assert finality_probe._ts(v) == expected, v
    for garbage in (None, "", "garbage", "not-a-time", "2023-99-99",
                    True, False, float("nan"), float("inf"), float("-inf"),
                    "nan", "inf", "-inf"):
        assert finality_probe._ts(garbage) is None, garbage


def test_ok_receipt_mismatch_tamper_skipped(tmp_path, orats_key, capsys):
    def five_hundred(req):
        return 500, {"content-type": "application/json"}, _body()

    rc = finality_probe.capture(_ns(tmp_path, label="L1"), http_get=five_hundred, now=_at(3))
    assert rc == 1
    meta_f = tmp_path / "2024-01-02" / "L1" / "orats__hist.strikes__SPY.meta.json"
    m = json.loads(meta_f.read_text())
    assert m["status"] == 500 and m["ok"] is False
    m["ok"] = True
    meta_f.write_text(json.dumps(m, sort_keys=True, indent=2) + "\n")

    payload = json.dumps({"error": "x"}).encode("utf-8")
    rc = finality_probe.capture(_ns(tmp_path, label="L2"), http_get=_fake_get(payload), now=_at(4))
    assert rc == 1
    meta_f = tmp_path / "2024-01-02" / "L2" / "orats__hist.strikes__SPY.meta.json"
    m = json.loads(meta_f.read_text())
    assert m["status"] == 200 and m["ok"] is False
    m["ok"] = True
    meta_f.write_text(json.dumps(m, sort_keys=True, indent=2) + "\n")

    capsys.readouterr()
    finality_probe.compare(argparse.Namespace(out=str(tmp_path), session=None))
    captured = capsys.readouterr()
    assert captured.err.count("ok receipt mismatch") == 2
    assert "skipped" in captured.err
    assert "- groups: 0" in captured.out


def test_massive_next_url_other_ticker_stops_paging(tmp_path, massive_key):
    other = "https://api.massive.com/v3/snapshot/options/AAPL?cursor=x"

    def http_get(req):
        return 200, {"content-type": "application/json"}, _mbody(MROWS, other)

    rc = finality_probe.capture(_ns(tmp_path, endpoints="", massive=True), http_get=http_get, now=_at(3))
    assert rc == 1
    unit = tmp_path / "2024-01-02" / "L1"
    assert not list(unit.glob("*__p1.*"))
    m = json.loads((unit / "massive__v3.snapshot.options__SPY__p0.meta.json").read_text())
    assert m.get("truncated") is True and m["ok"] is False


def test_sentinel_leak_guard_refuses_lowercase_percent_encoding(tmp_path, monkeypatch, capsys):
    key = "ab+cd/ef=="
    monkeypatch.setenv("ORATS_API_KEY", key)
    low = "ab%2bcd%2fef%3d%3d"
    body = json.dumps({"data": ROWS, "note": low}).encode("utf-8")
    assert key.encode("utf-8") not in body
    assert quote(key, safe="").encode("utf-8") not in body
    assert low.encode("utf-8") in body
    with pytest.raises(SystemExit) as exc:
        finality_probe.capture(_ns(tmp_path), http_get=_fake_get(body))
    assert exc.value.code == 2
    assert "refusing to write provider record containing API key" in capsys.readouterr().err
    unit = tmp_path / "2024-01-02" / "L1"
    assert not list(unit.glob("*.body")) and not list(unit.glob("*.meta.json"))


def test_replace_drops_unrequested_ticker_files(tmp_path, orats_key, capsys):
    rc = finality_probe.capture(_ns(tmp_path, tickers="SPY,AAPL"), http_get=_fake_get(), now=_at(3))
    assert rc == 0
    unit = tmp_path / "2024-01-02" / "L1"
    assert (unit / "orats__hist.strikes__SPY.body").is_file()
    assert (unit / "orats__hist.strikes__AAPL.body").is_file()
    assert (unit / "orats__hist.strikes__AAPL.meta.json").is_file()

    rc = finality_probe.capture(_ns(tmp_path, tickers="SPY", replace=True), http_get=_fake_get(), now=_at(4))
    assert rc == 0
    assert (unit / "orats__hist.strikes__SPY.body").is_file()
    assert (unit / "orats__hist.strikes__SPY.meta.json").is_file()
    assert not (unit / "orats__hist.strikes__AAPL.body").exists()
    assert not (unit / "orats__hist.strikes__AAPL.meta.json").exists()
    capsys.readouterr()
    finality_probe.compare(argparse.Namespace(out=str(tmp_path), session=None))
    out = capsys.readouterr().out
    assert "## 2024-01-02 orats hist/strikes SPY page=None" in out
    assert "AAPL" not in out


def test_compare_corrupt_meta_reports_invalid_json_with_relative_name(tmp_path, orats_key, capsys):
    finality_probe.capture(_ns(tmp_path), http_get=_fake_get(), now=_at(3))
    meta_f = tmp_path / "2024-01-02" / "L1" / "orats__hist.strikes__QQQ.meta.json"
    meta_f.write_bytes(b"not json at all")
    capsys.readouterr()
    finality_probe.compare(argparse.Namespace(out=str(tmp_path), session=None))
    captured = capsys.readouterr()
    assert "invalid JSON" in captured.err
    assert "2024-01-02/L1/orats__hist.strikes__QQQ.meta.json" in captured.err
    assert str(tmp_path) not in captured.err
    assert "Expecting" not in captured.err


def test_endpoint_with_dot_edited_skipped(tmp_path, orats_key, capsys):
    finality_probe.capture(_ns(tmp_path), http_get=_fake_get(), now=_at(3))
    meta_f = tmp_path / "2024-01-02" / "L1" / "orats__hist.strikes__SPY.meta.json"
    m = json.loads(meta_f.read_text())
    m["endpoint"] = "hist.strikes"
    meta_f.write_text(json.dumps(m, sort_keys=True, indent=2) + "\n")
    capsys.readouterr()
    finality_probe.compare(argparse.Namespace(out=str(tmp_path), session=None))
    captured = capsys.readouterr()
    assert "invalid endpoint" in captured.err
    assert "2024-01-02/L1/orats__hist.strikes__SPY.meta.json" in captured.err
    assert str(tmp_path) not in captured.err
    assert "- groups: 0" in captured.out


def test_massive_null_page_edited_skipped(tmp_path, massive_key, capsys):
    rc = finality_probe.capture(_ns(tmp_path, endpoints="", massive=True),
                                http_get=_fake_get(_mbody(MROWS)), now=_at(3))
    assert rc == 0
    unit = tmp_path / "2024-01-02" / "L1"
    body_f = unit / "massive__v3.snapshot.options__SPY__p0.body"
    meta_f = unit / "massive__v3.snapshot.options__SPY__p0.meta.json"
    body_f.rename(unit / "massive__v3.snapshot.options__SPY.body")
    meta_f.rename(unit / "massive__v3.snapshot.options__SPY.meta.json")
    meta_f = unit / "massive__v3.snapshot.options__SPY.meta.json"
    m = json.loads(meta_f.read_text())
    m["page"] = None
    meta_f.write_text(json.dumps(m, sort_keys=True, indent=2) + "\n")
    capsys.readouterr()
    finality_probe.compare(argparse.Namespace(out=str(tmp_path), session=None))
    captured = capsys.readouterr()
    assert "invalid page" in captured.err
    assert "2024-01-02/L1/massive__v3.snapshot.options__SPY.meta.json" in captured.err
    assert str(tmp_path) not in captured.err
    assert "- groups: 0" in captured.out


def test_orats_page_zero_edited_skipped(tmp_path, orats_key, capsys):
    finality_probe.capture(_ns(tmp_path), http_get=_fake_get(), now=_at(3))
    unit = tmp_path / "2024-01-02" / "L1"
    body_f = unit / "orats__hist.strikes__SPY.body"
    meta_f = unit / "orats__hist.strikes__SPY.meta.json"
    body_f.rename(unit / "orats__hist.strikes__SPY__p0.body")
    meta_f.rename(unit / "orats__hist.strikes__SPY__p0.meta.json")
    meta_f = unit / "orats__hist.strikes__SPY__p0.meta.json"
    m = json.loads(meta_f.read_text())
    m["page"] = 0
    meta_f.write_text(json.dumps(m, sort_keys=True, indent=2) + "\n")
    capsys.readouterr()
    finality_probe.compare(argparse.Namespace(out=str(tmp_path), session=None))
    captured = capsys.readouterr()
    assert "invalid page" in captured.err
    assert "2024-01-02/L1/orats__hist.strikes__SPY__p0.meta.json" in captured.err
    assert str(tmp_path) not in captured.err
    assert "- groups: 0" in captured.out


def test_rule_r_strict_boundary_max_updated_equals_fetch(tmp_path, orats_key, capsys):
    epoch = int(datetime(2024, 1, 3, tzinfo=timezone.utc).timestamp())
    rows = {
        "SPY": [{"updatedAt": str(epoch), "strike": 1}],
        "QQQ": [{"updatedAt": str(epoch - 1), "strike": 1}],
    }

    def http_get(req):
        return 200, {"content-type": "application/json"}, _body(rows[req["params"]["ticker"]])

    for i, label in enumerate(("A", "B")):
        rc = finality_probe.capture(_ns(tmp_path, label=label, tickers="SPY,QQQ"),
                                    http_get=http_get, now=_at(3 + i))
        assert rc == 0
    capsys.readouterr()
    finality_probe.compare(argparse.Namespace(out=str(tmp_path), session=None))
    out = capsys.readouterr().out
    spy = _group_block(out, "2024-01-02 orats hist/strikes SPY page=None")
    qqq = _group_block(out, "2024-01-02 orats hist/strikes QQQ page=None")
    assert "- pair A->B: R_declares_final_at_i=False" in spy
    assert "- pair A->B: R_declares_final_at_i=True" in qqq


def _mk_get3(page0, page1, page2):
    def http_get(req):
        u = req["url"]
        body = page2 if "cursor=2" in u else page1 if "cursor=1" in u else page0
        return 200, {"content-type": "application/json"}, body
    return http_get


def test_composite_digest_last_and_middle_page_changes(tmp_path, massive_key, capsys):
    p0 = _mbody(MROWS, NEXT1)
    p1 = _mbody([{"day": {"last_updated": 1700000300}, "strike": 102}], NEXT2)
    p2 = _mbody([{"day": {"last_updated": 1700000500}, "strike": 103}])
    p1c = _mbody([{"day": {"last_updated": 1700000300}, "strike": 999}], NEXT2)
    p2c = _mbody([{"day": {"last_updated": 1700000500}, "strike": 999}])
    variants = (("A", p1, p2), ("B", p1, p2), ("C", p1, p2c), ("D", p1c, p2))
    for i, (lab, pg1, pg2) in enumerate(variants):
        rc = finality_probe.capture(_ns(tmp_path, label=lab, endpoints="", massive=True),
                                    http_get=_mk_get3(p0, pg1, pg2), now=_at(3 + i))
        assert rc == 0
    capsys.readouterr()
    finality_probe.compare(argparse.Namespace(out=str(tmp_path), session=None))
    out = capsys.readouterr().out
    assert "- pair A->B: R_declares_final_at_i=True false_final=True true_not_final=False" in out
    assert "- pair B->C: R_declares_final_at_i=False false_final=False true_not_final=True" in out
    assert "- pair C->D: R_declares_final_at_i=False false_final=False true_not_final=True" in out
    block = _group_block(out, "2024-01-02 massive v3/snapshot/options SPY page=None")
    shas = {}
    for line in block.splitlines():
        if not line.startswith("| "):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 8 and cells[0] in ("A", "B", "C", "D") and cells[7] == "pages=3":
            shas[cells[0]] = cells[6]
    assert shas["A"] == shas["B"]
    assert shas["C"] != shas["A"]
    assert shas["D"] != shas["A"]


def test_main_cli_dispatch_compare_and_capture_missing_key(tmp_path, orats_key, capsys, monkeypatch):
    rc = finality_probe.capture(_ns(tmp_path), http_get=_fake_get(), now=_at(3))
    assert rc == 0
    capsys.readouterr()
    finality_probe.main(["compare", "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert "# finality compare" in out
    assert "## 2024-01-02 orats hist/strikes SPY page=None" in out
    assert "- groups: 1" in out

    monkeypatch.delenv("ORATS_API_KEY", raising=False)
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc:
        finality_probe.main(["capture", "--session", "2024-01-02", "--tickers", "SPY",
                             "--label", "L9", "--out", str(tmp_path)])
    assert exc.value.code == 2
    assert "missing ORATS_API_KEY" in capsys.readouterr().err


def test_capture_invalid_argument_types_exit_2(tmp_path, orats_key, capsys):
    for field, value in (("tickers", None), ("label", 5), ("out", None)):
        ns = _ns(tmp_path)
        setattr(ns, field, value)
        with pytest.raises(SystemExit) as exc:
            finality_probe.capture(ns, http_get=_fake_get())
        assert exc.value.code == 2
        assert "invalid argument types" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


def test_run_id_differs_across_captures(tmp_path, orats_key):
    for i, label in enumerate(("L1", "L2")):
        rc = finality_probe.capture(_ns(tmp_path, label=label), http_get=_fake_get(), now=_at(3 + i))
        assert rc == 0
    runs = []
    for label in ("L1", "L2"):
        meta_f = tmp_path / "2024-01-02" / label / "orats__hist.strikes__SPY.meta.json"
        runs.append(json.loads(meta_f.read_text())["run_id"])
    assert all(isinstance(r, str) and len(r) == 32 for r in runs)
    assert runs[0] != runs[1]


def test_elapsed_ms_measures_http_time(tmp_path, orats_key, monkeypatch):
    class _StepTime:
        def __init__(self):
            self.t = 0.0

        def perf_counter(self):
            self.t += 0.25
            return self.t

    monkeypatch.setattr(finality_probe, "time", _StepTime())
    rc = finality_probe.capture(_ns(tmp_path), http_get=_fake_get(), now=_at(3))
    assert rc == 0
    meta = json.loads((tmp_path / "2024-01-02" / "L1" / "orats__hist.strikes__SPY.meta.json").read_text())
    assert meta["elapsed_ms"] == 250


def test_orats_numeric_updated_at_max_and_bool_nan_ignored(tmp_path, orats_key):
    mixed = [{"updatedAt": "1700000000"}, {"updatedAt": 1700000100}, {"updatedAt": 1700000050.5}]
    rc = finality_probe.capture(_ns(tmp_path, label="L1"), http_get=_fake_get(_body(mixed)), now=_at(3))
    assert rc == 0
    meta = json.loads((tmp_path / "2024-01-02" / "L1" / "orats__hist.strikes__SPY.meta.json").read_text())
    assert meta["max_updated_at"] == 1700000100

    junk = [{"updatedAt": True}, {"updatedAt": float("nan")}, {"updatedAt": "garbage"}]
    rc = finality_probe.capture(_ns(tmp_path, label="L2"), http_get=_fake_get(_body(junk)), now=_at(4))
    assert rc == 0
    meta = json.loads((tmp_path / "2024-01-02" / "L2" / "orats__hist.strikes__SPY.meta.json").read_text())
    assert meta["max_updated_at"] is None


def test_header_secret_leak_refuses_and_sensitive_headers_dropped(tmp_path, orats_key, capsys):
    with pytest.raises(SystemExit) as exc:
        finality_probe.capture(_ns(tmp_path), http_get=_fake_get(headers={"x-echo": SENTINEL}))
    assert exc.value.code == 2
    assert "refusing to write provider record containing API key" in capsys.readouterr().err
    unit = tmp_path / "2024-01-02" / "L1"
    assert not list(unit.glob("*.body")) and not list(unit.glob("*.meta.json"))

    hdrs = {"content-type": "application/json", "Set-Cookie": "id=1",
            "Authorization": "Bearer arbitrary", "x-echo": "fine"}
    rc = finality_probe.capture(_ns(tmp_path, label="L2"), http_get=_fake_get(headers=hdrs), now=_at(4))
    assert rc == 0
    meta = json.loads((tmp_path / "2024-01-02" / "L2" / "orats__hist.strikes__SPY.meta.json").read_text())
    assert "set-cookie" not in meta["headers"]
    assert "authorization" not in meta["headers"]
    assert meta["headers"]["content-type"] == "application/json"
    assert meta["headers"]["x-echo"] == "fine"


def test_header_tracking_last_modified_yes(tmp_path, orats_key, capsys):
    a = _body()
    b = _body([{"updatedAt": "1700000000", "strike": 999}, {"updatedAt": "1700000100", "strike": 101}])
    finality_probe.capture(_ns(tmp_path, label="L1"), http_get=_fake_get(a, {"last-modified": "l1"}), now=_at(3))
    finality_probe.capture(_ns(tmp_path, label="L2"), http_get=_fake_get(b, {"last-modified": "l2"}), now=_at(4))
    capsys.readouterr()
    finality_probe.compare(argparse.Namespace(out=str(tmp_path), session=None))
    out = capsys.readouterr().out
    block = _group_block(out, "2024-01-02 orats hist/strikes SPY page=None")
    assert "last-modified=yes" in block
    assert "- header last-modified: header tracks content" in out


def test_capture_direct_arg_type_validation_exit_2_no_files(tmp_path, orats_key, capsys):
    with pytest.raises(SystemExit) as exc:
        finality_probe.capture(_ns(tmp_path), http_get=_fake_get(), now=0)
    assert exc.value.code == 2
    assert "invalid argument types" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())

    with pytest.raises(SystemExit) as exc:
        finality_probe.capture(_ns(tmp_path), http_get="x", now=_at(3))
    assert exc.value.code == 2
    assert "invalid argument types" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())

    ns = _ns(tmp_path)
    ns.replace = "false"
    with pytest.raises(SystemExit) as exc:
        finality_probe.capture(ns, http_get=_fake_get(), now=_at(3))
    assert exc.value.code == 2
    assert "invalid argument types" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())

    ns = _ns(tmp_path)
    ns.massive = 1
    with pytest.raises(SystemExit) as exc:
        finality_probe.capture(ns, http_get=_fake_get(), now=_at(3))
    assert exc.value.code == 2
    assert "invalid argument types" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())

    with pytest.raises(SystemExit) as exc:
        finality_probe.capture(_ns(str(tmp_path) + "\x00"), http_get=_fake_get(), now=_at(3))
    assert exc.value.code == 2
    assert "invalid argument types" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


def test_compare_direct_arg_type_validation_exit_2(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        finality_probe.compare(argparse.Namespace(out=None, session=None))
    assert exc.value.code == 2
    assert "invalid argument types" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        finality_probe.compare(argparse.Namespace(out=str(tmp_path) + "\x00", session=None))
    assert exc.value.code == 2
    assert "invalid argument types" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        finality_probe.compare(argparse.Namespace(out=str(tmp_path), session="2024-01-02\x00"))
    assert exc.value.code == 2
    assert "invalid argument types" in capsys.readouterr().err


def test_massive_last_updated_bool_and_nonintegral_float_ignored(tmp_path, massive_key):
    for i, lu in enumerate((True, 12.5)):
        label = f"L{i}"
        rc = finality_probe.capture(_ns(tmp_path, label=label, endpoints="", massive=True),
                                    http_get=_fake_get(_mbody([{"day": {"last_updated": lu}, "strike": 1}])),
                                    now=_at(3 + i))
        assert rc == 0
        meta = json.loads((tmp_path / "2024-01-02" / label / "massive__v3.snapshot.options__SPY__p0.meta.json").read_text())
        assert meta["max_updated_at"] is None


def test_massive_periodic_progress_includes_eta_max_and_tickers_left(tmp_path, massive_key, monkeypatch, capsys):
    class _StepTime:
        def __init__(self):
            self.t = 0.0

        def perf_counter(self):
            self.t += 31.0
            return self.t

    monkeypatch.setattr(finality_probe, "time", _StepTime())
    p0 = _mbody(MROWS, NEXT1)
    p1 = _mbody([{"day": {"last_updated": 1700000300}, "strike": 102}])
    rc = finality_probe.capture(_ns(tmp_path, endpoints="", massive=True, tickers="SPY,QQQ"),
                                http_get=_mk_get(p0, p1), now=_at(3))
    assert rc == 0
    err = capsys.readouterr().err
    lines = [ln for ln in err.splitlines() if "eta_max=" in ln]
    assert lines, err
    page_budget = finality_probe.MAX_MASSIVE_PAGES
    pages_per_ticker = 2 if json.loads(p0).get("next_url") else 1
    for ln in lines:
        total = int(ln.split("[capture] ")[1].split(" pages")[0])
        el = int(ln.split("elapsed=")[1].split("s")[0])
        page = (total - 1) % pages_per_ticker
        expected_eta = int((el - 62) / total * (page_budget - (page + 1)))
        actual_eta = int(ln.split("eta_max=")[1].split("s")[0])
        assert actual_eta == expected_eta, ln
    assert any("tickers_left=2" in ln for ln in lines), err
    assert any("tickers_left=1" in ln for ln in lines), err
    assert "massive ticker 1/2" in err
    assert "massive ticker 2/2" in err


def test_default_http_get_kwargs_close_and_success(monkeypatch):
    import requests

    calls = {}

    class FakeResp:
        status_code = 201
        headers = {"Content-Type": "application/json", "X-Mix": "Ab"}

        def __init__(self):
            self.closed = False

        def iter_content(self, chunk_size):
            yield b'{"data":[]}'

        def close(self):
            self.closed = True

    holder = {}

    def fake_get(url, **kwargs):
        calls["url"] = url
        calls["kwargs"] = kwargs
        holder["r"] = FakeResp()
        return holder["r"]

    monkeypatch.setattr(requests, "get", fake_get)
    status, hdrs, body = finality_probe._default_http_get(
        {"url": "https://api.orats.io/datav2/hist/strikes",
         "params": {"ticker": "SPY", "tradeDate": "2024-01-02", "token": SENTINEL},
         "headers": {"Accept-Encoding": "identity"}, "timeout": 30})
    assert calls["url"] == "https://api.orats.io/datav2/hist/strikes"
    assert calls["kwargs"]["params"] == {"ticker": "SPY", "tradeDate": "2024-01-02", "token": SENTINEL}
    timeout = calls["kwargs"]["timeout"]
    assert isinstance(timeout, (int, float)) and not isinstance(timeout, bool)
    assert timeout <= 60
    assert calls["kwargs"]["allow_redirects"] is False
    assert calls["kwargs"]["stream"] is True
    assert holder["r"].closed is True
    assert status == 201
    assert hdrs == {"content-type": "application/json", "x-mix": "Ab"}
    assert body == b'{"data":[]}'


def test_default_http_get_closes_when_body_cap_exceeded(monkeypatch):
    import requests

    monkeypatch.setattr(finality_probe, "MAX_BODY_BYTES", 10)

    class BigResp:
        status_code = 200
        headers = {}

        def __init__(self):
            self.closed = False

        def iter_content(self, chunk_size):
            yield b"x" * 64

        def close(self):
            self.closed = True

    holder = {}
    monkeypatch.setattr(requests, "get", lambda *a, **k: holder.setdefault("r", BigResp()))
    with pytest.raises(ValueError, match="BodyTooLarge"):
        finality_probe._default_http_get({"url": "https://example.com/x"})
    assert holder["r"].closed is True


def test_ts_two_instants_one_hundred_seconds_apart():
    pairs = [
        (1700000000, 1700000100),
        (1700000000000, 1700000100000),
        (1700000000000000000, 1700000100000000000),
        ("1700000000", "1700000100"),
        ("1700000000000", "1700000100000"),
        ("1700000000000000000", "1700000100000000000"),
        ("2023-11-14T22:13:20Z", "2023-11-14T22:15:00Z"),
    ]
    for a, b in pairs:
        ta, tb = finality_probe._ts(a), finality_probe._ts(b)
        assert ta is not None and tb is not None, (a, b)
        assert tb - ta == 100.0, (a, b, ta, tb)


def test_capture_preflight_duplicate_endpoints_exits_2_before_dirs(tmp_path, orats_key, capsys):
    called = []

    def boom(req):
        called.append(req)
        raise AssertionError("network must not be called")

    with pytest.raises(SystemExit) as exc:
        finality_probe.capture(_ns(tmp_path, endpoints="hist/strikes,hist/strikes"), http_get=boom)
    assert exc.value.code == 2
    assert "duplicate ORATS endpoints" in capsys.readouterr().err
    assert called == []
    assert not any(tmp_path.iterdir())


def test_capture_lock_preexisting_refuses_and_writes_nothing(tmp_path, orats_key, capsys):
    unit = tmp_path / "2024-01-02" / "L1"
    unit.mkdir(parents=True, mode=0o700)
    lock = unit / ".capture.lock"
    lock.write_text("99999", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        finality_probe.capture(_ns(tmp_path), http_get=_fake_get(), now=_at(3))
    assert exc.value.code == 2
    assert "locked by another capture" in capsys.readouterr().err
    assert not list(unit.glob("*.body"))
    assert not list(unit.glob("*.meta.json"))
    assert lock.read_text(encoding="utf-8") == "99999"


def test_capture_lock_removed_after_normal_and_failed_capture(tmp_path, orats_key, capsys):
    unit = tmp_path / "2024-01-02" / "L1"
    rc = finality_probe.capture(_ns(tmp_path), http_get=_fake_get(), now=_at(3))
    assert rc == 0
    assert not (unit / ".capture.lock").exists()

    leaky = json.dumps({"data": ROWS, "note": SENTINEL}).encode("utf-8")
    with pytest.raises(SystemExit) as exc:
        finality_probe.capture(_ns(tmp_path, label="L2"), http_get=_fake_get(leaky))
    assert exc.value.code == 2
    capsys.readouterr()
    assert not (tmp_path / "2024-01-02" / "L2" / ".capture.lock").exists()


def test_compare_ignores_stale_capture_lock(tmp_path, orats_key, capsys):
    finality_probe.capture(_ns(tmp_path), http_get=_fake_get(), now=_at(3))
    unit = tmp_path / "2024-01-02" / "L1"
    (unit / ".capture.lock").write_text("12345", encoding="utf-8")
    capsys.readouterr()
    finality_probe.compare(argparse.Namespace(out=str(tmp_path), session=None))
    captured = capsys.readouterr()
    assert "capture.lock" not in captured.out
    assert "capture.lock" not in captured.err
    assert "- groups: 1" in captured.out
    assert "## 2024-01-02 orats hist/strikes SPY page=None" in captured.out


def test_fetch_time_taken_immediately_after_http_get_returns(tmp_path, orats_key):
    class _Clock:
        def __init__(self):
            self.n = 0

        def now(self):
            v = datetime(2024, 1, 3, 0, 0, self.n, tzinfo=timezone.utc)
            self.n += 1
            return v

    clock = _Clock()
    state = {"before": None}

    def http_get(req):
        state["before"] = clock.n
        return 200, {"content-type": "application/json"}, _body()

    finality_probe.capture(_ns(tmp_path), http_get=http_get, now=clock.now)
    assert state["before"] == 0
    meta = json.loads((tmp_path / "2024-01-02" / "L1" / "orats__hist.strikes__SPY.meta.json").read_text())
    assert meta["fetched_at_utc"] == datetime(2024, 1, 3, 0, 0, 0, tzinfo=timezone.utc).isoformat()


def test_compare_never_prints_response_body_marker(tmp_path, orats_key, capsys):
    rows = [{"updatedAt": "1700000000", "strike": 100, "note": "BODYMARKER-123"}]
    finality_probe.capture(_ns(tmp_path), http_get=_fake_get(_body(rows)), now=_at(3))
    capsys.readouterr()
    finality_probe.compare(argparse.Namespace(out=str(tmp_path), session=None))
    captured = capsys.readouterr()
    assert "BODYMARKER-123" not in captured.out
    assert "BODYMARKER-123" not in captured.err
    assert "- groups: 1" in captured.out


def test_capture_preflight_massive_key_ticker_endpoint_and_repo(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ORATS_API_KEY", SENTINEL)
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    called = []

    def boom(req):
        called.append(req)
        raise AssertionError("network must not be called")

    with pytest.raises(SystemExit) as exc:
        finality_probe.capture(_ns(tmp_path, endpoints="", massive=True), http_get=boom)
    assert exc.value.code == 2
    assert "MASSIVE_API_KEY" in capsys.readouterr().err

    with pytest.raises(SystemExit) as exc:
        finality_probe.capture(_ns(tmp_path, tickers="SP Y"), http_get=boom)
    assert exc.value.code == 2
    assert "invalid ticker" in capsys.readouterr().err

    with pytest.raises(SystemExit) as exc:
        finality_probe.capture(_ns(tmp_path, endpoints="../x"), http_get=boom)
    assert exc.value.code == 2
    assert "invalid ORATS endpoint" in capsys.readouterr().err

    repo_root = Path(__file__).resolve().parents[3]
    with pytest.raises(SystemExit) as exc:
        finality_probe.capture(_ns(repo_root / "tests"), http_get=boom)
    assert exc.value.code == 2
    assert "refusing to write provider data inside a repository" in capsys.readouterr().err

    assert called == []


def test_injected_oversized_body_never_persisted(tmp_path, orats_key, monkeypatch):
    monkeypatch.setattr(finality_probe, "MAX_BODY_BYTES", 50)
    big = json.dumps({"data": [{"updatedAt": "1700000000", "strike": "x" * 200}]}).encode("utf-8")
    assert len(big) > finality_probe.MAX_BODY_BYTES
    rc = finality_probe.capture(_ns(tmp_path), http_get=_fake_get(big), now=_at(3))
    assert rc == 1
    unit = tmp_path / "2024-01-02" / "L1"
    meta = json.loads((unit / "orats__hist.strikes__SPY.meta.json").read_text())
    assert meta["status"] == 0
    assert meta["error"] == "ValueError"
    assert meta["ok"] is False
    assert meta["body_bytes"] == 0
    assert (unit / "orats__hist.strikes__SPY.body").read_bytes() == b""


def test_capture_no_provider_selected_exits_2_before_dirs(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        finality_probe.capture(_ns(tmp_path, endpoints="", massive=False), http_get=_fake_get())
    assert exc.value.code == 2
    assert "no provider selected" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


def test_post_lock_label_recheck_removes_lock(tmp_path, orats_key, monkeypatch, capsys):
    unit = tmp_path / "2024-01-02" / "L1"
    real_open = finality_probe.os.open
    seen = []

    def fake_open(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        if not seen and str(path).endswith(".capture.lock"):
            seen.append(True)
            unit.mkdir(parents=True, exist_ok=True)
            (unit / "x.meta.json").write_text("{}", encoding="utf-8")
        return fd

    monkeypatch.setattr(finality_probe.os, "open", fake_open)
    with pytest.raises(SystemExit) as exc:
        finality_probe.capture(_ns(tmp_path), http_get=_fake_get(), now=_at(3))
    assert exc.value.code == 2
    assert "label already captured; pass --replace to invalidate the prior capture" in capsys.readouterr().err
    assert not (unit / ".capture.lock").exists()
