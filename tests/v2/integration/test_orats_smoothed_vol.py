"""tools/orats_smoothed_vol.py on synthetic raw and curated trees (all under tmp_path)."""
# packages: engine.v2.foundation
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))

import orats_smoothed_vol as tool  # noqa: E402

K_EARLY_BIG = "f" * 64   # fetched first, LARGER cache key
K_LATE_SMALL = "1" * 64  # fetched later, SMALLER cache key: must win
K_NO_SMV = "2" * 64      # latest fetch, no smvVol column: must not override
K_RATES = "4" * 64
K_BROKEN = "3" * 64


def _payload(rows):
    return {"data": rows}


def _quote(ticker, strike, smv=None):
    row = {"ticker": ticker, "tradeDate": "2025-01-02", "expirDate": "2025-02-21", "strike": strike,
           "callMidIv": 0.21, "putMidIv": 0.22, "stockPrice": 500.0}
    if smv is not None:
        row["smvVol"] = smv
    return row


def _entry(raw, endpoint, key, payload, fetched_at, body=None):
    d = raw / "orats" / key[:2]
    d.mkdir(parents=True, exist_ok=True)
    data = body if body is not None else gzip.compress(json.dumps(payload).encode("utf-8"))
    (d / f"{key}.body.gz").write_bytes(data)
    meta = {"key": key, "source": "orats", "endpoint": endpoint, "params": {},
            "fetched_at": fetched_at, "note": "synthetic"}
    (d / f"{key}.meta.json").write_text(json.dumps(meta), encoding="utf-8")


def _body(rows):
    return gzip.compress(json.dumps(_payload(rows)).encode("utf-8"), compresslevel=0, mtime=0)


def _chains(chains, year, rows):
    d = chains / f"year={year}"
    d.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    df["expiry"] = pd.to_datetime(df["expiry"])
    df.to_parquet(d / "part-0000.parquet", index=False)


def _world(base: Path):
    raw, chains = base / "raw", base / "curated" / "option_chains"
    _entry(raw, "hist/strikes", K_EARLY_BIG,
           _payload([_quote("SPY", 100.0, 0.20), _quote("SPY", 105.0, 3.4e38)]),
           "2025-01-01T00:00:00+00:00")
    _entry(raw, "hist/strikes", K_LATE_SMALL, _payload([_quote("SPY", 100.0, 0.25)]),
           "2025-01-05T00:00:00+00:00")
    _entry(raw, "hist/strikes", K_NO_SMV, _payload([_quote("SPY", 100.0)]),
           "2025-01-09T00:00:00+00:00")
    _entry(raw, "hist/cores", K_RATES, _payload([
        {"ticker": "SPY", "tradeDate": "2025-01-02", "iRate5wk": 0.04, "iRateLt": 0.045, "annActDiv": 1.5},
        {"ticker": "SPY", "tradeDate": "2025-01-03", "iRate5wk": 0.04, "iRateLt": 0.045, "annActDiv": None},
    ]), "2025-01-02T00:00:00+00:00")
    _chains(chains, 2025, [
        {"ticker": "SPY", "obs_date": "2025-01-02", "expiry": "2025-02-21", "strike": 100.0,
         "chain_kind": "american"},
        {"ticker": "SPY", "obs_date": "2025-01-02", "expiry": "2025-02-21", "strike": 999.0,
         "chain_kind": "american"},
        {"ticker": "SPX", "obs_date": "2025-01-02", "expiry": "2025-02-21", "strike": 100.0,
         "chain_kind": "european"},
    ])
    return raw, chains


def _args(raw, chains, out, work):
    return ["--raw-root", str(raw), "--curated-chains", str(chains),
            "--out", str(out), "--work-dir", str(work)]


def _tree(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_end_to_end(tmp_path):
    raw, chains = _world(tmp_path)
    before = (_tree(raw), _tree(chains))
    out = tmp_path / "out"
    assert tool.main(_args(raw, chains, out, tmp_path / "work")) == 0

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    quotes_m = manifest["tables"]["quotes"]
    assert quotes_m["duplicates_dropped"] == 1
    assert quotes_m["conflicting_keys"] == 1
    assert quotes_m["without_required_fields"] == 1

    q = pd.read_parquet(out / "quotes" / "year=2025" / "month=01.parquet").set_index("strike")
    assert q.loc[100.0, "smvVol"] == 0.25        # later fetched_at wins, not key order
    assert pd.isna(q.loc[105.0, "smvVol"])       # 3.4e38 sentinel -> NaN
    row = q.loc[100.0]
    assert (row["smvVol"], row["callMidIv"], row["putMidIv"], row["stockPrice"]) == (0.25, 0.21, 0.22, 500.0)

    r = pd.read_parquet(out / "rates" / "year=2025" / "month=01.parquet")
    assert list(r.columns)[:2] == ["ticker", "obs_date"]
    assert len(r) == 2
    assert r["iRate5wk"].tolist() == [0.04, 0.04]
    assert r["iRateLt"].tolist() == [0.045, 0.045]
    assert r["obs_date"].dt.strftime("%Y-%m-%d").tolist() == ["2025-01-02", "2025-01-03"]
    assert r["iRate5wk"].notna().all()

    cov = json.loads((out / "coverage.json").read_text(encoding="utf-8"))
    rows = {(c["year"], c["chain_kind"]): c for c in cov["coverage"]}
    am = rows[(2025, "american")]
    assert (am["rows"], am["key_matched"], am["smv_matched"], am["rates_matched"]) == (2, 1, 1, 2)
    assert rows[(2025, "european")]["rows"] == 1
    assert rows[(2025, "all")]["rows"] == 3
    assert rows[(2025, "all")]["smv_share"] == 0.333333
    assert rows[(2019, "all")] == {"year": 2019, "chain_kind": "all", "rows": 0, "key_matched": 0,
                                   "smv_matched": 0, "rates_matched": 0, "smv_share": None}
    assert cov["dividend_fields"] == {"annActDiv": 1}
    assert (_tree(raw), _tree(chains)) == before


def test_idempotent_and_deterministic(tmp_path):
    raw, chains = _world(tmp_path)
    out1, out2 = tmp_path / "o1", tmp_path / "o2"
    assert tool.main(_args(raw, chains, out1, tmp_path / "w1")) == 0
    assert tool.main(_args(raw, chains, out2, tmp_path / "w2")) == 0
    assert _tree(out1) == _tree(out2)
    assert not (tmp_path / "w1").exists()

    before = _tree(out1)
    assert tool.main(_args(raw, chains, out1, tmp_path / "w3")) == 0
    assert _tree(out1) == before
    assert not (tmp_path / "w3").exists()


def test_work_dir_keeps_unrelated_files(tmp_path):
    raw, chains = _world(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    (work / "keep.txt").write_text("mine", encoding="utf-8")
    assert tool.main(_args(raw, chains, tmp_path / "out", work)) == 0
    assert (work / "keep.txt").read_text(encoding="utf-8") == "mine"
    assert not (work / "shards").exists()
    assert not (work / "markers").exists()


def test_resume_after_merge_crash(tmp_path, monkeypatch):
    raw, chains = _world(tmp_path)
    out, work = tmp_path / "out", tmp_path / "work"
    args = _args(raw, chains, out, work)

    def crash(*_a, **_k):
        raise RuntimeError("simulated merge crash")

    monkeypatch.setattr(tool, "merge_stage", crash)
    with pytest.raises(RuntimeError):
        tool.main(args)
    assert not (out / "manifest.json").exists()
    assert list((work / "markers").rglob("*.json"))
    monkeypatch.undo()

    parsed = []
    real = tool.parse_entry

    def counting(*args):
        parsed.append(args[1].key)
        return real(*args)

    monkeypatch.setattr(tool, "parse_entry", counting)
    assert tool.main(args) == 0
    assert parsed == []
    assert tool.main(_args(raw, chains, tmp_path / "fresh", tmp_path / "fresh_work")) == 0
    assert _tree(out) == _tree(tmp_path / "fresh")


def test_resume_reparses_damaged_stage(tmp_path, monkeypatch):
    raw, chains = _world(tmp_path)
    out, work = tmp_path / "out", tmp_path / "work"
    args = _args(raw, chains, out, work)

    def crash(*_a, **_k):
        raise RuntimeError("simulated merge crash")

    monkeypatch.setattr(tool, "merge_stage", crash)
    with pytest.raises(RuntimeError):
        tool.main(args)
    monkeypatch.undo()

    (work / "shards" / "quotes" / "2025-01" / f"{K_EARLY_BIG}.parquet").unlink()
    (work / "shards" / "quotes" / "2025-01" / f"{K_LATE_SMALL}.parquet").write_bytes(b"junk")
    (work / "markers" / "rates" / f"{K_RATES}.json").write_bytes(b"{not json")

    parsed = []
    real = tool.parse_entry

    def counting(*args):
        parsed.append(args[1].key)
        return real(*args)

    monkeypatch.setattr(tool, "parse_entry", counting)
    assert tool.main(args) == 0
    assert sorted(parsed) == sorted([K_EARLY_BIG, K_LATE_SMALL, K_RATES])
    assert tool.main(_args(raw, chains, tmp_path / "fresh", tmp_path / "fresh_work")) == 0
    assert _tree(out) == _tree(tmp_path / "fresh")


def test_digest_matches_parsed_bytes(tmp_path, monkeypatch):
    raw, chains = _world(tmp_path)
    out = tmp_path / "out"
    real = tool.parse_entry

    def tamper(table, entry, blob):
        if entry.key == K_LATE_SMALL:
            Path(entry.path).write_bytes(b"junk")
        return real(table, entry, blob)

    monkeypatch.setattr(tool, "parse_entry", tamper)
    assert tool.main(_args(raw, chains, out, tmp_path / "work")) == 0
    q = pd.read_parquet(out / "quotes" / "year=2025" / "month=01.parquet").set_index("strike")
    assert q.loc[100.0, "smvVol"] == 0.25


def test_unreadable_body_is_counted(tmp_path):
    raw, chains = _world(tmp_path)
    _entry(raw, "hist/strikes", K_BROKEN, None, "2025-01-10T00:00:00+00:00", body=b"not gzip")
    _entry(raw, "hist/strikes", "5" * 64, None, "2025-01-11T00:00:00+00:00",
           body=b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03" + b"\xff" * 32)
    out = tmp_path / "out"
    assert tool.main(_args(raw, chains, out, tmp_path / "work")) == 1
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["tables"]["quotes"]["unreadable"] == 2
    assert manifest["tables"]["quotes"]["rows_out"] == 2


@pytest.mark.parametrize("where", ["inside_curated", "equals_raw", "ancestor_of_curated"])
def test_guard_refuses_protected_out(tmp_path, tmp_path_factory, where):
    raw, chains = _world(tmp_path / "base")
    out = {"inside_curated": chains / "sub", "equals_raw": raw, "ancestor_of_curated": tmp_path}[where]
    before = _tree(tmp_path)
    with pytest.raises(SystemExit, match="overlaps"):
        tool.main(_args(raw, chains, out, tmp_path_factory.mktemp("work")))
    assert _tree(tmp_path) == before


def test_equal_fetched_at_ties(tmp_path):
    raw, chains = tmp_path / "raw", tmp_path / "curated" / "option_chains"
    t = "2025-01-04T00:00:00+00:00"
    # "a" body: strike 100 (0.1), then strike 105 twice (0.4 then 0.5)
    _entry(raw, "hist/strikes", "a" * 64, _payload([
        _quote("SPY", 100.0, 0.1), _quote("SPY", 105.0, 0.4), _quote("SPY", 105.0, 0.5)]), t)
    # "b" body: strike 100 twice (0.2 then 0.3); greater cache key wins the tie
    _entry(raw, "hist/strikes", "b" * 64, _payload([
        _quote("SPY", 100.0, 0.2), _quote("SPY", 100.0, 0.3)]), t)
    _entry(raw, "hist/cores", K_RATES, _payload([
        {"ticker": "SPY", "tradeDate": "2025-01-02", "iRate5wk": 0.04, "iRateLt": 0.045}]), t)
    _chains(chains, 2025, [{"ticker": "SPY", "obs_date": "2025-01-02", "expiry": "2025-02-21",
                            "strike": 100.0, "chain_kind": "american"}])
    out = tmp_path / "out"
    assert tool.main(_args(raw, chains, out, tmp_path / "work")) == 0
    q = pd.read_parquet(out / "quotes" / "year=2025" / "month=01.parquet").set_index("strike")
    assert q.loc[100.0, "smvVol"] == 0.3
    assert q.loc[105.0, "smvVol"] == 0.5


def test_changed_body_same_size_is_reparsed(tmp_path, monkeypatch):
    raw, chains = _world(tmp_path)
    out, work = tmp_path / "out", tmp_path / "work"
    args = _args(raw, chains, out, work)
    early = _body([_quote("SPY", 100.0, 0.25)])
    late = _body([_quote("SPY", 100.0, 0.35)])
    assert len(early) == len(late)
    _entry(raw, "hist/strikes", K_LATE_SMALL, None, "2025-01-05T00:00:00+00:00", body=early)

    def crash(*_a, **_k):
        raise RuntimeError("simulated merge crash")

    monkeypatch.setattr(tool, "merge_stage", crash)
    with pytest.raises(RuntimeError):
        tool.main(args)
    _entry(raw, "hist/strikes", K_LATE_SMALL, None, "2025-01-05T00:00:00+00:00", body=late)
    monkeypatch.undo()
    assert tool.main(args) == 0
    q = pd.read_parquet(out / "quotes" / "year=2025" / "month=01.parquet").set_index("strike")
    assert q.loc[100.0, "smvVol"] == 0.35


def test_body_size_cap_counts_unreadable(tmp_path, monkeypatch, caplog):
    raw, chains = _world(tmp_path)
    out = tmp_path / "out"
    monkeypatch.setattr(tool, "MAX_BODY_BYTES", 10)
    caplog.set_level("WARNING", logger="smvvol")
    assert tool.main(_args(raw, chains, out, tmp_path / "work")) == 1
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["tables"]["quotes"]["unreadable"] == 3
    assert manifest["tables"]["rates"]["unreadable"] == 1
    assert "compressed body exceeds 10 bytes" in caplog.text
    assert "decompressed body" not in caplog.text


def test_decompressed_cap_counts_unreadable(tmp_path, monkeypatch, caplog):
    raw, chains = _world(tmp_path)
    out = tmp_path / "out"
    cap = 20000
    big = gzip.compress(json.dumps(_payload([_quote("SPY", 100.0, 0.2)] * 2000)).encode("utf-8"))
    big_compressed = len(big)
    big_decompressed = len(gzip.decompress(big))
    assert big_compressed < cap < big_decompressed
    _entry(raw, "hist/strikes", "6" * 64, None, "2025-01-12T00:00:00+00:00", body=big)
    for p in raw.rglob("*.body.gz"):
        if p.name == f"{'6' * 64}.body.gz":
            continue
        assert len(p.read_bytes()) < cap
        assert len(gzip.decompress(p.read_bytes())) < cap
    monkeypatch.setattr(tool, "MAX_BODY_BYTES", cap)
    caplog.set_level("WARNING", logger="smvvol")
    assert tool.main(_args(raw, chains, out, tmp_path / "work")) == 1
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["tables"]["quotes"]["unreadable"] == 1
    assert "decompressed body exceeds 20000 bytes" in caplog.text


def test_meta_size_cap_refuses_before_writing(tmp_path, monkeypatch):
    raw, chains = _world(tmp_path)
    out = tmp_path / "out"
    monkeypatch.setattr(tool, "MAX_META_BYTES", 10)
    with pytest.raises(ValueError, match="exceeds cap"):
        tool.main(_args(raw, chains, out, tmp_path / "work"))
    assert not out.exists()
