"""Incremental Tier-2 rebuild: reused parses must give byte-identical tables."""
from __future__ import annotations

import gzip
import hashlib
import json
import random
import string

import pytest

from engine.data import rebuild_cache


def _gz(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as fh:
        json.dump(obj, fh)


def _summ(day, px):
    return {"tradeDate": day, "stockPrice": px, "iv10d": 0.3, "iv30d": 0.31,
            "exErnIv10d": 0.29, "exErnIv30d": 0.3, "impliedMove": 0.05, "rVol30": 0.2,
            "skewing": 0.1, "contango": 0.02, "fwd90_30": 0.3, "fexErn90_30": 0.3,
            "ieeEarnEffect": 0.1}


def _strike(ticker, day, strike, bid=2.0):
    return {"ticker": ticker, "tradeDate": day, "expirDate": "2026-09-18", "strike": strike,
            "stockPrice": 100.0, "spotPrice": 100.0, "callBidPrice": bid,
            "callAskPrice": bid + 0.4, "callMidIv": 0.3, "putBidPrice": 1.0,
            "putAskPrice": 1.4, "putMidIv": 0.33, "delta": 0.5}


def _fetch(paths, endpoint, params, payload):
    key = hashlib.sha256(json.dumps([endpoint, params], sort_keys=True).encode()).hexdigest()
    base = paths.RAW_FETCH / "orats" / key[:2]
    _gz(base / f"{key}.body.gz", payload)
    (base / f"{key}.meta.json").write_text(json.dumps(
        {"key": key, "source": "orats", "endpoint": endpoint, "params": params}))
    return base / f"{key}.body.gz"


@pytest.fixture
def world(tmp_root, monkeypatch):
    from engine import paths
    from engine.data.normalize import n_daily

    monkeypatch.delenv(rebuild_cache.FORCE_ENV, raising=False)
    n_daily.fetch_daily_index.cache_clear()
    for i, t in enumerate(("AAA", "BBB", "CCC")):
        _gz(paths.RAW_ORATS_SUMMARIES / f"{t}.json.gz",
            [_summ(f"2024-01-0{d}", 10.0 + i + d) for d in range(2, 6)])
        _gz(paths.RAW_ORATS_CORES / f"{t}.json.gz",
            [{"tradeDate": "2024-01-02", "mktCap": 5000.0 + i}])
        _gz(paths.RAW_ORATS_STRIKES / f"2024-01-0{2 + i}_b1.json.gz",
            {"rows": [_strike(t, f"2024-01-0{2 + i}", s) for s in (90.0, 100.0)]})
    # A crossed-quote + duplicate file that quarantines, and a fetch overlap.
    _gz(paths.RAW_ORATS_STRIKES / "2024-01-02_c2_b1.json.gz",
        {"rows": [_strike("AAA", "2024-01-02", 100.0, bid=9.0)] * 2})
    _fetch(paths, "hist/strikes", {"ticker": "AAA", "tradeDate": "2024-01-03"},
           {"data": [_strike("AAA", "2024-01-03", 95.0)]})
    _fetch(paths, "hist/strikes", {"ticker": "BBB", "tradeDate": "2024-01-04"}, {"data": []})
    _fetch(paths, "hist/summaries", {"tradeDate": "2024-01-08"},
           {"data": [dict(_summ("2024-01-08", 20.0), ticker="AAA")]})
    return paths


def _snapshot(paths):
    out = {}
    for table in ("daily_market", "option_chains"):
        base = paths.curated_table(table)
        for p in sorted(base.rglob("*")):
            if p.is_file():
                out[str(p.relative_to(paths.CURATED))] = p.read_bytes()
    return out


def _full(paths):
    from engine.data import rebuild

    reports = (rebuild.build_daily_table(), rebuild.build_chains_table())
    return _snapshot(paths), reports


def _incr(paths, **kw):
    from engine.data import rebuild
    from engine.data.normalize import n_daily

    n_daily.fetch_daily_index.cache_clear()
    caches = [rebuild_cache.InputCache(t, **kw) for t in ("daily_market", "option_chains")]
    reports = (rebuild.build_daily_table(None, caches[0]), rebuild.build_chains_table(None, caches[1]))
    for c in caches:
        c.commit()
    return _snapshot(paths), reports, caches


def _parsed(caches):
    return [c.stats["parsed"] for c in caches]


def test_cold_run_is_full_and_matches(world):
    full = _full(world)
    snap, reports, caches = _incr(world)
    assert snap == full[0] and reports == full[1]
    assert all(c.full for c in caches) and all(p > 0 for p in _parsed(caches))


def test_warm_run_parses_nothing_and_matches(world):
    full = _full(world)
    _incr(world)
    snap, reports, caches = _incr(world)
    assert snap == full[0] and reports == full[1]
    assert _parsed(caches) == [0, 0] and not any(c.full for c in caches)
    assert all(c.stats["reused"] > 0 for c in caches)


@pytest.mark.parametrize("change", ["changed", "added", "removed"])
def test_one_file_change_reparses_only_it(world, change):
    _incr(world)
    target = world.RAW_ORATS_STRIKES / "2024-01-03_b1.json.gz"
    if change == "changed":
        _gz(target, {"rows": [_strike("BBB", "2024-01-03", 100.0, bid=3.0)]})
    elif change == "added":
        _gz(world.RAW_ORATS_STRIKES / "2024-01-09_b1.json.gz",
            {"rows": [_strike("CCC", "2024-01-09", 100.0)]})
    else:
        target.unlink()
    snap, reports, caches = _incr(world)
    assert _parsed(caches) == [0, 0 if change == "removed" else 1]
    if change == "removed":
        assert caches[1].stats["removed"] == 1
    assert (snap, reports) == _full(world)


def test_daily_file_change_and_new_market_day(world):
    _incr(world)
    _gz(world.RAW_ORATS_SUMMARIES / "BBB.json.gz", [_summ("2024-01-02", 55.0)])
    _fetch(world, "hist/summaries", {"tradeDate": "2024-01-09"},
           {"data": [dict(_summ("2024-01-09", 21.0), ticker=t) for t in ("AAA", "CCC")]})
    snap, reports, caches = _incr(world)
    assert caches[0].stats["parsed"] == 1
    assert (snap, reports) == _full(world)


def test_corrupt_manifest_falls_back_to_full(world):
    _incr(world)
    (rebuild_cache.cache_root() / "option_chains" / "manifest.json").write_text("{not json")
    snap, reports, caches = _incr(world)
    assert caches[1].full and caches[1].stats["reused"] == 0
    assert (snap, reports) == _full(world)


def test_corrupt_entry_is_reparsed(world):
    _incr(world)
    for p in (rebuild_cache.cache_root() / "option_chains" / "entries").rglob("*.pkl"):
        p.write_bytes(b"garbage")
    snap, reports, caches = _incr(world)
    assert caches[1].stats["reused"] == 0 and caches[1].stats["parsed"] > 0
    assert (snap, reports) == _full(world)


def test_code_version_change_forces_full(world, monkeypatch):
    _incr(world)
    monkeypatch.setattr(rebuild_cache, "CACHE_SCHEMA", rebuild_cache.CACHE_SCHEMA + 1)
    snap, reports, caches = _incr(world)
    assert all(c.fallback_reason == "code version changed" for c in caches)
    assert (snap, reports) == _full(world)


def test_env_var_forces_full(world, monkeypatch):
    _incr(world)
    monkeypatch.setenv(rebuild_cache.FORCE_ENV, "1")
    snap, reports, caches = _incr(world)
    assert all(c.fallback_reason == "forced" for c in caches)
    assert (snap, reports) == _full(world)


def test_default_rebuild_never_touches_the_cache(world, monkeypatch):
    from engine.data import rebuild

    def refuse(*a, **k):
        raise AssertionError("default rebuild constructed an InputCache")

    monkeypatch.setattr(rebuild, "InputCache", refuse)
    rebuild.rebuild(tables=("daily", "chains"))
    assert not rebuild_cache.cache_root().exists()


def test_incremental_rebuild_writes_the_cache(world):
    from engine.data import rebuild

    rebuild.rebuild(tables=("daily", "chains"), incremental=True)
    assert (rebuild_cache.cache_root() / "option_chains" / "manifest.json").exists()


def _rewrite_same_size(path, obj, size):
    """Rewrite ``path`` as gzipped ``obj``, padded to exactly ``size`` bytes."""
    base = json.dumps(obj)
    rng = random.Random(0)
    pad = "".join(rng.choices(string.ascii_letters + string.digits, k=8192))
    for n in range(len(pad) + 1):
        text = base[:-1] + ', "pad": "' + pad[:n] + '"}'
        with gzip.open(path, "wt") as fh:
            fh.write(text)
        if path.stat().st_size == size:
            return
    raise AssertionError(f"could not rewrite {path.name} to {size} bytes")


def test_same_size_rewrite_with_restored_mtime_is_reparsed(world):
    """size+mtime are forgeable (cp -p, touch -r); the inode/ctime guard is not."""
    import os

    _incr(world)
    path = world.RAW_ORATS_STRIKES / "2024-01-03_b1.json.gz"
    st = path.stat()
    _rewrite_same_size(path, {"rows": []}, st.st_size)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert path.stat().st_size == st.st_size
    assert path.stat().st_mtime_ns == st.st_mtime_ns
    snap, reports, caches = _incr(world)
    assert caches[1].stats["parsed"] == 1
    assert caches[1].stats["hashed"] >= 1
    assert (snap, reports) == _full(world)


def test_old_format_manifest_entry_is_rehashed_not_an_error(world):
    _incr(world)
    manifest_path = rebuild_cache.cache_root() / "option_chains" / "manifest.json"
    doc = json.loads(manifest_path.read_text())
    for rec in doc["inputs"].values():
        rec.pop("st_ino", None)
        rec.pop("st_ctime_ns", None)
    manifest_path.write_text(json.dumps(doc, sort_keys=True))
    snap, reports, caches = _incr(world)
    assert caches[1].stats["hashed"] >= 1
    assert caches[1].stats["reused"] > 0
    assert (snap, reports) == _full(world)


def test_sample_with_incremental_is_refused(world):
    from engine.data import rebuild

    with pytest.raises(ValueError, match="--sample cannot be combined with --incremental"):
        rebuild.rebuild(tables=("daily", "chains"), sample=1, incremental=True)
    assert not rebuild_cache.cache_root().exists()


def test_cli_refuses_sample_with_incremental(capsys):
    from engine.data import rebuild

    with pytest.raises(SystemExit) as exc:
        rebuild.main(["--sample", "1", "--incremental"])
    assert exc.value.code == 2
    assert "--sample cannot be combined with --incremental" in capsys.readouterr().err


@pytest.mark.parametrize("cached_int", [True, False])
def test_mixed_mktcap_dtypes_match_full_rebuild(world, cached_int):
    """A cached part's int mktCap and a fresh part's None (and the reverse)."""
    from engine.data.normalize import n_daily

    legacy = world.RAW_ORATS_CORES / "AAA.json.gz"
    history = n_daily.history_body_path("AAA", "cores")
    _gz(history, {"data": [{"tradeDate": "2024-01-03",
                            "mktCap": 5000 if cached_int else None}]})
    _incr(world)
    _gz(legacy, [{"tradeDate": "2024-01-02",
                  "mktCap": None if cached_int else 5000}])
    snap, reports, caches = _incr(world)
    assert (snap, reports) == _full(world)


def test_proof_tool_compare_trees_names_differences(tmp_path):
    from tools.verify_incremental_rebuild import compare_trees

    a, b = tmp_path / "a", tmp_path / "b"
    for base, payload in ((a, b"x"), (b, b"y")):
        (base / "t").mkdir(parents=True)
        (base / "t" / "p.parquet").write_bytes(payload)
    (a / "t" / "only.parquet").write_bytes(b"z")
    assert compare_trees(a, b) == ["only in full: t/only.parquet", "bytes differ: t/p.parquet"]
    assert compare_trees(a, a) == []
