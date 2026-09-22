"""Change C: the tier-0 battery verdict cache in ``checks.phase4_real``.

Unit-level only -- these exercise the pure cache helpers directly
(``_battery_cache_path``, ``_peek_corpus_hash``, ``_load_battery_cache_entry``,
``_write_battery_cache_entry``, ``_implementation_hash``) with small synthetic
receipts and a ``tmp_path`` cache file. None of this runs ``build_evidence``,
``run_corpus`` or anything that touches a real corpus -- that stays out of
scope for a unit test (see the change's own hand-back for what is measured
vs. left unverified).
"""
from __future__ import annotations

import json

import pytest

from checks import phase4_real as p4
from engine.v2.diagnosis import AGREE, DIFFER, compare_records


def _agree_receipt(kind: str = "x"):
    return compare_records({"a": 1}, {"a": 1}, comparison_kind=kind)


def _differ_receipt(kind: str = "y"):
    return compare_records({"a": 1}, {"a": 2}, comparison_kind=kind)


@pytest.fixture
def cache_path(tmp_path, monkeypatch):
    path = tmp_path / "cache" / "phase4_battery_cache.json"
    monkeypatch.setenv("PHASE4_BATTERY_CACHE_PATH", str(path))
    return path


# --------------------------------------------------------------------------
# location
# --------------------------------------------------------------------------


def test_battery_cache_path_honors_env_override(cache_path):
    assert p4._battery_cache_path() == cache_path


def test_battery_cache_path_default_is_outside_the_repo(monkeypatch):
    monkeypatch.delenv("PHASE4_BATTERY_CACHE_PATH", raising=False)
    path = p4._battery_cache_path()
    assert "fixtures" not in path.parts
    assert str(path).startswith("/root/")


# --------------------------------------------------------------------------
# fail-open reads
# --------------------------------------------------------------------------


def test_missing_cache_file_is_a_miss(cache_path):
    assert p4._load_battery_cache_entry("sha256:aa", "sha256:bb") is None


def test_corrupt_cache_file_is_a_miss(cache_path):
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("{not json")
    assert p4._load_battery_cache_entry("sha256:aa", "sha256:bb") is None


def test_wrong_schema_version_is_a_miss(cache_path):
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text(json.dumps({"schema_version": "bogus.v9", "entries": {}}))
    assert p4._load_battery_cache_entry("sha256:aa", "sha256:bb") is None


def test_a_malformed_entries_table_is_a_miss(cache_path):
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text(json.dumps({
        "schema_version": p4._BATTERY_CACHE_SCHEMA, "entries": "not-a-dict",
    }))
    assert p4._load_battery_cache_entry("sha256:aa", "sha256:bb") is None


def test_an_entry_that_is_not_a_genuine_agree_is_never_a_hit(cache_path):
    """A cache that could turn a broken verdict into a skip is unacceptable
    -- even a hand-edited/corrupted file claiming DIFFER under 'verdict'
    must still be treated as a miss, never resurrected as a pass."""
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text(json.dumps({
        "schema_version": p4._BATTERY_CACHE_SCHEMA,
        "entries": {"sha256:aa::sha256:bb": {
            "corpus_hash": "sha256:aa", "implementation_hash": "sha256:bb",
            "verdict": DIFFER, "population": {"expected": 1, "compared": 1},
            "cases": {}, "problem_codes": [],
        }},
    }))
    assert p4._load_battery_cache_entry("sha256:aa", "sha256:bb") is None


# --------------------------------------------------------------------------
# writes only after a genuine AGREE
# --------------------------------------------------------------------------


def test_write_battery_cache_entry_refuses_a_non_agree_verdict(cache_path):
    merged = _differ_receipt()
    with pytest.raises(AssertionError):
        p4._write_battery_cache_entry("sha256:aa", "sha256:bb", merged, {"x": merged})
    assert not cache_path.exists()


def test_write_then_load_is_a_hit_on_matching_hashes(cache_path):
    merged = _agree_receipt("tier0_corpus")
    cases = {"manifest": _agree_receipt("m"), "coverage": _agree_receipt("c")}
    p4._write_battery_cache_entry("sha256:aa", "sha256:bb", merged, cases)
    entry = p4._load_battery_cache_entry("sha256:aa", "sha256:bb")
    assert entry is not None
    assert entry["verdict"] == AGREE
    assert entry["cases"] == {"manifest": AGREE, "coverage": AGREE}


def test_a_mismatch_on_corpus_hash_alone_is_a_miss(cache_path):
    merged = _agree_receipt()
    p4._write_battery_cache_entry("sha256:aa", "sha256:bb", merged, {})
    assert p4._load_battery_cache_entry("sha256:zz", "sha256:bb") is None


def test_a_mismatch_on_implementation_hash_alone_is_a_miss(cache_path):
    merged = _agree_receipt()
    p4._write_battery_cache_entry("sha256:aa", "sha256:bb", merged, {})
    assert p4._load_battery_cache_entry("sha256:aa", "sha256:zz") is None


def test_the_cache_entry_carries_no_numeric_model_values(cache_path):
    """Value-free, the same standard the evidence artifact itself holds to:
    statuses, codes, counts, hashes and field names only."""
    merged = _agree_receipt()
    cases = {"manifest": _agree_receipt("m")}
    p4._write_battery_cache_entry("sha256:aa", "sha256:bb", merged, cases)
    raw = json.loads(cache_path.read_text())
    entry = raw["entries"]["sha256:aa::sha256:bb"]
    assert set(entry) == {"corpus_hash", "implementation_hash", "verdict",
                          "population", "cases", "problem_codes"}
    assert set(entry["population"]) == {"expected", "compared"}
    assert all(isinstance(v, int) for v in entry["population"].values())
    assert all(v == AGREE for v in entry["cases"].values())


def test_a_write_failure_does_not_raise(tmp_path, monkeypatch):
    # The cache directory's parent is a FILE, not a directory -- mkdir must
    # fail, and the write path must swallow that rather than raise, since
    # it must never fail a run that just proved AGREE the hard way.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    monkeypatch.setenv("PHASE4_BATTERY_CACHE_PATH", str(blocker / "cache.json"))
    merged = _agree_receipt()
    p4._write_battery_cache_entry("sha256:aa", "sha256:bb", merged, {})  # must not raise
    assert not blocker.is_dir()


# --------------------------------------------------------------------------
# the two hash inputs
# --------------------------------------------------------------------------


def test_implementation_hash_is_corpus_independent_and_stable():
    a = p4._implementation_hash()
    b = p4._implementation_hash()
    assert a == b
    assert a.startswith("sha256:")


def test_peek_corpus_hash_reads_index_without_loading(tmp_path):
    root = tmp_path / "tier0"
    root.mkdir()
    (root / "INDEX.json").write_text(json.dumps({"corpus_hash": "sha256:" + "a" * 64}))
    assert p4._peek_corpus_hash(root) == "sha256:" + "a" * 64


def test_peek_corpus_hash_fails_open_on_a_missing_index(tmp_path):
    assert p4._peek_corpus_hash(tmp_path / "missing") is None


def test_peek_corpus_hash_fails_open_on_corrupt_json(tmp_path):
    root = tmp_path / "tier0"
    root.mkdir()
    (root / "INDEX.json").write_text("{not json")
    assert p4._peek_corpus_hash(root) is None


# --------------------------------------------------------------------------
# the regression: None corpus hash must never be a cache hit
# --------------------------------------------------------------------------


def test_none_corpus_hash_is_a_miss_even_when_entry_exists(cache_path):
    """Regression test: a None corpus hash is always a MISS, never a HIT.

    This test first hand-writes an entry under the 'None::<impl>' key to
    simulate the collision that would happen if _write_battery_cache_entry
    did not guard against None. Then it verifies that even with that entry
    present, _load_battery_cache_entry still returns None (MISS).
    """
    merged = _agree_receipt()
    # Hand-write an entry to simulate the collision key that would have been
    # created by the pre-fix _write_battery_cache_entry
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text(json.dumps({
        "schema_version": p4._BATTERY_CACHE_SCHEMA,
        "entries": {"None::sha256:bb": {
            "corpus_hash": None, "implementation_hash": "sha256:bb",
            "verdict": AGREE, "population": {"expected": 1, "compared": 1},
            "cases": {}, "problem_codes": [],
        }},
    }))
    # Now verify that _load_battery_cache_entry returns None (MISS), not the entry (HIT)
    assert p4._load_battery_cache_entry(None, "sha256:bb") is None


def test_none_corpus_hash_does_not_write_an_entry(cache_path):
    """A None corpus hash must never write a cache entry."""
    merged = _agree_receipt()
    p4._write_battery_cache_entry(None, "sha256:bb", merged, {})
    # Cache file should not exist, or if it exists, should have no entries
    if cache_path.exists():
        raw = json.loads(cache_path.read_text())
        assert raw.get("entries", {}) == {}
    else:
        assert not cache_path.exists()
