"""P2-C02 (Phase 2 review closeout): pure Tier-4 serving-cache coverage logic
-- ``engine.v2.data.tier4_coverage``. Synthetic joblib caches only (small
dicts with the real key names and a tiny estimator); no market data, no
network, no real registry or panel.
"""
from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.data.features import tier4 as legacy_tier4  # noqa: E402
from engine.v2.data import legacy_adapter  # noqa: E402
from engine.v2.data import tier4_coverage as tc  # noqa: E402
from engine.v2.foundation import ArtifactStore  # noqa: E402

PANEL_SHA = "a" * 64
PANEL12 = PANEL_SHA[:12]
MODELS = ({"model_id": "size", "features": ("a", "b"), "produces": "pred_abs_move"},
         {"model_id": "iv_crush", "features": ("c",), "produces": "pred_iv_crush_30"})


# --------------------------------------------------------------------------
# champion_producer_models
# --------------------------------------------------------------------------


def test_champion_producer_models_keeps_only_tier4_producers():
    champions = [
        {"id": "size_v1", "champion": True, "produces": "pred_abs_move", "features": ["a", "b"]},
        {"id": "gate_v1", "champion": True, "produces": None, "features": ["x"]},
        {"id": "iv_crush_v1", "champion": True, "produces": "pred_iv_crush_30", "features": []},
    ]
    out = tc.champion_producer_models(champions)
    assert {m["model_id"] for m in out} == {"size_v1", "iv_crush_v1"}
    (size,) = [m for m in out if m["model_id"] == "size_v1"]
    assert size["features"] == ("a", "b")
    assert size["produces"] == "pred_abs_move"


# --------------------------------------------------------------------------
# fold rule parity: the v2 adapter accessor vs the legacy scorer's own call
# --------------------------------------------------------------------------


@pytest.mark.parametrize("event_date, as_of", [
    ("2026-01-15", "2026-01-10"),   # same month, event after as_of
    ("2026-02-03", "2026-01-28"),   # event's month has not started yet at as_of -> falls back
    ("2026-03-01", "2026-03-01"),   # decided on the event's own day
    ("2025-12-20", "2026-01-05"),   # as_of after the event (a historical replay row)
])
def test_fold_rule_parity_with_legacy_scorer(event_date, as_of):
    """``Scorer._size_from_forecast``/``_forecast_for_gate`` etc. all call
    ``tier4.serving_fold(result.event_date, result.as_of)`` (engine/score.py,
    grep ``_serving(``) -- the v2 accessor must return exactly that, not a
    reimplementation of it."""
    expected = legacy_tier4.serving_fold(pd.Timestamp(event_date), pd.Timestamp(as_of))
    got = legacy_adapter.legacy_serving_fold(event_date, as_of)
    assert got == expected


# --------------------------------------------------------------------------
# required_serving_triples
# --------------------------------------------------------------------------


def test_required_serving_triples_crosses_events_with_every_producer():
    population = [("2026-01-15", "2026-01-10"), ("2026-02-20", "2026-02-10")]
    required = tc.required_serving_triples(population, MODELS, PANEL_SHA)
    assert required == frozenset({
        ("size", "202601", PANEL_SHA), ("iv_crush", "202601", PANEL_SHA),
        ("size", "202602", PANEL_SHA), ("iv_crush", "202602", PANEL_SHA),
    })


def test_required_serving_triples_dedupes_repeated_event_as_of_pairs():
    """Two events sharing an (event_date, as_of) pair compute the fold once
    (an implementation property, checked by asserting the *set* is right
    rather than any internal call count) and the result is still a plain set
    -- no duplicate triples, however many events share a fold."""
    population = [("2026-01-15", "2026-01-10"), ("2026-01-16", "2026-01-10")]
    required = tc.required_serving_triples(population, MODELS[:1], PANEL_SHA)
    assert required == frozenset({("size", "202601", PANEL_SHA)})


# --------------------------------------------------------------------------
# missing_triples
# --------------------------------------------------------------------------


def _store(tmp_path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "store")


def _cache_bytes(*, model_id="size", fold_start="2026-01-01", tier3_snapshot=PANEL_SHA,
                 features=("a", "b")) -> bytes:
    buf = io.BytesIO()
    joblib.dump({"estimator": object(), "model_id": model_id, "fold_start": fold_start,
                "tier3_snapshot": tier3_snapshot, "features": list(features)}, buf)
    return buf.getvalue()


def _publish(store, data: bytes) -> str:
    return store.publish_bytes(data, schema_ref="legacy_pinned_ref.v1").content_hash


def _cache_path(model_id: str, fold: str, panel12: str = PANEL12) -> str:
    return f"data/models/tier4/{model_id}_{fold}_{panel12}.joblib"


def test_covered_triple_launches_clean(tmp_path):
    store = _store(tmp_path)
    digest = _publish(store, _cache_bytes())
    required = frozenset({("size", "202601", PANEL_SHA)})
    pinned = {_cache_path("size", "202601"): digest}
    assert tc.missing_triples(required, pinned, store, registry_models=MODELS[:1]) == []


def test_missing_reason_absent_when_nothing_is_pinned(tmp_path):
    store = _store(tmp_path)
    required = frozenset({("size", "202601", PANEL_SHA)})
    missing = tc.missing_triples(required, {}, store, registry_models=MODELS[:1])
    assert missing == [{"model_id": "size", "fold": "202601", "panel_sha12": PANEL12,
                        "reason": "absent"}]


def test_missing_reason_fold_mismatch_for_a_new_month_beyond_pinned_folds(tmp_path):
    """A pinned cache exists for the SAME model and panel but an EARLIER
    fold -- the population has moved into a new month no cache was ever
    fitted for."""
    store = _store(tmp_path)
    digest = _publish(store, _cache_bytes(fold_start="2025-12-01"))
    required = frozenset({("size", "202601", PANEL_SHA)})
    pinned = {_cache_path("size", "202512"): digest}
    missing = tc.missing_triples(required, pinned, store, registry_models=MODELS[:1])
    assert missing == [{"model_id": "size", "fold": "202601", "panel_sha12": PANEL12,
                        "reason": "fold_mismatch"}]


def test_missing_reason_panel_mismatch_when_header_disagrees_with_filename(tmp_path):
    """The filename's 12-char panel prefix matches, but the file's own
    header names a DIFFERENT full panel sha -- caught only by reading the
    header, never by the filename alone."""
    store = _store(tmp_path)
    digest = _publish(store, _cache_bytes(tier3_snapshot="b" * 64))
    required = frozenset({("size", "202601", PANEL_SHA)})
    pinned = {_cache_path("size", "202601"): digest}
    missing = tc.missing_triples(required, pinned, store, registry_models=MODELS[:1])
    assert missing == [{"model_id": "size", "fold": "202601", "panel_sha12": PANEL12,
                        "reason": "panel_mismatch"}]


def test_missing_reason_features_mismatch(tmp_path):
    store = _store(tmp_path)
    digest = _publish(store, _cache_bytes(features=("a", "c")))
    required = frozenset({("size", "202601", PANEL_SHA)})
    pinned = {_cache_path("size", "202601"): digest}
    missing = tc.missing_triples(required, pinned, store, registry_models=MODELS[:1])
    assert missing == [{"model_id": "size", "fold": "202601", "panel_sha12": PANEL12,
                        "reason": "features_mismatch"}]


def test_multiple_misses_are_all_listed(tmp_path):
    store = _store(tmp_path)
    required = frozenset({("size", "202601", PANEL_SHA), ("iv_crush", "202601", PANEL_SHA)})
    missing = tc.missing_triples(required, {}, store, registry_models=MODELS)
    assert sorted(m["model_id"] for m in missing) == ["iv_crush", "size"]
    assert all(m["reason"] == "absent" for m in missing)


def test_oversized_cache_file_is_never_unpickled(tmp_path):
    """A file bigger than the header cap is treated exactly like an absent
    one -- refused, never opened. Uses a tiny cap so the test stays fast."""
    store = _store(tmp_path)
    digest = _publish(store, _cache_bytes())
    required = frozenset({("size", "202601", PANEL_SHA)})
    pinned = {_cache_path("size", "202601"): digest}
    missing = tc.missing_triples(required, pinned, store, registry_models=MODELS[:1],
                                 max_header_bytes=4)
    assert missing == [{"model_id": "size", "fold": "202601", "panel_sha12": PANEL12,
                        "reason": "absent"}]


# --------------------------------------------------------------------------
# legacy_adapter.legacy_tier4_serving_header directly -- size cap and hash
# verification, exercised without going through missing_triples
# --------------------------------------------------------------------------


def test_serving_header_accessor_returns_none_past_the_cap(tmp_path):
    path = tmp_path / "cache.joblib"
    path.write_bytes(_cache_bytes())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert legacy_adapter.legacy_tier4_serving_header(
        path, expected_sha256_hex=digest, max_bytes=path.stat().st_size - 1) is None


def test_serving_header_accessor_returns_none_on_hash_mismatch(tmp_path):
    path = tmp_path / "cache.joblib"
    path.write_bytes(_cache_bytes())
    assert legacy_adapter.legacy_tier4_serving_header(
        path, expected_sha256_hex="0" * 64, max_bytes=1 << 20) is None


def test_serving_header_accessor_returns_none_for_a_missing_file(tmp_path):
    assert legacy_adapter.legacy_tier4_serving_header(
        tmp_path / "nope.joblib", expected_sha256_hex="0" * 64, max_bytes=1 << 20) is None


def test_serving_header_accessor_reads_plain_fields_not_the_estimator(tmp_path):
    path = tmp_path / "cache.joblib"
    path.write_bytes(_cache_bytes(model_id="size", fold_start="2026-01-01",
                                  tier3_snapshot=PANEL_SHA, features=("a", "b")))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    header = legacy_adapter.legacy_tier4_serving_header(
        path, expected_sha256_hex=digest, max_bytes=1 << 20)
    assert header["model_id"] == "size"
    assert header["tier3_snapshot"] == PANEL_SHA
    assert header["features"] == ("a", "b")
    assert "estimator" not in header
