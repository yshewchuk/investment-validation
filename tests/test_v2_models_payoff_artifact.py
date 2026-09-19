"""``engine.v2.models.payoff_artifact`` -- the frozen P5-4 artifact type.

guides/rearchitecture_phase5_models.md P5-4: "a versioned, content-hashed
payoff-calibration artifact type... Load it through a verified read-only
loader (hash check, as FrozenInference/loader.py do)." These tests cover the
type and loader in isolation (no scoring, no training) -- parity against the
inline fit and the end-to-end score-path wiring live in
``tests/test_v2_scoring_native_payoff.py``; training's fit-and-freeze parity
lives in ``tests/test_v2_models_training_payoff.py``.
"""
from __future__ import annotations

import math

import pytest

from engine.v2.models.payoff_artifact import (
    PayoffArtifactError,
    PayoffArtifactLoader,
    PayoffArtifactRef,
    PayoffLineArtifact,
    PayoffSurfaceArtifact,
    make_payoff_line_artifact,
    make_payoff_surface_artifact,
    payoff_artifact_key,
    serialize_payoff_artifact,
)

_LINE_FIT = {
    "intercept": 0.05, "slope": 0.004, "resid_sd": 0.01, "n": 250,
    "r": 0.42, "residuals": [0.01, -0.02, 0.03, -0.01],
}
_SURFACE_FIT = {
    "coefficients": [0.02, 0.004, 0.001, 0.0005, 0.0007, 0.0003],
    "resid_sd": 0.008, "n": 300, "r": 0.31,
    "residuals": [0.001, -0.002, 0.0015],
}


# ---------------------------------------------------------------------------
# key identity
# ---------------------------------------------------------------------------


def test_payoff_artifact_key_rounds_alpha_and_normalizes_cutoff():
    assert payoff_artifact_key("STR-THRU", 0.50001, "2026-09-16") == (
        "STR-THRU", 0.5, "2026-09-16",
    )
    assert payoff_artifact_key("STR-THRU", 0.5, None) == ("STR-THRU", 0.5, None)
    # A full timestamp normalizes to its date, same as native_payoff's own
    # cutoff parsing.
    assert payoff_artifact_key("STR-THRU", 0.5, "2026-09-16T00:00:00")[2] == "2026-09-16"


def test_artifact_key_property_matches_the_free_function():
    artifact = make_payoff_line_artifact(
        _LINE_FIT, strategy="STR-THRU", driver="abs_move", alpha=0.5,
        cutoff="2026-09-16",
    )
    assert artifact.key == payoff_artifact_key("STR-THRU", 0.5, "2026-09-16")


# ---------------------------------------------------------------------------
# construction: provenance and hashing
# ---------------------------------------------------------------------------


def test_make_payoff_line_artifact_carries_provenance():
    artifact = make_payoff_line_artifact(
        _LINE_FIT, strategy="STR-THRU", driver="abs_move", alpha=0.5,
        cutoff="2026-09-16", window=("2020-01-01", "2026-09-01"),
    )
    assert artifact.strategy == "STR-THRU"
    assert artifact.driver == "abs_move"
    assert artifact.alpha == 0.5
    assert artifact.cutoff == "2026-09-16"
    assert artifact.window_start == "2020-01-01"
    assert artifact.window_end == "2026-09-01"
    assert artifact.n == 250
    assert artifact.intercept == pytest.approx(0.05)
    assert artifact.slope == pytest.approx(0.004)
    assert artifact.residuals == (0.01, -0.02, 0.03, -0.01)
    assert artifact.content_hash.startswith("sha256:")
    assert len(artifact.content_hash) == len("sha256:") + 64


def test_make_payoff_surface_artifact_carries_provenance():
    artifact = make_payoff_surface_artifact(
        _SURFACE_FIT, alpha=0.5, cutoff="2026-09-16",
        window=("2020-01-01", "2026-09-01"),
    )
    assert artifact.strategy == "STR-RUNUP"
    assert artifact.coefficients == tuple(_SURFACE_FIT["coefficients"])
    assert artifact.n == 300
    assert artifact.content_hash.startswith("sha256:")


def test_content_hash_is_deterministic_and_field_sensitive():
    a = make_payoff_line_artifact(
        _LINE_FIT, strategy="STR-THRU", driver="abs_move", alpha=0.5,
        cutoff="2026-09-16",
    )
    b = make_payoff_line_artifact(
        _LINE_FIT, strategy="STR-THRU", driver="abs_move", alpha=0.5,
        cutoff="2026-09-16",
    )
    assert a.content_hash == b.content_hash

    perturbed = make_payoff_line_artifact(
        {**_LINE_FIT, "slope": 0.005}, strategy="STR-THRU", driver="abs_move",
        alpha=0.5, cutoff="2026-09-16",
    )
    assert perturbed.content_hash != a.content_hash


def test_content_hash_differs_by_strategy_alpha_and_cutoff():
    base = make_payoff_line_artifact(
        _LINE_FIT, strategy="STR-THRU", driver="abs_move", alpha=0.5,
        cutoff="2026-09-16",
    )
    other_strategy = make_payoff_line_artifact(
        _LINE_FIT, strategy="CTR5", driver="abs_move", alpha=0.5,
        cutoff="2026-09-16",
    )
    other_alpha = make_payoff_line_artifact(
        _LINE_FIT, strategy="STR-THRU", driver="abs_move", alpha=0.75,
        cutoff="2026-09-16",
    )
    other_cutoff = make_payoff_line_artifact(
        _LINE_FIT, strategy="STR-THRU", driver="abs_move", alpha=0.5,
        cutoff="2026-09-17",
    )
    hashes = {
        base.content_hash, other_strategy.content_hash,
        other_alpha.content_hash, other_cutoff.content_hash,
    }
    assert len(hashes) == 4


def test_degenerate_resid_sd_round_trips_as_nan_not_a_tagged_object():
    """canonical_json tags NaN as {"__nonfinite__": ...} internally; the
    artifact's own field must stay a real float, never that tag leaking
    through the public dataclass."""
    fit = {**_LINE_FIT, "resid_sd": float("nan")}
    artifact = make_payoff_line_artifact(
        fit, strategy="STR-THRU", driver="abs_move", alpha=0.5,
    )
    assert isinstance(artifact.resid_sd, float)
    assert math.isnan(artifact.resid_sd)


# ---------------------------------------------------------------------------
# the verified read-only loader
# ---------------------------------------------------------------------------


def _write(tmp_path, name, artifact):
    path = tmp_path / name
    path.write_bytes(serialize_payoff_artifact(artifact))
    return PayoffArtifactRef(path=name, content_hash=artifact.content_hash)


def test_loader_round_trips_a_line_artifact(tmp_path):
    artifact = make_payoff_line_artifact(
        _LINE_FIT, strategy="STR-THRU", driver="abs_move", alpha=0.5,
        cutoff="2026-09-16", window=("2020-01-01", "2026-09-01"),
    )
    ref = _write(tmp_path, "line.json", artifact)
    loaded = PayoffArtifactLoader(tmp_path).load(ref)
    assert isinstance(loaded, PayoffLineArtifact)
    assert loaded == artifact


def test_loader_round_trips_a_surface_artifact(tmp_path):
    artifact = make_payoff_surface_artifact(_SURFACE_FIT, alpha=0.5)
    ref = _write(tmp_path, "surface.json", artifact)
    loaded = PayoffArtifactLoader(tmp_path).load(ref)
    assert isinstance(loaded, PayoffSurfaceArtifact)
    assert loaded == artifact


def test_loader_caches_by_content_hash(tmp_path):
    artifact = make_payoff_line_artifact(
        _LINE_FIT, strategy="STR-THRU", driver="abs_move", alpha=0.5,
    )
    ref = _write(tmp_path, "line.json", artifact)
    loader = PayoffArtifactLoader(tmp_path)
    first = loader.load(ref)
    assert loader.cache_size == 1
    second = loader.load(ref)
    assert second is first
    assert loader.cache_size == 1


def test_loader_refuses_a_tampered_file(tmp_path):
    artifact = make_payoff_line_artifact(
        _LINE_FIT, strategy="STR-THRU", driver="abs_move", alpha=0.5,
    )
    ref = _write(tmp_path, "line.json", artifact)
    (tmp_path / "line.json").write_bytes(
        serialize_payoff_artifact(artifact) + b" "
    )
    with pytest.raises(PayoffArtifactError):
        PayoffArtifactLoader(tmp_path).load(ref)


def test_loader_refuses_a_missing_file(tmp_path):
    ref = PayoffArtifactRef(path="absent.json", content_hash="sha256:" + "0" * 64)
    with pytest.raises(PayoffArtifactError):
        PayoffArtifactLoader(tmp_path).load(ref)


def test_loader_refuses_a_path_escaping_the_root(tmp_path):
    outside = tmp_path.parent / "outside.json"
    outside.write_bytes(b"{}")
    try:
        ref = PayoffArtifactRef(path="../outside.json", content_hash="sha256:" + "0" * 64)
        with pytest.raises(PayoffArtifactError):
            PayoffArtifactLoader(tmp_path).load(ref)
    finally:
        outside.unlink(missing_ok=True)


def test_loader_refuses_malformed_json(tmp_path):
    (tmp_path / "bad.json").write_bytes(b"not json")
    import hashlib
    digest = "sha256:" + hashlib.sha256(b"not json").hexdigest()
    ref = PayoffArtifactRef(path="bad.json", content_hash=digest)
    with pytest.raises(PayoffArtifactError):
        PayoffArtifactLoader(tmp_path).load(ref)


def test_loader_refuses_an_unknown_schema_version(tmp_path):
    import hashlib
    import json

    document = {"schema_version": "some_future_schema.v9.0"}
    raw = json.dumps(document).encode("utf-8")
    (tmp_path / "future.json").write_bytes(raw)
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    ref = PayoffArtifactRef(path="future.json", content_hash=digest)
    with pytest.raises(PayoffArtifactError):
        PayoffArtifactLoader(tmp_path).load(ref)


# ---------------------------------------------------------------------------
# provenance fields the tests above never read back (mutation-pilot triage)
# ---------------------------------------------------------------------------


def test_surface_artifact_records_cutoff_window_r_and_key():
    """A surface artifact's cutoff is its fold identity (the key); losing it
    would let a wrong-fold surface match any request."""
    artifact = make_payoff_surface_artifact(
        _SURFACE_FIT, alpha=0.5, cutoff="2026-09-16",
        window=("2020-01-01", "2026-09-01"),
    )
    assert artifact.cutoff == "2026-09-16"
    assert (artifact.window_start, artifact.window_end) == ("2020-01-01", "2026-09-01")
    assert artifact.r == pytest.approx(0.31)
    assert artifact.key == ("STR-RUNUP", 0.5, "2026-09-16")
    undated = make_payoff_surface_artifact(_SURFACE_FIT, alpha=0.5)
    assert undated.content_hash != artifact.content_hash


def test_loader_round_trips_a_dated_surface_artifact(tmp_path):
    artifact = make_payoff_surface_artifact(
        _SURFACE_FIT, alpha=0.5, cutoff="2026-09-16",
        window=("2020-01-01", "2026-09-01"),
    )
    ref = _write(tmp_path, "surface_dated.json", artifact)
    loaded = PayoffArtifactLoader(tmp_path).load(ref)
    assert loaded == artifact
    assert loaded.cutoff == "2026-09-16"
    assert (loaded.window_start, loaded.window_end) == ("2020-01-01", "2026-09-01")


def test_line_artifact_records_r_and_accepts_a_fit_without_r(tmp_path):
    artifact = make_payoff_line_artifact(
        _LINE_FIT, strategy="STR-THRU", driver="abs_move", alpha=0.5,
    )
    assert artifact.r == pytest.approx(0.42)
    no_r = make_payoff_line_artifact(
        {k: v for k, v in _LINE_FIT.items() if k != "r"},
        strategy="STR-THRU", driver="abs_move", alpha=0.5,
    )
    assert no_r.r is None
    assert no_r.content_hash != artifact.content_hash
    ref = _write(tmp_path, "line_no_r.json", no_r)
    assert PayoffArtifactLoader(tmp_path).load(ref) == no_r


def test_artifact_alpha_is_rounded_like_its_key():
    """The payload's alpha is rounded to 4 places, the same as
    ``payoff_artifact_key``, so the stored alpha and the key agree."""
    artifact = make_payoff_line_artifact(
        _LINE_FIT, strategy="STR-THRU", driver="abs_move", alpha=0.123456,
    )
    assert artifact.alpha == 0.1235
    assert artifact.key[1] == artifact.alpha
