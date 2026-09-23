"""P5-4: the frozen win-rate recalibration-map artifact.

Legacy reference: ``engine/recalibrate.py`` (``fit_recalibration``,
``RecalibrationMap.transform``) and ``engine/score.py:2336-2341``
(``Scorer._score_model`` recalibrates STR-THRU's raw win rate). This test
imports legacy directly to prove the v2 builder bit-identical; ``engine/v2``
itself never does.

Covers: builder == legacy fit (thresholds, n, base_rate, transform); frozen
"no map" below ``min_pairs``; causality; loader verification; the P5-3 job
writing the artifact; scoring applying it under a full causal-key check,
MODEL_NOT_READY when missing or mismatched, and the undeclared path
unchanged. Synthetic data only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine import recalibrate
from engine.v2.models.recalibration_artifact import (
    RECALIBRATION_MAP_ARTIFACT_V1,
    RecalibrationArtifactError,
    RecalibrationArtifactLoader,
    RecalibrationArtifactRef,
    RecalibrationMapArtifact,
    make_recalibration_map_artifact,
    serialize_recalibration_artifact,
)
from engine.v2.models.training.recalibration import (
    MIN_PAIRS,
    build_recalibration_map_artifact,
    fit_recalibration_map,
)
from engine.v2.scoring import application
from engine.v2.scoring.source_inputs import build_native_score_inputs

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_v2_scoring_native_payoff import (  # noqa: E402
    _ZERO_MODEL_RESIDUAL_ROWS,
    _bundle,
    _request,
    _rows_from_trades,
    _runup_bundle,
    _runup_request,
    _synthetic_trades,
)

CUTOFF = "2020-09-01"


def _pairs(n=900, seed=5) -> pd.DataFrame:
    """Mixed strategies/alphas, post-cutoff rows, NaNs and tied raw_win values."""
    rng = np.random.default_rng(seed)
    raw = np.round(rng.uniform(0.0, 1.0, n), 2)  # ties exercise isotonic's tie handling
    frame = pd.DataFrame({
        "strategy": np.where(rng.random(n) < 0.8, "STR-THRU", "STR-RUNUP"),
        "fill_alpha": np.where(rng.random(n) < 0.8, 0.5, 0.25),
        "event_id": [f"E{i}" for i in range(n)],
        "ticker": "AAA",
        "event_date": pd.Timestamp("2019-01-01"),
        "exit_date": pd.Timestamp("2019-01-01") + pd.to_timedelta(rng.integers(0, 800, n), "D"),
        "raw_win": raw,
        "outcome": (rng.uniform(0, 1, n) < 0.2 + 0.5 * raw).astype(float),
    })
    frame.loc[:6, "raw_win"] = np.nan
    frame.loc[7:9, "outcome"] = np.nan
    return frame


def _legacy(pairs, strategy="STR-THRU", alpha=0.5, before=CUTOFF, min_pairs=MIN_PAIRS):
    return recalibrate.fit_recalibration(strategy, alpha, before=before, pairs=pairs,
                                         min_pairs=min_pairs)


# --------------------------------------------------------------------------
# builder == legacy fit, bit for bit
# --------------------------------------------------------------------------


def test_min_pairs_matches_legacy():
    assert MIN_PAIRS == recalibrate.MIN_PAIRS


@pytest.mark.parametrize(("strategy", "alpha", "before"), [
    ("STR-THRU", 0.5, CUTOFF), ("STR-THRU", 0.5, None), ("STR-RUNUP", 0.5, "2020-12-01"),
    ("STR-THRU", 0.25, "2021-03-01"),
])
def test_native_fit_is_bit_identical_to_legacy(strategy, alpha, before):
    pairs = _pairs()
    legacy = _legacy(pairs, strategy, alpha, before, min_pairs=50)
    native = fit_recalibration_map(pairs, strategy, alpha, before=before, min_pairs=50)
    assert legacy is not None and native is not None
    assert native["n"] == legacy.n
    assert native["base_rate"] == legacy.base_rate
    np.testing.assert_array_equal(native["x_thresholds"], legacy.x_thresholds)
    np.testing.assert_array_equal(native["y_thresholds"], legacy.y_thresholds)


def test_artifact_transform_is_bit_identical_to_legacy_after_a_disk_round_trip(tmp_path):
    pairs = _pairs()
    legacy = _legacy(pairs)
    artifact = build_recalibration_map_artifact(pairs, strategy="STR-THRU", alpha=0.5,
                                                before=CUTOFF)
    path = tmp_path / "recal.json"
    path.write_bytes(serialize_recalibration_artifact(artifact))
    loaded = RecalibrationArtifactLoader(tmp_path).load(
        RecalibrationArtifactRef(path="recal.json", content_hash=artifact.content_hash))
    assert loaded == artifact
    assert (loaded.fitted, loaded.n, loaded.cutoff, loaded.alpha) == (True, legacy.n, CUTOFF, 0.5)
    for raw in np.concatenate([np.linspace(-0.2, 1.2, 141), legacy.x_thresholds]):
        expected = float(np.ravel(legacy.transform(raw))[0])
        assert loaded.transform(float(raw)) == expected


def test_below_min_pairs_freezes_legacys_no_map_answer():
    pairs = _pairs(n=150)
    assert _legacy(pairs, before="2019-03-01") is None
    artifact = build_recalibration_map_artifact(pairs, strategy="STR-THRU", alpha=0.5,
                                                before="2019-03-01")
    assert artifact.fitted is False
    assert (artifact.n, artifact.base_rate, artifact.x_thresholds) == (None, None, ())
    assert artifact.transform(0.4321) == 0.4321
    empty = build_recalibration_map_artifact(pairs.iloc[0:0], strategy="STR-THRU", alpha=0.5)
    assert empty.fitted is False and _legacy(pairs.iloc[0:0], before=None) is None


def test_a_post_cutoff_pair_cannot_enter_the_artifact():
    pairs = _pairs()
    base = build_recalibration_map_artifact(pairs, strategy="STR-THRU", alpha=0.5, before=CUTOFF)
    leaked = pd.concat([pairs, pd.DataFrame([{
        "strategy": "STR-THRU", "fill_alpha": 0.5, "event_id": "FUTURE", "ticker": "AAA",
        "event_date": pd.Timestamp(CUTOFF), "exit_date": pd.Timestamp(CUTOFF),
        "raw_win": 0.01, "outcome": 1.0}])], ignore_index=True)
    assert build_recalibration_map_artifact(
        leaked, strategy="STR-THRU", alpha=0.5, before=CUTOFF).content_hash == base.content_hash
    assert base.window_end < CUTOFF


def test_content_hash_is_field_sensitive_and_serialization_hashes_to_it():
    import hashlib

    pairs = _pairs()
    a = build_recalibration_map_artifact(pairs, strategy="STR-THRU", alpha=0.5, before=CUTOFF)
    raw = serialize_recalibration_artifact(a)
    assert "sha256:" + hashlib.sha256(raw).hexdigest() == a.content_hash
    later = build_recalibration_map_artifact(pairs, strategy="STR-THRU", alpha=0.5,
                                             before="2020-07-01")
    other_alpha = build_recalibration_map_artifact(pairs, strategy="STR-THRU", alpha=0.25,
                                                   before=CUTOFF)
    assert len({a.content_hash, later.content_hash, other_alpha.content_hash}) == 3


# --------------------------------------------------------------------------
# loader verification
# --------------------------------------------------------------------------


def _written(tmp_path):
    artifact = build_recalibration_map_artifact(_pairs(), strategy="STR-THRU", alpha=0.5,
                                                before=CUTOFF)
    (tmp_path / "recal.json").write_bytes(serialize_recalibration_artifact(artifact))
    return artifact, RecalibrationArtifactRef(path="recal.json", content_hash=artifact.content_hash)


def test_loader_caches_by_content_hash(tmp_path):
    artifact, ref = _written(tmp_path)
    loader = RecalibrationArtifactLoader(tmp_path)
    assert loader.load(ref) is loader.load(ref)
    assert loader.cache_size == 1


def test_loader_refuses_tampered_missing_escaping_and_malformed(tmp_path):
    artifact, ref = _written(tmp_path)
    document = json.loads((tmp_path / "recal.json").read_text())
    document["y_thresholds"][0] = 0.999
    (tmp_path / "recal.json").write_text(json.dumps(document))
    with pytest.raises(RecalibrationArtifactError, match="hash mismatch"):
        RecalibrationArtifactLoader(tmp_path).load(ref)
    with pytest.raises(RecalibrationArtifactError, match="missing"):
        RecalibrationArtifactLoader(tmp_path).load(
            RecalibrationArtifactRef(path="absent.json", content_hash=ref.content_hash))
    with pytest.raises(RecalibrationArtifactError, match="escapes"):
        RecalibrationArtifactLoader(tmp_path / "sub").load(
            RecalibrationArtifactRef(path="../recal.json", content_hash=ref.content_hash))
    import hashlib

    for body in (b"not json", b'{"schema_version": "recalibration_map_artifact.v0"}'):
        (tmp_path / "bad.json").write_bytes(body)
        bad = RecalibrationArtifactRef(path="bad.json",
                                       content_hash="sha256:" + hashlib.sha256(body).hexdigest())
        with pytest.raises(RecalibrationArtifactError):
            RecalibrationArtifactLoader(tmp_path).load(bad)


def test_loader_refuses_bytes_whose_own_payload_hash_disagrees(tmp_path):
    """Same content, non-canonical bytes: the file hash matches its ref but
    the artifact's recomputed identity does not -- refused."""
    import hashlib

    artifact, _ = _written(tmp_path)
    body = json.dumps(json.loads(serialize_recalibration_artifact(artifact)), indent=2).encode()
    (tmp_path / "pretty.json").write_bytes(body)
    ref = RecalibrationArtifactRef(path="pretty.json",
                                   content_hash="sha256:" + hashlib.sha256(body).hexdigest())
    with pytest.raises(RecalibrationArtifactError, match="own hash"):
        RecalibrationArtifactLoader(tmp_path).load(ref)


def test_an_unfitted_map_cannot_carry_thresholds():
    with pytest.raises(RecalibrationArtifactError):
        RecalibrationMapArtifact(
            strategy="STR-THRU", alpha=0.5, cutoff=None, min_pairs=120, fitted=False, n=3,
            base_rate=None, x_thresholds=(), y_thresholds=(), window_start=None,
            window_end=None, content_hash="sha256:x")._payload()
    assert RECALIBRATION_MAP_ARTIFACT_V1.startswith("recalibration_map_artifact.")


# --------------------------------------------------------------------------
# the P5-3 recipe fits it through the job
# --------------------------------------------------------------------------


def _recal_recipe(strategy="STR-THRU"):
    from engine.v2.models.training import current_recipes
    from engine.v2.models.training.recipes import RecipeKey

    return current_recipes()[RecipeKey("recalibration_map", strategy, "calibration")]


def test_recalibration_recipe_fits_the_frozen_artifact_through_the_job(tmp_path):
    from engine.v2.models.training import run_training_job
    from engine.v2.models.training.estimators import UnsupportedEstimator

    recipe, pairs = _recal_recipe(), _pairs()
    with pytest.raises(UnsupportedEstimator, match="fill alpha"):
        run_training_job(recipe, pairs, tmp_path / "noalpha", cutoffs=(CUTOFF,))
    result = run_training_job(recipe, pairs, tmp_path / "fit", cutoffs=(CUTOFF, "2019-02-01"),
                              alpha=0.5)
    assert [o.status for o in result.outcomes] == ["fitted", "passthrough"]
    fdir = tmp_path / "fit/folds" / f"cut-{CUTOFF}"
    expected = build_recalibration_map_artifact(pairs, strategy="STR-THRU", alpha=0.5,
                                                before=CUTOFF)
    assert (fdir / "recalibration_artifact.json").read_bytes() == \
        serialize_recalibration_artifact(expected)
    legacy = _legacy(pairs)
    membership = json.loads((fdir / "membership_receipt.json").read_text())
    assert membership["n_train"] == legacy.n == expected.n  # members == pairs the fit keeps
    early = json.loads((tmp_path / "fit/folds/cut-2019-02-01/recalibration_artifact.json")
                       .read_text())
    assert early["fitted"] is False and _legacy(pairs, before="2019-02-01") is None


# --------------------------------------------------------------------------
# scoring reads it: full causal key, MODEL_NOT_READY, undeclared unchanged
# --------------------------------------------------------------------------


def _line():
    from engine.v2.models.training.payoff import build_payoff_line_artifact

    return build_payoff_line_artifact(_rows_from_trades(_synthetic_trades(300, seed=11)),
                                      strategy="STR-THRU", driver="abs_move", alpha=0.5,
                                      before=CUTOFF)


_RESIDUALS = [{"prediction": 7.0, "residual": -0.8}, {"prediction": 7.0, "residual": 0.8}]


def _score(**overrides):
    base = dict(payoff_artifact_recipe={"before": CUTOFF, "seed": 7, "draw_count": 500},
                payoff_artifact=_line(), model_residual_rows=_RESIDUALS)
    base.update(overrides)
    return application.score_one(_request(), build_native_score_inputs(_bundle(**base)))


def _map(**kw):
    args = dict(strategy="STR-THRU", alpha=0.5, before=CUTOFF)
    args.update(kw)
    return build_recalibration_map_artifact(_pairs(), **args)


def test_scoring_applies_the_frozen_map_to_win_model_exactly_as_legacy():
    raw = _score()
    recal = _map()
    calibrated = _score(recalibration_artifact=recal)
    raw_win = raw.resolved_request["win_model"]
    assert 0.0 < raw_win < 1.0
    legacy = _legacy(_pairs())
    expected = float(np.ravel(legacy.transform(raw_win))[0])
    assert calibrated.resolved_request["win_model"] == expected != raw_win
    assert calibrated.resolved_request["exp_pnl_model"] == raw.resolved_request["exp_pnl_model"]
    assert calibrated.validation_status == "scored"


def test_win_model_raw_survives_recalibration_unchanged():
    """win_model_raw is the pre-recalibration Monte Carlo win rate
    (engine/score.py:3012's result.win_model_raw = raw_win) and must not
    move when a recalibration artifact changes win_model."""
    raw = _score()
    recal = _map()
    calibrated = _score(recalibration_artifact=recal)
    # Undeclared recalibration: win_model_raw equals win_model exactly.
    assert raw.resolved_request["win_model_raw"] == raw.resolved_request["win_model"]
    # Declared, fitted recalibration: win_model moves, win_model_raw does not.
    assert calibrated.resolved_request["win_model_raw"] == raw.resolved_request["win_model_raw"]
    assert calibrated.resolved_request["win_model"] != calibrated.resolved_request["win_model_raw"]


def test_undeclared_recalibration_leaves_the_record_unchanged():
    """Phase 4 captures declare nothing: their records must not move."""
    assert _score() == _score(recalibration_declared=False, recalibration_artifact=None)
    inline = dict(payoff_artifact_recipe={}, payoff_artifact=None,
                  payoff_recipe={"before": CUTOFF, "seed": 7, "draw_count": 500},
                  payoff_source_rows=_rows_from_trades(_synthetic_trades(300, seed=11)))
    assert "recalibration_artifact" not in build_native_score_inputs(_bundle(
        model_residual_rows=_RESIDUALS, **inline)).model


def test_scoring_with_the_compatibility_payoff_path_also_reads_the_map():
    inline = dict(payoff_artifact_recipe={}, payoff_artifact=None,
                  payoff_recipe={"before": CUTOFF, "seed": 7, "draw_count": 500},
                  payoff_source_rows=_rows_from_trades(_synthetic_trades(300, seed=11)))
    raw = _score(**inline)
    calibrated = _score(recalibration_artifact=_map(), **inline)
    assert calibrated.resolved_request["win_model"] == _map().transform(
        raw.resolved_request["win_model"])


def test_unfitted_map_ships_the_raw_win_like_legacy():
    unfitted = _map(min_pairs=10_000)
    assert unfitted.fitted is False
    assert _legacy(_pairs(), min_pairs=10_000) is None
    raw = _score()
    frozen = _score(recalibration_artifact=unfitted)
    assert frozen.resolved_request == raw.resolved_request
    assert frozen.validation_status == "scored"


MODEL_NOT_READY_CODE = "MODEL_NOT_READY"


@pytest.mark.parametrize("bad", ["missing", "alpha", "cutoff_later", "cutoff_earlier",
                                 "strategy", "wrong_type"])
def test_missing_or_mismatched_map_is_model_not_ready(bad):
    if bad == "missing":
        overrides = {"recalibration_declared": True, "recalibration_artifact": None}
    elif bad == "alpha":
        overrides = {"recalibration_artifact": _map(alpha=0.25)}
    elif bad == "cutoff_later":
        overrides = {"recalibration_artifact": _map(before="2020-10-01")}
    elif bad == "cutoff_earlier":
        overrides = {"recalibration_artifact": _map(before="2020-08-01")}
    elif bad == "strategy":
        overrides = {"recalibration_artifact": _map(strategy="STR-RUNUP")}
    else:
        with pytest.raises(ValueError, match="RecalibrationMapArtifact"):
            _score(recalibration_artifact=_line())
        return
    record = _score(**overrides)
    assert MODEL_NOT_READY_CODE in record.reason_codes
    assert record.resolved_request.get("win_model") is None
    assert record.resolved_request.get("exp_pnl_model") is None
    assert record.validation_status == "refused"


def test_declared_map_without_a_payoff_recipe_is_rejected_at_build_time():
    with pytest.raises(ValueError, match="without a payoff recipe"):
        build_native_score_inputs(_bundle(recalibration_declared=True))


def test_str_runup_never_recalibrates():
    """Legacy keeps STR-RUNUP's win raw (engine/score.py:2508-2513)."""
    recal = build_recalibration_map_artifact(_pairs(), strategy="STR-RUNUP", alpha=0.5,
                                             before=None, min_pairs=50)
    record = application.score_one(_runup_request(), build_native_score_inputs(
        _runup_bundle(recalibration_artifact=recal)))
    assert "UNSUPPORTED_RECALIBRATION" in record.reason_codes
    assert record.resolved_request.get("win_model") is None
    assert record.validation_status == "refused"


def test_make_artifact_wraps_a_fit_without_fitting():
    from engine.v2.models.no_fit import no_fit_guard

    fit = fit_recalibration_map(_pairs(), "STR-THRU", 0.5, before=CUTOFF)
    with no_fit_guard():
        artifact = make_recalibration_map_artifact(fit, strategy="STR-THRU", alpha=0.5,
                                                   cutoff=CUTOFF, min_pairs=MIN_PAIRS)
    assert artifact.fitted and artifact.n == fit["n"]


# --------------------------------------------------------------------------
# provenance fields no test above read back (mutation-pilot triage)
# --------------------------------------------------------------------------


def test_fitted_artifact_freezes_legacys_base_rate():
    pairs = _pairs()
    legacy = _legacy(pairs)
    artifact = build_recalibration_map_artifact(pairs, strategy="STR-THRU", alpha=0.5,
                                                before=CUTOFF)
    assert artifact.base_rate == legacy.base_rate


def test_window_is_stored_as_iso_days_and_an_unfitted_map_has_none():
    fit = fit_recalibration_map(_pairs(), "STR-THRU", 0.5, before=CUTOFF)
    artifact = make_recalibration_map_artifact(
        fit, strategy="STR-THRU", alpha=0.5, cutoff=CUTOFF, min_pairs=MIN_PAIRS,
        window=(pd.Timestamp("2019-01-05"), pd.Timestamp("2020-08-30")))
    assert (artifact.window_start, artifact.window_end) == ("2019-01-05", "2020-08-30")
    unfitted = build_recalibration_map_artifact(_pairs(n=150), strategy="STR-THRU",
                                                alpha=0.5, before="2019-03-01")
    assert (unfitted.window_start, unfitted.window_end) == (None, None)


def test_loader_refuses_a_document_missing_a_field(tmp_path):
    import hashlib

    artifact, _ = _written(tmp_path)
    document = json.loads(serialize_recalibration_artifact(artifact))
    del document["base_rate"]
    body = json.dumps(document).encode("utf-8")
    (tmp_path / "partial.json").write_bytes(body)
    ref = RecalibrationArtifactRef(path="partial.json",
                                   content_hash="sha256:" + hashlib.sha256(body).hexdigest())
    with pytest.raises(RecalibrationArtifactError, match="missing"):
        RecalibrationArtifactLoader(tmp_path).load(ref)
