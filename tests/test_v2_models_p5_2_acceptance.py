"""P5-2 acceptance: read-only inference, cold == warm, missing -> MODEL_NOT_READY.

``guides/rearchitecture_phase5_models.md`` P5-2: "Rig all fitting/provider/
cache-write paths to fail; cold and warm requests agree; missing artifact
returns MODEL_NOT_READY, never trains."

The per-call-site rigging tests live in ``tests/test_v2_models_no_fit.py``
(legacy guard) and ``tests/test_v2_models_no_fit_native.py`` (v2 guard).
This file covers the two acceptance points they did not:

- **cold vs warm.** A fresh interpreter (every loader cache empty) and a
  second request in the same process, through the same loaders, produce the
  identical v2 model, payoff and recalibration outputs and the identical
  canonical score record.
- **every v2 model consumer in ``engine/v2/scoring``** refuses a missing
  artifact with MODEL_NOT_READY, with BOTH no-fit guards active at once and
  no file written under the artifact root.

Synthetic ``tmp_path`` artifacts only; nothing reads ``data/``.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.models.no_fit import RuntimeFitForbidden as LegacyFitForbidden
from engine.models.no_fit import no_fit_guard as legacy_no_fit_guard
from engine.v2.foundation import to_document
from engine.v2.foundation.canonical import canonical_json
from engine.v2.models import (
    MODEL_NOT_READY,
    MODEL_READY,
    FrozenInference,
    InferenceRequest,
    ModelBinding,
    ModelRelease,
    RuntimeFitForbidden,
)
from engine.v2.models.contracts import ArtifactMember
from engine.v2.models.no_fit import no_fit_guard as v2_no_fit_guard
from engine.v2.models.payoff_artifact import (
    PayoffArtifactLoader,
    PayoffArtifactRef,
    serialize_payoff_artifact,
)
from engine.v2.models.recalibration_artifact import (
    RecalibrationArtifactLoader,
    RecalibrationArtifactRef,
    serialize_recalibration_artifact,
)
from engine.v2.scoring import application, native_payoff
from engine.v2.scoring.frozen_executor import FrozenStageExecutor, FrozenStageRefusal
from engine.v2.scoring.source_inputs import build_native_score_inputs

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_v2_scoring_native_payoff import (  # noqa: E402
    _GOOD_RUNUP_PAYOFF_ROWS,
    _ZERO_MODEL_RESIDUAL_ROWS,
    _bundle,
    _request,
    _rows_from_trades,
    _runup_bundle,
    _runup_request,
    _synthetic_trades,
)

REPO = Path(__file__).resolve().parents[1]
CUTOFF = "2020-10-01"


@contextmanager
def both_guards():
    with legacy_no_fit_guard(), v2_no_fit_guard():
        yield


def _tree(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


# --------------------------------------------------------------------------
# a synthetic release: one frozen driver estimator, one payoff line, one map
# --------------------------------------------------------------------------


def _pairs(n=400, seed=3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.0, 1.0, n)
    return pd.DataFrame({
        "strategy": "STR-THRU", "fill_alpha": 0.5, "event_id": [f"E{i}" for i in range(n)],
        "exit_date": pd.Timestamp("2019-01-01") + pd.to_timedelta(rng.integers(0, 600, n), "D"),
        "raw_win": raw, "outcome": (rng.uniform(0, 1, n) < 0.3 + 0.4 * raw).astype(float),
    })


def _write_release(root: Path) -> dict:
    """Build every artifact with the training-side builders, write it, return refs."""
    from engine.v2.models.training.payoff import build_payoff_line_artifact
    from engine.v2.models.training.recalibration import build_recalibration_map_artifact

    root.mkdir(parents=True, exist_ok=True)
    estimator = json.dumps({
        "schema_version": "linear_estimator.v1.0", "feature_order": ["x"],
        "outputs": [{"name": "driver_prediction", "intercept": 1.0, "coefficients": [2.0]}],
    }, sort_keys=True).encode()
    (root / "estimator.json").write_bytes(estimator)
    trades = _synthetic_trades(300, seed=11)
    line = build_payoff_line_artifact(_rows_from_trades(trades), strategy="STR-THRU",
                                      driver="abs_move", alpha=0.5, before=CUTOFF)
    (root / "payoff.json").write_bytes(serialize_payoff_artifact(line))
    recal = build_recalibration_map_artifact(_pairs(), strategy="STR-THRU", alpha=0.5,
                                             before=CUTOFF)
    assert recal.fitted
    (root / "recal.json").write_bytes(serialize_recalibration_artifact(recal))
    return {"estimator": "sha256:" + hashlib.sha256(estimator).hexdigest(),
            "payoff": line.content_hash, "recal": recal.content_hash}


def _release(hashes: dict) -> ModelRelease:
    member = ArtifactMember(name="estimator", path="estimator.json",
                            content_hash=hashes["estimator"])
    binding = ModelBinding(
        binding_id="driver", model_id="m-driver", role="driver", strategy_id="STR-THRU",
        decision_clock_id="entry-close", adapter="json-linear.v1", feature_order=("x",),
        output_names=("driver_prediction",), members=(member,),
    )
    return ModelRelease(release_id="r1", deployment_id="dep-1", bindings=(binding,))


_INFERENCE_REQUEST = InferenceRequest(release_id="r1", binding_id="driver",
                                      feature_order=("x",), rows=((2.5,),))


class _Session:
    """One process's loaders: warm requests reuse these caches."""

    def __init__(self, root: Path, hashes: dict) -> None:
        self.root, self.hashes = root, hashes
        self.inference = FrozenInference(root)
        self.payoffs = PayoffArtifactLoader(root)
        self.recals = RecalibrationArtifactLoader(root)
        self.release = _release(hashes)

    def outputs(self) -> dict:
        inferred = self.inference.infer(self.release, _INFERENCE_REQUEST)
        line = self.payoffs.load(PayoffArtifactRef(path="payoff.json",
                                                   content_hash=self.hashes["payoff"]))
        recal = self.recals.load(RecalibrationArtifactRef(path="recal.json",
                                                          content_hash=self.hashes["recal"]))
        bundle = _bundle(
            feature_vector={"x": 2.5}, feature_missing_mask={"x": False},
            payoff_artifact_recipe={"before": CUTOFF, "seed": 7, "draw_count": 500},
            payoff_artifact=line, recalibration_artifact=recal,
            model_residual_rows=[{"prediction": 6.0, "residual": -0.5},
                                 {"prediction": 6.0, "residual": 0.5}],
        )
        record = application.score_frozen(
            _request(), self.inference, self.release, _INFERENCE_REQUEST,
            {"_native_inputs": build_native_score_inputs(bundle)},
        )
        return json.loads(canonical_json({
            "inference": to_document(inferred),
            "payoff": [line.content_hash, line.intercept, line.slope, list(line.residuals)],
            "recal": [recal.content_hash, list(recal.x_thresholds), list(recal.y_thresholds),
                      recal.transform(0.37)],
            "record": to_document(record),
        }))


def _cold_outputs(root: Path, hashes: dict) -> dict:
    """The same request in a brand-new interpreter: nothing cached anywhere."""
    script = (
        "import json, sys; from pathlib import Path\n"
        f"sys.path[:0] = [{str(REPO)!r}, {str(REPO / 'tests')!r}]\n"
        "from test_v2_models_p5_2_acceptance import _Session, both_guards\n"
        "with both_guards():\n"
        f"    out = _Session(Path({str(root)!r}), {hashes!r}).outputs()\n"
        "print(json.dumps(out))\n"
    )
    env = {**os.environ, "PYTHONHASHSEED": "12345"}
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          timeout=240, cwd=str(REPO), env=env)
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_cold_process_and_warm_same_process_requests_agree(tmp_path):
    root = tmp_path / "release"
    hashes = _write_release(root)
    before = _tree(root)

    with both_guards():
        session = _Session(root, hashes)
        first = session.outputs()
        assert (session.inference.cache_size, session.payoffs.cache_size,
                session.recals.cache_size) == (1, 1, 1)
        warm = session.outputs()                      # every cache hit
        cleared = _Session(root, hashes).outputs()    # caches cleared, same process
    cold = _cold_outputs(root, hashes)                # fresh process

    assert first["inference"]["status"] == MODEL_READY
    assert first["record"]["validation_status"] == "scored"
    assert first["record"]["resolved_request"]["win_model"] is not None
    assert first == warm == cleared == cold
    assert _tree(root) == before                      # nothing written


def test_cold_warm_comparison_detects_a_changed_artifact(tmp_path):
    """Negative control: the comparison above can fail. A different payoff
    fold (a later cutoff) gives a different record."""
    root = tmp_path / "release"
    hashes = _write_release(root)
    baseline = _Session(root, hashes).outputs()
    from engine.v2.models.training.payoff import build_payoff_line_artifact

    other = build_payoff_line_artifact(_rows_from_trades(_synthetic_trades(300, seed=12)),
                                       strategy="STR-THRU", driver="abs_move", alpha=0.5,
                                       before=CUTOFF)
    (root / "payoff.json").write_bytes(serialize_payoff_artifact(other))
    changed = _Session(root, {**hashes, "payoff": other.content_hash}).outputs()
    assert changed["record"] != baseline["record"]


# --------------------------------------------------------------------------
# every v2 model consumer: missing artifact -> MODEL_NOT_READY, both guards on
# --------------------------------------------------------------------------


def _missing_estimator_release(tmp_path):
    root = tmp_path / "release"
    hashes = _write_release(root)
    (root / "estimator.json").unlink()
    return root, _release(hashes)


def test_frozen_inference_missing_member_is_model_not_ready_under_both_guards(tmp_path):
    root, release = _missing_estimator_release(tmp_path)
    before = _tree(root)
    with both_guards():
        result = FrozenInference(root).infer(release, _INFERENCE_REQUEST)
    assert result.status == MODEL_NOT_READY
    assert result.predictions == ()
    assert _tree(root) == before


def test_frozen_stage_executor_missing_member_is_model_not_ready_under_both_guards(tmp_path):
    root, release = _missing_estimator_release(tmp_path)
    executor = FrozenStageExecutor(inference=FrozenInference(root), release=release,
                                   binding_id="driver")
    with both_guards():
        with pytest.raises(FrozenStageRefusal) as refused:
            executor.execute({"x": 2.5})
    assert refused.value.code == MODEL_NOT_READY


def test_score_frozen_missing_member_is_model_not_ready_under_both_guards(tmp_path):
    root, release = _missing_estimator_release(tmp_path)
    before = _tree(root)
    bundle = _bundle(feature_vector={"x": 2.5}, feature_missing_mask={"x": False})
    with both_guards():
        record = application.score_frozen(
            _request(), FrozenInference(root), release, _INFERENCE_REQUEST,
            {"_native_inputs": build_native_score_inputs(bundle)},
        )
    assert MODEL_NOT_READY in record.reason_codes
    assert "ARTIFACT_INVALID" in record.reason_codes
    assert record.forecasts["driver_prediction"] is None
    assert record.validation_status == record.readiness == "refused"
    assert _tree(root) == before


@pytest.mark.parametrize("case", ["payoff_line", "payoff_surface", "recalibration"])
def test_model_stage_missing_artifact_is_model_not_ready_under_both_guards(case):
    """The payoff/recalibration consumers in stages.py: declared artifact,
    none supplied. Both guards are on, so an inline-fit fallback would raise
    RuntimeFitForbidden instead of returning a record."""
    if case == "payoff_line":
        bundle, request = _bundle(payoff_artifact_recipe={"seed": 1}, payoff_artifact=None,
                                  model_residual_rows=_ZERO_MODEL_RESIDUAL_ROWS), _request()
    elif case == "payoff_surface":
        bundle, request = _runup_bundle(payoff_recipe={}, payoff_source_rows=(),
                                        payoff_artifact_recipe={"seed": 1},
                                        payoff_artifact=None), _runup_request()
    else:
        from engine.v2.models.training.payoff import build_payoff_line_artifact

        line = build_payoff_line_artifact(
            [{"driver": 0.0, "spot_entry": 100.0, "exit_value": 2.0, "exit_date": "2026-09-01"},
             {"driver": 10.0, "spot_entry": 100.0, "exit_value": 6.0, "exit_date": "2026-09-01"}],
            strategy="STR-THRU", driver="abs_move", alpha=0.5, min_trades=2)
        bundle, request = _bundle(payoff_artifact_recipe={"seed": 1}, payoff_artifact=line,
                                  recalibration_declared=True, recalibration_artifact=None,
                                  model_residual_rows=_ZERO_MODEL_RESIDUAL_ROWS), _request()
    with both_guards():
        record = application.score_one(request, build_native_score_inputs(bundle))
    assert MODEL_NOT_READY in record.reason_codes
    assert record.resolved_request.get("exp_pnl_model") is None
    assert record.resolved_request.get("win_model") is None
    assert record.validation_status == "refused"


def test_both_guards_really_rig_the_fit_and_write_paths(tmp_path):
    """Positive control for the refusals above: under the same combined
    guard every fitting, provider-side model write and cache write raises."""
    from engine import recalibrate
    from engine.models.registry import Registry
    from engine.v2.models.training.recalibration import fit_recalibration_map

    with both_guards():
        with pytest.raises(RuntimeFitForbidden):
            native_payoff.fit_payoff_line(_GOOD_RUNUP_PAYOFF_ROWS, min_trades=2)
        with pytest.raises(RuntimeFitForbidden):
            native_payoff.fit_runup_payoff_surface(_GOOD_RUNUP_PAYOFF_ROWS, min_trades=2)
        with pytest.raises((RuntimeFitForbidden, LegacyFitForbidden)):
            fit_recalibration_map(_pairs(), "STR-THRU", 0.5, before=None)
        with pytest.raises(LegacyFitForbidden):
            Registry(entries=[], path=tmp_path / "registry.json").save()
        with pytest.raises(LegacyFitForbidden):
            recalibrate.build_pairs(scorer=object(), path=tmp_path / "pairs.parquet")
    assert not any(tmp_path.iterdir())


def test_each_guard_alone_rigs_the_recalibration_builder():
    from engine.v2.models.training.recalibration import fit_recalibration_map

    with legacy_no_fit_guard():
        with pytest.raises(LegacyFitForbidden):
            fit_recalibration_map(_pairs(), "STR-THRU", 0.5, before=None)
    with v2_no_fit_guard():
        with pytest.raises(RuntimeFitForbidden):
            fit_recalibration_map(_pairs(), "STR-THRU", 0.5, before=None)
