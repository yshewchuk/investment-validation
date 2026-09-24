"""Phase 4 capture: the residual-only model block, end to end.

An unpriced STR-THRU/STR-RUNUP row whose chain lookup came back empty still
carries the driver residual pools legacy banded BEFORE its entry-cost/payoff
guard (``engine/score.py``:2976-2982 / :3142-3163, fixture 019). Capture freezes
those pools into a RESIDUAL-ONLY ``blocks["model"]`` -- the
:data:`engine.v2.scoring.stages.RESIDUAL_ONLY_FIELD` marker plus the declared
pools and NO payoff (:func:`_model_block_from_frozen`, and the same model layer
of :func:`frozen_source_declarations`). The strict trace serializes that block
(:func:`_model_document`), and the checker reads it back
(:func:`checks.phase4_real._decode_model_block`) and hands native a block it can
band from (:func:`engine.v2.scoring.stages._residual_only_bands`).

These tests pin that whole chain, and the blocker Astra found: the serializer
used to treat ANY truthy marker as residual-only and silently drop whatever
payoff rode beside it, laundering a malformed/contradictory block into the exact
shape the decoder then accepted. Now BOTH ends reject it, so a contradiction
cannot survive a round trip -- it fails on the way out and again on the way in.
The last tests score the round-tripped block natively (not just build it) to
prove the four bands, their legacy draw order, and the withheld P&L survive.
"""
from __future__ import annotations

import json
import math

import pytest

from checks import phase4_real
from engine.v2.foundation import content_hash, tag_nonfinite, to_document
from engine.v2.models.lineage import Lineage
from engine.v2.models.payoff_artifact import make_payoff_line_artifact
from engine.v2.models.residual_artifact import make_driver_residual_pool_artifact
from engine.v2.scoring.stages import RESIDUAL_ONLY_FIELD, NativeScoreInputs, assemble_native_values
from tools.capture_tier0_corpus import (
    StrictTraceCaptureError,
    _model_block_from_frozen,
    _model_document,
    frozen_source_declarations,
)

# Reuse the native residual-band oracle and request identity from the pinned
# native-execution test: the artifact path must band identically to the rows
# path those tests already fix against legacy's draw order (implied pool drawn
# first, runup-move pool second, from one rng, non-negative clipped, horizon
# scaled). Importing them rather than restating keeps the two files honest about
# being the SAME behaviour reached through two different bundles.
import tests.test_v2_scoring_native_residual_only_bands as bands  # noqa: E402

IMPLIED_ROLE, IMPLIED_MODEL = "implied_t1", "implied_v1"
MOVE_ROLE, MOVE_MODEL = "runup_move", "runup_v1"


def _driver_state(role: str, model_id: str, residuals, buckets=None) -> dict:
    """One recorded champion residual pool, exactly as
    ``engine.score._phase4_record_driver_pool`` stores it (flat, no fold)."""
    return {"role": role, "model_id": model_id,
            "flat_residuals": [float(r) for r in residuals],
            "buckets": None if buckets is None else dict(buckets)}


def _residual_declaration(slot: str, state_name: str, role: str, model_id: str) -> dict:
    return {f"model_residual:{slot}": {"state": state_name, "role": role,
                                       "model_id": model_id}}


def _runup_frozen(*, implied_buckets=None, move_buckets=None) -> dict:
    """A STR-RUNUP unpriced row's ``source_inputs.frozen``: implied pool under
    slot ``driver``, runup-move pool under slot ``runup_move``, no payoff."""
    return {
        "states": {
            f"driver_pool:{IMPLIED_MODEL}": _driver_state(
                IMPLIED_ROLE, IMPLIED_MODEL, bands.IMPLIED_RESIDUALS, implied_buckets),
            f"driver_pool:{MOVE_MODEL}": _driver_state(
                MOVE_ROLE, MOVE_MODEL, bands.MOVE_RESIDUALS, move_buckets),
        },
        "declarations": {
            **_residual_declaration("driver", f"driver_pool:{IMPLIED_MODEL}",
                                    IMPLIED_ROLE, IMPLIED_MODEL),
            **_residual_declaration("runup_move", f"driver_pool:{MOVE_MODEL}",
                                    MOVE_ROLE, MOVE_MODEL),
        },
    }


def _thru_frozen() -> dict:
    """A STR-THRU unpriced row's frozen declarations: a single driver pool."""
    return {
        "states": {f"driver_pool:{IMPLIED_MODEL}": _driver_state(
            IMPLIED_ROLE, IMPLIED_MODEL, bands.IMPLIED_RESIDUALS)},
        "declarations": _residual_declaration(
            "driver", f"driver_pool:{IMPLIED_MODEL}", IMPLIED_ROLE, IMPLIED_MODEL),
    }


def _priced_frozen() -> dict:
    """A priced STR-THRU row: a payoff fit plus the driver residual pool."""
    frozen = _thru_frozen()
    frozen["states"]["payoff:STR-THRU|0.5000|2026-09-01"] = {
        "kind": "line", "strategy": "STR-THRU", "alpha": 0.5,
        "cutoff": "2026-09-01", "n": 40, "resid_sd": 1.5, "r": 0.6,
        "residuals": [0.1, -0.2, 0.3], "driver": "abs_move",
        "intercept": 0.05, "slope": 0.9,
    }
    frozen["declarations"]["payoff"] = {
        "state": "payoff:STR-THRU|0.5000|2026-09-01", "before": "2026-09-16",
        "seed": 7, "draw_count": 2000,
    }
    return frozen


def _candidate(frozen: dict) -> dict:
    """A legacy_trace whose source_inputs checkpoint carries ``frozen`` (the
    shape ``_checkpoint_value`` reads inside ``frozen_source_declarations``)."""
    value = {"frozen": frozen}
    return {"legacy_trace": {"checkpoints": {
        "source_inputs": {"value": value, "content_hash": content_hash(value)},
    }}}


def _declarations(frozen: dict, tmp_path, release_states=()) -> dict:
    return frozen_source_declarations(
        _candidate(frozen), release_root=tmp_path / "release",
        deployment_id="deployment-1", release_states=list(release_states))


def _block(frozen: dict) -> dict:
    """``_model_block_from_frozen`` reads a source VALUE (whose ``"frozen"`` key
    holds these declarations), so wrap the frozen content before calling it."""
    return _model_block_from_frozen({"frozen": frozen})


def _driver_pool() -> object:
    """A real inline driver artifact (for tests that only need the value)."""
    return _block(_thru_frozen())["model_residual_artifacts"]["driver"]


# ---------------------------------------------------------------------------
# 1. both builders preserve the residual declarations and drop the payoff
# ---------------------------------------------------------------------------


def test_model_block_from_frozen_str_thru_is_residual_only_without_payoff():
    block = _block(_thru_frozen())
    assert block[RESIDUAL_ONLY_FIELD] is True
    assert "payoff_recipe" not in block and "payoff_artifact" not in block
    assert set(block["model_residual_artifacts"]) == {"driver"}
    assert set(block["model_residual_artifact_recipe"]) == {"driver"}


def test_model_block_from_frozen_str_runup_keeps_implied_and_runup_pools():
    block = _block(_runup_frozen())
    assert block[RESIDUAL_ONLY_FIELD] is True
    assert "payoff_recipe" not in block and "payoff_artifact" not in block
    assert set(block["model_residual_artifacts"]) == {"driver", "runup_move"}
    assert block["model_residual_artifacts"]["driver"].role == IMPLIED_ROLE
    assert block["model_residual_artifacts"]["runup_move"].role == MOVE_ROLE


def test_frozen_source_declarations_preserve_residuals_without_payoff(tmp_path):
    declared = _declarations(_runup_frozen(), tmp_path)
    assert set(declared["model_residual_artifacts"]) == {"driver", "runup_move"}
    assert set(declared["model_residual_artifact_recipe"]) == {"driver", "runup_move"}
    # No payoff is declared, so none is emitted -- the residual_only marker is
    # added later by ``source_inputs._model_block``, never by this builder.
    assert "payoff_artifact" not in declared
    assert "payoff_artifact_recipe" not in declared
    assert RESIDUAL_ONLY_FIELD not in declared


def test_empty_no_residual_no_payoff_stays_empty(tmp_path):
    empty = {"states": {}, "declarations": {}}
    assert _block(empty) == {}
    assert _model_document(_block(empty)) == {}
    declared = _declarations(empty, tmp_path)
    assert "model_residual_artifacts" not in declared
    assert "payoff_artifact" not in declared


def test_priced_path_is_unchanged_by_the_residual_only_marker(tmp_path):
    block = _block(_priced_frozen())
    assert RESIDUAL_ONLY_FIELD not in block
    assert block["payoff_recipe"] == {"before": "2026-09-16", "seed": 7,
                                      "draw_count": 2000}
    assert block["payoff_artifact"].intercept == 0.05
    assert set(block["model_residual_artifacts"]) == {"driver"}

    doc = _model_document(block)
    assert RESIDUAL_ONLY_FIELD not in doc
    assert doc["payoff_recipe"] == block["payoff_recipe"]
    assert doc["payoff_artifact"]["kind"] == "line"

    declared = _declarations(_priced_frozen(), tmp_path)
    assert declared["payoff_artifact"].slope == 0.9
    assert declared["payoff_artifact_recipe"]["seed"] == 7
    assert set(declared["model_residual_artifacts"]) == {"driver"}


# ---------------------------------------------------------------------------
# 2. declared-state integrity and release reuse
# ---------------------------------------------------------------------------


def test_missing_declared_state_is_refused(tmp_path):
    frozen = {
        "states": {},  # the pool the declaration points at was never recorded
        "declarations": _residual_declaration("driver", "driver_pool:ghost",
                                              IMPLIED_ROLE, IMPLIED_MODEL),
    }
    with pytest.raises(StrictTraceCaptureError, match="was not recorded"):
        _block(frozen)
    # The other builder, through the release/deployment path, refuses the very
    # same dangling declaration: both model layers read via the same state()
    # lookup and neither may silently drop a declared-but-missing pool.
    with pytest.raises(StrictTraceCaptureError, match="was not recorded"):
        _declarations(frozen, tmp_path)


def test_matching_release_artifact_is_reused_instead_of_the_inline_copy(tmp_path):
    frozen = _thru_frozen()
    inline = _block(frozen)["model_residual_artifacts"]["driver"]
    release = make_driver_residual_pool_artifact(
        role=inline.role, model_id=inline.model_id, fold=None,
        flat_residuals=list(inline.flat_residuals), buckets=None,
        deciles=inline.deciles, min_pool=inline.min_pool, lineage=Lineage())
    declared = _declarations(frozen, tmp_path, release_states=[release])
    # Same key AND same content -> the release object is frozen in, not the
    # inline reconstruction (the release is the canonical copy).
    assert declared["model_residual_artifacts"]["driver"] is release
    assert (declared["model_residual_artifact_recipe"]["driver"]["content_hash"]
            == release.content_hash)


def test_release_artifact_with_same_key_altered_pool_is_refused(tmp_path):
    frozen = _thru_frozen()
    inline = _block(frozen)["model_residual_artifacts"]["driver"]
    altered = list(inline.flat_residuals)
    altered[0] = altered[0] + 100.0  # same key, different pool content
    release = make_driver_residual_pool_artifact(
        role=inline.role, model_id=inline.model_id, fold=None,
        flat_residuals=altered, buckets=None, deciles=inline.deciles,
        min_pool=inline.min_pool, lineage=Lineage())
    with pytest.raises(StrictTraceCaptureError, match="not what legacy used"):
        _declarations(frozen, tmp_path, release_states=[release])


# ---------------------------------------------------------------------------
# 3. serializer -> tag_nonfinite -> JSON -> decoder round trip
# ---------------------------------------------------------------------------


def _roundtrip(block: dict) -> dict:
    tagged = tag_nonfinite(_model_document(block))
    return phase4_real._decode_model_block(json.loads(json.dumps(tagged, allow_nan=False)))


def test_residual_only_roundtrip_retains_marker_identities_and_infinite_edges():
    # A bucketed implied pool whose outermost decile edges are +/-inf (a real
    # champion pool, like the one the existing JSON round-trip test pins).
    buckets = {"edges": [-math.inf, -1.0, 0.0, 1.0, math.inf],
               "pools": [[-2.0, -1.5], [-0.5, -0.1], [0.2, 0.4], [1.2, 2.0]],
               "min_pool": 2}
    block = _block(_runup_frozen(implied_buckets=buckets))
    original = block["model_residual_artifacts"]["driver"]

    doc = _model_document(block)
    assert doc[RESIDUAL_ONLY_FIELD] is True
    assert "payoff_recipe" not in doc and "payoff_artifact" not in doc

    decoded = _roundtrip(block)
    assert decoded[RESIDUAL_ONLY_FIELD] is True
    assert "payoff_recipe" not in decoded and "payoff_artifact" not in decoded

    kept = decoded["model_residual_artifacts"]["driver"]
    assert (kept.role, kept.model_id, kept.fold) == \
        (original.role, original.model_id, original.fold)
    assert kept.content_hash == original.content_hash
    assert kept.bucket_edges[0] == -math.inf
    assert kept.bucket_edges[-1] == math.inf
    assert (decoded["model_residual_artifact_recipe"]["driver"]["content_hash"]
            == original.content_hash)


@pytest.mark.parametrize("marker", [1, "yes", 0, False, None])
def test_malformed_marker_is_rejected_by_serializer_and_decoder(marker):
    pool = _driver_pool()
    block = {RESIDUAL_ONLY_FIELD: marker,
             "model_residual_artifacts": {"driver": pool}}
    with pytest.raises(StrictTraceCaptureError, match="literal True"):
        _model_document(block)
    # The decoder is handed only the marker; the malformed marker must be what
    # it refuses, before anything else in the block is inspected.
    with pytest.raises(phase4_real._TraceError, match="literal true"):
        phase4_real._decode_model_block({RESIDUAL_ONLY_FIELD: marker})


def test_decoder_refuses_marker_without_any_residual_declaration():
    # A well-formed ``True`` marker, but the block declares no residual pools
    # at all: nothing for native to band from. The decoder must refuse it for
    # the missing declarations, not silently restore an empty residual-only
    # block the scoring stage would then have to reject downstream.
    with pytest.raises(phase4_real._TraceError, match="no residual declarations"):
        phase4_real._decode_model_block({RESIDUAL_ONLY_FIELD: True})


def test_payoff_recipe_contradiction_is_rejected_by_both_ends():
    pool = _driver_pool()
    recipe = {"role": pool.role, "model_id": pool.model_id, "fold": pool.fold,
              "content_hash": pool.content_hash}
    block = {RESIDUAL_ONLY_FIELD: True, "payoff_recipe": {"seed": 7},
             "model_residual_artifact_recipe": {"driver": recipe},
             "model_residual_artifacts": {"driver": pool}}
    with pytest.raises(StrictTraceCaptureError, match="payoff recipe"):
        _model_document(block)
    with pytest.raises(phase4_real._TraceError, match="payoff recipe"):
        phase4_real._decode_model_block(
            {RESIDUAL_ONLY_FIELD: True, "payoff_recipe": {"seed": 7},
             "model_residual_artifact_recipe": {"driver": recipe}})


def test_payoff_artifact_contradiction_is_rejected_by_both_ends():
    payoff = make_payoff_line_artifact(
        {"n": 40, "resid_sd": 1.5, "r": 0.6, "residuals": [0.1, -0.2, 0.3],
         "intercept": 0.05, "slope": 0.9},
        strategy="STR-THRU", driver="abs_move", alpha=0.5, cutoff="2026-09-01")
    tagged_artifact = _model_document({"payoff_artifact": payoff})["payoff_artifact"]

    with pytest.raises(StrictTraceCaptureError, match="payoff artifact"):
        _model_document({RESIDUAL_ONLY_FIELD: True, "payoff_artifact": payoff})
    with pytest.raises(phase4_real._TraceError, match="payoff artifact"):
        phase4_real._decode_model_block(
            {RESIDUAL_ONLY_FIELD: True, "payoff_artifact": tagged_artifact})


def test_priced_block_without_marker_roundtrips_and_is_accepted():
    # The positive control for the two rejection tests: identical payload to the
    # contradiction cases but WITHOUT the marker -> the block is priced, so it
    # round-trips clean (a marker plus payoff is the only contradiction).
    block = _block(_priced_frozen())
    decoded = _roundtrip(block)
    assert RESIDUAL_ONLY_FIELD not in decoded
    assert decoded["payoff_artifact"].intercept == 0.05


# ---------------------------------------------------------------------------
# 4. the round-tripped block actually bands natively (not just builder parity)
# ---------------------------------------------------------------------------


def _residual_inputs(model_block, *, snapshot="snap-1") -> NativeScoreInputs:
    context = dict(bands._identity_context())
    context["snapshot"] = snapshot
    context.pop("expiry", None)
    context.pop("post_event_expiry", None)
    context["quotes"] = {}  # observed-empty chain: NO_CHAIN, but bands survive
    return NativeScoreInputs(
        context=context, features={"model_inputs": {}},
        forecast={"driver_name": "implied_t1", "models": bands._runup_forecast_models()},
        geometry=None, pricing=None, analogs={"recipe": None},
        simulation={"mode": "not_applicable"}, gate={"mode": "not_applicable"},
        chooser={}, diagnostics={}, source_ref="capture-residual-only-roundtrip",
        stage_receipts=bands._RECEIPTS, model=model_block,
    )


def _band_fields(values):
    return {k: values.get(k) for k in (
        "driver_p10", "driver_p90", "runup_move_p10", "runup_move_p90")}


def test_roundtripped_residual_only_block_bands_all_four_without_payoff():
    roundtripped = _roundtrip(_block(_runup_frozen()))
    # The frozen-artifact path (not the compatibility rows path): _driver_pool
    # must reach driver_pool_from_artifact, so prove the rows fields are absent.
    assert "model_residual_rows" not in roundtripped
    assert "model_residual_artifacts" in roundtripped

    values = assemble_native_values(_residual_inputs(roundtripped), strategy="STR-RUNUP")
    seed = bands._derived_seed(bands._identity_context())
    bands_present = _band_fields(values)
    assert all(v is not None for v in bands_present.values())
    assert bands_present == pytest.approx(bands._runup_oracle(seed), rel=1e-12)
    # No payoff/P&L field is produced: the row is refused for the chain only.
    assert "NO_CHAIN" in values["flags"]
    for field in ("exp_pnl_model", "win_model", "win_model_raw", "payoff",
                  "model_p10", "model_p90"):
        assert values.get(field) is None, field


def test_roundtripped_bands_are_seed_deterministic_and_draw_ordered():
    roundtripped = _roundtrip(_block(_runup_frozen()))
    first = assemble_native_values(_residual_inputs(roundtripped), strategy="STR-RUNUP")
    again = assemble_native_values(_residual_inputs(roundtripped), strategy="STR-RUNUP")
    # Deterministic: the same request seed reproduces the same four bands.
    assert _band_fields(first) == _band_fields(again)

    # A different snapshot is a different request seed: every band moves, and
    # each agrees with the implied-then-runup oracle for that seed -- a
    # swapped draw order would not match this oracle on either seed.
    other = assemble_native_values(
        _residual_inputs(roundtripped, snapshot="snap-2"), strategy="STR-RUNUP")
    assert _band_fields(other) != _band_fields(first)
    assert _band_fields(other) == pytest.approx(
        bands._runup_oracle(bands._derived_seed(
            bands._identity_context(snapshot="snap-2"))), rel=1e-12)
