"""Verified RUNUP model state: semantic slots, causal surface and typed gaps."""
from dataclasses import replace

import pytest

from engine.v2.models.no_fit import no_fit_guard
from engine.v2.models.payoff_artifact import make_payoff_line_artifact
from engine.v2.ops import native_score_batch as batch
from engine.v2.scoring.native_residuals import driver_pool_from_artifact
from engine.v2.scoring.release_bindings import ModelNotReady, resolve_release_binding
from tests.v2.ops.runup_helpers import pool, state_row, surface
from tests.v2.ops.test_native_driver_declarations import (
    EXACT,
    IMPLIED,
    MOVE,
    POLICY,
    RUNUP_GATE,
    _event,
)
from tests.v2.ops.test_native_score_batch_driver_alias import (
    GATE,
    SIZE,
    _assemble,
    _event_inputs,
    _stage,
)


def _binding(root, *, pools=None, surfaces=None, extra=(), reverse=False):
    pools = [pool("implied_t1", residuals=(1.0,)), pool("runup_move", residuals=(-0.5,))] if pools is None else pools
    surfaces = [surface()] if surfaces is None else surfaces
    rows = [state_row(root, f"driver_residual_pool:{item.role}", [item]) for item in pools]
    if surfaces:
        rows.append(state_row(root, "payoff_surface:STR-RUNUP", surfaces))
    if reverse:
        rows.reverse()
    with no_fit_guard():
        return _stage(root, [IMPLIED, MOVE, RUNUP_GATE, *extra], rows=rows)


def _build(binding, event=None, **kwargs):
    event = event or _event()
    rows, refusals = _assemble(binding, [event], gate_policy=POLICY, **kwargs)
    return rows, refusals, event


def _labelled_refusal_detail(cause):
    label = batch._RELEASE_ARTIFACT_REFUSAL_LABELS[("MODEL_NOT_READY", cause)]
    return f"nightly_source_bundle refused: MODEL_NOT_READY: {label}"


def _assert_model_refusal(binding, detail="nightly_source_bundle refused: MODEL_NOT_READY: RUNUP champion pool missing or unusable"):
    rows, refusals, event = _build(binding)
    assert rows == {}
    assert [(item.key, item.code, item.detail) for item in refusals] == [(
        event.key, "MODEL_NOT_READY", detail)]


@pytest.mark.parametrize("reverse", [False, True])
def test_both_verified_pools_surface_and_simulation_contract_survive(tmp_path, reverse):
    binding = _binding(tmp_path, reverse=reverse)
    rows, refusals, event = _build(binding)
    assert refusals == ()
    inputs = rows[event.key][1]
    expected_pools = {"driver": binding.driver_residual_artifacts["implied_t1"],
                      "runup_move": binding.driver_residual_artifacts["runup_move"]}
    assert inputs.model["model_residual_artifacts"] == expected_pools
    assert inputs.model["model_residual_artifact_recipe"] == {
        slot: {"role": item.role, "model_id": item.model_id, "fold": None,
               "content_hash": item.content_hash} for slot, item in expected_pools.items()}
    assert inputs.model["payoff_artifact"] == surface()
    assert inputs.model["payoff_recipe"] == {"before": "2026-01-09"}
    assert inputs.simulation == {"mode": "not_applicable"}
    assert "recalibration_artifact" not in inputs.model
    assert inputs.analogs == {"recipe": None}
    for slot, expected in (("driver", [1.0]), ("runup_move", [-0.5])):
        values, flag = driver_pool_from_artifact(inputs.model, slot, 11.0)
        assert flag is None
        assert values.tolist() == expected


@pytest.mark.parametrize("missing", ["implied_t1", "runup_move", "surface"])
def test_missing_required_state_refuses_only_runup_without_inventing_inputs(tmp_path, missing):
    pools = [pool(role) for role in ("implied_t1", "runup_move") if role != missing]
    binding = _binding(tmp_path, pools=pools, surfaces=[] if missing == "surface" else None,
                       extra=(SIZE, GATE))
    runup, thru = _event(), _event_inputs()
    rows, refused = _assemble(binding, [runup, thru], gate_policy=POLICY)
    assert list(rows) == [thru.key]
    assert [(item.key, item.code) for item in refused] == [(runup.key, "MODEL_NOT_READY")]
    assert rows[thru.key][1].simulation == {}


@pytest.mark.parametrize("role", ["implied_t1", "runup_move"])
@pytest.mark.parametrize("change", [
    {"model_id": "wrong-model"}, {"fold": "2026-01-01"}, {"fold": "2026-02-01"},
    {"residuals": ()}, {"residuals": (float("nan"),)}, {"residuals": (float("inf"),)},
    pytest.param({"residuals": (1.0, float("nan"))}, id="mixed-flat-invalid-after-finite"),
    {"buckets": {"edges": (0.0,), "pools": ()}},
    {"buckets": {"edges": (1.0, 0.0), "pools": ((1.0,),)}},
    pytest.param({"buckets": {"edges": (0.0, 2.0, 1.0), "pools": ((1.0,), (2.0,))}},
                 id="mixed-order-invalid-after-valid"),
    {"buckets": {"edges": (0.0, float("nan")), "pools": ((1.0,),)}},
    {"buckets": {"edges": (-float("inf"), float("inf")), "pools": ((float("nan"),),)}},
    pytest.param({"buckets": {"edges": (-float("inf"), float("inf")),
                             "pools": ((1.0, float("nan")),)}}, id="mixed-bucket-invalid-after-finite"),
    {"buckets": {"edges": (-float("inf"), float("inf")), "pools": ((),)}, "min_pool": 0},
    pytest.param({"buckets": {"edges": (-float("inf"), 0.0, float("inf")),
                             "pools": ((1.0,), ())}, "min_pool": 0},
                 id="mixed-zero-floor-empty-after-nonempty"),
])
def test_hash_valid_but_unusable_champion_pool_is_refused(tmp_path, role, change):
    pools = [pool(name, **(change if name == role else {})) for name in ("implied_t1", "runup_move")]
    binding = _binding(tmp_path, pools=pools)
    _assert_model_refusal(binding)


@pytest.mark.parametrize("point,expected", [(-1.0, [9.0]), (1.0, [2.0, 3.0])])
def test_valid_bucket_selection_and_thin_bucket_flat_fallback(tmp_path, point, expected):
    artifact = pool("runup_move", residuals=(9.0,), buckets={
        "edges": (-float("inf"), 0.0, float("inf")), "pools": ((), (2.0, 3.0))}, min_pool=1)
    rows, refused, event = _build(_binding(tmp_path, pools=[pool("implied_t1"), artifact]))
    assert refused == ()
    selected, flag = driver_pool_from_artifact(rows[event.key][1].model, "runup_move", point)
    assert flag is None
    assert selected.tolist() == expected


def test_nonempty_bucket_with_zero_floor_remains_usable(tmp_path):
    artifact = pool("runup_move", residuals=(9.0,), buckets={
        "edges": (-float("inf"), float("inf")), "pools": ((2.0,),)}, min_pool=0)
    rows, refused, event = _build(_binding(tmp_path, pools=[pool("implied_t1"), artifact]))
    assert refused == ()
    selected, flag = driver_pool_from_artifact(rows[event.key][1].model, "runup_move", 1.0)
    assert flag is None
    assert selected.tolist() == [2.0]


@pytest.mark.parametrize("changes", [
    {"coefficients": (0.0,) * 5}, {"coefficients": (0.0,) * 7},
    {"coefficients": (float("nan"),) + (0.0,) * 5},
    {"coefficients": (float("inf"),) + (0.0,) * 5},
    {"residuals": ()}, {"residuals": (float("nan"),)}, {"residuals": (float("inf"),)},
    pytest.param({"residuals": (1.0, float("nan"))}, id="mixed-surface-invalid-after-finite"),
])
def test_hash_valid_but_unusable_surface_is_refused(tmp_path, changes):
    _assert_model_refusal(_binding(tmp_path, surfaces=[surface(**changes)]),
                          detail=_labelled_refusal_detail(
                              "RUNUP payoff surface is numerically unusable"))


def _line():
    return make_payoff_line_artifact(
        {"n": 2, "intercept": 10.0, "slope": 20.0, "resid_sd": 0.0, "r": 1.0, "residuals": (0.0,)},
        strategy="STR-RUNUP", driver="driver_prediction", alpha=0.5, cutoff="2026-01-09")


@pytest.mark.parametrize("reverse", [False, True])
def test_surface_selection_ignores_line_and_other_keys_regardless_of_order(tmp_path, reverse):
    selected = surface()
    candidates = [_line(), surface(alpha=0.6), surface(cutoff="2026-01-08"),
                  surface(cutoff="2026-01-10"), surface(strategy="STR-THRU"), selected]
    binding = _binding(tmp_path, surfaces=candidates[::-1] if reverse else candidates)
    rows, refused, event = _build(binding)
    assert refused == ()
    assert rows[event.key][1].model["payoff_artifact"] == selected


@pytest.mark.parametrize("candidates", [
    [_line()], [surface(alpha=0.6)], [surface(cutoff="2026-01-08")],
    [surface(cutoff="2026-01-10")], [surface(cutoff=None)], [surface(strategy="STR-THRU")],
    [surface(), surface(coefficients=(1.0,) * 6)], [surface(), surface()],
])
def test_missing_or_ambiguous_exact_surface_never_falls_back(tmp_path, candidates):
    _assert_model_refusal(_binding(tmp_path, surfaces=candidates),
                          detail=_labelled_refusal_detail(
                              "RUNUP requires one exact-key payoff surface"))


@pytest.mark.parametrize("as_of,entry,expected", [
    ("2026-01-10", "2026-01-08", "2026-01-08"),
    ("2026-01-07", "2026-01-09", "2026-01-07"),
])
def test_surface_cutoff_is_minimum_of_validated_decision_and_entry(tmp_path, as_of, entry, expected):
    event = _event()
    event = replace(event, panel_anchor="2026-01-06",
                    calendar_row={**event.calendar_row, "entry_date": entry,
                                  "calendar_observed_through": "2026-01-06"},
                    quote_rows=[{**quote, "observed_at": "2026-01-06"} for quote in event.quote_rows])
    binding = _binding(tmp_path, surfaces=[surface(cutoff=expected)])
    rows, refused, _ = _build(binding, event, as_of=as_of)
    assert refused == ()
    assert rows[event.key][1].model["payoff_recipe"] == {"before": expected}


def test_exact_driver_pool_is_required_even_with_a_healthy_alias_pool(tmp_path):
    binding = _binding(tmp_path, extra=(EXACT,))
    _assert_model_refusal(binding)
    binding = _binding(tmp_path / "complete", extra=(EXACT,), pools=[
        pool("driver", model_id="m-driver-STR-RUNUP", residuals=(4.0,)),
        pool("implied_t1", residuals=(9.0,)), pool("runup_move")])
    rows, refused, event = _build(binding)
    assert refused == ()
    assert rows[event.key][1].model["model_residual_artifacts"]["driver"].role == "driver"
    assert rows[event.key][1].model["model_residual_artifact_recipe"]["driver"] == {
        "role": "driver", "model_id": "m-driver-STR-RUNUP", "fold": None,
        "content_hash": binding.driver_residual_artifacts["driver"].content_hash}


@pytest.mark.parametrize("member", ["driver_residual_pool:implied_t1", "driver_residual_pool:runup_move",
                                     "payoff_surface:STR-RUNUP"])
def test_corrupt_required_staged_bytes_refuse_release_before_any_row(tmp_path, member):
    binding = _binding(tmp_path)
    assert binding.release_id == "r1"
    artifact = (binding.payoff_artifacts["STR-RUNUP"][0] if member.startswith("payoff")
                else binding.driver_residual_artifacts[member.split(":")[1]])
    row = state_row(tmp_path, member, [artifact])
    (tmp_path / "deployment" / row["objects"][0]["path"]).write_bytes(b"corrupt")
    with pytest.raises(ModelNotReady) as error:
        resolve_release_binding(tmp_path)
    assert error.value.code == "MODEL_NOT_READY"
    assert error.value.member_id == member


def test_missing_forecast_identity_refuses_model_declaration_without_partial_output(tmp_path):
    binding = _binding(tmp_path)
    spec = replace(batch._RUNUP_SPEC, drivers=(batch._RUNUP_SPEC.drivers[0],))
    code, detail, pair = batch._matched_decision_clock(binding.model_identity, spec)
    assert (code, detail) == (None, None)
    with pytest.raises(batch.NightlySourceBundleRefusal) as error:
        batch.runup_model_declarations(binding, {driver.output: driver.identity for driver in pair[0]},
                                      cutoff="2026-01-09", fill_alpha=0.5)
    assert error.value.code == "MODEL_NOT_READY"
