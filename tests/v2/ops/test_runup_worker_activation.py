"""Real RUNUP worker activation from synthetic, frozen, staged release bytes."""
import hashlib
import json
import math
from dataclasses import asdict, replace
from types import SimpleNamespace

import pandas as pd
import pytest

from engine.models.training.gate import FEATURES as GATE_FEATURES
from engine.models.training.implied_t1 import FEATURES as IMPLIED_FEATURES
from engine.models.training.runup_move import FEATURES as MOVE_FEATURES
from engine.v2.models import ArtifactInventoryMember, ArtifactMember, deployment
from engine.v2.models.no_fit import no_fit_guard
from engine.v2.ops import native_score_batch as batch
from engine.v2.ops.native_board_universe import BoardRequest
from engine.v2.scoring.release_bindings import resolve_gate_policy
from tests.v2.ops import test_native_score_batch_driver_alias as release_helpers
from tests.v2.ops.runup_helpers import pool, state_row, surface

AS_OF = "2026-01-05"
EVENT_DATE = "2026-01-15"
KEY = f"TEST|STR-RUNUP|{EVENT_DATE}|BMO"
IDENTITY_KEYS = ("implied_t1:*", "runup_move:*", "gate:STR-RUNUP")
SPECS = (
    ("implied_t1", "*", IMPLIED_FEATURES),
    ("runup_move", "*", MOVE_FEATURES),
    ("gate", "STR-RUNUP", GATE_FEATURES),
)


@pytest.fixture(autouse=True)
def forbid_fitting():
    with no_fit_guard():
        yield


def _linear_payload(order, intercept, coefficients=None):
    coefficients = coefficients or {}
    return json.dumps({
        "schema_version": "linear_estimator.v1.0",
        "feature_order": list(order),
        "outputs": [{"name": "prediction", "intercept": intercept,
                     "coefficients": [coefficients.get(name, 0.0) for name in order]}],
    }, sort_keys=True).encode()


def _stage_with_threshold(root, release, inventory, payloads):
    """Extend only the synthetic fixture; the real deployment writes/verifies it."""
    payload = json.dumps({"models": [{"id": "m-gate-STR-RUNUP", "threshold": 1.0}]}).encode()
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    member = ArtifactMember(name="threshold", path="unused-threshold.json", content_hash=digest)
    inventory_member = ArtifactInventoryMember(
        member_id="m-gate-STR-RUNUP:threshold", kind="threshold",
        artifact_ref="artifact://synthetic-gate-threshold", content_hash=digest)
    release = replace(release, bindings=tuple(
        replace(binding, members=(*binding.members, member)) if binding.role == "gate" else binding
        for binding in release.bindings))
    inventory = replace(
        inventory,
        artifacts=tuple(replace(artifact, members=(*artifact.members, inventory_member))
                        if artifact.role == "gate" else artifact for artifact in inventory.artifacts),
        bindings=tuple(replace(binding, required_member_kinds=("estimator", "threshold"))
                       if binding.role == "gate" else binding for binding in inventory.bindings))
    return deployment.stage_release(root, release, inventory, {**payloads, digest: payload})


@pytest.fixture
def stage_release(tmp_path, monkeypatch):
    def stage(raw_move=2.0):
        root = tmp_path / "release"
        pools = {"driver": pool("implied_t1", residuals=(1.0, 1.0)),
                 "runup_move": pool("runup_move", residuals=(-0.5, -0.5))}
        payoff = surface(cutoff=AS_OF)
        rows = [state_row(root, f"driver_residual_pool:{artifact.role}", [artifact])
                for artifact in pools.values()]
        rows.append(state_row(root, "payoff_surface:STR-RUNUP", [payoff]))
        payloads = iter((
            _linear_payload(IMPLIED_FEATURES, 7.0),
            _linear_payload(MOVE_FEATURES, raw_move),
            _linear_payload(GATE_FEATURES, 0.5,
                            {"days_to_print": 0.01, "entry_cost_pct": 0.02, "dte_entry": 0.03}),
        ))
        # Patch only the shared test helper's namespace. Production deployment,
        # release resolution, registry, assembly and scoring remain untouched.
        with monkeypatch.context() as fixture_patch:
            fixture_patch.setattr(release_helpers, "_linear_payload", lambda order: next(payloads))
            fixture_patch.setattr(release_helpers, "deployment", SimpleNamespace(
                stage_release=_stage_with_threshold,
                mark_staging_succeeded=deployment.mark_staging_succeeded,
                promote=deployment.promote))
            binding = release_helpers._stage(root, SPECS, rows=rows)
        return root, binding, pools, payoff

    return stage


def _event_document(horizon=7):
    features = {name: 1.0 for name in set(IMPLIED_FEATURES) | set(MOVE_FEATURES) | set(GATE_FEATURES)}
    features.update(date=EVENT_DATE, spot=100.0, mcap_log=math.log(1e10), im=7.0)
    # These namesakes must lose to calendar ownership and native gate derivation.
    features.update(days_before_print=999, days_to_print=999, entry_cost_pct=999, dte_entry=999)
    return {
        "key": {"ticker": "TEST", "strategy": "STR-RUNUP",
                "event_date": EVENT_DATE, "session": "BMO"},
        "calendar_row": {
            "event_id": "evt-runup", "ticker": "TEST", "event_date": EVENT_DATE,
            "entry_date": AS_OF, "exit_date": "2026-01-14", "expiry": "2026-01-16",
            "spot": 100.0, "calendar_observed_through": AS_OF, "days_before_print": horizon,
        },
        "panel_row": features, "panel_anchor": AS_OF, "tier4_row": {},
        "quote_rows": [{"right": right, "strike": 100.0, "expiry": "2026-01-16",
                        "bid": 2.0, "ask": 2.0, "observed_at": AS_OF,
                        "chain_spot": 100.0}
                       for right in ("C", "P")],
    }


def _worker(release_root, root, document):
    root.mkdir()
    event_bytes = json.dumps([document], sort_keys=True).encode()
    (root / "events.json").write_bytes(event_bytes)
    batch.run_native_score_batch_worker({
        "release_root": str(release_root), "as_of": AS_OF,
        "snapshot_id": "synthetic-snap", "calendar_revision": "synthetic-cal",
        "feature_names": [], "expected_ids": ["evt-runup"],
        # Deliberately no caller gate_policy: resolve the frozen threshold member.
    }, root)
    assert (root / "events.json").read_bytes() == event_bytes
    records = (root / "records.json").read_bytes()
    refusals = (root / "refusals.json").read_bytes()
    assert json.loads(refusals) == {
        "schema_version": "native_score_batch_refusals.v2.0",
        "refusals": {}, "unkeyable_refusals": [],
    }
    document = json.loads(records)
    assert document["schema_version"] == "native_score_batch_records.v2.0"
    assert document["authoritative"] is False
    assert document["known_gaps"] == []
    assert set(document["records"]) == {KEY}
    return document["records"][KEY], (records, refusals)


@pytest.mark.parametrize("horizon,raw_move,published,band,expected_pnl", [
    pytest.param(7, 2.0, 1.0, 0.75, 0.3375, id="scale-raw-D14-once"),
    pytest.param(0, 2.0, 0.0, 0.0, 0.3, id="zero-horizon"),
    pytest.param(7, -2.0, 0.0, 0.0, 0.3, id="negative-raw-clamped"),
])
def test_real_worker_scores_runup_and_repeats_byte_identically(
        tmp_path, stage_release, horizon, raw_move, published, band, expected_pnl):
    root, binding, _, _ = stage_release(raw_move)
    document = _event_document(horizon)
    record, first_bytes = _worker(root, tmp_path / "first", document)
    _, second_bytes = _worker(root, tmp_path / "second", document)
    assert second_bytes == first_bytes
    assert record["validation_status"] == "scored"
    assert record["readiness"] == "ready"
    assert record["reason_codes"] == ["PROJECTED_CALENDAR"]
    assert record["forecasts"]["driver_prediction"] == 7.0
    assert record["forecasts"]["runup_move_raw_d14"] == raw_move
    assert record["forecasts"]["runup_move_prediction"] == published

    resolved = record["resolved_request"]
    assert resolved["days_before_print"] == horizon
    assert resolved["driver_name"] == "im_t1"
    assert resolved["driver_p10"] == resolved["driver_p90"] == 8.0
    assert resolved["runup_move_p10"] == resolved["runup_move_p90"] == band
    # Independent oracle: (.02 + .004 * 8 + .002 * .75) * 100 / 4 - 1
    # is .3375; either clamping case has zero move and therefore returns .3.
    assert resolved["exp_pnl_model"] == pytest.approx(expected_pnl, rel=0.0, abs=1e-12)
    assert resolved["win_model"] == 1.0
    assert resolved["win_model_raw"] == 1.0
    assert resolved["entry_cost"] == 4.0
    assert resolved["entry_cost_pct"] == 4.0
    assert resolved["days_to_print"] == 10.0
    assert resolved["dte_entry"] == 11
    assert record["gate_terms"] == {
        "gate_score": pytest.approx(0.5 + 0.01 * 10 + 0.02 * 4 + 0.03 * 11),
        "gate_threshold": 1.0, "gate_pass": True,
    }
    assert record["forecasts"]["exp_pnl_sim"] is None
    assert not {"exp_pnl_analog", "win_analog", "n_analogs", "exp_pnl_sim", "win_sim"} & resolved.keys()
    identities = {key: json.loads(json.dumps(asdict(binding.model_identity[key])))
                  for key in IDENTITY_KEYS}
    references = [binding.model_identity[key].artifact_hash for key in IDENTITY_KEYS]
    assert resolved["model_identity"] == identities
    assert record["model_artifact_ids"] == references
    assert record["canonical_request"]["model_artifact_refs"] == references
    assert record["uncertainty"]["runup_move_p10"] == band
    assert record["uncertainty"]["runup_move_p90"] == band


def test_assembler_retains_frozen_pool_surface_and_executor_recipes(stage_release):
    root, binding, pools, payoff = stage_release()
    document = _event_document()
    event = batch.NightlyEventInputs(
        **{name: value for name, value in document.items() if name != "key"},
        key=BoardRequest(**{**document["key"], "event_date": pd.Timestamp(EVENT_DATE)}))
    assembled, refusals = batch.assemble_score_batch_inputs(
        as_of=AS_OF, snapshot_id="synthetic-snap", calendar_revision="synthetic-cal",
        binding=binding, events=[event], feature_names=[],
        gate_policy=resolve_gate_policy(binding, root))
    assert refusals == ()
    request, inputs = assembled[event.key]
    assert inputs.model == {
        "payoff_recipe": {"before": AS_OF}, "payoff_artifact": payoff,
        "model_residual_artifacts": pools,
        "model_residual_artifact_recipe": {
            slot: {"role": artifact.role, "model_id": artifact.model_id,
                   "fold": None, "content_hash": artifact.content_hash}
            for slot, artifact in pools.items()},
    }
    assert inputs.simulation == {"mode": "not_applicable"}
    assert inputs.analogs == {"recipe": None}
    assert inputs.features["model_inputs"]["days_before_print"] == 7
    assert not {"days_to_print", "entry_cost_pct", "dte_entry"} & inputs.features["model_inputs"].keys()
    assert inputs.forecast["driver_name"] == "im_t1"
    assert inputs.forecast["models"] == {}
    outputs = {"driver_prediction": "implied_t1:*", "runup_move_prediction": "runup_move:*"}
    assert inputs.forecast["model_artifact_refs"] == {
        output: binding.model_identity[key].artifact_hash for output, key in outputs.items()}
    assert {output: executor.binding_id for output, executor in inputs.forecast["executors"].items()} == {
        output: binding.model_identity[key].binding_id for output, key in outputs.items()}
    assert inputs.forecast["artifact_hashes"] == request.model_artifact_refs[:2]
    assert inputs.gate["binding_id"] == "b-gate-STR-RUNUP"
    assert inputs.gate["executors"]["gate_score"].binding_id == "b-gate-STR-RUNUP"
    assert inputs.gate["threshold"] == 1.0
