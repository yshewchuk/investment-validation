"""Batch assembly and worker semantics for ``engine/v2/ops/native_score_batch``."""
import hashlib
import json

import pandas as pd
import pytest

from engine.v2.contracts import ScoreRecord
from engine.v2.foundation import content_hash
from engine.v2.models import (
    ArtifactInventoryMember,
    ArtifactMember,
    ModelArtifactInventory,
    ModelBinding,
    ModelRelease,
    ModelReleaseInventory,
    ReleaseBinding,
    ReleaseRequirement,
)
from engine.v2.models import deployment
from engine.v2.ops.native_board_universe import BoardRequest
from engine.v2.ops.native_score_batch import (
    NativeScoreBatchRowRefusal,
    NightlyEventInputs,
    assemble_score_batch_inputs,
    run_native_score_batch_worker,
)
from engine.v2.scoring.application import score_one
from engine.v2.scoring.identity import bootstrap_seed, score_request_key
from engine.v2.scoring.release_bindings import resolve_release_binding
from engine.v2.scoring.stages import _model_seed

_AS_OF = "2026-01-10"
_SNAPSHOT = "snap-1"
_CALENDAR_REVISION = "cal-1"
_GATE_POLICY = {"STR-THRU": {"threshold": 0.0}}


def _sha(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _linear_payload(intercept=1.0, coefficient=2.0) -> bytes:
    return json.dumps({
        "schema_version": "linear_estimator.v1.0",
        "feature_order": ["x"],
        "outputs": [{"name": "prediction", "intercept": intercept,
                     "coefficients": [coefficient]}],
    }, sort_keys=True).encode()


def _write_empty_catalog(root, release_id: str) -> None:
    """A members-empty ``phase5_release.json``: resolve_release_binding
    requires the catalog to exist even when it declares no state objects."""
    body = {"schema_version": "phase5_staged_release.v1.0", "release_id": release_id,
            "deployment_id": "d1", "members": [], "sources": {}}
    body["manifest_hash"] = content_hash(
        {k: v for k, v in body.items() if k != "manifest_hash"})
    (root / "phase5_release.json").write_text(json.dumps(body, sort_keys=True))


def _stage_release(tmp_path, *, roles=("driver", "gate"),
                   clock_ids=None, release_id="r1"):
    """Stage one binding per role in ``roles`` for STR-THRU, promote, and
    return the real, resolved ScoringReleaseBinding. ``clock_ids`` maps
    role -> decision_clock_id string; defaults every role to
    "entry-close"."""
    clock_ids = clock_ids or {}
    bindings = []
    artifacts = []
    release_bindings = []
    requirements = []
    payloads = {}
    for role in roles:
        payload = _linear_payload()
        member_hash = _sha(payload)
        payloads[member_hash] = payload
        member = ArtifactMember(name="estimator", path="unused.json",
                                content_hash=member_hash)
        clock_id = clock_ids.get(role, "entry-close")
        binding_id = f"b-{role}"
        model_id = f"m-{role}"
        bindings.append(ModelBinding(
            binding_id=binding_id, model_id=model_id, role=role,
            strategy_id="STR-THRU", decision_clock_id=clock_id,
            adapter="json-linear.v1", feature_order=("x",),
            output_names=("prediction",), members=(member,)))
        artifacts.append(ModelArtifactInventory(
            artifact_id=model_id, role=role, strategy_ids=("STR-THRU",),
            compatible_clock_ids=(clock_id,), target_contract_ref="return.v1",
            ordered_features=("x",),
            members=(ArtifactInventoryMember(
                member_id=f"{model_id}:estimator", kind="estimator",
                artifact_ref=f"artifact://{model_id}",
                content_hash=member_hash),)))
        release_bindings.append(ReleaseBinding(
            role=role, strategy_id="STR-THRU", clock_id=clock_id,
            artifact_id=model_id, ordered_features=("x",),
            required_member_kinds=("estimator",)))
        requirements.append(ReleaseRequirement(
            role=role, strategy_id="STR-THRU", clock_id=clock_id))
    release = ModelRelease(release_id=release_id, deployment_id="d1",
                           bindings=tuple(bindings))
    inventory = ModelReleaseInventory(
        release_id=release_id, deployment_id="d1",
        known_clock_ids=tuple(sorted({clock_ids.get(r, "entry-close") for r in roles})),
        artifacts=tuple(artifacts), bindings=tuple(release_bindings),
        requirements=tuple(requirements), artifact_manifest_ref="manifest://r",
        evidence_refs=("evidence://r",))
    dep_root = tmp_path / "deployment"
    deployment.stage_release(dep_root, release, inventory, payloads)
    deployment.promote(dep_root, release_id)
    _write_empty_catalog(tmp_path, release_id)
    return resolve_release_binding(tmp_path)


def _calendar_row(**overrides) -> dict:
    row = {
        "event_id": "evt-1",
        "ticker": "TEST", "event_date": "2026-01-15",
        "entry_date": "2026-01-15", "exit_date": "2026-01-16",
        "expiry": "2026-01-16", "spot": 100.0,
        "calendar_observed_through": "2026-01-09",
    }
    row.update(overrides)
    return row


def _event_inputs(strategy="STR-THRU", ticker="TEST",
                  event_date=pd.Timestamp("2026-01-15"), **overrides) -> NightlyEventInputs:
    base = dict(
        key=BoardRequest(ticker=ticker, strategy=strategy,
                         event_date=event_date, session="am"),
        calendar_row=_calendar_row(),
        panel_row={"date": str(event_date.date()), "signal": 1.5},
        # A forward event's real anchor is FeatureVector.as_of (issue #53,
        # fixed by #67) -- calendar_observed_through is already staged
        # before as_of ("2026-01-10") in this fixture, so it doubles as a
        # convenient, always-valid default anchor here.
        panel_anchor="2026-01-09",
        tier4_row={"pred_abs_move": 0.05, "pred_abs_move_fold_start": "2026-01-01"},
        quote_rows=[
            {"right": "C", "strike": 100.0, "expiry": "2026-01-16",
             "bid": 1.0, "ask": 1.2, "observed_at": "2026-01-09"},
            {"right": "P", "strike": 100.0, "expiry": "2026-01-16",
             "bid": 1.1, "ask": 1.3, "observed_at": "2026-01-09"},
        ],
        quote_status=None,
    )
    base.update(overrides)
    return NightlyEventInputs(**base)


def _assemble(binding, events, **overrides):
    kwargs = dict(
        as_of=_AS_OF, snapshot_id=_SNAPSHOT, calendar_revision=_CALENDAR_REVISION,
        binding=binding, events=events,
        feature_names=("signal", "pred_abs_move"), gate_policy=_GATE_POLICY,
    )
    kwargs.update(overrides)
    return assemble_score_batch_inputs(**kwargs)


def test_happy_path_assembles_score_request_and_native_inputs(tmp_path):
    binding = _stage_release(tmp_path)
    event = _event_inputs()
    assembled, refusals = _assemble(binding, [event])
    assert refusals == ()
    assert list(assembled) == [event.key]
    request, native_inputs = assembled[event.key]
    assert request.strategy_version == "STR-THRU"
    assert request.mode == "shadow"
    assert request.deployment_id == "d1"
    assert request.decision_clock_id == "entry-close"
    assert request.event_id == "evt-1"
    record = score_one(request, native_inputs)
    assert isinstance(record, ScoreRecord)


def test_unsupported_strategy_refuses_without_sinking_batch(tmp_path):
    binding = _stage_release(tmp_path)
    good = _event_inputs()
    bad = _event_inputs(strategy="TWIN-P")
    assembled, refusals = _assemble(binding, [good, bad])
    assert list(assembled) == [good.key]
    assert len(refusals) == 1
    assert isinstance(refusals[0], NativeScoreBatchRowRefusal)
    assert refusals[0].code == "UNSUPPORTED_STRATEGY"
    assert refusals[0].key == bad.key


def test_release_missing_driver_role_refuses(tmp_path):
    binding = _stage_release(tmp_path, roles=("gate",))
    event = _event_inputs()
    assembled, refusals = _assemble(binding, [event])
    assert assembled == {}
    assert len(refusals) == 1
    assert refusals[0].code == "RELEASE_MISSING_ROLE"
    assert "driver:STR-THRU" in refusals[0].detail


def test_gate_policy_not_staged_refuses(tmp_path):
    binding = _stage_release(tmp_path)
    event = _event_inputs()
    assembled, refusals = _assemble(binding, [event], gate_policy={})
    assert assembled == {}
    assert len(refusals) == 1
    assert refusals[0].code == "GATE_POLICY_NOT_STAGED"


def test_ambiguous_decision_clock_refuses(tmp_path):
    binding = _stage_release(tmp_path, clock_ids={"driver": "entry-close",
                                                  "gate": "entry-open"})
    event = _event_inputs()
    assembled, refusals = _assemble(binding, [event])
    assert assembled == {}
    assert len(refusals) == 1
    assert refusals[0].code == "AMBIGUOUS_DECISION_CLOCK"


def test_post_as_of_panel_anchor_refuses_without_sinking_batch(tmp_path):
    """Issue #53 (fixed by #67): a planted panel_anchor after as_of is a
    per-row POST_AS_OF_ROW refusal, re-wrapped from
    assemble_nightly_source_bundle's own NightlySourceBundleRefusal --
    never a batch-level exception."""
    binding = _stage_release(tmp_path)
    good = _event_inputs()
    bad = _event_inputs(
        key=BoardRequest(ticker="OTHER", strategy="STR-THRU",
                         event_date=pd.Timestamp("2026-01-15"), session="am"),
        panel_anchor="2026-01-11")  # after as_of (2026-01-10)
    assembled, refusals = _assemble(binding, [good, bad])
    assert list(assembled) == [good.key]
    assert len(refusals) == 1
    assert refusals[0].code == "POST_AS_OF_ROW"
    assert refusals[0].key == bad.key


def test_duplicate_event_key_raises(tmp_path):
    binding = _stage_release(tmp_path)
    with pytest.raises(ValueError):
        _assemble(binding, [_event_inputs(), _event_inputs()])


def test_mc_seed_matches_bootstrap_seed_from_snapshot_and_key(tmp_path):
    binding = _stage_release(tmp_path)
    event = _event_inputs()
    assembled, refusals = _assemble(binding, [event])
    assert refusals == ()
    _, native_inputs = assembled[event.key]
    key = score_request_key(native_inputs.context)
    expected = bootstrap_seed(native_inputs.context["snapshot"], key)
    assert _model_seed(native_inputs, {}) == expected


def _event_doc(strategy="STR-THRU", event_id="evt-1") -> dict:
    calendar_row = _calendar_row(event_id=event_id)
    return {
        "key": {"ticker": "TEST", "strategy": strategy,
                "event_date": "2026-01-15", "session": "am"},
        "calendar_row": calendar_row,
        "panel_row": {"date": "2026-01-15", "signal": 1.5},
        "panel_anchor": "2026-01-09",
        "tier4_row": {"pred_abs_move": 0.05,
                      "pred_abs_move_fold_start": "2026-01-01"},
        "quote_rows": [
            {"right": "C", "strike": 100.0, "expiry": "2026-01-16",
             "bid": 1.0, "ask": 1.2, "observed_at": "2026-01-09"},
            {"right": "P", "strike": 100.0, "expiry": "2026-01-16",
             "bid": 1.1, "ask": 1.3, "observed_at": "2026-01-09"},
        ],
        "quote_status": None,
    }


def _worker_parameters(tmp_path, *, expected_ids) -> dict:
    return {
        "release_root": str(tmp_path), "as_of": _AS_OF, "snapshot_id": _SNAPSHOT,
        "calendar_revision": _CALENDAR_REVISION,
        "feature_names": ["signal", "pred_abs_move"],
        "gate_policy": {"STR-THRU": {"threshold": 0.0}},
        "expected_ids": expected_ids,
    }


def test_run_native_score_batch_worker_writes_records_and_refusals(tmp_path):
    _stage_release(tmp_path)
    root = tmp_path / "staging"
    root.mkdir()
    (root / "events.json").write_text(json.dumps(
        [_event_doc(), _event_doc(strategy="TWIN-P", event_id="evt-2")]))
    result = run_native_score_batch_worker(
        _worker_parameters(tmp_path, expected_ids=["native_score_batch"]), root)
    assert result["completed_ids"] == ["native_score_batch"]
    assert result["no_work"] is False
    records_document = json.loads((root / "records.json").read_text())
    assert records_document["authoritative"] is False
    assert records_document["known_gaps"] == ["PANEL_ANCHOR_UNVERIFIED"]
    assert len(records_document["records"]) == 1
    refusals_document = json.loads((root / "refusals.json").read_text())
    assert len(refusals_document) == 1
    assert refusals_document[0]["code"] == "UNSUPPORTED_STRATEGY"


def test_empty_events_is_a_legitimate_no_op(tmp_path):
    binding = _stage_release(tmp_path)
    assert _assemble(binding, ()) == ({}, ())
    root = tmp_path / "staging"
    root.mkdir()
    (root / "events.json").write_text("[]")
    result = run_native_score_batch_worker(
        _worker_parameters(tmp_path, expected_ids=[]), root)
    assert result["no_work"] is True
    assert result["completed_ids"] == []
    assert json.loads((root / "records.json").read_text())["records"] == []
    assert json.loads((root / "refusals.json").read_text()) == []
