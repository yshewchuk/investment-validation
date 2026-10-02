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
    _decode_producer_refusals,
    _native_score_batch_documents,
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
        calendar_row=_calendar_row(ticker="OTHER"),
        panel_anchor="2026-01-11")  # after as_of (2026-01-10)
    assembled, refusals = _assemble(binding, [good, bad])
    assert list(assembled) == [good.key]
    assert len(refusals) == 1
    assert refusals[0].code == "POST_AS_OF_ROW"
    assert refusals[0].key == bad.key


def test_calendar_row_key_mismatch_refuses_without_sinking_batch(tmp_path):
    """CodeRabbit round 2 (PR #66): calendar_row is never checked against its
    own BoardRequest key anywhere else -- a staged calendar_row for the
    wrong ticker/event_date is a per-row refusal, checked before every other
    per-row check."""
    binding = _stage_release(tmp_path)
    good = _event_inputs()
    bad = _event_inputs(
        key=BoardRequest(ticker="OTHER", strategy="STR-THRU",
                         event_date=pd.Timestamp("2026-01-15"), session="am"),
        calendar_row=_calendar_row())  # calendar_row.ticker stays "TEST"
    assembled, refusals = _assemble(binding, [good, bad])
    assert list(assembled) == [good.key]
    assert len(refusals) == 1
    assert refusals[0].code == "CALENDAR_ROW_KEY_MISMATCH"
    assert refusals[0].key == bad.key


def test_calendar_row_unparseable_event_date_refuses_without_raising(tmp_path):
    """CodeRabbit round 3 (PR #66): _iso raises on an unparseable value --
    _calendar_row_problem must convert that into this row's own refusal
    rather than letting the exception escape and abort every other row's
    assembly."""
    binding = _stage_release(tmp_path)
    good = _event_inputs()
    bad = _event_inputs(
        key=BoardRequest(ticker="OTHER", strategy="STR-THRU",
                         event_date=pd.Timestamp("2026-01-15"), session="am"),
        calendar_row=_calendar_row(ticker="OTHER", event_date="not-a-date"))
    assembled, refusals = _assemble(binding, [good, bad])
    assert list(assembled) == [good.key]
    assert len(refusals) == 1
    assert refusals[0].code == "CALENDAR_ROW_INVALID"
    assert refusals[0].key == bad.key


def test_calendar_row_not_a_mapping_refuses_without_raising(tmp_path):
    """CodeRabbit round 5 (PR #66): a null calendar_row in events.json must
    not reach calendar_row.get() and raise AttributeError -- it is this
    row's own CALENDAR_ROW_INVALID refusal instead."""
    binding = _stage_release(tmp_path)
    good = _event_inputs()
    bad = _event_inputs(
        key=BoardRequest(ticker="OTHER", strategy="STR-THRU",
                         event_date=pd.Timestamp("2026-01-15"), session="am"),
        calendar_row=None)
    assembled, refusals = _assemble(binding, [good, bad])
    assert list(assembled) == [good.key]
    assert len(refusals) == 1
    assert refusals[0].code == "CALENDAR_ROW_INVALID"
    assert refusals[0].key == bad.key


def test_calendar_row_unparseable_expiry_refuses_without_raising(tmp_path):
    """CodeRabbit round 5 (PR #66): _identity_context parses calendar_row
    ["expiry"] outside every try/except in _assemble_one_event -- an
    unparseable expiry must be caught in _calendar_row_problem, before that
    point is ever reached."""
    binding = _stage_release(tmp_path)
    good = _event_inputs()
    bad = _event_inputs(
        key=BoardRequest(ticker="OTHER", strategy="STR-THRU",
                         event_date=pd.Timestamp("2026-01-15"), session="am"),
        calendar_row=_calendar_row(ticker="OTHER", expiry="not-a-date"))
    assembled, refusals = _assemble(binding, [good, bad])
    assert list(assembled) == [good.key]
    assert len(refusals) == 1
    assert refusals[0].code == "CALENDAR_ROW_INVALID"
    assert refusals[0].key == bad.key


def test_duplicate_event_key_raises(tmp_path):
    binding = _stage_release(tmp_path)
    with pytest.raises(ValueError):
        _assemble(binding, [_event_inputs(), _event_inputs()])


@pytest.mark.parametrize("bad_as_of", [None, "not-a-date",
                                       pd.Timestamp("2026-01-10", tz="UTC")])
def test_invalid_as_of_raises_batch_level(tmp_path, bad_as_of):
    """Opus gate (PR #66): as_of is a batch-level, not per-row, argument, so
    a None/unparseable/timezone-aware value must raise here rather than
    only surfacing once assemble_nightly_source_bundle re-validates it
    inside every single row (which would refuse every row while the
    attempt still reported success)."""
    binding = _stage_release(tmp_path)
    with pytest.raises(ValueError):
        _assemble(binding, [_event_inputs()], as_of=bad_as_of)


@pytest.mark.parametrize("field", ["snapshot_id", "calendar_revision"])
@pytest.mark.parametrize("bad_value", [None, "", 123])
def test_invalid_snapshot_or_calendar_revision_raises_batch_level(
        tmp_path, field, bad_value):
    """Opus gate (PR #66): a non-string or empty snapshot_id/
    calendar_revision must raise here rather than flowing straight into
    every row's ScoreRequest as the literal string "None" or "" via
    str()."""
    binding = _stage_release(tmp_path)
    with pytest.raises(ValueError):
        _assemble(binding, [_event_inputs()], **{field: bad_value})


def test_duplicate_request_hash_across_distinct_keys_raises(tmp_path):
    """CodeRabbit round 2 (PR #66): ScoreRequest carries no ticker of its
    own, so two DISTINCT BoardRequest keys whose calendar_row shares the
    same event_id (a caller-side data bug) collide into the same
    request_hash. Nothing in this module can say which row is the bad one,
    so the whole batch raises rather than letting one row silently clobber
    the other's inputs in fields_by_request."""
    binding = _stage_release(tmp_path)
    first = _event_inputs(
        key=BoardRequest(ticker="AAPL", strategy="STR-THRU",
                         event_date=pd.Timestamp("2026-01-15"), session="am"),
        calendar_row=_calendar_row(ticker="AAPL", event_id="evt-dup"))
    second = _event_inputs(
        key=BoardRequest(ticker="MSFT", strategy="STR-THRU",
                         event_date=pd.Timestamp("2026-01-15"), session="am"),
        calendar_row=_calendar_row(ticker="MSFT", event_id="evt-dup"))
    with pytest.raises(ValueError, match="duplicate request_hash"):
        _assemble(binding, [first, second])


def test_mc_seed_matches_bootstrap_seed_from_snapshot_and_key(tmp_path):
    """CodeRabbit round 2 (PR #66): the expected key/seed is built here from
    the fixture's own declared literals, never read back out of
    native_inputs.context -- a comparison against the object under test's
    own field would pass even if assembly wrote the wrong snapshot/date.
    Also checks repeatability (acceptance criterion 5): assembling the same
    event twice must draw the identical seed."""
    binding = _stage_release(tmp_path)
    event = _event_inputs()
    expected_context = {
        "ticker": "TEST", "strategy": "STR-THRU",
        "requested_as_of": _AS_OF, "requested_event_date": "2026-01-15",
        "requested_strike": None, "requested_expiry": "2026-01-16",
        "fill_alpha": 0.5, "variant": None, "decision_offset": None,
        "quote_max_age_sessions": None, "chain_as_of": _AS_OF,
    }
    expected = bootstrap_seed(_SNAPSHOT, score_request_key(expected_context))

    for _ in range(2):  # repeatability: same event, same seed, every time
        assembled, refusals = _assemble(binding, [event])
        assert refusals == ()
        _, native_inputs = assembled[event.key]
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


def _record_matches_identity(record, *, event_id, strategy_version, deployment_id) -> bool:
    """Independent identity check for one score record's canonical_request
    (CodeRabbit round 2, PR #66) -- shared by the regression assertion below
    and the planted-defect test that proves it can actually fail."""
    req = record["canonical_request"]
    return (req["event_id"] == event_id and req["strategy_version"] == strategy_version
            and req["deployment_id"] == deployment_id)


def test_planted_record_identity_corruption_is_caught():
    """Proves _record_matches_identity is not a tautology that would pass
    regardless of the worker's real output: it must fail on a corrupted
    canonical_request field."""
    good = {"canonical_request": {"event_id": "evt-1", "strategy_version": "STR-THRU",
                                  "deployment_id": "d1"}}
    assert _record_matches_identity(good, event_id="evt-1", strategy_version="STR-THRU",
                                    deployment_id="d1")
    corrupted = {"canonical_request": {**good["canonical_request"], "event_id": "evt-WRONG"}}
    assert not _record_matches_identity(
        corrupted, event_id="evt-1", strategy_version="STR-THRU", deployment_id="d1")


def _refusal_matches_key(document, canonical_key, expected_code) -> None:
    """Independent identity check for one serialized keyed refusal (v2.0
    shape) -- the assembly-level tests above only check the in-memory
    BoardRequest key; this checks the published refusals.json document
    itself."""
    assert document["refusals"][canonical_key]["code"] == expected_code


def test_planted_refusal_key_corruption_is_caught():
    """Proves _refusal_matches_key is not a tautology: it must fail on a
    corrupted serialized refusal code."""
    good = {"refusals": {"TEST|TWIN-P|2026-01-15|am": {
        "code": "UNSUPPORTED_STRATEGY", "detail": "only STR-THRU is supported"}}}
    _refusal_matches_key(good, "TEST|TWIN-P|2026-01-15|am", "UNSUPPORTED_STRATEGY")
    corrupted = {"refusals": {"TEST|TWIN-P|2026-01-15|am": {
        "code": "WRONG_CODE", "detail": "only STR-THRU is supported"}}}
    with pytest.raises(AssertionError):
        _refusal_matches_key(corrupted, "TEST|TWIN-P|2026-01-15|am",
                             "UNSUPPORTED_STRATEGY")


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
    assert records_document["schema_version"] == "native_score_batch_records.v2.0"
    assert records_document["authoritative"] is False
    # Issue #53 (fixed by #67): known_gaps is empty now that
    # assemble_nightly_source_bundle verifies panel_anchor itself.
    assert records_document["known_gaps"] == []
    assert len(records_document["records"]) == 1
    # CodeRabbit round 2 (PR #66): compare actual record CONTENT against an
    # independently-known expected identity, not just the record count.
    assert _record_matches_identity(
        records_document["records"]["TEST|STR-THRU|2026-01-15|am"],
        event_id="evt-1", strategy_version="STR-THRU", deployment_id="d1")
    refusals_document = json.loads((root / "refusals.json").read_text())
    assert refusals_document["schema_version"] == "native_score_batch_refusals.v2.0"
    assert len(refusals_document["refusals"]) == 1
    assert refusals_document["unkeyable_refusals"] == []
    # CodeRabbit round 5 (PR #66): compare the refusal's fully serialized
    # canonical key against an independently-known expected identity, not
    # just its code.
    _refusal_matches_key(refusals_document, "TEST|TWIN-P|2026-01-15|am",
                         "UNSUPPORTED_STRATEGY")


def test_run_native_score_batch_worker_merges_producer_refusals_when_present(tmp_path):
    _stage_release(tmp_path)
    root = tmp_path / "staging"
    root.mkdir()
    (root / "events.json").write_text(json.dumps([_event_doc()]))
    (root / "producer_refusals.json").write_text(json.dumps({
        "schema_version": "native_score_batch_producer_refusals.v1.0",
        "refusals": [{
            "key": {"ticker": "OTHER", "strategy": "STR-THRU",
                    "event_date": "2026-01-20", "session": "am"},
            "code": "EVENT_NOT_FOUND",
            "detail": "no matching calendar event",
        }],
    }))
    run_native_score_batch_worker(
        _worker_parameters(tmp_path, expected_ids=["native_score_batch"]), root)
    records_document = json.loads((root / "records.json").read_text())
    assert len(records_document["records"]) == 1
    assert _record_matches_identity(
        records_document["records"]["TEST|STR-THRU|2026-01-15|am"],
        event_id="evt-1", strategy_version="STR-THRU", deployment_id="d1")
    refusals_document = json.loads((root / "refusals.json").read_text())
    _refusal_matches_key(refusals_document, "OTHER|STR-THRU|2026-01-20|am",
                         "EVENT_NOT_FOUND")


def test_run_native_score_batch_worker_without_producer_refusals_file_matches_absent_and_empty(
        tmp_path):
    _stage_release(tmp_path)
    for root_name, producer_body in (
        ("staging_absent", None),
        ("staging_empty", {"schema_version": "native_score_batch_producer_refusals.v1.0",
                           "refusals": []}),
    ):
        root = tmp_path / root_name
        root.mkdir()
        (root / "events.json").write_text(json.dumps([_event_doc()]))
        if producer_body is not None:
            (root / "producer_refusals.json").write_text(json.dumps(producer_body))
        run_native_score_batch_worker(
            _worker_parameters(tmp_path, expected_ids=["native_score_batch"]), root)
    absent_root = tmp_path / "staging_absent"
    empty_root = tmp_path / "staging_empty"
    assert (json.loads((absent_root / "refusals.json").read_text())
            == json.loads((empty_root / "refusals.json").read_text()))
    assert (json.loads((absent_root / "records.json").read_text())
            == json.loads((empty_root / "records.json").read_text()))


def test_run_native_score_batch_worker_producer_refusal_colliding_with_record_raises(tmp_path):
    _stage_release(tmp_path)
    root = tmp_path / "staging"
    root.mkdir()
    (root / "events.json").write_text(json.dumps([_event_doc()]))
    (root / "producer_refusals.json").write_text(json.dumps({
        "schema_version": "native_score_batch_producer_refusals.v1.0",
        "refusals": [{
            "key": {"ticker": "TEST", "strategy": "STR-THRU",
                    "event_date": "2026-01-15", "session": "am"},
            "code": "EVENT_NOT_FOUND",
            "detail": "no matching calendar event",
        }],
    }))
    with pytest.raises(ValueError):
        run_native_score_batch_worker(
            _worker_parameters(tmp_path, expected_ids=["native_score_batch"]), root)


def test_decode_producer_refusals_rejects_wrong_schema_version():
    with pytest.raises(ValueError):
        _decode_producer_refusals({"schema_version": "wrong", "refusals": []})
    with pytest.raises(ValueError):
        _decode_producer_refusals({
            "schema_version": "native_score_batch_producer_refusals.v1.0",
            "refusals": "not-a-list",
        })


def test_run_native_score_batch_worker_scores_under_no_fit_guard(tmp_path, monkeypatch):
    """CodeRabbit round 5 (PR #66): assert directly that fitting is forbidden
    at the score_batch call site, so a mutation that drops the no_fit_guard
    wrapping (while the model itself never fits) is caught. The count/content
    checks above only prove scoring happened, not that it happened guarded."""
    from engine.v2.models.no_fit import fitting_forbidden
    from engine.v2.ops import native_score_batch as nsb_module

    _stage_release(tmp_path)
    root = tmp_path / "staging"
    root.mkdir()
    (root / "events.json").write_text(json.dumps([_event_doc()]))

    real_score_batch = nsb_module.score_batch
    seen: dict = {}

    def spy(*args, **kwargs):
        seen["forbidden"] = fitting_forbidden()
        return real_score_batch(*args, **kwargs)

    monkeypatch.setattr(nsb_module, "score_batch", spy)
    run_native_score_batch_worker(
        _worker_parameters(tmp_path, expected_ids=["native_score_batch"]), root)
    assert seen["forbidden"] is True


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
    assert json.loads((root / "records.json").read_text())["records"] == {}
    refusals_document = json.loads((root / "refusals.json").read_text())
    assert refusals_document["refusals"] == {}
    assert refusals_document["unkeyable_refusals"] == []


def test_invalid_key_field_ticker_delimiter_is_unkeyable(tmp_path):
    binding = _stage_release(tmp_path)
    good = _event_inputs()
    bad = _event_inputs(ticker="TE|ST", calendar_row=_calendar_row(ticker="TE|ST"))
    assembled, refusals = _assemble(binding, [good, bad])
    assert list(assembled) == [good.key]
    assert len(refusals) == 1
    assert refusals[0].code == "INVALID_KEY_FIELD"

    root = tmp_path / "staging"
    root.mkdir()
    bad_doc = _event_doc(event_id="evt-2")
    bad_doc["key"]["ticker"] = "TE|ST"
    bad_doc["calendar_row"]["ticker"] = "TE|ST"
    (root / "events.json").write_text(json.dumps([_event_doc(), bad_doc]))
    run_native_score_batch_worker(
        _worker_parameters(tmp_path, expected_ids=["native_score_batch"]), root)
    refusals_document = json.loads((root / "refusals.json").read_text())
    assert refusals_document["refusals"] == {}
    assert len(refusals_document["unkeyable_refusals"]) == 1
    unkeyable = refusals_document["unkeyable_refusals"][0]
    assert unkeyable["code"] == "INVALID_KEY_FIELD"
    assert unkeyable["key"]["ticker"] == "TE|ST"
    records_document = json.loads((root / "records.json").read_text())
    assert len(records_document["records"]) == 1


def test_invalid_key_field_session_delimiter_is_unkeyable(tmp_path):
    binding = _stage_release(tmp_path)
    good = _event_inputs()
    bad = _event_inputs(key=BoardRequest(ticker="TEST", strategy="STR-THRU",
                                         event_date=pd.Timestamp("2026-01-15"),
                                         session="a|m"))
    assembled, refusals = _assemble(binding, [good, bad])
    assert list(assembled) == [good.key]
    assert len(refusals) == 1
    assert refusals[0].code == "INVALID_KEY_FIELD"

    root = tmp_path / "staging"
    root.mkdir()
    bad_doc = _event_doc(event_id="evt-2")
    bad_doc["key"]["session"] = "a|m"
    (root / "events.json").write_text(json.dumps([_event_doc(), bad_doc]))
    run_native_score_batch_worker(
        _worker_parameters(tmp_path, expected_ids=["native_score_batch"]), root)
    refusals_document = json.loads((root / "refusals.json").read_text())
    assert refusals_document["refusals"] == {}
    assert len(refusals_document["unkeyable_refusals"]) == 1
    unkeyable = refusals_document["unkeyable_refusals"][0]
    assert unkeyable["code"] == "INVALID_KEY_FIELD"
    assert unkeyable["key"]["session"] == "a|m"
    records_document = json.loads((root / "records.json").read_text())
    assert len(records_document["records"]) == 1


def test_invalid_key_field_takes_priority_over_unsupported_strategy(tmp_path):
    binding = _stage_release(tmp_path)
    bad = _event_inputs(strategy="TWIN|P")
    assembled, refusals = _assemble(binding, [bad])
    assert assembled == {}
    assert len(refusals) == 1
    assert refusals[0].code == "INVALID_KEY_FIELD"


def test_native_score_batch_documents_rejects_length_mismatch():
    with pytest.raises(ValueError):
        _native_score_batch_documents(
            (BoardRequest(ticker="A", strategy="STR-THRU",
                          event_date=pd.Timestamp("2026-01-15"), session="am"),
             BoardRequest(ticker="B", strategy="STR-THRU",
                          event_date=pd.Timestamp("2026-01-15"), session="am")),
            [{"stub": True}], ())


def test_native_score_batch_documents_rejects_duplicate_canonical_keys_in_records():
    """CodeRabbit round 2: two DISTINCT BoardRequests differing only by time
    of day collapse to one canonical key once _iso reduces both to the same
    calendar date -- the later row must never silently overwrite the earlier
    one's record."""
    first = BoardRequest(ticker="TEST", strategy="STR-THRU",
                         event_date=pd.Timestamp("2026-01-15 09:00"), session="am")
    second = BoardRequest(ticker="TEST", strategy="STR-THRU",
                          event_date=pd.Timestamp("2026-01-15 16:00"), session="am")
    assert first != second
    with pytest.raises(ValueError):
        _native_score_batch_documents((first, second), [{"a": 1}, {"b": 2}], ())


def test_native_score_batch_documents_rejects_duplicate_canonical_keys_in_refusals():
    """CodeRabbit round 2: the same canonical-key collision applies to the
    keyed-refusals dict, not just records -- both routes must raise."""
    first = NativeScoreBatchRowRefusal(
        BoardRequest(ticker="TEST", strategy="STR-THRU",
                     event_date=pd.Timestamp("2026-01-15 09:00"), session="am"),
        "UNSUPPORTED_STRATEGY", "first detail")
    second = NativeScoreBatchRowRefusal(
        BoardRequest(ticker="TEST", strategy="STR-THRU",
                     event_date=pd.Timestamp("2026-01-15 16:00"), session="am"),
        "UNSUPPORTED_STRATEGY", "second detail")
    assert first.key != second.key
    with pytest.raises(ValueError):
        _native_score_batch_documents((), (), (first, second))


def test_native_score_batch_documents_rejects_cross_dict_canonical_key_collision():
    """CodeRabbit round 3: two DISTINCT BoardRequests differing only by time
    of day collide onto one canonical key -- when one succeeds (a record)
    and the other fails (a refusal), neither dict's own internal duplicate
    check can see the other, so the cross-dict overlap must raise too."""
    record_key = BoardRequest(ticker="TEST", strategy="STR-THRU",
                              event_date=pd.Timestamp("2026-01-15 09:00"), session="am")
    refusal = NativeScoreBatchRowRefusal(
        BoardRequest(ticker="TEST", strategy="STR-THRU",
                     event_date=pd.Timestamp("2026-01-15 16:00"), session="am"),
        "UNSUPPORTED_STRATEGY", "some detail")
    with pytest.raises(ValueError):
        _native_score_batch_documents((record_key,), [{"a": 1}], (refusal,))
