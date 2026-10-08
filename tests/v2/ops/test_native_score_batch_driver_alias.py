"""Driver alias and derived feature names for engine/v2/ops/native_score_batch (synthetic releases only)."""
import dataclasses
import hashlib
import json

import pandas as pd
import pytest

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
    deployment,
)
from engine.v2.ops import native_score_batch
from engine.v2.ops.native_board_universe import BoardRequest
from engine.v2.ops.native_score_batch import (
    NightlyEventInputs,
    assemble_score_batch_inputs,
    run_native_score_batch_worker,
)
from engine.v2.models.frozen_state import serialize_frozen_state
from engine.v2.models.lineage import Lineage
from engine.v2.models.payoff_artifact import (
    make_payoff_line_artifact,
    serialize_payoff_artifact,
)
from engine.v2.models.residual_artifact import make_driver_residual_pool_artifact
from engine.v2.scoring import native_gate_features
from engine.v2.scoring.release_bindings import ModelNotReady, resolve_release_binding
from tests.test_v2_scoring_release_bindings import (
    _dep_root,
    _obj,
    _row,
    _write_catalog,
    _write_object,
)

_AS_OF = "2026-01-10"
_SNAPSHOT = "snap-1"
_CALENDAR_REVISION = "cal-1"
_GATE_POLICY = {"STR-THRU": {"threshold": 0.0}}

#: (role, strategy_id, feature_order) specs. GATE's order deliberately names
#: stage-derived columns (pred_abs_move/forecast_edge/analog_n) so the
#: derived-names tests can prove they are projected away.
GATE = ("gate", "STR-THRU", ("x", "sig", "pred_abs_move", "forecast_edge", "analog_n"))
SIZE = ("size", "*", ("x", "iv", "zz"))
DRIVER = ("driver", "STR-THRU", ("x", "iv"))


def _sha(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _linear_payload(feature_order) -> bytes:
    return json.dumps({
        "schema_version": "linear_estimator.v1.0",
        "feature_order": list(feature_order),
        "outputs": [{"name": "prediction", "intercept": 1.0,
                     "coefficients": [1.0] * len(feature_order)}],
    }, sort_keys=True).encode()


def _write_empty_catalog(root, release_id: str) -> None:
    """A members-empty ``phase5_release.json``: resolve_release_binding
    requires the catalog to exist even when it declares no state objects."""
    body = {"schema_version": "phase5_staged_release.v1.0", "release_id": release_id,
            "deployment_id": "d1", "members": [], "sources": {}}
    body["manifest_hash"] = content_hash(
        {k: v for k, v in body.items() if k != "manifest_hash"})
    root.mkdir(parents=True, exist_ok=True)
    (root / "phase5_release.json").write_text(json.dumps(body, sort_keys=True))


def _stage(tmp_path, specs, release_id="r1", rows=None):
    """Stage one binding per ``(role, strategy_id, feature_order)`` spec,
    promote, and return the real, resolved ScoringReleaseBinding. Every
    clock id is "entry-close". ``rows`` (when given) becomes the release's
    staged state catalog, so release-backed artifacts resolve like
    production; no rows leaves it members-empty."""
    bindings = []
    artifacts = []
    release_bindings = []
    requirements = []
    payloads = {}
    for role, strategy_id, feature_order in specs:
        payload = _linear_payload(feature_order)
        member_hash = _sha(payload)
        payloads[member_hash] = payload
        member = ArtifactMember(name="estimator", path="unused.json",
                                content_hash=member_hash)
        binding_id = f"b-{role}-{strategy_id}"
        model_id = f"m-{role}-{strategy_id}"
        bindings.append(ModelBinding(
            binding_id=binding_id, model_id=model_id, role=role,
            strategy_id=strategy_id, decision_clock_id="entry-close",
            adapter="json-linear.v1", feature_order=tuple(feature_order),
            output_names=("prediction",), members=(member,)))
        artifacts.append(ModelArtifactInventory(
            artifact_id=model_id, role=role, strategy_ids=(strategy_id,),
            compatible_clock_ids=("entry-close",), target_contract_ref="return.v1",
            ordered_features=tuple(feature_order),
            members=(ArtifactInventoryMember(
                member_id=f"{model_id}:estimator", kind="estimator",
                artifact_ref=f"artifact://{model_id}",
                content_hash=member_hash),)))
        release_bindings.append(ReleaseBinding(
            role=role, strategy_id=strategy_id, clock_id="entry-close",
            artifact_id=model_id, ordered_features=tuple(feature_order),
            required_member_kinds=("estimator",)))
        requirements.append(ReleaseRequirement(
            role=role, strategy_id=strategy_id, clock_id="entry-close"))
    release = ModelRelease(release_id=release_id, deployment_id="d1",
                           bindings=tuple(bindings))
    inventory = ModelReleaseInventory(
        release_id=release_id, deployment_id="d1", known_clock_ids=("entry-close",),
        artifacts=tuple(artifacts), bindings=tuple(release_bindings),
        requirements=tuple(requirements), artifact_manifest_ref="manifest://r",
        evidence_refs=("evidence://r",))
    dep_root = tmp_path / "deployment"
    deployment.stage_release(dep_root, release, inventory, payloads)
    deployment.mark_staging_succeeded(dep_root, release_id)
    deployment.promote(dep_root, release_id)
    if rows is None:
        _write_empty_catalog(tmp_path, release_id)
    else:
        _write_catalog(tmp_path, release_id=release_id, rows=rows)
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


def _assemble(binding, events, feature_names=(), **overrides):
    kwargs = dict(
        as_of=_AS_OF, snapshot_id=_SNAPSHOT, calendar_revision=_CALENDAR_REVISION,
        binding=binding, events=events, feature_names=feature_names,
        gate_policy=_GATE_POLICY,
    )
    kwargs.update(overrides)
    return assemble_score_batch_inputs(**kwargs)


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


def _worker_parameters(tmp_path, *, expected_ids,
                       feature_names=("signal", "pred_abs_move")) -> dict:
    return {
        "release_root": str(tmp_path), "as_of": _AS_OF, "snapshot_id": _SNAPSHOT,
        "calendar_revision": _CALENDAR_REVISION,
        "feature_names": list(feature_names), "gate_policy": _GATE_POLICY,
        "expected_ids": expected_ids,
    }


def test_dedicated_driver_binding_wins_over_alias(tmp_path):
    """The named test ARCHITECTURE.md cites for alias removal: when the
    release carries a dedicated ``driver:STR-THRU`` binding, that exact
    identity is used -- never displaced by the ``size:*`` alias key, even
    though a size binding is present in the same release."""
    binding = _stage(tmp_path, [DRIVER, SIZE, GATE])
    event = _event_inputs()
    assembled, refusals = _assemble(binding, [event])
    assert refusals == ()
    _, native_inputs = assembled[event.key]
    identity = native_inputs.features["model_identity"]
    assert "driver:STR-THRU" in identity
    assert "size:*" not in identity


def test_alias_used_when_dedicated_binding_absent_and_recorded(tmp_path):
    """No real release carries ``driver:STR-THRU`` (the inventory emits
    ``size``), so the alias key is consulted and the bundle's
    ``model_identity`` is keyed by it -- the record shows ``size:*`` and the
    size binding's own id, never a fabricated ``driver:STR-THRU`` key."""
    binding = _stage(tmp_path, [SIZE, GATE])
    event = _event_inputs()
    assembled, refusals = _assemble(binding, [event])
    assert refusals == ()
    _, native_inputs = assembled[event.key]
    identity = native_inputs.features["model_identity"]
    assert "size:*" in identity
    assert "gate:STR-THRU" in identity
    assert "driver:STR-THRU" not in identity
    assert identity["size:*"]["binding_id"] == "b-size-*"


def test_neither_binding_refuses_release_missing_role(tmp_path):
    """Neither the exact ``driver:STR-THRU`` identity nor the alias target
    staged: the row refuses, naming the exact key that was missing."""
    binding = _stage(tmp_path, [GATE])
    event = _event_inputs()
    assembled, refusals = _assemble(binding, [event])
    assert assembled == {}
    assert len(refusals) == 1
    assert refusals[0].code == "RELEASE_MISSING_ROLE"
    assert refusals[0].detail == "driver:STR-THRU"


def test_alias_is_data_driven(tmp_path, monkeypatch):
    """The alias is one data table, not a per-strategy code path: emptying
    it must make even a ``size:*``-backed release refuse, exactly as if no
    alias existed."""
    binding = _stage(tmp_path, [SIZE, GATE])
    monkeypatch.setattr(native_score_batch, "_DRIVER_ROLE_ALIAS", {})
    assembled, refusals = _assemble(binding, [_event_inputs()])
    assert assembled == {}
    assert len(refusals) == 1
    assert refusals[0].code == "RELEASE_MISSING_ROLE"


def test_derived_feature_names_are_the_union_minus_stage_owned_columns_and_deterministic(
        tmp_path):
    """Empty ``feature_names`` derives, per row, the sorted de-duplicated
    union of the driver and gate ``feature_order``s minus the stage-owned
    gate columns -- here ``{"x","iv","zz"} | {"x","sig","pred_abs_move",
    "forecast_edge","analog_n"}`` minus GATE_FORECAST/GATE_ANALOG names,
    i.e. exactly {"iv","sig","x","zz"} -- a pure function of the recorded
    identities, so the same specs staged in reversed order give the same
    names."""
    binding = _stage(tmp_path, [SIZE, GATE])
    event = _event_inputs()
    assembled, refusals = _assemble(binding, [event])
    assert refusals == ()
    _, native_inputs = assembled[event.key]
    assert sorted(native_inputs.features["missing_mask"]) == ["iv", "sig", "x", "zz"]
    again = _stage(tmp_path / "again", [GATE, SIZE])
    again_assembled, again_refusals = _assemble(again, [event])
    assert again_refusals == ()
    _, again_inputs = again_assembled[event.key]
    assert sorted(again_inputs.features["missing_mask"]) == \
        sorted(native_inputs.features["missing_mask"])


def test_derived_names_never_contain_stage_owned_groups(tmp_path):
    """The derived names are stage-owned-column-free on every axis of the
    assembled row: neither ``model_inputs`` nor ``missing_mask`` ever
    carries a GATE_FORECAST_COLUMNS/GATE_ANALOG_COLUMNS name, because
    projecting one would suppress that column's native derivation."""
    binding = _stage(tmp_path, [SIZE, GATE])
    event = _event_inputs()
    assembled, refusals = _assemble(binding, [event])
    assert refusals == ()
    _, native_inputs = assembled[event.key]
    features = native_inputs.features
    stage_owned = set(native_gate_features.GATE_FORECAST_COLUMNS
                      + native_gate_features.GATE_ANALOG_COLUMNS)
    assert not stage_owned & set(features["model_inputs"])
    assert not stage_owned & set(features["missing_mask"])


def test_explicit_feature_names_win(tmp_path):
    """A non-empty explicit ``feature_names`` is used as given -- the
    derived union is never merged in behind it."""
    binding = _stage(tmp_path, [SIZE, GATE])
    event = _event_inputs()
    assembled, refusals = _assemble(binding, [event], feature_names=("signal",))
    assert refusals == ()
    _, native_inputs = assembled[event.key]
    features = native_inputs.features
    assert set(features["missing_mask"]) == {"signal"}
    assert set(features["model_inputs"]) == {"signal"}


def test_empty_feature_order_refuses_release_missing_feature_order(tmp_path):
    """Empty ``feature_names`` over an identity with no ``feature_order``
    refuses RELEASE_MISSING_FEATURE_ORDER; the same release with explicit
    ``feature_names`` assembles -- explicit names win even then."""
    binding = _stage(tmp_path, [SIZE, GATE])
    event = _event_inputs()
    identities = dict(binding.model_identity)
    identities["gate:STR-THRU"] = dataclasses.replace(
        binding.model_identity["gate:STR-THRU"], feature_order=())
    binding = dataclasses.replace(binding, model_identity=identities)
    assembled, refusals = _assemble(binding, [event])
    assert assembled == {}
    assert len(refusals) == 1
    assert refusals[0].code == "RELEASE_MISSING_FEATURE_ORDER"
    assembled, refusals = _assemble(binding, [event], feature_names=("signal",))
    assert refusals == ()
    assert list(assembled) == [event.key]


def test_worker_without_feature_names_gets_past_release_missing_role(tmp_path):
    """The nightly worker stages ``feature_names: []`` (the derived union is
    the contract), so the alias/derived-names path must get this release
    past RELEASE_MISSING_ROLE in the real worker too. Per-row feature gaps
    (every derived name is absent from this minimal fixture's rows) are
    record flags, not worker refusals, and are out of scope here."""
    _stage(tmp_path, [SIZE, GATE])
    root = tmp_path / "work"
    root.mkdir()
    (root / "events.json").write_text(json.dumps([_event_doc()]))
    run_native_score_batch_worker(
        _worker_parameters(tmp_path, expected_ids=["evt-1"], feature_names=[]), root)
    document = json.loads((root / "refusals.json").read_text())
    assert "RELEASE_MISSING_ROLE" not in json.dumps(document)
    assert document["refusals"] == {}
    assert document["unkeyable_refusals"] == []


@pytest.mark.parametrize("bad", [0, {}, "signal", 1.5, False])
def test_worker_rejects_non_sequence_feature_names(tmp_path, bad):
    """A supplied, malformed ``feature_names`` raises -- truthiness must not
    silently turn ``0``/``{}`` into the empty (derive-everything) default."""
    _stage(tmp_path, [SIZE, GATE])
    root = tmp_path / "work"
    root.mkdir()
    (root / "events.json").write_text(json.dumps([_event_doc()]))
    params = _worker_parameters(tmp_path, expected_ids=["evt-1"])
    params["feature_names"] = bad
    with pytest.raises(ValueError, match="feature_names must be a list or tuple"):
        run_native_score_batch_worker(params, root)


@pytest.mark.parametrize("bad", [0, False, {}, "", [1], ("signal", 2)])
def test_direct_caller_malformed_feature_names_refuse_invalid_feature_names(tmp_path, bad):
    """The shape check sits at the shared assembly boundary: a direct
    ``assemble_score_batch_inputs`` caller passing a falsey or malformed
    ``feature_names`` gets the per-row INVALID_FEATURE_NAMES refusal, never a
    silent derive-everything default."""
    binding = _stage(tmp_path, [SIZE, GATE])
    event = _event_inputs()
    assembled, refusals = _assemble(binding, [event], feature_names=bad)
    assert assembled == {}
    assert [r.code for r in refusals] == ["INVALID_FEATURE_NAMES"]


@pytest.mark.parametrize("empty", [None, [], ()])
def test_direct_caller_none_or_empty_feature_names_derive(tmp_path, empty):
    binding = _stage(tmp_path, [SIZE, GATE])
    event = _event_inputs()
    assembled, refusals = _assemble(binding, [event], feature_names=empty)
    assert refusals == ()
    _, native_inputs = assembled[event.key]
    assert sorted(native_inputs.features["missing_mask"]) == ["iv", "sig", "x", "zz"]


def _staged_driver_pool(tmp_path, *, role="size", model_id="m-size-*"):
    """A verified driver residual pool written as a real staged catalog row."""
    artifact = make_driver_residual_pool_artifact(
        role=role, model_id=model_id, fold=None,
        flat_residuals=(0.1, -0.2, 0.3), buckets=None, deciles=10,
        min_pool=2, lineage=Lineage())
    path = _write_object(_dep_root(tmp_path), serialize_frozen_state(artifact))
    row = _row(f"driver_residual_pool:{role}", [_obj(path, artifact.content_hash)])
    return artifact, row


def _staged_payoff_line(tmp_path, *, alpha=0.5, cutoff="2026-01-10"):
    """A verified payoff line written as a real staged catalog row."""
    artifact = make_payoff_line_artifact(
        {"n": 2, "intercept": 0.1, "slope": 0.2, "resid_sd": 0.01, "r": 0.5,
         "residuals": [0.01, -0.01]},
        strategy="STR-THRU", driver="driver_prediction", alpha=alpha, cutoff=cutoff)
    path = _write_object(_dep_root(tmp_path), serialize_payoff_artifact(artifact))
    row = _row("payoff_line:STR-THRU", [_obj(path, artifact.content_hash)])
    return artifact, row


def test_release_backed_bundle_declares_driver_pool_and_payoff_artifact(tmp_path):
    """A promoted release carrying a verified driver residual pool and payoff
    line makes the real ``_assemble_one_event``/``build_native_score_inputs``
    path declare all four SourceBundle fields: the pool under slot ``driver``
    with its causal recipe, and the payoff artifact selected by the canonical
    ``(strategy, 0.5, cutoff)`` key with its ``before`` recipe. The declared
    identities/content are the release's own artifact objects."""
    driver, driver_row = _staged_driver_pool(tmp_path)
    payoff, payoff_row = _staged_payoff_line(tmp_path, cutoff="2026-01-08")
    binding = _stage(tmp_path, [SIZE, GATE], rows=[driver_row, payoff_row])
    event = _event_inputs(calendar_row=_calendar_row(entry_date="2026-01-08"))
    assembled, refusals = _assemble(binding, [event])
    assert refusals == ()
    _, native_inputs = assembled[event.key]
    model = native_inputs.model
    assert model["model_residual_artifacts"] == {"driver": driver}
    assert model["model_residual_artifact_recipe"] == {"driver": {
        "role": "size", "model_id": "m-size-*", "fold": None,
        "content_hash": driver.content_hash}}
    assert model["payoff_artifact"] == payoff
    assert model["payoff_recipe"] == {"before": "2026-01-08"}
    assert native_inputs.simulation == {}


def test_null_entry_date_with_pool_but_no_payoff_declares_no_cutoff(tmp_path):
    """The payoff cutoff runs only when the resolved binding actually has
    payoff artifacts: a release carrying a valid driver pool but no payoff
    member assembles a row with a null ``entry_date`` (the "no entry_date"
    case -- a key literally absent would refuse at the bundle's staged-input
    check before the release declarations run). :func:`_row_cutoff`, which
    would refuse a null entry date with ``MISSING_STAGED_INPUT``, and
    ``payoff_artifact_key`` are never evaluated -- the model block declares
    the release's pool with its recipe, nothing payoff-related, and the
    simulation mapping stays empty."""
    driver, driver_row = _staged_driver_pool(tmp_path)
    binding = _stage(tmp_path, [SIZE, GATE], rows=[driver_row])
    event = _event_inputs(calendar_row=_calendar_row(entry_date=None))
    assembled, refusals = _assemble(binding, [event])
    assert refusals == ()
    _, native_inputs = assembled[event.key]
    model = native_inputs.model
    assert model["model_residual_artifacts"] == {"driver": driver}
    assert model["model_residual_artifact_recipe"] == {"driver": {
        "role": "size", "model_id": "m-size-*", "fold": None,
        "content_hash": driver.content_hash}}
    assert "payoff_artifact" not in model
    assert "payoff_recipe" not in model
    assert native_inputs.simulation == {}


def test_release_without_artifacts_ignores_row_markers(tmp_path):
    """The release is the sole source: a release with no driver-pool or payoff
    member leaves every declaration absent even when the event's Tier-4 and
    panel rows carry residual/payoff-like marker values, so request rows can
    never populate the model block."""
    binding = _stage(tmp_path, [SIZE, GATE])
    event = _event_inputs(
        tier4_row={"pred_abs_move": 0.05,
                   "pred_abs_move_fold_start": "2026-01-01",
                   "model_residual_artifacts": {"driver": "bogus"},
                   "model_residual_artifact_recipe": {"driver": {"role": "size"}},
                   "payoff_artifact": "bogus",
                   "payoff_artifact_recipe": {"before": "2026-01-10"}},
        panel_row={"date": "2026-01-15", "signal": 1.5,
                   "model_residual_artifacts": {"driver": "bogus"},
                   "payoff_artifact": "bogus"},
    )
    assembled, refusals = _assemble(binding, [event])
    assert refusals == ()
    _, native_inputs = assembled[event.key]
    assert native_inputs.model == {}


def test_malformed_driver_pool_refuses_model_not_ready(tmp_path):
    """A malformed staged driver pool refuses through the real release
    resolver with the typed ``ModelNotReady`` naming its member, before any
    bundle is returned."""
    junk = b"not valid json"
    row = _row("driver_residual_pool:size",
               [_obj(_write_object(_dep_root(tmp_path), junk), _sha(junk))])
    with pytest.raises(ModelNotReady) as error:
        _stage(tmp_path, [SIZE, GATE], rows=[row])
    assert error.value.code == "MODEL_NOT_READY"
    assert error.value.member_id == "driver_residual_pool:size"


def test_malformed_payoff_artifact_refuses_model_not_ready(tmp_path):
    """A malformed staged payoff member refuses through the real release
    resolver with the typed ``ModelNotReady`` naming its member, before any
    bundle is returned."""
    junk = b"not valid json"
    row = _row("payoff_line:STR-THRU",
               [_obj(_write_object(_dep_root(tmp_path), junk), _sha(junk))])
    with pytest.raises(ModelNotReady) as error:
        _stage(tmp_path, [SIZE, GATE], rows=[row])
    assert error.value.code == "MODEL_NOT_READY"
    assert error.value.member_id == "payoff_line:STR-THRU"


def test_malformed_entry_date_refuses_calendar_row_invalid(tmp_path):
    """``entry_date`` is calendar date validation too: on a real release that
    carries a driver pool and a payoff line (so the payoff causal cutoff
    actually runs), a malformed non-null ``entry_date`` refuses the existing
    row-level ``CALENDAR_ROW_INVALID`` from ``_calendar_row_problem`` before
    :func:`_row_cutoff` ever sees it -- never the cutoff helper's own
    ``INVALID_DATE`` re-wrap -- and no row is assembled."""
    driver, driver_row = _staged_driver_pool(tmp_path)
    payoff, payoff_row = _staged_payoff_line(tmp_path)
    binding = _stage(tmp_path, [SIZE, GATE], rows=[driver_row, payoff_row])
    event = _event_inputs(calendar_row=_calendar_row(entry_date="not-a-date"))
    assembled, refusals = _assemble(binding, [event])
    assert assembled == {}
    assert [refusal.code for refusal in refusals] == ["CALENDAR_ROW_INVALID"]
