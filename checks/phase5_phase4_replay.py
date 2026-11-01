"""P5-6 Phase 4 replay: a Phase 4 corpus's traced pairs, scored from a staged release.

Each pair's strict trace is verified by Phase 4's own verifier
(``checks.phase4_real._verified_trace_bundle``, which builds the frozen replay
plan through ``checks.phase4_frozen_bridge.prepare_frozen_replay``). The plan's
bindings are then *rebound* to the staged release: every model member is
looked up by content hash among the release's staged objects (model bindings
and Tier-4 serving folds) and served from the staged deployment store instead
of the corpus's own copy. The trace's binding contract (release id, binding
ids, role, feature order, output names, adapter) is kept as captured: those
ids enter the stage inputs, and legacy capture names outputs per context
(``driver_prediction``, ``pred_abs_move``, ``gate_score``...), so only the
bytes change hands. The rebound plan runs through
``application.score_frozen`` and the runtime stage receipts are compared with
the captured ones by ``checks.phase4_real._verify_runtime_execution``.

Dispositions per pair (value-free: ids, codes, counts, field paths):

* ``replayed`` -- every stage receipt and identity matches the capture.
* ``mismatch`` (``P5_PHASE4_MISMATCH``) -- scored from the staged bytes, a
  stage receipt or identity differs from the capture.
* ``member_absent`` (``P5_PHASE4_MEMBER_ABSENT``) -- the trace binds a model
  whose bytes are not a staged member of this release, or its entry-rule gate
  pins a trailing ``pnl_sim`` cutoff that is not a staged
  ``trailing_pnl_cutoff`` object (the cutoff document travels inside the
  hash-bound trace, so the release serves it by identity); nothing is scored.
* ``unverified`` (``P5_PHASE4_UNVERIFIED``) -- the trace fails Phase 4's own
  verification, so there is nothing trustworthy to replay.
* ``error`` (``P5_PHASE4_ERROR``) -- scoring raised.
* ``untraced`` / ``not_frozen`` -- no strict trace, or a trace with no frozen
  model binding: the pair never reads the release. Counted, not a finding:
  trace coverage is the Phase 4 gate's subject, not this one (my judgement
  call). If no pair replays at all the subject fails ``P5_PHASE4_EMPTY``.
"""
from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from engine.v2.models.release_layout import deployment_root, object_relpath

PHASE4_UNVERIFIED = "P5_PHASE4_UNVERIFIED"
PHASE4_MEMBER_ABSENT = "P5_PHASE4_MEMBER_ABSENT"
PHASE4_MISMATCH = "P5_PHASE4_MISMATCH"
PHASE4_ERROR = "P5_PHASE4_ERROR"
PHASE4_EMPTY = "P5_PHASE4_EMPTY"

REPLAY_CODES = (PHASE4_UNVERIFIED, PHASE4_MEMBER_ABSENT, PHASE4_MISMATCH,
                PHASE4_ERROR, PHASE4_EMPTY)
DISPOSITIONS = ("replayed", "mismatch", "member_absent", "unverified", "error",
                "untraced", "not_frozen")

REPLAY_VALIDATION_VERSION = 3
MAX_CHECKPOINT_BYTES = 64 * 1024 * 1024


def _checkpoint_scope(corpus_hash: str | None, manifest: Mapping[str, Any]) -> str:
    """Return the content hash of the canonical checkpoint scope: corpus, full
    staged manifest and validation version."""
    from engine.v2.foundation import content_hash

    scope = {"corpus_hash": corpus_hash, "manifest": manifest,
             "validation_version": REPLAY_VALIDATION_VERSION}
    return content_hash(scope)


#: Staged state members a replayed pair may read by content hash: the Tier-4
#: serving folds (frozen bindings, including the chooser's producer folds),
#: the entry-rule trailing cutoff, and the chooser's frozen state.
_STAGED_STATE_PREFIXES = ("tier4_folds:", "trailing_pnl_cutoff", "chooser_analog_pool",
                          "admissible_table:")


def staged_model_objects(model_release, manifest: Mapping[str, Any]) -> dict[str, str]:
    """``content_hash -> member id`` for every object a frozen binding, or an
    entry-rule gate's trailing cutoff, may read."""
    from engine.v2.foundation import from_document
    from engine.v2.models.serving_folds import ChooserServingFoldDescriptor

    staged: dict[str, str] = {}
    for binding in model_release.bindings:
        for member in binding.members:
            staged.setdefault(member.content_hash,
                              f"model:{binding.role}:{binding.strategy_id}")
    for row in manifest.get("members", ()):
        if not isinstance(row, Mapping):
            continue
        member_id = row.get("member_id", "")
        if not isinstance(member_id, str) or not member_id.startswith(_STAGED_STATE_PREFIXES):
            continue
        for obj in row.get("objects", ()):
            if not isinstance(obj, Mapping):
                continue
            content_hash = obj.get("content_hash")
            if not isinstance(content_hash, str):
                continue
            staged.setdefault(content_hash, member_id)
            doc = obj.get("serving_fold")
            if isinstance(doc, Mapping):
                try:
                    descriptor = from_document(ChooserServingFoldDescriptor, doc)
                except Exception:  # noqa: BLE001 -- malformed descriptor never binds
                    continue
                staged[f"__chooser_descriptor__:{descriptor.descriptor_hash}"] = (
                    f"{descriptor.producer_output}\x1f{descriptor.training_cutoff}"
                    f"\x1f{descriptor.fold_start}")
    return staged


def rebind_to_release(plan, release_root: Path, staged: Mapping[str, str]):
    """``(rebound plan, absent [(binding_id, hash)], used member ids)``.

    Only member paths change: each points at the staged store's
    content-addressed object, and ``FrozenInference`` is rooted at the staged
    deployment store, so the corpus's own artifact copies are never read.
    """
    from checks.phase4_frozen_bridge import FrozenReplayPlan
    from engine.v2.models.loader import FrozenInference

    release, absent, used = _rebind_release(plan.release, staged)
    rebound = FrozenReplayPlan(
        inference=FrozenInference(deployment_root(release_root)),
        release=release,
        requests=plan.requests,
        receipt=plan.receipt,
    )
    return rebound, absent, sorted(used)


def _rebind_release(release, staged: Mapping[str, str]):
    """``(release with every member at its staged object, absent, used)``."""
    absent, used, bindings = [], set(), []
    for binding in release.bindings:
        members = []
        for member in binding.members:
            if member.content_hash not in staged:
                absent.append((binding.binding_id, member.content_hash))
            else:
                used.add(staged[member.content_hash])
            members.append(replace(member, path=object_relpath(member.content_hash)))
        bindings.append(replace(binding, members=tuple(members)))
    return replace(release, bindings=tuple(bindings)), absent, used


def _staged_state(release_root: Path, staged: Mapping[str, str], prefix: str, artifact):
    """``(the staged state equal to artifact, member id)`` or ``(None, None)``.

    Frozen state is matched by its own content hash after loading the staged
    object through the typed loader, so the replay serves the release's bytes.
    """
    from engine.v2.models.frozen_state import FrozenStateLoader, FrozenStateRef

    loader = FrozenStateLoader(deployment_root(release_root))
    for object_hash, member_id in staged.items():
        if not member_id.startswith(prefix):
            continue
        state = loader.load(FrozenStateRef(path=object_relpath(object_hash),
                                           content_hash=object_hash))
        if getattr(state, "content_hash", None) == artifact.content_hash:
            return state, member_id
    return None, None


def _chooser_replay_dates(request: Any, legacy_event_date: Any = None):
    from datetime import date, datetime

    def parse(value):
        if not isinstance(value, str):
            return None
        try:
            parsed = date.fromisoformat(value)
        except ValueError:
            try:
                parsed_datetime = datetime.fromisoformat(value)
            except ValueError:
                return None
            if parsed_datetime.tzinfo is not None or parsed_datetime.isoformat() != value:
                return None
            return parsed_datetime.date()
        if parsed.isoformat() != value:
            return None
        return parsed

    event_day = parse(legacy_event_date)
    decision_day = parse(getattr(request, "requested_decision_at", None))
    if event_day is None or decision_day is None:
        return None
    return event_day, decision_day


def _has_causal_chooser_fold_provenance(recipe: Any, request: Any,
                                        staged: Mapping[str, str],
                                        legacy_event_date: Any = None) -> bool:
    from datetime import date

    if not isinstance(recipe, Mapping):
        return False
    provenance = recipe.get("fold_provenance")
    expected = {"pred_im_t1_d14", "pred_runup_abs_move_d14"}
    if not isinstance(provenance, Mapping) or set(provenance) != expected:
        return False
    artifact_refs = recipe.get("artifact_refs")
    if (not isinstance(artifact_refs, (list, tuple))
            or any(not isinstance(ref, str) for ref in artifact_refs)):
        return False
    replay_dates = _chooser_replay_dates(request, legacy_event_date)
    if replay_dates is None:
        return False
    event_day, decision_day = replay_dates
    expected_fold_start = min(event_day.replace(day=1), decision_day.replace(day=1)).isoformat()
    for producer_output, item in provenance.items():
        if not isinstance(item, Mapping) or set(item) != {
                "training_cutoff", "fold_start", "descriptor_hash"}:
            return False
        cutoff_text, fold_text = item.get("training_cutoff"), item.get("fold_start")
        descriptor_hash = item.get("descriptor_hash")
        if (not isinstance(cutoff_text, str) or not isinstance(fold_text, str)
                or not isinstance(descriptor_hash, str) or descriptor_hash not in artifact_refs):
            return False
        if staged.get(f"__chooser_descriptor__:{descriptor_hash}") != (
                f"{producer_output}\x1f{cutoff_text}\x1f{fold_text}"):
            return False
        try:
            cutoff_day, fold_day = date.fromisoformat(cutoff_text), date.fromisoformat(fold_text)
        except ValueError:
            return False
        if (cutoff_day.isoformat() != cutoff_text or fold_day.isoformat() != fold_text
                or fold_day.day != 1 or fold_text != expected_fold_start
                or cutoff_day >= decision_day or cutoff_day >= fold_day):
            return False
    return True


def _staged_chooser_pools(chooser, request, release_root: Path, manifest,
                          legacy_event_date: Any = None):
    if not isinstance(manifest, Mapping):
        return None
    replay_dates = _chooser_replay_dates(request, legacy_event_date)
    if replay_dates is None:
        return None
    event_day, decision_day = replay_dates
    expected_fold_start = min(event_day.replace(day=1), decision_day.replace(day=1)).isoformat()
    from numpy import array_equal, asarray
    from engine.v2.foundation import from_document
    from engine.v2.models import deployment
    from engine.v2.models.serving_fold_loader import (
        load_chooser_serving_fold, load_serving_fold, validate_serving_fold_descriptor,
        validate_serving_fold_object,
    )
    from engine.v2.models.serving_folds import (
        ChooserServingFoldDescriptor, ServingFoldDescriptor,
    )
    from engine.v2.models.loader import FrozenInference

    members = manifest.get("members")
    if not isinstance(members, list):
        return None
    docs = {}
    size_docs = []
    seen_members = set()
    chooser_months = set()
    size_months = set()
    try:
        for row in members:
            if not isinstance(row, Mapping):
                return None
            member_id = row.get("member_id", "")
            if not isinstance(member_id, str):
                return None
            if member_id not in {"tier4_folds:implied_t1", "tier4_folds:runup_move", "tier4_folds:size"}:
                continue
            if member_id in seen_members or row.get("status") != "STAGED":
                return None
            seen_members.add(member_id)
            objects = row.get("objects", ())
            if not isinstance(objects, list):
                return None
            if member_id in {"tier4_folds:implied_t1", "tier4_folds:runup_move"}:
                for obj in objects:
                    doc = obj.get("serving_fold") if isinstance(obj, Mapping) else None
                    if not isinstance(doc, Mapping) or doc.get("fold_start") != expected_fold_start:
                        continue
                    descriptor = from_document(ChooserServingFoldDescriptor, doc)
                    chooser_month = (member_id, descriptor.fold_start)
                    if chooser_month in chooser_months or descriptor.descriptor_hash in docs:
                        return None
                    chooser_months.add(chooser_month)
                    docs[descriptor.descriptor_hash] = descriptor
            else:
                for obj in objects:
                    if not isinstance(obj, Mapping):
                        return None
                    doc = obj.get("serving_fold")
                    if not isinstance(doc, Mapping):
                        return None
                    descriptor = from_document(ServingFoldDescriptor, doc)
                    validate_serving_fold_descriptor(descriptor)
                    if descriptor.fold_start in size_months:
                        return None
                    size_months.add(descriptor.fold_start)
                    if type(obj.get("bytes")) is not int:
                        return None
                    if (obj.get("path") != descriptor.estimator.path
                            or obj.get("content_hash") != descriptor.estimator.content_hash):
                        return None
                    validate_serving_fold_object(obj, descriptor)
                    size_docs.append((obj, descriptor))
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError):
        return None

    recipe = getattr(chooser, "recipe", None)
    artifact_refs = recipe.get("artifact_refs") if isinstance(recipe, Mapping) else None
    if (not isinstance(artifact_refs, (list, tuple))
            or any(not isinstance(ref, str) for ref in artifact_refs)):
        return None
    size_matches = [(obj, descriptor) for obj, descriptor in size_docs
                    if descriptor.estimator.content_hash in artifact_refs
                    and descriptor.fold_start == expected_fold_start]
    if len(size_matches) != 1:
        return None
    size_obj, size_descriptor = size_matches[0]
    try:
        validate_serving_fold_object(size_obj, size_descriptor)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError):
        return None

    recipe_provenance = recipe.get("fold_provenance", {})
    fold_pools = getattr(chooser, "fold_pools", None)
    if not isinstance(fold_pools, Mapping):
        return None
    catalog_hash = manifest.get("manifest_hash")
    if not isinstance(catalog_hash, str):
        return None
    root = deployment_root(release_root)
    verified = dict(fold_pools)
    try:
        parent = deployment.read_manifest(root, size_descriptor.parent_release_id)
        if parent is None:
            return None
        size_ref = load_serving_fold(
            FrozenInference(root), size_descriptor,
            verified_parent_manifest=parent, catalog_hash=catalog_hash,
            expected_bytes=size_obj["bytes"])
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError):
        return None
    expected_size = {
        "predictions": size_ref.pool_pred, "residuals": size_ref.pool_res,
        "interval_floor": size_descriptor.policy.interval_floor,
    }
    actual_size = fold_pools.get("pred_abs_move")
    if (not isinstance(actual_size, Mapping) or set(actual_size) != set(expected_size)
            or actual_size.get("interval_floor") != expected_size["interval_floor"]):
        return None
    try:
        if (not array_equal(asarray(actual_size["predictions"]), asarray(expected_size["predictions"]))
                or not array_equal(asarray(actual_size["residuals"]), asarray(expected_size["residuals"]))):
            return None
    except (TypeError, ValueError, KeyError):
        return None
    verified["pred_abs_move"] = expected_size

    for output in ("pred_im_t1_d14", "pred_runup_abs_move_d14"):
        item = recipe_provenance.get(output) if isinstance(recipe_provenance, Mapping) else None
        descriptor = docs.get(item.get("descriptor_hash")) if isinstance(item, Mapping) else None
        if (descriptor is None or descriptor.producer_output != output
                or descriptor.training_cutoff != item.get("training_cutoff")
                or descriptor.fold_start != item.get("fold_start")):
            return None
        try:
            parent = deployment.read_manifest(root, descriptor.parent_release_id)
            if parent is None:
                return None
            ref = load_chooser_serving_fold(
                FrozenInference(root), descriptor,
                verified_parent_manifest=parent, verified_catalog=dict(manifest))
        except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError):
            return None
        producer_refs = recipe.get("producers")
        producer = producer_refs.get(output) if isinstance(producer_refs, Mapping) else None
        binding_id = producer.get("binding_id") if isinstance(producer, Mapping) else None
        if not isinstance(binding_id, str) or not binding_id or producer.get("output") != output:
            return None
        trace_bindings = [binding for binding in getattr(chooser.release, "bindings", ())
                          if getattr(binding, "binding_id", None) == binding_id]
        fold_bindings = getattr(ref.release, "bindings", ())
        if len(trace_bindings) != 1 or len(fold_bindings) != 1:
            return None
        alias, verified_binding = trace_bindings[0], fold_bindings[0]
        semantic_fields = ("role", "model_id", "strategy_id", "feature_order",
                           "decision_clock_id", "output_names")
        if any(getattr(alias, field, None) != getattr(verified_binding, field, None)
               for field in semantic_fields):
            return None
        alias_members = getattr(alias, "members", ())
        verified_members = getattr(verified_binding, "members", ())
        alias_hashes = tuple(getattr(member, "content_hash", None) for member in alias_members)
        verified_hashes = tuple(getattr(member, "content_hash", None) for member in verified_members)
        if (not alias_hashes or any(not isinstance(value, str) or not value for value in alias_hashes)
                or alias_hashes != verified_hashes):
            return None
        expected = {"predictions": ref.pool_pred, "residuals": ref.pool_res,
                    "interval_floor": descriptor.policy.interval_floor}
        actual = fold_pools.get(output)
        if (not isinstance(actual, Mapping) or set(actual) != set(expected)
                or actual.get("interval_floor") != expected["interval_floor"]):
            return None
        try:
            if (not array_equal(asarray(actual["predictions"]), asarray(expected["predictions"]))
                    or not array_equal(asarray(actual["residuals"]), asarray(expected["residuals"]))):
                return None
        except (TypeError, ValueError, KeyError):
            return None
        verified[output] = expected
    return verified


def rebind_chooser(chooser, request, release_root: Path, staged: Mapping[str, str],
                   manifest=None, legacy_event_date: Any = None):
    """``(chooser block served from the staged release, absent, used)``.

    The frozen chooser's champion and Tier-4 producer folds are rebound by
    content hash exactly like the scoring bindings; its k-NN analog pool and
    n_admissible table must be staged ``chooser_analog_pool`` /
    ``admissible_table:`` objects with the same content hash, and are served
    from the staged store. The recipe remains trace-bound; producer descriptors
    and pools come from the staged catalog and parent manifests. Replay refuses
    unless trace arrays and interval floors match those loaded values exactly.
    """
    from engine.v2.models.loader import FrozenInference
    from engine.v2.scoring.chooser_inputs import frozen_chooser_block

    if not _has_causal_chooser_fold_provenance(
            chooser.recipe, request, staged, legacy_event_date):
        return None, [("chooser:fold_provenance", "unverifiable")], set()
    fold_pools = _staged_chooser_pools(
        chooser, request, release_root, manifest, legacy_event_date)
    if fold_pools is None:
        return None, [("chooser:fold_pools", "unverifiable")], set()
    release, absent, used = _rebind_release(chooser.release, staged)
    served = {}
    for name, prefix in (("analog_pool", "chooser_analog_pool"),
                         ("admissible_table", "admissible_table:")):
        declared = getattr(chooser, name)
        served[name], member_id = ((None, None) if declared is None else
                                   _staged_state(release_root, staged, prefix, declared))
        if declared is not None and served[name] is None:
            absent.append((f"chooser:{name}", declared.content_hash))
        elif member_id is not None:
            used.add(member_id)
    if absent:
        return None, absent, used
    block = frozen_chooser_block(
        strategy=request.strategy_version, recipe=chooser.recipe,
        fold_pools=fold_pools, analog_pool=served["analog_pool"],
        admissible_table=served["admissible_table"],
        inference=FrozenInference(deployment_root(release_root)), release=release)
    return block, absent, used


def _replay_pair(pair, corpus_root: Path, release_root: Path,
                 staged: Mapping[str, str], manifest=None) -> dict:
    from checks import phase4_real
    from engine.v2.scoring import application

    payload = pair.get("payload") or {}
    if not payload.get("input_trace"):
        return {"disposition": "untraced"}
    if payload.get("record_kind") == "dyn_sv_choice":
        return _replay_chooser_pair(pair, corpus_root, release_root, staged, manifest)
    try:
        verified = phase4_real._verified_trace_bundle(pair, corpus_root)
    except Exception as exc:  # noqa: BLE001 -- any refusal means "not verifiable"
        return {"disposition": "unverified", "code": PHASE4_UNVERIFIED,
                "detail": f"{type(exc).__name__}: {exc}"}
    if verified["frozen_replay"] is None and verified.get("frozen_chooser") is None:
        return {"disposition": "not_frozen"}
    try:
        rebound, inputs, absent, used = _rebound_inputs(verified, release_root, staged, manifest)
    except Exception as exc:  # noqa: BLE001 -- a staged chooser that cannot be built
        return {"disposition": "error", "code": PHASE4_ERROR,
                "detail": f"rebind: {type(exc).__name__}", "members": []}
    if absent:
        return {"disposition": "member_absent", "code": PHASE4_MEMBER_ABSENT,
                "detail": ",".join(f"{b}:{h}" for b, h in absent), "members": used}
    try:
        native = (application.score_one(verified["request"], inputs) if rebound is None
                  else application.score_frozen(
                      verified["request"], rebound.inference, rebound.release,
                      rebound.requests, {"_native_inputs": inputs}))
    except Exception as exc:  # noqa: BLE001
        return {"disposition": "error", "code": PHASE4_ERROR,
                "detail": type(exc).__name__, "members": used}
    try:
        receipts, _identities = phase4_real._verify_runtime_execution(verified, native)
    except phase4_real._TraceError as exc:
        return {"disposition": "mismatch", "code": PHASE4_MISMATCH, "detail": str(exc),
                "members": used}
    return {"disposition": "replayed", "members": used,
            "stages": len(verified["captured_receipts"]),
            "runtime_stages": len(receipts)}


def _rebound_inputs(verified, release_root: Path, staged: Mapping[str, str], manifest=None):
    """``(rebound frozen plan or None, native inputs, absent, used)``: every
    model, fold, cutoff and chooser state the pair reads, from the release."""
    plan, chooser = verified["frozen_replay"], verified.get("frozen_chooser")
    inputs, absent, used, rebound = verified["inputs"], [], set(), None
    if plan is not None:
        rebound, absent, plan_used = rebind_to_release(plan, release_root, staged)
        used.update(plan_used)
    if chooser is not None:
        block, chooser_absent, chooser_used = rebind_chooser(
            chooser, verified["request"], release_root, staged, manifest,
            legacy_event_date=verified.get("legacy_event_date"))
        absent.extend(chooser_absent)
        used.update(chooser_used)
        if block is not None:
            inputs = replace(inputs, chooser=block)
    cutoff = _entry_rule_cutoff(inputs)
    if cutoff is not None:
        if cutoff in staged:
            used.add(staged[cutoff])
        else:
            absent.append(("gate:entry_rule", cutoff))
    return rebound, inputs, absent, sorted(used)


def _entry_rule_cutoff(inputs) -> str | None:
    """The content hash of the trailing cutoff an entry-rule gate pins.

    The cutoff document travels inside the hash-bound trace; the release
    serves it by identity: the pair replays only if the staged
    ``trailing_pnl_cutoff`` member holds the same bytes (same content hash).
    """
    gate = inputs.gate
    if gate.get("mode") != "entry_rule":
        return None
    key = gate.get("trailing_cutoff_key") or {}
    return key.get("content_hash") or "unpinned"


def _replay_chooser_pair(pair, corpus_root: Path, release_root: Path,
                         staged: Mapping[str, str], manifest=None) -> dict:
    """A ``dyn_sv_choice`` pair replays when every ranked member replays.

    Each member is its own strict trace (``phase4_real._chooser_members``);
    the first member that does not replay decides the disposition.
    """
    from checks import phase4_real

    try:
        members = phase4_real._chooser_members(pair)
    except Exception as exc:  # noqa: BLE001 -- any refusal means "not verifiable"
        return {"disposition": "unverified", "code": PHASE4_UNVERIFIED,
                "detail": f"{type(exc).__name__}: {exc}"}
    used, stages, runtime = set(), 0, 0
    for index, (member_pair, _record) in enumerate(members):
        row = _replay_pair(member_pair, corpus_root, release_root, staged, manifest)
        used.update(row.get("members", ()))
        if row["disposition"] != "replayed":
            detail = row.get("detail")
            return {**row, "members": sorted(used),
                    **({"detail": f"member {index}: {detail}"} if detail else {})}
        stages += row["stages"]
        runtime += row["runtime_stages"]
    return {"disposition": "replayed", "members": sorted(used),
            "stages": stages, "runtime_stages": runtime}


def _log(tag: str, message: str, started: float) -> None:
    """Same ``[tag HH:MM:SS] message (+elapsed)`` idiom as
    ``checks.phase5_acceptance._progress`` -- this module's own corpus load
    and per-pair loop are the phase that actually dominates a real run (a
    single ~120MB pair measured at ~111s, a ~7MB pair at ~4s; the corpus can
    hold pairs up to 750MB), so each pair gets its own line rather than one
    line for the whole phase."""
    print(f"[{tag} {time.strftime('%H:%M:%S')}] {message} (+{time.perf_counter()-started:.1f}s)",
          flush=True)


def _checkpoint_row_hash(row: Mapping[str, Any], scope: str) -> str:
    """Hash a checkpoint row, bound to the replay scope, for corruption detection."""
    from engine.v2.foundation import content_hash

    payload = {key: value for key, value in row.items()
               if key != "checkpoint_row_hash"}
    return content_hash({"checkpoint_scope": scope, "row": payload})


def _load_checkpoint(checkpoint_path: Path, scope: str | None) -> dict[str, dict]:
    """Rows already replayed by an earlier, killed run of the SAME scope.

    Keyed by ``fixture_id``. Each row's ``checkpoint_row_hash`` is bound to the
    replay scope so a modified row is rejected; a file larger than
    ``MAX_CHECKPOINT_BYTES`` or whose bytes exceed the read cap is discarded
    without being parsed. A checkpoint file from a different scope (any row's
    ``checkpoint_scope`` disagrees, a scoped hash mismatches, or the file is
    unreadable/corrupt) is discarded, never silently reused -- resuming against
    the wrong corpus, manifest or validation version would be a correctness
    defect, not a convenience."""
    if not checkpoint_path.is_file():
        return {}
    import json

    try:
        if checkpoint_path.stat().st_size > MAX_CHECKPOINT_BYTES:
            return {}
    except OSError:
        return {}

    done: dict[str, dict] = {}
    consumed = 0
    try:
        with checkpoint_path.open("rb") as fh:
            while True:
                line = fh.readline(MAX_CHECKPOINT_BYTES - consumed + 1)
                if not line:
                    break
                consumed += len(line)
                if consumed > MAX_CHECKPOINT_BYTES:
                    return {}
                text = line.decode("utf-8").strip()
                if not text:
                    continue
                row = json.loads(text)
                if not isinstance(row, dict):
                    return {}
                if row.get("checkpoint_scope") != scope:
                    return {}
                fixture_id = row.get("fixture_id")
                if not isinstance(fixture_id, str) or not fixture_id:
                    return {}
                if row.get("disposition") not in DISPOSITIONS:
                    return {}
                if row.get("checkpoint_row_hash") != _checkpoint_row_hash(row, scope):
                    return {}
                done[fixture_id] = row
    except (OSError, ValueError, UnicodeDecodeError, TypeError):
        return {}
    return done


def replay_corpus(corpus: Path, release_root: Path, model_release,
                  manifest: Mapping[str, Any],
                  checkpoint_path: Path | None = None,
                  ) -> tuple[dict, list[tuple[str, str, str]]]:
    """Replay every pair; return ``(summary, findings as (code, subject, detail))``.

    ``checkpoint_path``, if given, makes the pair loop resumable: each pair's
    row is appended (one JSON line, flushed immediately -- survives SIGKILL)
    as soon as it is computed, and a prior file for the SAME scope is read
    back at the top so a killed or capped run picks up where it left off
    instead of re-paying every already-replayed pair. Added because a single
    pair has been measured at ~111s and the corpus can hold pairs up to
    ~750MB -- losing all 20 pairs' work to one OOM kill near the end was the
    concrete risk this closes. Not wired for the pre-phase4 build_evidence
    phases: those are comparatively cheap (~10s of seconds total, measured)
    and none of them is naturally per-item."""
    from checks.tier0_corpus import load, resolve_corpus

    started = time.perf_counter()
    _log("p5-phase4", f"loading corpus {corpus}", started)
    loaded = load(resolve_corpus(Path(corpus)))
    total = len(loaded.ordered_ids)
    corpus_hash = loaded.index.get("corpus_hash")
    _log("p5-phase4", f"corpus loaded: {total} pairs", started)
    staged = staged_model_objects(model_release, manifest)

    done: dict[str, dict] = {}
    checkpoint_fh = None
    checkpoint_scope = _checkpoint_scope(corpus_hash, manifest)
    if checkpoint_path is not None:
        done = _load_checkpoint(checkpoint_path, checkpoint_scope)
        if done:
            _log("p5-phase4",
                 f"checkpoint: resuming, {len(done)}/{total} pairs already replayed", started)
        checkpoint_fh = checkpoint_path.open("a", buffering=1)  # line-buffered: survives SIGKILL

    import json as _json

    rows, findings = [], []
    counts = {name: 0 for name in DISPOSITIONS}
    try:
        for index, fixture_id in enumerate(loaded.ordered_ids):
            cached = done.get(fixture_id)
            if cached is not None:
                row = {k: v for k, v in cached.items()
                       if k not in ("corpus_hash", "checkpoint_scope",
                                    "checkpoint_row_hash")}
                _log("p5-phase4",
                     f"pair {index + 1}/{total} {fixture_id}: {row['disposition']} "
                     "(from checkpoint)", started)
            else:
                pair_started = time.perf_counter()
                replay_row = _replay_pair(loaded.pairs[fixture_id], loaded.root, release_root,
                                          staged, manifest)
                pair_elapsed = time.perf_counter() - pair_started
                row = {"fixture_id": fixture_id, **replay_row}
                _log("p5-phase4",
                     f"pair {index + 1}/{total} {fixture_id}: {row['disposition']} "
                     f"(pair +{pair_elapsed:.1f}s)", started)
                if checkpoint_fh is not None:
                    checkpoint_row = {
                        **row,
                        "corpus_hash": corpus_hash,
                        "checkpoint_scope": checkpoint_scope,
                    }
                    checkpoint_row["checkpoint_row_hash"] = _checkpoint_row_hash(
                        checkpoint_row, checkpoint_scope)
                    checkpoint_fh.write(_json.dumps(checkpoint_row) + "\n")
            counts[row["disposition"]] += 1
            rows.append(row)
            if "code" in row:
                findings.append((row["code"], f"phase4:{fixture_id}", row["detail"]))
    finally:
        if checkpoint_fh is not None:
            checkpoint_fh.close()
    if not counts["replayed"] and not findings:
        findings.append((PHASE4_EMPTY, "phase4",
                         f"0/{len(rows)} pairs replayed against the staged release"))
    used = sorted({m for row in rows for m in row.get("members", ())})
    summary = {"status": "FAIL" if findings else "PASS", "pairs": len(rows),
               "dispositions": counts, "members_exercised": used,
               "corpus_hash": corpus_hash, "rows": rows}
    _log("p5-phase4", f"done: {counts}", started)
    return summary, findings


__all__ = ["DISPOSITIONS", "PHASE4_EMPTY", "PHASE4_ERROR", "PHASE4_MEMBER_ABSENT",
           "PHASE4_MISMATCH", "PHASE4_UNVERIFIED", "REPLAY_CODES", "rebind_chooser",
           "rebind_to_release", "replay_corpus", "staged_model_objects"]
