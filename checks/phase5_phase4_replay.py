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

from checks.phase5_release import deployment_root, object_relpath

PHASE4_UNVERIFIED = "P5_PHASE4_UNVERIFIED"
PHASE4_MEMBER_ABSENT = "P5_PHASE4_MEMBER_ABSENT"
PHASE4_MISMATCH = "P5_PHASE4_MISMATCH"
PHASE4_ERROR = "P5_PHASE4_ERROR"
PHASE4_EMPTY = "P5_PHASE4_EMPTY"

REPLAY_CODES = (PHASE4_UNVERIFIED, PHASE4_MEMBER_ABSENT, PHASE4_MISMATCH,
                PHASE4_ERROR, PHASE4_EMPTY)
DISPOSITIONS = ("replayed", "mismatch", "member_absent", "unverified", "error",
                "untraced", "not_frozen")


#: Staged state members a replayed pair may read by content hash: the Tier-4
#: serving folds (frozen bindings, including the chooser's producer folds),
#: the entry-rule trailing cutoff, and the chooser's frozen state.
_STAGED_STATE_PREFIXES = ("tier4_folds:", "trailing_pnl_cutoff", "chooser_analog_pool",
                          "admissible_table:")


def staged_model_objects(model_release, manifest: Mapping[str, Any]) -> dict[str, str]:
    """``content_hash -> member id`` for every object a frozen binding, or an
    entry-rule gate's trailing cutoff, may read."""
    staged: dict[str, str] = {}
    for binding in model_release.bindings:
        for member in binding.members:
            staged.setdefault(member.content_hash,
                              f"model:{binding.role}:{binding.strategy_id}")
    for row in manifest.get("members", ()):
        if str(row.get("member_id", "")).startswith(_STAGED_STATE_PREFIXES):
            for obj in row.get("objects", ()):
                staged.setdefault(obj["content_hash"], row["member_id"])
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


def rebind_chooser(chooser, request, release_root: Path, staged: Mapping[str, str]):
    """``(chooser block served from the staged release, absent, used)``.

    The frozen chooser's champion and Tier-4 producer folds are rebound by
    content hash exactly like the scoring bindings; its k-NN analog pool and
    n_admissible table must be staged ``chooser_analog_pool`` /
    ``admissible_table:`` objects with the same content hash, and are served
    from the staged store. The recipe and the fold pools are the trace's own
    declaration (hash-bound by the trace; the fold pools' hashes enter the
    chooser stage receipt, so a staged fold with other pools would mismatch).
    """
    from engine.v2.models.loader import FrozenInference
    from engine.v2.scoring.chooser_inputs import frozen_chooser_block

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
        fold_pools=chooser.fold_pools, analog_pool=served["analog_pool"],
        admissible_table=served["admissible_table"],
        inference=FrozenInference(deployment_root(release_root)), release=release)
    return block, absent, used


def _replay_pair(pair, corpus_root: Path, release_root: Path,
                 staged: Mapping[str, str]) -> dict:
    from checks import phase4_real
    from engine.v2.scoring import application

    payload = pair.get("payload") or {}
    if not payload.get("input_trace"):
        return {"disposition": "untraced"}
    if payload.get("record_kind") == "dyn_sv_choice":
        return _replay_chooser_pair(pair, corpus_root, release_root, staged)
    try:
        verified = phase4_real._verified_trace_bundle(pair, corpus_root)
    except Exception as exc:  # noqa: BLE001 -- any refusal means "not verifiable"
        return {"disposition": "unverified", "code": PHASE4_UNVERIFIED,
                "detail": f"{type(exc).__name__}: {exc}"}
    if verified["frozen_replay"] is None and verified.get("frozen_chooser") is None:
        return {"disposition": "not_frozen"}
    try:
        rebound, inputs, absent, used = _rebound_inputs(verified, release_root, staged)
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


def _rebound_inputs(verified, release_root: Path, staged: Mapping[str, str]):
    """``(rebound frozen plan or None, native inputs, absent, used)``: every
    model, fold, cutoff and chooser state the pair reads, from the release."""
    plan, chooser = verified["frozen_replay"], verified.get("frozen_chooser")
    inputs, absent, used, rebound = verified["inputs"], [], set(), None
    if plan is not None:
        rebound, absent, plan_used = rebind_to_release(plan, release_root, staged)
        used.update(plan_used)
    if chooser is not None:
        block, chooser_absent, chooser_used = rebind_chooser(
            chooser, verified["request"], release_root, staged)
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
                         staged: Mapping[str, str]) -> dict:
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
        row = _replay_pair(member_pair, corpus_root, release_root, staged)
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


def _load_checkpoint(checkpoint_path: Path, corpus_hash: str | None) -> dict[str, dict]:
    """Rows already replayed by an earlier, killed run of the SAME corpus.

    Keyed by ``fixture_id``. A checkpoint file from a different corpus (its
    first line's ``corpus_hash`` disagrees, or the file is unreadable/corrupt)
    is discarded, never silently reused -- resuming against the wrong corpus
    would be a correctness defect, not a convenience."""
    if not checkpoint_path.is_file():
        return {}
    import json

    done: dict[str, dict] = {}
    try:
        with checkpoint_path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("corpus_hash") != corpus_hash:
                    return {}
                done[row["fixture_id"]] = row
    except (OSError, ValueError):
        return {}
    return done


def replay_corpus(corpus: Path, release_root: Path, model_release,
                  manifest: Mapping[str, Any],
                  checkpoint_path: Path | None = None,
                  ) -> tuple[dict, list[tuple[str, str, str]]]:
    """Replay every pair; return ``(summary, findings as (code, subject, detail))``.

    ``checkpoint_path``, if given, makes the pair loop resumable: each pair's
    row is appended (one JSON line, flushed immediately -- survives SIGKILL)
    as soon as it is computed, and a prior file for the SAME corpus is read
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
    if checkpoint_path is not None:
        done = _load_checkpoint(checkpoint_path, corpus_hash)
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
                row = {k: v for k, v in cached.items() if k != "corpus_hash"}
                _log("p5-phase4",
                     f"pair {index + 1}/{total} {fixture_id}: {row['disposition']} "
                     "(from checkpoint)", started)
            else:
                pair_started = time.perf_counter()
                replay_row = _replay_pair(loaded.pairs[fixture_id], loaded.root, release_root,
                                          staged)
                pair_elapsed = time.perf_counter() - pair_started
                row = {"fixture_id": fixture_id, **replay_row}
                _log("p5-phase4",
                     f"pair {index + 1}/{total} {fixture_id}: {row['disposition']} "
                     f"(pair +{pair_elapsed:.1f}s)", started)
                if checkpoint_fh is not None:
                    checkpoint_fh.write(_json.dumps({**row, "corpus_hash": corpus_hash}) + "\n")
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
