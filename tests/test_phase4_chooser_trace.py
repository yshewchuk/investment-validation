"""Tier-0 fix 5: the DYN-SV choice replayed natively from a multi-member trace.

Two layers, both on synthetic inputs:

* the frozen chooser declaration a DYN-SV menu row's strict trace carries
  (``tools.capture_tier0_corpus.frozen_chooser_declaration``), resolved by
  ``checks.phase4_frozen_bridge.prepare_frozen_chooser`` into the block the
  native chooser stage runs, must give legacy ``Scorer._score_chooser``'s
  ``chooser_score`` (real legacy scorer, real native stage);
* a ``dyn_sv_choice`` pair carries one strict trace per ranked member
  (``chooser_trace``) and ``checks/phase4_real.py`` verifies each member,
  scores it natively, runs the native chooser over the members and compares
  the choice with the one legacy ``dynamic_short_vol`` made.
"""
from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

import engine.score as score_mod
import tests.test_v2_scoring_native_chooser_features as cf
from checks import phase4_real
from checks.phase4_frozen_bridge import (
    FROZEN_CHOOSER_FIELD,
    FrozenBridgeError,
    prepare_frozen_chooser,
    with_frozen_chooser,
)
from engine.v2.foundation import content_hash
from engine.v2.models.chooser_analog_pool import ChooserAnalogPoolArtifact
from engine.v2.scoring import application
from engine.v2.scoring.source_inputs import SourceBundle, build_native_score_inputs
from tests.test_phase4_capture_strict import (
    _artifact,
    _candidate,
    _chooser_case,
    _chooser_scorer,
    _collector,
    _full_strict_candidate,
    _run_chooser,
    chooser_root,  # noqa: F401 -- pytest fixture
)
from tools.capture_tier0_corpus import (
    CHOOSER_TRACE_SCHEMA,
    StrictTraceCaptureError,
    _chooser_consumed_rows,
    _hydrate_trace,
    _package_resources,
    attach_strict_probe,
    frozen_chooser_declaration,
    make_pair,
)

CLOCK = "legacy.decision_offset.0"


# -- the frozen chooser declaration ---------------------------------------------------


def _captured_chooser(chooser_root, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr("engine.paths.ROOT", tmp_path)
    case = _chooser_case()
    collector = _collector()
    _run_chooser(_chooser_scorer(tmp_path, monkeypatch, chooser_root), case, collector)
    return case, _candidate(collector)


def _chooser_request(case):
    return replace(cf._request(case["strategy"]), decision_clock_id=CLOCK)


def _declared_plan(candidate, case, tmp_path, *, tamper=None):
    request = _chooser_request(case)
    declaration, package, extra = frozen_chooser_declaration(
        candidate, deployment_id=request.deployment_id, release_root=tmp_path / "release")
    if tamper is not None:
        declaration = tamper(copy.deepcopy(declaration))
    resources = _package_resources(package) + extra
    inputs = SimpleNamespace(chooser={FROZEN_CHOOSER_FIELD: declaration})
    plan = prepare_frozen_chooser(
        release_root=tmp_path / "release", resource_rows=resources,
        verified_documents={declaration["release_resource_id"]: package.sidecar_document},
        request=request, inputs=inputs)
    return declaration, package, plan, request


def _native_with(plan, case, candidate):
    """The same bundle as the R4-19 chooser test, but its chooser block is the
    one the trace declaration resolves to, and its chooser features are the
    rows the capture merges into ``model_inputs``."""
    vector = {name: value for name, value in _chooser_consumed_rows(candidate).items()
              if value is not None}
    bundle = SourceBundle(
        source_ref="capture-chooser", strategy=case["strategy"],
        context={"ticker": "AAA", "event_date": cf.EVENT, "entry_date": case["entry_date"],
                 "exit_date": "2026-09-09", "expiry": cf.EXPIRY, "spot": case["spot"]},
        raw_quotes=case["quotes"], feature_vector=vector,
        feature_missing_mask={name: False for name in vector},
        model_identity={"size": {"model_id": "synthetic"}},
        forecast_recipes={"forecast_abs_move": {"intercept": 6.0, "coefficients": {}},
                          "pred_iv_crush": {"intercept": -20.0, "coefficients": {}}},
        model_artifact_refs={"forecast_abs_move": "sha256:a", "pred_iv_crush": "sha256:b"},
        residual_recipe={"mode": "planned_exit", "pre_iv30": 40.0},
        paired_residual_rows=tuple(
            {"event_date": f"2025-{1 + i % 12:02d}-{1 + i % 27:02d}",
             "pred_abs_move": 3.0 + i % 9, "err_move": float(i % 7) - 3.0,
             "err_crush": -5.0 + float(i % 5)}
            for i in range(400)),
        analog_recipe={},
        gate_recipe={"model": {"intercept": 1.0, "coefficients": {}}, "threshold": 0.0})
    inputs = with_frozen_chooser(build_native_score_inputs(bundle), plan)
    return application.score_one(cf._request(case["strategy"]), inputs)


def test_frozen_chooser_declaration_is_json_and_resolves_to_the_chooser_block(
        chooser_root, tmp_path, monkeypatch):  # noqa: F811
    case, candidate = _captured_chooser(chooser_root, tmp_path, monkeypatch)
    declaration, package, plan, _request = _declared_plan(candidate, case, tmp_path)

    import json
    assert json.loads(json.dumps(declaration)) == declaration
    roles = {row["role"] for row in package.sidecar_document["bindings"]}
    assert roles == {"chooser", "implied_t1", "runup_move"}
    assert set(declaration["recipe"]["producers"]) == set(cf.FOLDS)
    assert set(declaration["fold_pools"]) == {*cf.FOLDS, "pred_abs_move"}
    assert declaration["admissible_table"] is not None
    assert isinstance(plan.block["analog_pool"], ChooserAnalogPoolArtifact)
    assert plan.request_refs == frozenset(package.request_refs)


def test_frozen_chooser_from_the_trace_declaration_equals_legacy(
        chooser_root, tmp_path, monkeypatch):  # noqa: F811
    case, candidate = _captured_chooser(chooser_root, tmp_path, monkeypatch)
    _declaration, _package, plan, _request = _declared_plan(candidate, case, tmp_path)

    record = _native_with(plan, case, candidate)
    legacy_case = cf._legacy_case_from_record(case, record)
    legacy = _run_chooser(_chooser_scorer(tmp_path, monkeypatch, chooser_root), legacy_case)
    assert legacy.chooser_score is not None, legacy.flags
    assert record.forecasts["chooser_score"] == legacy.chooser_score

    # Planted defect: a perturbed producer pool in the declaration is seen.
    def perturb(declaration):
        pool = declaration["fold_pools"]["pred_im_t1_d14"]
        pool["residuals"] = [3.0 * value for value in pool["residuals"]]
        return declaration

    _d, _p, planted, _r = _declared_plan(candidate, case, tmp_path, tamper=perturb)
    assert _native_with(planted, case, candidate).forecasts["chooser_score"] != (
        legacy.chooser_score)


def test_frozen_chooser_refuses_a_tampered_pool_file_and_stray_chooser_keys(
        chooser_root, tmp_path, monkeypatch):  # noqa: F811
    case, candidate = _captured_chooser(chooser_root, tmp_path, monkeypatch)
    declaration, package, _plan, request = _declared_plan(candidate, case, tmp_path)
    resources = _package_resources(package)
    documents = {declaration["release_resource_id"]: package.sidecar_document}
    state = next((tmp_path / "release" / "resources" / "states").iterdir())
    extra = [{"resource_id": declaration["analog_pool_resource_id"], "ref": "state:x",
              "kind": "artifact", "path": str(state.relative_to(tmp_path / "release")),
              "sha256": "sha256:" + "0" * 64}]
    with pytest.raises(FrozenBridgeError, match="sha256: mismatch"):
        prepare_frozen_chooser(
            release_root=tmp_path / "release", resource_rows=resources + extra,
            verified_documents=documents, request=request,
            inputs=SimpleNamespace(chooser={FROZEN_CHOOSER_FIELD: declaration}))
    with pytest.raises(FrozenBridgeError, match="must stand alone"):
        prepare_frozen_chooser(
            release_root=tmp_path / "release", resource_rows=resources,
            verified_documents=documents, request=request,
            inputs=SimpleNamespace(chooser={FROZEN_CHOOSER_FIELD: declaration, "x": 1}))


def test_a_pool_file_changed_since_legacy_read_it_is_refused(
        chooser_root, tmp_path, monkeypatch):  # noqa: F811
    case, candidate = _captured_chooser(chooser_root, tmp_path, monkeypatch)
    frame = cf.pool_frame()
    frame.iloc[::-1].to_parquet(tmp_path / score_mod.CHOOSER_ANALOG_POOL, index=False)
    import tools.capture_tier0_corpus as capture

    monkeypatch.setattr(capture, "_CHOOSER_POOL_STATES", {})
    with pytest.raises(StrictTraceCaptureError, match="differs from the one legacy loaded"):
        frozen_chooser_declaration(candidate, deployment_id="dep-1",
                                   release_root=tmp_path / "release")


# -- the multi-member dyn_sv_choice trace ----------------------------------------------


SPOTS = ([95.0, 105.0], [80.0, 120.0], [99.0, 101.0])


def _member(tmp_path, index, spots, path, digest):
    candidate = _full_strict_candidate(
        fixture_id=f"m{index}", ticker="AAA", driver_vector={"x": 2.0 + index},
        gate_vector={"x": 9.0, "n_prior": 5.0}, path=path, digest=digest)
    checkpoint = candidate["legacy_trace"]["checkpoints"]["source_inputs"]
    for binding in checkpoint["value"]["model_bindings"]:
        binding["decision_clock"] = CLOCK
    checkpoint["value"]["native_recipes"]["simulation"] = {
        "terminal_spots": spots, "capital_at_risk": 3.0}
    checkpoint["content_hash"] = content_hash(checkpoint["value"])
    return candidate


def _choice(tmp_path, monkeypatch):
    """A dyn_sv_choice candidate over three traced members. The members are
    synthetic STR-THRU rows (the traced path is strategy-agnostic); their
    legacy records carry the member values legacy would rank, and the legacy
    choice is the REAL ``dynamic_short_vol`` over those rows."""
    monkeypatch.setattr("engine.paths.ROOT", tmp_path)
    path, digest = _artifact(tmp_path)
    members = [_member(tmp_path, i, spots, path, digest) for i, spots in enumerate(SPOTS)]
    # Legacy member records: the native member values (as if member parity
    # held), so the choice comparison is the only thing under test here.
    from tools.capture_tier0_corpus import strict_trace_one

    for member in members:
        _trace, native = strict_trace_one(copy.deepcopy(member), "snapshot-1",
                                          tmp_path / "probe")
        member["record"] = {"strategy": "STR-THRU", "ticker": "AAA",
                            "event_date": "2026-09-17", "session": "AMC",
                            "exp_pnl_sim": native.forecasts["exp_pnl_sim"],
                            "strike_offset": None}
    frame = pd.DataFrame([m["record"] for m in members])
    chosen = score_mod.dynamic_short_vol(frame, menu=("STR-THRU",))
    record = {key: (None if isinstance(value, float) and value != value else value)
              for key, value in chosen.iloc[0].to_dict().items()}
    candidate = {
        "fixture_id": "dyn-0", "kind": "dyn_sv_choice", "covers": [],
        "request": {"kind": "dyn_sv_resolution",
                    "entry_point": "engine.score.dynamic_short_vol",
                    "menu": ["STR-THRU"], "frame": "forward",
                    "frame_rows": [{"request": m["request"], "record": m["record"]}
                                   for m in members]},
        "record": record, "duration": 0.0, "relations": {},
        "members": [{**m, "kind": "score_result"} for m in members],
    }
    return candidate


def _pair(candidate):
    return make_pair(candidate["fixture_id"], [], candidate["request"], candidate["record"],
                     record_kind="dyn_sv_choice", duration=0.0,
                     input_trace=_hydrate_trace(candidate.get("input_trace")),
                     legacy_input_hash=candidate.get("legacy_input_hash"))


def _corpus(tmp_path, *pairs):
    return SimpleNamespace(
        root=tmp_path, index={"pairs": {p["fixture_id"]: {} for p in pairs}},
        ordered_ids=[p["fixture_id"] for p in pairs],
        pairs={p["fixture_id"]: p for p in pairs})


def test_dyn_sv_choice_carries_one_strict_trace_per_ranked_member(tmp_path, monkeypatch):
    candidate = _choice(tmp_path, monkeypatch)
    attached, gaps = attach_strict_probe([candidate], "snapshot-1", tmp_path)
    assert attached == ("dyn-0",) and gaps == {}

    trace = _hydrate_trace(candidate["input_trace"])
    assert trace["schema_version"] == CHOOSER_TRACE_SCHEMA
    assert [m["member_index"] for m in trace["members"]] == [0, 1, 2]
    assert [m["request_hash"] for m in trace["members"]] == [
        content_hash(row["request"]) for row in candidate["request"]["frame_rows"]]
    pair = _pair(candidate)
    assert pair["payload"]["trace_disposition"] == "complete"


def test_phase4_real_replays_the_choice_natively_and_it_matches_legacy(
        tmp_path, monkeypatch):
    candidate = _choice(tmp_path, monkeypatch)
    attach_strict_probe([candidate], "snapshot-1", tmp_path / "captured")
    pair = _pair(candidate)
    # The corpus is replayed from where it was published, not where it was
    # built (the capture writes a temporary sibling and renames it).
    import shutil

    shutil.copytree(tmp_path / "captured", tmp_path / "published")
    shutil.rmtree(tmp_path / "captured")
    tmp_path = tmp_path / "published"

    members, choice = phase4_real._replayed_chooser(pair, tmp_path)
    assert len(members) == 3
    checks = phase4_real._chooser_selection_checks(pair["payload"]["record"], choice)
    assert checks == {"chosen_strategy": True, "menu_size": True, "chosen_margin": True}
    assert choice.chooser_selection["ranking_key"] == "exp_pnl_sim"

    _release, parity = phase4_real._native_parity(_corpus(tmp_path, pair))
    row = _release["dispositions"][0]
    assert row["disposition"] == "compared", row
    assert parity["population"]["compared"] == 1


def test_a_wrong_legacy_choice_is_a_chooser_disagreement(tmp_path, monkeypatch):
    candidate = _choice(tmp_path, monkeypatch)
    attach_strict_probe([candidate], "snapshot-1", tmp_path)
    # Planted defect: legacy recorded a different margin (another runner-up).
    candidate["record"] = dict(candidate["record"],
                               chosen_margin=candidate["record"]["chosen_margin"] + 1.0)
    pair = _pair(candidate)
    _members, choice = phase4_real._replayed_chooser(pair, tmp_path)
    assert phase4_real._chooser_selection_checks(pair["payload"]["record"], choice)[
        "chosen_margin"] is False
    release, _parity = phase4_real._native_parity(_corpus(tmp_path, pair))
    assert release["dimension_agreement"]["chooser"] is False


def test_a_member_trace_bound_to_another_frame_row_is_incomparable(tmp_path, monkeypatch):
    candidate = _choice(tmp_path, monkeypatch)
    attach_strict_probe([candidate], "snapshot-1", tmp_path)
    tampered = copy.deepcopy(candidate)
    # The pair's legacy frame row now names another request than the one
    # its member trace was captured from.
    rows = tampered["request"]["frame_rows"]
    rows[0]["request"] = dict(rows[0]["request"], ticker="ZZZ")
    with pytest.raises(phase4_real._TraceError, match="not the frame row's request"):
        phase4_real._replayed_chooser(_pair(tampered), tmp_path)
    release, _parity = phase4_real._native_parity(_corpus(tmp_path, _pair(tampered)))
    assert release["population"]["incomparable"] == 1


def test_one_untraceable_member_makes_the_whole_choice_a_gap(tmp_path, monkeypatch):
    candidate = _choice(tmp_path, monkeypatch)
    del candidate["members"][1]["legacy_trace"]["checkpoints"]["source_inputs"]
    scored = _full_strict_candidate(
        fixture_id="s0", ticker="BBB", driver_vector={"x": 1.0},
        gate_vector={"x": 1.0, "n_prior": 1.0}, path=tmp_path / "model.joblib",
        digest=_artifact(tmp_path)[1])
    attached, gaps = attach_strict_probe([candidate, scored], "snapshot-1", tmp_path)
    assert "dyn-0" not in attached
    assert gaps["dyn-0"].startswith("member 1:")


# -- the other consumers of a traced corpus ------------------------------------------


def test_phase5_replay_replays_a_choice_member_by_member(tmp_path, monkeypatch):
    from checks.phase5_phase4_replay import PHASE4_MEMBER_ABSENT, PHASE4_UNVERIFIED, _replay_pair

    candidate = _choice(tmp_path, monkeypatch)
    attach_strict_probe([candidate], "snapshot-1", tmp_path)
    pair = _pair(candidate)
    # Nothing staged: the first member's bindings are absent from the release.
    row = _replay_pair(pair, tmp_path, tmp_path / "staged", {})
    assert (row["disposition"], row["code"]) == ("member_absent", PHASE4_MEMBER_ABSENT)
    assert row["detail"].startswith("member 0: ")
    broken = copy.deepcopy(pair)
    broken["payload"]["input_trace"]["members"].pop()
    row = _replay_pair(broken, tmp_path, tmp_path / "staged", {})
    assert (row["disposition"], row["code"]) == ("unverified", PHASE4_UNVERIFIED)


def test_calibration_keys_come_from_every_member_of_a_choice(tmp_path, monkeypatch):
    from tools.phase5_calibration_keys import _keyed_pairs, _pair_key

    candidate = _choice(tmp_path, monkeypatch)
    attach_strict_probe([candidate], "snapshot-1", tmp_path)
    pair = _pair(candidate)
    keyed = list(_keyed_pairs("dyn-0", pair))
    assert [label for label, _ in keyed] == [f"dyn-0#member{i}" for i in range(3)]
    for _label, member in keyed:
        key, reason = _pair_key(member)
        assert key is not None, reason
        assert key[0] == "STR-THRU"


def _menu_row_with_frozen_chooser(chooser_root, tmp_path, monkeypatch):  # noqa: F811
    """``(pair, captured trace, native record)`` of one TWIN-P menu row whose
    trace carries the frozen chooser, published under ``tmp_path/published``."""
    import shutil

    from tools.capture_tier0_corpus import strict_trace_one

    _case, chooser_candidate = _captured_chooser(chooser_root, tmp_path, monkeypatch)
    source = chooser_candidate["legacy_trace"]["checkpoints"]["source_inputs"]["value"]
    path, digest = _artifact(tmp_path)
    candidate = _full_strict_candidate(
        fixture_id="twin", ticker="AAA", strategy="TWIN-P", driver_vector={"x": 2.0},
        gate_vector={"x": 9.0, "n_prior": 5.0}, path=path, digest=digest)
    checkpoint = candidate["legacy_trace"]["checkpoints"]["source_inputs"]
    for binding in checkpoint["value"]["model_bindings"]:
        binding["decision_clock"] = CLOCK
        binding["strategy"] = "TWIN-P"
    checkpoint["value"]["frozen"] = source["frozen"]
    checkpoint["content_hash"] = content_hash(checkpoint["value"])

    trace, native = strict_trace_one(candidate, "snapshot-1", tmp_path / "captured")
    shutil.copytree(tmp_path / "captured", tmp_path / "published")
    pair = {"payload": {"request": candidate["request"], "input_trace": trace,
                        "input_trace_hash": trace["trace_hash"],
                        "legacy_input_hash": trace["shared_input_hash"]}}
    return pair, trace, native


def test_a_menu_row_with_a_frozen_chooser_traces_and_replays_identically(
        chooser_root, tmp_path, monkeypatch):  # noqa: F811
    """Capture -> phase4_real round trip of one DYN-SV menu row carrying the
    frozen chooser: the chooser release and pool are declared resources, the
    request names their refs, and the replayed stage receipts (the chooser's
    binding and state identities included) equal the captured ones."""
    pair, trace, native = _menu_row_with_frozen_chooser(chooser_root, tmp_path, monkeypatch)
    assert set(trace["native_inputs"]["chooser"]) == {FROZEN_CHOOSER_FIELD}
    # driver + gate bindings, then the chooser and its two producer folds
    assert len(set(trace["request"]["model_artifact_refs"])) == 5

    verified = phase4_real._verified_trace_bundle(pair, tmp_path / "published")
    assert "executors" in verified["inputs"].chooser
    replayed, _receipts, _ids = phase4_real._replayed_member(verified)
    assert replayed.score_id == native.score_id


def _stage_everything(verified, corpus_root, release_root):
    """Stage every model member, the chooser pool and table: ``staged`` map."""
    from checks import phase5_release as layout
    from engine.v2.models.frozen_state import serialize_frozen_state

    staged = {}
    chooser = verified["frozen_chooser"]
    for release, label in ((verified["frozen_replay"].release, "model"),
                           (chooser.release, "tier4_folds:chooser")):
        for binding in release.bindings:
            for member in binding.members:
                digest, _ = layout.write_object(
                    release_root, (corpus_root / member.path).read_bytes())
                assert digest == member.content_hash
                staged[digest] = f"{label}:{binding.role}"
    for state, member_id in ((chooser.analog_pool, "chooser_analog_pool"),
                             (chooser.admissible_table, "admissible_table:dyn_sv")):
        digest, _ = layout.write_object(release_root, serialize_frozen_state(state))
        staged[digest] = member_id
    return staged


def test_phase5_replay_serves_the_frozen_chooser_from_the_staged_release(
        chooser_root, tmp_path, monkeypatch):  # noqa: F811
    """The P5-6 replay rebinds the chooser champion, its producer folds, the
    k-NN pool and the n_admissible table to the staged release by content
    hash; the pair replays, and each missing piece is ``member_absent``."""
    from checks import phase5_phase4_replay as replay
    from checks import phase5_release as layout

    pair, _trace, _native = _menu_row_with_frozen_chooser(chooser_root, tmp_path, monkeypatch)
    corpus = tmp_path / "published"
    verified = phase4_real._verified_trace_bundle(pair, corpus)
    staged_root = tmp_path / "staged"
    staged = _stage_everything(verified, corpus, staged_root)

    row = replay._replay_pair(pair, corpus, staged_root, staged)
    assert row["disposition"] == "replayed", row
    assert {"chooser_analog_pool", "admissible_table:dyn_sv",
            "tier4_folds:chooser:chooser"} <= set(row["members"])

    chooser = verified["frozen_chooser"]
    champion = next(b for b in chooser.release.bindings if b.role == "chooser")
    for dropped, marker in ((champion.members[0].content_hash, champion.binding_id),
                            (chooser.analog_pool.content_hash, "chooser:analog_pool")):
        without = {h: m for h, m in staged.items() if h != dropped}
        if marker == "chooser:analog_pool":
            without = {h: m for h, m in staged.items() if m != "chooser_analog_pool"}
        row = replay._replay_pair(pair, corpus, staged_root, without)
        assert row["disposition"] == "member_absent", row
        assert marker in row["detail"]

    # The staged bytes are what the rebuilt champion executes (this synthetic
    # menu row declines the ranking, so drive the executor directly): with the
    # staged object removed it refuses, although the corpus copy is intact.
    block, absent, _used = replay.rebind_chooser(chooser, verified["request"], staged_root,
                                                 staged)
    assert absent == [] and block["analog_pool"] is not None
    facts = {name: 0.0 for name in champion.feature_order}
    assert "chooser_score" in block["executors"]["chooser_score"].predict(facts)
    (layout.deployment_root(staged_root)
     / layout.object_relpath(champion.members[0].content_hash)).unlink()
    block, _absent, _used = replay.rebind_chooser(chooser, verified["request"], staged_root,
                                                  staged)
    with pytest.raises(Exception):  # noqa: B017 -- any refusal: the bytes are gone
        block["executors"]["chooser_score"].predict(facts)
