"""P5-6 acceptance gate on synthetic tmp_path releases.

Every release here is built by the preparer's own ``write_release`` from
json-linear model bindings and real P5-4 payoff artifacts (built by the
layer-6 builders from synthetic rows). No real data is read.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from checks import phase5_acceptance as gate
from checks import phase5_release as layout
from engine.v2.models import (
    ArtifactInventoryMember,
    ArtifactMember,
    ModelArtifactInventory,
    ModelBinding,
    ModelRelease,
    ModelReleaseInventory,
    ReleaseBinding,
    ReleaseRequirement,
    current_pointer,
    promote,
    stage_release,
)
from engine.v2.models.payoff_artifact import serialize_payoff_artifact
from engine.v2.models.training.payoff import (
    build_payoff_line_artifact,
    build_payoff_surface_artifact,
)
from tools import phase5_prepare_release as prep

CLOCK = "legacy.entry_close.v1"
_ROWS = [
    {"driver": 0.0, "spot_entry": 100.0, "exit_value": 2.0, "exit_date": "2026-09-01"},
    {"driver": 10.0, "spot_entry": 100.0, "exit_value": 6.0, "exit_date": "2026-09-01"},
]
_RUNUP_ROWS = [dict(row, spot_exit=100.0, strike=100.0) for row in _ROWS]


def _hash(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _linear(intercept: float) -> bytes:
    return json.dumps({"schema_version": "linear_estimator.v1.0", "feature_order": ["x"],
                       "outputs": [{"name": "y", "intercept": intercept,
                                    "coefficients": [1.0]}]}, sort_keys=True).encode()


def _models(release_id: str, *, intercept: float = 1.0,
            keys=layout.EXPECTED_MODEL_BINDINGS):
    """A json-linear release over ``keys`` plus its matching inventory."""
    bindings, artifacts, inv_bindings, requirements, payloads = [], [], [], [], {}
    for index, (role, strategy) in enumerate(keys):
        data = _linear(intercept + index)
        digest = _hash(data)
        payloads[digest] = data
        model_id = f"m-{role}-{strategy}".replace("*", "all")
        bindings.append(ModelBinding(
            binding_id=f"{role}:{strategy}", model_id=model_id, role=role,
            strategy_id=strategy, decision_clock_id=CLOCK, adapter="json-linear.v1",
            feature_order=("x",), output_names=("y",),
            members=(ArtifactMember(name="estimator", path="src", content_hash=digest),)))
        artifacts.append(ModelArtifactInventory(
            artifact_id=model_id, role=role, strategy_ids=(strategy,),
            compatible_clock_ids=(CLOCK,), target_contract_ref="t", ordered_features=("x",),
            members=(ArtifactInventoryMember(member_id=f"{model_id}:estimator",
                                             kind="estimator", artifact_ref="src",
                                             content_hash=digest),)))
        inv_bindings.append(ReleaseBinding(
            role=role, strategy_id=strategy, clock_id=CLOCK, artifact_id=model_id,
            ordered_features=("x",), required_member_kinds=("estimator",)))
        requirements.append(ReleaseRequirement(role=role, strategy_id=strategy, clock_id=CLOCK))
    inventory = ModelReleaseInventory(
        release_id=release_id, deployment_id="dep", known_clock_ids=(CLOCK,),
        artifacts=tuple(artifacts), bindings=tuple(inv_bindings),
        requirements=tuple(requirements), artifact_manifest_ref="registry.json",
        evidence_refs=("evidence",))
    release = ModelRelease(release_id=release_id, deployment_id="dep",
                           bindings=tuple(bindings))
    return release, inventory, payloads


def _state_payloads() -> dict[str, dict[str, bytes]]:
    line = build_payoff_line_artifact(_ROWS, strategy="STR-THRU", driver="abs_move",
                                      alpha=0.5, min_trades=2)
    dated = build_payoff_line_artifact(_ROWS, strategy="STR-THRU", driver="abs_move",
                                       alpha=0.25, before="2026-09-10", min_trades=2)
    surface = build_payoff_surface_artifact(_RUNUP_ROWS, alpha=0.5, min_trades=2)
    return {
        "payoff_line:STR-THRU": {"a": serialize_payoff_artifact(line),
                                 "b": serialize_payoff_artifact(dated)},
        "payoff_surface:STR-RUNUP": {"a": serialize_payoff_artifact(surface)},
        "tier4_folds:size": {"m_202609_abcdefabcdef.joblib": b"fold-bytes"},
    }


#: A reduced catalog whose every member can be built here: the two payoff
#: states (real typed loader + real v2 consumer) and one raw state.
SPECS = tuple(s for s in layout.STATE_SPECS if s.member_id in _state_payloads())


def _stub_probe(member_id: str, consumer: str):
    """A consumer that reads its member from the resolved release only."""
    def probe(ctx):
        state = ctx.states.get(member_id)
        if state is None:
            return [{"consumer": consumer, "member_id": member_id, "blocked": True}]
        return [{"consumer": consumer, "member_id": member_id, "resolved": True,
                 "refused_when_missing": True, "detail": ""}]
    return probe


CONSUMERS = {
    "frozen_stage_executor": gate.CONSUMERS["frozen_stage_executor"],
    "model_stage.payoff_line": gate.CONSUMERS["model_stage.payoff_line"],
    "model_stage.payoff_surface": gate.CONSUMERS["model_stage.payoff_surface"],
    "features.tier4_serving_folds": _stub_probe("tier4_folds:size",
                                                "features.tier4_serving_folds"),
}


def _incumbent(tmp_path: Path) -> Path:
    store = tmp_path / "incumbent"
    release, inventory, payloads = _models("rel-incumbent", intercept=100.0)
    stage_release(store, release, inventory, payloads)
    promote(store, "rel-incumbent")
    return store


def _release(tmp_path: Path, *, states=None, keys=layout.EXPECTED_MODEL_BINDINGS,
             incumbent=True, specs=SPECS) -> Path:
    out = tmp_path / "release"
    release, inventory, payloads = _models("rel-candidate", keys=keys)
    builds = prep.build_states(_state_payloads() if states is None else states)
    builds = [b for b in builds if b.spec in specs]
    prep.write_release(out, release, inventory, payloads, builds,
                       incumbent=_incumbent(tmp_path) if incumbent else None)
    return out


def _run(tmp_path: Path, release_root: Path, **kwargs) -> dict:
    cache = tmp_path / "model-cache"
    cache.mkdir(exist_ok=True)
    kwargs.setdefault("consumers", CONSUMERS)
    kwargs.setdefault("state_specs", SPECS)
    return gate.build_evidence(
        release_root, report_dir=tmp_path / "report",
        watch_dirs=[cache, layout.deployment_root(release_root)], **kwargs)


def test_complete_release_passes(tmp_path):
    root = _release(tmp_path)
    pointer_before = (layout.deployment_root(root) / "DEPLOYED").read_bytes()
    evidence = _run(tmp_path, root)

    assert evidence["findings"] == []
    assert evidence["release_ok"] is True
    # No Phase 4 corpus: release acceptance only, never the full PASS.
    assert evidence["status"] == "RELEASE_PASS"
    assert evidence["phase4"] == {"status": "NOT_RUN"}
    assert {r["member_id"] for r in evidence["members"]} == (
        {f"model:{r}:{s}" for r, s in layout.EXPECTED_MODEL_BINDINGS}
        | {s.member_id for s in SPECS})
    consumers = {(r["consumer"], r["member_id"]) for r in evidence["consumers"]}
    assert ("model_stage.payoff_line", "payoff_line:STR-THRU") in consumers
    assert ("model_stage.payoff_surface", "payoff_surface:STR-RUNUP") in consumers
    assert all(r["status"] == "ok" for r in evidence["consumers"])
    # both (alpha, cutoff) folds of the line were scored, each by its own key
    assert sum(1 for c, m in [(r["consumer"], r["member_id"]) for r in evidence["consumers"]]
               if c == "model_stage.payoff_line") == 2
    assert evidence["scoring_pass"]["fit_attempts"] == 0
    assert all(evidence["scoring_pass"]["controls"].values())
    # The gate never moves the staged root's own pointer.
    assert (layout.deployment_root(root) / "DEPLOYED").read_bytes() == pointer_before
    report = Path(evidence["report"]).read_text()
    assert "Release members" in report and "payoff_line:STR-THRU" in report


def test_missing_state_member_fails_and_blocks_its_consumer(tmp_path):
    states = _state_payloads()
    del states["payoff_line:STR-THRU"]
    evidence = _run(tmp_path, _release(tmp_path, states=states))

    assert evidence["release_ok"] is False
    assert {"P5_MEMBER_MISSING", "P5_CONSUMER_BLOCKED"} <= set(evidence["finding_codes"])
    row = next(r for r in evidence["members"] if r["member_id"] == "payoff_line:STR-THRU")
    assert row["status"] == "MISSING" and row["verdict"] == "P5_MEMBER_MISSING"


def test_missing_model_binding_fails(tmp_path):
    keys = layout.EXPECTED_MODEL_BINDINGS[:-1]  # no chooser
    evidence = _run(tmp_path, _release(tmp_path, keys=keys))

    assert "P5_MEMBER_MISSING" in evidence["finding_codes"]
    assert any(f["subject"] == "model:chooser:DYN-SV" for f in evidence["findings"])


def test_tampered_state_object_fails_with_hash_mismatch(tmp_path):
    root = _release(tmp_path)
    manifest = layout.read_manifest(root)
    row = next(r for r in manifest["members"] if r["member_id"] == "payoff_surface:STR-RUNUP")
    (layout.deployment_root(root) / row["objects"][0]["path"]).write_bytes(b"{}")
    evidence = _run(tmp_path, root)

    assert "P5_MEMBER_HASH_MISMATCH" in evidence["finding_codes"]
    assert "P5_CONSUMER_BLOCKED" in evidence["finding_codes"]


def test_tampered_model_object_fails_hash_and_consumer(tmp_path):
    root = _release(tmp_path)
    member = gate.deployment._read_manifest(
        layout.deployment_root(root), "rel-candidate").release.bindings[0].members[0]
    (layout.deployment_root(root) / member.path).write_bytes(_linear(-5.0))
    evidence = _run(tmp_path, root)

    assert {"P5_MEMBER_HASH_MISMATCH", "P5_CONSUMER_UNRESOLVED"} <= set(
        evidence["finding_codes"])


def test_tampered_manifest_row_fails_layout(tmp_path):
    root = _release(tmp_path)
    path = root / layout.MANIFEST_NAME
    body = json.loads(path.read_text())
    body["members"] = body["members"][1:]
    path.write_text(json.dumps(body))
    evidence = _run(tmp_path, root)

    assert evidence["finding_codes"] == ["P5_RELEASE_LAYOUT"]


@pytest.mark.parametrize("module_name", ["engine.models.no_fit", "engine.v2.models.no_fit"])
def test_fitting_attempt_under_scoring_fails(tmp_path, module_name):
    import importlib

    module = importlib.import_module(module_name)

    def fitting_probe(ctx):
        try:  # a consumer that swallows the refusal is still caught
            module.forbid_fitting("synthetic.consumer.fit")
        except Exception:
            pass
        return []

    evidence = _run(tmp_path, _release(tmp_path), consumers={**CONSUMERS, "fit": fitting_probe})

    assert "P5_RUNTIME_FIT" in evidence["finding_codes"]
    assert any(f["detail"] == "synthetic.consumer.fit" for f in evidence["findings"])


def test_model_cache_write_under_scoring_fails(tmp_path):
    cache = tmp_path / "model-cache"

    def writing_probe(ctx):
        import joblib

        try:
            joblib.dump({"m": 1}, cache / "fold.joblib")
        except gate.ModelCacheWrite:
            pass
        (cache / "side-channel.bin").write_bytes(b"x")
        return []

    evidence = _run(tmp_path, _release(tmp_path),
                    consumers={**CONSUMERS, "write": writing_probe})

    details = [f["detail"] for f in evidence["findings"] if f["code"] == "P5_MODEL_CACHE_WRITE"]
    assert "joblib.dump x1" in details
    assert any(d.endswith("side-channel.bin") for d in details)


def test_rollback_is_exact(tmp_path):
    evidence = _run(tmp_path, _release(tmp_path))
    rollback = evidence["rollback"]

    assert rollback["status"] == "ok" and rollback["incumbent"] == "rel-incumbent"
    assert rollback["checks"] == {
        "promoted_resolves_candidate": True, "pointer_release_id_restored": True,
        "target_release_byte_exact": True, "manifests_and_history_prefix_byte_exact": True,
        "history_appended_two": True,
    }


def test_rollback_that_lands_elsewhere_is_caught(tmp_path, monkeypatch):
    def wrong_rollback(root, **_):
        return gate.deployment.promote(root, "rel-candidate")

    monkeypatch.setattr(gate.deployment, "rollback", wrong_rollback)
    evidence = _run(tmp_path, _release(tmp_path))

    assert "P5_ROLLBACK_NOT_EXACT" in evidence["finding_codes"]
    assert evidence["rollback"]["checks"]["pointer_release_id_restored"] is False


def test_no_incumbent_is_red_unless_first_deployment(tmp_path):
    root = _release(tmp_path, incumbent=False)
    assert "P5_ROLLBACK_NO_INCUMBENT" in _run(tmp_path, root)["finding_codes"]

    first = _run(tmp_path, root, first_deployment=True)
    assert first["release_ok"] is True
    assert first["rollback"]["checks"]["first_deployment_rollback_refused"] is True
    assert current_pointer(layout.deployment_root(root)) is None  # scratch copy only


def test_full_catalog_never_skips_a_member(tmp_path):
    """The production catalog and consumer table: every pending member and
    consumer is an explicit row and a finding, so this tree stays red."""
    root = _release(tmp_path, specs=layout.STATE_SPECS)
    evidence = _run(tmp_path, root, consumers=gate.CONSUMERS,
                    state_specs=layout.STATE_SPECS)

    ids = {r["member_id"] for r in evidence["members"]}
    assert {s.member_id for s in layout.STATE_SPECS} <= ids
    assert evidence["status"] == "FAIL"
    assert "P5_CONSUMER_PENDING" in evidence["finding_codes"]
    pending = {r["consumer"] for r in evidence["consumers"] if r["status"] == "PENDING"}
    assert {"simulation.paired_residual_pool", "model_stage.recalibration",
            "analogs.board_analog_matcher"} <= pending
    by_id = {r["member_id"]: r for r in evidence["members"]}
    for member_id in ("board_analog_matcher", "recalibration_map:STR-THRU",
                      "trailing_pnl_cutoff"):
        assert by_id[member_id]["status"] in ("PENDING", "MISSING")
        assert by_id[member_id]["verdict"] in ("P5_MEMBER_PENDING", "P5_MEMBER_MISSING")


def test_report_refused_inside_repo(tmp_path):
    root = _release(tmp_path)
    with pytest.raises(gate.ReportPathError):
        gate.build_evidence(root, report_dir=gate.ROOT / "reports" / "p5-6",
                            consumers=CONSUMERS, state_specs=SPECS,
                            watch_dirs=[layout.deployment_root(root)])


def test_preparer_marks_every_catalog_state(tmp_path):
    builds = prep.build_states({}, notes={"payoff_line:STR-THRU": "no training root"})
    by_id = {b.spec.member_id: b for b in builds}

    assert set(by_id) == {s.member_id for s in layout.STATE_SPECS}
    assert by_id["payoff_line:STR-THRU"].status == "MISSING"
    assert by_id["payoff_line:STR-THRU"].detail == "no training root"
    assert by_id["board_analog_matcher"].status == "PENDING"


def test_preparer_selects_current_snapshot_folds(tmp_path):
    for name in ("size_v1_202608_aaaaaaaaaaaa.joblib", "size_v1_202609_aaaaaaaaaaaa.joblib",
                 "size_v1_202609_bbbbbbbbbbbb.joblib", "other_202609_aaaaaaaaaaaa.joblib"):
        (tmp_path / name).write_bytes(name.encode())
    found = prep.tier4_fold_payloads(tmp_path, {"size": "size_v1"}, "a" * 64, None)
    assert sorted(found["tier4_folds:size"]) == [
        "size_v1_202608_aaaaaaaaaaaa.joblib", "size_v1_202609_aaaaaaaaaaaa.joblib"]
    month = prep.tier4_fold_payloads(tmp_path, {"size": "size_v1"}, "a" * 64, "202609")
    assert list(month["tier4_folds:size"]) == ["size_v1_202609_aaaaaaaaaaaa.joblib"]


def test_preparer_refuses_out_inside_data(monkeypatch, tmp_path):
    from engine import paths

    monkeypatch.setattr(paths, "DATA", tmp_path / "data")
    with pytest.raises(prep.PrepareRefused):
        prep._refuse_data_dir(tmp_path / "data" / "release")
    prep._refuse_data_dir(tmp_path / "elsewhere")
