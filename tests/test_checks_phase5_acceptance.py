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
from engine.v2.models.admissible_table import legacy_n_admissible_table
from engine.v2.models.frozen_state import serialize_frozen_state
from engine.v2.models.lineage import DataDependency, Lineage
from engine.v2.models.payoff_artifact import serialize_payoff_artifact
from engine.v2.models.recalibration_artifact import serialize_recalibration_artifact
from engine.v2.models.training.payoff import (
    build_payoff_line_artifact,
    build_payoff_surface_artifact,
)
from engine.v2.models.training.recalibration import build_recalibration_map_artifact
from engine.v2.models.training.residuals import (
    build_driver_residual_pool_artifact,
    build_paired_residual_pool_artifact,
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


def _recal_pairs(n: int = 400, seed: int = 3):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.0, 1.0, n)
    return pd.DataFrame({
        "strategy": "STR-THRU", "fill_alpha": 0.5, "event_id": [f"E{i}" for i in range(n)],
        "exit_date": pd.Timestamp("2019-01-01") + pd.to_timedelta(rng.integers(0, 600, n), "D"),
        "raw_win": raw, "outcome": (rng.uniform(0, 1, n) < 0.3 + 0.4 * raw).astype(float),
    })


def _recal(alpha: float = 0.5, before=None) -> bytes:
    return serialize_recalibration_artifact(build_recalibration_map_artifact(
        _recal_pairs(), strategy="STR-THRU", alpha=alpha, before=before))


LINEAGE = Lineage(data=(DataDependency(table="tier3.panel", end_exclusive="2026-09-01"),))


def _driver_pool(role: str, *, lineage=LINEAGE, seed: int = 1) -> bytes:
    import numpy as np

    rng = np.random.default_rng(seed)
    rows = [{"prediction": float(p), "residual": float(r)}
            for p, r in zip(rng.uniform(0, 12, 60), rng.normal(0, 1.5, 60))]
    return serialize_frozen_state(build_driver_residual_pool_artifact(
        rows, role=role, model_id=f"m-{role}", fold=None, lineage=lineage))


def _paired_pool() -> bytes:
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(17)
    forecasts, outcomes, crush = [], [], []
    for index, day in enumerate(pd.date_range("2025-06-01", periods=400, freq="D")):
        ticker, stamp = ("AAA", "BBB", "CCC")[index % 3], str(day.date())
        forecasts.append({"ticker": ticker, "event_date": stamp,
                          "pred_abs_move": 4.0 + index % 20 / 10.0, "pred_iv_crush_30": -20.0})
        outcomes.append({"ticker": ticker, "event_date": stamp,
                         "abs_move": float(rng.uniform(0, 12))})
        crush.append({"ticker": ticker, "event_date": stamp,
                      "crush_pct_iv30": float(rng.uniform(-45, 0))})
    return serialize_frozen_state(build_paired_residual_pool_artifact(
        forecasts, outcomes, crush, move_model_id="size_v1_4",
        crush_model_id="iv_crush_v1_gbm", cutoff="2026-09-01", lineage=LINEAGE))


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
        # keyed like the undated line: (STR-THRU, 0.5, no cutoff)
        "recalibration_map:STR-THRU": {"a": _recal()},
        "tier4_folds:size": {"m_202609_abcdefabcdef.joblib": b"fold-bytes"},
        "driver_residual_pool:size": {"m-size|champion": _driver_pool("size")},
        "driver_residual_pool:implied_t1": {"m-implied_t1|champion": _driver_pool("implied_t1")},
        "driver_residual_pool:runup_move": {"m-runup_move|champion": _driver_pool("runup_move")},
        "paired_residual_pool": {"size_v1_4|iv_crush_v1_gbm|2026-09-01": _paired_pool()},
        "admissible_table:dyn_sv": {"dyn_sv.n_admissible_by_depth|v1": serialize_frozen_state(
            legacy_n_admissible_table())},
    }


#: A reduced catalog whose every member can be built here: payoff,
#: recalibration, residual pools and the admissible table (real typed loaders,
#: real v2 consumers where one exists) and one raw state.
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
    "model_stage.recalibration": gate.CONSUMERS["model_stage.recalibration"],
    "model_stage.driver_residual_pool": gate.CONSUMERS["model_stage.driver_residual_pool"],
    "simulation.paired_residual_pool": gate.CONSUMERS["simulation.paired_residual_pool"],
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
    assert ("model_stage.recalibration", "recalibration_map:STR-THRU") in consumers
    for role in ("size", "implied_t1", "runup_move"):
        assert ("model_stage.driver_residual_pool", f"driver_residual_pool:{role}") in consumers
    assert ("simulation.paired_residual_pool", "paired_residual_pool") in consumers
    assert evidence["lineage"] == {"status": "ok", "states": 5, "valid": 5}
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
    assert "analogs.board_analog_matcher" in pending
    assert "model_stage.recalibration" not in pending  # a real probe since 3dea05e
    assert "model_stage.driver_residual_pool" not in pending  # real since 52ef989
    assert "simulation.paired_residual_pool" not in pending
    assert "chooser.admissible_table" in pending  # no v2 consumer reads it yet
    by_id = {r["member_id"]: r for r in evidence["members"]}
    for member_id in ("board_analog_matcher", "recalibration_map:STR-RUNUP",
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


def test_recalibration_map_without_matching_payoff_line_is_unresolved(tmp_path):
    states = _state_payloads()
    states["recalibration_map:STR-THRU"] = {"a": _recal(alpha=0.75)}
    evidence = _run(tmp_path, _release(tmp_path, states=states))

    assert "P5_CONSUMER_UNRESOLVED" in evidence["finding_codes"]
    row = next(f for f in evidence["findings"] if f["code"] == "P5_CONSUMER_UNRESOLVED")
    assert row["subject"] == "model_stage.recalibration"


def test_missing_recalibration_member_is_missing_not_pending(tmp_path):
    states = _state_payloads()
    del states["recalibration_map:STR-THRU"]
    evidence = _run(tmp_path, _release(tmp_path, states=states))

    row = next(r for r in evidence["members"] if r["member_id"] == "recalibration_map:STR-THRU")
    assert row["status"] == "MISSING"
    assert {"P5_MEMBER_MISSING", "P5_CONSUMER_BLOCKED"} <= set(evidence["finding_codes"])


def test_preparer_collects_recalibration_and_payoff_artifacts(tmp_path):
    fold = tmp_path / "train" / "recal" / "folds" / "c1"
    fold.mkdir(parents=True)
    (fold / "recalibration_artifact.json").write_bytes(_recal())
    (fold / "payoff_artifact.json").write_bytes(_state_payloads()["payoff_line:STR-THRU"]["a"])
    recal = prep.recalibration_payloads([tmp_path / "train"])
    payoff = prep.payoff_payloads([tmp_path / "train"])

    assert list(recal) == ["recalibration_map:STR-THRU"]
    assert list(payoff) == ["payoff_line:STR-THRU"]
    (fold / "recalibration_artifact.json").write_bytes(b"{}")
    with pytest.raises(ValueError):
        prep.recalibration_payloads([tmp_path / "train"])


def test_driver_pool_staged_under_the_wrong_role_fails_identity(tmp_path):
    states = _state_payloads()
    states["driver_residual_pool:size"] = {"x": _driver_pool("implied_t1")}
    evidence = _run(tmp_path, _release(tmp_path, states=states))

    assert {"P5_MEMBER_IDENTITY", "P5_CONSUMER_BLOCKED"} <= set(evidence["finding_codes"])


def test_missing_paired_pool_blocks_the_simulation_consumer(tmp_path):
    states = _state_payloads()
    del states["paired_residual_pool"]
    evidence = _run(tmp_path, _release(tmp_path, states=states))

    blocked = [f for f in evidence["findings"] if f["code"] == "P5_CONSUMER_BLOCKED"]
    assert [f["subject"] for f in blocked] == ["simulation.paired_residual_pool"]


def test_unknown_lineage_upstream_fails(tmp_path):
    states = _state_payloads()
    orphan = Lineage(data=LINEAGE.data, upstream=("fold:size:2026-09",))
    states["driver_residual_pool:size"] = {"m-size|champion": _driver_pool("size", lineage=orphan)}
    evidence = _run(tmp_path, _release(tmp_path, states=states))

    assert "P5_LINEAGE_INVALID" in evidence["finding_codes"]
    assert evidence["lineage"]["code"] == "UNKNOWN_UPSTREAM"


def test_lineage_upstream_naming_a_staged_state_passes(tmp_path):
    states = _state_payloads()
    parent = "driver_residual_pool:implied_t1/m-implied_t1|champion"
    child = Lineage(data=LINEAGE.data, upstream=(parent,))
    states["driver_residual_pool:size"] = {"m-size|champion": _driver_pool("size", lineage=child)}
    evidence = _run(tmp_path, _release(tmp_path, states=states))

    assert evidence["lineage"]["status"] == "ok"
    assert evidence["release_ok"] is True


def test_preparer_classifies_prebuilt_frozen_states(tmp_path):
    paths = []
    for name, data in (("d.json", _driver_pool("runup_move")), ("p.json", _paired_pool())):
        (tmp_path / name).write_bytes(data)
        paths.append(tmp_path / name)
    found = prep.frozen_state_payloads(paths)
    assert sorted(found) == ["driver_residual_pool:runup_move", "paired_residual_pool"]


def test_frozen_state_dir_skips_training_job_summaries(tmp_path):
    out = tmp_path / "states"
    out.mkdir()
    (out / "driver_residual_pool__size.json").write_bytes(_driver_pool("size"))
    (out / "driver_residual_pool__size.summary.json").write_text('{"state": "x"}')
    (out / "paired_residual_pool.json").write_bytes(_paired_pool())
    (out / "paired_residual_pool.summary.json").write_text('{"state": "y"}')

    assert [p.name for p in prep.frozen_state_files([out])] == [
        "driver_residual_pool__size.json", "paired_residual_pool.json"]
    assert prep.frozen_state_files([out / "paired_residual_pool.summary.json"]) == []
    found = prep.frozen_state_payloads([out])
    assert sorted(found) == ["driver_residual_pool:size", "paired_residual_pool"]
