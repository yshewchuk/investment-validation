import hashlib
import json
from pathlib import Path

import joblib
import pytest

from checks import phase4_real
from checks.phase4_frozen_bridge import prepare_frozen_replay
from engine.v2.contracts import ScoreRequest
from engine.v2.foundation import content_hash
from engine.v2.scoring.stages import STAGE_NAMES, NativeScoreInputs, receipt
from tools.phase4_frozen_resources import (
    FrozenResourceError,
    package_frozen_resources,
)


class _Estimator:
    def predict(self, rows):
        return [2.0 * float(row[0]) + 1.0 for row in rows]


def _artifact(source: Path, name: str = "model.joblib") -> tuple[Path, str]:
    path = source / name
    joblib.dump(_Estimator(), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, digest


def _binding(path: Path, digest: str, **changes):
    row = {
        "model_id": "size-model-v1",
        "artifact": path.name,
        "artifact_sha256": digest,
        "role": "size",
        "feature_order": ["x"],
        "output_names": ["forecast_abs_move"],
        "strategy": "STR-THRU",
        "decision_clock": "entry-close",
        "adapter": "joblib-estimator.v1",
    }
    row.update(changes)
    return row


def _native_inputs():
    return NativeScoreInputs(
        context={"strategy": "STR-THRU"},
        features={"model_inputs": {"x": 3.0}},
        forecast={},
        geometry=None,
        pricing=None,
        analogs={},
        simulation={},
        gate={},
        chooser={},
        diagnostics={},
        source_ref="sha256:" + "1" * 64,
        stage_receipts=tuple(
            receipt(stage, {"source": "test"}, {"execution": "pending"})
            for stage in STAGE_NAMES
            if stage != "diagnostics"
        ),
    )


def test_package_is_consumable_by_strict_frozen_bridge(tmp_path):
    source = tmp_path / "source"
    release = tmp_path / "release"
    source.mkdir()
    path, digest = _artifact(source)
    package = package_frozen_resources(
        model_bindings=[_binding(path, digest)],
        deployment_id="deployment-1",
        release_root=release,
        source_root=source,
    )

    sidecar_row = package.resource_rows[0]
    assert sidecar_row["kind"] == "sidecar"
    assert content_hash(package.sidecar_document) == sidecar_row["content_hash"]
    for row in package.resource_rows:
        resource_path = release / row["path"]
        assert resource_path.is_file()
        assert row["sha256"] == "sha256:" + hashlib.sha256(resource_path.read_bytes()).hexdigest()
        assert not Path(row["path"]).is_absolute()
        assert ".." not in Path(row["path"]).parts
    verified_documents = phase4_real._verified_resources(release, list(package.resource_rows))
    assert verified_documents[sidecar_row["resource_id"]] == package.sidecar_document

    request = ScoreRequest(
        event_id="event-1",
        event_revision="event-revision-1",
        calendar_revision="calendar-1",
        strategy_version="STR-THRU",
        deployment_id="deployment-1",
        decision_clock_id="entry-close",
        requested_decision_at="2026-09-16",
        snapshot_id="snapshot-1",
        mode="replay",
        fill_model={"alpha": 0.5},
        model_artifact_refs=package.request_refs,
    )
    plan = prepare_frozen_replay(
        release_root=release,
        resource_rows=list(package.resource_rows),
        verified_documents=verified_documents,
        metadata={"frozen_inference": package.trace_declaration},
        request=request,
        inputs=_native_inputs(),
    )

    assert plan is not None
    result = plan.inference.infer(plan.release, plan.requests[0])
    assert result.status == "READY"
    assert result.predictions == ((7.0,),)


def test_package_identity_is_deterministic_and_deduplicates_artifact(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    path, digest = _artifact(source)
    first = _binding(path, digest)
    second = _binding(
        path,
        digest,
        model_id="gate-model-v1",
        role="gate",
        output_names=["gate_score"],
    )
    package_a = package_frozen_resources(
        model_bindings=[first, second],
        deployment_id="deployment-1",
        release_root=tmp_path / "release-a",
        source_root=source,
    )
    package_b = package_frozen_resources(
        model_bindings=[second, first],
        deployment_id="deployment-1",
        release_root=tmp_path / "release-b",
        source_root=source,
    )

    assert package_a.sidecar_document == package_b.sidecar_document
    assert package_a.trace_declaration == package_b.trace_declaration
    assert package_a.request_refs == package_b.request_refs
    assert len([row for row in package_a.resource_rows if row["kind"] == "artifact"]) == 1


def test_legacy_binding_clock_and_role_are_normalized(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    path, digest = _artifact(source)
    package = package_frozen_resources(
        model_bindings=[_binding(
            path, digest, role="abs_move", decision_offset=None,
            decision_clock=None,
        )],
        deployment_id="deployment-1",
        release_root=tmp_path / "release",
        source_root=source,
    )
    binding = package.sidecar_document["bindings"][0]
    assert binding["role"] == "size"
    assert binding["decision_clock_id"] == "legacy.decision_offset.0"


def test_output_whitelists_metadata_and_never_copies_answers(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    path, digest = _artifact(source)
    binding = _binding(
        path,
        digest,
        prediction=999.0,
        predictions=[999.0],
        gate_pass=True,
        threshold=0.6,
    )
    package = package_frozen_resources(
        model_bindings=[binding],
        deployment_id="deployment-1",
        release_root=tmp_path / "release",
        source_root=source,
    )

    encoded = json.dumps(package.sidecar_document, sort_keys=True)
    assert "999" not in encoded
    assert "gate_pass" not in encoded
    assert "threshold" not in encoded
    assert set(package.sidecar_document["bindings"][0]) == {
        "adapter",
        "binding_id",
        "decision_clock_id",
        "feature_order",
        "members",
        "model_id",
        "output_names",
        "request_ref",
        "role",
        "strategy_id",
    }


def test_missing_or_mismatched_artifact_is_refused(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    path, digest = _artifact(source)
    path.unlink()
    with pytest.raises(FrozenResourceError, match="artifact missing"):
        package_frozen_resources(
            model_bindings=[_binding(path, digest)],
            deployment_id="deployment-1",
            release_root=tmp_path / "missing-release",
            source_root=source,
        )

    path, digest = _artifact(source)
    with pytest.raises(FrozenResourceError, match="sha256 mismatch"):
        package_frozen_resources(
            model_bindings=[_binding(path, "0" * 64)],
            deployment_id="deployment-1",
            release_root=tmp_path / "mismatch-release",
            source_root=source,
        )


def test_source_escape_and_conflicting_aliases_are_refused(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    outside, digest = _artifact(tmp_path, "outside.joblib")
    with pytest.raises(FrozenResourceError, match="escapes source root"):
        package_frozen_resources(
            model_bindings=[_binding(outside, digest, artifact="../outside.joblib")],
            deployment_id="deployment-1",
            release_root=tmp_path / "escape-release",
            source_root=source,
        )

    inside, digest = _artifact(source)
    with pytest.raises(FrozenResourceError, match="conflicting artifact and artifact_path"):
        package_frozen_resources(
            model_bindings=[
                _binding(
                    inside,
                    digest,
                    artifact_path="different.joblib",
                )
            ],
            deployment_id="deployment-1",
            release_root=tmp_path / "alias-release",
            source_root=source,
        )


def test_duplicate_bridge_role_is_refused_before_sidecar_publication(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    path, digest = _artifact(source)
    first = _binding(path, digest)
    second = _binding(path, digest, model_id="other", role="size:monthly")

    with pytest.raises(FrozenResourceError, match="duplicate role"):
        package_frozen_resources(
            model_bindings=[first, second],
            deployment_id="deployment-1",
            release_root=tmp_path / "release",
            source_root=source,
        )


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ({"strategy": "STR-RUNUP"}, "mixed strategies"),
        ({"decision_clock": "pre-open"}, "mixed decision clocks"),
    ),
)
def test_package_requires_one_request_scope(tmp_path, change, message):
    source = tmp_path / "source"
    source.mkdir()
    path, digest = _artifact(source)
    first = _binding(path, digest)
    second = _binding(path, digest, model_id="gate-model-v1", role="gate", **change)

    with pytest.raises(FrozenResourceError, match=message):
        package_frozen_resources(
            model_bindings=[first, second],
            deployment_id="deployment-1",
            release_root=tmp_path / "release",
            source_root=source,
        )
