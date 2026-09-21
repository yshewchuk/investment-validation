from __future__ import annotations

import copy
import hashlib
import json

import pytest

from checks.phase4_checkpoints import (
    ASSERTED_UNEXERCISED_BRANCHES,
    REQUIRED_BRANCHES,
    SCHEMA_VERSION,
    CheckpointError,
    load_bundle,
    validate_bundle,
)
from engine.v2.foundation import content_hash


def _hashed(value):
    return {"value": value, "content_hash": content_hash(value)}


def _case(case_id, branches, *, disposition="compared", checkpoints=None,
          executable_inputs="available", first_gap=None):
    request = {
        "event_id": case_id,
        "strategy_version": "STR-THRU",
        "mode": "replay",
    }
    if checkpoints is None:
        checkpoints = {
            "features": _hashed({
                "feature_vector": {"spot": 100.0},
                "missing_mask": {"spot": False},
                "model_identity": {"artifact": "model-a"},
            }),
            "selection_pricing": _hashed({
                "selected_legs": [{"right": "call", "side": "long", "quantity": 1}],
                "entry_cost": 2.5,
            }),
            "simulation": _hashed({
                "horizon": "planned_exit",
                "capital_denominator": 2.5,
                "residual_population_identity": {"ref": "residual-a"},
                "draw_count": 1000,
                "seed": 7,
            }),
            "gate_inputs": _hashed({"entry_cost": 2.5}),
        }
    row = {
        "case_id": case_id,
        "request": request,
        "request_hash": content_hash(request),
        "strategy": "STR-THRU",
        "branches": branches,
        "resource_refs": ["models"],
        "executable_inputs": executable_inputs,
        "disposition": disposition,
        "checkpoints": checkpoints,
        "first_gap": first_gap,
    }
    return {**row, "case_hash": content_hash(row)}


def _resign_case(case):
    case["case_hash"] = content_hash({
        key: value for key, value in case.items() if key != "case_hash"
    })


def _write_case_file(tmp_path, document):
    """Write a case document to <tmp_path>/cases/<case_id>.json and return its pointer."""
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    data = json.dumps(document, sort_keys=True).encode("utf-8")
    (cases_dir / (document["case_id"] + ".json")).write_bytes(data)
    return {
        "case_id": document["case_id"],
        "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
    }


def _bundle(tmp_path, document=None):
    raw = b'{"artifact":"frozen"}\n'
    (tmp_path / "models.json").write_bytes(raw)
    branches = sorted(REQUIRED_BRANCHES)
    if document is None:
        document = _case("case-1", branches)
    pointer = _write_case_file(tmp_path, document)
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "release_id": "phase4-test-release",
        "metadata": {
            "status": "diagnostic_only",
            # `ties` is asserted ABSENT with evidence rather than required
            # as a captured instance (see ASSERTED_UNEXERCISED_BRANCHES):
            # 12 choices had a runner-up, none tied, closest margin 0.017.
            "tie_audit": {"examined": 12, "exercised": 0, "closest": 0.017},
        },
        "resources": [{
            "resource_id": "models",
            "path": "models.json",
            "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        }],
        "coverage": {
            "strategies": {"STR-THRU": ["case-1"]},
            "branches": {branch: ["case-1"] for branch in branches},
        },
        "cases": [pointer],
    }
    return {**bundle, "manifest_hash": content_hash(bundle)}


def _resign_bundle(bundle):
    bundle["manifest_hash"] = content_hash({
        key: value for key, value in bundle.items() if key != "manifest_hash"
    })


def test_valid_bundle_has_shared_resources_and_complete_coverage(tmp_path):
    verified = validate_bundle(_bundle(tmp_path), tmp_path)
    assert verified["case_ids"] == ("case-1",)
    assert verified["strategies"] == ("STR-THRU",)
    assert set(verified["branches"]) == REQUIRED_BRANCHES


def test_load_bundle_resolves_case_pointers_from_release_root(tmp_path):
    bundle = _bundle(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps(bundle), encoding="utf-8")
    verified = load_bundle(tmp_path / "manifest.json")
    assert verified["case_ids"] == ("case-1",)


@pytest.mark.parametrize("field", sorted({
    "schema_version", "release_id", "metadata", "resources", "coverage",
    "cases", "manifest_hash",
}))
def test_missing_required_top_level_field_is_rejected(tmp_path, field):
    bundle = _bundle(tmp_path)
    bundle.pop(field)
    with pytest.raises(CheckpointError, match="unexpected or missing fields"):
        validate_bundle(bundle, tmp_path)


@pytest.mark.parametrize("mutation, message", [
    (
        lambda bundle: bundle["resources"][0].update({"path": "../models.json"}),
        "escapes release root",
    ),
    (
        lambda bundle: bundle["resources"][0].update({"sha256": "sha256:" + "0" * 64}),
        "sha256: mismatch",
    ),
])
def test_resource_path_and_hash_corruption_are_rejected(tmp_path, mutation, message):
    bundle = _bundle(tmp_path)
    mutation(bundle)
    _resign_bundle(bundle)
    with pytest.raises(CheckpointError, match=message):
        validate_bundle(bundle, tmp_path)


def test_case_pointer_wrong_sha256_is_rejected(tmp_path):
    bundle = _bundle(tmp_path)
    bundle["cases"][0]["sha256"] = "sha256:" + "0" * 64
    _resign_bundle(bundle)
    with pytest.raises(CheckpointError, match="sha256: mismatch"):
        validate_bundle(bundle, tmp_path)


@pytest.mark.parametrize("bad_case_id", ["../models", "..", "nested/case", "", "."])
def test_case_pointer_path_traversal_case_id_is_rejected(tmp_path, bad_case_id):
    bundle = _bundle(tmp_path)
    bundle["cases"][0]["case_id"] = bad_case_id
    _resign_bundle(bundle)
    with pytest.raises(CheckpointError, match="expected a safe identifier|expected nonempty string"):
        validate_bundle(bundle, tmp_path)


def test_case_pointer_missing_file_is_rejected(tmp_path):
    bundle = _bundle(tmp_path)
    bundle["cases"][0]["case_id"] = "no-such-case"
    _resign_bundle(bundle)
    with pytest.raises(CheckpointError, match="missing case file"):
        validate_bundle(bundle, tmp_path)


def test_duplicate_case_id_is_rejected(tmp_path):
    bundle = _bundle(tmp_path)
    bundle["cases"].append(dict(bundle["cases"][0]))
    _resign_bundle(bundle)
    with pytest.raises(CheckpointError, match="case_id.*duplicate"):
        validate_bundle(bundle, tmp_path)


def test_missing_checkpoint_group_is_rejected(tmp_path):
    branches = sorted(REQUIRED_BRANCHES)
    document = _case("case-1", branches)
    document["checkpoints"].pop("gate_inputs")
    _resign_case(document)
    bundle = _bundle(tmp_path, document=document)
    with pytest.raises(CheckpointError, match="missing required group"):
        validate_bundle(bundle, tmp_path)


def test_incomparable_case_missing_required_groups_needs_first_gap(tmp_path):
    """2026-09-21 (coordinator): a non-``compared`` case may be missing
    required groups, but only when it explains itself -- no ``first_gap``
    with missing groups is still rejected, the same as a fabricated group
    would be."""
    branches = sorted(REQUIRED_BRANCHES)
    checkpoints = {"gate_inputs": _hashed({"entry_cost": 2.5})}
    document = _case(
        "case-1", branches, disposition="incomparable", checkpoints=checkpoints,
        first_gap=None,
    )
    bundle = _bundle(tmp_path, document=document)
    with pytest.raises(CheckpointError, match="expected stage and reason"):
        validate_bundle(bundle, tmp_path)


def test_incomparable_case_with_first_gap_and_partial_groups_is_accepted(tmp_path):
    """The relaxed side of the same contract change: a genuinely partial,
    non-``compared`` case that DOES explain where it stopped validates,
    carrying only the groups it actually reached."""
    branches = sorted(REQUIRED_BRANCHES)
    checkpoints = {"gate_inputs": _hashed({"entry_cost": 2.5})}
    document = _case(
        "case-1", branches, disposition="refused_as_expected", checkpoints=checkpoints,
        first_gap={"stage": "resolve_context", "reason": "superseded strategy"},
    )
    bundle = _bundle(tmp_path, document=document)
    verified = validate_bundle(bundle, tmp_path)
    assert verified["case_ids"] == ("case-1",)


def test_complete_case_must_not_carry_a_first_gap(tmp_path):
    """A case with all four required groups has nothing to explain; a
    stray ``first_gap`` there would be a lie about where it stopped."""
    branches = sorted(REQUIRED_BRANCHES)
    document = _case(
        "case-1", branches, disposition="incomparable",
        first_gap={"stage": "resolve_context", "reason": "superseded strategy"},
    )
    bundle = _bundle(tmp_path, document=document)
    with pytest.raises(CheckpointError, match="must be null"):
        validate_bundle(bundle, tmp_path)


def test_compared_case_still_requires_all_four_groups_even_with_first_gap(tmp_path):
    """Coordinator: keep FULL strictness for ``compared``. Supplying a
    ``first_gap`` does not buy a ``compared`` case an exemption from the
    four-group requirement -- that relaxation is for non-``compared``
    dispositions only."""
    branches = sorted(REQUIRED_BRANCHES)
    document = _case("case-1", branches)
    document["checkpoints"].pop("gate_inputs")
    document["first_gap"] = {"stage": "resolve_context", "reason": "superseded strategy"}
    _resign_case(document)
    bundle = _bundle(tmp_path, document=document)
    with pytest.raises(CheckpointError, match="missing required group"):
        validate_bundle(bundle, tmp_path)


def test_incomplete_declared_coverage_is_rejected(tmp_path):
    bundle = _bundle(tmp_path)
    bundle["coverage"]["branches"].pop("multi_expiry")
    _resign_bundle(bundle)
    with pytest.raises(CheckpointError, match="incomplete critical branches"):
        validate_bundle(bundle, tmp_path)


def test_nonfinite_checkpoint_value_is_rejected(tmp_path):
    branches = sorted(REQUIRED_BRANCHES)
    document = _case("case-1", branches)
    checkpoint = document["checkpoints"]["features"]
    checkpoint["value"] = copy.deepcopy(checkpoint["value"])
    checkpoint["value"]["feature_vector"]["spot"] = float("nan")
    checkpoint["content_hash"] = content_hash(checkpoint["value"])
    _resign_case(document)
    bundle = _bundle(tmp_path, document=document)
    with pytest.raises(CheckpointError, match="non-finite value"):
        validate_bundle(bundle, tmp_path)


def test_missing_executable_inputs_cannot_be_success(tmp_path):
    branches = sorted(REQUIRED_BRANCHES)
    document = _case("case-1", branches)
    document["executable_inputs"] = "missing"
    _resign_case(document)
    bundle = _bundle(tmp_path, document=document)
    with pytest.raises(CheckpointError, match="must be incomparable"):
        validate_bundle(bundle, tmp_path)


def test_load_bundle_rejects_malformed_json(tmp_path):
    path = tmp_path / "checkpoints.json"
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(CheckpointError, match="invalid JSON"):
        load_bundle(path)


# --------------------------------------------------------------------------
# ties: asserted absent, with evidence (user decision, 2026-09-21)
# --------------------------------------------------------------------------


def test_ties_is_asserted_absent_and_is_not_a_required_branch():
    """The contract moved from "show me a tie" to "prove none happened".

    Without this the two constants could drift back together and `ties`
    would silently become a coverage requirement again -- the incoherent
    demand this decision removed (a path neither implementation takes
    cannot diverge).
    """
    assert "ties" not in REQUIRED_BRANCHES
    assert "ties" in ASSERTED_UNEXERCISED_BRANCHES
    assert not (REQUIRED_BRANCHES & ASSERTED_UNEXERCISED_BRANCHES)


def test_a_bundle_without_a_tie_audit_is_rejected(tmp_path):
    """An unmentioned tie is ambiguous between "none happened", "one
    happened and was not recorded" and "nobody looked". Dropping `ties`
    from the required set without this check would accept all three."""
    bundle = _bundle(tmp_path)
    del bundle["metadata"]["tie_audit"]
    _resign_bundle(bundle)
    with pytest.raises(CheckpointError, match="tie_audit"):
        validate_bundle(bundle, tmp_path)


def test_a_tie_audit_that_examined_nothing_is_rejected(tmp_path):
    """`examined == 0` is an assertion from zero observations -- exactly
    the absence-of-evidence move the audit exists to stop."""
    bundle = _bundle(tmp_path)
    bundle["metadata"]["tie_audit"] = {
        "examined": 0, "exercised": 0, "closest": None,
    }
    _resign_bundle(bundle)
    with pytest.raises(CheckpointError, match="absence of evidence"):
        validate_bundle(bundle, tmp_path)


def test_a_tie_that_really_happened_must_be_covered_by_a_case(tmp_path):
    """If the tie path DOES start executing (a score discretised or
    rounded upstream), the audit must fail loudly rather than pass with a
    nonzero count nobody captured."""
    bundle = _bundle(tmp_path)
    bundle["metadata"]["tie_audit"] = {
        "examined": 12, "exercised": 1, "closest": 0.0,
    }
    _resign_bundle(bundle)
    with pytest.raises(CheckpointError, match="no case covers it"):
        validate_bundle(bundle, tmp_path)


def test_a_zero_margin_the_audit_did_not_count_is_rejected(tmp_path):
    """`closest == 0.0` with `exercised == 0` is self-contradictory: a
    zero margin IS a tie. Without this the producer could report the
    margin honestly and the count wrongly and still pass."""
    bundle = _bundle(tmp_path)
    bundle["metadata"]["tie_audit"] = {
        "examined": 12, "exercised": 0, "closest": 0.0,
    }
    _resign_bundle(bundle)
    with pytest.raises(CheckpointError, match="did not count"):
        validate_bundle(bundle, tmp_path)
