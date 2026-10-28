"""Pure chooser metadata validation through constructors and typed decoding."""
from dataclasses import FrozenInstanceError
from datetime import date

import pytest

from engine.v2.foundation import DocumentError, content_hash, from_document, to_document
from engine.v2.models import ArtifactMember
from engine.v2.models.serving_folds import ChooserFoldPolicy, ChooserServingFoldDescriptor


OUTPUTS = {
    "implied_t1": ("driver_prediction", "pred_im_t1_d14"),
    "runup_move": ("pred_runup_abs_move_d14", "pred_runup_abs_move_d14"),
}


class _Count(int):
    """An int subclass must not satisfy the exact pool-count contract."""


def _policy_document(role):
    return {
        "schema_version": "chooser_fold_policy.v1.0", "role": role,
        "model_id": f"synthetic-{role}", "feature_order": ["zeta", "alpha", "middle"],
        "panel_sha256": "a" * 64, "interval_floor": 0.0,
        "interval_policy": "chooser-heldout-deciles.v1",
    }


def _descriptor_document(role):
    return {
        "schema_version": "chooser_serving_fold_descriptor.v1.0",
        "parent_release_id": "synthetic-parent", "parent_release_hash": "sha256:" + "b" * 64,
        "policy": _policy_document(role), "fold_start": "2026-02-01",
        "decision_clock_id": "entry-close", "pool_count": 2,
        "output_name": OUTPUTS[role][0], "producer_output": OUTPUTS[role][1],
        "pool_pred_field": "pool_pred", "pool_res_field": "pool_res",
        "training_cutoff": None,
        "estimator": {"schema_version": "artifact_member.v1.0", "name": "estimator",
                      "path": "objects/" + "c" * 64, "content_hash": "sha256:" + "c" * 64},
    }


def _construct_descriptor(document):
    return ChooserServingFoldDescriptor(**{
        **document, "policy": ChooserFoldPolicy(**document["policy"]),
        "estimator": ArtifactMember(**document["estimator"]),
    })


def _assert_error(error, code, path, message, *, cause_path=None):
    assert type(error) is DocumentError
    assert (error.code, error.path, error.message) == (code, path, message)
    assert str(error) == f"{code} at {path}: {message}"
    if cause_path is None:
        assert error.__cause__ is None
    else:
        assert type(error.__cause__) is DocumentError
        assert (error.__cause__.code, error.__cause__.path, error.__cause__.message) == (
            code, cause_path, message)


@pytest.mark.parametrize("role", OUTPUTS)
@pytest.mark.parametrize("route", ["constructor", "document", "nested"])
@pytest.mark.parametrize("field,value,code,message", [
    ("feature_order", [], "BAD_FEATURE_ORDER", "expected nonempty unique feature names"),
    ("feature_order", (), "BAD_FEATURE_ORDER", "expected nonempty unique feature names"),
    ("feature_order", ["x", "x"], "BAD_FEATURE_ORDER", "expected nonempty unique feature names"),
    ("feature_order", ["x", "y", "x"], "BAD_FEATURE_ORDER", "expected nonempty unique feature names"),
    ("feature_order", [""], "BAD_FEATURE_ORDER", "expected nonempty unique feature names"),
    ("feature_order", ["x", ""], "BAD_FEATURE_ORDER", "expected nonempty unique feature names"),
    ("panel_sha256", "a" * 63, "BAD_PANEL_HASH", "expected 64 lowercase hex digits"),
    ("panel_sha256", "A" * 64, "BAD_PANEL_HASH", "expected 64 lowercase hex digits"),
])
def test_policy_semantic_refusals_keep_exact_paths(role, route, field, value, code, message):
    document = _policy_document(role)
    document[field] = value
    with pytest.raises(DocumentError) as caught:
        if route == "constructor":
            ChooserFoldPolicy(**document)
        elif route == "document":
            from_document(ChooserFoldPolicy, document)
        else:
            fold = _descriptor_document(role)
            fold["policy"] = document
            from_document(ChooserServingFoldDescriptor, fold, path="$.folds[1]")
    path = f"$.folds[1].policy.{field}" if route == "nested" else f"$.{field}"
    _assert_error(caught.value, code, path, message,
                  cause_path=f"$.{field}" if route == "nested" else None)


@pytest.mark.parametrize("role", OUTPUTS)
@pytest.mark.parametrize("field,value,code,message", [
    ("feature_order", "feature", "BAD_FEATURE_ORDER", "expected nonempty unique feature names"),
    ("feature_order", None, "BAD_FEATURE_ORDER", "expected nonempty unique feature names"),
    ("feature_order", {"x": 1}, "BAD_FEATURE_ORDER", "expected nonempty unique feature names"),
    ("feature_order", ["x", 1], "BAD_FEATURE_ORDER", "expected nonempty unique feature names"),
    ("feature_order", ["x", None], "BAD_FEATURE_ORDER", "expected nonempty unique feature names"),
    ("feature_order", ["x", []], "BAD_FEATURE_ORDER", "expected nonempty unique feature names"),
    ("role", "size", "BAD_CHOOSER_ROLE", "unsupported chooser fold role"),
    ("role", None, "BAD_CHOOSER_ROLE", "unsupported chooser fold role"),
    ("interval_policy", "size-heldout-deciles.v1", "BAD_INTERVAL_POLICY", "unsupported chooser interval policy"),
    ("interval_policy", None, "BAD_INTERVAL_POLICY", "unsupported chooser interval policy"),
    ("interval_floor", True, "BAD_INTERVAL_FLOOR", "expected a finite number or null"),
    ("interval_floor", "0.0", "BAD_INTERVAL_FLOOR", "expected a finite number or null"),
    ("interval_floor", [], "BAD_INTERVAL_FLOOR", "expected a finite number or null"),
    ("interval_floor", float("nan"), "BAD_INTERVAL_FLOOR", "expected a finite number or null"),
    ("interval_floor", float("inf"), "BAD_INTERVAL_FLOOR", "expected a finite number or null"),
    ("interval_floor", -float("inf"), "BAD_INTERVAL_FLOOR", "expected a finite number or null"),
    pytest.param("interval_floor", 10**400, "BAD_INTERVAL_FLOOR", "expected a finite number or null",
                 id="floor-overflows-finite-check"),
])
def test_policy_constructor_rejects_invalid_values(role, field, value, code, message):
    with pytest.raises(DocumentError) as caught:
        ChooserFoldPolicy(**{**_policy_document(role), field: value})
    _assert_error(caught.value, code, f"$.{field}", message)


@pytest.mark.parametrize("role", OUTPUTS)
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("field,value,code,suffix,message", [
    ("feature_order", "feature", "BAD_TYPE", "feature_order", "expected an array"),
    ("feature_order", None, "BAD_TYPE", "feature_order", "expected an array"),
    ("feature_order", {"x": 1}, "BAD_TYPE", "feature_order", "expected an array"),
    ("feature_order", ["x", 1], "BAD_TYPE", "feature_order[1]", "expected str"),
    ("feature_order", ["x", None], "BAD_TYPE", "feature_order[1]", "expected str"),
    ("feature_order", ["x", []], "BAD_TYPE", "feature_order[1]", "expected str"),
    ("role", "size", "BAD_ENUM", "role", "not one of the declared values"),
    ("role", None, "BAD_ENUM", "role", "not one of the declared values"),
    ("interval_policy", "size-heldout-deciles.v1", "BAD_ENUM", "interval_policy", "not one of the declared values"),
    ("interval_policy", None, "BAD_ENUM", "interval_policy", "not one of the declared values"),
    ("interval_floor", True, "BAD_TYPE", "interval_floor", "expected float"),
    ("interval_floor", "0.0", "BAD_TYPE", "interval_floor", "expected float"),
    ("interval_floor", [], "BAD_TYPE", "interval_floor", "expected float"),
    ("interval_floor", float("nan"), "BAD_TYPE", "interval_floor", "non-finite numbers are not JSON"),
    ("interval_floor", float("inf"), "BAD_TYPE", "interval_floor", "non-finite numbers are not JSON"),
    ("interval_floor", -float("inf"), "BAD_TYPE", "interval_floor", "non-finite numbers are not JSON"),
])
def test_policy_decoder_keeps_earlier_type_and_enum_refusals(role, nested, field, value, code, suffix, message):
    document = {**_policy_document(role), field: value}
    with pytest.raises(DocumentError) as caught:
        if nested:
            fold = {**_descriptor_document(role), "policy": document}
            from_document(ChooserServingFoldDescriptor, fold, path="$.folds[1]")
        else:
            from_document(ChooserFoldPolicy, document)
    prefix = "$.folds[1].policy" if nested else "$"
    _assert_error(caught.value, code, f"{prefix}.{suffix}", message)


@pytest.mark.parametrize("role", OUTPUTS)
@pytest.mark.parametrize("route", ["constructor", "document", "nested"])
@pytest.mark.parametrize("field,value,code,message", [
    *[("fold_start", value, "BAD_FOLD_MONTH", "expected canonical ISO month start") for value in (
        "", "2026-02-02", "2026-02-28", "2026-02-30", "2026-13-01", "2026-2-1",
        "20260201", "2026-W05-7", "2026-02-01T00:00:00", "2026-02-01\n")],
    ("pool_count", -1, "BAD_POOL_COUNT", "expected a nonnegative integer"),
    ("pool_count", _Count(1), "BAD_POOL_COUNT", "expected a nonnegative integer"),
    *[("parent_release_hash", value, "BAD_PARENT_HASH", "expected canonical SHA-256 content hash") for value in (
        "", "b" * 64, "SHA256:" + "b" * 64, "sha256:" + "B" * 64,
        "sha256:" + "g" * 64, "sha256:" + "b" * 63, "sha256:" + "b" * 65,
        "sha256:" + "b" * 64 + "\n")],
    ("decision_clock_id", "", "BAD_FOLD_CLOCK", "expected a nonempty clock identity"),
])
def test_descriptor_semantic_refusals_keep_exact_paths(role, route, field, value, code, message):
    document = {**_descriptor_document(role), field: value}
    with pytest.raises(DocumentError) as caught:
        if route == "constructor":
            _construct_descriptor(document)
        else:
            from_document(ChooserServingFoldDescriptor, document,
                          path="$.folds[1]" if route == "nested" else "$")
    path = f"$.folds[1].{field}" if route == "nested" else f"$.{field}"
    _assert_error(caught.value, code, path, message,
                  cause_path=f"$.{field}" if route == "nested" else None)


@pytest.mark.parametrize("role", OUTPUTS)
@pytest.mark.parametrize("field,value,code,message,type_message", [
    ("fold_start", None, "BAD_FOLD_MONTH", "expected canonical ISO month start", "expected str"),
    ("fold_start", date(2026, 2, 1), "BAD_FOLD_MONTH", "expected canonical ISO month start", "expected str"),
    *[("pool_count", value, "BAD_POOL_COUNT", "expected a nonnegative integer", "expected int")
      for value in (True, False, 0.0, 1.0, "1", None)],
    ("parent_release_hash", None, "BAD_PARENT_HASH", "expected canonical SHA-256 content hash", "expected str"),
    ("parent_release_hash", 123, "BAD_PARENT_HASH", "expected canonical SHA-256 content hash", "expected str"),
    ("decision_clock_id", None, "BAD_FOLD_CLOCK", "expected a nonempty clock identity", "expected str"),
    ("decision_clock_id", 123, "BAD_FOLD_CLOCK", "expected a nonempty clock identity", "expected str"),
])
def test_descriptor_constructor_and_decoder_keep_distinct_type_refusals(
        role, field, value, code, message, type_message):
    document = {**_descriptor_document(role), field: value}
    with pytest.raises(DocumentError) as direct:
        _construct_descriptor(document)
    _assert_error(direct.value, code, f"$.{field}", message)
    with pytest.raises(DocumentError) as decoded:
        from_document(ChooserServingFoldDescriptor, document, path="$.folds[1]")
    _assert_error(decoded.value, "BAD_TYPE", f"$.folds[1].{field}", type_message)


@pytest.mark.parametrize("role", OUTPUTS)
@pytest.mark.parametrize("features", [["single"], ("zeta", "alpha", "middle"), [" raw ", "alpha"], ["A", "a"]])
def test_policy_preserves_valid_feature_names_and_order(role, features):
    document = {**_policy_document(role), "feature_order": features}
    for policy in (ChooserFoldPolicy(**document), from_document(ChooserFoldPolicy, document)):
        assert type(policy.feature_order) is tuple
        assert policy.feature_order == tuple(features)
        assert to_document(policy) == {**document, "feature_order": list(features)}


@pytest.mark.parametrize("role", OUTPUTS)
@pytest.mark.parametrize("floor", [None, -1.25, 0, 2, 1.25])
@pytest.mark.parametrize("count,month,clock", [
    (0, "0001-01-01", "entry-close"), (1, "2024-02-01", "next-open"),
    (57, "9999-12-01", "custom-clock-v2"),
])
def test_valid_metadata_preserves_values_order_and_complete_identity(role, floor, count, month, clock):
    expected = _descriptor_document(role)
    expected.update(pool_count=count, fold_start=month, decision_clock_id=clock)
    expected["policy"]["interval_floor"] = floor
    constructed = _construct_descriptor(expected)
    decoded = from_document(ChooserServingFoldDescriptor, expected, path="$.folds[1]")
    for descriptor in (constructed, decoded):
        assert descriptor == constructed
        assert descriptor.policy.feature_order == ("zeta", "alpha", "middle")
        assert type(descriptor.policy.feature_order) is tuple
        assert type(descriptor.pool_count) is int
        assert to_document(descriptor) == expected
        assert descriptor.descriptor_hash == content_hash(expected)
    assert constructed.policy.interval_floor is floor
    if floor is None:
        assert decoded.policy.interval_floor is None
    else:
        assert type(decoded.policy.interval_floor) is float
    expected["policy"]["feature_order"].reverse()
    assert constructed.policy.feature_order == decoded.policy.feature_order == ("zeta", "alpha", "middle")
    with pytest.raises(FrozenInstanceError):
        constructed.policy.feature_order = ("changed",)
    with pytest.raises(FrozenInstanceError):
        constructed.pool_count = 3


@pytest.mark.parametrize("role", OUTPUTS)
@pytest.mark.parametrize("field", [
    "parent_release_id", "parent_release_hash", "model_id", "feature_order", "panel_sha256",
    "interval_floor", "role", "fold_start", "decision_clock_id", "pool_count", "estimator", "estimator_name",
])
def test_descriptor_hash_binds_every_variable_identity_field(role, field):
    original = _construct_descriptor(_descriptor_document(role))
    document = _descriptor_document(role)
    policy_changes = {"model_id": "another-model", "feature_order": ["middle", "alpha", "zeta"],
                      "panel_sha256": "d" * 64, "interval_floor": 1.25}
    descriptor_changes = {"parent_release_id": "another-parent", "parent_release_hash": "sha256:" + "e" * 64,
                          "fold_start": "2026-03-01", "decision_clock_id": "next-open", "pool_count": 3}
    if field in policy_changes:
        document["policy"][field] = policy_changes[field]
    elif field in descriptor_changes:
        document[field] = descriptor_changes[field]
    elif field == "role":
        other = next(other for other in OUTPUTS if other != role)
        document["policy"]["role"] = other
        document.update(output_name=OUTPUTS[other][0], producer_output=OUTPUTS[other][1])
    elif field == "estimator_name":
        document["estimator"]["name"] = "another-estimator"
    else:
        document["estimator"].update(content_hash="sha256:" + "f" * 64, path="objects/" + "f" * 64)
    for changed in (_construct_descriptor(document), from_document(ChooserServingFoldDescriptor, document)):
        assert to_document(changed) == document
        assert changed.descriptor_hash == content_hash(document)
        assert changed.descriptor_hash != original.descriptor_hash
