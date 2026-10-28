"""Verify native registration evidence owned by the existing catalog reservation."""
import json
import re
from datetime import date

from engine.v2.contracts import ArtifactRef, SnapshotRef
from engine.v2.foundation import (
    ArtifactError,
    artifact_reference,
    canonical_json,
    content_hash,
    from_document,
    safe_relative_path,
    to_document,
)
from engine.v2.ops.catalog import decode_bounded_text, sql_text_budget
from engine.v2.ops.errors import OpsError, fail

MAX_NATIVE_EVIDENCE_BYTES = 16 * 1024 * 1024
MAX_NATIVE_SIZE_ARTIFACT_BYTES = 1024 * 1024 * 1024


def read_native_evidence(conn, run_id, *, max_bytes=MAX_NATIVE_EVIDENCE_BYTES):
    """Return verifier metadata and bounded object evidence from one SQL row."""
    encoding, limit = sql_text_budget(conn, max_bytes)
    row = conn.execute(
        "SELECT e.run_id, e.spec_hash, e.mode, e.input_hash, "
        "CASE WHEN typeof(e.evidence_json)='text' AND length(CAST(e.evidence_json AS BLOB))<=? "
        "THEN CAST(e.evidence_json AS BLOB) END AS evidence_json, "
        "h.input_hash AS registered_input, h.payload_hash AS registered_payload, "
        "CASE WHEN h.payload_json='{}' THEN '{}' END AS registered_document FROM experiment_runs e "
        "LEFT JOIN hypotheses h ON h.run_id=e.run_id AND h.spec_hash=e.spec_hash "
        "WHERE e.run_id=?", (limit, run_id)).fetchone()
    if row is None:
        return None
    try:
        payload = decode_bounded_text(row["evidence_json"], encoding, max_bytes)
        evidence = json.loads(payload)
        if not isinstance(evidence, dict):
            raise ValueError
        native_evidence_json(evidence, max_bytes=max_bytes)
    except (ValueError, TypeError, RecursionError):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "native evidence requires bounded JSON object text") from None
    return row, evidence


def native_evidence_json(document, *, max_bytes=MAX_NATIVE_EVIDENCE_BYTES):
    """Encode canonical object evidence within the native UTF-8 byte ceiling."""
    try:
        if not isinstance(document, dict):
            raise ValueError
        payload = canonical_json(document)
        if len(payload) > max_bytes or len(payload.encode("utf-8")) > max_bytes:
            raise ValueError
        return payload
    except (ValueError, TypeError, RecursionError):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "native evidence exceeds its canonical object contract") from None


def verified_registration_bytes(conn, store, run_id, *, expected_ref=None, max_evidence_bytes=None,
                                 verified_size_rows=None):
    """Return exact verified registration bytes, or None for an absent reservation."""
    if max_evidence_bytes is None:
        row = conn.execute(
            "SELECT e.*, h.input_hash AS registered_input, h.payload_hash AS registered_payload, "
            "h.payload_json AS registered_document FROM experiment_runs e "
            "LEFT JOIN hypotheses h ON h.run_id=e.run_id AND h.spec_hash=e.spec_hash "
            "WHERE e.run_id=?", (run_id,)).fetchone()
    else:
        saved = read_native_evidence(conn, run_id, max_bytes=max_evidence_bytes)
        row, evidence = (None, None) if saved is None else saved
    if row is None:
        return None
    try:
        if max_evidence_bytes is None:
            evidence = json.loads(row["evidence_json"])
        ref = from_document(ArtifactRef, evidence["native_registration"])
        if (ref.schema_ref not in ("native_experiment_registration.v1.0", "native_experiment_registration.v1.1",
                                  "native_experiment_registration.v1.2", "native_experiment_registration.v1.3",
                                  "native_experiment_registration.v1.4", "native_experiment_registration.v1.5",
                                  "native_experiment_registration.v1.6",
                                  "native_experiment_registration.v1.7")
                or (expected_ref is not None and ref != expected_ref)):
            raise ValueError
        if max_evidence_bytes is not None and ref.byte_size > max_evidence_bytes:
            raise fail("RESOURCE_LIMIT_EXCEEDED", "native registration exceeds its UTF-8 byte budget",
                       details={"registration_bytes": ref.byte_size, "max_registration_bytes": max_evidence_bytes})
        binding = store.read_verified(ref)
        document = json.loads(binding)
        _check_size_artifact_budget(document)
        verified_size_rows = _size_verification_cache(document, verified_size_rows)
        valid = _matches(row, evidence, ref, binding, document, store, verified_size_rows)
    except (ArtifactError, ValueError, TypeError, KeyError):
        valid = False
    except RecursionError:
        if max_evidence_bytes is None:
            raise
        valid = False
    if not valid:
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "native registration differs from its immutable binding")
    return binding


def _size_verification_cache(document, verified_size_rows):
    if (verified_size_rows is None and isinstance(document, dict)
            and document.get("schema_version") == "native_experiment_registration.v1.7"):
        return {}
    return verified_size_rows


def _size_artifact_byte_total(document):
    """Sum v1.7 shard bytes; malformed shapes surface as identity conflicts."""
    try:
        return sum(
            from_document(ArtifactRef, shard["ref"]).byte_size
            for descriptor in document["size_artifacts"].values()
            for shard in descriptor["entry"]["shards"])
    except (ArtifactError, ValueError, TypeError, KeyError, AttributeError):
        raise ValueError("malformed size artifact descriptor") from None


def _check_size_artifact_budget(document):
    """Enforce the v1.7 aggregate size artifact byte cap."""
    if not isinstance(document, dict) or document.get("schema_version") != "native_experiment_registration.v1.7":
        return
    size_artifact_bytes = _size_artifact_byte_total(document)
    if size_artifact_bytes > MAX_NATIVE_SIZE_ARTIFACT_BYTES:
        raise fail("RESOURCE_LIMIT_EXCEEDED", "native registration size artifacts exceed their byte budget",
                   details={"size_artifact_bytes": size_artifact_bytes,
                            "max_size_artifact_bytes": MAX_NATIVE_SIZE_ARTIFACT_BYTES})


def _size_artifacts_verified(store, document, verified_size_rows=None):
    """Hash-check each v1.7 shard independently without assembling section JSON."""
    if document.get("schema_version") != "native_experiment_registration.v1.7":
        return True
    try:
        artifacts = document["size_artifacts"]
        for artifact_id, descriptor in artifacts.items():
            if not _size_artifact_verified(store, descriptor, verified_size_rows):
                return False
        return True
    except (ArtifactError, ValueError, TypeError, KeyError, AttributeError, RecursionError):
        return False


def _size_artifact_descriptor_parts(descriptor):
    if (not isinstance(descriptor, dict)
            or set(descriptor) != {"kind", "section", "entry"}
            or descriptor["kind"] not in ("list", "map", "object")
            or not isinstance(descriptor["section"], str)):
        return None
    entry = descriptor["entry"]
    if (not isinstance(entry, dict) or set(entry) != {"row_count", "shards"}
            or type(entry["row_count"]) is not int or entry["row_count"] < 0
            or not isinstance(entry["shards"], list)):
        return None
    return descriptor["kind"], descriptor["section"], entry


def _size_artifact_boundaries_valid(shard, position):
    first_key, last_key = shard["first_key"], shard["last_key"]
    if (not isinstance(first_key, list) or len(first_key) != 1
            or type(first_key[0]) is not int):
        return False
    if (not isinstance(last_key, list) or len(last_key) != 1
            or type(last_key[0]) is not int):
        return False
    return first_key == [position] and last_key == [position + shard["rows"] - 1]


def _size_artifact_shard_ref(shard, section, position):
    if (not isinstance(shard, dict)
            or set(shard) != {"ref", "rows", "first_key", "last_key"}
            or type(shard["rows"]) is not int or shard["rows"] <= 0):
        return None
    ref = from_document(ArtifactRef, shard["ref"])
    if (ref.schema_ref != "magnitude_search_shard.v1"
            or type(ref.byte_size) is not int or not 0 < ref.byte_size <= 8 * 1024 * 1024):
        return None
    safe_relative_path(ref.storage_key)
    if not _size_artifact_boundaries_valid(shard, position):
        return None
    return ref


def _size_artifact_rows_valid(rows, position, kind, map_names):
    for offset, row in enumerate(rows):
        if (not isinstance(row, dict) or set(row) != {"position", "row"}
                or type(row["position"]) is not int
                or row["position"] != position + offset):
            return False
        if kind == "map":
            value = row["row"]
            if (not isinstance(value, list) or len(value) != 2
                    or not isinstance(value[0], str) or value[0] in map_names):
                return False
            map_names.add(value[0])
    return True


def _size_artifact_total_valid(kind, entry, position):
    if position != entry["row_count"]:
        return False
    return kind != "object" or position == 1


def _size_artifact_verified(store, descriptor, verified_size_rows=None):
    """Verify one artifact descriptor's shards, row totals, and object cardinality."""
    try:
        parts = _size_artifact_descriptor_parts(descriptor)
        if parts is None:
            return False
        kind, section, entry = parts
        position = 0
        map_names = set()
        for shard in entry["shards"]:
            ref = _size_artifact_shard_ref(shard, section, position)
            if ref is None:
                return False
            stored = json.loads(store.read_verified(ref))
            if (not isinstance(stored, dict)
                    or set(stored) != {"schema_version", "section", "rows"}
                    or stored["schema_version"] != "magnitude_search_shard.v1"
                    or stored["section"] != section):
                return False
            rows = stored["rows"]
            if (not isinstance(rows, list) or len(rows) != shard["rows"]
                    or not _size_artifact_rows_valid(rows, position, kind, map_names)):
                return False
            if verified_size_rows is not None:
                verified_size_rows[(section, ref.schema_ref, ref.storage_key, ref.content_hash,
                                    ref.byte_size)] = rows
            position += shard["rows"]
        if not _size_artifact_total_valid(kind, entry, position):
            return False
        return True
    except (ArtifactError, ValueError, TypeError, KeyError, AttributeError, RecursionError):
        return False


def _matches(row, evidence, ref, binding, document, store, verified_size_rows=None):
    variant = content_hash(document)
    return (_magnitude_binding(document) and _size_artifacts_verified(store, document, verified_size_rows)
            and (document["schema_version"] != "native_experiment_registration.v1.7"
                 or _magnitude_binding(document, verified_size_rows))
            and document["schema_version"] == ref.schema_ref
            and canonical_json(document).encode() == binding
            and artifact_reference(binding, ref.schema_ref) == ref
            and row["spec_hash"] == document["spec_hash"] and row["mode"] == "primary"
            and row["input_hash"] == row["registered_input"] == variant
            and row["registered_payload"] == content_hash({"spec": row["spec_hash"], "input": variant})
            and row["registered_document"] == "{}" and evidence["variant_id"] == variant
            and type(evidence["planned_variants"]) is int and evidence["planned_variants"] == 1)


def validate_magnitude_execution(document):
    """Check the executable envelope only; catalog ownership and admission remain separate."""
    try:
        if document["schema_version"] == "native_experiment_registration.v1.3" and _magnitude_binding(document):
            return
    except (KeyError, TypeError, ValueError, OverflowError):
        pass
    raise fail("EXPERIMENT_IDENTITY_CONFLICT", "magnitude execution requires a sealed v1.3 plan")


def _magnitude_binding(document, verified_size_rows=None):
    """Check stored plan integrity and cross-links, never reconstruct or authorize."""
    if document["schema_version"] == "native_experiment_registration.v1.7":
        return _size_binding_v17(document, verified_size_rows)
    if document["schema_version"] == "native_experiment_registration.v1.6":
        return _size_binding(document)
    if document["schema_version"] == "native_experiment_registration.v1.5":
        return _fitted_binding(document)
    if document["schema_version"] == "native_experiment_registration.v1.4":
        return _history_binding(document)
    executable = document["schema_version"] == "native_experiment_registration.v1.3"
    if document["schema_version"] not in ("native_experiment_registration.v1.2",
                                           "native_experiment_registration.v1.3"):
        return not {"magnitude_request", "magnitude_plan", "magnitude_execution",
                    "magnitude_history_request", "magnitude_history"}.intersection(document)
    if set(document) != {"schema_version", "spec_hash", "execution_plan", "snapshot", "scope",
                         "source_closure", "environment", "event_ids", "holdouts",
                         "magnitude_request", "magnitude_plan"} | ({"magnitude_execution"} if executable else set()):
        return False
    plan, request = document["magnitude_plan"], document["magnitude_request"]
    if not isinstance(plan, dict) or not isinstance(request, dict):
        return False
    if set(plan) != {"schema_version", "snapshot_id", "holdout_as_of_month", "random_membership_version",
                     "rolling_membership_version", "training_start", "evaluation_years", "horizons",
                     "target_recipe", "fold_policy", "population_input", "calendar_input", "events",
                     "folds", "availability", "plan_id"}:
        return False
    return (_magnitude_metadata(plan) and _magnitude_sources(plan, document["snapshot"])
            and _magnitude_context(document, plan, request)
            and (not executable or _magnitude_execution(document, plan)))


def _size_binding_v17(document, verified_size_rows=None):
    """Structural v1.7 placeholder envelope; full SIZE validation follows hydration."""
    try:
        if set(document) != {"schema_version", "spec_hash", "execution_plan", "snapshot", "scope",
                             "source_closure", "environment", "event_ids", "holdouts",
                             "magnitude_request", "magnitude_plan", "magnitude_execution",
                             "size_request", "size_panel_request", "size_panel",
                             "size_comparison", "size_artifacts"}:
            return False
        return (_size_artifact_map_v17(document, verified_size_rows)
                and _size_plan_v17(document)
                and _size_panel_v17(document))
    except (KeyError, TypeError, ValueError, AttributeError, ArtifactError, RecursionError):
        return False


def _size_artifact_map_v17(document, verified_size_rows=None):
    artifacts = document["size_artifacts"]
    if not isinstance(artifacts, dict):
        return False
    found = []
    for key, value in document.items():
        if key != "size_artifacts" and not _size_placeholders_v17(value, found):
            return False
    if verified_size_rows is not None:
        for rows in verified_size_rows.values():
            if not _size_placeholders_v17(rows, found):
                return False
        if set(found) != set(artifacts):
            return False
    elif not set(found) <= set(artifacts):
        return False
    shard_total = 0
    for artifact_id, descriptor in artifacts.items():
        if not _size_descriptor_v17(descriptor, artifact_id):
            return False
        shard_total += len(descriptor["entry"]["shards"])
    return shard_total <= 8192


def _size_plan_v17(document):
    plan = document["magnitude_plan"]
    return (isinstance(plan, dict)
            and set(plan) == {"schema_version", "snapshot_id", "holdout_as_of_month",
                              "random_membership_version", "rolling_membership_version",
                              "training_start", "evaluation_years", "horizons",
                              "target_recipe", "fold_policy", "population_input",
                              "calendar_input", "events", "folds", "availability", "plan_id"}
            and _size_placeholder_v17(plan["events"])
            and _size_placeholder_v17(document["event_ids"]))


def _size_panel_v17(document):
    panel = document["size_panel"]
    if not isinstance(panel, dict) or set(panel) != {"context", "feature_snapshot"}:
        return False
    context, feature = panel["context"], panel["feature_snapshot"]
    if not isinstance(context, dict) or not isinstance(feature, dict):
        return False
    request = context.get("request")
    if (context.get("schema_version") != "magnitude_panel_context.v3"
            or not isinstance(context.get("context_id"), str)
            or context.get("source_snapshot") != document["snapshot"]
            or not _size_placeholder_v17(context.get("target_events"))
            or not isinstance(request, dict)
            or not _size_placeholder_v17(request.get("event_ids"))
            or not isinstance(feature.get("snapshot_id"), str)):
        return False
    return (document["size_request"]["schema_version"] == "magnitude_search_request.v2"
            and document["size_comparison"]["schema_version"] == "size_causal_comparator.v1"
            and isinstance(document["size_panel_request"], dict))


def _size_placeholders_v17(node, found):
    """Gather exact compact placeholders, rejecting mixed $size_artifact nodes."""
    if isinstance(node, dict):
        if "$size_artifact" in node:
            if set(node) != {"$size_artifact"} or not isinstance(node["$size_artifact"], str):
                return False
            found.append(node["$size_artifact"])
            return True
        return all(_size_placeholders_v17(value, found) for value in node.values())
    if isinstance(node, list):
        return all(_size_placeholders_v17(item, found) for item in node)
    return True


def _size_placeholder_v17(value):
    return (isinstance(value, dict) and set(value) == {"$size_artifact"}
            and isinstance(value["$size_artifact"], str))


def _size_descriptor_v17(descriptor, artifact_id):
    if (not isinstance(descriptor, dict) or set(descriptor) != {"kind", "section", "entry"}
            or descriptor["kind"] not in ("list", "object", "map")
            or descriptor["section"] != f"size_registration.v1.7:{artifact_id}"):
        return False
    entry = descriptor["entry"]
    if (not isinstance(entry, dict) or set(entry) != {"row_count", "shards"}
            or type(entry["row_count"]) is not int or entry["row_count"] < 0
            or not isinstance(entry["shards"], list)):
        return False
    return all(_size_shard_v17(shard) for shard in entry["shards"])


def _size_shard_v17(shard):
    if not isinstance(shard, dict) or set(shard) != {"ref", "rows", "first_key", "last_key"}:
        return False
    ref = from_document(ArtifactRef, shard["ref"])
    return (ref.schema_ref == "magnitude_search_shard.v1"
            and 1 <= ref.byte_size <= 8 * 1024 * 1024
            and type(shard["rows"]) is int and shard["rows"] > 0
            and isinstance(shard["first_key"], list)
            and isinstance(shard["last_key"], list))


def _size_binding(document):
    """Integrity-only additive panel envelope; research owns current source admission."""
    additions = ("size_request", "size_panel_request", "size_panel", "size_comparison")
    if not all(isinstance(document.get(key), dict) for key in additions):
        return False
    baseline = {key: value for key, value in document.items() if key not in additions}
    baseline["schema_version"] = "native_experiment_registration.v1.3"
    panel, request = document["size_panel"], document["size_request"]
    context = panel["context"]
    return (_size_panel_request(document) and document["size_comparison"]["schema_version"] == "size_causal_comparator.v1"
        and _magnitude_binding(baseline)
        and context["schema_version"] == "magnitude_panel_context.v3"
        and context["context_id"] == content_hash({k: v for k, v in context.items() if k != "context_id"})
        and context["source_snapshot"] == document["snapshot"]
        and request["schema_version"] == "magnitude_search_request.v2"
        and request["feature_snapshot_id"] == panel["feature_snapshot"]["snapshot_id"]
        and request["history_snapshot_id"] == context["history_snapshot"]["snapshot_id"]
        and {e["event_id"] for e in context["target_events"]} == set(document["event_ids"]))


def _size_panel_request(document):
    panel = document["size_panel"]
    context = panel.get("context")
    if set(panel) != {"context", "feature_snapshot"} or not isinstance(context, dict):
        return False
    request = context["request"]
    return (isinstance(request, dict) and set(request) == {"as_of_month", "policy", "event_ids",
        "horizons", "history_start", "limits", "source_start", "metadata_limits"}
        and document["size_panel_request"] == {**request,
            "feature_snapshot_id": panel["feature_snapshot"]["snapshot_id"],
            "history_snapshot": context["history_snapshot"]["snapshot_id"],
            "price_history_snapshot": context["price_history_snapshot"]["snapshot_id"]})


def magnitude_execution_document(labels_before):
    """Declare the fixed baseline policy; the evaluation clock has no implicit default."""
    try:
        if not _day(labels_before):
            raise ValueError
    except (ValueError, TypeError, OverflowError):
        raise fail("INVALID_EXPERIMENT_SPEC", "a canonical magnitude evaluation cutoff is required") from None
    return {"schema_version": "magnitude_execution.v1", "evaluation_labels_before": labels_before,
            "training_labels": "omit_skipped_unavailable_or_late_before_numeric_parse",
            "evaluation_labels": "complete_finite_strictly_before_cutoff",
            "model": "fold_local_arithmetic_mean", "features": ["constant_intercept"],
            "primary_metric": "event_pooled_rmse", "secondary_metrics": ["mae"],
            "quantile_calibration": "not_implemented"}


def _magnitude_execution(document, plan):
    """Validate sealed execution policy without reading labels or granting execution."""
    execution = document["magnitude_execution"]
    if (not isinstance(execution, dict) or not _day(execution.get("evaluation_labels_before"))
            or execution != magnitude_execution_document(execution["evaluation_labels_before"])):
        return False
    evaluation = {key for fold in plan["folds"] for key in fold["evaluation_event_ids"]}
    return (type(document["execution_plan"]["seed"]) is int and document["execution_plan"]["seed"] == 0
            and document["execution_plan"]["economic_params"] == {}
            and all(event["event_date"] < execution["evaluation_labels_before"]
                    for event in plan["events"] if event["event_id"] in evaluation))


def magnitude_history_request(feature_snapshot_id, history_start):
    """Validate the literal stored-feature request before repository access."""
    try:
        if _text(feature_snapshot_id) and _day(history_start):
            return {"feature_snapshot_id": feature_snapshot_id, "history_start": history_start}
    except (ValueError, TypeError, OverflowError):
        pass
    raise fail("INVALID_EXPERIMENT_SPEC", "history requires an exact feature snapshot and canonical start")


def magnitude_history_execution_document(labels_before):
    """Fixed stored ticker mean with an explicitly sealed empty-history fallback."""
    return {**magnitude_execution_document(labels_before), "schema_version": "magnitude_history_execution.v1",
            "model": "stored_ticker_arithmetic_mean", "features": ["n_prior", "mean_prior_abs_move"],
            "empty_history": "fold_local_arithmetic_mean", "primary_metric": "event_horizon_pooled_rmse",
            "weighting": "equal_event_and_horizon"}


def validate_magnitude_history_execution(document):
    """Validate a closed v1.4 envelope without reconstructing or authorizing reads."""
    try:
        if document["schema_version"] == "native_experiment_registration.v1.4" and _history_binding(document):
            return
    except (KeyError, TypeError, ValueError, OverflowError, ArtifactError):
        pass
    raise fail("EXPERIMENT_IDENTITY_CONFLICT", "history execution requires a sealed v1.4 plan")


def _fitted_magnitude_recipe(kind: str) -> dict:
    """Return a detached JSON policy; accepting caller overrides is unsupported."""
    if kind not in ("history_ridge", "history_hgb"):
        raise ValueError("unknown fitted magnitude model")
    ridge = kind == "history_ridge"
    params = ({"alpha": 1.0, "fit_intercept": True, "copy_X": True,
               "max_iter": None, "tol": 0.0001, "solver": "svd", "positive": False,
               "random_state": 0} if ridge else {
        "loss": "squared_error", "quantile": None, "learning_rate": 0.1,
        "max_iter": 100, "max_leaf_nodes": 7, "max_depth": None,
        "min_samples_leaf": 20, "l2_regularization": 0.0, "max_features": 1.0,
        "max_bins": 255, "categorical_features": None, "monotonic_cst": None,
        "interaction_cst": None, "warm_start": False, "early_stopping": False,
        "scoring": "loss", "validation_fraction": 0.1, "n_iter_no_change": 10,
        "tol": 1e-7, "verbose": 0, "random_state": 0})
    preprocessing = ({
        "imputer": {"strategy": "median", "keep_empty_features": True},
        "missing_indicator": {"features": "all", "sparse": False},
        "scaler": {"with_mean": True, "with_std": True}, "output_features": 22,
    } if ridge else {"missing_values": "native", "scaling": "none",
                    "all_empty_training_columns": "constant_zero_train_and_predict"})
    return {"schema_version": "magnitude_fitted_recipe.v1", "model": kind,
        "features": ["n_prior", "mean_prior_move", "mean_prior_abs_move",
                     "ema2_prior_move", "ema2_prior_abs_move", "ema4_prior_move",
                     "ema4_prior_abs_move", "ema8_prior_move", "ema8_prior_abs_move",
                     "ema12_prior_move", "ema12_prior_abs_move"],
        "estimator": {"class": "Ridge" if ridge else "HistGradientBoostingRegressor",
                      "params": params}, "preprocessing": preprocessing,
        "target_transform": "none", "prediction_projection": "clip_zero",
        "fit_scope": "fold_and_horizon", "numerical_threads": 1}



def magnitude_fitted_execution_document(labels_before, model):
    """Seal one fixed fitted recipe; arbitrary estimator overrides are unsupported."""
    try:
        recipe = _fitted_magnitude_recipe(model)
    except (ValueError, TypeError):
        raise fail("INVALID_EXPERIMENT_SPEC", "unsupported fitted magnitude model") from None
    execution = magnitude_history_execution_document(labels_before)
    execution.pop("empty_history")
    return {**execution, "schema_version": "magnitude_fitted_execution.v1", "model": model,
            "features": recipe["features"], "fit_recipe": recipe}


def _fitted_binding(document):
    """Require the literal recipe, then apply the unchanged stored-history proof."""
    execution = document["magnitude_execution"]
    try:
        expected = magnitude_fitted_execution_document(execution["evaluation_labels_before"], execution["model"])
        if canonical_json(execution) != canonical_json(expected):
            return False
    except (KeyError, TypeError, ValueError, OpsError):
        return False
    baseline = {**document, "schema_version": "native_experiment_registration.v1.4",
                "magnitude_execution": magnitude_history_execution_document(execution["evaluation_labels_before"])}
    return _history_binding(baseline)


def validate_magnitude_fitted_execution(document):
    """Validate closed v1.5 without granting payload access or fitting."""
    try:
        if document["schema_version"] == "native_experiment_registration.v1.5" and _fitted_binding(document):
            return
    except (KeyError, TypeError, ValueError, OverflowError, ArtifactError):
        pass
    raise fail("EXPERIMENT_IDENTITY_CONFLICT", "fitted magnitude execution requires a sealed v1.5 plan")


def _history_binding(document):
    """Cross-link the closed history binding, then reuse the v1.3 plan proof."""
    execution = document["magnitude_execution"]
    cutoff = execution["evaluation_labels_before"]
    if not _day(cutoff) or execution != magnitude_history_execution_document(cutoff):
        return False
    request, binding = document["magnitude_history_request"], document["magnitude_history"]
    if (set(request) != {"feature_snapshot_id", "history_start"} or not _text(request["feature_snapshot_id"])
            or not _day(request["history_start"]) or set(binding) != {"schema_version", "feature_snapshot", "context"}
            or binding["schema_version"] != "magnitude_history_binding.v1"):
        return False
    if not _history_context(document, binding, request) or not _history_producer(document, binding["context"]):
        return False
    baseline = {k: v for k, v in document.items() if k not in {"magnitude_history_request", "magnitude_history"}}
    baseline.update(schema_version="native_experiment_registration.v1.3",
                    magnitude_execution=magnitude_execution_document(cutoff))
    return (document["execution_plan"]["runner"] == "experiments/v2_prediction.py"
            and all(request["history_start"] <= clock["source_session_ceiling"]
                    for event in document["magnitude_plan"]["events"] for clock in event["forecasts"])
            and _magnitude_binding(baseline))


def _history_context(document, binding, request):
    """Cross-link the exact producer context and typed features-only snapshot."""
    context, feature = binding["context"], binding["feature_snapshot"]
    return (set(context) == {"schema_version", "source_snapshot", "temporal_plan", "history_start",
                            "feature_recipe", "producer", "context_id"}
            and context["schema_version"] == "magnitude_materialization.v1"
            and context["context_id"] == content_hash({k: v for k, v in context.items() if k != "context_id"})
            and context["source_snapshot"] == document["snapshot"]
            and context["temporal_plan"] == document["magnitude_plan"]
            and context["history_start"] == request["history_start"]
            and feature == to_document(from_document(SnapshotRef, feature))
            and feature["snapshot_id"] == request["feature_snapshot_id"]
            and set(feature["table_versions"]) == {"research_magnitude_history"})


def _history_producer(document, context):
    """Check the fixed recipe and producer subset of the sealed runner closure."""
    names = ["n_prior", "mean_prior_move", "mean_prior_abs_move"] + [
        f"ema{span}_prior_{kind}" for span in (2, 4, 8, 12) for kind in ("move", "abs_move")]
    recipe = {"schema_version": "stored_magnitude_history.v1", "arithmetic": "panel_math.history_features",
        "features": [{"name": name, "on_missing": "zero_count" if name == "n_prior" else "null"}
                     for name in names], "units": "percentage_points", "imputation": "none"}
    producer = context["producer"]
    sources = producer["source_closure"]
    return (context["feature_recipe"] == recipe and set(producer) == {"source_closure", "environment"}
            and producer["environment"] == document["environment"] and isinstance(sources, dict)
            and isinstance(document["source_closure"], dict)
            and {"experiments/magnitude_history_store.py", "engine/v2/research/magnitude_history_store.py",
                 "engine/v2/foundation/experiment_holdouts.py", "engine/v2/features/panel_math.py"} <= sources.keys()
            and all(_digest(value) and document["source_closure"].get(key) == value for key, value in sources.items()))


def _text(value):
    """Require nonblank, unpadded wire strings without coercion."""
    return isinstance(value, str) and bool(value.strip()) and value == value.strip()


def _day(value):
    """Validate the JSON wire date, without accepting timestamps or compact ISO dates."""
    return (isinstance(value, str) and re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value) is not None
            and date.fromisoformat(value).isoformat() == value)


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None


def _magnitude_metadata(plan):
    """Validate fixed v1 wire descriptors, not live calendar or fold semantics."""
    recipe = {"schema_version": "earnings_magnitude.v1", "units": "percentage_points",
        "target": "absolute_earnings_close_to_close_return", "price_basis": "split_neutral_non_dividend_adjusted",
        "estimand": "conditional_mean", "primary_loss": "pooled_rmse",
        "secondary_diagnostics": ["mae", "quantile_calibration"]}
    policy = {"kind": "expanding_event_year", "group_key": "event_id",
        "fit_boundary": "min_year_start_and_evaluation_forecast_ceilings",
        "training_label_availability": "strictly_before_fit_boundary_required_later",
        "issuer_overlap": "allowed_across_time", "inner_search": "not_defined"}
    availability = {"basis": "reconstructed_date_bounds", "event_schedule_as_known": "unverified",
                    "source_publication": "unverified"}
    return (plan["target_recipe"] == recipe and plan["fold_policy"] == policy
            and plan["availability"] == availability and _magnitude_window(plan)
            and plan["random_membership_version"] == "canonical-event-sha256.v1"
            and plan["rolling_membership_version"] == "calendar-months.v1"
            and _text(plan["snapshot_id"]) and _text(plan["holdout_as_of_month"])
            and len(plan["holdout_as_of_month"]) == 7 and _day(plan["holdout_as_of_month"] + "-01"))


def _magnitude_window(plan):
    years = plan["evaluation_years"]
    return (isinstance(years, list) and bool(years)
            and all(isinstance(year, str) and re.fullmatch(r"[0-9]{4}", year)
                    and 2 <= int(year) <= 9998 for year in years)
            and years == sorted(set(years)) and _day(plan["training_start"])
            and plan["training_start"] < years[0] + "-01-01")


def _magnitude_sources(plan, snapshot):
    """Bind closed provenance records to the already sealed snapshot, without I/O."""
    columns = ["event_id", "ticker", "event_date", "session", "date_conflict", "event_cluster_id"]
    population, calendar = plan["population_input"], plan["calendar_input"]
    if (not _magnitude_source(population, "earnings_events", columns, snapshot)
            or not _magnitude_source(calendar, "daily_market", ["date"], snapshot)):
        return False
    return (calendar["snapshot_id"] == plan["snapshot_id"]
            and _text(calendar["calendar_version"]) and calendar["calendar_version"] == snapshot["calendar_version"])


def _magnitude_source(source, table, columns, snapshot):
    fields = {"table", "columns", "dataset_version", "objects"}
    if table == "daily_market":
        fields |= {"snapshot_id", "calendar_version"}
    if not isinstance(source, dict) or set(source) != fields:
        return False
    return (source["table"] == table and source["columns"] == columns and _text(source["dataset_version"])
            and source["dataset_version"] == snapshot["table_versions"][table]["dataset_version_id"]
            and _magnitude_objects(source["objects"]))


def _magnitude_objects(objects):
    if not isinstance(objects, list) or not objects:
        return False
    return (all(isinstance(obj, dict) and set(obj) == {"fragment_id", "object_id", "content_hash"}
                and _text(obj["fragment_id"]) and _text(obj["object_id"]) and _digest(obj["content_hash"])
                for obj in objects)
            and len({obj["fragment_id"] for obj in objects}) == len(objects))


def _magnitude_context(document, plan, request):
    """Cross-link the sealed request and selected membership to the outer identity."""
    return (plan["schema_version"] == "magnitude_temporal_plan.v1"
            and plan["plan_id"] == content_hash({k: v for k, v in plan.items() if k != "plan_id"})
            and request == {"training_start": plan["training_start"], "evaluation_years": plan["evaluation_years"]}
            and plan["evaluation_years"] == document["execution_plan"]["folds"]
            and document["execution_plan"]["price_source"] == "computed_moves"
            and plan["snapshot_id"] == document["snapshot"]["snapshot_id"]
            and document["holdouts"] == {name: plan[name] for name in (
                "holdout_as_of_month", "random_membership_version", "rolling_membership_version")}
            and _magnitude_events(plan["events"], document["event_ids"])
            and _magnitude_folds(plan))


def _magnitude_events(events, expected):
    """Enclosing IDs must equal the complete nonempty unique canonical plan IDs."""
    fields = {"event_id", "ticker", "event_date", "session", "population_use", "forecasts"}
    if (not isinstance(events, list) or any(not isinstance(event, dict) or set(event) != fields
                                          for event in events)):
        return False
    ids = [event["event_id"] for event in events]
    return (bool(ids) and all(_magnitude_event(event) for event in events)
            and len(set(ids)) == len(ids) and sorted(ids) == expected
            and events == sorted(events, key=lambda event: (event["event_date"], event["event_id"]))
            and len({clock["calendar_id"] for event in events for clock in event["forecasts"]}) == 1)


def _magnitude_event(event):
    return (all(_text(event[key]) for key in ("event_id", "ticker", "event_date", "session", "population_use"))
            and _day(event["event_date"]) and event["session"] in ("AMC", "BMO")
            and event["population_use"] == "post-release selection" and _magnitude_forecasts(event))


def _magnitude_forecasts(event):
    """Each closed clock binds its canonical event and fixed horizon with an intact digest."""
    clocks = event["forecasts"]
    if not isinstance(clocks, list) or len(clocks) != 3:
        return False
    return (all(_magnitude_clock(clock, horizon, event) for clock, horizon in zip(clocks, (5, 3, 1)))
            and len({clock["pre_announcement_close"] for clock in clocks}) == 1)


def _magnitude_clock(clock, horizon, event):
    fields = {"schema_version", "event_date", "session", "horizon_sessions", "pre_announcement_close",
              "source_session_ceiling", "calendar_id", "cutoff_id"}
    return (isinstance(clock, dict) and set(clock) == fields
        and clock["schema_version"] == "earnings_forecast_cutoff.v1"
        and clock["event_date"] == event["event_date"] and clock["session"] == event["session"]
        and type(clock["horizon_sessions"]) is int and clock["horizon_sessions"] == horizon
        and _day(clock["pre_announcement_close"]) and _day(clock["source_session_ceiling"])
        and _digest(clock["calendar_id"])
        and clock["cutoff_id"] == content_hash({k: v for k, v in clock.items() if k != "cutoff_id"}))


def _magnitude_folds(plan):
    """Stored folds must cover the declared years and refer only to their sealed population."""
    if plan["horizons"] != [5, 3, 1] or any(type(value) is not int for value in plan["horizons"]):
        return False
    folds = plan["folds"]
    if not isinstance(folds, list) or len(folds) != len(plan["evaluation_years"]):
        return False
    events = {event["event_id"]: event["event_date"][:4] for event in plan["events"]}
    return all(_magnitude_fold(fold, year, events) for fold, year in zip(folds, plan["evaluation_years"]))


def _magnitude_fold(fold, year, events):
    """Validate structural membership; fresh reconstruction owns the actual date boundary."""
    if not isinstance(fold, dict) or set(fold) != {
            "year", "training_labels_before", "training_candidate_event_ids", "evaluation_event_ids"}:
        return False
    train, test = fold["training_candidate_event_ids"], fold["evaluation_event_ids"]
    return (fold["year"] == year and _day(fold["training_labels_before"])
            and all(_magnitude_member_order(ids, events) for ids in (train, test))
            and not set(train).intersection(test)
            and set(test) == {key for key, event_year in events.items() if event_year == year})


def _magnitude_member_order(ids, events):
    """One linear pass preserves the canonical order and rejects duplicates/foreign IDs."""
    if not isinstance(ids, list) or not ids:
        return False
    members = set(ids)
    return ids == [key for key in events if key in members]
