"""Research-only sealed nested-search metadata with bounded label-bounds admission.

Planning reads only the registration-pinned snapshot's planned bounds keys and
publishes or verifies canonical plan bytes; it writes no ledger, outcome or
catalog reservation.
"""
import itertools
import json
import math
from collections import Counter
from datetime import date
from pathlib import Path

from engine.v2.contracts import ArtifactRef
from engine.v2.contracts.data import DataQuery, KeyPredicate, SnapshotRef
from engine.v2.data import bounded_snapshot
from engine.v2.data.computed_move_bounds import CONTRACT as BOUNDS_CONTRACT
from engine.v2.data.computed_move_bounds import TABLE as BOUNDS_TABLE
from engine.v2.data.errors import DataError, fail as data_fail
from engine.v2.data.repository import Repository
from engine.v2.foundation import (
    ArtifactError,
    artifact_reference,
    canonical_json,
    content_hash,
    from_document,
    to_document,
)
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.fingerprints import environment_identity, source_closure
from engine.v2.ops.native_registration import (
    magnitude_fitted_execution_document,
    validate_magnitude_fitted_execution,
)
from engine.v2.research import magnitude_folds
from engine.v2.research.event_history import MAX_OUTCOME_OBJECT_BYTES
from engine.v2.research.progress import Progress, decision, stage
from experiments.label_availability import label_cutoff_reason
from experiments.magnitude_search_errors import artifact_failure
from experiments.magnitude_search_request_schema import (
    qualified as is_qualified_request,
    request_version as parse_request_version,
)
from experiments.native_registration import verify_native_registration

SCHEMA = "magnitude_search_plan.v1"
PLAN_V2 = "magnitude_search_plan.v2"
PLAN_V3 = "magnitude_search_plan.v3"
_HORIZONS = (5, 3, 1)
_AXES = {"history_ridge": {"alpha"}, "history_hgb": {
    "learning_rate", "max_iter", "max_leaf_nodes", "min_samples_leaf", "l2_regularization"}}
_INTEGERS = {"max_iter", "max_leaf_nodes", "min_samples_leaf"}
_BOUNDS_CATALOG_LIMITS = {"max_catalog_tables": 256, "max_catalog_fragments": 65_536,
    "max_catalog_bytes": 64 * 1024 * 1024, "max_fragments": 65_536,
    "max_partitions": 65_536}


def plan_schema_for_request(request):
    """Return the default plan schema for a supported request version.

    Research-only: the standalone ``experiments.v2_magnitude_search`` CLI and the
    qualified SIZE binding call this mapping; there is no production entrypoint.
    """
    version = request.get("schema_version") if isinstance(request, dict) else None
    return PLAN_V2 if version == "magnitude_search_request.v3" else PLAN_V3


def _request(request):
    """Admit the complete bounded Cartesian request before any artifact access."""
    try:
        if (not isinstance(request, dict) or set(request) != {
                "schema_version", "inner_years", "space", "max_candidates", "max_fits"}
                or request["schema_version"] != "magnitude_search_request.v1"
                or type(request["max_candidates"]) is not int or not 2 <= request["max_candidates"] <= 4096
                or type(request["max_fits"]) is not int or request["max_fits"] < 1):
            raise ValueError
        years, space = request["inner_years"], request["space"]
        if (not isinstance(years, dict) or not years or not isinstance(space, dict)
                or not space or not space.keys() <= _AXES.keys()):
            raise ValueError
        for outer, inner in years.items():
            if (not _year(outer) or not isinstance(inner, list) or not inner
                    or any(not _year(year) or year >= outer for year in inner)
                    or inner != sorted(set(inner))):
                raise ValueError
        count = 0
        for kind, axes in space.items():
            if not isinstance(axes, dict) or axes.keys() != _AXES[kind]:
                raise ValueError
            for name, values in axes.items():
                if (not isinstance(values, list) or not values or len(values) > request["max_candidates"]
                        or any(not _parameter(name, value) for value in values)
                        or values != sorted(set(values))):
                    raise ValueError
            count += math.prod(len(values) for values in axes.values())
        if not 2 <= count <= request["max_candidates"]:
            raise ValueError
        return json.loads(canonical_json(request)), count
    except (ValueError, TypeError, OverflowError):
        raise fail("INVALID_EXPERIMENT_SPEC", "search requires explicit valid axes, inner years and count caps") from None


def _year(value):
    return (isinstance(value, str) and len(value) == 4 and value.isascii()
            and value.isdecimal() and 2 <= int(value) <= 9998)


def _parameter(name, value):
    if name in _INTEGERS:
        return type(value) is int and value >= (2 if name == "max_leaf_nodes" else 1)
    return (type(value) in (int, float) and math.isfinite(value)
            and (value >= 0 if name == "l2_regularization" else value > 0))


def _v3(plan):
    return plan.get("schema_version") == PLAN_V3


def trial_key(plan, outer_year, inner_year, candidate_id):
    """Shared trial identity: v3 ignores the outer year, older plans keep it."""
    if _v3(plan):
        return content_hash([plan["plan_id"], inner_year, candidate_id])
    return content_hash([plan["plan_id"], outer_year, inner_year, candidate_id])


def cohort_key(plan, outer_year, inner_year):
    """Shared cohort identity: v3 ignores the outer year, older plans keep it."""
    if _v3(plan):
        return content_hash([plan["plan_id"], inner_year])
    return content_hash([plan["plan_id"], outer_year, inner_year])


def canonical_inner(plan, outer_year, inner_year):
    """Return the owning outer year and canonical inner fold for a trial."""
    if _v3(plan):
        for owner in plan["folds"]:
            for inner in owner["inner"]:
                if inner["year"] == inner_year:
                    return owner["outer"]["year"], inner
    else:
        try:
            owner = next(fold for fold in plan["folds"] if fold["outer"]["year"] == outer_year)
            inner = next(fold for fold in owner["inner"] if fold["year"] == inner_year)
            return outer_year, inner
        except (StopIteration, KeyError, TypeError):
            pass
    raise fail("INVALID_EXPERIMENT_SPEC", "trial is outside the sealed search plan")


def _label_end(value):
    """Accept canonical stored dates and null finalized-unavailable labels.

    A null bound is a present row whose label the data contract finalized as
    unavailable; malformed non-null artifacts remain identity conflicts.
    """
    if value is None:
        return None
    try:
        if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "planned label bounds are not canonical dates") from None
    return value


def _label_bounds(conn, store, binding):
    """Read only planned event keys after validating the native bounds table/contract.

    A missing table or a different bound contract refuses the whole plan with
    ``CONTRACT_MISMATCH`` before any scan. An absent row for one planned event is
    different: it stays out of the result and is sealed as
    ``label_bounds_missing`` for that event only. A present row with a null
    post-close date is likewise event-local, but seals
    ``label_bounds_unavailable`` because the data contract finalizes that label
    as unavailable.

    The bounds source is the registration-pinned ``binding["snapshot"]``, not
    the separate feature snapshot id.

    ``computed_move_bounds`` declares ``(ticker, event_date)`` as its unique
    primary key and both columns filterable. Group the plan's events by ticker
    and query each ticker's exact planned dates with ``ticker eq`` and
    ``event_date in``; the unique key makes every returned row the planned row
    for that ticker/date, so cross-pairs and repeated session scans cannot
    satisfy an unplanned key. Catalog membership and bytes are admitted through
    the public ``bounded_snapshot.open_bounded_snapshot`` API under fixed caps
    before table access, and each scan is bounded by its
    ``scan_population_bound``, the contract's batch/result caps and the same
    decoded-object cap as native magnitude outcome reads.

    ``_prepare`` admits only a registration whose validated magnitude plan has
    events, but an empty population still validates the table/contract: the
    empty ticker loop is safe and reads no bounds row.
    """
    events = binding["magnitude_plan"]["events"]
    snapshot = from_document(SnapshotRef, binding["snapshot"])
    days_by_ticker = {}
    for event in events:
        days_by_ticker.setdefault(event["ticker"], set()).add(event["event_date"])
    try:
        with stage("label bounds"):
            repository = Repository(conn, store, pin_snapshots=True)
            snapshot_id = snapshot.snapshot_id
            with bounded_snapshot.open_bounded_snapshot(repository, snapshot,
                    table_name=BOUNDS_TABLE, limits=_BOUNDS_CATALOG_LIMITS) as session:
                resolved = session.resolve_full(snapshot_id)
                if (BOUNDS_TABLE not in resolved.snapshot.table_versions
                        or session.table_contract(resolved.snapshot, BOUNDS_TABLE) != BOUNDS_CONTRACT):
                    raise data_fail("CONTRACT_MISMATCH",
                               "planned label bounds require the native bounds contract")
                contract_ref = resolved.snapshot.table_versions[BOUNDS_TABLE].table_contract_ref
                found, empty_bounds = {}, 0
                planned_tickers = sorted(days_by_ticker)
                total = len(planned_tickers)
                progress = Progress("label bounds tickers", total)
                decision("label bounds", f"planned tickers={total} events={len(events)}")
                for ticker in planned_tickers:
                    key_filter = (
                        KeyPredicate(column="ticker", operator="eq", values=(ticker,)),
                        KeyPredicate(column="event_date", operator="in",
                                     values=tuple(sorted(days_by_ticker[ticker]))))
                    bound = session.scan_population_bound(snapshot_id, table_name=BOUNDS_TABLE,
                        table_contract_ref=contract_ref, key_filter=key_filter)
                    if not bound:
                        empty_bounds += 1
                    else:
                        request = DataQuery(snapshot_id=snapshot_id, table_contract_ref=contract_ref,
                            columns=("ticker", "event_date", "post_close_date"),
                            key_filter=key_filter, order_by=BOUNDS_CONTRACT.primary_key,
                            max_batch_rows=min(bound, BOUNDS_CONTRACT.maximum_batch_rows), max_result_rows=bound)
                        for batch in session.scan(request, table_name=BOUNDS_TABLE,
                                                  max_object_bytes=MAX_OUTCOME_OBJECT_BYTES):
                            for row in batch.to_pylist():
                                found[(row["ticker"], row["event_date"])] = _label_end(row["post_close_date"])
                    progress.tick()
                missing_rows = sum(1 for event in events
                                   if (event["ticker"], event["event_date"]) not in found)
                unavailable_rows = sum(1 for event in events
                                       if (event["ticker"], event["event_date"]) in found
                                       and found[(event["ticker"], event["event_date"])] is None)
                decision("label bounds", f"planned tickers={total} empty_bounds={empty_bounds} "
                         f"missing_rows={missing_rows} unavailable_rows={unavailable_rows}")
        return {event["event_id"]: found[(event["ticker"], event["event_date"])]
                for event in events if (event["ticker"], event["event_date"]) in found}
    except DataError as error:
        if (error.code == "CONTRACT_MISMATCH"
                and error.problem.message == "catalog probe table unavailable"):
            problem = data_fail("CONTRACT_MISMATCH",
                       "planned label bounds require the native bounds contract").problem
            raise OpsError(problem) from None
        if error.problem.category == "integrity":
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "planned label bounds could not be read") from None
        raise OpsError(error.problem) from None


def _filter_evaluation(event_ids, bounds, cutoff):
    """Keep labels available strictly before the cutoff; report one reason per drop.

    ``bounds`` contains only planned events with a bounds row, so membership
    distinguishes an absent row from a present row whose label is unavailable.
    """
    retained, excluded = [], []
    for event_id in event_ids:
        if event_id not in bounds:
            reason = "label_bounds_missing"
        else:
            label_end = bounds[event_id]
            reason = ("label_bounds_unavailable" if label_end is None
                      else label_cutoff_reason(label_end, cutoff))
        if reason is None:
            retained.append(event_id)
        else:
            excluded.append({"event_id": event_id, "reasons": [reason]})
    return retained, excluded


def _exclusion_summary(counts):
    return ", ".join(f"{reason}={count}" for reason, count in sorted(counts.items()))


def _refuse_empty_evaluation(scope, year, excluded, total_counts):
    """Refuse a fold whose label exclusions removed every evaluation event."""
    decision("search evaluation exclusions", _exclusion_summary(total_counts))
    fold_counts = Counter(reason for item in excluded for reason in item["reasons"])
    raise fail("INVALID_EXPERIMENT_SPEC",
        f"{scope} evaluation fold {year}: no eligible evaluation events remain after label exclusions "
        f"({_exclusion_summary(fold_counts)})")


def _nested(binding, request, *, temporal=None, label_bounds=None):
    """Preserve annual event groups; inner scoring labels must predate outer fitting."""
    temporal = binding["magnitude_plan"] if temporal is None else temporal
    try:
        expected = [magnitude_folds.annual_fold(year, temporal["events"]) for year in temporal["evaluation_years"]]
        if expected != temporal["folds"]:
            raise ValueError
    except (DataError, ValueError):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "outer fold metadata differs from its declared boundaries") from None
    if set(request["inner_years"]) != set(temporal["evaluation_years"]):
        raise fail("INVALID_EXPERIMENT_SPEC", "search inner years must cover every outer fold")
    outer_cutoff = (binding["magnitude_execution"]["evaluation_labels_before"]
                    if label_bounds is not None else None)
    nested = []
    exclusion_counts = Counter()
    try:
        for outer in expected:
            if label_bounds is not None:
                retained, excluded = _filter_evaluation(
                    outer["evaluation_event_ids"], label_bounds, outer_cutoff)
                outer = {**outer, "evaluation_event_ids": retained}
                if excluded:
                    outer["evaluation_exclusions"] = excluded
                    for item in excluded:
                        exclusion_counts.update(item["reasons"])
                if not retained:
                    _refuse_empty_evaluation("outer", outer["year"], excluded, exclusion_counts)
            members = set(outer["training_candidate_event_ids"])
            population = [event for event in temporal["events"] if event["event_id"] in members]
            inner = []
            for year in request["inner_years"][outer["year"]]:
                fold = magnitude_folds.annual_fold(year, population)
                if label_bounds is not None:
                    retained, excluded = _filter_evaluation(
                        fold["evaluation_event_ids"], label_bounds, outer["training_labels_before"])
                    fold = {**fold, "evaluation_event_ids": retained}
                    if excluded:
                        fold["evaluation_exclusions"] = excluded
                        for item in excluded:
                            exclusion_counts.update(item["reasons"])
                    if not retained:
                        _refuse_empty_evaluation("inner", fold["year"], excluded, exclusion_counts)
                inner.append({**fold, "evaluation_labels_before": outer["training_labels_before"]})
            nested.append({"outer": outer, "inner": inner})
    except DataError:
        raise fail("INVALID_EXPERIMENT_SPEC", "every search fold needs earlier training and evaluation candidates") from None
    if exclusion_counts:
        decision("search evaluation exclusions", _exclusion_summary(exclusion_counts))
    return nested


def nested_folds(binding, request):
    """Research-only nested-fold adapter used by SIZE; the research planner and
    regression binding call ``_nested`` directly."""
    return _nested(binding, request)


def _candidates(binding, request):
    """Use fixed-policy constructors and stdlib product, without constructing estimators."""
    candidates = []
    for kind, axes in sorted(request["space"].items()):
        names = sorted(axes)
        for values in itertools.product(*(axes[name] for name in names)):
            recipe = magnitude_fitted_execution_document(
                binding["magnitude_execution"]["evaluation_labels_before"], kind)["fit_recipe"]
            recipe["estimator"]["params"].update(zip(names, values))
            candidates.append({"candidate_id": content_hash(recipe), "recipe": recipe})
    return sorted(candidates, key=lambda candidate: candidate["candidate_id"])


def _implementation(binding, *, entrypoints=None):
    """Bind installed planning code and require the original execution sources/environment."""
    root = Path(__file__).resolve().parents[1]
    try:
        sources = source_closure(root, entrypoints or ["experiments/v2_magnitude_search.py", "experiments/v2_prediction.py"])
        anchor_matches = (source_closure(root, [binding["execution_plan"]["runner"]]) == binding["source_closure"]
            if entrypoints is not None else all(sources.get(key) == digest
                for key, digest in binding["source_closure"].items()))
    except (OpsError, ArtifactError, OSError, SyntaxError, ValueError):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "search implementation sources cannot be verified") from None
    environment = environment_identity(1)
    if environment != binding["environment"] or not anchor_matches:
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "search anchor sources or environment differ from this installation")
    return {"source_closure": sources, "environment": environment}


def _prepare(conn, store, registration, request, *, plan_schema=None):
    request_version = request.get("schema_version") if isinstance(request, dict) else None
    anchor_version = getattr(registration, "document", {}).get("schema_version")
    if anchor_version == "native_experiment_registration.v1.8":
        from experiments.iv_crush_search import prepare_plan
        return prepare_plan(conn, store, registration, request)
    qualified = anchor_version in ("native_experiment_registration.v1.6", "native_experiment_registration.v1.7")
    expected = (is_qualified_request(parse_request_version(request_version)) if qualified
                else parse_request_version(request_version) == 1)
    if not expected:
        raise fail("INVALID_EXPERIMENT_SPEC", "search requires explicit valid axes, inner years and count caps")
    if qualified:
        from experiments.size_search_binding import prepare_plan
        default_schema = plan_schema_for_request(request)
        return prepare_plan(conn, store, registration, request, plan_schema=plan_schema or default_schema)
    request, count = _request(request)
    verify_native_registration(conn, store, registration)
    binding = registration.document
    validate_magnitude_fitted_execution(binding)
    nested = _nested(binding, request, label_bounds=_label_bounds(conn, store, binding))
    fits = len(_HORIZONS) * (count * sum(len(fold["inner"]) for fold in nested) + len(nested))
    if fits > request["max_fits"]:
        raise fail("INVALID_EXPERIMENT_SPEC", "planned search fits exceed the explicit cap")
    identity = {"schema_version": SCHEMA, "request": request,
        "anchor": {"run_id": registration.run_id, "variant_id": registration.variant_id,
            "registration_ref": to_document(artifact_reference(registration.binding_json, binding["schema_version"]))},
        "implementation": _implementation(binding), "candidates": _candidates(binding, request),
        "folds": nested, "horizons": list(_HORIZONS), "candidate_count": count, "planned_fit_count": fits,
        "selection": {"primary_metric": "inner_event_horizon_pooled_rmse", "secondary_metrics": ["mae"],
            "tie_break": "candidate_id_ascending", "cohort": "shared_qualified_labels_across_candidates",
            "training_labels": "strictly_before_inner_fit", "validation_labels": "complete_before_outer_fit",
            "attempts": "all_candidates_before_selection", "failed_candidate": "exclude_entire_score",
            "all_candidates_failed": "refuse", "retries": 0,
            "outer_evaluation": "selected_candidate_only_after_freezing", "outer_fit_failure": "refuse",
            "winner": "latest_outer_inner_selection_before_its_evaluation",
            "final_read_authority": "none"}}
    return json.loads(canonical_json({**identity, "plan_id": content_hash(identity)}))


def seal_search_plan(conn, store, registration, request):
    """Publish immutable planning bytes only; no catalog reservation, attempt or result."""
    plan = _prepare(conn, store, registration, request)
    try:
        ref = store.publish_bytes(canonical_json(plan).encode(), schema_ref=plan["schema_version"])
    except ArtifactError as error:
        raise _artifact_failure(error) from None
    return {"plan": plan, "plan_ref": to_document(ref)}


def read_search_plan(conn, store, registration, request, ref):
    """Reconstruct and revalidate the complete binding, repeating bounded bounds admission."""
    version = request.get("schema_version") if isinstance(request, dict) else None
    parsed_version = parse_request_version(version)
    if is_qualified_request(parsed_version) and parsed_version != 3:
        schemas = (PLAN_V2, PLAN_V3)
    elif version == "magnitude_search_request.v3":
        schemas = (PLAN_V2,)
    elif version == "regression_search_request.v1":
        schemas = ("regression_search_plan.v1",)
    elif version == "magnitude_search_request.v1":
        schemas = (SCHEMA,)
    else:
        schemas = ()
    if not isinstance(ref, ArtifactRef) or ref.schema_ref not in schemas:
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "search plan reference has the wrong schema")
    expected = _prepare(conn, store, registration, request,
                        plan_schema=ref.schema_ref if is_qualified_request(parsed_version) else None)
    payload = canonical_json(expected).encode()
    if ref != artifact_reference(payload, ref.schema_ref):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "search plan reference differs from its complete binding")
    try:
        store.read_verified(ref)
    except ArtifactError as error:
        raise _artifact_failure(error) from None
    return expected


def _artifact_failure(error):
    """Share the publication/read integrity distinction without leaking paths."""
    return artifact_failure(error, integrity_message="stored search plan failed integrity verification",
                            resource_message="search planning request could not be completed")
