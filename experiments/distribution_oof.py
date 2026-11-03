"""Causal shared residual controls from sealed earlier inner-fit completions.

No label reader, calibrator, fit, fallback or probability smoothing lives here.
Native admission owns availability/holdouts; fit replay owns forecast validation.
"""
from math import fsum, isfinite

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import fail
from engine.v2.research.magnitude_distribution import recipe, validate_distribution
from engine.v2.research.magnitude_distribution_math import mean, mean_matched_control
from engine.v2.research.progress import Progress, stage

__all__ = ["residual_pool", "control_for_prediction"]
_SCHEMA = "distribution_oof_pool.v1"
_POLICY = "earlier_inner_equal_weight_per_horizon.v1"


def _require(condition):
    if not condition:
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "distribution residual evidence differs")


def _number(value):
    try:
        _require(type(value) in (int, float))
        value = float(value)
        if isfinite(value):
            return value
    except (OverflowError, ValueError):
        pass
    raise fail("EXPERIMENT_VARIANT_FAILED", "distribution residual arithmetic is nonfinite")


def _sum(values):
    try:
        return _number(fsum(values))
    except (OverflowError, ValueError):
        raise fail("EXPERIMENT_VARIANT_FAILED", "distribution residual arithmetic is nonfinite") from None


def residual_pool(fits, role, outer_year, horizon, inner_year=None):
    """Read every sealed earlier inner donor; join destination training labels.

    Outer uses all declared inners; inner uses only its strict earlier prefix.
    Missing completions fail, never disappear. Donor tails are their own recorded
    training tails. Any unresolved donor tail disables the entire histogram
    control, not the separately usable point pool. No cross-call cache or writes.
    Shared outputs are O(rows), never predictions times residuals.
    """
    context = fits.trials.read_fit_context(role, outer_year, inner_year)
    plan, fold, admission = (context[key] for key in ("plan", "fold", "admission"))
    _require(type(horizon) is int and horizon in plan["recipe"]["horizons"])
    groups = [group for group in context["native_plan"]["folds"] if group["outer"]["year"] == outer_year]
    _require(len(groups) == 1 and role in ("inner", "outer"))
    inners = groups[0]["inner"]
    years = [item["year"] for item in inners]
    _require(years == sorted(set(years)) and all(year < outer_year for year in years))
    _require((role == "outer" and inner_year is None) or (role == "inner" and inner_year in years))
    donors = inners if role == "outer" else inners[:years.index(inner_year)]
    training = context["inputs"]["training"]
    labels = {row["event_id"]: _number(row["absolute_move_pct"]) for row in training}
    _require(len(labels) == len(training) and all(value >= 0 for value in labels.values()))
    _require(not set(labels).intersection(fold["evaluation_event_ids"]))
    point, histogram, keys, exclusions, sources, seen = [], [], [], [], [], set()
    unresolved = False
    progress = Progress("distribution earlier OOF residuals", sum(len(item["evaluation_event_ids"]) for item in donors))
    with stage("distribution earlier OOF residuals"):
        for donor in donors:
            result = fits.read("inner", outer_year, horizon, inner_year=donor["year"])
            _require(type(result) is dict)
            receipt, rows, tail = (result[key] for key in ("receipt", "rows", "tail"))
            claim, trial = receipt["claim"], receipt["claim"]["trial"]
            _require(receipt["status"] == "completed" and receipt["result_id"] == content_hash(
                {key: value for key, value in receipt.items() if key != "result_id"}))
            _require((claim["role"], claim["outer_year"], claim["inner_year"], claim["plan_id"]) ==
                     ("inner", outer_year, donor["year"], plan["plan_id"]))
            expected = {"recipe_id": plan["recipe"]["recipe_id"], "context_id": plan["context_id"],
                "budget_id": plan["budget"]["budget_id"], "horizon": horizon, "role": "inner",
                "fold_id": content_hash({"fold": donor, "role": "inner", "outer_year": outer_year})}
            _require(all(trial[key] == value for key, value in expected.items()))
            events = [row["event_id"] for row in rows]
            _require(len(set(events)) == len(events) and set(events) == set(donor["evaluation_event_ids"]))
            _require(not set(events).intersection(fold["evaluation_event_ids"]))
            unresolved |= tail["policy"] == "unresolved"
            sources.append({"inner_year": donor["year"], "result_id": receipt["result_id"], "tail_id": content_hash(tail)})
            basis = None
            if tail["policy"] != "unresolved":
                # Reuse one conditional tail moment per donor, not per forecast.
                tail_mean = mean({**recipe(), "probabilities": [0] * 80 + [1], "tail": tail})
                basis = [index / 4 + 0.125 for index in range(80)] + [tail_mean]
            for row in sorted(rows, key=lambda row: row["event_id"]):
                event, prediction = row["event_id"], _number(row["point_prediction"])
                key = (event, horizon)
                _require(key not in seen and prediction >= 0)
                seen.add(key)
                if event not in labels:
                    exclusions.append({"event_id": event, "horizon": horizon,
                        "result_id": receipt["result_id"], "reason": "not_destination_training"})
                else:
                    keys.append([event, horizon])
                    point.append(_number(labels[event] - prediction))
                    if basis is not None:
                        masses = row["probabilities"]
                        total = _sum(masses)
                        if masses[-1] and not (masses[-1] / total / tail["sample_count"]):
                            # Delegate the public reconstruction's underflow refusal.
                            mean({**recipe(), "probabilities": masses, "tail": tail})
                        forecast = _sum((p / total) * value for p, value in zip(masses, basis))
                        histogram.append(_number(labels[event] - forecast))
                progress.tick()
    empty = "no_earlier_inner" if not donors else "no_eligible_rows"
    body = {"schema_version": _SCHEMA, "policy": _POLICY, "calibration_identity": "uncalibrated",
        "reconstruction": "uniform", "support_policy": "zero_atom_histogram_mean_anchor_point_actual_mean",
        "destination": {"plan_id": plan["plan_id"], "recipe_id": plan["recipe"]["recipe_id"],
            "context_id": plan["context_id"], "inputs_id": admission["inputs_id"], "fold_id": admission["fold_id"],
            "admission_id": content_hash(admission), "budget_id": plan["budget"]["budget_id"],
            "role": role, "outer_year": outer_year, "inner_year": inner_year, "horizon": horizon,
            "training_labels_before": fold["training_labels_before"]},
        "sources": sources, "keys": keys, "exclusions": exclusions,
        "coverage": {"source_rows": len(seen), "contributing_rows": len(keys), "excluded_rows": len(exclusions)},
        "point_status": "ready" if point else empty, "point_residuals": point,
        "histogram_status": "unresolved_donor_tail" if unresolved else ("ready" if point else empty),
        "histogram_residuals": None if unresolved else histogram, "final_read_authority": "none"}
    return {**body, "pool_id": content_hash(body)}


def control_for_prediction(pool, histogram_document, point_prediction):
    """Apply a verified pool document without fitting or reauthenticating sources.

    Use only residual_pool output; its hash detects mutation, not forged source
    authority. Undefined histogram anchoring retains the existing numeric refusal.
    Returned empirical draws preserve zero atoms; no histogram rebucketing occurs.
    """
    _require(type(pool) is dict and pool.get("schema_version") == _SCHEMA and pool.get("policy") == _POLICY)
    _require(pool["pool_id"] == content_hash({key: value for key, value in pool.items() if key != "pool_id"}))
    document = validate_distribution(histogram_document)
    prediction = _number(point_prediction)
    _require(prediction >= 0)
    point, histogram = {"status": pool["point_status"]}, {"status": pool["histogram_status"]}
    if point["status"] == "ready":
        draws = [max(0.0, _number(prediction + value)) for value in pool["point_residuals"]]
        point.update(magnitudes=draws, mean=_sum(value / len(draws) for value in draws),
                     zero_count=sum(value == 0 for value in draws))
    if histogram["status"] == "ready":
        if document["tail"]["policy"] == "unresolved":
            histogram["status"] = "unresolved_prediction_tail"
        else:
            histogram.update(status="ready", **mean_matched_control(mean(document), pool["histogram_residuals"]))
    return {"pool_id": pool["pool_id"], "calibration_identity": "uncalibrated", "point": point, "histogram": histogram}
