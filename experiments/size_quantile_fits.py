"""EXP-190 Q3a (design sections 4, 7): one fold's quantile fits, walk-forward configuration selection, the
non-crossing sort and arm ids. No file I/O; the caller supplies labels. The yearly loop, its `bind_grid` call, label
reads and fit receipts are in `size_quantile_walk`, its only caller (nothing calls the walk yet); real-data fits need
`size_quantile_prereg.require_preregistration` first. Uses v14's fit path (`fit_regression_predictions`: signed
domain so raw outputs are not clipped, native NaN, all-empty training columns zeroed, one thread). Row masks are
v14's: every training pair the caller passes, no feature filter. A `ValueError` from the fit itself propagates: the
caller records the fold's arm as `fit_failed` and does not retry. Labels must be finite and non-negative; input
defects refuse before any fit."""
import itertools
import math
import numbers

import numpy as np

from engine.v2.foundation import content_hash
from engine.v2.models.regression_ridge_hgb_recipe import regression_recipe
from engine.v2.models.training.magnitude_ridge_hgb import fit_regression_predictions
from engine.v2.ops.errors import fail
from engine.v2.research.progress import Progress, decision, stage
from experiments.size_quantile_metrics import pinball

FIRST_CONFIGURATION = (7, 300)


def _refuse(reason, message):
    return fail("EXPERIMENT_IDENTITY_CONFLICT", f"{reason}: {message}", details={"reason": reason})


def _label_array(values, reason):
    for v in values:
        if not (isinstance(v, numbers.Real) and not isinstance(v, bool)) or not math.isfinite(v) or v < 0:
            raise _refuse(reason, "a label is not a finite non-negative number")
    return np.asarray(values, dtype=float)


def _key(c):
    return f"{c[0]}x{c[1]}"


def configurations(spec):
    axes = spec["grid"]["axes"]
    return sorted(itertools.product(axes["max_leaf_nodes"], axes["min_samples_leaf"]))


def q_raw(raw):
    return np.maximum(np.sort(np.asarray(raw, float), axis=1), 0.0)


def arm_id(spec, sealed, source, configuration):
    recipes = _recipes(spec, sealed, configuration)
    params = recipes[(configuration, spec["levels"][0])]["estimator"]["params"]
    cfg = {k: params[k] for k in ("max_leaf_nodes", "min_samples_leaf", *spec["grid"]["fixed"])}
    return content_hash({"source": source, "levels": list(spec["levels"]), "configuration": cfg})


def recipe_for(spec, sealed, configuration, level):
    fixed = spec["grid"]["fixed"]
    recipe = regression_recipe(target_domain="signed", model="history_hgb", columns=sealed["model_columns"],
                               feature_dictionary_hash=sealed["dictionary_hash"], max_leaf_nodes=configuration[0],
                               min_samples_leaf=configuration[1], learning_rate=fixed["learning_rate"],
                               l2_regularization=fixed["l2_regularization"], loss="quantile", quantile=level,
                               max_iter=fixed["max_iter"])
    params = recipe["estimator"]["params"]
    for key, value in fixed.items():
        if key not in params or (type(params[key]) is bool, params[key]) != (type(value) is bool, value):
            raise _refuse("fixed_parameters_mismatch", f"{key} is not the value fitted")
    return recipe


def _recipes(spec, sealed, required=None):
    configs = configurations(spec)
    levels = list(spec["levels"])
    real = all(isinstance(v, numbers.Real) and not isinstance(v, bool) and 0 < v < 1 for v in levels)
    if not (levels and configs and real and all(a < b for a, b in zip(levels, levels[1:]))
            and required in (None, *configs)):
        raise _refuse("grid_invalid", "levels real, in (0, 1), strictly increasing; grid complete; required in grid")
    try:
        return {(c, level): recipe_for(spec, sealed, c, level) for c, level in itertools.product(configs, levels)}
    except ValueError:
        raise _refuse("grid_invalid", "a grid cell is not an admitted recipe") from None


def _matrix(by_id, event_ids, columns):
    for e in event_ids:
        if e not in by_id:
            raise _refuse("panel_row_missing", f"{e} has no lead-1 row")
        if any(c not in by_id[e] for c in columns):
            raise _refuse("panel_column_missing", f"{e} lacks a model column")
    try:
        matrix = np.asarray([[by_id[e][c] for c in columns] for e in event_ids], dtype=float)
    except (TypeError, ValueError):
        raise _refuse("feature_invalid", "a feature value is not numeric") from None
    if np.isinf(matrix).any():
        raise _refuse("feature_infinite", "a feature value is infinite")
    return matrix


def select_configuration(spec, sealed, outputs, labels):
    """`outputs`: earlier years' `fit_fold` results; `labels`: {event_id: y} labelled before the cutoff."""
    _recipes(spec, sealed, FIRST_CONFIGURATION)
    if not outputs:
        if labels:
            raise _refuse("selection_inputs_invalid", "labels were given but there are no earlier outputs")
        return FIRST_CONFIGURATION, {}
    levels = list(spec["levels"])
    pairs = [(out, i, labels[e]) for out in outputs for i, e in enumerate(out["event_ids"]) if e in labels]
    if not pairs:
        raise fail("INVALID_EXPERIMENT_SPEC", "no earlier out-of-sample event has a label",
                   details={"reason": "selection_labels_empty"})
    y = _label_array([v for _, _, v in pairs], "selection_label_invalid")
    losses = {}
    for c in configurations(spec):
        rows = np.asarray([out["raw"][_key(c)][i] for out, i, _ in pairs], float).reshape(len(pairs), len(levels))
        losses[_key(c)] = float(pinball(y, np.sort(rows, axis=1), levels).mean())
    winner = min(configurations(spec), key=lambda c: (losses[_key(c)], c[0], -c[1]))
    decision("quantile selection", f"configuration={_key(winner)} selection_labels={len(pairs)}")
    return winner, losses


def fit_fold(spec, sealed, training, evaluation_ids, *, validate_only=False):
    """`training`: [(event_id, y)] label-qualified before the fold cutoff by the caller."""
    raw = {}
    try:
        by_id = {r["event_id"]: r for r in sealed["rows"]}
        if len(by_id) != len(sealed["rows"]):
            raise _refuse("panel_row_duplicate", "an event has more than one lead-1 row")
        columns, levels = sealed["model_columns"], list(spec["levels"])
        training_ids = [e for e, _ in training]
        ids = set(training_ids) | set(evaluation_ids)
        if not training_ids or not evaluation_ids or len(ids) != len(training_ids) + len(evaluation_ids):
            raise _refuse("fold_inputs_invalid", "training and evaluation ids must be non-empty, unique and disjoint")
        y = _label_array([v for _, v in training], "training_label_invalid")
        train_x = _matrix(by_id, training_ids, columns)
        eval_x = _matrix(by_id, evaluation_ids, columns)
        configs = configurations(spec)
        recipes = _recipes(spec, sealed)
        if validate_only:
            return None
        progress = Progress("quantile fits", len(configs) * len(levels))
        with stage("quantile fold fits"):
            for c in configs:
                cols = []
                for level in levels:
                    recipe = recipes[(c, level)]
                    fit = fit_regression_predictions(recipe, train_x, y, eval_x, columns=columns)
                    cols.append(np.asarray(fit["predictions"], float))
                    progress.tick()
                raw[_key(c)] = np.column_stack(cols).tolist()
    except MemoryError:
        raise fail("RESOURCE_LIMIT_EXCEEDED", "a quantile fold ran out of memory") from None
    return {"event_ids": list(evaluation_ids), "raw": raw}
