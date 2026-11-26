"""EXP-190 Q3a (design sections 4, 7): one fold's quantile fits, walk-forward configuration selection, the
non-crossing sort and arm ids. Pure: reads and writes no files; labels are supplied by the caller. No production
caller yet. The year-by-year loop, label reads, `bind_grid` and the fit receipts are the next slice; real-data
fits also need `size_quantile_prereg.require_preregistration` first. Uses v14's fit path
(`fit_regression_predictions`: signed domain so raw outputs are not clipped, native NaN, all-empty training columns
zeroed, one thread). Row masks are v14's: every training pair the caller passes, no feature filter. A `ValueError`
from a fit propagates: the caller records the fold's arm as `fit_failed` and does not retry."""
import itertools

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


def _key(c):
    return f"{c[0]}x{c[1]}"


def configurations(spec):
    axes = spec["grid"]["axes"]
    return sorted(itertools.product(axes["max_leaf_nodes"], axes["min_samples_leaf"]))


def q_raw(raw):
    return np.maximum(np.sort(np.asarray(raw, float), axis=1), 0.0)


def arm_id(spec, sealed, source, configuration):
    params = recipe_for(spec, sealed, configuration, spec["levels"][0])["estimator"]["params"]  # admits the recipe
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


def select_configuration(spec, outputs, labels):
    """`outputs`: fit_fold results of earlier out-of-sample years.
    `labels`: {event_id: y} for those events labelled before the cutoff."""
    configs = configurations(spec)
    if not outputs:
        if labels:
            raise _refuse("selection_inputs_invalid", "labels were given but there are no earlier outputs")
        return FIRST_CONFIGURATION, {}
    levels = list(spec["levels"])
    pairs = [(out, i, labels[e]) for out in outputs for i, e in enumerate(out["event_ids"]) if e in labels]
    if not pairs:
        raise fail("INVALID_EXPERIMENT_SPEC", "no earlier out-of-sample event has a label",
                   details={"reason": "selection_labels_empty"})
    try:
        y = np.asarray([v for _, _, v in pairs], dtype=float)
    except (TypeError, ValueError):
        raise _refuse("selection_label_invalid", "a selection label is not a finite number") from None
    if y.ndim != 1 or not np.isfinite(y).all():
        raise _refuse("selection_label_invalid", "a selection label is not a finite number")
    losses = {}
    for c in configs:
        rows = np.asarray([out["raw"][_key(c)][i] for out, i, _ in pairs], float).reshape(len(pairs), len(levels))
        losses[_key(c)] = float(pinball(y, np.sort(rows, axis=1), levels).mean())
    winner = min(configs, key=lambda c: (losses[_key(c)], c[0], -c[1]))
    decision("quantile selection", f"configuration={_key(winner)} selection_labels={len(pairs)}")
    return winner, losses


def fit_fold(spec, sealed, training, evaluation_ids):
    """`training`: [(event_id, y)] label-qualified before the fold cutoff by the caller."""
    raw = {}
    try:
        by_id = {r["event_id"]: r for r in sealed["rows"]}
        if len(by_id) != len(sealed["rows"]):
            raise _refuse("panel_row_duplicate", "an event has more than one lead-1 row")
        columns, levels = sealed["model_columns"], list(spec["levels"])
        train_x = _matrix(by_id, [e for e, _ in training], columns)
        y = np.asarray([v for _, v in training], dtype=float)
        eval_x = _matrix(by_id, evaluation_ids, columns)
        configs = configurations(spec)
        progress = Progress("quantile fits", len(configs) * len(levels))
        with stage("quantile fold fits"):
            for c in configs:
                cols = []
                for level in levels:
                    recipe = recipe_for(spec, sealed, c, level)
                    fit = fit_regression_predictions(recipe, train_x, y, eval_x, columns=columns)
                    cols.append(np.asarray(fit["predictions"], float))
                    progress.tick()
                raw[_key(c)] = np.column_stack(cols).tolist()
    except MemoryError:
        raise fail("RESOURCE_LIMIT_EXCEEDED", "a quantile fold ran out of memory") from None
    return {"event_ids": list(evaluation_ids), "raw": raw}
