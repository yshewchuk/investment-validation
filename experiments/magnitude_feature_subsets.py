"""Bounded static feature-neighborhood previews; no sources, fits or admission."""
import json
import re

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import fail

__all__ = ["plan_subsets"]

_KEYS = {"schema_version", "dictionary", "column_list_id", "core", "optional_blocks",
         "model_configuration_ids", "horizons", "outer_years", "inner_years",
         "max_subsets", "max_candidates", "max_fits"}


def _require(condition, message):
    if not condition:
        raise fail("INVALID_EXPERIMENT_SPEC", "feature subsets: " + message)


def _identifier(value):
    return (type(value) is str and 1 <= len(value) <= 64
            and re.fullmatch(r"[a-z][a-z0-9_]*", value) is not None)


def _hash(value):
    return (type(value) is str and len(value) == 71
            and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None)


def _columns(values):
    return (type(values) is list and 1 <= len(values) <= 512
            and all(_identifier(value) for value in values) and len(set(values)) == len(values))


def _ordered(values, positions):
    return (_columns(values) and all(value in positions for value in values)
            and values == sorted(values, key=positions.__getitem__))


def _partition(request):
    dictionary, core, blocks = request["dictionary"], request["core"], request["optional_blocks"]
    _require(_columns(dictionary) and _hash(request["column_list_id"]),
             "dictionary requires bounded unique names and a canonical column_list_id")
    positions = {name: index for index, name in enumerate(dictionary)}
    _require(_ordered(core, positions), "core must be nonempty and dictionary-ordered")
    _require(type(blocks) is list and len(blocks) <= 64, "at most 64 optional blocks are allowed")
    seen, names = set(core), []
    for block in blocks:
        _require(type(block) is dict and len(block) == 2
                 and all(type(key) is str for key in block) and set(block) == {"name", "columns"}
                 and _identifier(block["name"]) and _ordered(block["columns"], positions),
                 "blocks require a name and nonempty dictionary-ordered columns")
        _require(seen.isdisjoint(block["columns"]), "core and blocks must be disjoint")
        seen.update(block["columns"])
        names.append(block["name"])
    _require(names == sorted(set(names)), "block names must be unique and lexically ordered")
    _require(seen == set(dictionary), "core and blocks must partition the full dictionary")


def _years(values):
    return (type(values) is list and 1 <= len(values) <= 64
            and all(type(year) is str and len(year) == 4
                    and re.fullmatch(r"[0-9]{4}", year) is not None
                    and 2 <= int(year) <= 9998 for year in values)
            and values == sorted(set(values)))


def _schedule(request):
    models, horizons = request["model_configuration_ids"], request["horizons"]
    _require(type(models) is list and 1 <= len(models) <= 256
             and all(_hash(model) for model in models) and models == sorted(set(models)),
             "model configuration IDs must be bounded, unique and sorted")
    _require(type(horizons) is list and len(horizons) == 3
             and all(type(horizon) is int for horizon in horizons) and horizons == [5, 3, 1],
             "horizons must be exactly integer [5, 3, 1]")
    outer, inner = request["outer_years"], request["inner_years"]
    _require(_years(outer) and type(inner) is dict and len(inner) == len(outer)
             and all(type(key) is str for key in inner) and set(inner) == set(outer),
             "ordered outer years require exactly matching inner-year keys")
    _require(all(_years(inner[year]) and inner[year][-1] < year for year in outer),
             "nonempty ordered inner years must strictly precede each outer year")


def _counts(request):
    for name in ("max_subsets", "max_candidates", "max_fits"):
        value = request[name]
        _require(type(value) is int and 1 <= value <= 2**31 - 1,
                 "caps must be explicit bounded positive integers")
    blocks = len(request["optional_blocks"])
    subsets = (1, 2, 4)[blocks] if blocks < 3 else 2 * blocks + 2
    models = len(request["model_configuration_ids"])
    candidates = subsets * models
    inners, outers = sum(map(len, request["inner_years"].values())), len(request["outer_years"])
    fits = len(request["horizons"]) * (candidates * inners + outers)
    _require(subsets <= request["max_subsets"] and candidates <= request["max_candidates"]
             and fits <= request["max_fits"], "subset, candidate or fit cap is insufficient")
    return {"subsets": subsets, "model_configurations": models, "candidates": candidates,
            "inner_folds": inners, "outer_folds": outers, "search_fits": fits}


def _subsets(request):
    dictionary, core = request["dictionary"], set(request["core"])
    full = set(dictionary)
    choices = {tuple(request["core"]), tuple(dictionary)}
    for block in request["optional_blocks"]:
        columns = set(block["columns"])
        for members in (core | columns, full - columns):
            choices.add(tuple(column for column in dictionary if column in members))
    result = []
    for columns in choices:
        subset = {"schema_version": "magnitude_feature_subset.v1",
                  "column_list_id": request["column_list_id"], "columns": list(columns)}
        result.append({**subset, "subset_id": content_hash(subset)})
    return sorted(result, key=lambda subset: subset["subset_id"])


def plan_subsets(request):
    """Preview an explicit static neighborhood, never an executable search plan."""
    _require(type(request) is dict and len(request) == len(_KEYS)
             and all(type(key) is str for key in request) and set(request) == _KEYS
             and type(request["schema_version"]) is str
             and request["schema_version"] == "magnitude_feature_subsets_request.v1",
             "request requires its closed v1 schema")
    _partition(request)
    _schedule(request)
    counts = _counts(request)
    _require(request["column_list_id"] == content_hash(request["dictionary"]),
             "column_list_id differs from the ordered dictionary names")
    request = json.loads(json.dumps(request, sort_keys=True))
    return {"schema_version": "magnitude_feature_subset_plan.v1", "request": request,
            "plan_id": content_hash(request), "horizons": list(request["horizons"]),
            "subsets": _subsets(request), "counts": counts}
