import pytest

from experiments.magnitude_search_refits import blend_components, refit_units, required_units

BLENDS = [{"blend_id": "blend_ab", "name": "ab", "component_candidate_ids": ["b", "a"]},
          {"blend_id": "blend_cd", "name": "cd", "component_candidate_ids": ["d", "c"]}]


def _plan(extra=None, blends=True):
    selection = {"winner_rules": ["overall", "rmse", "mae"]}
    if extra is not None:
        selection["extra_refits"] = extra
    plan = {"plan_id": "plan", "selection": selection,
            "candidates": [{"candidate_id": value} for value in ("a", "b", "c", "d", "e", "f")],
            "folds": [{"outer": {"year": "2024"}}]}
    if blends:
        plan["blends"] = BLENDS
    return plan


def _item(overall, winners=(), best=None):
    item = {"selected_candidate_id": overall, "distinct_winner_ids": list(winners)}
    if best is not None:
        item["best_non_blend_id"] = best
    return item


def test_blend_components_returns_sorted_components_of_a_blend_only():
    assert blend_components(_plan(), "blend_ab") == ["a", "b"]
    assert blend_components(_plan(), "c") is None


def test_blend_components_is_none_without_a_blends_key():
    assert blend_components(_plan(blends=False), "blend_ab") is None


def test_required_units_expands_a_blend_overall_and_keeps_a_plain_overall():
    assert required_units(_plan(), _item("blend_cd")) == ["c", "d"]
    assert required_units(_plan(), _item("c")) == ["c"]


def test_blend_overall_expands_to_sorted_components_first():
    assert refit_units(_plan(), _item("blend_ab", ["blend_ab"], best="a")) == ["a", "b"]


def test_blend_overall_precedes_other_winners_and_the_best_non_blend():
    plan = _plan()
    item = _item("blend_ab", ["blend_cd", "e"], best="f")
    assert refit_units(plan, item) == ["a", "b", "c", "d", "e", "f"]


def test_winners_expand_and_dedupe_with_the_overall_unit():
    plan = _plan()
    item = _item("c", ["blend_cd", "blend_ab"], best="e")
    assert refit_units(plan, item) == ["c", "a", "b", "d", "e"]


def test_best_non_blend_follows_the_winners_and_precedes_the_extras():
    plan = _plan(extra=["f", "b"])
    item = _item("c", ["a"], best="e")
    assert refit_units(plan, item) == ["c", "a", "e", "b", "f"]


def test_best_non_blend_already_a_component_is_deduped():
    plan = _plan()
    item = _item("blend_ab", ["blend_ab"], best="b")
    assert refit_units(plan, item) == ["a", "b"]


@pytest.mark.parametrize("item", [
    _item("c", ["c"]),
    _item("c", ["c"], best=3),
])
def test_blend_plans_require_a_string_best_non_blend_id(item):
    with pytest.raises(ValueError, match="malformed refit selection"):
        refit_units(_plan(), item)


def test_legacy_plan_units_are_unchanged_without_blends():
    assert refit_units({"plan_id": "plan", "selection": {}}, {"selected_candidate_id": "a"}) == [None]
    plan = _plan(extra=["f"], blends=False)
    assert refit_units(plan, _item("c", ["c", "a"])) == ["c", "a", "f"]
