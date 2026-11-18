import pytest

from engine.v2.foundation import content_hash
from experiments.magnitude_search_refits import (
    failed_terminal,
    per_candidate,
    possible_slots,
    refit_key,
    refit_units,
    terminal_keys,
)

LEGACY = {"plan_id": "plan", "candidates": [{"candidate_id": "a"}, {"candidate_id": "b"}],
          "folds": [{"outer": {"year": "2024"}}, {"outer": {"year": "2025"}}]}


def _winner(extra=None):
    selection = {"winner_rules": ["overall", "rmse", "mae"]}
    if extra is not None:
        selection["extra_refits"] = extra
    return {"plan_id": "plan", "selection": selection,
            "candidates": [{"candidate_id": value} for value in ("a", "b", "c", "d")],
            "folds": [{"outer": {"year": "2024"}}, {"outer": {"year": "2025"}}]}


def _item(overall, winners):
    return {"selected_candidate_id": overall, "distinct_winner_ids": sorted(winners)}


def test_legacy_plan_keeps_one_unit_and_the_original_key_per_fold():
    assert per_candidate(LEGACY) is False
    assert refit_units(LEGACY, {"selected_candidate_id": "a"}) == [None]
    assert refit_key(LEGACY, "2025") == content_hash(["plan", "2025"])
    assert possible_slots(LEGACY) == [
        ("2024", None, content_hash(["plan", "2024"])),
        ("2025", None, content_hash(["plan", "2025"])),
    ]


def test_winner_units_put_the_overall_winner_first_then_sorted_winners_and_extras():
    plan = _winner(extra=["d", "b", "a"])
    assert per_candidate(plan) is True
    assert refit_units(plan, _item("c", ["c", "b", "a"])) == ["c", "a", "b", "d"]


def test_shared_winners_and_extras_are_deduped_to_one_unit():
    plan = _winner(extra=["a"])
    assert refit_units(plan, _item("a", ["a", "a", "a"])) == ["a"]


@pytest.mark.parametrize("plan, item", [
    (_winner(), {"selected_candidate_id": "c", "distinct_winner_ids": "c"}),
    (_winner(), {"selected_candidate_id": 1, "distinct_winner_ids": ["c"]}),
    (_winner(extra="d"), {"selected_candidate_id": "c", "distinct_winner_ids": ["c"]}),
])
def test_malformed_selection_is_rejected(plan, item):
    with pytest.raises(ValueError, match="malformed refit selection"):
        refit_units(plan, item)


def test_winner_possible_slots_cover_every_fold_and_candidate_with_distinct_keys():
    plan = _winner()
    slots = possible_slots(plan)
    assert len(slots) == len(plan["folds"]) * len(plan["candidates"])
    assert {(year, candidate_id) for year, candidate_id, _ in slots} == {
        (fold["outer"]["year"], candidate["candidate_id"])
        for fold in plan["folds"] for candidate in plan["candidates"]}
    keys = [key for _, _, key in slots]
    assert len(set(keys)) == len(keys)
    legacy = {content_hash(["plan", fold["outer"]["year"]]) for fold in plan["folds"]}
    assert not legacy & set(keys)
    for year, candidate_id, key in slots:
        assert key == refit_key(plan, year, candidate_id)


def test_terminal_keys_list_every_fold_unit_with_the_overall_first():
    plan = _winner()
    selection = {"selections": [
        {"outer_year": "2024", "selected_candidate_id": "c", "distinct_winner_ids": ["b", "c"]},
        {"outer_year": "2025", "selected_candidate_id": "a", "distinct_winner_ids": ["a"]},
    ]}
    assert terminal_keys(plan, selection) == [
        refit_key(plan, "2024", "c"),
        refit_key(plan, "2024", "b"),
        refit_key(plan, "2025", "a"),
    ]


def test_failed_terminal_has_the_exact_durable_shape():
    assert failed_terminal("key", "2024", "candidate", "inputs") == {
        "schema_version": "magnitude_search_outer_failed.v1",
        "outer_id": "key",
        "outer_year": "2024",
        "candidate_id": "candidate",
        "inputs_id": "inputs",
        "status": "failed",
        "code": "EXPERIMENT_VARIANT_FAILED",
    }
