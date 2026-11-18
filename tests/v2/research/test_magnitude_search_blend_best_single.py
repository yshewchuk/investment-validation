"""Exact-value oracles for the best non-blend (best-single) selection."""
import pytest

from engine.v2.ops.errors import OpsError
from experiments.magnitude_search_blends import best_non_blend_id


def _score(candidate_id, blend_score, kind=None):
    row = {"candidate_id": candidate_id, "blend_score": blend_score}
    if kind is not None:
        row["kind"] = kind
    return row


def test_lowest_scoring_blend_is_ignored():
    eligible = [
        _score("blend:x", 0.10, kind="blend"),
        _score("c-beta", 0.40),
        _score("c-alpha", 0.30),
    ]
    assert best_non_blend_id(eligible) == "c-alpha"


def test_non_blend_without_kind_key_counts_as_non_blend():
    eligible = [_score("blend:x", 0.05, kind="blend"), _score("c-only", 0.9)]
    assert best_non_blend_id(eligible) == "c-only"


def test_blend_score_tie_breaks_to_lowest_candidate_id():
    eligible = [
        _score("c-zeta", 0.25),
        _score("c-alpha", 0.25),
        _score("c-mid", 0.25),
        _score("blend:x", 0.01, kind="blend"),
    ]
    assert best_non_blend_id(eligible) == "c-alpha"


def test_lowest_blend_score_wins_over_tie_breaking_id():
    eligible = [_score("c-alpha", 0.5), _score("c-zeta", 0.2)]
    assert best_non_blend_id(eligible) == "c-zeta"


def test_only_blend_rows_raise_variant_failed():
    eligible = [_score("blend:x", 0.1, kind="blend"), _score("blend:y", 0.2, kind="blend")]
    with pytest.raises(OpsError) as caught:
        best_non_blend_id(eligible)
    assert caught.value.problem.code == "EXPERIMENT_VARIANT_FAILED"
    assert caught.value.problem.message == "all candidates failed for an outer fold"


def test_no_rows_raise_variant_failed():
    with pytest.raises(OpsError) as caught:
        best_non_blend_id([])
    assert caught.value.problem.code == "EXPERIMENT_VARIANT_FAILED"
