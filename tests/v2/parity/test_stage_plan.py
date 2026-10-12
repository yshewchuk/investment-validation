"""The scorer-computed keys belong to the stage that produces them.

``engine.v2.scoring.stages`` writes all three of these during the score, and
the record comparator attributes a mismatch on each by looking the key up in
``SCORER_V1``:

* ``days_to_print`` — ``_stage_days_to_print``, from the resolved context
  (``event_date`` and the decision date ``as_of``) before the forecast, so it
  is a ``resolve_context`` value; a corrupt context date moves it together
  with ``as_of``/``event_date`` rather than looking like a separate root.
* ``dte_entry`` and ``entry_cost_pct`` — ``_check_pricing_boundary``, from the
  priced first leg and the quote date, so both belong to ``pricing``.

Listing ``dte_entry`` under ``resolve_context`` (its old home) blamed a pricing
mismatch on the context stage; these assertions pin the correct owners with
literal stage names so that cannot regress silently.
"""
from __future__ import annotations

from engine.v2.parity.stage_plan import SCORER_V1, load_stage_plan


def _stages_with(field: str) -> list[str]:
    return [stage.stage_id for stage in SCORER_V1.stages if field in stage.fields]


def test_scorer_computed_keys_each_belong_to_exactly_one_stage():
    expected = {
        "days_to_print": "resolve_context",
        "dte_entry": "pricing",
        "entry_cost_pct": "pricing",
    }
    for field, stage_id in expected.items():
        owners = _stages_with(field)
        assert owners == [stage_id], field
        assert SCORER_V1.stage_of(field) == stage_id, field


def test_scorer_computed_keys_attributed_to_producing_stage():
    expected = {
        "days_to_print": "resolve_context",
        "dte_entry": "pricing",
        "entry_cost_pct": "pricing",
    }
    for field, stage_id in expected.items():
        owners = _stages_with(field)
        assert owners == [stage_id], field
        assert SCORER_V1.stage_of(field) == stage_id, field

    resolve_context = next(
        stage for stage in SCORER_V1.stages if stage.stage_id == "resolve_context"
    )
    assert "dte_entry" not in resolve_context.fields


def test_dte_entry_is_not_a_resolve_context_field():
    resolve_context = next(
        stage for stage in SCORER_V1.stages if stage.stage_id == "resolve_context"
    )
    assert "dte_entry" not in resolve_context.fields


def test_the_plan_resolves_by_its_declared_id():
    assert load_stage_plan("scorer.v1") is SCORER_V1
