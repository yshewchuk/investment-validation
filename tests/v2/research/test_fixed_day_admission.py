"""Direct fixed-day admission shares shape validation but retains its own kind."""
import pytest

from engine.v2.data.errors import DataError
from engine.v2.foundation.experiment_exits import valid_exit_recipe
from engine.v2.research import experiment_exits


def assert_fixed_day_refusal(error):
    assert error.code == error.problem.code == "EXPERIMENT_VARIANT_FAILED"
    assert error.problem.category == "internal"
    assert error.problem.message == "invalid fixed-day exit economics"
    assert error.problem.retryable is False
    assert error.problem.details == {}


def test_direct_fixed_day_obeys_shared_admission_before_reads(monkeypatch):
    recipe = {"kind": "fixed_day", "trading_days": 2}
    seen = []

    def refuse_shared_recipe(value):
        seen.append(value)
        return False

    monkeypatch.setattr(experiment_exits, "valid_exit_recipe", refuse_shared_recipe)
    with pytest.raises(DataError) as caught:
        experiment_exits.walk_fixed_day(None, None, (),
                                        economic_params={"exit": recipe, "fill": 0.25})
    assert_fixed_day_refusal(caught.value)
    assert seen == [recipe]
    assert seen[0] is recipe


@pytest.mark.parametrize("units", [None, "quote_price_times_quantity", "entry_debit_fraction"])
def test_direct_fixed_day_still_rejects_valid_target_stop_recipe(units):
    recipe = {"kind": "target_stop", "trading_days": 2, "target_pnl": 0.5, "stop_pnl": -0.5}
    if units is not None:
        recipe["threshold_units"] = units
    assert valid_exit_recipe(recipe) is True
    with pytest.raises(DataError) as caught:
        experiment_exits.walk_fixed_day(None, None, (),
                                        economic_params={"exit": recipe, "fill": 0.25})
    assert_fixed_day_refusal(caught.value)
