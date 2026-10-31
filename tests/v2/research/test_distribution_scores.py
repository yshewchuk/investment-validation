"""Pure score/selection controls; observations are never decision inputs."""
from copy import deepcopy
from math import fsum, log
from types import SimpleNamespace

import pytest

from engine.v2.foundation import content_hash
from engine.v2.research.magnitude_distribution import histogram
from engine.v2.research.magnitude_distribution_math import bin_scores, crps, empirical_crps, mean
from experiments.distribution_scores import arm_forecasts, economic_forecast, score_forecasts


def distribution(weights=None, tail=(0,)):
    masses = [0.0] * 81
    for index, mass in (weights or {0: 1}).items():
        masses[index] = mass
    return histogram(masses, tail_excesses=list(tail) if tail is not None else None)


def destination(*, outer_year="2026", horizon=1):
    return dict(plan_id="plan", admission_id="admission", recipe_id="recipe", context_id="context",
                fold_id="fold", inputs_id="inputs", budget_id="budget", role="outer",
                outer_year=outer_year, inner_year=None, horizon=horizon)


def receipt(scope=None):
    scope = destination() if scope is None else scope
    claim = {key: scope[key] for key in ("plan_id", "admission_id", "role", "outer_year", "inner_year")}
    claim["trial"] = {key: scope[key] for key in
        ("recipe_id", "context_id", "fold_id", "inputs_id", "budget_id", "horizon", "role")}
    return dict(status="completed", claim=claim)


def pool(*, residuals=(-.125, .875), status="ready", outer_year="2026", horizon=1):
    # A synthetic numerical pool, not a claimed authenticated historical pool.
    body = dict(schema_version="distribution_oof_pool.v1", policy="earlier_inner_equal_weight_per_horizon.v1",
        point_status=status, histogram_status=status, point_residuals=list(residuals),
        histogram_residuals=list(residuals), destination=destination(outer_year=outer_year, horizon=horizon))
    return {**body, "pool_id": content_hash(body)}


def forecast(document=None, point=.125, event="e", *, scope=None):
    document = document or distribution()
    return dict(rows=[dict(event_id=event, probabilities=document["probabilities"], point_prediction=point)],
                tail=document["tail"], receipt=receipt(scope))


def scores(result, control=None, target=.125):
    rows = score_forecasts(result, control or pool(), {row["event_id"]: target for row in result["rows"]},
                          outer_year="2026", horizon=1)
    return {row["arm"]: row for row in rows}


def test_uniform_and_empirical_scores_use_existing_proper_scoring_oracles():
    result, control = forecast(), pool()
    actual = scores(result, control)
    assert actual["histogram"]["crps"] == pytest.approx(1 / 48)
    values = arm_forecasts(result["rows"][0], result["tail"], control, receipt=result["receipt"])
    assert values["matched_point"] == [0, 1]
    assert actual["matched_point"]["crps"] == empirical_crps([0, 1], .125) == .25
    assert actual["same_mean"]["crps"] == empirical_crps(values["same_mean"], .125)
    assert mean(values["histogram"]) == fsum(values["same_mean"]) / len(values["same_mean"])
    assert actual["histogram"]["rps"] == 0 and actual["histogram"]["log_loss"] == 0


def test_impossible_bin_uses_explicit_infinite_log_loss_not_json_nan():
    actual = scores(forecast(), target=.25)
    assert actual["histogram"]["log_loss"] == "infinite"
    assert actual["histogram"]["rps"] == 1
    assert actual["histogram"]["crps"] == pytest.approx(1 / 12)


def test_empirical_zero_atoms_remain_atoms_not_uniform_first_bin():
    actual = scores(forecast(point=.125), target=0)
    assert actual["matched_point"]["crps"] == .25
    assert actual["matched_point"]["log_loss"] == pytest.approx(log(2))
    assert actual["histogram"]["crps"] == pytest.approx(1 / 12)


@pytest.mark.parametrize("observation", [5, 10, 20, 25])
def test_thresholds_are_inclusive_and_empirical_tail_duplicates_retained(observation):
    document = distribution({0: .25, 20: .25, 40: .25, 80: .25}, tail=(0, 0, 5))
    actual = scores(forecast(document), pool(status="no_earlier_inner"), target=observation)["histogram"]
    assert actual["tails"] == [dict(threshold=t, probability_ge=p, observed_ge=observation >= t)
        for t, p in [(5, .75), (10, .5), (20, .25)]]
    assert actual["crps"] == crps(document, observation)
    assert actual["log_loss"] == bin_scores(document, observation)["log_loss"]


def test_tail_boundary_twenty_is_overflow_not_last_finite_bin():
    actual = scores(forecast(distribution({80: 1}, tail=(0, 0, 2))), target=20)["histogram"]
    assert actual["log_loss"] == 0 and actual["rps"] == 0
    assert actual["tails"][-1] == dict(threshold=20, probability_ge=1, observed_ge=True)
    assert actual["crps"] == pytest.approx(2 / 9)


@pytest.mark.parametrize("weights", [{0: 1}, {80: 1}])
def test_unresolved_tail_preserves_bin_scores_but_no_full_crps_or_economics(weights):
    document = distribution(weights, tail=None)
    actual = scores(forecast(document), target=20)
    assert actual["histogram"]["status"] == "available"
    assert actual["histogram"]["crps"] is None
    assert actual["same_mean"]["status"] == "unavailable"
    assert actual["matched_point"]["status"] == "available"
    assert economic_forecast(document, None, SimpleNamespace(spot=100, cost=5), scale=100, round_trip_cost=2) is None


def test_unavailable_earlier_controls_retain_every_arm_row_without_fallback():
    actual = scores(forecast(), pool(status="no_earlier_inner"))
    assert set(actual) == {"histogram", "same_mean", "matched_point"}
    assert actual["histogram"]["status"] == "available"
    assert all(actual[name]["status"] == "unavailable" for name in ("same_mean", "matched_point"))


def test_missing_outcome_keeps_explicit_unavailable_score_rows():
    actual = scores(forecast(), target={"omitted": "label_not_available"})
    assert len(actual) == 3
    assert all(row["status"] == "missing_target" and row["observation"] is None
        and row["reason"] == "label_not_available" and "crps" not in row for row in actual.values())


def test_output_event_horizon_identity_and_input_immutability():
    result, control = forecast(scope=destination(horizon=5)), pool(horizon=5)
    before = deepcopy((result, control))
    outputs = list(score_forecasts(result, control, {"e": .125}, outer_year="2026", horizon=5))
    assert all(row["event_id"] == "e" and row["horizon"] == 5 and row["outer_year"] == "2026" for row in outputs)
    assert (result, control) == before


@pytest.mark.parametrize("section,key", [("claim", key) for key in
    ("plan_id", "admission_id", "role", "outer_year", "inner_year")] + [("trial", key) for key in
    ("recipe_id", "context_id", "fold_id", "inputs_id", "budget_id", "horizon", "role")])
@pytest.mark.parametrize("empty", [False, True])
def test_same_label_foreign_receipt_refuses_before_target_access(section, key, empty):
    from engine.v2.ops.errors import OpsError

    result, control = forecast(), pool()
    claim = result["receipt"]["claim"]
    binding = claim if section == "claim" else claim["trial"]
    binding[key] = "foreign"
    if empty:
        result["rows"] = []
    before = deepcopy((result, control))
    with pytest.raises(OpsError) as caught:
        list(score_forecasts(result, control, {"e": .125}, outer_year="2026", horizon=1))
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.retryable is False
    assert (result, control) == before


@pytest.mark.parametrize("mutation", [
    lambda result, control: result.pop("receipt"),
    lambda result, control: result.update(receipt=None),
    lambda result, control: result["receipt"].update(status="failed"),
    lambda result, control: result["receipt"].update(claim=None),
    lambda result, control: result["receipt"]["claim"].update(trial=None),
    lambda result, control: result["receipt"]["claim"].pop("admission_id"),
    lambda result, control: result["receipt"]["claim"]["trial"].pop("inputs_id"),
    lambda result, control: control["destination"].pop("plan_id"),
])
def test_missing_binding_refuses_before_row_or_target_access(mutation):
    from engine.v2.ops.errors import OpsError

    result, control = forecast(), pool()
    mutation(result, control)
    result.pop("rows")
    with pytest.raises(OpsError) as caught:
        list(score_forecasts(result, control, None, outer_year="2026", horizon=1))
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"


def test_economic_controls_refuse_foreign_receipt_before_numeric_work(monkeypatch):
    from engine.v2.ops.errors import OpsError
    from experiments import distribution_scores as module

    result, control = forecast(), pool()
    result["receipt"]["claim"]["plan_id"] = "foreign"
    monkeypatch.setattr(module, "control_for_prediction", lambda *args: pytest.fail("controls reached"))
    with pytest.raises(OpsError) as caught:
        arm_forecasts(None, None, control, receipt=result["receipt"])
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"


@pytest.mark.parametrize("outer_year,horizon", [("2024", 1), ("2023", 5)])
@pytest.mark.parametrize("empty", [False, True])
def test_score_refuses_actual_residual_pool_from_another_scope_before_scoring(monkeypatch, outer_year, horizon, empty):
    from engine.v2.ops.errors import OpsError
    from experiments import distribution_scores as module
    from experiments.distribution_oof import residual_pool
    from tests.v2.research.test_distribution_oof import Fits

    control = residual_pool(Fits(), "outer", "2023", 1)
    result = forecast(scope=control["destination"])
    if empty:
        result["rows"] = []
    before = deepcopy((result, control))
    monkeypatch.setattr(module, "arm_forecasts", lambda *args, **kwargs: pytest.fail("scope mismatch reached scoring"))
    module._require_pool_binding(result["receipt"], control)  # Isolate the destination-scope guard.
    rows = score_forecasts(result, control, None, outer_year=outer_year, horizon=horizon)
    with pytest.raises(OpsError) as caught:
        next(rows)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.retryable is False
    assert (result, control) == before


@pytest.mark.parametrize("role,inner_year", [("inner", None), ("outer", "2022"), ("inner", "2022")])
@pytest.mark.parametrize("empty", [False, True])
def test_score_refuses_matching_receipt_pool_outside_outer_role_before_targets(role, inner_year, empty):
    from engine.v2.ops.errors import OpsError
    from experiments import distribution_scores as module

    control = pool()
    control["destination"].update(role=role, inner_year=inner_year)
    control["pool_id"] = content_hash({key: value for key, value in control.items() if key != "pool_id"})
    result = forecast(scope=control["destination"])
    if empty:
        result["rows"] = []
    before = deepcopy((result, control))
    module._require_pool_binding(result["receipt"], control)  # The preceding guard must pass.
    with pytest.raises(OpsError) as caught:
        next(score_forecasts(result, control, None, outer_year="2026", horizon=1))
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.retryable is False
    assert (result, control) == before


@pytest.mark.parametrize("horizon", [5, 3, 1])
def test_score_accepts_honest_current_residual_pool_under_both_guards(horizon):
    from engine.models.no_fit import no_fit_guard as legacy_guard
    from engine.v2.models.no_fit import no_fit_guard as native_guard
    from experiments.distribution_oof import residual_pool
    from tests.v2.research.test_distribution_oof import Fits

    control = residual_pool(Fits(), "outer", "2023", horizon)
    result = forecast(distribution({80: 1}), point=20, scope=control["destination"])
    before = deepcopy((result, control))
    with legacy_guard(), native_guard():
        rows = list(score_forecasts(result, control, {"e": 20}, outer_year="2023", horizon=horizon))
    assert {row["arm"] for row in rows} == {"histogram", "same_mean", "matched_point"}
    assert all(row["status"] == "available" and row["outer_year"] == "2023" and row["horizon"] == horizon for row in rows)
    assert (result, control) == before


def test_matching_mean_has_identical_after_cost_linear_ev_despite_profit_probability_difference():
    result = forecast(distribution({8: 1}), point=2.125)
    values = arm_forecasts(result["rows"][0], result["tail"], pool(residuals=(-2, 2)), receipt=result["receipt"])
    bridge = SimpleNamespace(intercept=.1, slope=.01, residuals=(0,))
    decision = SimpleNamespace(spot=100, cost=12)
    histogram_result, control_result = [economic_forecast(values[name], bridge, decision,
        scale=200, round_trip_cost=20) for name in ("histogram", "same_mean")]
    assert histogram_result["expected_net_profit_usd"] == pytest.approx(5)
    assert control_result["expected_net_profit_usd"] == pytest.approx(5)
    assert histogram_result["probability_profit"] == pytest.approx(.6)
    assert control_result["probability_profit"] == .5
    # Costs are subtracted exactly once, not multiplied a second time.
    no_fees = economic_forecast(values["histogram"], bridge, decision, scale=200, round_trip_cost=0)
    assert no_fees["expected_net_profit_usd"] - histogram_result["expected_net_profit_usd"] == pytest.approx(20)


def test_after_cost_zero_profit_is_not_a_profitable_atom():
    bridge, decision = SimpleNamespace(intercept=.1, slope=0, residuals=(0,)), SimpleNamespace(spot=100, cost=9)
    result = economic_forecast([0, 5], bridge, decision, scale=200, round_trip_cost=200)
    assert result["expected_net_profit_usd"] == 0 and result["probability_profit"] == 0


def test_negative_post_residual_exit_remains_diagnostic_not_clipped():
    bridge, decision = SimpleNamespace(intercept=.1, slope=0, residuals=(-.2, .2)), SimpleNamespace(spot=100, cost=9)
    result = economic_forecast([1], bridge, decision, scale=100, round_trip_cost=0)
    assert result["probability_negative_exit"] == .5
    assert result["expected_net_profit_usd"] == 100


def test_missing_decision_or_arm_is_unavailable_not_zero_ev():
    assert economic_forecast(None, None, None, scale=100, round_trip_cost=0) is None
    assert economic_forecast(distribution(), None, None, scale=100, round_trip_cost=0) is None


def test_optional_incumbent_draws_are_not_synthesized_from_gate_decision():
    result = forecast()
    values = arm_forecasts(result["rows"][0], result["tail"], pool(),
                          dict(incumbent_magnitudes=[0, 3], gate_selected=True), receipt=result["receipt"])
    assert values["incumbent"] == [0, 3] and "gate" not in values
    values = arm_forecasts(result["rows"][0], result["tail"], pool(), receipt=result["receipt"])
    assert "incumbent" not in values


def test_admitted_roundoff_normalization_applies_to_tail_probabilities_too():
    document = distribution({0: .5 - 5e-13, 80: .5})
    actual = scores(forecast(document), pool(status="no_earlier_inner"), target=20)["histogram"]
    assert actual["tails"][-1]["probability_ge"] == pytest.approx(.5 / fsum(document["probabilities"]), abs=1e-15, rel=0)


def test_large_concentrated_empirical_controls_count_before_mass_division():
    actual = scores(forecast(), pool(residuals=[0.0] * 100_000))
    for arm in ("same_mean", "matched_point"):
        assert actual[arm]["status"] == "available"
        assert actual[arm]["rps"] == actual[arm]["log_loss"] == actual[arm]["crps"] == 0


def test_profit_scores_include_unselected_opportunities_and_after_cost_truth():
    from experiments.distribution_scores import profit_scores

    intents = [{"event_id": "E", "arms": {"histogram": {"selected": False}},
                "forecasts": {"histogram": {"probability_profit": .75}, "same_mean": None}}]
    rows = list(profit_scores(intents, {"E": {"entry_cash": 110, "exit_cash": 105}}))
    assert rows[0]["observed_profit"] is False and rows[0]["brier"] == .75 ** 2
    assert rows[1]["observed_profit"] is False and rows[1]["brier"] is None


@pytest.mark.parametrize("quotes", [{}, {"E": {"entry_cash": None, "exit_cash": 105}},
                                    {"E": {"entry_cash": 110, "exit_cash": None}}])
def test_profit_scores_missing_entry_or_exit_is_not_a_false_loss(quotes):
    from experiments.distribution_scores import profit_scores

    row, = profit_scores([{"event_id": "E", "forecasts": {"histogram": {"probability_profit": .75}}}], quotes)
    assert row["observed_profit"] is None and row["brier"] is None and row["probability_profit"] == .75


@pytest.mark.parametrize("changes", [dict(scale=0), dict(scale=-1), dict(scale=True),
    dict(scale=float("inf")), dict(round_trip_cost=-1), dict(round_trip_cost=True),
    dict(round_trip_cost=float("nan"))])
def test_economic_units_require_positive_scale_and_nonnegative_total_cost(changes):
    from engine.v2.ops.errors import OpsError

    bridge, decision = SimpleNamespace(intercept=.1, slope=0, residuals=(0,)), SimpleNamespace(spot=100, cost=9)
    with pytest.raises(OpsError) as caught:
        economic_forecast([0, 5], bridge, decision, **(dict(scale=100, round_trip_cost=0) | changes))
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"


@pytest.mark.parametrize("availability", ["available", "missing_arm", "missing_decision", "missing_both", "unresolved_tail"])
@pytest.mark.parametrize("changes", [dict(scale=0), dict(scale=-1), dict(scale=True), dict(scale=None),
    dict(scale="100"), dict(scale=float("nan")), dict(scale=float("inf")), dict(scale=float("-inf")),
    dict(round_trip_cost=-1), dict(round_trip_cost=True), dict(round_trip_cost=None),
    dict(round_trip_cost="0"), dict(round_trip_cost=float("nan")),
    dict(round_trip_cost=float("inf")), dict(round_trip_cost=float("-inf"))])
def test_explicit_economics_refuse_before_every_availability_branch(availability, changes):
    from engine.v2.ops.errors import OpsError

    class PoisonBridge:
        def __getattribute__(self, name):
            raise AssertionError("invalid economics reached bridge")
    value, decision = [0, 5], SimpleNamespace(spot=100, cost=9)
    if availability in ("missing_arm", "missing_both"):
        value = None
    if availability in ("missing_decision", "missing_both"):
        decision = None
    if availability == "unresolved_tail":
        value = distribution(tail=None)
    with pytest.raises(OpsError) as caught:
        economic_forecast(value, PoisonBridge(), decision, **(dict(scale=100, round_trip_cost=0) | changes))
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.retryable is False
    assert caught.value.problem.message == "invalid distribution run contract"


@pytest.mark.parametrize("availability", ["available", "missing_arm", "missing_decision", "missing_both", "unresolved_tail"])
@pytest.mark.parametrize("scale,cost,profit,probability", [(100, 0, 100, 1), (.5, 1.25, -.75, 0)])
def test_valid_explicit_economics_keep_availability_and_usd_results(availability, scale, cost, profit, probability):
    bridge, decision = SimpleNamespace(intercept=.1, slope=0, residuals=(0,)), SimpleNamespace(spot=100, cost=9)
    value = distribution(tail=None) if availability == "unresolved_tail" else [0, 5]
    if availability in ("missing_arm", "missing_both"):
        value = None
    if availability in ("missing_decision", "missing_both"):
        decision = None
    before = deepcopy((value, bridge, decision))
    result = economic_forecast(value, bridge, decision, scale=scale, round_trip_cost=cost)
    if availability == "available":
        assert result["expected_net_profit_usd"] == profit
        assert result["probability_profit"] == probability
    else:
        assert result is None
    assert (value, bridge, decision) == before


def test_score_progress_brackets_iteration_and_ticks_once_per_event(monkeypatch):
    from contextlib import contextmanager

    import experiments.distribution_scores as module

    events = []
    @contextmanager
    def stage(name):
        events.append(("start", name))
        yield
        events.append(("end", name))
    class Progress:
        def __init__(self, name, total):
            events.append(("total", total))
        def tick(self):
            events.append(("tick",))
    monkeypatch.setattr(module, "stage", stage)
    monkeypatch.setattr(module, "Progress", Progress)
    assert len(scores(forecast())) == 3
    assert events == [("total", 1), ("start", "distribution proper scores"),
                      ("tick",), ("end", "distribution proper scores")]


@pytest.mark.parametrize("quotes", [{}, {"E": {"entry_cash": None, "exit_cash": 105}},
    {"E": {"entry_cash": 110, "exit_cash": None}}, {"E": {"entry_cash": None, "exit_cash": None}},
    {"E": {"entry_cash": 110, "exit_cash": 105}}])
@pytest.mark.parametrize("probability", [True, False, -.1, 1.1, float("nan"), float("inf"),
    -float("inf"), ".5", None, complex(.5), 10 ** 1000])
def test_profit_probability_refuses_before_outcome_availability(quotes, probability):
    from engine.v2.data.errors import DataError
    from experiments.distribution_scores import profit_scores

    with pytest.raises(DataError) as caught:
        list(profit_scores([{"event_id": "E", "forecasts": {"histogram": {"probability_profit": probability}}}], quotes))
    assert caught.value.code == "CONTRACT_MISMATCH"
    assert caught.value.problem.retryable is False
    assert caught.value.problem.message == "invalid magnitude distribution arithmetic"


@pytest.mark.parametrize("leg", ["entry_cash", "exit_cash"])
@pytest.mark.parametrize("other", [None, 100])
@pytest.mark.parametrize("forecast", [None, {"probability_profit": .75}])
@pytest.mark.parametrize("value", ["9", True, False, float("nan"), float("inf"), -float("inf"), complex(5), []])
def test_profit_cash_refuses_before_availability_or_forecast_branch(leg, other, forecast, value):
    from engine.v2.ops.errors import OpsError
    from experiments.distribution_scores import profit_scores

    quote = dict(entry_cash=other, exit_cash=other)
    quote[leg] = value
    with pytest.raises(OpsError) as caught:
        list(profit_scores([{"event_id": "E", "forecasts": {"histogram": forecast}}], {"E": quote}))
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.retryable is False
    assert caught.value.problem.message == "invalid distribution run contract"


def test_profit_cash_strings_never_form_a_lexicographic_profit():
    from engine.v2.ops.errors import OpsError
    from experiments.distribution_scores import profit_scores

    with pytest.raises(OpsError) as caught:
        list(profit_scores([{"event_id": "E", "forecasts": {"histogram": {"probability_profit": .25}}}],
                           {"E": {"entry_cash": "100", "exit_cash": "9"}}))
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"


@pytest.mark.parametrize("quote,observed,loss", [
    ({"entry_cash": 100, "exit_cash": -3}, False, .0625),
    ({"entry_cash": 100, "exit_cash": 100}, False, .0625),
    ({"entry_cash": 100, "exit_cash": 101}, True, .5625),
    ({"entry_cash": 0, "exit_cash": 0}, False, .0625),
    ({"entry_cash": 10 ** 400, "exit_cash": 10 ** 400 + 1}, True, .5625),
    ({"entry_cash": -10 ** 400, "exit_cash": -10 ** 400 - 1}, False, .0625),
    ({"entry_cash": None, "exit_cash": -3}, None, None),
    ({"entry_cash": 100, "exit_cash": None}, None, None),
    (None, None, None),
])
def test_profit_valid_total_cash_and_unknown_truth_preserve_exact_output(quote, observed, loss):
    from experiments.distribution_scores import profit_scores

    quotes = {} if quote is None else {"E": quote}
    intents = [{"event_id": "E", "forecasts": {"histogram": {"probability_profit": .25}, "same_mean": None}}]
    before = deepcopy((intents, quotes))
    assert list(profit_scores(intents, quotes)) == [
        {"event_id": "E", "arm": "histogram", "observed_profit": observed, "probability_profit": .25, "brier": loss},
        {"event_id": "E", "arm": "same_mean", "observed_profit": observed, "probability_profit": None, "brier": None},
    ]
    assert (intents, quotes) == before
