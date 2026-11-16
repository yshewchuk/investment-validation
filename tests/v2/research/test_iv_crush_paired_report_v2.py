"""Independent four-arm IV crush v2 coverage, abstention and bootstrap oracles."""
import math

import pytest

from engine.v2.foundation import content_hash
from experiments.iv_crush_paired_report import paired_report, paired_report_v2


def case():
    events = [{"event_id": e, "ticker": "AAA" if i < 2 else "BBB",
               "event_date": f"202{year}-0{month}-02"}
              for i, (e, year, month) in enumerate(
                  (("a", 3, 6), ("b", 3, 6), ("c", 4, 7), ("d", 4, 7)))]
    targets = {"a": -140., "b": 20., "c": 60., "d": -40.}
    challenger = [{"event_id": e, "horizon_sessions": h, "predicted_crush_pct_iv30": targets[e] + 2.}
                  for e in targets for h in (1, 3, 5)]
    incumbent = [{"event_id": e, "horizon_sessions": h, "predicted_crush_pct_iv30": targets[e] + 4.,
                  "abstention": None, "training_mean": -60.} for e in targets for h in (1, 3, 5)]
    implied = {e: {"iv30": 50., "exern_iv30": 40.} for e in targets}
    return events, challenger, incumbent, targets, implied


def test_implied_arm_value_identity_order_and_v1_unchanged():
    events, challenger, incumbent, targets, implied = case()
    result = paired_report_v2(events, challenger, incumbent, targets, implied)
    assert result["result_id"] == content_hash(
        {key: value for key, value in result.items() if key != "result_id"})
    arm = result["arm_metrics"]["implied_ex_earnings"]
    assert arm["bias"] == pytest.approx(5.0)
    assert arm["count"] == 12
    assert result["common_matched"]["metrics"]["implied_ex_earnings"] == arm
    shuffled = ([*reversed(events)], [*reversed(challenger)], [*reversed(incumbent)],
                targets, implied)
    assert paired_report_v2(*shuffled) == result
    assert paired_report(events, challenger, incumbent, targets)["schema_version"] == \
        "iv_crush_paired_report.v1"


def test_implied_abstention_shrinks_implied_and_common_sets_without_imputation():
    events, challenger, incumbent, targets, implied = case()
    implied["a"] = {"iv30": None, "exern_iv30": 40.}
    implied["b"] = {"iv30": 50., "exern_iv30": 0.}
    result = paired_report_v2(events, challenger, incumbent, targets, implied)

    implied_metrics = result["arm_metrics"]["implied_ex_earnings"]
    assert implied_metrics["count"] == 6
    assert implied_metrics["bias"] == pytest.approx(-30.0)
    assert result["common_matched_events"] == 2
    assert result["common_matched"]["events"] == 2
    assert result["common_matched"]["metrics"]["implied_ex_earnings"]["count"] == 6
    for arm in ("challenger", "incumbent", "training_mean"):
        assert result["arm_metrics"][arm]["count"] == 12
    statuses = {}
    for row in result["coverage"]:
        if row["arm"] == "implied_ex_earnings":
            statuses[row["forecast_status"]] = (
                statuses.get(row["forecast_status"], 0) + row["count"])
    assert statuses["abstained_missing_or_nonpositive_input"] == 6
    assert statuses["predicted"] == 6


def test_incumbent_horizon_abstention_removes_event_from_common_and_incumbent_sets():
    events, challenger, incumbent, targets, implied = case()
    row = next(r for r in incumbent
               if r["event_id"] == "a" and r["horizon_sessions"] == 1)
    row.update(abstention="unavailable_input", predicted_crush_pct_iv30=None)
    result = paired_report_v2(events, challenger, incumbent, targets, implied)

    assert result["forecast_matched_events"] == 3
    assert result["arm_metrics"]["incumbent"]["count"] == 9
    assert result["arm_metrics"]["implied_ex_earnings"]["count"] == 12
    assert result["common_matched_events"] == 3
    assert result["common_matched"]["metrics"]["incumbent"]["count"] == 9
    assert any(row["arm"] == "incumbent"
               and row["forecast_status"] == "unavailable_input"
               and row["count"] == 1 for row in result["coverage"])


def test_bootstrap_clusters_pairs_and_deterministic_intervals():
    args = case()
    result = paired_report_v2(*args)
    assert set(result["bootstrap"]) == {"ticker", "event_month"}
    names = {"challenger_minus_incumbent", "challenger_minus_training_mean",
             "challenger_minus_implied_ex_earnings", "incumbent_minus_training_mean"}
    for cluster in result["bootstrap"].values():
        assert set(cluster) == names
        for pair in cluster.values():
            assert pair["resamples"] == 2000 and pair["seed"] == 0
            assert set(pair["percentile_95"]) == {"rmse", "mae"}
            for interval in pair["percentile_95"].values():
                assert len(interval) == 2
                assert all(isinstance(value, float) and math.isfinite(value)
                           for value in interval)
                assert interval[0] <= interval[1]
    assert result["bootstrap"] == paired_report_v2(*args)["bootstrap"]


@pytest.mark.parametrize("mutation", ["missing_event", "extra_event", "bool", "nan",
                                      "extra_key", "challenger_coverage"])
def test_invalid_implied_or_coverage_refuses(mutation):
    events, challenger, incumbent, targets, implied = case()
    if mutation == "missing_event":
        del implied["a"]
    elif mutation == "extra_event":
        implied["e"] = {"iv30": 50., "exern_iv30": 40.}
    elif mutation == "bool":
        implied["a"] = {"iv30": True, "exern_iv30": 40.}
    elif mutation == "nan":
        implied["a"] = {"iv30": float("nan"), "exern_iv30": 40.}
    elif mutation == "extra_key":
        implied["a"] = {"iv30": 50., "exern_iv30": 40., "extra": 1}
    else:
        challenger.pop()
    messages = {"missing_event": "IV implied ex-earnings coverage differs",
                "extra_event": "IV implied ex-earnings coverage differs",
                "extra_key": "IV implied ex-earnings coverage differs",
                "bool": "invalid IV implied input", "nan": "invalid IV implied input"}
    with pytest.raises(ValueError, match=messages.get(mutation)):
        paired_report_v2(events, challenger, incumbent, targets, implied)


def test_all_incumbent_abstain_has_no_common_set_but_keeps_training_mean():
    events, challenger, incumbent, targets, implied = case()
    for row in incumbent:
        row.update(abstention="insufficient_training", predicted_crush_pct_iv30=None)
    result = paired_report_v2(events, challenger, incumbent, targets, implied)
    assert result["comparison_status"] == "unavailable_no_common_matched_events"
    assert result["common_matched_events"] == 0
    assert result["common_matched"] is None
    assert result["bootstrap"] is None
    assert result["arm_metrics"]["incumbent"] is None
    assert result["arm_metrics"]["training_mean"]["count"] == 12


def test_challenger_distribution_defaults_embeds_supplied_and_refuses_non_dict():
    args = case()
    assert paired_report_v2(*args)["challenger_distribution"] == {"status": "not_supplied"}
    supplied = {"schema_version": "iv_distribution.v1", "median": 1.5}
    result = paired_report_v2(*args, distribution=supplied)
    assert result["challenger_distribution"] == supplied
    with pytest.raises(ValueError, match="invalid challenger distribution"):
        paired_report_v2(*args, distribution=[1])


def oracle_case():
    """Constant per-arm errors so every bootstrap resample equals the hand-computed pooled difference."""
    events = [{"event_id": e, "ticker": "AAA" if i < 2 else "BBB", "event_date": f"202{y}-0{m}-02"}
              for i, (e, y, m) in enumerate((("a", 3, 6), ("b", 3, 7), ("c", 4, 6), ("d", 4, 7)))]
    targets = {"a": -30., "b": 20., "c": 60., "d": -40.}
    challenger = [{"event_id": e, "horizon_sessions": h, "predicted_crush_pct_iv30": targets[e] + 2.}
                  for e in targets for h in (1, 3, 5)]
    incumbent = [{"event_id": e, "horizon_sessions": h, "predicted_crush_pct_iv30": targets[e] + 4.,
                  "abstention": None, "training_mean": targets[e] - 10.} for e in targets for h in (1, 3, 5)]
    # implied forecast = target + 6, from distinct ratios: 100*(exern/100 - 1) = target + 6
    implied = {e: {"iv30": 100., "exern_iv30": 106. + targets[e]} for e in targets}
    return events, challenger, incumbent, targets, implied


def test_common_matched_differences_and_bootstrap_match_hand_oracle():
    result = paired_report_v2(*oracle_case())
    # errors: challenger +2, incumbent +4, training_mean -10, implied +6 (RMSE == MAE == |error|)
    expected = {"challenger_minus_incumbent": -2.0, "challenger_minus_training_mean": -8.0,
                "challenger_minus_implied_ex_earnings": -4.0, "incumbent_minus_training_mean": -6.0}
    bias = {"challenger": 2.0, "incumbent": 4.0, "training_mean": -10.0, "implied_ex_earnings": 6.0}
    for arm, value in bias.items():
        metrics = result["common_matched"]["metrics"][arm]
        assert metrics["bias"] == pytest.approx(value) and metrics["count"] == 12
        assert metrics["mae"] == pytest.approx(abs(value)) and metrics["rmse"] == pytest.approx(abs(value))
    for name, value in expected.items():
        assert result["common_matched"]["difference"][name] == {"rmse": pytest.approx(value), "mae": pytest.approx(value)}
        for cluster, blocks in (("ticker", 2), ("event_month", 4)):
            pair = result["bootstrap"][cluster][name]
            assert pair["blocks"] == blocks and pair["events"] == 4
            assert pair["percentile_95"]["rmse"] == pytest.approx([value, value])
            assert pair["percentile_95"]["mae"] == pytest.approx([value, value])


@pytest.mark.parametrize("record", [{"iv30": 0., "exern_iv30": 40.}, {"iv30": 50., "exern_iv30": -1.},
                                    {"iv30": -5., "exern_iv30": 40.}, {"iv30": 50., "exern_iv30": None}])
def test_nonpositive_or_missing_implied_input_abstains(record):
    events, challenger, incumbent, targets, implied = case()
    implied["a"] = record
    result = paired_report_v2(events, challenger, incumbent, targets, implied)
    assert result["arm_metrics"]["implied_ex_earnings"]["count"] == 9
    assert result["common_matched_events"] == 3


def test_infinite_implied_input_refuses():
    events, challenger, incumbent, targets, implied = case()
    implied["a"] = {"iv30": math.inf, "exern_iv30": 40.}
    with pytest.raises(ValueError, match="invalid IV implied input"):
        paired_report_v2(events, challenger, incumbent, targets, implied)
