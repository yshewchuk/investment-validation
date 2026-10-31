"""Synthetic public replay/admission seams: no model, native labels or store."""
from contextlib import nullcontext
from copy import deepcopy
from math import fsum
from types import SimpleNamespace

import pytest

from engine.v2.data.errors import DataError
from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError, fail
from engine.v2.research.magnitude_distribution import histogram
from engine.v2.research.magnitude_distribution_math import mean
from experiments import distribution_oof as oof


def _identified(body, key):
    return {**body, key: content_hash(body)}


def _distribution(index=0, tail=None):
    masses = [0.0] * 81
    masses[index] = 1.0
    return histogram(masses, tail_excesses=[0] if tail is None else tail)


class Fits:
    def __init__(self):
        self.calls, self.context_calls, self.edits = [], [], {}
        self.plan = {"plan_id": content_hash("plan"), "context_id": content_hash("context"),
            "recipe": {"recipe_id": content_hash("recipe"), "horizons": [5, 3, 1]},
            "budget": {"budget_id": content_hash("budget")}}
        self.labels = {"base": 50.0, "e0": 2.125, "e1": 4.125, "e2": 1.125}
        self.inners = [{"year": str(2020 + index), "training_labels_before": f"{2020 + index}-01-01",
            "evaluation_event_ids": [f"e{index}"] + (["omitted"] if index == 0 else [])}
            for index in range(3)]
        self.outer = {"year": "2023", "training_labels_before": "2023-01-01", "evaluation_event_ids": ["future"]}
        self.native = {"folds": [{"outer": self.outer, "inner": self.inners}]}
        self.trials = SimpleNamespace(read_fit_context=self.context)

    def context(self, role, outer_year, inner_year=None):
        self.context_calls.append((role, outer_year, inner_year))
        assert outer_year == "2023"
        if role == "outer":
            fold, count = self.outer, 3
        else:
            count = [item["year"] for item in self.inners].index(inner_year)
            fold = self.inners[count]
        training = [{"event_id": event, "absolute_move_pct": value} for event, value in self.labels.items()
                    if event == "base" or event in [f"e{index}" for index in range(count)]]
        admission = {"inputs_id": content_hash(training), "fold_id": content_hash(
            {"fold": fold, "role": role, "outer_year": outer_year})}
        return deepcopy({"plan": self.plan, "native_plan": self.native, "fold": fold,
            "admission": admission, "inputs": {"training": training, "features": []}})

    def read(self, role, outer_year, horizon, inner_year=None):
        self.calls.append((role, outer_year, horizon, inner_year))
        assert role == "inner" and outer_year == "2023", "outer/future result was used"
        donor = next(item for item in self.inners if item["year"] == inner_year)
        position = int(inner_year) - 2019
        document = _distribution(position * 4 * horizon)
        rows = [{"event_id": event, "probabilities": document["probabilities"],
                 "point_prediction": float(position * horizon)} for event in donor["evaluation_event_ids"]]
        trial = {"recipe_id": self.plan["recipe"]["recipe_id"], "context_id": self.plan["context_id"],
            "inputs_id": content_hash([inner_year, horizon]), "budget_id": self.plan["budget"]["budget_id"],
            "horizon": horizon, "role": role,
            "fold_id": content_hash({"fold": donor, "role": role, "outer_year": outer_year})}
        claim = {"plan_id": self.plan["plan_id"], "admission_id": content_hash(inner_year),
            "trial": _identified(trial, "trial_id"), "role": role,
            "outer_year": outer_year, "inner_year": inner_year}
        result = {"receipt": _identified({"status": "completed", "claim": claim}, "result_id"),
                  "rows": rows, "tail": document["tail"]}
        return self.edits.get(inner_year, lambda result: result)(deepcopy(result))


@pytest.fixture
def fits(monkeypatch):
    monkeypatch.setattr(oof, "stage", lambda *args: nullcontext())
    monkeypatch.setattr(oof, "Progress", lambda *args: SimpleNamespace(tick=lambda: None))
    return Fits()


def _pool(fits, role="outer", inner=None, horizon=1):
    return oof.residual_pool(fits, role, "2023", horizon, inner_year=inner)


def _conflict(action):
    with pytest.raises(OpsError) as error:
        action()
    assert error.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert error.value.problem.retryable is False


def _rehash(result):
    result["receipt"]["result_id"] = content_hash({key: value for key, value in result["receipt"].items() if key != "result_id"})
    return result


def test_exact_outer_pool_uses_all_inner_sources_and_destination_labels(fits):
    pool = _pool(fits)
    assert fits.calls == [("inner", "2023", 1, str(year)) for year in (2020, 2021, 2022)]
    assert fits.context_calls == [("outer", "2023", None)]
    assert pool["point_residuals"] == [1.125, 2.125, -1.875]
    assert pool["histogram_residuals"] == [1.0, 2.0, -2.0]
    assert pool["keys"] == [["e0", 1], ["e1", 1], ["e2", 1]]
    assert pool["coverage"] == {"source_rows": 4, "contributing_rows": 3, "excluded_rows": 1}
    assert pool["exclusions"] == [{"event_id": "omitted", "horizon": 1,
        "result_id": pool["sources"][0]["result_id"], "reason": "not_destination_training"}]
    assert pool["point_status"] == pool["histogram_status"] == "ready"
    assert pool["destination"]["training_labels_before"] == "2023-01-01"
    assert pool["destination"]["plan_id"] == fits.plan["plan_id"]
    assert pool["destination"]["recipe_id"] == fits.plan["recipe"]["recipe_id"]
    assert pool["calibration_identity"] == "uncalibrated" and pool["final_read_authority"] == "none"
    assert pool["pool_id"] == content_hash({key: value for key, value in pool.items() if key != "pool_id"})
    assert _pool(fits) == pool


def test_three_period_inner_prefix_is_immune_to_future_labels_and_predictions(fits):
    original = _pool(fits, "inner", "2021")
    assert fits.calls == [("inner", "2023", 1, "2020")]
    fits.labels["e1"], fits.labels["e2"] = 99999, 88888
    fits.edits["2021"] = fits.edits["2022"] = lambda result: pytest.fail("future donor read")
    assert _pool(fits, "inner", "2021") == original
    assert original["point_residuals"] == [1.125]
    assert original["histogram_residuals"] == [1]
    assert original["destination"]["training_labels_before"] == "2021-01-01"


def test_first_inner_explicit_unavailable_never_reads_a_donor(fits):
    pool = _pool(fits, "inner", "2020")
    assert fits.calls == []
    assert pool["point_status"] == pool["histogram_status"] == "no_earlier_inner"
    assert pool["sources"] == pool["keys"] == pool["point_residuals"] == []
    controls = oof.control_for_prediction(pool, _distribution(), 1)
    assert controls["point"] == controls["histogram"] == {"status": "no_earlier_inner"}


def test_no_eligible_labels_reports_all_exclusions_but_requires_all_sources(fits):
    fits.labels = {"base": 1.0}
    pool = _pool(fits)
    assert len(fits.calls) == 3
    assert pool["point_status"] == pool["histogram_status"] == "no_eligible_rows"
    assert pool["coverage"] == {"source_rows": 4, "contributing_rows": 0, "excluded_rows": 4}


def test_missing_failed_donor_never_silently_disappears(fits):
    failure = fail("RESOURCE_UNAVAILABLE", "synthetic missing completion")
    def missing(result):
        raise failure
    fits.edits["2021"] = missing
    with pytest.raises(OpsError) as error:
        _pool(fits)
    assert error.value is failure
    assert [call[-1] for call in fits.calls] == ["2020", "2021"]
    fits.edits["2021"] = lambda result: None
    _conflict(lambda: _pool(fits))


@pytest.mark.parametrize("field,value", [("role", "outer"), ("outer_year", "2024"),
    ("inner_year", "2022"), ("plan_id", "other")])
def test_wrong_donor_claims_refuse_even_with_valid_document_hash(fits, field, value):
    def changed(result):
        result["receipt"]["claim"][field] = value
        return _rehash(result)
    fits.edits["2020"] = changed
    _conflict(lambda: _pool(fits))


@pytest.mark.parametrize("field,value", [("recipe_id", "other"), ("context_id", "other"),
    ("budget_id", "other"), ("fold_id", "other"), ("horizon", 3), ("role", "outer")])
def test_wrong_trial_identity_refuses(fits, field, value):
    def changed(result):
        result["receipt"]["claim"]["trial"][field] = value
        return _rehash(result)
    fits.edits["2020"] = changed
    _conflict(lambda: _pool(fits))


def test_incorrect_completion_hash_and_duplicate_or_unknown_rows_refuse(fits):
    for edit in (lambda result: {**result, "receipt": {**result["receipt"], "result_id": "unknown"}},
                 lambda result: {**result, "rows": [result["rows"][0]] * 2},
                 lambda result: {**result, "rows": [{**result["rows"][0], "event_id": "unknown"}, result["rows"][1]]}):
        fits.edits["2020"] = edit
        _conflict(lambda: _pool(fits))


def test_cross_donor_duplicate_and_self_overlap_refuse(fits):
    fits.inners[1]["evaluation_event_ids"] = ["e0"]
    _conflict(lambda: _pool(fits))
    fits.inners[1]["evaluation_event_ids"] = ["future"]
    _conflict(lambda: _pool(fits))


@pytest.mark.parametrize("adjustment", [0, -5e-13])
def test_donor_specific_tail_not_outer_tail_and_optimized_mean_matches_public_oracle(fits, adjustment):
    document = histogram([0.25 + adjustment] + [0] * 79 + [0.75], tail_excesses=[0, 0, 20])
    def changed(result):
        return {**result, "tail": document["tail"], "rows": [
            {**row, "probabilities": document["probabilities"]} for row in result["rows"]]}
    fits.edits["2020"] = changed
    expected = fits.labels["e0"] - mean(document)
    pool = _pool(fits, "inner", "2021")
    assert pool["histogram_residuals"] == pytest.approx([expected], rel=1e-14, abs=1e-14)
    assert pool["sources"][0]["tail_id"] == content_hash(document["tail"])
    assert all(call[0] == "inner" for call in fits.calls)
    document["tail"]["excesses"] = [40, 40, 60]
    new = _pool(fits, "inner", "2021")
    assert new["histogram_residuals"] != pool["histogram_residuals"]
    assert new["point_residuals"] == pool["point_residuals"]


def test_unresolved_donor_disables_whole_histogram_pool_without_row_dropping(fits):
    def unresolved(result):
        return {**result, "tail": {"policy": "unresolved", "sample_count": 0, "excesses": []}}
    fits.edits["2021"] = unresolved
    pool = _pool(fits)
    assert pool["histogram_status"] == "unresolved_donor_tail"
    assert pool["histogram_residuals"] is None
    assert pool["point_status"] == "ready" and len(pool["point_residuals"]) == 3
    assert pool["coverage"]["contributing_rows"] == 3
    result = oof.control_for_prediction(pool, _distribution(), 0)
    assert result["histogram"] == {"status": "unresolved_donor_tail"}
    assert result["point"]["status"] == "ready"
    del fits.labels["e1"]  # Even an entirely excluded donor cannot be hidden.
    excluded = _pool(fits)
    assert excluded["histogram_status"] == "unresolved_donor_tail"
    assert excluded["coverage"]["contributing_rows"] == 2


def test_primary_mean_anchor_secondary_actual_mean_and_zero_atoms(fits):
    pool = _pool(fits)
    result = oof.control_for_prediction(pool, _distribution(), 0)
    assert result["point"]["magnitudes"] == [1.125, 2.125, 0]
    assert result["point"]["mean"] == pytest.approx(3.25 / 3)
    assert result["point"]["zero_count"] == 1
    assert result["histogram"]["anchored_mean"] == pytest.approx(0.125)
    assert result["histogram"]["zero_count"] == 1
    assert result["histogram"]["magnitudes"][-1] == 0
    assert fsum(result["histogram"]["magnitudes"]) / 3 == pytest.approx(0.125)
    result["point"]["magnitudes"][0] = 99
    assert oof.control_for_prediction(pool, _distribution(), 0)["point"]["magnitudes"][0] == 1.125


def test_unresolved_prediction_keeps_point_control_and_all_zero_anchor_refuses(fits):
    pool = _pool(fits)
    unresolved = histogram([1] + [0] * 80)
    result = oof.control_for_prediction(pool, unresolved, 0)
    assert result["histogram"]["status"] == "unresolved_prediction_tail"
    assert result["point"]["status"] == "ready"
    fits.labels["e0"] = 0
    pool = _pool(fits, "inner", "2021")
    with pytest.raises(DataError) as error:
        oof.control_for_prediction(pool, _distribution(), 0)
    assert error.value.code == "CONTRACT_MISMATCH"


def test_horizon_isolation_and_pool_mutation_refusal(fits):
    one, three = _pool(fits), _pool(fits, horizon=3)
    assert one["point_residuals"] != three["point_residuals"]
    assert three["keys"] == [["e0", 3], ["e1", 3], ["e2", 3]]
    assert one["pool_id"] != three["pool_id"]
    for horizon in (True, 0, 2, "1"):
        _conflict(lambda: _pool(fits, horizon=horizon))
    one["point_residuals"][0] += 1
    _conflict(lambda: oof.control_for_prediction(one, _distribution(), 0))


def test_public_tail_weight_underflow_refusal_is_not_bypassed(fits):
    document = histogram([1] + [0] * 79 + [5e-324], tail_excesses=[0, 1])
    fits.edits["2020"] = lambda result: {**result, "tail": document["tail"], "rows": [
        {**row, "probabilities": document["probabilities"]} for row in result["rows"]]}
    with pytest.raises(DataError) as expected:
        mean(document)
    with pytest.raises(DataError) as actual:
        _pool(fits, "inner", "2021")
    assert actual.value.code == expected.value.code == "CONTRACT_MISMATCH"
    assert actual.value.problem.message == expected.value.problem.message


def test_duplicate_destination_training_and_self_training_refuse(fits):
    original = fits.trials.read_fit_context
    def duplicate(*args):
        context = original(*args)
        context["inputs"]["training"].append(deepcopy(context["inputs"]["training"][0]))
        return context
    fits.trials.read_fit_context = duplicate
    _conflict(lambda: _pool(fits))
    def self_training(*args):
        context = original(*args)
        context["inputs"]["training"].append({"event_id": "future", "absolute_move_pct": 1})
        return context
    fits.trials.read_fit_context = self_training
    _conflict(lambda: _pool(fits))


def test_source_rows_have_deterministic_key_order(fits):
    expected = _pool(fits)
    fits.edits["2020"] = lambda result: {**result, "rows": list(reversed(result["rows"]))}
    assert _pool(fits) == expected


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), 10 ** 1000])
def test_nonfinite_arithmetic_is_native_failure(fits, bad):
    fits.labels["e0"] = bad
    with pytest.raises(OpsError) as error:
        _pool(fits)
    assert error.value.code == "EXPERIMENT_VARIANT_FAILED"


def test_point_addition_overflow_refuses_instead_of_clipping_nonfinite(fits):
    fits.labels["e0"] = 1e308
    pool = _pool(fits, "inner", "2021")
    with pytest.raises(OpsError) as error:
        oof.control_for_prediction(pool, _distribution(), 1e308)
    assert error.value.code == "EXPERIMENT_VARIANT_FAILED"


def test_no_direct_io_or_new_label_reader(fits, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("OOF controls attempted direct I/O")
    monkeypatch.setattr("builtins.open", forbidden)
    pool = _pool(fits)
    assert oof.control_for_prediction(pool, _distribution(), 0)["point"]["status"] == "ready"
