"""Admitted reference binding and hand-derived empirical forecast scores."""
from copy import deepcopy
from math import log

import pytest

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError
from experiments import distribution_prediction_baseline as baseline
from tests.v2.research.test_distribution_fitted import case as case


def prepared(case, horizon=1):
    result = case.fits.fit("outer", "2024", horizon)
    return baseline.training_baseline(case.fits, "2024", horizon), result


@pytest.mark.parametrize("horizon", [1, 3, 5])
def test_baseline_uses_exact_admitted_training_and_fit_claim(case, horizon):
    document, result = prepared(case, horizon)
    assert document["fit_claim"] == result["receipt"]["claim"]
    assert document["result_id"] == result["receipt"]["result_id"]
    assert document["training"] == case.inputs["training"]
    assert [row["absolute_move_pct"] for row in document["training"]] == [0, .25, 20, 22]
    assert document["evaluation_event_ids"] == ["Z", "A"]
    assert document["training_labels_before"] == "2023-01-01"
    assert document["baseline_id"] == content_hash({k: v for k, v in document.items() if k != "baseline_id"})
    assert not set(document["evaluation_event_ids"]) & {row["event_id"] for row in document["training"]}
    before = deepcopy(document)
    case.inputs["training"][0]["absolute_move_pct"] = 999
    assert document == before


def test_empirical_reference_has_hand_derived_scores_for_all_admitted_events(case):
    document, result = prepared(case)
    rows = list(baseline.score_training_baseline(document, {"Z": 5, "A": 0}, receipt=result["receipt"]))
    # Sum unordered distances for [0,1/4,20,22] is 343/4. CRPS subtracts 343/64.
    assert [row["event_id"] for row in rows] == ["Z", "A"]
    assert all(row["arm"] == "training_empirical" and row["horizon"] == 1 for row in rows)
    assert rows[0]["crps"] == 325 / 64 and rows[1]["crps"] == 333 / 64
    assert rows[0]["rps"] == 317 / 16
    assert rows[0]["log_loss"] == "infinite" and rows[1]["log_loss"] == log(4)
    assert rows[0]["tails"] == [dict(threshold=t, probability_ge=.5, observed_ge=5 >= t) for t in (5, 10, 20)]


def test_missing_target_retains_reference_row_and_does_not_replace_it_with_zero(case):
    document, result = prepared(case)
    rows = list(baseline.score_training_baseline(document,
        {"Z": {"omitted": "no_price_window"}, "A": 22}, receipt=result["receipt"]))
    assert rows[0] == dict(event_id="Z", outer_year="2024", horizon=1, arm="training_empirical",
                           observation=None, status="missing_target", reason="no_price_window")
    assert rows[1]["crps"] == 389 / 64


@pytest.mark.parametrize("mutation", [
    lambda context: context["inputs"]["training"][0].update(absolute_move_pct=7),
    lambda context: context["inputs"]["training"].reverse(),
    lambda context: next(row for row in context["inputs"]["features"] if row["horizon_sessions"] == 1).update(context_id="foreign"),
    lambda context: context["fold"]["evaluation_event_ids"].reverse(),
    lambda context: context["plan"].update(plan_id="foreign"),
])
def test_mutated_context_under_same_inputs_id_cannot_supply_baseline(case, mutation):
    case.fits.fit("outer", "2024", 1)
    original, calls = case.trials.read_fit_context, []
    def changed(*args):
        context = original(*args)
        if not calls:
            mutation(context)
        calls.append(1)
        return context
    case.trials.read_fit_context = changed
    with pytest.raises(OpsError, match="baseline differs"):
        baseline.training_baseline(case.fits, "2024", 1)


@pytest.mark.parametrize("mutation", [
    lambda context: context["inputs"]["training"].append(context["inputs"]["training"][0]),
    lambda context: context["fold"]["evaluation_event_ids"].append("Z"),
    lambda context: context["fold"]["evaluation_event_ids"].append("Tz"),
])
def test_duplicate_or_overlapping_event_axes_refuse(case, mutation):
    _, result = prepared(case)
    original = case.trials.read_fit_context
    def changed(*args):
        context = original(*args)
        mutation(context)
        return context
    case.trials.read_fit_context = changed
    case.fits.read = lambda *args: result
    with pytest.raises(OpsError, match="baseline differs"):
        baseline.training_baseline(case.fits, "2024", 1)


@pytest.mark.parametrize("field", ["plan_id", "admission_id", "outer_year", "inner_year", "ordered_events_id"])
def test_foreign_receipt_refuses_before_target_or_sample_access(case, monkeypatch, field):
    document, result = prepared(case)
    result["receipt"]["claim"][field] = "foreign"
    monkeypatch.setattr(baseline, "prepare_empirical_crps", lambda *args: pytest.fail("sample access"))
    with pytest.raises(OpsError, match="baseline differs"):
        next(baseline.score_training_baseline(document, None, receipt=result["receipt"]))


@pytest.mark.parametrize("targets", [{}, {"Z": 0}, {"Z": 0, "A": 0, "extra": 0}])
def test_scoring_requires_complete_exact_evaluation_roster(case, targets):
    document, result = prepared(case)
    with pytest.raises(OpsError, match="baseline differs"):
        next(baseline.score_training_baseline(document, targets, receipt=result["receipt"]))


def test_changed_saved_baseline_refuses_before_targets(case):
    document, result = prepared(case)
    document["training"][0]["absolute_move_pct"] = 7
    with pytest.raises(OpsError, match="baseline differs"):
        next(baseline.score_training_baseline(document, None, receipt=result["receipt"]))


def test_scoring_prepares_shared_distribution_once_for_the_event_cohort(case, monkeypatch):
    document, result = prepared(case)
    original, calls = baseline.prepare_empirical_crps, []
    def prepare(samples):
        calls.append(list(samples))
        return original(samples)
    monkeypatch.setattr(baseline, "prepare_empirical_crps", prepare)
    assert len(list(baseline.score_training_baseline(document, {"Z": 0, "A": 22}, receipt=result["receipt"]))) == 2
    assert calls == [[0, .25, 20, 22]]


@pytest.mark.parametrize("horizon", [True, 1.0, "1", None, 2])
def test_invalid_horizon_refuses_before_reading_fit(case, horizon):
    case.fits.read = lambda *args: pytest.fail("fit read")
    with pytest.raises(OpsError, match="baseline differs"):
        baseline.training_baseline(case.fits, "2024", horizon)


@pytest.mark.parametrize("mutation", [
    lambda result: result["receipt"].update(status="started"),
    lambda result: result["receipt"].update(result_id="foreign"),
    lambda result: result["rows"].reverse(),
])
def test_builder_checks_completed_fit_and_exact_event_order(case, mutation):
    _, result = prepared(case)
    mutation(result)
    case.fits.read = lambda *args: result
    with pytest.raises(OpsError, match="baseline differs"):
        baseline.training_baseline(case.fits, "2024", 1)


def test_rebuild_and_numeric_replay_need_no_fitting_or_new_record(case, monkeypatch):
    from engine.models.no_fit import no_fit_guard as legacy_no_fit
    from engine.v2.models.no_fit import no_fit_guard
    from experiments import distribution_fitted
    document, result = prepared(case)
    before = deepcopy(case.records)
    def forbidden(*args, **kwargs):
        pytest.fail("fit or journal write during baseline replay")
    monkeypatch.setattr(distribution_fitted, "fit_candidate", forbidden)
    monkeypatch.setattr(distribution_fitted, "append_record", forbidden)
    with no_fit_guard(), legacy_no_fit():
        assert baseline.training_baseline(case.fits, "2024", 1) == document
        assert len(list(baseline.score_training_baseline(document,
            {"Z": 0, "A": 22}, receipt=result["receipt"]))) == 2
    assert case.records == before
    assert sum(call[0] == "fit" for call in case.calls) == 1


@pytest.mark.parametrize('mutation', [
    lambda doc, receipt: doc.pop('baseline_id'),
    lambda doc, receipt: doc.pop('evaluation_event_ids'),
    lambda doc, receipt: doc.update(evaluation_event_ids=None),
    lambda doc, receipt: doc.update(evaluation_event_ids=['Z', 'Z']),
    lambda doc, receipt: doc.update(training=None),
    lambda doc, receipt: doc.update(training=[{}]),
    lambda doc, receipt: receipt.clear(),
    lambda doc, receipt: receipt.update(claim=None),
])
def test_malformed_saved_evidence_has_exact_typed_refusal_before_preparation(case, monkeypatch, mutation):
    document, result = prepared(case)
    mutation(document, result['receipt'])
    if 'baseline_id' in document:
        document['baseline_id'] = content_hash({k: v for k, v in document.items() if k != 'baseline_id'})
    monkeypatch.setattr(baseline, 'prepare_empirical_crps', lambda *args: pytest.fail('score preparation'))
    with pytest.raises(OpsError) as error:
        next(baseline.score_training_baseline(document, {'Z': 0, 'A': 22}, receipt=result['receipt']))
    assert error.value.code == 'EXPERIMENT_IDENTITY_CONFLICT'
    assert error.value.problem.message == 'distribution training baseline differs'
    assert error.value.problem.retryable is False


@pytest.mark.parametrize('receipt', [None, {}, [], 'receipt'])
def test_malformed_receipt_is_typed_before_any_target_iteration(case, receipt):
    document, _ = prepared(case)
    with pytest.raises(OpsError) as error:
        next(baseline.score_training_baseline(document, None, receipt=receipt))
    assert error.value.code == 'EXPERIMENT_IDENTITY_CONFLICT'


@pytest.mark.parametrize('bad', [{}, {'omitted': None}, {'omitted': ''}, {'omitted': 'foreign'},
    {'omitted': 'no_price_window', 'value': 0}, [], None, True, float('nan'), float('inf'), -1, 10**1000])
def test_all_target_statuses_checked_before_first_row_or_score_preparation(case, monkeypatch, bad):
    document, result = prepared(case)
    monkeypatch.setattr(baseline, 'prepare_empirical_crps', lambda *args: pytest.fail('score preparation'))
    # Z is valid and ordered first: invalid A must still prevent any first yield.
    with pytest.raises(OpsError) as error:
        next(baseline.score_training_baseline(document, {'Z': 0, 'A': bad}, receipt=result['receipt']))
    assert error.value.code == 'EXPERIMENT_IDENTITY_CONFLICT'
    assert error.value.problem.retryable is False


@pytest.mark.parametrize('reason', sorted(baseline.OMISSION_REASONS))
def test_each_native_omission_reason_is_retained_exactly(case, reason):
    document, result = prepared(case)
    rows = list(baseline.score_training_baseline(document,
        {'Z': {'omitted': reason}, 'A': 0}, receipt=result['receipt']))
    assert rows[0] == dict(event_id='Z', outer_year='2024', horizon=1, arm='training_empirical',
        observation=None, status='missing_target', reason=reason)
    assert rows[1]['crps'] == 333 / 64


@pytest.mark.parametrize('mutation', ['append', 'remove', 'reorder', 'claim', 'target', 'status', 'training', 'clear'])
def test_between_yield_mutation_cannot_change_validated_roster_identity_or_targets(case, mutation):
    document, result = prepared(case)
    targets = {'Z': 0, 'A': {'omitted': 'no_price_window'} if mutation == 'status' else 22}
    expected = list(baseline.score_training_baseline(document, targets, receipt=result['receipt']))
    stream = baseline.score_training_baseline(document, targets, receipt=result['receipt'])
    first = next(stream)
    assert first['event_id'] == 'Z' and first['outer_year'] == '2024' and first['horizon'] == 1
    if mutation == 'append':
        document['evaluation_event_ids'].append('OUTSIDE')
        targets['OUTSIDE'] = 0
    elif mutation == 'remove':
        document['evaluation_event_ids'].remove('A')
        targets.pop('A')
    elif mutation == 'reorder':
        document['evaluation_event_ids'].reverse()
    elif mutation == 'claim':
        document['fit_claim']['outer_year'] = '2099'
        document['fit_claim']['trial']['horizon'] = 5
    elif mutation == 'target':
        targets['A'] = 999
    elif mutation == 'status':
        targets['A']['omitted'] = 'unvalidated replacement'
    elif mutation == 'training':
        document['training'][0]['absolute_move_pct'] = 999
    else:
        document.clear()
        result['receipt'].clear()
        targets.clear()
    rows = [first, *stream]
    assert rows == expected
    assert [(row['event_id'], row['outer_year'], row['horizon']) for row in rows] == [('Z', '2024', 1), ('A', '2024', 1)]
    if mutation == 'status':
        assert rows[1]['reason'] == 'no_price_window' and rows[1]['observation'] is None
    else:
        assert rows[1]['observation'] == 22
