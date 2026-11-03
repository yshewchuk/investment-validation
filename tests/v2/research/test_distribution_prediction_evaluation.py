"""Native-receipt target admission and real CAS score/diagnostic replay."""
from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError
from engine.v2.research.magnitude_inputs import magnitude_recipe
from experiments import distribution_prediction_evaluation as evaluation
from tests.v2.research.test_distribution_prediction_plan import case as case
from tests.v2.research.test_distribution_scores import distribution, forecast, pool


@pytest.fixture
def evaluated(case, monkeypatch):
    run = case.run
    plan = run.seal(clock=case.clock)
    common = {'snapshot_id': 'snapshot', 'holdout_as_of_month': '2025-01',
        'random_membership_version': 'canonical-event-sha256.v1', 'rolling_membership_version': 'calendar-months.v1',
        'population_use': 'post-release selection', 'purpose': 'selection', 'computed_moves_version': 'moves'}
    physical = [{'event_id': e, 'fragment_id': 'fragment'+e, 'object_id': 'object'+e, 'content_hash': 'hash'+e} for e in ('Z', 'A')]
    inputs = {'preflight': {**common, 'planned_inputs': physical}}
    admission = {'manifest': inputs, 'inputs_id': content_hash(inputs)}
    fold = {'year': '2024', 'evaluation_event_ids': ['Z', 'A'], 'evaluation_labels_before': '2025-01-01'}
    recipe = magnitude_recipe()
    body = {'schema_version': 'research_magnitude_targets.v1', **common, 'labels_before': fold['evaluation_labels_before'],
        'target_recipe': recipe, 'target_recipe_id': content_hash(recipe), 'consumed_inputs': physical,
        'excluded_future_events': [], 'omitted_labels': [{'event_id': 'A', 'reason': 'no_price_window'}]}
    receipt = {**body, 'inputs_id': content_hash(body), 'holdout_exclusions': []}
    frame = pd.DataFrame([{'event_id': 'Z', 'absolute_move_pct': .125}])
    frame.attrs = deepcopy(receipt)
    values = {}
    for horizon in (1, 3, 5):
        controls = pool(outer_year='2024', horizon=horizon)
        result = forecast(distribution(), event='Z', scope=controls['destination'])
        result['rows'].append({**result['rows'][0], 'event_id': 'A'})
        result['receipt']['result_id'] = content_hash(result['receipt'])
        baseline = {'schema_version': 'distribution_training_baseline.v1', 'policy': 'same_admitted_training_equal_weight.v1',
            'fit_claim': result['receipt']['claim'], 'result_id': result['receipt']['result_id'],
            'evaluation_event_ids': ['Z', 'A'], 'training': [{'event_id': 'T1', 'absolute_move_pct': 0}, {'event_id': 'T2', 'absolute_move_pct': 1}],
            'training_labels_before': '2024-01-01', 'final_read_authority': 'none'}
        baseline['baseline_id'] = content_hash(baseline)
        values[horizon] = {'result': result, 'pool': controls, 'baseline': baseline}
    frozen = {'run_plan_id': plan['run_plan_id'], 'fold': fold, 'admission_id': content_hash(admission)}
    context = {'fold': fold, 'admission': admission}
    calls = []
    def frozen_read(*args):
        calls.append('frozen')
        return deepcopy((frozen, values))
    monkeypatch.setattr(evaluation, 'read_frozen_outer', frozen_read)
    fits = SimpleNamespace(trials=SimpleNamespace(read_fit_context=lambda *args: deepcopy(context)))
    run.trials.repository = object()
    run.trials.source = SimpleNamespace(document={'source': 'binding'})
    def targets(repository, binding, selected):
        assert calls[-1] == 'frozen' and selected == fold
        assert run.record('evaluation-start:2024') is not None
        calls.append('target-read')
        return frame.copy(deep=True)
    monkeypatch.setattr(evaluation, 'evaluation_targets', targets)
    return SimpleNamespace(run=run, fits=fits, context=context, frozen=frozen, values=values,
        calls=calls, frame=frame, case=case)


def test_native_receipt_scores_diagnostics_and_no_source_replay(evaluated, monkeypatch):
    saved = evaluation.evaluate_outer(evaluated.run, evaluated.fits, '2024')
    assert evaluated.calls == ['frozen', 'target-read']
    before = evaluated.case.conn.total_changes
    monkeypatch.setattr(evaluation, 'evaluation_targets', lambda *args: pytest.fail('target reread'))
    record, scores, diagnostics = evaluation.read_evaluated_outer(evaluated.run, evaluated.fits, '2024')
    assert record == saved and len(scores) == len(diagnostics) == 24
    assert evaluated.case.conn.total_changes == before
    assert sum(row['status'] == 'missing_target' for row in scores) == 12
    assert sum(row['status'] == 'missing_target' for row in diagnostics) == 12
    hist = next(row for row in scores if row['horizon'] == 1 and row['arm'] == 'histogram' and row['event_id'] == 'Z')
    assert hist['crps'] == pytest.approx(1 / 48) and hist['rps'] == 0 and hist['log_loss'] == 0
    baseline = next(row for row in scores if row['horizon'] == 1 and row['arm'] == 'training_empirical' and row['event_id'] == 'Z')
    assert baseline['crps'] == .25
    diagnostic = next(row for row in diagnostics if row['horizon'] == 1 and row['arm'] == 'histogram' and row['event_id'] == 'Z')
    assert diagnostic['pit_lower'] == diagnostic['pit_upper'] == .5
    assert diagnostic['intervals'][0] == {'level': .5, 'lower': .0625, 'upper': .1875, 'width': .125, 'covered': True}


@pytest.mark.parametrize('field,value', [('labels_before', '2026-01-01'), ('inputs_id', 'foreign'), ('holdout_exclusions', ['event'])])
def test_foreign_native_receipt_consumes_attempt_without_publishing_scores(evaluated, field, value):
    evaluated.frame.attrs[field] = value
    with pytest.raises(OpsError):
        evaluation.evaluate_outer(evaluated.run, evaluated.fits, '2024')
    assert evaluated.run.record('evaluated:2024') is None
    before = list(evaluated.calls)
    with pytest.raises(OpsError):
        evaluation.evaluate_outer(evaluated.run, evaluated.fits, '2024')
    assert evaluated.calls == before


def test_changed_native_context_refuses_before_target_read(evaluated):
    evaluated.context['fold'] = {**evaluated.context['fold'], 'evaluation_labels_before': '2030-01-01'}
    with pytest.raises(OpsError):
        evaluation.evaluate_outer(evaluated.run, evaluated.fits, '2024')
    assert evaluated.calls == ['frozen']


@pytest.mark.parametrize('field', ['scores', 'diagnostics'])
def test_rehashed_substituted_scores_or_diagnostics_are_not_trusted(evaluated, monkeypatch, field):
    saved = evaluation.evaluate_outer(evaluated.run, evaluated.fits, '2024')
    changed = deepcopy(saved)
    rows = list(evaluated.run.read_section(changed['horizons'][0][field]))
    if field == 'scores':
        rows[0]['crps'] = 999
    else:
        rows[0]['pit_lower'] = .1
    changed['horizons'][0][field] = evaluated.run.publish('distribution_prediction_changed', rows, count=len(rows))
    original = evaluated.run.record
    monkeypatch.setattr(evaluated.run, 'record', lambda slot, body=None: changed if slot == 'evaluated:2024' else original(slot, body))
    monkeypatch.setattr(evaluation, 'evaluation_targets', lambda *args: pytest.fail('source reread'))
    with pytest.raises(OpsError):
        evaluation.read_evaluated_outer(evaluated.run, evaluated.fits, '2024')


def test_replay_requires_final_completion_and_complete_horizons(evaluated, monkeypatch):
    with pytest.raises(OpsError):
        evaluation.read_evaluated_outer(evaluated.run, evaluated.fits, '2024')
    saved = evaluation.evaluate_outer(evaluated.run, evaluated.fits, '2024')
    original = evaluated.run.record
    monkeypatch.setattr(evaluated.run, 'record', lambda slot, body=None:
        {**saved, 'horizons': saved['horizons'][:2]} if slot == 'evaluated:2024' else original(slot, body))
    with pytest.raises(OpsError):
        evaluation.read_evaluated_outer(evaluated.run, evaluated.fits, '2024')


def test_unresolved_histogram_tail_keeps_scores_but_not_invented_diagnostics(evaluated):
    for value in evaluated.values.values():
        value['result']['tail'] = distribution(tail=None)['tail']
    evaluation.evaluate_outer(evaluated.run, evaluated.fits, '2024')
    _, scores, diagnostics = evaluation.read_evaluated_outer(evaluated.run, evaluated.fits, '2024')
    histograms = [row for row in scores if row['arm'] == 'histogram' and row['event_id'] == 'Z']
    assert len(histograms) == 3 and all(row['status'] == 'available' and row['crps'] is None for row in histograms)
    histogram_diagnostics = [row for row in diagnostics if row['arm'] == 'histogram' and row['event_id'] == 'Z']
    assert len(histogram_diagnostics) == 3
    assert all(row['status'] == 'unavailable' and row['reason'] == 'unresolved_tail' and 'intervals' not in row for row in histogram_diagnostics)
    assert all(row['status'] == 'available' for row in diagnostics if row['arm'] == 'training_empirical' and row['event_id'] == 'Z')


def test_native_receipt_omission_union_cannot_hide_missing_event(evaluated):
    body = {key: value for key, value in evaluated.frame.attrs.items() if key not in ('inputs_id', 'holdout_exclusions')}
    body.pop('omitted_labels')
    evaluated.frame.attrs = {**body, 'inputs_id': content_hash(body), 'holdout_exclusions': []}
    with pytest.raises(OpsError):
        evaluation.evaluate_outer(evaluated.run, evaluated.fits, '2024')
    assert evaluated.run.record('evaluated:2024') is None


def test_evaluation_target_receipt_persists_full_native_physical_identity(evaluated):
    saved = evaluation.evaluate_outer(evaluated.run, evaluated.fits, '2024')
    receipt = list(evaluated.run.read_section(saved['receipt']))[0]
    assert receipt == evaluated.frame.attrs
    assert [row['event_id'] for row in receipt['consumed_inputs']] == ['Z', 'A']
    assert receipt['labels_before'] == '2025-01-01'
    assert receipt['holdout_exclusions'] == []
    assert receipt['omitted_labels'] == [{'event_id': 'A', 'reason': 'no_price_window'}]


@pytest.mark.parametrize('year', [None, 2024, True, 2024.0, b'2024', [], {}])
def test_non_string_year_refuses_before_admission_or_evidence_effects(evaluated, monkeypatch, tmp_path, year):
    conn = evaluated.case.conn
    changes, records = conn.total_changes, list(conn.execute('SELECT * FROM magnitude_search_records'))
    before = {path.relative_to(tmp_path): path.read_bytes() for path in (tmp_path / 'store').rglob('*') if path.is_file()}
    def refused_effect(*args, **kwargs):
        pytest.fail('invalid year reached admission, journal, stored evidence or targets')
    monkeypatch.setattr(evaluated.run, 'admit', refused_effect)
    monkeypatch.setattr(evaluated.run, 'record', refused_effect)
    monkeypatch.setattr(evaluation, '_context', refused_effect)
    monkeypatch.setattr(evaluation, 'evaluation_targets', refused_effect)
    with pytest.raises(OpsError) as caught:
        evaluation.evaluate_outer(evaluated.run, evaluated.fits, year)
    assert caught.value.code == 'INVALID_EXPERIMENT_SPEC' and not caught.value.problem.retryable
    assert conn.total_changes == changes and list(conn.execute('SELECT * FROM magnitude_search_records')) == records
    assert {path.relative_to(tmp_path): path.read_bytes() for path in (tmp_path / 'store').rglob('*') if path.is_file()} == before
    assert evaluated.calls == []
