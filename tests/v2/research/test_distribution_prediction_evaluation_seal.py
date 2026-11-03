"""Actual P7 replay integration and P11 implementation seal regressions."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError
from engine.v2.ops.fingerprints import source_closure
from experiments import distribution_prediction_evaluation as evaluation
from experiments import distribution_prediction_evidence as evidence
from experiments import distribution_prediction_plan as metadata
from tests.v2.research.test_distribution_prediction_evaluation import evaluated as evaluated
from tests.v2.research.test_distribution_prediction_plan import case as native_case

ROOT = Path(metadata.__file__).resolve().parents[1]
EVALUATION_SOURCE = 'experiments/distribution_prediction_evaluation.py'
BOUND_SOURCES = (EVALUATION_SOURCE, 'experiments/distribution_prediction_scores.py',
    'engine/v2/research/magnitude_distribution_diagnostics.py')


class InheritedPlan(metadata.PredictionPlan):
    pass


class YearString(str):
    pass


@pytest.fixture(scope='module')
def code_snapshot(tmp_path_factory):
    destination = tmp_path_factory.mktemp('prediction-evaluation-source')
    for relative in source_closure(ROOT, [*metadata.PredictionPlan.source_roots, EVALUATION_SOURCE]):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    return destination


@pytest.fixture(params=[metadata.PredictionPlan, InheritedPlan], ids=['default', 'inherited'])
def case(native_case, request, tmp_path, monkeypatch, code_snapshot):
    code = tmp_path / 'code'
    shutil.copytree(code_snapshot, code)
    monkeypatch.setattr(metadata, 'source_closure', lambda root, paths: source_closure(code, paths))
    native_case.run = request.param(native_case.trials, native_case.run.request)
    native_case.code = code
    return native_case


def snapshot(case):
    return (case.conn.total_changes,
        tuple(tuple(row) for row in case.conn.execute('SELECT * FROM magnitude_search_records ORDER BY slot')),
        tuple(sorted((str(path.relative_to(case.store.root)), sha256(path.read_bytes()).hexdigest())
            for path in case.store.root.rglob('*') if path.is_file())))


@pytest.fixture
def composed(evaluated, monkeypatch):
    # Keep synthetic completed native fit/control seams, but replace the old
    # fixture's fake frozen reader with actual P7 publication and CAS replay.
    run, values = evaluated.run, evaluated.values
    plan = run.admit()
    evaluated.context['plan'] = {'plan_id': plan['admission_plan_id'], 'recipe': {'horizons': [1, 3, 5]}}
    admission_id = content_hash(evaluated.context['admission'])
    for value in values.values():
        receipt, pool, baseline = value['result']['receipt'], value['pool'], value['baseline']
        receipt['schema_version'] = 'distribution_fitted.v2'
        receipt['claim'].update(plan_id=plan['admission_plan_id'], admission_id=admission_id)
        receipt['result_id'] = content_hash({key: val for key, val in receipt.items() if key != 'result_id'})
        pool['destination'].update(plan_id=plan['admission_plan_id'], admission_id=admission_id)
        pool['pool_id'] = content_hash({key: val for key, val in pool.items() if key != 'pool_id'})
        baseline.update(fit_claim=deepcopy(receipt['claim']), result_id=receipt['result_id'])
        baseline['baseline_id'] = content_hash({key: val for key, val in baseline.items() if key != 'baseline_id'})
    evaluated.fits.read = lambda role, year, horizon: deepcopy(values[horizon]['result'])
    monkeypatch.setattr(evidence, 'residual_pool', lambda fits, role, year, horizon: deepcopy(values[horizon]['pool']))
    monkeypatch.setattr(evidence, 'training_baseline', lambda fits, year, horizon: deepcopy(values[horizon]['baseline']))
    monkeypatch.setattr(evaluation, 'read_frozen_outer', evidence.read_frozen_outer)
    assert evaluation.read_frozen_outer is evidence.read_frozen_outer
    evidence.freeze_outer(run, evaluated.fits, '2024')
    def targets(*args):
        assert [entry['horizon'] for entry in run.record('frozen:2024')['horizons']] == [1, 3, 5]
        assert run.record('evaluation-start:2024') is not None
        evaluated.calls.append('target-read')
        return evaluated.frame.copy(deep=True)
    monkeypatch.setattr(evaluation, 'evaluation_targets', targets)
    return evaluated


@pytest.mark.parametrize('plan_type', [metadata.PredictionPlan, InheritedPlan], ids=['default', 'inherited'])
def test_default_and_inherited_roots_bind_evaluation_scores_and_diagnostics(plan_type):
    closure = source_closure(ROOT, list(plan_type.source_roots))
    for relative in BOUND_SOURCES:
        assert closure.get(relative) == 'sha256:' + sha256((ROOT / relative).read_bytes()).hexdigest()


def test_identical_evaluation_bytes_allow_real_frozen_read_and_readonly_replay(composed, monkeypatch):
    run, case = composed.run, composed.case
    path = case.code / EVALUATION_SOURCE
    path.write_bytes(path.read_bytes())
    assert run.admit() == run.record('seal')
    saved = evaluation.evaluate_outer(run, composed.fits, '2024')
    assert composed.calls == ['target-read']
    before = snapshot(case)
    monkeypatch.setattr(evaluation, 'evaluation_targets', lambda *args: pytest.fail('target reread'))
    record, scores, diagnostics = evaluation.read_evaluated_outer(run, composed.fits, '2024')
    assert record == saved and len(scores) == len(diagnostics) == 24
    assert snapshot(case) == before


@pytest.mark.parametrize('action', ['evaluate', 'replay'])
def test_evaluation_source_drift_refuses_before_start_fits_cas_or_targets(composed, monkeypatch, action):
    run, case = composed.run, composed.case
    if action == 'replay':
        evaluation.evaluate_outer(run, composed.fits, '2024')
    sealed, before = run.record('seal'), snapshot(case)
    path = case.code / EVALUATION_SOURCE
    path.write_bytes(path.read_bytes() + b'\n# Evaluation source-only mutation probe.\n')
    changed = run.preflight()
    assert changed['run_plan_id'] != sealed['run_plan_id']
    assert set(changed['source_closure']) == set(sealed['source_closure'])
    assert {key for key in changed['source_closure']
        if changed['source_closure'][key] != sealed['source_closure'][key]} == {EVALUATION_SOURCE}
    assert changed['source_closure'][EVALUATION_SOURCE] == 'sha256:' + sha256(path.read_bytes()).hexdigest()
    def forbidden(*args, **kwargs):
        pytest.fail('source drift reached journal, fits, stored evidence or targets')
    original_record = run.record
    def only_seal(slot, body=None):
        if slot != 'seal' or body is not None:
            forbidden()
        return original_record(slot, body)
    monkeypatch.setattr(run, 'record', only_seal)
    for name in ('publish', 'read_section'):
        monkeypatch.setattr(run, name, forbidden)
    monkeypatch.setattr(composed.fits, 'read', forbidden)
    monkeypatch.setattr(composed.fits.trials, 'read_fit_context', forbidden)
    monkeypatch.setattr(evidence, 'residual_pool', forbidden)
    monkeypatch.setattr(evidence, 'training_baseline', forbidden)
    monkeypatch.setattr(evaluation, 'load_inputs', forbidden)
    monkeypatch.setattr(evaluation, 'evaluation_targets', forbidden)
    assert evaluation.read_frozen_outer is evidence.read_frozen_outer
    operation = evaluation.evaluate_outer if action == 'evaluate' else evaluation.read_evaluated_outer
    with pytest.raises(OpsError) as error:
        operation(run, composed.fits, '2024')
    assert error.value.code == 'EXPERIMENT_IDENTITY_CONFLICT' and not error.value.problem.retryable
    assert snapshot(case) == before


@pytest.mark.parametrize('year', [None, 2024, True, 2024.0, b'2024', [], {}, (), YearString('2024')])
def test_replay_invalid_year_uses_real_p7_guard_before_any_effect(native_case, monkeypatch, year):
    run, before = native_case.run, snapshot(native_case)
    def forbidden(*args, **kwargs):
        pytest.fail('invalid replay year reached admission, journal, fits, CAS or targets')
    for name in ('admit', 'record', 'publish', 'read_section'):
        monkeypatch.setattr(run, name, forbidden)
    fits = SimpleNamespace(read=forbidden, trials=SimpleNamespace(read_fit_context=forbidden))
    monkeypatch.setattr(evaluation, 'load_inputs', forbidden)
    monkeypatch.setattr(evaluation, 'evaluation_targets', forbidden)
    # No evaluated fixture or fake frozen reader participates in this path.
    assert evaluation.read_frozen_outer is evidence.read_frozen_outer
    with pytest.raises(OpsError) as error:
        evaluation.read_evaluated_outer(run, fits, year)
    assert error.value.code == 'INVALID_EXPERIMENT_SPEC' and not error.value.problem.retryable
    assert snapshot(native_case) == before
