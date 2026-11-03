"""Real source traversal and before-effect source-drift refusal for P7 callers."""
from hashlib import sha256
from pathlib import Path
import shutil

import pytest

from engine.v2.ops.errors import OpsError
from engine.v2.ops.fingerprints import source_closure
from experiments import distribution_prediction_evidence as evidence
from experiments import distribution_prediction_plan as metadata
from experiments import magnitude_search_evidence as native_evidence
from tests.v2.research.test_distribution_prediction_evidence import frozen as frozen
from tests.v2.research.test_distribution_prediction_plan import case as native_case

ROOT = Path(metadata.__file__).resolve().parents[1]
BOUND_SOURCES = (
    'experiments/distribution_prediction_evidence.py',
    'experiments/distribution_prediction_baseline.py',
)


class InheritedPlan(metadata.PredictionPlan):
    pass


@pytest.fixture(scope='module')
def code_snapshot(tmp_path_factory):
    destination = tmp_path_factory.mktemp('prediction-source-snapshot')
    # Include the required paths even in the old-roots negative control. Copy
    # only source-closure files, never catalogs, study data or installed state.
    paths = source_closure(ROOT, [*metadata.PredictionPlan.source_roots, *BOUND_SOURCES])
    for relative in paths:
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


@pytest.mark.parametrize('plan_type', [metadata.PredictionPlan, InheritedPlan], ids=['default', 'inherited'])
def test_default_and_inherited_roots_bind_actual_evidence_and_baseline_bytes(plan_type):
    closure = source_closure(ROOT, list(plan_type.source_roots))
    for relative in BOUND_SOURCES:
        assert closure.get(relative) == 'sha256:' + sha256((ROOT / relative).read_bytes()).hexdigest()


def test_identical_source_bytes_preserve_seal_and_freeze_replay(frozen):
    run, case = frozen.run, frozen.case
    sealed = run.record('seal')
    for relative in BOUND_SOURCES:
        path = case.code / relative
        path.write_bytes(path.read_bytes())
    assert run.admit() == sealed
    saved = evidence.freeze_outer(run, frozen.fits, '2024')
    before = snapshot(case)
    record, values = evidence.read_frozen_outer(run, frozen.fits, '2024')
    assert record == saved and set(values) == {1, 3, 5}
    assert snapshot(case) == before


@pytest.mark.parametrize('relative', BOUND_SOURCES, ids=['evidence', 'baseline'])
@pytest.mark.parametrize('action', ['freeze', 'replay'])
def test_source_only_drift_refuses_old_seal_before_effects(frozen, monkeypatch, relative, action):
    run, case = frozen.run, frozen.case
    if action == 'replay':
        evidence.freeze_outer(run, frozen.fits, '2024')
    sealed, before = run.record('seal'), snapshot(case)
    path = case.code / relative
    path.write_bytes(path.read_bytes() + b'\n# Source-only mutation probe.\n')
    changed = run.preflight()
    assert changed['run_plan_id'] != sealed['run_plan_id']
    assert set(changed['source_closure']) == set(sealed['source_closure'])
    assert {key for key in changed['source_closure']
        if changed['source_closure'][key] != sealed['source_closure'][key]} == {relative}
    assert changed['source_closure'][relative] == 'sha256:' + sha256(path.read_bytes()).hexdigest()
    with pytest.raises(OpsError) as error:
        run.admit()
    assert error.value.code == 'EXPERIMENT_IDENTITY_CONFLICT' and not error.value.problem.retryable

    def forbidden(*args, **kwargs):
        pytest.fail('source drift reached fit/context, evidence, journal publication or target access')
    original_record = run.record
    def only_seal(slot, body=None):
        if slot != 'seal' or body is not None:
            forbidden()
        return original_record(slot, body)
    monkeypatch.setattr(run, 'record', only_seal)
    for name in ('publish', 'read_section'):
        monkeypatch.setattr(run, name, forbidden)
    monkeypatch.setattr(frozen.fits, 'read', forbidden)
    monkeypatch.setattr(frozen.fits.trials, 'read_fit_context', forbidden)
    monkeypatch.setattr(evidence, 'residual_pool', forbidden)
    monkeypatch.setattr(evidence, 'training_baseline', forbidden)
    monkeypatch.setattr(native_evidence, 'evaluation_targets', forbidden)
    operation = evidence.freeze_outer if action == 'freeze' else evidence.read_frozen_outer
    with pytest.raises(OpsError) as error:
        operation(run, frozen.fits, '2024')
    assert error.value.code == 'EXPERIMENT_IDENTITY_CONFLICT' and not error.value.problem.retryable
    assert snapshot(case) == before
