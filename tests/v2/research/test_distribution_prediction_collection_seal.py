"""Real source closure binds collection/reporting before evaluated replay effects."""
import shutil
from hashlib import sha256
from pathlib import Path

import pytest

from engine.v2.foundation import content_hash
from engine.v2.models.no_fit import no_fit_guard
from engine.v2.ops.errors import OpsError
from engine.v2.ops.fingerprints import source_closure
from experiments import distribution_prediction_collection as collection
from experiments import distribution_prediction_evaluation as evaluation
from experiments import distribution_prediction_plan as metadata
from experiments import distribution_prediction_uncertainty as uncertainty
from experiments import magnitude_search_evidence as native_evidence
from tests.v2.research import test_distribution_prediction_plan as plan_tests
from tests.v2.research.test_distribution_prediction_collection import replay as replay
from tests.v2.research.test_distribution_prediction_evaluation import evaluated as evaluated

native_case = plan_tests.case

ROOT = Path(metadata.__file__).resolve().parents[1]
BOUND_SOURCES = (
    'experiments/distribution_prediction_collection.py',
    'experiments/distribution_prediction_report.py',
    'experiments/distribution_prediction_uncertainty.py',
    'experiments/distribution_prediction_diagnostic_report.py',
)


class InheritedPlan(metadata.PredictionPlan):
    pass


@pytest.fixture(scope='module')
def code_snapshot(tmp_path_factory):
    destination = tmp_path_factory.mktemp('prediction-collection-source')
    # Explicitly copy the required roots even for the old-roots negative
    # control. Only copied implementation bytes change, never installed code.
    for relative in source_closure(ROOT, [*metadata.PredictionPlan.source_roots, *BOUND_SOURCES]):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    return destination


@pytest.fixture(params=[metadata.PredictionPlan, InheritedPlan], ids=['default', 'inherited'])
def case(native_case, request, tmp_path, monkeypatch, code_snapshot):
    code = tmp_path / 'code'
    shutil.copytree(code_snapshot, code)
    # Replace the metadata fixture's fake digests with the actual import walker.
    monkeypatch.setattr(metadata, 'source_closure', lambda root, paths: source_closure(code, paths))
    native_case.run = request.param(native_case.trials, native_case.run.request)
    native_case.code = code
    return native_case


def snapshot(case):
    return (case.conn.total_changes, tuple(case.conn.iterdump()),
        tuple(sorted((str(path.relative_to(case.store.root)), sha256(path.read_bytes()).hexdigest())
            for path in case.store.root.rglob('*') if path.is_file())))


@pytest.mark.parametrize('plan_type', [metadata.PredictionPlan, InheritedPlan], ids=['default', 'inherited'])
def test_default_and_inherited_roots_bind_collection_and_reporting_bytes(plan_type):
    closure = source_closure(ROOT, list(plan_type.source_roots))
    for relative in BOUND_SOURCES:
        assert closure.get(relative) == 'sha256:' + sha256((ROOT / relative).read_bytes()).hexdigest()


def test_identical_source_bytes_allow_collection_from_real_readonly_evaluated_replay(replay):
    run, case = replay.run, replay.case
    sealed, before = run.record('seal'), snapshot(case)
    for relative in BOUND_SOURCES:
        path = case.code / relative
        path.write_bytes(path.read_bytes())
    assert run.preflight() == run.admit() == sealed
    assert collection.read_evaluated_outer is evaluation.read_evaluated_outer
    calls = list(replay.calls)
    with no_fit_guard():
        groups, summaries = collection.report_evidence(run, replay.fits)
        assert collection.report_evidence(run, replay.fits) == (groups, summaries)
    assert groups == [{'outer_year': '2024', 'evaluation_id': content_hash(replay.saved)}]
    assert summaries['forecast']['population'] == summaries['diagnostics']['population'] == 2
    assert summaries['forecast']['schema_version'] == 'distribution_prediction_score_summary.v1'
    assert summaries['diagnostics']['schema_version'] == 'distribution_prediction_diagnostic_summary.v1'
    assert replay.calls == [*calls, 'frozen', 'frozen']
    assert snapshot(case) == before


@pytest.mark.parametrize('relative', BOUND_SOURCES,
    ids=['collection', 'report', 'uncertainty', 'diagnostic_report'])
def test_source_only_drift_refuses_collection_before_evaluated_reads_or_effects(replay, monkeypatch, relative):
    run, case = replay.run, replay.case
    sealed, before = run.record('seal'), snapshot(case)
    calls = list(replay.calls)
    path = case.code / relative
    path.write_bytes(path.read_bytes() + b'\n# Collection source-only mutation probe.\n')
    changed = run.preflight()
    assert changed['run_plan_id'] != sealed['run_plan_id']
    assert set(changed['source_closure']) == set(sealed['source_closure'])
    assert {key for key in changed['source_closure']
        if changed['source_closure'][key] != sealed['source_closure'][key]} == {relative}
    assert changed['source_closure'][relative] == 'sha256:' + sha256(path.read_bytes()).hexdigest()

    def forbidden(*args, **kwargs):
        pytest.fail('source drift reached evaluated reads, evidence, publication, reports or targets')
    original_record = run.record
    def only_seal(slot, body=None):
        if slot != 'seal' or body is not None:
            forbidden()
        return original_record(slot, body)
    monkeypatch.setattr(run, 'record', only_seal)
    for name in ('publish', 'read_section'):
        monkeypatch.setattr(run, name, forbidden)
    for name in ('read_evaluated_outer', 'prediction_uncertainty', 'prediction_diagnostic_report'):
        monkeypatch.setattr(collection, name, forbidden)
    monkeypatch.setattr(uncertainty, 'prediction_report', forbidden)
    monkeypatch.setattr(replay.fits.trials, 'read_fit_context', forbidden)
    for name in ('read_frozen_outer', 'load_inputs', 'evaluation_targets'):
        monkeypatch.setattr(evaluation, name, forbidden)
    monkeypatch.setattr(native_evidence, 'evaluation_targets', forbidden)
    with pytest.raises(OpsError) as error:
        collection.report_evidence(run, replay.fits)
    assert error.value.code == 'EXPERIMENT_IDENTITY_CONFLICT' and not error.value.problem.retryable
    assert replay.calls == calls
    assert snapshot(case) == before
