"""Prediction metadata seals and real create-only journal/CAS boundaries."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from engine.v2.foundation import ArtifactStore, SystemClock, content_hash
from engine.v2.ops.catalog import connect
from engine.v2.ops.errors import OpsError
from engine.v2.ops.migrations import migrate
from experiments import distribution_prediction_plan as metadata
from experiments.magnitude_search_records import MIGRATIONS, OWNER, append_record


def request():
    return {"schema_version": "distribution_prediction_request.v1", "max_events": 100,
        "limits": {"parent": {"memory_bytes": 3 << 30, "cpu_seconds": 60, "wall_seconds": 60}, "job_wall_seconds": 60},
        "uncertainty": {"seed": 0, "resamples": 100, "block_days": 30, "block_origin": "2020-01-01", "confidence": .95}}


@pytest.fixture
def case(tmp_path, monkeypatch):
    conn, store, clock = connect(tmp_path / 'catalog'), ArtifactStore(tmp_path / 'store'), SystemClock()
    conn.execute('CREATE TABLE experiment_runs(run_id TEXT PRIMARY KEY)')
    conn.execute("INSERT INTO experiment_runs VALUES ('prediction')")
    migrate(conn, OWNER, MIGRATIONS, clock=clock)
    base = {"plan_id": content_hash('native'), "recipe": {"horizons": [5, 3, 1]},
            "budget": {"child_limits": {"memory_bytes": 2 << 30}}}
    native = {"folds": [{"outer": {"year": "2024", "evaluation_event_ids": ["Z", "A"]}}]}
    state = {"closure": content_hash('source')}
    calls = []
    def seal(*, clock):
        calls.append('native-seal')
        return append_record(conn, store, 'prediction', 'distribution:seal', base)
    trials = SimpleNamespace(conn=conn, store=store, own=SimpleNamespace(run_id='prediction'), source=object(),
        request={}, source_plan_ref=object(), recipe={}, caps={}, seal=seal)
    monkeypatch.setattr(metadata, 'prepare_plan', lambda *args: deepcopy((base, native)))
    monkeypatch.setattr(metadata, 'source_closure', lambda root, paths: {path: state['closure'] for path in paths})
    run = metadata.PredictionPlan(trials, request())
    yield SimpleNamespace(run=run, trials=trials, conn=conn, store=store, clock=clock, base=base, native=native, state=state, calls=calls)
    conn.close()


@pytest.mark.parametrize('field,value', [
    ('max_events', 0), ('max_events', True), ('max_events', 1.5), ('schema_version', 'strategy'),
    ('limits', None), ('uncertainty', None), ('unknown', 1),
])
def test_closed_request_refuses(field, value):
    value_request = request()
    value_request[field] = value
    with pytest.raises(OpsError) as error:
        metadata.validate_request(value_request)
    assert error.value.code == 'INVALID_EXPERIMENT_SPEC'


@pytest.mark.parametrize('field,value', [('seed', -1), ('seed', True), ('resamples', 0), ('block_days', 0),
    ('confidence', True), ('confidence', float('nan')), ('confidence', 1), ('block_origin', '2020-1-1')])
def test_invalid_uncertainty_refuses(field, value):
    value_request = request()
    value_request['uncertainty'][field] = value
    with pytest.raises(OpsError):
        metadata.validate_request(value_request)


def test_constructor_detaches_request_and_preflight_has_no_writes(case):
    original = request()
    run = metadata.PredictionPlan(case.trials, original)
    original['limits']['parent']['cpu_seconds'] = 999
    assert run.request == request()
    plan = run.preflight()
    assert plan['evaluation_event_ids'] == ['Z', 'A'] and plan['event_count'] == 2
    assert plan['arms'] == ['histogram', 'same_mean', 'matched_point', 'training_empirical']
    assert plan['run_plan_id'] == content_hash({k: v for k, v in plan.items() if k != 'run_plan_id'})
    assert plan['calibration_identity'] == 'uncalibrated' and plan['final_read_authority'] == 'none'
    assert case.calls == [] and case.conn.execute('SELECT count(*) FROM magnitude_search_records').fetchone()[0] == 0


@pytest.mark.parametrize('change', ['policy', 'limits', 'source', 'native'])
def test_seal_binds_policy_limits_installed_source_and_admission(case, change):
    expected = case.run.seal(clock=case.clock)
    assert case.run.admit() == expected
    if change == 'policy':
        case.run.request['uncertainty']['resamples'] += 1
    if change == 'limits':
        case.run.request['limits']['parent']['cpu_seconds'] += 1
    if change == 'source':
        case.state['closure'] = content_hash('changed')
    if change == 'native':
        case.base['plan_id'] = content_hash('changed')
    with pytest.raises(OpsError) as error:
        case.run.admit()
    assert error.value.code == 'EXPERIMENT_IDENTITY_CONFLICT'


@pytest.mark.parametrize('change', ['empty', 'duplicate', 'horizons', 'memory', 'count'])
def test_invalid_preflight_refuses_before_native_seal(case, change):
    if change == 'empty':
        case.native['folds'].clear()
    if change == 'duplicate':
        case.native['folds'][0]['outer']['evaluation_event_ids'].append('Z')
    if change == 'horizons':
        case.base['recipe']['horizons'] = [1, 3]
    if change == 'memory':
        case.base['budget']['child_limits']['memory_bytes'] = 4 << 30
    if change == 'count':
        case.run.request['max_events'] = 1
    with pytest.raises(OpsError):
        case.run.seal(clock=case.clock)
    assert case.calls == []


def test_namespaced_record_is_create_only_even_for_identical_retry(case):
    case.run.seal(clock=case.clock)
    case.run.record('started', {'test': 'claim'})
    assert case.run.record('started') == {'test': 'claim'}
    with pytest.raises(OpsError):
        case.run.record('started', {'test': 'claim'})
    assert case.conn.execute("SELECT count(*) FROM magnitude_search_records WHERE slot='distribution:prediction-started'").fetchone()[0] == 1


def test_bounded_section_roundtrip_is_detached_and_tampering_refuses(case):
    rows = [{'event_id': 'Z', 'value': 1}, {'event_id': 'A', 'value': 2}]
    section = case.run.publish('distribution_prediction_test', rows, count=2)
    assert list(case.run.read_section(section)) == rows
    rows[0]['value'] = 999
    assert list(case.run.read_section(section))[0]['value'] == 1
    section['descriptor']['row_count'] += 1
    with pytest.raises(OpsError):
        list(case.run.read_section(section))


@pytest.mark.parametrize('name,rows', [('other', []), ('distribution_prediction_test', None),
    ('distribution_prediction_test', [{'position': 1}]), ('distribution_prediction_test', [5])])
def test_section_shapes_refuse(case, name, rows):
    with pytest.raises(OpsError):
        case.run.publish(name, rows, count=1)


@pytest.mark.parametrize('field,value', [('memory_bytes', True), ('cpu_seconds', 0), ('wall_seconds', None)])
def test_invalid_parent_limit_refuses(field, value):
    document = request()
    document['limits']['parent'][field] = value
    with pytest.raises(OpsError):
        metadata.validate_request(document)


def test_metadata_import_and_preflight_do_not_depend_on_economics_or_future_runner():
    import os
    import subprocess
    import sys
    script = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname in ('experiments.distribution_run_request', 'experiments.distribution_market',
                        'experiments.distribution_book', 'experiments.distribution_run',
                        'experiments.distribution_prediction_run'):
            raise AssertionError('forbidden import: ' + fullname)
sys.meta_path.insert(0, Block())
from experiments.distribution_prediction_plan import PredictionPlan
assert not hasattr(PredictionPlan, 'run') and not hasattr(PredictionPlan, 'collect')
print('metadata only')
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True,
        timeout=15, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'metadata only'


@pytest.mark.parametrize('count', [0, 2, True, None, -1])
def test_publication_requires_declared_exact_row_count(case, count):
    with pytest.raises(OpsError):
        case.run.publish('distribution_prediction_test', [{'value': 1}], count=count)
