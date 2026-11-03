"""All-horizon freeze order, provenance recheck and read-only evidence replay."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError
from experiments import distribution_prediction_evidence as evidence
from tests.v2.research.test_distribution_prediction_plan import case as case


@pytest.fixture
def frozen(case, monkeypatch):
    run, calls = case.run, []
    plan = run.seal(clock=case.clock)
    context = {"plan": {"plan_id": plan['admission_plan_id'], "recipe": {"horizons": [5, 3, 1]}},
        "fold": {"year": '2024', "evaluation_event_ids": ['Z', 'A'], 'training_labels_before': '2024-01-01'},
        "admission": {"native": 'admission'}}
    results = {h: {"receipt": {"schema_version": 'distribution_fitted.v2', "status": 'completed',
        "result_id": content_hash(h), "claim": {"admission_id": content_hash(context['admission']),
        "plan_id": plan['admission_plan_id'], "role": 'outer', "outer_year": '2024', "inner_year": None,
        "trial": {"horizon": h}}}, "rows": [{'event_id': e, 'probabilities': [h]} for e in ('Z', 'A')],
        "tail": {"policy": 'test'}} for h in (1, 3, 5)}
    def read(role, year, horizon):
        calls.append(('fit-read', horizon))
        return deepcopy(results[horizon])
    fits = SimpleNamespace(trials=SimpleNamespace(read_fit_context=lambda *args: deepcopy(context)), read=read)
    pools = {h: {'pool_id': content_hash(['pool', h]), 'destination': h} for h in results}
    baselines = {h: {'result_id': results[h]['receipt']['result_id'], 'training': [{'event_id': 'T', 'absolute_move_pct': h}]} for h in results}
    monkeypatch.setattr(evidence, 'residual_pool', lambda fits, role, year, horizon: deepcopy(pools[horizon]))
    monkeypatch.setattr(evidence, 'training_baseline', lambda fits, year, horizon: deepcopy(baselines[horizon]))
    original = run.publish
    def publish(*args, **kwargs):
        calls.append(('publish', args[0]))
        return original(*args, **kwargs)
    monkeypatch.setattr(run, 'publish', publish)
    return SimpleNamespace(run=run, fits=fits, context=context, results=results, pools=pools,
        baselines=baselines, calls=calls, case=case)


def test_every_horizon_validated_before_first_section_then_exact_readonly_replay(frozen, monkeypatch):
    saved = evidence.freeze_outer(frozen.run, frozen.fits, '2024')
    assert [entry['horizon'] for entry in saved['horizons']] == [1, 3, 5]
    assert frozen.calls[:3] == [('fit-read', h) for h in (1, 3, 5)]
    assert len([x for x in frozen.calls if x[0] == 'publish']) == 6
    for entry in saved['horizons']:
        assert entry['forecasts']['descriptor']['row_count'] == 2
        assert entry['reference']['descriptor']['row_count'] == 1
    before = frozen.case.conn.total_changes
    monkeypatch.setattr(frozen.run, 'publish', lambda *a, **k: pytest.fail('replay publication'))
    record, values = evidence.read_frozen_outer(frozen.run, frozen.fits, '2024')
    assert record == saved and set(values) == {1, 3, 5}
    assert frozen.case.conn.total_changes == before


@pytest.mark.parametrize('change', ['incomplete', 'foreign', 'order', 'unbounded', 'baseline'])
def test_bad_horizon_refuses_before_any_forecast_publication(frozen, change):
    row = frozen.results[5]
    if change == 'incomplete':
        row['receipt']['status'] = 'started'
    if change == 'foreign':
        row['receipt']['claim']['admission_id'] = 'foreign'
    if change == 'order':
        row['rows'].reverse()
    if change == 'unbounded':
        row['receipt']['schema_version'] = 'distribution_fitted.v1'
    if change == 'baseline':
        frozen.baselines[5]['result_id'] = 'foreign'
    with pytest.raises(OpsError):
        evidence.freeze_outer(frozen.run, frozen.fits, '2024')
    assert not any(call[0] == 'publish' for call in frozen.calls)
    assert frozen.run.record('freeze-start:2024') is not None
    assert frozen.run.record('frozen:2024') is None
    with pytest.raises(OpsError):
        evidence.freeze_outer(frozen.run, frozen.fits, '2024')


@pytest.mark.parametrize('change', ['pool', 'training', 'forecasts', 'tail'])
def test_replay_rebuilds_native_evidence_not_just_saved_hashes(frozen, change):
    evidence.freeze_outer(frozen.run, frozen.fits, '2024')
    if change == 'pool':
        frozen.pools[1]['pool_id'] = content_hash('changed')
    if change == 'training':
        frozen.baselines[1]['training'][0]['absolute_move_pct'] = 999
    if change == 'forecasts':
        frozen.results[1]['rows'][0]['probabilities'] = [999]
    if change == 'tail':
        frozen.results[1]['tail']['policy'] = 'changed'
    with pytest.raises(OpsError):
        evidence.read_frozen_outer(frozen.run, frozen.fits, '2024')


def test_duplicate_freeze_refuses_before_second_fit_read(frozen):
    evidence.freeze_outer(frozen.run, frozen.fits, '2024')
    before = list(frozen.calls)
    with pytest.raises(OpsError):
        evidence.freeze_outer(frozen.run, frozen.fits, '2024')
    assert frozen.calls == before


@pytest.mark.parametrize('change', ['horizons', 'roster', 'plan'])
def test_invalid_native_axes_refuse_before_fit_reads(frozen, change):
    if change == 'horizons':
        frozen.context['plan']['recipe']['horizons'] = [1, 3]
    if change == 'roster':
        frozen.context['fold']['evaluation_event_ids'].append('foreign')
    if change == 'plan':
        frozen.context['plan']['plan_id'] = content_hash('foreign')
    with pytest.raises(OpsError):
        evidence.freeze_outer(frozen.run, frozen.fits, '2024')
    assert frozen.calls == []


def test_read_requires_final_frozen_marker_and_all_horizon_entries(frozen, monkeypatch):
    with pytest.raises(OpsError):
        evidence.read_frozen_outer(frozen.run, frozen.fits, '2024')
    saved = evidence.freeze_outer(frozen.run, frozen.fits, '2024')
    original = frozen.run.record
    def changed(slot, body=None):
        if slot == 'frozen:2024':
            return {**saved, 'horizons': saved['horizons'][:-1]}
        return original(slot, body)
    monkeypatch.setattr(frozen.run, 'record', changed)
    with pytest.raises(OpsError):
        evidence.read_frozen_outer(frozen.run, frozen.fits, '2024')
