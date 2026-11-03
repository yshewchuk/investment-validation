"""Typed freeze admission and bounded, source-authenticated reference transport."""
from copy import deepcopy

import pytest

from engine.v2.foundation import canonical_json
from engine.v2.ops.errors import OpsError, fail
from experiments import distribution_prediction_evidence as evidence
from experiments.magnitude_search_shards import MAX_SHARD_BYTES
from tests.v2.research.test_distribution_prediction_evidence import frozen as frozen
from tests.v2.research.test_distribution_prediction_plan import case as case


def snapshot(frozen):
    return (frozen.case.conn.total_changes,
        sorted((str(path.relative_to(frozen.case.store.root)), path.read_bytes())
            for path in frozen.case.store.root.rglob('*') if path.is_file()))


@pytest.mark.parametrize('operation', [evidence.freeze_outer, evidence.read_frozen_outer])
@pytest.mark.parametrize('year', [None, 2024, True, 2024.0, [], {}, ['2024']])
def test_invalid_year_refuses_before_any_admission_or_effect(frozen, monkeypatch, operation, year):
    before = snapshot(frozen)
    monkeypatch.setattr(frozen.run, 'admit', lambda: pytest.fail('invalid year reached admission'))
    with pytest.raises(OpsError) as error:
        operation(frozen.run, frozen.fits, year)
    assert error.value.code == 'INVALID_EXPERIMENT_SPEC'
    assert frozen.calls == [] and snapshot(frozen) == before


@pytest.mark.parametrize('sections', [None, {}, 1, [], [None] * 3, [{}] * 3,
    [{'horizon': 1}] * 3, [[], [], []], ['1', '3', '5'],
    [{'horizon': h, 'forecasts': {}, 'reference': {}} for h in (True, 3, 5)],
    [{'horizon': h, 'forecasts': {}, 'reference': {}} for h in ('1', 3, 5)],
    [{'horizon': h, 'forecasts': {}, 'reference': {}} for h in (1, 5, 3)],
    [{'horizon': h, 'forecasts': {}, 'reference': {}, 'extra': 1} for h in (1, 3, 5)]])
def test_saved_horizon_shape_refuses_before_section_reads(frozen, monkeypatch, sections):
    saved = evidence.freeze_outer(frozen.run, frozen.fits, '2024')
    before, original = snapshot(frozen), frozen.run.record
    monkeypatch.setattr(frozen.run, 'record', lambda slot, body=None:
        {**saved, 'horizons': sections} if slot == 'frozen:2024' else original(slot, body))
    monkeypatch.setattr(frozen.run, 'read_section', lambda *args: pytest.fail('malformed roster reached CAS'))
    with pytest.raises(OpsError) as error:
        evidence.read_frozen_outer(frozen.run, frozen.fits, '2024')
    assert error.value.code == 'EXPERIMENT_IDENTITY_CONFLICT'
    assert snapshot(frozen) == before


def test_large_native_reference_freezes_with_original_shard_cap_and_readonly_replay(frozen, monkeypatch):
    frozen.baselines[1]['training'] = [
        {'event_id': format(index, '064x'), 'absolute_move_pct': 1} for index in range(85_000)]
    large = canonical_json(evidence._reference(1, {'result': frozen.results[1],
        'pool': frozen.pools[1], 'baseline': frozen.baselines[1]}))
    assert len(large.encode()) > MAX_SHARD_BYTES
    saved = evidence.freeze_outer(frozen.run, frozen.fits, '2024')
    assert saved['schema_version'] == 'distribution_prediction_frozen.v2'
    assert saved['reference_encoding'] == 'canonical_json_fragments.v1'
    assert saved['reference_chunk_chars'] == 65_536
    section = saved['horizons'][0]['reference']
    assert section['descriptor']['row_count'] == (len(large) + 65_535) // 65_536
    assert len(section['descriptor']['shards']) >= 3
    for path in (frozen.case.store.root / 'objects').rglob('*'):
        if path.is_file():
            assert path.stat().st_size <= MAX_SHARD_BYTES
    original, before = frozen.run.read_section, snapshot(frozen)
    def streaming(section):
        class Rows:
            def __iter__(self):
                return original(section)
            def __length_hint__(self):
                pytest.fail('reference rows materialized')
        return Rows() if section['name'].endswith('_reference') else original(section)
    monkeypatch.setattr(frozen.run, 'read_section', streaming)
    monkeypatch.setattr(frozen.run, 'publish', lambda *a, **k: pytest.fail('replay publication'))
    record, values = evidence.read_frozen_outer(frozen.run, frozen.fits, '2024')
    assert record == saved and values[1]['baseline'] == frozen.baselines[1]
    assert snapshot(frozen) == before


@pytest.mark.parametrize('text', ['"\\\n\x00', 'é漢😀', '\\u0022', '__nonfinite__'])
def test_fragment_boundaries_preserve_nested_escaped_native_evidence(frozen, text):
    # Deliberately cross token/escape boundaries, including nested receipt lists.
    payload = text * 70_000
    frozen.results[1]['receipt']['nested'] = [{'value': payload, 'null': None, 'empty': []}]
    frozen.results[1]['tail']['samples'] = [0, 1.5, 1.5]
    frozen.pools[1]['residuals'] = [None, [], -2.5, 0]
    saved = evidence.freeze_outer(frozen.run, frozen.fits, '2024')
    assert saved['horizons'][0]['reference']['descriptor']['row_count'] > 1
    _, values = evidence.read_frozen_outer(frozen.run, frozen.fits, '2024')
    assert values[1]['result'] == frozen.results[1]
    assert values[1]['pool'] == frozen.pools[1]


@pytest.mark.parametrize('change', ['missing', 'extra', 'reordered', 'modified', 'extra_key', 'old_row'])
def test_valid_cas_substitution_cannot_authenticate_reference(frozen, monkeypatch, change):
    frozen.baselines[1]['nested'] = 'x' * 140_000
    saved = evidence.freeze_outer(frozen.run, frozen.fits, '2024')
    entry = saved['horizons'][0]
    rows = list(frozen.run.read_section(entry['reference']))
    if change == 'missing': rows.pop()
    if change == 'extra': rows.append(rows[-1])
    if change == 'reordered': rows.reverse()
    if change == 'modified': rows[0]['canonical_json'] += 'changed'
    if change == 'extra_key': rows[0]['untrusted'] = 1
    if change == 'old_row': rows = [evidence._reference(1, {'result': frozen.results[1],
        'pool': frozen.pools[1], 'baseline': frozen.baselines[1]})]
    # Republish with genuine content hashes, manifest and ordered positions.
    entry['reference'] = frozen.run.publish('distribution_prediction_forged_reference', rows, count=len(rows))
    original, before = frozen.run.record, snapshot(frozen)
    monkeypatch.setattr(frozen.run, 'record', lambda slot, body=None:
        saved if slot == 'frozen:2024' else original(slot, body))
    with pytest.raises(OpsError) as error:
        evidence.read_frozen_outer(frozen.run, frozen.fits, '2024')
    assert error.value.code == 'EXPERIMENT_IDENTITY_CONFLICT'
    assert snapshot(frozen) == before


@pytest.mark.parametrize('field,value', [('schema_version', 'distribution_prediction_frozen.v1'),
    ('reference_encoding', 'unknown.v1'), ('reference_chunk_chars', 65_535)])
def test_old_or_changed_reference_layout_refuses_without_reading_sections(frozen, monkeypatch, field, value):
    saved = evidence.freeze_outer(frozen.run, frozen.fits, '2024')
    saved[field] = value
    original, before = frozen.run.record, snapshot(frozen)
    monkeypatch.setattr(frozen.run, 'record', lambda slot, body=None:
        saved if slot == 'frozen:2024' else original(slot, body))
    monkeypatch.setattr(frozen.run, 'read_section', lambda *a: pytest.fail('old layout read'))
    with pytest.raises(OpsError) as error:
        evidence.read_frozen_outer(frozen.run, frozen.fits, '2024')
    assert error.value.code == 'EXPERIMENT_IDENTITY_CONFLICT'
    assert snapshot(frozen) == before


def test_interrupted_reference_publication_consumes_freeze_without_final_marker(frozen, monkeypatch):
    original = frozen.run.publish
    def interrupted(name, rows, *, count):
        if name.endswith('_reference'):
            raise fail('RESOURCE_UNAVAILABLE', 'injected storage interruption')
        return original(name, rows, count=count)
    monkeypatch.setattr(frozen.run, 'publish', interrupted)
    with pytest.raises(OpsError) as error:
        evidence.freeze_outer(frozen.run, frozen.fits, '2024')
    assert error.value.code == 'RESOURCE_UNAVAILABLE'
    assert frozen.run.record('freeze-start:2024') is not None
    assert frozen.run.record('frozen:2024') is None
    before = deepcopy(frozen.calls), snapshot(frozen)
    with pytest.raises(OpsError):
        evidence.freeze_outer(frozen.run, frozen.fits, '2024')
    assert (frozen.calls, snapshot(frozen)) == before
