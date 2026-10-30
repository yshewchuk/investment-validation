"""Qualified panel values stage keeps anchor caches per active ticker."""
from types import SimpleNamespace

from engine.v2.research import _magnitude_panel_sources as sources
from engine.v2.research import magnitude_panel_qualified as panel
from experiments import size_panel
from tests.v2.research.test_magnitude_panel_qualified import panel_data  # noqa: F401  (fixture)


def _records(count=3):
    ref = SimpleNamespace(object_id="object", content_hash="sha256:" + "a" * 64)
    return [SimpleNamespace(partition_key=f"T{i}", primary_key_min=(f"T{i}", "2020-01-01"),
        primary_key_max=(f"T{i}", "2020-12-31"), fragment_id=f"f{i}", object_ref=ref) for i in range(count)]


def test_anchor_lookup_clear_drops_entries_and_recomputes_identical_values():
    lookup = sources.anchor_lookup(_records())
    first = [lookup(f"T{i}", "2020-06-01", {"capture_id": "v1"}) for i in range(3)]
    assert lookup.size() == 3
    lookup.clear()
    assert lookup.size() == 0
    assert [lookup(f"T{i}", "2020-06-01", {"capture_id": "v1"}) for i in range(3)] == first


def _collect(data, monkeypatch, *, keep_cache):
    built = []
    real = sources.anchor_lookup

    def tracked(*args, **kwargs):
        lookup = real(*args, **kwargs)
        calls = []
        wrapper = lambda *a, **k: lookup(*a, **k)  # noqa: E731
        wrapper.size = lookup.size
        wrapper.clear = (lambda: calls.append(lookup.size())) if keep_cache else (
            lambda: (calls.append(lookup.size()), lookup.clear()))
        wrapper.calls = calls
        built.append(wrapper)
        return wrapper

    monkeypatch.setattr(sources, "anchor_lookup", tracked)
    request = data.panel_request
    prepared = panel._prepare(data.repository, data.panel_source, producer=size_panel._producer(request),
        history_admitted=size_panel._history(data.repository, data.panel_source, request), **request)
    rows = list(panel._rows(data.repository, data.panel_source, request["price_history_snapshot"], prepared))
    return rows, built


def test_values_stage_clears_ticker_anchor_caches_without_changing_rows(panel_data, monkeypatch):
    kept, _ = _collect(panel_data, monkeypatch, keep_cache=True)
    cleared, built = _collect(panel_data, monkeypatch, keep_cache=False)
    assert cleared and cleared == kept
    assert len(built) >= 3
    ticker_lookups = [b for b in built if b.calls]
    assert len(ticker_lookups) == 2 and all(b.calls == [0] for b in ticker_lookups)  # cleared once, at first ticker
    assert all(b.size() > 0 for b in ticker_lookups)  # and refilled by the active ticker
    regime = [b for b in built if not b.calls and b.size()]
    assert len(regime) == 1  # the ^GSPC lookup is never cleared
