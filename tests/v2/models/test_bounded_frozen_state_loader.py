"""Bounded frozen-state reads over synthetic temporary artifacts only."""
import builtins
import os
import sys
from pathlib import Path

import pytest

from engine.v2.models import frozen_state, frozen_state_io
from engine.v2.models.admissible_table import make_admissible_depth_table
from engine.v2.models.frozen_state import (
    FrozenStateError, FrozenStateLoader, FrozenStateRef, serialize_frozen_state,
)
from engine.v2.models.lineage import Lineage


def _staged(tmp_path):
    state = make_admissible_depth_table(
        table_id="synthetic-table", version="v1", breakpoints=((2.0, 3.0),),
        fallback=4.0, provenance="synthetic", lineage=Lineage())
    payload = serialize_frozen_state(state)
    path = tmp_path / "table.json"
    path.write_bytes(payload)
    return state, path, payload, FrozenStateRef(path=path.name, content_hash=state.content_hash)


@pytest.mark.parametrize("limit", [0, -1, True, False, 1.5, "1024", sys.maxsize, sys.maxsize + 1])
def test_invalid_bound_refuses_before_io(tmp_path, limit):
    with pytest.raises(FrozenStateError, match="positive integer"):
        FrozenStateLoader(tmp_path, max_bytes=limit)


@pytest.mark.parametrize("headroom", [0, 1])
def test_at_or_below_bound_keeps_typed_values_and_cache(tmp_path, headroom):
    state, _path, payload, ref = _staged(tmp_path)
    loader = FrozenStateLoader(tmp_path, max_bytes=len(payload) + headroom)
    loaded = loader.load(ref)
    assert loaded == state
    assert loaded.key == ("synthetic-table", "v1")
    assert loaded.breakpoints == ((2.0, 3.0),)
    assert loader.cache_size == 1
    assert loader.load(ref) is loaded


def test_omitted_bound_preserves_existing_read_path(tmp_path, monkeypatch):
    state, _path, _payload, ref = _staged(tmp_path)

    def never_bounded(*args):
        pytest.fail("an existing unbounded caller entered the new bounded path")

    monkeypatch.setattr(frozen_state, "read_bounded_state", never_bounded)
    assert FrozenStateLoader(tmp_path).load(ref) == state


def _watch_reads(monkeypatch, *, chunk_size=None):
    requests = []

    class WatchedStream:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def fileno(self):
            return self.stream.fileno()

        def read(self, count):
            requests.append(count)
            return self.stream.read(min(count, chunk_size) if chunk_size else count)

    def watched_open(*args, **kwargs):
        assert kwargs.get("buffering") == 0  # A buffered reader can read ahead past the ceiling.
        return WatchedStream(builtins.open(*args, **kwargs))

    monkeypatch.setattr(frozen_state_io, "open", watched_open, raising=False)
    return requests


def test_oversize_refuses_before_content_read_hash_or_decode(tmp_path, monkeypatch):
    _state, _path, payload, ref = _staged(tmp_path)
    requests = _watch_reads(monkeypatch)

    def never_reached(*args, **kwargs):
        pytest.fail("oversized artifact reached hashing or JSON decoding")

    monkeypatch.setattr(frozen_state.hashlib, "sha256", never_reached)
    monkeypatch.setattr(frozen_state.json, "loads", never_reached)
    loader = FrozenStateLoader(tmp_path, max_bytes=len(payload) - 1)
    with pytest.raises(FrozenStateError, match="exceeds byte limit"):
        loader.load(ref)
    assert requests == []
    assert loader.cache_size == 0


def test_growth_after_size_check_still_uses_bounded_read_before_decode(tmp_path, monkeypatch):
    _state, path, payload, ref = _staged(tmp_path)
    limit = len(payload)
    requests = _watch_reads(monkeypatch)
    real_fstat = os.fstat

    def grow_after_fstat(fd):
        old = real_fstat(fd)
        with path.open("ab") as stream:
            stream.write(b"padding after size check")
        return old

    def never_decode(*args, **kwargs):
        pytest.fail("oversized artifact reached JSON decoding")

    monkeypatch.setattr(frozen_state_io.os, "fstat", grow_after_fstat)
    monkeypatch.setattr(frozen_state.json, "loads", never_decode)
    loader = FrozenStateLoader(tmp_path, max_bytes=limit)
    with pytest.raises(FrozenStateError, match="exceeds byte limit"):
        loader.load(ref)
    assert requests == [limit + 1]
    assert loader.cache_size == 0


def test_valid_read_is_bounded_and_never_calls_path_read_bytes(tmp_path, monkeypatch):
    state, _path, payload, ref = _staged(tmp_path)
    requests = _watch_reads(monkeypatch)

    def never_read_bytes(*args):
        pytest.fail("unbounded read_bytes was called")

    monkeypatch.setattr(Path, "read_bytes", never_read_bytes)
    assert FrozenStateLoader(tmp_path, max_bytes=len(payload)).load(ref) == state
    assert requests == [len(payload) + 1, 1]


def test_bounded_short_reads_preserve_full_artifact(tmp_path, monkeypatch):
    state, _path, payload, ref = _staged(tmp_path)
    requests = _watch_reads(monkeypatch, chunk_size=17)
    assert FrozenStateLoader(tmp_path, max_bytes=len(payload)).load(ref) == state
    assert requests == list(range(len(payload) + 1, 1, -17)) + [1]


def test_fifo_is_opened_nonblocking_and_refused_before_read(tmp_path, monkeypatch):
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    requests = _watch_reads(monkeypatch)
    real_open = os.open
    flags_seen = []

    def checked_open(path, flags, *args, **kwargs):
        flags_seen.append(flags)
        assert flags & os.O_NONBLOCK  # Fails before a mutant can block opening the FIFO.
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(frozen_state_io.os, "open", checked_open)
    ref = FrozenStateRef(path="fifo", content_hash="sha256:" + "0" * 64)
    loader = FrozenStateLoader(tmp_path, max_bytes=1)
    with pytest.raises(FrozenStateError, match="regular file"):
        loader.load(ref)
    assert len(flags_seen) == 1
    assert requests == []
    assert loader.cache_size == 0
