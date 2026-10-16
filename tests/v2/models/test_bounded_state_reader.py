"""Public bounded-reader contract over synthetic temporary files only."""
import builtins
import errno
import os
import sys
from types import SimpleNamespace

import pytest

from engine.v2.models import frozen_state_io
from engine.v2.models.frozen_state_io import BoundedStateReadError, read_bounded_state


@pytest.fixture
def observed_io(monkeypatch):
    observed = SimpleNamespace(streams=[], requests=[], chunks=[], flags=[],
                               chunk_size=None, read_error=None)
    real_open = os.open

    def checked_open(path, flags, *args, **kwargs):
        observed.flags.append(flags)
        # Fail before opening a FIFO if a regression removes nonblocking mode.
        assert flags & os.O_NONBLOCK
        return real_open(path, flags, *args, **kwargs)

    class ObservedStream:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def fileno(self):
            return self.stream.fileno()

        def read(self, count):
            observed.requests.append(count)
            assert count > 0
            if observed.read_error is not None:
                raise observed.read_error
            if observed.chunk_size is not None:
                count = min(count, observed.chunk_size)
            chunk = self.stream.read(count)
            observed.chunks.append(chunk)
            return chunk

    def watched_open(*args, **kwargs):
        assert args[1] == "rb"
        # Buffered read-ahead would violate the byte budget.
        assert kwargs.get("buffering") == 0
        stream = builtins.open(*args, **kwargs)
        observed.streams.append(stream)
        return ObservedStream(stream)

    monkeypatch.setattr(frozen_state_io.os, "open", checked_open)
    monkeypatch.setattr(frozen_state_io, "open", watched_open, raising=False)
    try:
        yield observed
    finally:
        try:
            assert all(stream.closed for stream in observed.streams)
        finally:
            for stream in observed.streams:
                stream.close()


def _assert_refusal(error, code, message):
    assert type(error) is BoundedStateReadError
    assert error.code == code
    assert str(error) == message
    assert error.args == (message,)
    assert error.__cause__ is None


class IntSubclass(int):
    pass


@pytest.mark.parametrize("limit", [
    None, -1, True, False, 0.0, 1.5, "1", b"1", IntSubclass(1),
    sys.maxsize, sys.maxsize + 1,
])
def test_invalid_limits_refuse_before_open_with_fixed_error(tmp_path, monkeypatch, limit):
    def forbidden_open(*args, **kwargs):
        pytest.fail("an invalid byte limit reached filesystem I/O")

    monkeypatch.setattr(frozen_state_io, "open", forbidden_open, raising=False)
    with pytest.raises(BoundedStateReadError) as caught:
        read_bounded_state(tmp_path / "private-state", limit)
    _assert_refusal(caught.value, "INVALID_LIMIT",
                    "byte limit must be a nonnegative integer below sys.maxsize")


@pytest.mark.parametrize("kwargs", [{}, {"no_follow": False}, {"no_follow": True}])
@pytest.mark.parametrize("payload, limit, requests", [
    (b"", 0, [1]),
    (b"a", 1, [2, 1]),
    (b"a\x00b\xff", 4, [5, 1]),
    (b"a\x00b\xff", 5, [6, 2]),
])
def test_regular_files_return_exact_bytes_and_close(tmp_path, observed_io, kwargs,
                                                   payload, limit, requests):
    path = tmp_path / "state"
    path.write_bytes(payload)
    result = read_bounded_state(path, limit, **kwargs)
    assert type(result) is bytes
    assert result == payload
    assert observed_io.requests == requests
    assert len(observed_io.flags) == 1
    assert bool(observed_io.flags[0] & os.O_NOFOLLOW) == kwargs.get("no_follow", False)
    assert len(observed_io.streams) == 1
    assert observed_io.streams[0].closed


@pytest.mark.parametrize("kwargs", [{}, {"no_follow": False}])
def test_default_symlink_policy_keeps_regular_target_readable(tmp_path, observed_io, kwargs):
    target = tmp_path / "target"
    target.write_bytes(b"synthetic")
    link = tmp_path / "link"
    link.symlink_to(target)
    assert read_bounded_state(link, 9, **kwargs) == b"synthetic"
    assert not observed_io.flags[0] & os.O_NOFOLLOW


@pytest.mark.parametrize("target_kind", ["regular", "dangling", "fifo"])
def test_no_follow_refuses_final_symlinks_before_stat_or_read(
        tmp_path, monkeypatch, observed_io, target_kind):
    target = tmp_path / "target"
    if target_kind == "regular":
        target.write_bytes(b"synthetic")
    elif target_kind == "fifo":
        os.mkfifo(target)
    link = tmp_path / "link"
    link.symlink_to(target)

    def forbidden_fstat(*args):
        pytest.fail("a final-component symlink reached fstat")

    monkeypatch.setattr(frozen_state_io.os, "fstat", forbidden_fstat)
    with pytest.raises(OSError) as caught:
        read_bounded_state(link, 9, no_follow=True)
    assert caught.value.errno == errno.ELOOP
    assert caught.value.__cause__ is None
    assert len(observed_io.flags) == 1
    assert observed_io.flags[0] & os.O_NOFOLLOW
    assert observed_io.streams == []
    assert observed_io.requests == []


def test_no_follow_still_allows_operator_trusted_parent_symlink(tmp_path, observed_io):
    directory = tmp_path / "directory"
    directory.mkdir()
    (directory / "state").write_bytes(b"x")
    link = tmp_path / "parent-link"
    link.symlink_to(directory, target_is_directory=True)
    assert read_bounded_state(link / "state", 1, no_follow=True) == b"x"


@pytest.mark.parametrize("limit, payload", [(0, b"x"), (3, b"four")])
def test_initial_oversize_refuses_before_read_and_closes(tmp_path, observed_io, limit, payload):
    path = tmp_path / "private-state"
    path.write_bytes(payload)
    with pytest.raises(BoundedStateReadError) as caught:
        read_bounded_state(path, limit)
    _assert_refusal(caught.value, "SIZE_LIMIT", "frozen state exceeds byte limit")
    assert observed_io.requests == []
    assert len(observed_io.streams) == 1
    assert observed_io.streams[0].closed


def test_short_reads_continue_until_eof_with_remaining_byte_budget(tmp_path, observed_io):
    path = tmp_path / "state"
    path.write_bytes(b"0123456")
    observed_io.chunk_size = 2
    assert read_bounded_state(path, 7) == b"0123456"
    assert observed_io.requests == [8, 6, 4, 2, 1]
    assert observed_io.chunks == [b"01", b"23", b"45", b"6", b""]


@pytest.mark.parametrize("limit, growth, chunk_size, requests", [
    (0, b"x", None, [1]),
    (3, b"x", None, [4]),
    (3, b"more than one byte", None, [4]),
    (3, b"more than one byte", 2, [4, 2]),
])
def test_growth_after_fstat_reads_only_limit_plus_one_and_closes(
        tmp_path, monkeypatch, observed_io, limit, growth, chunk_size, requests):
    path = tmp_path / "private-state"
    payload = b"a" * limit
    path.write_bytes(payload)
    observed_io.chunk_size = chunk_size
    real_fstat = os.fstat

    def grow_after_fstat(fd):
        original = real_fstat(fd)
        with path.open("ab") as stream:
            stream.write(growth)
        return original

    monkeypatch.setattr(frozen_state_io.os, "fstat", grow_after_fstat)
    with pytest.raises(BoundedStateReadError) as caught:
        read_bounded_state(path, limit)
    _assert_refusal(caught.value, "SIZE_LIMIT", "frozen state exceeds byte limit")
    assert observed_io.requests == requests
    assert b"".join(observed_io.chunks) == (payload + growth)[:limit + 1]
    assert observed_io.streams[0].closed


@pytest.mark.parametrize("link", [False, True])
def test_fifo_is_opened_nonblocking_and_refused_without_read(tmp_path, observed_io, link):
    path = tmp_path / "fifo"
    os.mkfifo(path)
    if link:
        alias = tmp_path / "fifo-link"
        alias.symlink_to(path)
        path = alias
    with pytest.raises(BoundedStateReadError) as caught:
        read_bounded_state(path, 1)
    _assert_refusal(caught.value, "NOT_REGULAR_FILE",
                    "bounded frozen state must be a regular file")
    assert len(observed_io.flags) == 1
    assert observed_io.flags[0] & os.O_NONBLOCK
    assert observed_io.requests == []
    assert observed_io.streams[0].closed


@pytest.mark.parametrize("stage", ["open", "fstat", "read"])
def test_io_errors_propagate_unchanged_with_original_cause_and_close(
        tmp_path, monkeypatch, observed_io, stage):
    path = tmp_path / "private-state"
    path.write_bytes(b"x")
    original_cause = RuntimeError("synthetic original cause")
    original_error = OSError(errno.EIO, "synthetic I/O failure", str(path))
    original_error.__cause__ = original_cause

    def fail(*args, **kwargs):
        raise original_error

    if stage == "open":
        monkeypatch.setattr(frozen_state_io.os, "open", fail)
    elif stage == "fstat":
        monkeypatch.setattr(frozen_state_io.os, "fstat", fail)
    else:
        observed_io.read_error = original_error
    with pytest.raises(OSError) as caught:
        read_bounded_state(path, 1)
    assert caught.value is original_error
    assert caught.value.errno == errno.EIO
    assert caught.value.filename == str(path)
    assert caught.value.__cause__ is original_cause
    assert len(observed_io.streams) == (0 if stage == "open" else 1)
    assert observed_io.requests == ([2] if stage == "read" else [])
