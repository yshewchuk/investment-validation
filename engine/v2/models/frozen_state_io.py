"""Bounded regular-file read used by the frozen-state loader's opt-in path."""
import os
import stat
import sys
from pathlib import Path


class BoundedStateReadError(ValueError):
    """A bounded read was refused; every message is fixed and path-free."""


def valid_byte_limit(limit: int | None) -> bool:
    return limit is None or (type(limit) is int and 0 < limit < sys.maxsize)


def read_bounded_state(path: Path, limit: int) -> bytes:
    # Nonblocking open lets fstat refuse a FIFO before any content read.
    with open(path, "rb", buffering=0,
              opener=lambda name, flags: os.open(name, flags | os.O_NONBLOCK)) as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise BoundedStateReadError("bounded frozen state must be a regular file")
        if info.st_size > limit:
            raise BoundedStateReadError("frozen state exceeds byte limit")
        raw = bytearray()
        while len(raw) <= limit:
            chunk = stream.read(limit + 1 - len(raw))
            if not chunk:
                break
            raw.extend(chunk)
    if len(raw) > limit:
        raise BoundedStateReadError("frozen state exceeds byte limit")
    return bytes(raw)
