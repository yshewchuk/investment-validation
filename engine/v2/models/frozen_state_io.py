"""Shared bounded regular-file admission for frozen state and offline publication."""
import os
import stat
import sys
from pathlib import Path


class BoundedStateReadError(ValueError):
    """A bounded read was refused; every message is fixed and path-free."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def valid_byte_limit(limit: int | None) -> bool:
    return limit is None or (type(limit) is int and 0 < limit < sys.maxsize)


def read_bounded_state(path: Path, limit: int, *, no_follow: bool = False) -> bytes:
    """Read at most limit + 1 bytes; optionally refuse final-component symlinks."""
    if type(limit) is not int or not 0 <= limit < sys.maxsize:
        raise BoundedStateReadError("INVALID_LIMIT", "byte limit must be a nonnegative integer below sys.maxsize")
    # Nonblocking open lets fstat refuse a FIFO before any content read.
    flags = os.O_NONBLOCK | (os.O_NOFOLLOW if no_follow else 0)
    with open(path, "rb", buffering=0,
              opener=lambda name, mode: os.open(name, mode | flags)) as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise BoundedStateReadError("NOT_REGULAR_FILE", "bounded frozen state must be a regular file")
        if info.st_size > limit:
            raise BoundedStateReadError("SIZE_LIMIT", "frozen state exceeds byte limit")
        raw = bytearray()
        while len(raw) <= limit:
            chunk = stream.read(limit + 1 - len(raw))
            if not chunk:
                break
            raw.extend(chunk)
    if len(raw) > limit:
        raise BoundedStateReadError("SIZE_LIMIT", "frozen state exceeds byte limit")
    return bytes(raw)
