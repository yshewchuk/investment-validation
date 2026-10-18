"""Opt-in memory diagnostics for the legacy nightly (Refs #866).

Off by default: when the enable flag is unset :func:`snapshot` and :func:`tick`
return immediately — no output, no gc walk, no import of heavy modules, no
behaviour change. The exported helpers (:func:`estimate_size`,
:func:`top_objects`, :func:`rss_bytes`, :func:`watch`) do their work when
called directly and do not check the flag. Stdlib only; pandas/numpy are
detected by duck typing and type names at call time, never imported at module
top.

Flags (read at call time; truthy is ``1``/``true``/``yes``/``on``):

- ``INVESTING_PLAN_MEMPROFILE``: enable snapshots.
- ``INVESTING_PLAN_MEMPROFILE_TRIM``: under enable, run ``gc.collect()`` then
  a best-effort libc ``malloc_trim(0)`` before measuring, and log RSS before
  and after the trim so retained-by-Python separates from freed-but-held-by-glibc.
- ``INVESTING_PLAN_MEMPROFILE_EVERY``: int, default 250; used by :func:`tick`.
- ``INVESTING_PLAN_MEMPROFILE_TOP``: int, default 12; rows of largest objects
  (negative values are clamped to 0 rows).

Output lines are prefixed with two spaces then ``[mem]`` and go to stdout,
flushed. Size estimates are sampled, not exact.

:func:`snapshot` and :func:`tick` never raise, except MemoryError which
propagates; when disabled they return before any diagnostic work. The exported
helpers do their work when called directly and do not check the flag.
"""
from __future__ import annotations

import ctypes
import gc
import itertools
import os
import sys
from typing import Any

__all__ = [
    "enabled",
    "trim_enabled",
    "rss_bytes",
    "peak_rss_bytes",
    "watch",
    "DEFAULT_WATCHES",
    "estimate_size",
    "top_objects",
    "snapshot",
    "tick",
]

_TRUTHY = {"1", "true", "yes", "on"}
_DEFAULT_EVERY = 250
_DEFAULT_TOP = 12
_PREFIX = "  [mem] "

#: (name, owner-id | None) -> (weakref.ref(owner) | module-name string, attr)
_registry: dict[tuple[str, int | None], tuple[Any, str]] = {}
_defaults_registered = False

DEFAULT_WATCHES: tuple[tuple[str, str, str], ...] = (
    ("replay.available_chain_keys", "engine.replay", "_AVAILABLE_KEYS"),
    ("replay.chain_dates_by_ticker", "engine.replay", "_CHAIN_DATES_BY_TICKER"),
)


def enabled() -> bool:
    return _flag("INVESTING_PLAN_MEMPROFILE")


def trim_enabled() -> bool:
    return enabled() and _flag("INVESTING_PLAN_MEMPROFILE_TRIM")


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def _int_flag(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except MemoryError:
        raise
    except (TypeError, ValueError):
        return default


def _malloc_trim() -> bool:
    """Best-effort libc ``malloc_trim(0)``; swallow OSError/AttributeError."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        return True
    except MemoryError:
        raise
    except (OSError, AttributeError, TypeError):
        return False


def _proc_field(field: str) -> int:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith(field):
                    parts = line.split()
                    return int(parts[1]) * 1024
    except MemoryError:
        raise
    except OSError:
        pass
    return 0


def rss_bytes() -> int:
    """Current RSS bytes; does not check the enable flag."""
    return _proc_field("VmRSS:")


def peak_rss_bytes() -> int:
    """Peak (high-water) RSS bytes; does not check the enable flag."""
    return _proc_field("VmHWM:")


def _fmt(n: int) -> str:
    try:
        return f"{int(n) / (1024 ** 3):.2f}G"
    except MemoryError:
        raise
    except (TypeError, ValueError):
        return "0.00G"


def _fmt_mb(n: int) -> str:
    try:
        return f"{int(n) / (1024 ** 2):.1f}M"
    except MemoryError:
        raise
    except (TypeError, ValueError):
        return "0.0M"


def watch(name: str, owner: Any, attr: str) -> None:
    """Register a named cache. ``owner`` is an object (stored as a weakref; if
    not weakref-able, store nothing) or a module-name string resolved through
    ``sys.modules`` at sample time (never imported). Keyed by
    ``(name, id(owner))`` for objects so multiple live Scorers each keep their
    own watch; module-string watches key by ``(name, None)``. Registering the
    same ``(name, owner)`` twice replaces. Dead weakref entries are dropped
    first so stale owners do not accumulate. Cheap and side-effect-free; does
    not check the enable flag; never keeps the owner alive."""
    import weakref

    for key, (target, _a) in list(_registry.items()):
        if isinstance(target, weakref.ref) and target() is None:
            del _registry[key]
    if isinstance(owner, str):
        _registry[(name, None)] = (owner, attr)
        return
    try:
        _registry[(name, id(owner))] = (weakref.ref(owner), attr)
    except MemoryError:
        raise
    except TypeError:
        pass


def _ensure_defaults() -> None:
    global _defaults_registered
    if _defaults_registered:
        return
    for name, mod, attr in DEFAULT_WATCHES:
        _registry.setdefault((name, None), (mod, attr))
    _defaults_registered = True


def _item_size(item: Any, sample: int) -> int:
    """Never materialise ``item``; extrapolate nested containers to full
    length from a first-N sample of members for sets/dicts (allocation-free,
    biased) or a strided sample for lists/tuples/Series (measured one level
    only)."""
    if type(item).__name__ in {"DataFrame", "Series"}:
        return estimate_size(item)
    try:
        base = sys.getsizeof(item)
    except MemoryError:
        raise
    except Exception:
        return 0
    try:
        if isinstance(item, (tuple, list)):
            if not item:
                return base
            members = list(itertools.islice(item, 8))
            mean = sum(_one_level(m) for m in members) / len(members)
            return int(base + len(item) * mean)
        if isinstance(item, dict):
            if not item:
                return base
            pairs = list(itertools.islice(item.items(), 8))
            mean = (
                sum(_one_level(k) + _one_level(v) for k, v in pairs) / len(pairs)
            )
            return int(base + len(item) * mean)
        if isinstance(item, (set, frozenset)):
            if not item:
                return base
            members = list(itertools.islice(item, 8))
            mean = sum(_one_level(m) for m in members) / len(members)
            return int(base + len(item) * mean)
    except MemoryError:
        raise
    except Exception:
        return base
    return base


def _one_level(m: Any) -> int:
    if type(m).__name__ in {"DataFrame", "Series"}:
        return estimate_size(m)
    try:
        return sys.getsizeof(m)
    except MemoryError:
        raise
    except Exception:
        return 0


def _sample_indices(n: int, sample: int) -> list[int]:
    if n <= sample:
        return list(range(n))
    step = n / sample
    return sorted({min(n - 1, int(i * step)) for i in range(sample)})


def _mean_item_size(items: list[Any], sample: int) -> int:
    """Mean ``_item_size`` over already-sampled items (no re-sampling)."""
    if not items:
        return 0
    total = 0
    count = 0
    for item in items:
        try:
            total += _item_size(item, sample)
            count += 1
        except MemoryError:
            raise
        except Exception:
            continue
    return total // count if count else 0


def estimate_size(obj: Any, *, sample: int = 200) -> int:
    """Approximate deep bytes without walking everything. Sampled, not exact:
    first-N sample for sets/dicts (allocation-free, biased), strided for
    lists/tuples/Series. Never raises, except MemoryError which propagates;
    returns 0 on any other failure; does not check the enable flag."""
    try:
        if type(obj).__name__ == "Series":
            total = int(obj.memory_usage(index=True, deep=False))
            if obj.dtype == object:
                n = len(obj)
                if n:
                    step = max(1, n // sample)
                    sample_vals = list(obj.iloc[::step])[:sample]
                    mean = (
                        sum(_one_level(v) for v in sample_vals) / len(sample_vals)
                        if sample_vals
                        else 0
                    )
                    total += int(mean * n)
            return int(total)
        if hasattr(obj, "nbytes") and hasattr(obj, "shape"):
            return int(obj.nbytes)
        if hasattr(obj, "memory_usage") and hasattr(obj, "columns"):
            total = int(obj.memory_usage(index=True, deep=False).sum())
            for col in obj.columns:
                try:
                    series = obj[col]
                    if series.dtype == object:
                        n = len(series)
                        if n:
                            step = max(1, n // sample)
                            sample_vals = list(series.iloc[::step])[:sample]
                            mean = (
                                sum(_one_level(v) for v in sample_vals)
                                / len(sample_vals)
                                if sample_vals
                                else 0
                            )
                            total += int(mean * n)
                except MemoryError:
                    raise
                except Exception:
                    continue
            return int(total)
        if isinstance(obj, dict):
            # first-N sample of items: biased but allocation-free
            base = sys.getsizeof(obj)
            pairs = list(itertools.islice(obj.items(), sample))
            mean = _mean_item_size([k for k, _ in pairs], sample) + _mean_item_size(
                [v for _, v in pairs], sample
            )
            return int(base + len(obj) * mean)
        if isinstance(obj, (list, tuple)):
            base = sys.getsizeof(obj)
            items = [obj[i] for i in _sample_indices(len(obj), sample)]
            mean = _mean_item_size(items, sample)
            return int(base + len(obj) * mean)
        if isinstance(obj, (set, frozenset)):
            # first-N sample of entries: biased but allocation-free
            base = sys.getsizeof(obj)
            items = list(itertools.islice(obj, sample))
            mean = _mean_item_size(items, sample)
            return int(base + len(obj) * mean)
        return int(sys.getsizeof(obj))
    except MemoryError:
        raise
    except Exception:
        return 0


def top_objects(n: int) -> list[tuple[int, str]]:
    """Walk ``gc.get_objects()``, keep DataFrame/Series with >= 1000 rows,
    ndarrays with nbytes >= 1_000_000, and dict/list/set with len >= 1000,
    estimate each, and return the n largest as ``(bytes, description)``.
    Standalone numpy arrays are NOT reported because ``gc.get_objects()``
    only returns gc-tracked objects; arrays are seen only through
    DataFrame/Series sizes or containers holding them. Deduplicates by id;
    ``n`` is coerced to a non-negative int at entry (non-numeric input → 0
    rows); never raises except MemoryError; does not check the enable flag."""
    try:
        n = max(0, int(n))
    except (TypeError, ValueError):
        n = 0
    seen: set[int] = set()
    candidates: list[tuple[int, str]] = []
    try:
        objs = gc.get_objects()
    except MemoryError:
        raise
    except Exception:
        return []
    for o in objs:
        try:
            if id(o) in seen:
                continue
            name = type(o).__name__
            is_df = name in {"DataFrame", "Series"}
            is_arr = name == "ndarray"
            if not (is_df or is_arr or isinstance(o, (dict, list, set))):
                continue
            if is_arr:
                if int(getattr(o, "nbytes", 0) or 0) < 1_000_000:
                    continue
            else:
                try:
                    if len(o) < 1000:
                        continue
                except MemoryError:
                    raise
                except Exception:
                    continue
            seen.add(id(o))
            nbytes = estimate_size(o)
            desc = _describe(o, name, is_df)
            candidates.append((nbytes, desc))
        except MemoryError:
            raise
        except Exception:
            continue
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[:n]


def _describe(o: Any, name: str, is_df: bool) -> str:
    if is_df and name == "DataFrame":
        try:
            shape = f"{o.shape[0]} x {o.shape[1]}"
            cols = ", ".join(str(c) for c in list(o.columns)[:3])
            return f"DataFrame len={shape} [{cols}]"
        except MemoryError:
            raise
        except Exception:
            return f"DataFrame len={len(o)}"
    if name == "Series":
        try:
            return f"Series len={len(o)}"
        except MemoryError:
            raise
        except Exception:
            return "Series"
    if name == "ndarray":
        try:
            return f"ndarray len={len(o)}"
        except MemoryError:
            raise
        except Exception:
            return "ndarray"
    if isinstance(o, dict):
        try:
            if o:
                k = repr(next(iter(o.keys())))[:60]
                return f"dict len={len(o)} key={k}"
        except MemoryError:
            raise
        except Exception:
            pass
        return f"dict len={len(o)}"
    try:
        return f"{name} len={len(o)}"
    except MemoryError:
        raise
    except Exception:
        return name


def snapshot(label: str, *, top: int | None = None) -> None:
    """Print one diagnostic block, or return immediately if not enabled —
    the disabled path returns before any diagnostic work. Never raises,
    except MemoryError which propagates."""
    if not enabled():
        return
    try:
        _ensure_defaults()
        for (key_name, owner_id), entry in list(_registry.items()):
            target, attr = entry
            if isinstance(target, str):
                continue
            try:
                if target() is None:
                    del _registry[(key_name, owner_id)]
            except MemoryError:
                raise
            except TypeError:
                del _registry[(key_name, owner_id)]
        seen_names: dict[str, int] = {}
        n = top if top is not None else _int_flag("INVESTING_PLAN_MEMPROFILE_TOP", _DEFAULT_TOP)
        n = max(0, n)
        pre = rss_bytes()
        trimmed = False
        if trim_enabled():
            gc.collect()
            trimmed = bool(_malloc_trim())
        post = rss_bytes()
        peak = peak_rss_bytes()
        head = f"{_PREFIX}{label} rss={_fmt(post)} peak={_fmt(peak)}"
        if trimmed:
            head += f" [trimmed from {_fmt(pre)}]"
        print(head, flush=True)
        for (key_name, owner_id), (target, attr) in list(_registry.items()):
            if isinstance(target, str):
                mod = sys.modules.get(target)
                val = getattr(mod, attr, None) if mod is not None else None
                if val is None:
                    continue
                display = key_name
            else:
                owner = target()
                if owner is None:
                    continue
                try:
                    val = getattr(owner, attr, None)
                except MemoryError:
                    raise
                except Exception:
                    continue
                if val is None:
                    continue
                k = seen_names.get(key_name, 0) + 1
                seen_names[key_name] = k
                display = key_name if k == 1 else f"{key_name}#{k}"
            try:
                length: Any = len(val)
            except MemoryError:
                raise
            except Exception:
                length = "-"
            try:
                size = estimate_size(val)
            except MemoryError:
                raise
            except Exception:
                size = 0
            size_str = _fmt_mb(size) if size else "0.0M"
            print(f"{_PREFIX}  cache {display}: {length} entries ~{size_str}", flush=True)
        for nbytes, desc in top_objects(n):
            print(f"{_PREFIX}  obj {_fmt_mb(nbytes)} {desc}", flush=True)
    except MemoryError:
        raise
    except Exception as exc:
        try:
            print(f"{_PREFIX}{label} probe failed: {type(exc).__name__}: {exc!r}", flush=True)
        except OSError:
            pass


def tick(label: str, i: int) -> None:
    """Call :func:`snapshot` every ``INVESTING_PLAN_MEMPROFILE_EVERY`` steps;
    returns immediately if not enabled."""
    if not enabled():
        return
    try:
        every = _int_flag("INVESTING_PLAN_MEMPROFILE_EVERY", _DEFAULT_EVERY)
        if every > 0 and i > 0 and i % every == 0:
            snapshot(f"{label} {i:,}")
    except MemoryError:
        raise
    except Exception as exc:
        try:
            print(f"{_PREFIX}{label} probe failed: {type(exc).__name__}: {exc!r}", flush=True)
        except OSError:
            pass