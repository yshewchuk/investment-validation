"""Stage-boundary memory lines for the shared search driver; logging only, never changes results."""
import itertools
import resource
import sys
import time

from engine.v2.research.progress import decision
from experiments.memory_guard import rss_bytes

_NODE_CAP = 200_000
_T0 = None


def deep_size(root, cap=_NODE_CAP):
    """Bounded sys.getsizeof walk over dict keys and values and list/tuple/set/frozenset items.

    Walks at most `cap` objects (visited plus queued) and never materialises more than
    room+1 children of any one object. Returns (bytes, truncated); when truncated is True
    the size is a lower bound.
    """
    seen, stack, total, truncated = set(), [root], 0, False
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        total += sys.getsizeof(obj)
        if isinstance(obj, dict):
            it = itertools.chain(obj.keys(), obj.values())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            it = iter(obj)
        else:
            it = None
        room = cap - len(seen) - len(stack)
        if room <= 0:
            truncated = True
            continue
        if it is None:
            continue
        kids = list(itertools.islice(it, room + 1))
        if len(kids) > room:
            truncated = True
            kids = kids[:room]
        stack.extend(kids)
    return total, truncated


def log_memory(label, **holders):
    """Emit one stage-boundary memory line; any Exception is swallowed so logging cannot fail a run; MemoryError (the RSS watchdog's signal) propagates."""
    global _T0
    try:
        rss = rss_bytes()
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        now = time.monotonic()
        if _T0 is None or label == "start":
            _T0 = now
        parts = [f"{label}: VmRSS {rss/2**20:.0f} MiB, peak {peak/1024:.0f} MiB, {now - _T0:.0f}s since search start"]
        for name, holder in holders.items():
            if holder is None:
                continue
            size, truncated = deep_size(holder)
            parts.append(f", {name} {size/2**20:.1f}{'+' if truncated else ''} MiB")
        decision("search memory", "".join(parts))
    except MemoryError:
        raise
    except Exception:
        return


def panel_memo_holder():
    """Return the loaded size_search_binding _PANEL_MEMO without importing that module."""
    module = sys.modules.get("experiments.size_search_binding")
    return None if module is None else getattr(module, "_PANEL_MEMO", None)
