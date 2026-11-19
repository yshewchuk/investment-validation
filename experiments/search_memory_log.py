"""Stage-boundary memory lines for the shared search driver; logging only, never changes results."""
import resource
import sys

from engine.v2.research.progress import decision
from experiments.memory_guard import rss_bytes

_NODE_CAP = 200_000


def deep_size(root, cap=_NODE_CAP):
    """Bounded sys.getsizeof walk over dict keys and values and list/tuple/set/frozenset items.

    Counts at most `cap` objects (visited plus queued); returns (bytes, truncated).
    When truncated is True the size is a lower bound.
    """
    seen, stack, total, visited = set(), [root], 0, 0
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        if visited >= cap:
            return total, True
        seen.add(id(obj))
        visited += 1
        total += sys.getsizeof(obj)
        if isinstance(obj, dict):
            stack.extend(obj.keys())
            stack.extend(obj.values())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            stack.extend(obj)
        if len(stack) + visited > cap:
            return total, True
    return total, False


def log_memory(label, **holders):
    """Emit one stage-boundary memory line; any failure is swallowed so logging cannot raise."""
    try:
        rss = rss_bytes()
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        parts = [f"{label}: VmRSS {rss/2**20:.0f} MiB, peak {peak/1024:.0f} MiB"]
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
