"""EXP-188 pricing seam: private copies of size_pnl_shortvol and
size_pnl_straddle that resolve strikes through
engine.chain_index.price_structure_indexed (the chain-index twin of
engine.structures.price_structure).

Each copy is a fresh re-execution of its source, not put into
sys.modules, so the originals stay on price_structure (EXP-189's test
asserts that). Each copy caches one chain snapshot per (event, kwargs)
of the current chunk: the cache is bounded to one chunk per copy.
"""
import importlib.util

from engine.chain_index import price_structure_indexed
from experiments import size_pnl_shortvol as sv
from experiments import size_pnl_straddle as sd


def _cached(build):
    """``build(chunk, i, **kw)``, one snapshot per (i, kwargs) of the
    CURRENT chunk. A new chunk object drops the cache and is held in its
    place, so its id cannot be reused."""
    held, snaps = None, {}

    def cached(chunk, i, **kw):
        nonlocal held, snaps
        if chunk is not held:
            held, snaps = chunk, {}
        key = (i, tuple(sorted(kw.items())))
        if key not in snaps:
            snaps[key] = build(chunk, i, **kw)
        return snaps[key]
    return cached


def _copy(source):
    """A private re-execution of ``source``; the real module is untouched."""
    spec = importlib.util.find_spec(source.__name__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_shortvol():
    module = _copy(sv)
    module.Refusal = sv.Refusal  # callers catch sv.Refusal; the copy raises the same class
    module.price_structure = price_structure_indexed
    module.chain_snapshot = _cached(module.chain_snapshot)
    return module


def load_straddle():
    module = _copy(sd)
    module.price_structure = price_structure_indexed
    module.chain_snapshot = _cached(module.chain_snapshot)
    return module


SHORTVOL = load_shortvol()
STRADDLE = load_straddle()
