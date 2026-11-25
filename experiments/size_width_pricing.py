"""EXP-189 pricing seam: a private copy of size_pnl_shortvol whose strike resolution runs through engine.chain_index.price_structure_indexed.

``load()`` re-executes the sv source into a fresh module (the real module is untouched); ``chain_snapshot`` caches one snapshot per event of the current chunk."""
import importlib.util

from engine.chain_index import price_structure_indexed
from experiments import size_pnl_shortvol as sv


def _cached(build):
    """``build(chunk, i)`` with one snapshot per event of the CURRENT chunk: a new chunk object drops the cache and is held in its place."""
    held, snaps = None, {}

    def cached(chunk, i):
        nonlocal held, snaps
        if chunk is not held:
            held, snaps = chunk, {}
        if i not in snaps:
            snaps[i] = build(chunk, i)
        return snaps[i]
    return cached


def load():
    """A private instance of size_pnl_shortvol on the chain-index pricer; the real module is untouched."""
    spec = importlib.util.find_spec(sv.__name__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.Refusal = sv.Refusal  # callers catch sv.Refusal; the copy raises the same class
    module.price_structure = price_structure_indexed
    module.chain_snapshot = _cached(module.chain_snapshot)
    return module


PRICING = load()
