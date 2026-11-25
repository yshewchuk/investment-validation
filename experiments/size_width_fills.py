"""EXP-189 fill-model seam: the marks through which the width sweep's entries and exits are priced.

The default is today's marks (EXP-188's own functions, called, never copied). A later PR plugs another model in via
``size_width_pricing.load(fills=...)`` without touching the sweep. Rung-3 rows (value 0.0, defined-risk floor) and strike
resolution are rules of ``score``, not marks, and do not pass through the seam.
"""
import typing

from experiments import size_pnl_shortvol as sv


class Fills(typing.Protocol):
    def entry(self, priced): ...   # (entry cost per share, mean relative spread) or None; ``priced`` is an engine.structures.StructurePrice

    def exit_value(self, chunk, traded, alpha, mid=False): ...   # per-share exit value of the traded legs (dicts with idx into chunk, side, qty)


class CurrentFills:
    """Today's marks: size_pnl_shortvol.entry and size_pnl_shortvol._exit_value."""
    def entry(self, priced):
        return sv.entry(priced)

    def exit_value(self, chunk, traded, alpha, mid=False):
        return sv._exit_value(chunk, traded, alpha, mid=mid)
