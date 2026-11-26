"""EXP-189 fill-model seam: the marks through which the width sweep's entries and exits are priced.

The default is today's marks (EXP-188's own functions, called, never copied). A later PR plugs another model in via
``size_width_pricing.load(fills=...)`` without touching the sweep. Rung-3 rows (value 0.0, defined-risk floor) still
bypass the seam: their exit flags are all False. Strike resolution (``size_pnl_shortvol.resolve_priced``, called by both ``score_events`` and ``score``) is not a mark and does not pass through the seam.
"""
import typing

from experiments import size_pnl_shortvol as sv


class Trade(typing.NamedTuple):
    chunk: typing.Any   # the ladder chunk
    i: int              # event index into the chunk
    event_id: str
    structure: str
    legs: tuple         # every resolved leg dict, reference legs of qty 0 included


class Skip(typing.NamedTuple):
    reason: str         # a counted non-trade with its own reason


class Exit(typing.NamedTuple):
    value: float
    flags: frozenset = frozenset()
    unquoted: bool = False   # the whole structure is at the defined-risk floor: value must be 0.0


class Fills(typing.Protocol):
    exit_flags: tuple   # names of the boolean flag columns this model adds to score rows

    def entry(self, trade, priced): ...   # (entry cost per share, mean relative spread), None, or Skip

    def exit_value(self, trade, traded, alpha, mid=False): ...   # float or Exit; traded legs as in size_pnl_shortvol


class CurrentFills:
    """Today's marks: size_pnl_shortvol.entry and size_pnl_shortvol._exit_value."""
    exit_flags = ()

    def entry(self, trade, priced):
        return sv.entry(priced)

    def exit_value(self, trade, traded, alpha, mid=False):
        return sv._exit_value(trade.chunk, traded, alpha, mid=mid)
