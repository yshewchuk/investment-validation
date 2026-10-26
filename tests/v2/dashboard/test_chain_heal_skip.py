"""The chain heal-skip: a mostly-held session stops being re-bought.

Weekend runs 10-18/19 spent ~114 calls healing sessions already 75%+ held —
the unheld names left are mostly ones with no chain on file (a transient
failure is possible and is not retried while the session stays skipped), so
the batches came back all-404. Past ``CHAIN_SKIP_HELD_FRACTION`` the session
is recorded in ``sessions_skipped`` and not fetched; a fresh (or unheld)
session still is.

Every dependency below the public signature — the fetcher, the store's held
set, the trading calendar — is a synthetic fake, so this needs no data root
and makes no provider call.
"""
from __future__ import annotations

import pandas as pd

from engine.dashboard import nightly

AS_OF = "2026-09-02"
TICKERS = [f"TKR{i:02d}" for i in range(10)]


class _Fetcher:
    """A ``Fetcher`` stand-in that answers every batch with the whole chunk."""

    def __init__(self):
        self.asked = []
        self.asked_tickers = []

    def fetch(self, source, endpoint, params, note=""):
        self.asked.append(params["tradeDate"])
        self.asked_tickers.extend(params["ticker"].split(","))

        class R:
            from_cache = False

            @staticmethod
            def json():
                return {"data": [{"ticker": t}
                                 for t in params["ticker"].split(",")]}

        return R()


class _Cal:
    """A one-session trading calendar; the real one reads the data root."""

    def __init__(self, stamps):
        self.days = pd.to_datetime(stamps)

    def index_of(self, date, *, side="exact"):
        return int(self.days.get_loc(pd.Timestamp(date)))


def _wire(monkeypatch, held):
    monkeypatch.setattr(nightly, "_unknown_symbols", lambda: set())
    monkeypatch.setattr("engine.calendar.trading_calendar",
                        lambda *a, **k: _Cal([AS_OF]))
    monkeypatch.setattr("engine.replay.available_chain_keys", lambda: held)


def test_a_mostly_held_session_is_not_fetched(monkeypatch, capsys):
    """8 of 10 held (80% >= 75%): the 2 unheld are mostly names with no chain
    on file (a transient failure is possible and is not retried) — skip,
    record, announce; the held count still lands in ``skipped_held``."""
    _wire(monkeypatch, {(t, pd.Timestamp(AS_OF)) for t in TICKERS[:8]})
    f = _Fetcher()

    out = nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=1)

    assert out["calls"] == 0
    assert f.asked == [], "a mostly-held session must not be fetched"
    assert out["sessions_skipped"] == [
        {"session": AS_OF, "held_fraction": 0.8, "unheld": 2}
    ]
    assert out["skipped_held"] == 8, "the held accounting is unchanged by the skip"
    err = capsys.readouterr().err
    assert ("[chains] session 2026-09-02: 80% held, skipping 2 unheld pairs "
            "(mostly no chain on file; a transient failure is possible and is "
            "not retried while the session stays skipped)") in err, \
        "the skip is never silent"


def test_a_zero_held_session_is_still_healed(monkeypatch):
    """A night that did not run leaves ~0% held — exactly what healing is for."""
    _wire(monkeypatch, set())
    f = _Fetcher()

    out = nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=1)

    assert out["calls"] == 1  # 10 tickers at CHAIN_BATCH=10 -> one call
    assert f.asked == [AS_OF]
    assert "sessions_skipped" not in out


def test_skip_held_false_fetches_regardless(monkeypatch):
    """The skip belongs to the heal pass only; an explicit full refresh is
    never skipped, even at 100% held."""
    _wire(monkeypatch, {(t, pd.Timestamp(AS_OF)) for t in TICKERS})
    f = _Fetcher()

    out = nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=1,
                                         skip_held=False)

    assert out["calls"] == 1
    assert f.asked == [AS_OF]
    assert "sessions_skipped" not in out
    assert out["skipped_held"] == 0


def test_exactly_75_pct_held_is_skipped(monkeypatch):
    """Boundary: ``>=`` means 3 of 4 held (exactly 75%) skips the fetch."""
    tickers = [f"TKR{i}" for i in range(4)]
    _wire(monkeypatch, {(t, pd.Timestamp(AS_OF)) for t in tickers[:3]})
    f = _Fetcher()

    out = nightly.refresh_forward_chains(tickers, AS_OF, fetcher=f, sessions=1)

    assert out["calls"] == 0
    assert f.asked == [], "exactly 75% held meets the >= bound and must skip"
    assert out["sessions_skipped"] == [
        {"session": AS_OF, "held_fraction": 0.75, "unheld": 1}
    ]
    assert out["skipped_held"] == 3


def test_74_pct_held_is_fetched_and_requests_only_the_unheld(monkeypatch):
    """Boundary: 74 of 100 (74% < 75%) is fetched, and the request names the
    26 unheld tickers and nothing already held."""
    tickers = [f"TKR{i:03d}" for i in range(100)]
    unheld = tickers[74:]
    _wire(monkeypatch, {(t, pd.Timestamp(AS_OF)) for t in tickers[:74]})
    f = _Fetcher()

    out = nightly.refresh_forward_chains(tickers, AS_OF, fetcher=f, sessions=1)

    assert out["calls"] == 3, "26 unheld at CHAIN_BATCH=10 -> ceil(26/10)"
    assert f.asked == [AS_OF, AS_OF, AS_OF]
    assert f.asked_tickers == unheld, "the request is exactly the unheld set"
    assert "sessions_skipped" not in out
