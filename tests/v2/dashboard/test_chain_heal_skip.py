"""The chain heal-skip: a mostly-held session stops re-buying the unheld tail.

Weekend runs 10-18/19 spent ~114 calls healing sessions already 75%+ held —
the unheld names left are mostly ones with no chain on file, and the batches
came back all-404. Past ``CHAIN_SKIP_HELD_FRACTION`` the session is recorded
in ``sessions_skipped`` and its tail is not bought; the names that failed
transiently or were cut off by quota are kept alive in ``CHAIN_RETRY_PATH``
and ARE re-asked even in a skipped session. Only definitive 404s and
never-attempted-and-failed names are dropped. A fresh (or unheld) session
still is healed in full.

Every dependency below the public signature — the fetcher, the store's held
set, the trading calendar, the retry file — is a synthetic fake or a tmp_path
patch, so this needs no data root and makes no provider call.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from engine.dashboard import nightly
from engine.data.throttle import QuotaExhausted

AS_OF = "2026-09-02"
TICKERS = [f"TKR{i:02d}" for i in range(10)]


class _Fetcher:
    """A ``Fetcher`` stand-in that answers every batch with the whole chunk.

    ``fail(params)`` may return an exception to raise instead of a record.
    """

    def __init__(self, fail=None):
        self.asked = []
        self.asked_tickers = []
        self.fail = fail

    def fetch(self, source, endpoint, params, note=""):
        self.asked.append(params["tradeDate"])
        self.asked_tickers.extend(params["ticker"].split(","))
        exc = self.fail(params) if self.fail else None
        if exc is not None:
            raise exc

        class R:
            from_cache = False

            @staticmethod
            def json():
                return {"data": [{"ticker": t}
                                 for t in params["ticker"].split(",")]}

        return R()


class _Cal:
    """A small trading calendar; the real one reads the data root."""

    def __init__(self, stamps):
        self.days = pd.to_datetime(stamps)

    def index_of(self, date, *, side="exact"):
        return int(self.days.get_loc(pd.Timestamp(date)))


def _wire(monkeypatch, held, tmp_path, stamps=(AS_OF,)):
    monkeypatch.setattr(nightly, "_unknown_symbols", lambda: set())
    monkeypatch.setattr(nightly, "CHAIN_RETRY_PATH", tmp_path / "orats_chain_retry.json")
    monkeypatch.setattr("engine.calendar.trading_calendar", lambda *a, **k: _Cal(stamps))
    monkeypatch.setattr("engine.replay.available_chain_keys", lambda: held)


def test_a_mostly_held_session_is_not_fetched(monkeypatch, capsys, tmp_path):
    """8 of 10 held (80% >= 75%) with nothing in the retry file: the 2 unheld
    are mostly names with no chain on file — skip, record, announce; the held
    count still lands in ``skipped_held``."""
    _wire(monkeypatch, {(t, pd.Timestamp(AS_OF)) for t in TICKERS[:8]}, tmp_path)
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
            "(mostly no chain on file; transient failures and quota-cut names "
            "are retried via the chain retry file)") in err, \
        "the skip is never silent"


def test_a_skipped_session_still_retries_the_failed_names(monkeypatch, tmp_path):
    """The skip trims to the retry file, it does not mute it: a session 8 of
    10 held with one unheld name in CHAIN_RETRY_PATH fetches exactly that
    name and none of the never-retried tail."""
    _wire(monkeypatch, {(t, pd.Timestamp(AS_OF)) for t in TICKERS[:8]}, tmp_path)
    nightly.CHAIN_RETRY_PATH.write_text(json.dumps({AS_OF: ["TKR09"]}))
    f = _Fetcher()

    out = nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=1)

    assert f.asked_tickers == ["TKR09"], "only the retry-file name is re-asked"
    assert out["sessions_skipped"] == [
        {"session": AS_OF, "held_fraction": 0.8, "unheld": 2, "retried": 1}
    ]
    assert nightly._read_chain_retry() == {}, "the OK fetch empties the entry"


def test_a_zero_held_session_is_still_healed(monkeypatch, tmp_path):
    """A night that did not run leaves ~0% held — exactly what healing is for."""
    _wire(monkeypatch, set(), tmp_path)
    f = _Fetcher()

    out = nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=1)

    assert out["calls"] == 1  # 10 tickers at CHAIN_BATCH=10 -> one call
    assert f.asked == [AS_OF]
    assert f.asked_tickers == sorted(TICKERS), "the heal asks the whole unheld set"
    assert "sessions_skipped" not in out


def test_transient_failure_is_retried_on_a_later_skipped_run(monkeypatch, tmp_path):
    """A non-404 exception puts the chunk in the retry file, and the next
    run — by then past the skip threshold — re-asks exactly those names."""
    _wire(monkeypatch, set(), tmp_path)
    f1 = _Fetcher(fail=lambda p: RuntimeError("orats hist/strikes: connection reset"))

    out = nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f1, sessions=1)

    assert out["failed"], "the failure is still reported"
    assert out["calls"] == 0
    assert nightly._read_chain_retry() == {AS_OF: set(TICKERS)}

    monkeypatch.setattr("engine.replay.available_chain_keys",
                        lambda: {(t, pd.Timestamp(AS_OF)) for t in TICKERS[:8]})
    f2 = _Fetcher()
    out2 = nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f2, sessions=1)

    assert f2.asked_tickers == ["TKR08", "TKR09"], "the skipped session re-asks the retry set"
    assert out2["sessions_skipped"][0]["retried"] == 2
    assert nightly._read_chain_retry() == {}, "the re-ask cleared it"


def test_a_404_exception_is_final_and_never_retried(monkeypatch, tmp_path):
    """"HTTP 404" is a definitive no-chain: recorded as missing (like the
    200-but-short path) and never entered in the retry file."""
    _wire(monkeypatch, set(), tmp_path)
    f = _Fetcher(fail=lambda p: RuntimeError("orats hist/strikes returned HTTP 404"))

    out = nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=1)

    assert out["missing"] == sorted(TICKERS)
    assert out["failed"], "the existing failed reporting is unchanged by the 404 split"
    assert nightly._read_chain_retry() == {}, "a 404 has no chain; retrying cannot help"


def test_quota_cut_persists_the_unasked_remainder_then_reraises(monkeypatch, tmp_path):
    """QuotaExhausted mid-loop must not swallow the tail: the cut chunk and
    every later unasked chunk land in the retry file before the raise."""
    tickers = [f"TKR{i:02d}" for i in range(25)]

    def fail(params):
        if params["ticker"].split(",")[0] == "TKR10":
            return QuotaExhausted("orats daily quota below the reserve floor")
        return None

    _wire(monkeypatch, set(), tmp_path)
    f = _Fetcher(fail=fail)

    with pytest.raises(QuotaExhausted):
        nightly.refresh_forward_chains(tickers, AS_OF, fetcher=f, sessions=1)

    assert f.asked_tickers[:10] == sorted(tickers)[:10]
    assert nightly._read_chain_retry() == {AS_OF: {t for t in tickers if t >= "TKR10"}}, \
        "the cut chunk plus all later unasked names are kept for retry"


def test_old_entries_are_pruned_past_the_window(monkeypatch, tmp_path):
    """A retry entry more than 14 days behind the fetched session is dropped
    on the next persist — that date is not coming back."""
    _wire(monkeypatch, set(), tmp_path)
    stale = str((pd.Timestamp(AS_OF) - pd.Timedelta(days=15)).date())
    fresh = str((pd.Timestamp(AS_OF) - pd.Timedelta(days=3)).date())
    nightly.CHAIN_RETRY_PATH.write_text(
        json.dumps({stale: ["OLD1"], fresh: ["OLD2"], AS_OF: ["TKR00"]}))
    f = _Fetcher(fail=lambda p: RuntimeError("HTTP 502"))

    nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=1)

    kept = nightly._read_chain_retry()
    assert stale not in kept, "the 15-day-old entry is pruned"
    assert kept[fresh] == {"OLD2"} and kept[AS_OF] == set(TICKERS)


def test_skip_held_false_fetches_regardless(monkeypatch, tmp_path):
    """The skip belongs to the heal pass only; an explicit full refresh is
    never skipped, even at 100% held."""
    _wire(monkeypatch, {(t, pd.Timestamp(AS_OF)) for t in TICKERS}, tmp_path)
    f = _Fetcher()

    out = nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=1,
                                         skip_held=False)

    assert out["calls"] == 1
    assert f.asked == [AS_OF]
    assert f.asked_tickers == sorted(TICKERS)
    assert "sessions_skipped" not in out
    assert out["skipped_held"] == 0


def test_exactly_75_pct_held_is_skipped(monkeypatch, tmp_path):
    """Boundary: ``>=`` means 3 of 4 held (exactly 75%) skips the fetch."""
    tickers = [f"TKR{i}" for i in range(4)]
    _wire(monkeypatch, {(t, pd.Timestamp(AS_OF)) for t in tickers[:3]}, tmp_path)
    f = _Fetcher()

    out = nightly.refresh_forward_chains(tickers, AS_OF, fetcher=f, sessions=1)

    assert out["calls"] == 0
    assert f.asked == [], "exactly 75% held meets the >= bound and must skip"
    assert out["sessions_skipped"] == [
        {"session": AS_OF, "held_fraction": 0.75, "unheld": 1}
    ]
    assert out["skipped_held"] == 3


def test_74_pct_held_is_fetched_and_requests_only_the_unheld(monkeypatch, tmp_path):
    """Boundary: 74 of 100 (74% < 75%) is fetched, and the request names the
    26 unheld tickers and nothing already held."""
    tickers = [f"TKR{i:03d}" for i in range(100)]
    unheld = tickers[74:]
    _wire(monkeypatch, {(t, pd.Timestamp(AS_OF)) for t in tickers[:74]}, tmp_path)
    f = _Fetcher()

    out = nightly.refresh_forward_chains(tickers, AS_OF, fetcher=f, sessions=1)

    assert out["calls"] == 3, "26 unheld at CHAIN_BATCH=10 -> ceil(26/10)"
    assert f.asked == [AS_OF, AS_OF, AS_OF]
    assert f.asked_tickers == unheld, "the request is exactly the unheld set"
    assert "sessions_skipped" not in out


def test_two_sessions_newest_healed_older_skipped(monkeypatch, tmp_path):
    """Through the public entry point: the newest session at 0% held is fetched
    in full, the older one at 80% held is skipped, and the two are recorded
    independently."""
    older = "2026-09-01"
    held = {(t, pd.Timestamp(older)) for t in TICKERS[:8]}
    _wire(monkeypatch, held, tmp_path, stamps=[older, AS_OF])
    f = _Fetcher()

    out = nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=2)

    assert out["sessions"] == [AS_OF, older]
    assert f.asked == [AS_OF], "only the newest session is fetched"
    assert f.asked_tickers == sorted(TICKERS)
    assert out["sessions_skipped"] == [
        {"session": older, "held_fraction": 0.8, "unheld": 2}
    ]
    assert out["skipped_held"] == 8
