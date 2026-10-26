"""The chain heal-skip: a mostly-held session stops re-buying the unheld tail.

Weekend runs 10-18/19 spent ~114 calls healing sessions already 75%+ held —
the unheld names left are mostly ones with no chain on file, and the batches
came back all-404. Past ``CHAIN_SKIP_HELD_FRACTION`` the session is recorded
in ``sessions_skipped`` and its tail is not bought; the names that failed
transiently or were cut off by quota are kept alive in ``CHAIN_RETRY_PATH``
and ARE re-asked even in a skipped session. Only definitive 404s are
dropped. A fresh (or unheld) session still is healed in full.

Every dependency below the public signature — the fetcher, the store's held
set, the trading calendar, the retry file — is a synthetic fake or a tmp_path
patch, so this needs no data root and makes no provider call.
"""
from __future__ import annotations

import json
import os
import threading

import pandas as pd
import pytest

from engine.dashboard import nightly
from engine.data.fetch import CredentialRotated, FetchError
from engine.data.throttle import QuotaExhausted

AS_OF = "2026-09-02"
TICKERS = [f"TKR{i:02d}" for i in range(10)]


class _Fetcher:
    """A ``Fetcher`` stand-in that answers every batch with the whole chunk.

    ``fail(params)`` may return an exception to raise instead of a record.
    ``bad_json(params)`` may return an exception for ``record.json()`` to
    raise on an otherwise-successful (HTTP 200) call.
    """

    def __init__(self, fail=None, bad_json=None):
        self.asked = []
        self.asked_tickers = []
        self.fail = fail
        self.bad_json = bad_json

    def fetch(self, source, endpoint, params, note=""):
        self.asked.append(params["tradeDate"])
        self.asked_tickers.extend(params["ticker"].split(","))
        exc = self.fail(params) if self.fail else None
        if exc is not None:
            raise exc
        bad = self.bad_json(params) if self.bad_json else None

        class R:
            from_cache = False

            @staticmethod
            def json():
                if bad is not None:
                    raise bad
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


def test_a_retry_name_that_is_now_held_is_not_re_asked(monkeypatch, tmp_path):
    """The skip re-asks the intersection of the memo and the UNHELD tail: a
    memo name that got held since is not bought again."""
    _wire(monkeypatch, {(t, pd.Timestamp(AS_OF)) for t in TICKERS[:8]}, tmp_path)
    nightly.CHAIN_RETRY_PATH.write_text(json.dumps({AS_OF: ["TKR00", "TKR09"]}))
    f = _Fetcher()

    out = nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=1)

    assert f.asked_tickers == ["TKR09"], "the held memo name must not be re-asked"
    assert out["sessions_skipped"][0]["retried"] == 1


def test_a_memo_value_that_is_a_string_is_ignored(monkeypatch, tmp_path):
    """A value is a LIST of names: a bare string for a date would split into
    characters if iterated naively, so it is skipped for that date."""
    _wire(monkeypatch, {(t, pd.Timestamp(AS_OF)) for t in TICKERS[:8]}, tmp_path)
    nightly.CHAIN_RETRY_PATH.write_text(json.dumps({AS_OF: "TKR09"}))
    f = _Fetcher()

    out = nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=1)

    assert nightly._read_chain_retry() == {}, "the string is dropped, not char-split"
    assert f.asked == [], "no pending names survived: the session just skips"
    assert "retried" not in out["sessions_skipped"][0]


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


def test_credential_rotated_persists_the_debt_then_reraises(monkeypatch, tmp_path):
    """A rotated key is a FetchError: the generic arm must not file it as
    transient and keep buying against a dead credential. Like the quota cut,
    the refused chunk and every later unasked name land in the retry file
    before the raise (run_nightly turns it into a NightlyStop)."""
    tickers = [f"TKR{i:02d}" for i in range(25)]

    def fail(params):
        if params["ticker"].split(",")[0] == "TKR10":
            return CredentialRotated("orats: HTTP 401, credential rotated")
        return None

    _wire(monkeypatch, set(), tmp_path)
    f = _Fetcher(fail=fail)

    with pytest.raises(CredentialRotated):
        nightly.refresh_forward_chains(tickers, AS_OF, fetcher=f, sessions=1)

    assert f.asked == [AS_OF, AS_OF], "the loop stops at the refusal: no later batch"
    assert "TKR20" not in f.asked_tickers, "no call is made after the raise"
    assert nightly._read_chain_retry() == {AS_OF: {t for t in tickers if t >= "TKR10"}}, \
        "the refused chunk plus all later unasked names are kept for retry"


def test_non_json_200_is_transient_and_the_session_continues(monkeypatch, tmp_path):
    """``record.json()`` raising ValueError on a 200 body must not abort the
    session: the chunk is owed the date, reported in ``failed``, and the later
    batches are still fetched."""
    tickers = [f"TKR{i:02d}" for i in range(25)]
    _wire(monkeypatch, set(), tmp_path)
    f = _Fetcher(bad_json=lambda p: ValueError("no JSON object could be decoded")
                 if p["ticker"].split(",")[0] == "TKR00" else None)

    out = nightly.refresh_forward_chains(tickers, AS_OF, fetcher=f, sessions=1)

    assert f.asked == [AS_OF, AS_OF, AS_OF], "the session continued past the bad body"
    assert out["failed"] and "TKR00" in out["failed"][0], "the failure is still reported"
    assert nightly._read_chain_retry() == {AS_OF: set(tickers[:10])}, \
        "the bad-body chunk is owed the date; the OK batches are not"
    assert out["returned"] == 15, "the two good batches still land their rows"


def test_json_body_raising_a_base_exception_keeps_the_debt(monkeypatch, tmp_path):
    """A KeyboardInterrupt surfacing from ``record.json()`` is not an
    Exception: the wrapper must still bank the current chunk and the unasked
    tail before the session unwinds, then re-raise."""
    tickers = [f"TKR{i:02d}" for i in range(25)]
    _wire(monkeypatch, set(), tmp_path)
    f = _Fetcher(bad_json=lambda p: KeyboardInterrupt()
                 if p["ticker"].split(",")[0] == "TKR10" else None)

    with pytest.raises(KeyboardInterrupt):
        nightly.refresh_forward_chains(tickers, AS_OF, fetcher=f, sessions=1)

    assert f.asked == [AS_OF, AS_OF], "the loop died at the interrupting batch"
    assert nightly._read_chain_retry() == {AS_OF: {t for t in tickers if t >= "TKR10"}}, \
        "the interrupted chunk plus all later unasked names are kept for retry"


@pytest.mark.parametrize("boom", [
    pytest.param(KeyboardInterrupt, id="keyboard-interrupt"),
    pytest.param(type("LoopDefect", (BaseException,), {}), id="other-base-exception"),
])
def test_abnormal_exit_keeps_the_retry_debt(monkeypatch, tmp_path, boom):
    """ANY other BaseException out of the fetch call — the generic
    ``except Exception`` arm never sees it — leaves the current chunk and
    every later unasked one in the retry file before the exception
    propagates; a lost debt is a silently unfetched date. (A plain Exception
    from fetch is transient by design and stays on the old path.)"""
    tickers = [f"TKR{i:02d}" for i in range(25)]

    def fail(params):
        if params["ticker"].split(",")[0] == "TKR10":
            return boom()
        return None

    _wire(monkeypatch, set(), tmp_path)
    f = _Fetcher(fail=fail)

    with pytest.raises(boom):
        nightly.refresh_forward_chains(tickers, AS_OF, fetcher=f, sessions=1)

    assert nightly._read_chain_retry() == {AS_OF: {t for t in tickers if t >= "TKR10"}}, \
        "the unasked remainder is persisted before the unwind"


def test_generic_non_404_fetch_error_stays_transient(monkeypatch, tmp_path):
    """A FetchError that is not CredentialRotated and not a 404 keeps the old
    behaviour: reported, chunked into the retry file, session completes."""
    _wire(monkeypatch, set(), tmp_path)
    f = _Fetcher(fail=lambda p: FetchError("orats hist/strikes returned HTTP 503"))

    out = nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=1)

    assert out["failed"], "the failure is reported"
    assert nightly._read_chain_retry() == {AS_OF: set(TICKERS)}, \
        "a plain FetchError is owed the date, not a stop-the-run signal"


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


def test_prune_keeps_exactly_14_days_and_drops_15(monkeypatch, tmp_path):
    """Boundary: the cutoff is ``>=``, so an entry exactly 14 days behind the
    fetched session survives its persist and 15 days does not."""
    _wire(monkeypatch, set(), tmp_path)
    keep = str((pd.Timestamp(AS_OF) - pd.Timedelta(days=14)).date())
    drop = str((pd.Timestamp(AS_OF) - pd.Timedelta(days=15)).date())
    nightly.CHAIN_RETRY_PATH.write_text(json.dumps({keep: ["OLD1"], drop: ["OLD2"]}))
    f = _Fetcher(fail=lambda p: RuntimeError("HTTP 502"))

    nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=1)

    kept = nightly._read_chain_retry()
    assert kept[keep] == {"OLD1"}, "exactly 14 days old is still inside the window"
    assert drop not in kept, "15 days old is past it"
    assert kept[AS_OF] == set(TICKERS)


def test_a_corrupt_memo_reads_empty_and_the_run_proceeds(monkeypatch, tmp_path):
    """Non-JSON at the memo path: treated as no debt, silently (the docstring
    promise is never raises) — and it never blocks the refresh."""
    _wire(monkeypatch, set(), tmp_path)
    nightly.CHAIN_RETRY_PATH.write_text("{not json at all")
    f = _Fetcher()

    out = nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=1)

    assert nightly._read_chain_retry() == {}
    assert out["calls"] == 1 and f.asked == [AS_OF], "the refresh proceeds"


def test_an_oversized_memo_is_ignored_with_one_stderr_line(monkeypatch, capsys, tmp_path):
    """A memo over CHAIN_RETRY_MAX_BYTES is announced once per read and never
    parsed — and the run proceeds as if there were no memo."""
    _wire(monkeypatch, set(), tmp_path)
    payload = json.dumps({AS_OF: ["TKR09"]})
    nightly.CHAIN_RETRY_PATH.write_text(payload + " " * nightly.CHAIN_RETRY_MAX_BYTES)

    assert nightly._read_chain_retry() == {}
    err = capsys.readouterr().err.splitlines()
    assert len([line for line in err if "[chains] ignoring" in line]) == 1, \
        "one line naming the file and the size cap, per read"
    assert "cap" in err[0]

    f = _Fetcher()
    out = nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=1)
    assert out["calls"] == 1, "the oversized memo never blocks the refresh"


def test_a_directory_at_the_memo_path_is_ignored_and_the_run_proceeds(
        monkeypatch, capsys, tmp_path):
    """A non-regular file at the path (a misplaced ``mv``) is fstat-rejected
    on the open fd before anything is read: reads as {}, the save failure it
    also causes is announced, and nothing raises."""
    _wire(monkeypatch, set(), tmp_path)
    nightly.CHAIN_RETRY_PATH.mkdir()
    f = _Fetcher(fail=lambda p: RuntimeError("HTTP 502"))

    out = nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=1)

    assert out["failed"], "the run proceeds and still reports its fetch failures"
    assert nightly._read_chain_retry() == {}, "a directory reads as no memo"
    err = capsys.readouterr().err
    assert "not a regular file" in err, "the ignore is announced with its reason"
    assert "could not write" in err, "the save onto it is announced, not raised"


def test_a_symlink_at_the_memo_path_is_ignored(monkeypatch, capsys, tmp_path):
    """``O_NOFOLLOW`` refuses the symlink at open (ELOOP): the target's
    contents never reach the run, one stderr line names the file, and the
    read is not silent."""
    _wire(monkeypatch, set(), tmp_path)
    real = tmp_path / "elsewhere.json"
    real.write_text(json.dumps({AS_OF: ["TKR09"]}))
    nightly.CHAIN_RETRY_PATH.symlink_to(real)

    assert nightly._read_chain_retry() == {}, "a symlink is never followed"
    err = capsys.readouterr().err
    assert f"ignoring {nightly.CHAIN_RETRY_PATH.name}" in err, \
        "the refusal is announced with the file's name"
    assert "TKR09" not in err, "the symlinked target's contents are not read"


def test_a_fifo_at_the_memo_path_returns_empty_without_hanging(
        monkeypatch, capsys, tmp_path):
    """A FIFO would block a plain read-only open forever with no writer on the
    other end; ``O_NONBLOCK`` opens it at once, the fstat sees a non-regular
    file, and the read returns {} promptly. The thread+timeout harness fails
    the test instead of hanging the suite if the flag is ever lost."""
    _wire(monkeypatch, set(), tmp_path)
    os.mkfifo(nightly.CHAIN_RETRY_PATH)
    done = threading.Event()
    box = []

    def read():
        box.append(nightly._read_chain_retry())
        done.set()

    t = threading.Thread(target=read, daemon=True)
    t.start()
    assert done.wait(5), "the FIFO read blocked — O_NONBLOCK is missing"
    assert box == [{}], "a FIFO reads as no memo"
    assert "not a regular file" in capsys.readouterr().err


def test_a_memo_write_failure_is_announced_not_raised(monkeypatch, capsys, tmp_path):
    """A lost memo costs at most a re-buy window: an OSError from the atomic
    writer is printed to stderr and the refresh finishes normally."""
    _wire(monkeypatch, set(), tmp_path)

    def boom(path, text):
        raise OSError("read-only filesystem")

    monkeypatch.setattr("engine.data.manifest.write_text_atomic", boom)
    f = _Fetcher(fail=lambda p: RuntimeError("HTTP 502"))

    out = nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=1)

    assert out["failed"] and out["calls"] == 0, "the run completes despite the lost memo"
    assert "could not write orats_chain_retry.json" in capsys.readouterr().err


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
