"""The chain heal-skip: a mostly-held session stops re-buying the unheld tail.

Weekend runs 10-18/19 spent ~114 calls healing sessions already 75%+ held —
the unheld names left are mostly ones with no chain on file, and the batches
came back all-404. Past ``CHAIN_SKIP_HELD_FRACTION`` the session is recorded
in ``sessions_skipped`` and its tail is not bought; the names that failed
transiently or were cut off by quota are kept alive in ``CHAIN_RETRY_PATH``
and ARE re-asked even in a skipped session. Only definitive 404s are
dropped. A fresh (or unheld) session still is healed in full. The memo is
also owed UP FRONT: a fetched session files its whole intended list before
its first fetch and every exit path replaces it with the real failures, so
a SIGKILL/OOM — which runs no handler — leaves the next run re-asking the
whole attempted list. And memo dates past the two-session window but inside
the 14-day retention are re-asked for exactly their owed unheld names. The memo is
also owed UP FRONT: a fetched session files its whole intended list before
its first fetch, so a SIGKILL/OOM — which runs no handler — leaves the next
run re-asking the entire attempted list. And memo dates past the two-session
heal window but inside the 14-day retention are re-asked, exactly their owed
unheld names, before they age out.

Every dependency below the public signature — the fetcher, the store's held
set, the trading calendar, the retry file — is a synthetic fake or a tmp_path
patch, so this needs no data root and makes no provider call.
"""
from __future__ import annotations

import json
import os
import stat
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
    raise on an otherwise-successful (HTTP 200) call. ``body(params)`` may
    return a WRONG-SHAPE success body — a list, a string — in place of the
    well-formed dict.
    """

    def __init__(self, fail=None, bad_json=None, body=None):
        self.asked = []
        self.asked_tickers = []
        self.fail = fail
        self.bad_json = bad_json
        self.body = body

    def fetch(self, source, endpoint, params, note=""):
        self.asked.append(params["tradeDate"])
        self.asked_tickers.extend(params["ticker"].split(","))
        exc = self.fail(params) if self.fail else None
        if exc is not None:
            raise exc
        bad = self.bad_json(params) if self.bad_json else None
        custom = self.body(params) if self.body else None

        class R:
            from_cache = False

            @staticmethod
            def json():
                if bad is not None:
                    raise bad
                if custom is not None:
                    return custom
                return {"data": [{"ticker": t}
                                 for t in params["ticker"].split(",")]}

        return R()


class _MemoSpyFetcher(_Fetcher):
    """Records the on-disk memo as each call sees it, so a test can assert
    the session owes its debt BEFORE the first fetch, not only on an exit
    path that a SIGKILL never reaches."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.memo_at_call: list[dict] = []

    def fetch(self, source, endpoint, params, note=""):
        self.memo_at_call.append(nightly._read_chain_retry())
        return super().fetch(source, endpoint, params, note=note)


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
    every later unasked chunk land in the retry file before it is re-raised
    out of refresh_forward_chains — run_nightly records a degraded refresh
    and scores from cached data (only CredentialRotated stops the run)."""
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
    before the raise — but only this one stops the run: run_nightly turns a
    CredentialRotated into a NightlyStop, where QuotaExhausted degrades the
    refresh and scoring proceeds on cached data."""
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


@pytest.mark.parametrize("shape", [
    pytest.param(["TKR00", "TKR01"], id="list-body"),
    pytest.param("<html>502 Bad Gateway</html>", id="string-body"),
])
def test_wrong_shape_json_is_transient_and_the_session_continues(monkeypatch, tmp_path,
                                                                 shape):
    """A 200 whose JSON parses but is not a dict with a list "data" must not
    reach the AttributeError the row bookkeeping would raise on it: same
    class as a non-JSON body — owed the date, reported in ``failed``, later
    batches still fetched, and its names NOT filed under definitive
    ``missing`` (a bad shape is no verdict on the chain)."""
    tickers = [f"TKR{i:02d}" for i in range(25)]
    _wire(monkeypatch, set(), tmp_path)
    f = _Fetcher(body=lambda p: shape if p["ticker"].split(",")[0] == "TKR00" else None)

    out = nightly.refresh_forward_chains(tickers, AS_OF, fetcher=f, sessions=1)

    assert f.asked == [AS_OF, AS_OF, AS_OF], "the session continued past the wrong shape"
    assert out["failed"] and "TKR00" in out["failed"][0], "the wrong shape is reported"
    assert out["missing"] == [], "a bad shape says nothing about who has a chain"
    assert out["returned"] == 15, "the two good batches still land their rows"
    assert nightly._read_chain_retry() == {AS_OF: set(tickers[:10])}, \
        "the wrong-shape chunk is owed the date; the OK batches are not"


@pytest.mark.parametrize("junk", [
    pytest.param("TKR00", id="string-row"),
    pytest.param(None, id="null-row"),
    pytest.param(5, id="number-row"),
])
def test_non_dict_rows_in_a_200_are_wrong_shape_and_the_session_continues(
        monkeypatch, tmp_path, junk):
    """A 200 whose "data" rows include a non-object is the shape that makes
    ``r.get`` raise AttributeError and take the whole session down: same
    class as a non-list body — the chunk is owed the date, reported in
    ``failed``, none of its rows booked, and the later batches are still
    fetched."""
    tickers = [f"TKR{i:02d}" for i in range(25)]

    def body(p):
        if p["ticker"].split(",")[0] == "TKR00":
            return {"data": [{"ticker": "TKR00"}, junk]}
        return None

    _wire(monkeypatch, set(), tmp_path)
    f = _Fetcher(body=body)

    out = nightly.refresh_forward_chains(tickers, AS_OF, fetcher=f, sessions=1)

    assert f.asked == [AS_OF, AS_OF, AS_OF], "the session continued past the junk row"
    assert out["failed"] and "TKR00" in out["failed"][0], "the junk rows are reported"
    assert "not all objects" in out["failed"][0]
    assert out["rows"] == 15 and out["returned"] == 15, "the bad chunk books nothing"
    assert out["missing"] == [], "a bad shape says nothing about who has a chain"
    assert nightly._read_chain_retry() == {AS_OF: set(tickers[:10])}, \
        "the junk-row chunk is owed the date; the OK batches are not"


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


def test_the_whole_intended_list_is_owed_before_the_first_fetch(monkeypatch, tmp_path):
    """Hard-kill safety: a SIGKILL/OOM runs no handler, so the session files
    the ENTIRE intended list as owed before its first fetch — every exit
    path below only ever replaces it with the real failures. Asserted from
    inside the first fetch call: the memo must already carry the whole list
    while the first chunk is in flight, and a BaseException mid-session must
    leave the unasked tail there."""
    tickers = [f"TKR{i:02d}" for i in range(25)]
    _wire(monkeypatch, set(), tmp_path)
    f = _MemoSpyFetcher(fail=lambda p: KeyboardInterrupt()
                        if p["ticker"].split(",")[0] == "TKR10" else None)

    with pytest.raises(KeyboardInterrupt):
        nightly.refresh_forward_chains(tickers, AS_OF, fetcher=f, sessions=1)

    assert f.memo_at_call[0] == {AS_OF: set(tickers)}, \
        "the first fetch is made with the whole unheld list already owed"
    assert nightly._read_chain_retry() == {AS_OF: {t for t in tickers if t >= "TKR10"}}, \
        "the unwind replaced the up-front debt with the unasked tail"


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


def test_a_re_persisted_older_entry_cannot_outlive_the_newest_key(monkeypatch, tmp_path):
    """The prune's reference is max(stamp, the newest key on file): an older
    memo date re-asked by the older pass that fails again is re-persisted at
    ITS OWN stamp, and pruning against that stamp alone kept it alive 14 days
    behind a fresher debt the file carries. Here the newest key is 16 days
    ahead of the failing re-ask: the stale entry drops out of its own
    persist, while one exactly 14 days behind the newest key survives."""
    older = str((pd.Timestamp("2026-09-10") - pd.Timedelta(days=16)).date())  # drops
    keep = str((pd.Timestamp("2026-09-10") - pd.Timedelta(days=14)).date())  # stays
    newest = "2026-09-10"  # newer than this run's session: asked never, prune ref
    held = {(t, pd.Timestamp(AS_OF)) for t in TICKERS}
    _wire(monkeypatch, held, tmp_path)
    nightly.CHAIN_RETRY_PATH.write_text(
        json.dumps({older: ["TKR08"], keep: ["TKR09"], newest: ["TKR00"]}))
    f = _Fetcher(fail=lambda p: RuntimeError("HTTP 502"))

    out = nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=1)

    assert f.asked == [older, keep], "the future key is inside neither the " \
        "heal window's older-pass bound nor the fetch: prune reference only"
    assert out["failed"], "both re-asks failed transiently"
    kept = nightly._read_chain_retry()
    assert older not in kept, \
        "re-persisted 16 days behind the newest key: the debt ages out here, " \
        "not 14 days behind its own stale stamp"
    assert kept[keep] == {"TKR09"}, "exactly 14 behind the newest key is kept"
    assert kept[newest] == {"TKR00"}, "the newest key itself is never pruned"


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


def test_the_memo_write_fsyncs_the_containing_directory(monkeypatch, tmp_path):
    """Atomic is not durable: write_text_atomic fsyncs the temp FILE, but the
    memo's name arrives via os.replace — a directory entry. So once the write
    succeeds the writer opens CHAIN_RETRY_PATH.parent and fsyncs that fd."""
    _wire(monkeypatch, set(), tmp_path)
    real_fsync = os.fsync
    synced: list[os.stat_result] = []

    def spy(fd):
        synced.append(os.fstat(fd))  # stat while the fd is still open
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", spy)
    f = _Fetcher(fail=lambda p: RuntimeError("HTTP 502"))

    out = nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=1)

    assert out["failed"], "a real persist ran — the 502 owes the date"
    parent = tmp_path.stat()
    dirs = [st for st in synced
            if stat.S_ISDIR(st.st_mode) and st.st_dev == parent.st_dev
            and st.st_ino == parent.st_ino]
    assert len(dirs) == 2, \
        "both persists sync the memo's parent dir: the up-front owed list, " \
        "then the exit-path failures that replace it"
    assert nightly._read_chain_retry() == {AS_OF: set(TICKERS)}, "the memo still lands"


def test_an_fsync_failure_is_announced_not_raised(monkeypatch, capsys, tmp_path):
    """The sync is a durability courtesy, not a gate: an OSError from the
    directory fsync prints on stderr exactly like the write failure it shares
    a handler with, and the refresh finishes normally."""
    _wire(monkeypatch, set(), tmp_path)
    real_fsync = os.fsync

    def refuse_dirs(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("fsync refused on the directory")
        return real_fsync(fd)  # write_text_atomic's own file fsync stays real

    monkeypatch.setattr(os, "fsync", refuse_dirs)
    f = _Fetcher(fail=lambda p: RuntimeError("HTTP 502"))

    out = nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=1)

    assert out["failed"] and out["calls"] == 0, "the run completes despite the failed sync"
    err = capsys.readouterr().err
    assert "could not write orats_chain_retry.json" in err, \
        "the fsync OSError is announced with the same prefix as a write failure"
    assert "fsync refused" in err


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


def test_an_older_memo_date_is_re_asked_for_exactly_its_owed_names(
        monkeypatch, tmp_path, capsys):
    """The heal window is two sessions but retention is fourteen days: a memo
    date in that gap is never touched by the session loop, so the pass after
    it re-asks it for exactly its owed unheld names — minus a name the store
    already holds for that date, minus a dropped symbol — and clears the
    entry on success. A 15-day-old entry is past the same cutoff the prune
    uses and must not be asked at all."""
    older = str((pd.Timestamp(AS_OF) - pd.Timedelta(days=12)).date())
    stale = str((pd.Timestamp(AS_OF) - pd.Timedelta(days=15)).date())
    held = {(t, pd.Timestamp(s)) for s in (AS_OF, "2026-09-01") for t in TICKERS}
    held.add(("TKR08", pd.Timestamp(older)))
    _wire(monkeypatch, held, tmp_path, stamps=["2026-09-01", AS_OF])
    nightly.CHAIN_RETRY_PATH.write_text(
        json.dumps({older: ["TKR08", "TKR09", "ZZZ9"], stale: ["TKR07"]}))
    f = _Fetcher()

    out = nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=2)

    assert f.asked == [older], "both window sessions were fully held; " \
        "only the older memo date is fetched"
    assert f.asked_tickers == ["TKR09"], "exactly the owed unheld names: not " \
        "the one already held there, not the dropped symbol"
    assert out["older_retried"] == [{"session": older, "names": 1}]
    err = capsys.readouterr().err
    assert err.count(f"[chains] session {older}:") == 1, "one stderr line per date"
    kept = nightly._read_chain_retry()
    assert older not in kept, "a successful re-ask clears the entry"
    assert kept == {stale: {"TKR07"}}, "the 15-day-old entry is neither asked nor cleared"


def test_non_canonical_memo_date_keys_are_ignored_with_one_stderr_line(
        monkeypatch, tmp_path, capsys):
    """A memo date is re-asked only when it is exactly its own "YYYY-MM-DD"
    string: keys carrying a time part or a bare yyyymmdd parse to a real day
    but must never reach the fetch as a tradeDate or the persist as a stamp —
    they are skipped, counted on one stderr line, and left in the file for
    the prune, while the canonical date is still asked by its canonical
    stamp."""
    canon = str((pd.Timestamp(AS_OF) - pd.Timedelta(days=8)).date())  # 2026-08-25
    with_time = f"{(pd.Timestamp(AS_OF) - pd.Timedelta(days=11)).date()} 10:00:00"
    compact = str((pd.Timestamp(AS_OF) - pd.Timedelta(days=9)).date()).replace("-", "")
    held = {(t, pd.Timestamp(AS_OF)) for t in TICKERS}
    _wire(monkeypatch, held, tmp_path)
    nightly.CHAIN_RETRY_PATH.write_text(
        json.dumps({canon: ["TKR08"], with_time: ["TKR09"], compact: ["TKR00"]}))
    f = _Fetcher()

    out = nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=1)

    assert f.asked == [canon], "only the canonical key is asked, by its " \
        "canonical date; the time-part and compact keys never reach tradeDate"
    assert f.asked_tickers == ["TKR08"]
    assert out["older_retried"] == [{"session": canon, "names": 1}]
    err = capsys.readouterr().err
    assert err.count("non-canonical") == 1, "one counted line for the whole pass"
    assert "ignored 2 non-canonical" in err, "both keys named by count, not silently"
    kept = nightly._read_chain_retry()
    assert canon not in kept, "the successful canonical re-ask cleared its entry"
    assert kept == {with_time: {"TKR09"}, compact: {"TKR00"}}, \
        "the ignored keys survive untouched for the prune to age out"


def test_skip_held_false_never_runs_the_older_pass(monkeypatch, tmp_path):
    """The older-date re-ask belongs to the heal pass; a forced full refresh
    re-buys everything for the window sessions anyway and must not spend the
    memo's older dates."""
    older = str((pd.Timestamp(AS_OF) - pd.Timedelta(days=12)).date())
    _wire(monkeypatch, set(), tmp_path, stamps=["2026-09-01", AS_OF])
    nightly.CHAIN_RETRY_PATH.write_text(json.dumps({older: ["TKR09"]}))
    f = _Fetcher()

    out = nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=2,
                                         skip_held=False)

    assert f.asked == [AS_OF, "2026-09-01"], "the window sessions, newest first"
    assert "older_retried" not in out
    assert nightly._read_chain_retry() == {older: {"TKR09"}}, \
        "the older entry survives untouched — the memo writes go to the fetched stamps"
