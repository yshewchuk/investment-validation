"""The chain heal pass: per-ticker skipping of chronic no-chain names.

Weekend runs 10-18/19 spent ~114 calls healing sessions that were already
mostly held, because the unheld names left were mostly ones with no chain on
file. The first fix skipped the whole session past a held fraction, which also
skipped the late recoveries sitting in the same tail. The rule now is PER
TICKER: every session is asked for every wanted name it does not hold, except
a name that has come back with no chain — absent from a 200 body, or a 404 —
for CHAIN_MISS_SKIP_AFTER consecutive runs. Such a name waits for its
every-CHAIN_MISS_REPROBE_DAYS re-probe; a probe that returns data clears the
count, one that misses buys another week of quiet. The streak counts ONCE per
anchor date — a same-day rerun does not deepen it — and the skip age is clamped
at zero, so a backdated rerun never re-probes a future-dated ``last``. The
skip is never silent: it is in the report (``miss_skipped``, ``miss_counts``)
and on stderr.

Transient debt is not evidence of no chain, so a name the ledger owes is never
skipped by the count, and a transient, wrong-shape, non-JSON, quota or
abnormal-exit ask leaves the count untouched — only a definitive answer moves
it. The ledger is one file with two sections, ``owed`` (per session date, the
retry memo) and ``misses`` (per ticker, the counts); a flat
``{"<date>": [names]}`` file — the format before counts existed — still reads.

The memo protocol is unchanged and is tested here too: a fetched session OWES
THE WHOLE INTENDED LIST before its first fetch, and every exit path MERGES the
run's real failures into that entry — (names on file minus the ones this call
attempted) plus this call's failures — so a filtered run never erases debt it
did not attempt, and a SIGKILL/OOM — which runs no handler — leaves the next
run re-asking the entire attempted list plus whatever else was on file. Memo
dates past the two-session heal window but inside the 14-day retention are
re-asked for exactly their owed unheld names, before they age out. Every memo
re-ask is fetched with a non-None ``refetch_token`` (the cached body may be
the malformed 200 that filed the debt); a first fetch passes none.

Every dependency below the public signature — the fetcher, the store's held
set, the trading calendar, the ledger file — is a synthetic fake or a tmp_path
patch, so this needs no data root and makes no provider call. The chunk loop
carries P19 progress logging: one stage-start line, progress lines at a
bounded cadence (never one per chunk) and one stage-end line naming the
session's own counts and RSS.
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


def _days(before: int, *, of: str = AS_OF) -> str:
    """A canonical date ``before`` days behind the run's anchor."""
    return str((pd.Timestamp(of) - pd.Timedelta(days=before)).date())


def _seed_ledger(*, owed: dict | None = None, misses: dict | None = None,
                 legacy: bool = False) -> None:
    """Write the ledger directly, in the new shape or the legacy flat one."""
    if legacy:
        payload = owed or {}
    else:
        payload = {"owed": owed or {}, "misses": misses or {}}
    nightly.CHAIN_RETRY_PATH.write_text(json.dumps(payload))


class _Fetcher:
    """A ``Fetcher`` stand-in that answers every batch with the whole chunk.

    ``fail(params)`` may return an exception to raise instead of a record.
    ``bad_json(params)`` may return an exception for ``record.json()`` to
    raise on an otherwise-successful (HTTP 200) call. ``body(params)`` may
    return a WRONG-SHAPE success body — a list, a string — in place of the
    well-formed dict. ``absent`` names are dropped from the well-formed body,
    which is a 200 that says those tickers have no chain on that date. Every
    call's ``refetch_token`` is recorded in ``asked_tokens``, parallel to
    ``asked``.
    """

    def __init__(self, fail=None, bad_json=None, body=None, absent=()):
        self.asked = []
        self.asked_tickers = []
        self.asked_tokens = []
        self.fail = fail
        self.bad_json = bad_json
        self.body = body
        self.absent = set(absent)

    def fetch(self, source, endpoint, params, note="", refetch_token=None):
        self.asked.append(params["tradeDate"])
        self.asked_tickers.extend(params["ticker"].split(","))
        self.asked_tokens.append(refetch_token)
        exc = self.fail(params) if self.fail else None
        if exc is not None:
            raise exc
        bad = self.bad_json(params) if self.bad_json else None
        custom = self.body(params) if self.body else None
        absent = self.absent

        class R:
            from_cache = False

            @staticmethod
            def json():
                if bad is not None:
                    raise bad
                if custom is not None:
                    return custom
                return {"data": [{"ticker": t}
                                 for t in params["ticker"].split(",")
                                 if t not in absent]}

        return R()


class _MemoSpyFetcher(_Fetcher):
    """Records the on-disk memo as each call sees it, so a test can assert
    the session owes its debt BEFORE the first fetch, not only on an exit
    path that a SIGKILL never reaches."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.memo_at_call: list[dict] = []

    def fetch(self, source, endpoint, params, note="", refetch_token=None):
        self.memo_at_call.append(nightly._read_chain_retry())
        return super().fetch(source, endpoint, params, note=note,
                             refetch_token=refetch_token)


class _OldSignatureFetcher:
    """A fetcher written before the ``refetch_token`` kwarg existed: the ``fetch``
    signature is exactly ``(self, source, endpoint, params, note=None)``. A first
    fetch must not hand it an unexpected keyword."""

    def __init__(self):
        self.asked_tickers = []
        self.notes = []

    def fetch(self, source, endpoint, params, note=None):
        self.asked_tickers.extend(params["ticker"].split(","))
        self.notes.append(note)

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


def test_a_mostly_held_session_is_still_fetched_for_its_unheld_names(
        monkeypatch, capsys, tmp_path):
    """8 of 10 held: the old rule skipped the whole session here and lost the
    recoveries sitting in that tail. Now the session is fetched for exactly its
    two unheld names and nothing else — the held accounting still lands in
    ``skipped_held``, but a held PAIR is the only thing that goes unasked."""
    _wire(monkeypatch, {(t, pd.Timestamp(AS_OF)) for t in TICKERS[:8]}, tmp_path)
    f = _Fetcher()

    out = nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=1)

    assert f.asked == [AS_OF], "the unheld tail is bought, not skipped"
    assert f.asked_tickers == ["TKR08", "TKR09"], "exactly the two unheld names"
    assert out["skipped_held"] == 8
    assert out["miss_skipped"]["tickers"] == 0, "nothing was skipped on the count"
    assert "skipping" not in capsys.readouterr().err, "no skip line without a skip"


def test_a_memo_name_that_is_now_held_is_not_re_asked(monkeypatch, tmp_path):
    """The held filter, not the memo, decides: a memo name the store already
    holds for that session is not bought again, and the session's unheld names
    are. The held name's debt survives the merge untouched — this run never
    attempted it, so it cannot speak for it — until the prune ages it out."""
    _wire(monkeypatch, {(t, pd.Timestamp(AS_OF)) for t in TICKERS[:8]}, tmp_path)
    _seed_ledger(owed={AS_OF: ["TKR00", "TKR09"]})
    f = _Fetcher()

    nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=1)

    assert f.asked_tickers == ["TKR08", "TKR09"], \
        "the held memo name must not be re-asked; the unheld ones are"
    assert nightly._read_chain_retry() == {AS_OF: {"TKR00"}}, \
        "the OK fetch cleared TKR09 and left the debt it never attempted"


def test_an_owed_name_is_never_skipped_by_the_miss_rule(monkeypatch, tmp_path):
    """The count skips a chronic no-chain name, but transient debt is not
    evidence of no chain: TKR09 is at the threshold AND owes the date (a 502,
    never a verdict), and is asked anyway; TKR08 is at the threshold with no
    debt and is not."""
    _wire(monkeypatch, set(), tmp_path)
    at_threshold = {"n": nightly.CHAIN_MISS_SKIP_AFTER, "last": AS_OF}
    _seed_ledger(owed={AS_OF: ["TKR09"]},
                  misses={"TKR08": at_threshold, "TKR09": dict(at_threshold)})
    f = _Fetcher()

    out = nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=1)

    assert "TKR08" not in f.asked_tickers, "the chronic name is skipped"
    assert "TKR09" in f.asked_tickers, "an owed name is re-asked whatever its count says"
    assert out["miss_skipped"]["tickers"] == 1, "the owed name is not counted as a saving"


def test_memo_driven_chunk_gets_refetch_token_first_fetch_does_not(
        monkeypatch, tmp_path):
    """The cached body may be the malformed 200 that filed the debt, so a memo
    re-ask — here an older memo date past the heal window — is fetched with a
    non-None ``refetch_token``: a distinct cache key makes it a fresh provider
    call. An ordinary first fetch is the cache-first path and passes none."""
    older = _days(12)
    held = {(t, pd.Timestamp(s)) for s in (AS_OF, _days(1)) for t in TICKERS}
    _wire(monkeypatch, held, tmp_path, stamps=(_days(1), AS_OF))
    _seed_ledger(owed={older: ["TKR09"]})
    f1 = _Fetcher()

    out = nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f1, sessions=2)

    assert f1.asked == [older], "both window sessions are held; the memo date is asked"
    assert f1.asked_tickers == ["TKR09"] and out["older_retried"] == [
        {"session": older, "names": 1}]
    assert len(f1.asked_tokens) == 1, "one memo-driven chunk"
    assert f1.asked_tokens[0] is not None, \
        "the re-ask must not replay the cached body that filed the debt"

    monkeypatch.setattr("engine.replay.available_chain_keys", lambda: set())
    f2 = _Fetcher()
    nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f2, sessions=2)

    assert f2.asked == [AS_OF, _days(1)], "the unheld sessions are healed in full"
    assert f2.asked_tokens == [None, None], "a first fetch gets no token"


def test_full_fetch_token_is_per_chunk(monkeypatch, tmp_path):
    """The refetch token is per CHUNK, not per session: a 0%-held session
    (whose whole universe is fetched) whose memo already owes TKR03 gets the
    run-day token only on the chunk containing TKR03 — that name's cached body
    may be the malformed 200 that filed the debt — and no token on the other
    chunk, which is all first fetches."""
    _wire(monkeypatch, set(), tmp_path)
    _seed_ledger(owed={AS_OF: ["TKR03"]})
    f = _Fetcher()

    nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=1, batch=5)

    assert f.asked_tickers == TICKERS, "the whole universe is still fetched"
    assert len(f.asked_tokens) == 2, "two chunks"
    assert "TKR03" in f.asked_tickers[:5]
    assert f.asked_tokens[0] is not None, \
        "the chunk holding the memo-owed name must not replay the cached body"
    assert f.asked_tokens[1] is None, \
        "the all-first-fetch chunk stays ordinary cache-first"


def test_an_old_signature_fetcher_works_for_a_normal_first_fetch(monkeypatch, tmp_path):
    """A first fetch of an unheld session is the all-cache-first path, so the
    token kwarg is never passed — an injected fetcher whose ``fetch`` predates
    ``refetch_token`` (``(self, source, endpoint, params, note=None)``) must not
    be handed an unexpected keyword (no TypeError) and still fetches the chains."""
    _wire(monkeypatch, set(), tmp_path)
    f = _OldSignatureFetcher()

    out = nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=1)

    assert f.asked_tickers == sorted(TICKERS), "the chains are fetched, no TypeError"
    assert out["calls"] == 1
    assert f.notes == ["phase3 forward chains"], "note is passed; refetch_token is not"


def test_a_memo_value_that_is_a_string_is_ignored(monkeypatch, tmp_path):
    """A value is a LIST of names: a bare string for a date would split into
    characters if iterated naively, so it is skipped for that date — and the
    unheld pairs of that session are still bought, junk memo or not."""
    _wire(monkeypatch, {(t, pd.Timestamp(AS_OF)) for t in TICKERS[:8]}, tmp_path)
    _seed_ledger(owed={AS_OF: "TKR09"}, legacy=True)
    f = _Fetcher()

    nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=1)

    assert nightly._read_chain_retry() == {}, "the string is dropped, not char-split"
    assert f.asked_tickers == ["TKR08", "TKR09"], \
        "no character of it was asked, and the session's unheld names were"
    assert nightly._read_chain_ledger()[1] == {}, "a junk memo value is no verdict either"


def test_a_zero_held_session_is_still_healed(monkeypatch, tmp_path):
    """A night that did not run leaves ~0% held — exactly what healing is for."""
    _wire(monkeypatch, set(), tmp_path)
    f = _Fetcher()

    out = nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=1)

    assert out["calls"] == 1  # 10 tickers at CHAIN_BATCH=10 -> one call
    assert f.asked == [AS_OF]
    assert f.asked_tickers == sorted(TICKERS), "the heal asks the whole unheld set"
    assert out["miss_skipped"] == {"tickers": 0, "pairs": 0,
                                   "est_weekly_calls_saved": 0.0}


def test_transient_failure_is_retried_on_the_next_run(monkeypatch, tmp_path):
    """A non-404 exception puts the chunk in the retry file, and the next run
    re-asks exactly those names: an owed name is never dropped by the miss
    count, and it is still unheld, so the ordinary heal asks for it."""
    _wire(monkeypatch, set(), tmp_path)
    f1 = _Fetcher(fail=lambda p: RuntimeError("orats hist/strikes: connection reset"))

    out = nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f1, sessions=1)

    assert out["failed"], "the failure is still reported"
    assert out["calls"] == 0
    assert nightly._read_chain_retry() == {AS_OF: set(TICKERS)}

    monkeypatch.setattr("engine.replay.available_chain_keys",
                        lambda: {(t, pd.Timestamp(AS_OF)) for t in TICKERS[:8]})
    f2 = _Fetcher()
    nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f2, sessions=1)

    assert f2.asked_tickers == ["TKR08", "TKR09"], "the retry set is asked again"
    assert nightly._read_chain_retry() == {AS_OF: set(TICKERS[:8])}, \
        "the merge cleared only the names this call attempted: the debt of " \
        "the now-held names was never attempted and stays on file"
    assert nightly._read_chain_ledger()[1] == {}, "a call that never answered is no miss"


def test_filtered_run_preserves_unattempted_owed_names(monkeypatch, tmp_path):
    """An older memo date's owed list is a FILTERED run: the persist merges —
    (names on file - attempted) + failures — it never replaces. A name owed on
    that date but off this run's universe (BBB) is unattempted by the
    owed-intersection, so a clean re-ask of AAA leaves {BBB} and a transient
    AAA failure leaves {AAA, BBB}: neither run erases debt it cannot speak
    for."""
    older = _days(12)
    tickers = ["AAA"] + TICKERS[:9]
    held = {(t, pd.Timestamp(day)) for day in (AS_OF, _days(1)) for t in tickers}
    _wire(monkeypatch, held, tmp_path, stamps=(_days(1), AS_OF))

    _seed_ledger(owed={older: ["AAA", "BBB"]})
    f = _Fetcher()
    nightly.refresh_forward_chains(tickers, AS_OF, fetcher=f, sessions=2)

    assert f.asked == [older], "both window sessions are held; the memo date is asked"
    assert f.asked_tickers == ["AAA"], "only memo ∩ wanted is attempted"
    assert "BBB" not in f.asked_tickers, "the off-universe name is never asked"
    assert nightly._read_chain_retry() == {older: {"BBB"}}, \
        "the OK verdict speaks only for AAA: unattempted BBB stays owed"

    _seed_ledger(owed={older: ["AAA", "BBB"]})
    f2 = _Fetcher(fail=lambda p: RuntimeError("orats hist/strikes: HTTP 502"))
    out2 = nightly.refresh_forward_chains(tickers, AS_OF, fetcher=f2, sessions=2)

    assert out2["failed"], "the transient failure is still reported"
    assert f2.asked_tickers == ["AAA"]
    assert nightly._read_chain_retry() == {older: {"AAA", "BBB"}}, \
        "the failure re-files AAA and the merge never touched BBB"
    assert nightly._read_chain_ledger()[1] == {}, \
        "and the 502 counts as no miss for AAA, whatever its ledger said"


def test_a_404_exception_is_final_and_never_retried(monkeypatch, tmp_path):
    """"HTTP 404" is a definitive no-chain: recorded as missing (like the
    200-but-short path) and never entered in the retry file."""
    _wire(monkeypatch, set(), tmp_path)
    f = _Fetcher(fail=lambda p: RuntimeError("orats hist/strikes returned HTTP 404"))

    out = nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=1)

    assert out["missing"] == sorted(TICKERS)
    assert out["failed"], "the existing failed reporting is unchanged by the 404 split"
    assert nightly._read_chain_retry() == {}, "a 404 has no chain; retrying cannot help"
    assert nightly._read_chain_ledger()[1] == {t: {"n": 1, "last": AS_OF} for t in TICKERS}, \
        "the 404 IS a verdict: every name in the chunk is one miss deeper"


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


def test_hard_kill_upfront_persist_keeps_unrelated_owed_names(monkeypatch, tmp_path):
    """The up-front owed persist is a MERGE too: a run wanting only AAA that
    dies on a BaseException from its very first fetch — the hard-kill
    analogue, the exit-path bookkeeping below never runs — must leave on file
    BOTH the debt it never attempted (ZZZ, off the universe) and the whole
    attempted list (AAA), which the up-front persist filed beside ZZZ and the
    unwind re-banked. Replace-on-persist would have erased ZZZ."""
    _wire(monkeypatch, set(), tmp_path)
    nightly.CHAIN_RETRY_PATH.write_text(json.dumps({AS_OF: ["ZZZ"]}))
    f = _Fetcher(fail=lambda p: KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        nightly.refresh_forward_chains(["AAA"], AS_OF, fetcher=f, sessions=1)

    assert f.asked == [AS_OF], "the kill came at the first fetch"
    assert nightly._read_chain_retry() == {AS_OF: {"AAA", "ZZZ"}}, \
        "the up-front merge filed AAA beside ZZZ and the unwind kept both"


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


def test_a_future_dated_key_does_not_move_the_prune_cutoff(monkeypatch, tmp_path):
    """The prune's reference is max(stamp, the canonical keys on file) but
    NEVER later than the persist's own stamp: a hand-edited key past this run
    used to move the cutoff years forward and the FIRST persist (the
    up-front owed list) erased every in-retention debt behind it before the
    older pass even read the memo, so that debt was never re-asked. Here the
    newest key is 8 days past the run; the entries 16 and 14 days behind it
    are 8 and 6 behind the run — inside retention — so they survive every
    persist and reach the older pass. The future key is kept but inert: it is
    not a past session, so it is never fetched."""
    older = str((pd.Timestamp(AS_OF) - pd.Timedelta(days=8)).date())   # 2026-08-25
    keep = str((pd.Timestamp(AS_OF) - pd.Timedelta(days=6)).date())    # 2026-08-27
    future = "2026-09-10"  # later than this run's newest session
    held = {(t, pd.Timestamp(AS_OF)) for t in TICKERS[:5]}
    _wire(monkeypatch, held, tmp_path)
    nightly.CHAIN_RETRY_PATH.write_text(
        json.dumps({older: ["TKR08"], keep: ["TKR09"], future: ["TKR00"]}))
    f = _Fetcher(fail=lambda p: RuntimeError("HTTP 502"))

    out = nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=1)

    assert f.asked == [AS_OF, older, keep], \
        "the up-front persist's prune kept the debts for the older pass; " \
        "the future key is never fetched"
    assert out["failed"], "every ask failed transiently"
    kept = nightly._read_chain_retry()
    assert kept[older] == {"TKR08"} and kept[keep] == {"TKR09"}, \
        "16 and 14 days behind the future key, inside retention of the run"
    assert kept[AS_OF] == {t for t in TICKERS if t >= "TKR05"}
    assert kept[future] == {"TKR00"}, "kept but inert: never a prune reference"


def test_exactly_14_days_behind_is_kept_by_the_prune_and_re_asked(
        monkeypatch, tmp_path):
    """The prune and the older pass cut at the SAME reference (stamps[0])
    minus 14 days: an entry exactly on that boundary is dropped by neither
    and is re-asked once more before it ages out. A persist cutting later
    (as the future-key bug did) or a pass asking from earlier would silently
    lose the pair either side of the boundary."""
    edge = str((pd.Timestamp(AS_OF) - pd.Timedelta(days=14)).date())  # 2026-08-19
    held = {(t, pd.Timestamp(AS_OF)) for t in TICKERS[:5]}
    _wire(monkeypatch, held, tmp_path)
    nightly.CHAIN_RETRY_PATH.write_text(json.dumps({edge: ["TKR08"]}))
    f = _Fetcher(fail=lambda p: RuntimeError("HTTP 502"))

    out = nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=1)

    assert f.asked == [AS_OF, edge], "the heal persist's prune kept the " \
        "boundary entry, and the older pass re-asks it under the same cutoff"
    assert out["older_retried"] == [{"session": edge, "names": 1}]
    kept = nightly._read_chain_retry()
    assert kept[edge] == {"TKR08"}, "exactly 14 behind is inside both cutoffs"
    assert kept[AS_OF] == {t for t in TICKERS if t >= "TKR05"}


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
    assert len(dirs) == 3, \
        "every persist syncs the ledger's parent dir: the up-front owed list, " \
        "the exit-path failures that replace it, then the end-of-run counts"
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
    """The held check belongs to the heal pass only; an explicit full refresh
    re-buys every pair, even at 100% held. Nothing in the ledger is on the
    count here, so there is no per-ticker skip to make."""
    _wire(monkeypatch, {(t, pd.Timestamp(AS_OF)) for t in TICKERS}, tmp_path)
    f = _Fetcher()

    out = nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=1,
                                         skip_held=False)

    assert out["calls"] == 1
    assert f.asked == [AS_OF]
    assert f.asked_tickers == sorted(TICKERS)
    assert out["miss_skipped"]["tickers"] == 0
    assert out["skipped_held"] == 0


def test_a_forced_refresh_still_drops_the_chronic_names(monkeypatch, tmp_path):
    """``skip_held=False`` forces the HELD check, not the count: a chronic
    no-chain name stays out of a forced refresh too, because the skip is a cost
    decision rather than a cache one."""
    _wire(monkeypatch, {(t, pd.Timestamp(AS_OF)) for t in TICKERS}, tmp_path)
    _seed_ledger(misses={"TKR09": {"n": nightly.CHAIN_MISS_SKIP_AFTER, "last": AS_OF}})
    f = _Fetcher()

    out = nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=1,
                                         skip_held=False)

    assert f.asked_tickers == [t for t in TICKERS if t != "TKR09"], \
        "every pair but the chronic one is re-bought"
    assert out["skipped_held"] == 0, "the held check is off, and only that"
    assert out["miss_skipped"] == {"tickers": 1, "pairs": 1,
                                   "est_weekly_calls_saved": 0.7}


def test_a_mostly_held_session_requests_only_its_unheld_names(monkeypatch, tmp_path):
    """No held fraction skips anything now: 74 of 100 held is fetched, and the
    request names the 26 unheld tickers and nothing already held. The pair, not
    the session, is the unit of the heal."""
    tickers = [f"TKR{i:03d}" for i in range(100)]
    unheld = tickers[74:]
    _wire(monkeypatch, {(t, pd.Timestamp(AS_OF)) for t in tickers[:74]}, tmp_path)
    f = _Fetcher()

    out = nightly.refresh_forward_chains(tickers, AS_OF, fetcher=f, sessions=1)

    assert out["calls"] == 3, "26 unheld at CHAIN_BATCH=10 -> ceil(26/10)"
    assert f.asked == [AS_OF, AS_OF, AS_OF]
    assert f.asked_tickers == unheld, "the request is exactly the unheld set"
    assert out["skipped_held"] == 74


def test_two_sessions_are_healed_independently_by_pair(monkeypatch, tmp_path):
    """Through the public entry point: the newest session at 0% held is fetched
    in full and the older one at 80% held is fetched for its two unheld names —
    the held fraction buys nothing back, and the two sessions are accounted
    independently."""
    older = "2026-09-01"
    held = {(t, pd.Timestamp(older)) for t in TICKERS[:8]}
    _wire(monkeypatch, held, tmp_path, stamps=[older, AS_OF])
    f = _Fetcher()

    out = nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=2)

    assert out["sessions"] == [AS_OF, older]
    assert f.asked == [AS_OF, older], "both sessions are asked"
    assert f.asked_tickers == sorted(TICKERS) + ["TKR08", "TKR09"]
    assert out["skipped_held"] == 8


def test_an_older_memo_date_is_re_asked_for_exactly_its_owed_names(
        monkeypatch, tmp_path, capsys):
    """The heal window is two sessions but retention is fourteen days: a memo
    date in that gap is never touched by the session loop, so the pass after
    it re-asks it for exactly its owed unheld names — minus a name the store
    already holds for that date, minus a dropped symbol — and merges those
    attempted names out of the entry on success: the names this call never
    attempted (the held one and the dropped symbol) stay owed until the prune
    ages them out. A 15-day-old entry is past the same cutoff the prune
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
    assert kept == {older: {"TKR08", "ZZZ9"}}, \
        "the merge cleared the one attempted name: the held name and the " \
        "dropped symbol were never attempted and stay owed. The 15-day-old " \
        "entry was neither asked (past the same cutoff the prune uses) nor " \
        "re-askable, and the run's final ledger write ages it out"


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
    assert kept == {}, "non-canonical keys are dropped by the persist that follows the pass"


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


def test_the_chunk_loop_logs_bounded_progress_not_one_line_per_chunk(
        monkeypatch, capsys, tmp_path):
    """P19: a fetched session's chunk loop logs one stage-start line, progress
    lines only at the bounded cadence (both gates: >= 60s AND >= 50 chunks
    since the last one — never per chunk) and one stage-end line carrying the
    session's own call/failure/missing counts and RSS. 120 wanted names at
    batch=1 under a fake monotonic advancing 1s per read stay far below one
    line per chunk."""
    names = [f"TKR{i:04d}" for i in range(120)]
    _wire(monkeypatch, set(), tmp_path)
    clock = [0.0]

    def fake_monotonic():
        now = clock[0]
        clock[0] += 1.0
        return now

    monkeypatch.setattr(nightly.time, "monotonic", fake_monotonic)
    f = _Fetcher()

    out = nightly.refresh_forward_chains(names, AS_OF, fetcher=f, sessions=1,
                                         batch=1)

    assert out["calls"] == 120, "the whole list was fetched"
    err = capsys.readouterr().err
    lines = [ln for ln in err.splitlines() if ln.startswith(f"[chains] {AS_OF}:")]
    assert lines.count(f"[chains] {AS_OF}: fetching 120 names in 120 calls") == 1, \
        "exactly one stage-start line"
    progress = [ln for ln in lines if " chunks, " in ln and " calls/s" in ln]
    assert 1 <= len(progress) < 12, \
        f"bounded cadence, far fewer than 120: {progress}"
    end = [ln for ln in lines if " done in " in ln]
    assert len(end) == 1, "exactly one stage-end line"
    assert f"calls={out['calls']}" in end[0] and "failed=0" in end[0] \
        and "missing=0" in end[0] and end[0].endswith("G"), \
        "the end line names this session's own counts and its RSS"


# -- the per-ticker miss skip ------------------------------------------------

def test_three_consecutive_misses_keep_asking_and_the_fourth_skips_it(
        monkeypatch, capsys, tmp_path):
    """The threshold is the measured one: pairs at 1, 2 or 3 batch 404s went on
    to be held 9.4% / 14% / 22% of the time, so those names stay asked. Of the
    pairs that reached a 4th, none ever were: the 5th run stops paying for it
    and says so, in the report and on stderr. Consecutive means consecutive
    ANCHOR DATES — a same-day rerun counts once, not twice (see the next
    test)."""
    _wire(monkeypatch, set(), tmp_path)
    threshold = nightly.CHAIN_MISS_SKIP_AFTER
    anchors = [str((pd.Timestamp(AS_OF) + pd.Timedelta(days=i)).date())
               for i in range(threshold + 1)]

    for run, anchor in enumerate(anchors[:threshold], start=1):
        f = _Fetcher(absent={"TKR09"})
        out = nightly.refresh_forward_chains(TICKERS, anchor, fetcher=f, sessions=1)
        assert "TKR09" in f.asked_tickers, f"miss {run} of {threshold} is still asked"
        assert out["miss_skipped"]["tickers"] == 0
        assert nightly._read_chain_ledger()[1] == {"TKR09": {"n": run, "last": anchor}}

    f = _Fetcher(absent={"TKR09"})
    out = nightly.refresh_forward_chains(TICKERS, anchors[threshold], fetcher=f,
                                         sessions=1)

    assert "TKR09" not in f.asked_tickers, "the run after the 4th miss does not ask"
    assert "TKR08" in f.asked_tickers, "and it asks every other name all the same"
    assert out["requested"] == len(TICKERS), "requested counts the universe, not the ask"
    assert out["miss_skipped"] == {"tickers": 1, "pairs": 1,
                                   "est_weekly_calls_saved": 0.7}
    assert out["miss_counts"] == {"tracked": 1, "at_threshold": 1, "reset": 0}
    err = capsys.readouterr().err
    assert ("[chains] skipping 1 tickers after 4 consecutive misses "
            "(re-probe every 7d): 1 pairs, ~0.7 calls/week saved") in err, \
        "any skip is in the report and on stderr, never silent"
    assert nightly._read_chain_ledger()[1] == {
        "TKR09": {"n": threshold, "last": anchors[threshold - 1]}}, \
        "a run that never asked the name leaves its count exactly where it was"


def test_two_runs_on_the_same_anchor_date_count_one_miss(monkeypatch, tmp_path):
    """A rerun is not a new night: when the ledger's ``last`` is already the
    run's anchor date, a MISS verdict does not deepen the streak. Only the
    next anchor date takes it to the next count — the same day rerunning the
    nightly twice must not push a name toward the skip twice as fast."""
    _wire(monkeypatch, set(), tmp_path)

    for run in (1, 2):
        f = _Fetcher(absent={"TKR09"})
        nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=1)
        assert "TKR09" in f.asked_tickers
        assert nightly._read_chain_ledger()[1] == {"TKR09": {"n": 1, "last": AS_OF}}, \
            f"rerun {run} on the entry's own anchor date counts once per day"

    next_anchor = str((pd.Timestamp(AS_OF) + pd.Timedelta(days=1)).date())
    f = _Fetcher(absent={"TKR09"})
    nightly.refresh_forward_chains(TICKERS, next_anchor, fetcher=f, sessions=1)

    assert nightly._read_chain_ledger()[1] == {"TKR09": {"n": 2, "last": next_anchor}}, \
        "the next anchor date is the second miss"


def test_a_same_anchor_rerun_still_clears_on_data(monkeypatch, tmp_path):
    """The once-per-day rule guards increments only: a rerun whose answer is
    data still wipes the entry, whatever its ``last`` says."""
    _wire(monkeypatch, set(), tmp_path)
    _seed_ledger(misses={"TKR09": {"n": 2, "last": AS_OF}})
    f = _Fetcher()

    out = nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=1)

    assert nightly._read_chain_ledger()[1] == {}, "DATA still resets on a rerun"
    assert out["miss_counts"] == {"tracked": 0, "at_threshold": 0, "reset": 1}


def test_a_backdated_anchor_reads_a_future_last_as_age_zero(monkeypatch, tmp_path):
    """The skip age is clamped at zero: a rerun anchored BEFORE the entry's
    ``last`` (a backdated replay, or a hand-edited future date) computes a
    negative age, which must never re-probe and never raise — a future ``last``
    is age 0, so the chronic name keeps its quiet."""
    _wire(monkeypatch, set(), tmp_path)
    future = str((pd.Timestamp(AS_OF) + pd.Timedelta(days=3)).date())
    entry = {"n": nightly.CHAIN_MISS_SKIP_AFTER, "last": future}
    _seed_ledger(misses={"TKR09": entry})
    f = _Fetcher(absent={"TKR09"})

    out = nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=1)

    assert "TKR09" not in f.asked_tickers, "a negative age is age 0: no re-probe"
    assert out["miss_skipped"]["tickers"] == 1
    assert nightly._read_chain_ledger()[1] == {"TKR09": entry}, \
        "the unasked name's count and its future last both stand"


def test_data_clears_the_count_and_removes_the_entry(monkeypatch, tmp_path):
    """One name that answers with a chain wipes its whole streak: the entry is
    gone, not zeroed, and the report counts the reset."""
    _wire(monkeypatch, set(), tmp_path)
    _seed_ledger(misses={"TKR09": {"n": nightly.CHAIN_MISS_SKIP_AFTER - 1,
                                   "last": AS_OF}})
    f = _Fetcher()

    out = nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=1)

    assert "TKR09" in f.asked_tickers, "three misses is under the threshold: keep asking"
    assert nightly._read_chain_ledger()[1] == {}, "the entry is dropped, not zeroed"
    assert out["miss_counts"] == {"tracked": 0, "at_threshold": 0, "reset": 1}


def test_a_name_held_for_one_session_is_data_not_a_miss(monkeypatch, tmp_path):
    """A pair already in the store IS a chain. Absent from the session the run
    asked for but held for the other one, the name is DATA and its streak
    clears — otherwise a name that lost a single night's chain would be skipped
    whole, which is the mistake the session-level rule made."""
    older = "2026-09-01"
    _wire(monkeypatch, {("TKR09", pd.Timestamp(older))}, tmp_path,
          stamps=[older, AS_OF])
    _seed_ledger(misses={"TKR09": {"n": nightly.CHAIN_MISS_SKIP_AFTER - 1,
                                   "last": AS_OF}})
    f = _Fetcher(absent={"TKR09"})

    out = nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=2)

    assert f.asked == [AS_OF, older], "the held pair is not bought; the rest is"
    assert f.asked_tickers.count("TKR09") == 1, "asked once, for the unheld session"
    assert nightly._read_chain_ledger()[1] == {}, "held elsewhere is a chain, not a miss"
    assert out["miss_counts"]["reset"] == 1


@pytest.mark.parametrize("kind", ["transient", "fetch-error", "bad-json",
                                  "wrong-shape", "junk-row"])
def test_a_call_that_never_answered_is_never_a_miss(monkeypatch, tmp_path, kind):
    """The count moves only on a definitive answer. A transport reset, a plain
    FetchError, a 200 that is not JSON, a 200 whose shape is wrong and a 200
    with junk rows all leave the streak exactly where it was, and the name
    stays owed the date."""
    kw = {
        "transient": {"fail": lambda p: RuntimeError("orats: connection reset")},
        "fetch-error": {"fail": lambda p: FetchError("orats hist/strikes: HTTP 503")},
        "bad-json": {"bad_json": lambda p: ValueError("no JSON object could be decoded")},
        "wrong-shape": {"body": lambda p: ["not", "a", "dict"]},
        "junk-row": {"body": lambda p: {"data": [{"ticker": "TKR00"}, 5]}},
    }[kind]
    _wire(monkeypatch, set(), tmp_path)
    before = {"TKR09": {"n": nightly.CHAIN_MISS_SKIP_AFTER - 1, "last": _days(1)}}
    _seed_ledger(misses=before)
    f = _Fetcher(**kw)

    out = nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=1)

    assert out["failed"], f"{kind} is still reported"
    assert nightly._read_chain_ledger()[1] == before, \
        f"a {kind} answer is no evidence of no chain: the count stands"
    assert out["miss_counts"]["tracked"] == 1, "one entry tracked, unmoved"
    assert out["miss_counts"]["reset"] == 0
    assert out["miss_skipped"]["tickers"] == 0, "and a name short of the bar is asked"
    assert nightly._read_chain_retry() == {AS_OF: set(TICKERS)}, \
        "and every name of the unanswered chunk stays owed the date"


def test_a_quota_cut_is_never_a_miss(monkeypatch, tmp_path):
    """QuotaExhausted re-raises out of the run, but the end-of-run persist is a
    finally: it still writes the ledger, and what it learned is nothing — the
    chunk it refused was asked and never answered, and the names past it were
    never asked at all."""
    _wire(monkeypatch, set(), tmp_path)
    before = {"TKR09": {"n": nightly.CHAIN_MISS_SKIP_AFTER - 1, "last": AS_OF}}
    _seed_ledger(misses=before)
    f = _Fetcher(fail=lambda p: QuotaExhausted("orats daily quota below the reserve floor"))

    with pytest.raises(QuotaExhausted):
        nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=1)

    assert nightly._read_chain_ledger()[1] == before, "the refused chunk counts as nothing"
    assert nightly._read_chain_retry() == {AS_OF: set(TICKERS)}, \
        "and the whole intended list stays owed, count or no count"


def test_a_skip_drops_only_that_name_from_the_heal_pass(monkeypatch, tmp_path):
    """Why the session-level rule went: a mostly-dead session still has
    partners that come back late. TKR09 is chronic and out of the batch; TKR08
    missed once, is asked beside its neighbours, and recovers on this run."""
    _wire(monkeypatch, set(), tmp_path)
    _seed_ledger(misses={"TKR09": {"n": nightly.CHAIN_MISS_SKIP_AFTER, "last": AS_OF},
                         "TKR08": {"n": 1, "last": AS_OF}})
    f = _Fetcher(absent={"TKR09"})

    out = nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=1)

    assert f.asked_tickers == [t for t in TICKERS if t != "TKR09"], \
        "the batch is the universe minus the one skipped name"
    assert out["calls"] == 1 and out["miss_skipped"]["tickers"] == 1
    assert nightly._read_chain_ledger()[1] == {
        "TKR09": {"n": nightly.CHAIN_MISS_SKIP_AFTER, "last": AS_OF}}, \
        "TKR08's recovery cleared it; the skipped name's count did not move"


@pytest.mark.parametrize(("age", "asked"), [
    pytest.param(3, False, id="3-days-quiet"),
    pytest.param(6, False, id="6-days-quiet"),
    pytest.param(7, True, id="7-days-reprobe"),
    pytest.param(8, True, id="8-days-reprobe"),
])
def test_the_reprobe_window_is_seven_days(monkeypatch, tmp_path, age, asked):
    """The skip is a week of quiet, not a ban: (anchor - last) >= 7 days asks
    the name again, anything younger leaves it alone."""
    _wire(monkeypatch, set(), tmp_path)
    entry = {"n": nightly.CHAIN_MISS_SKIP_AFTER, "last": _days(age)}
    _seed_ledger(misses={"TKR09": entry})
    f = _Fetcher(absent={"TKR09"})

    out = nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=1)

    assert ("TKR09" in f.asked_tickers) is asked, f"last {age} days old"
    assert out["miss_skipped"]["tickers"] == (0 if asked else 1)
    if asked:
        assert nightly._read_chain_ledger()[1] == {"TKR09": {
            "n": nightly.CHAIN_MISS_SKIP_AFTER + 1, "last": AS_OF}}, \
            "a probe that missed counts and re-stamps: another week of quiet"
    else:
        assert nightly._read_chain_ledger()[1] == {"TKR09": entry}, "the quiet stands"


def test_a_reprobe_that_returns_data_clears_the_count(monkeypatch, tmp_path):
    """The other half of the probe: a late chain resets the streak to nothing,
    however long it was."""
    _wire(monkeypatch, set(), tmp_path)
    _seed_ledger(misses={"TKR09": {"n": 9, "last": _days(8)}})
    f = _Fetcher()

    out = nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=1)

    assert "TKR09" in f.asked_tickers, "eight days old is past the re-probe window"
    assert nightly._read_chain_ledger()[1] == {}
    assert out["miss_counts"] == {"tracked": 0, "at_threshold": 0, "reset": 1}


def test_a_count_older_than_the_prune_window_is_dropped(monkeypatch, tmp_path):
    """A streak nobody has touched for two months is not a live count. A name
    still in the universe is re-probed long before that — every 7 days — and so
    either clears or re-stamps; the CHAIN_MISS_PRUNE_DAYS window is what drops
    the entries left behind when a ticker leaves the calendar frontier and is
    never asked again. Exactly 60 days is still inside it."""
    _wire(monkeypatch, set(), tmp_path)
    _seed_ledger(misses={"GONE1": {"n": nightly.CHAIN_MISS_SKIP_AFTER, "last": _days(61)},
                         "GONE2": {"n": nightly.CHAIN_MISS_SKIP_AFTER, "last": _days(60)},
                         "TKR09": {"n": nightly.CHAIN_MISS_SKIP_AFTER, "last": _days(30)}})
    f = _Fetcher(absent={"TKR09"})

    nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=1)

    assert "TKR09" in f.asked_tickers, "30 days old is past the 7-day window: re-probed"
    kept = nightly._read_chain_ledger()[1]
    assert "GONE1" not in kept, "a count 61 days behind the run is dropped"
    assert kept["GONE2"] == {"n": nightly.CHAIN_MISS_SKIP_AFTER, "last": _days(60)}, \
        "exactly 60 days is still inside the window"
    assert kept["TKR09"] == {"n": nightly.CHAIN_MISS_SKIP_AFTER + 1, "last": AS_OF}, \
        "and the probe that missed re-stamped rather than aged out"


def test_a_legacy_flat_ledger_reads_as_owed(monkeypatch, tmp_path):
    """The format the memo had before counts existed still loads: its top-level
    date keys ARE the owed section, nothing is skipped on no counts, and the
    older of the two dates still reaches the older-memo re-ask. The next write
    puts it in the new shape."""
    _wire(monkeypatch, set(), tmp_path, stamps=[AS_OF])
    older = _days(3)
    _seed_ledger(owed={AS_OF: ["TKR09"], older: ["TKR08"]}, legacy=True)

    assert nightly._read_chain_retry() == {AS_OF: {"TKR09"}, older: {"TKR08"}}
    assert nightly._read_chain_ledger()[1] == {}, "a flat file has no counts yet"

    f = _Fetcher()
    out = nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=1)

    assert "TKR09" in f.asked_tickers, "the heal session still re-asks its debt"
    assert out["older_retried"] == [{"session": older, "names": 1}], \
        "and the date past the window is re-asked from the same legacy file"
    assert json.loads(nightly.CHAIN_RETRY_PATH.read_text()) == {"owed": {}, "misses": {}}


def test_a_malformed_miss_entry_is_ignored(monkeypatch, tmp_path):
    """A count the run cannot read is no evidence to skip on: a junk entry is
    dropped for that ticker, never raised, never half-read — and the one
    readable entry still does its job."""
    junk = [f"JUNK{i}" for i in range(6)]
    broken = dict(zip(junk, [
        "not-a-dict",                       # a bare string
        {"last": AS_OF},                    # no count at all
        {"n": "4", "last": AS_OF},          # a count that is not an int
        {"n": 0, "last": AS_OF},            # a zero count (dropped by the schema)
        {"n": 4},                           # no run date
        {"n": 4, "last": AS_OF.replace("-", "")},  # a non-canonical date
    ]))
    universe = TICKERS + ["GOOD"] + junk
    _wire(monkeypatch, set(), tmp_path)
    _seed_ledger(owed={AS_OF: ["TKR00"]},
                 misses={**broken,
                         "GOOD": {"n": nightly.CHAIN_MISS_SKIP_AFTER, "last": AS_OF}})
    f = _Fetcher()

    out = nightly.refresh_forward_chains(universe, AS_OF, fetcher=f, sessions=1)

    assert f.asked_tickers == [t for t in sorted(universe) if t != "GOOD"], \
        "a junk count asks its ticker like any other name"
    assert out["miss_skipped"] == {"tickers": 1, "pairs": 1,
                                   "est_weekly_calls_saved": 0.7}
    assert nightly._read_chain_ledger()[1] == {
        "GOOD": {"n": nightly.CHAIN_MISS_SKIP_AFTER, "last": AS_OF}}, \
        "the unreadable entries are gone; the readable one is untouched"


def test_a_404_chunk_is_a_miss_for_every_name_in_it(monkeypatch, tmp_path):
    """The 404 arm already files the whole chunk as a definitive no-chain; the
    count follows exactly those names — not the chunks that answered, which are
    data — and the name still never enters the owed memo."""
    tickers = [f"TKR{i:02d}" for i in range(25)]
    _wire(monkeypatch, set(), tmp_path)
    f = _Fetcher(fail=lambda p: RuntimeError("orats hist/strikes returned HTTP 404")
                 if p["ticker"].split(",")[0] == "TKR10" else None)

    out = nightly.refresh_forward_chains(tickers, AS_OF, fetcher=f, sessions=1)

    missed = {t for t in tickers if "TKR10" <= t <= "TKR19"}
    assert set(out["missing"]) == missed, "the 404 chunk is the whole missing set"
    assert nightly._read_chain_ledger()[1] == {t: {"n": 1, "last": AS_OF} for t in missed}, \
        "every name in it is one miss deeper and no other name is"
    assert nightly._read_chain_retry() == {}, "a 404 is still never owed"


def test_a_name_with_data_in_one_session_and_a_404_in_another_is_data(
        monkeypatch, tmp_path):
    """The run's outcome is the STRONGEST answer it saw, not the last one: a
    name that 404s for the older session and answers for the newest is DATA, so
    its streak clears and the pair it holds stays held."""
    older = "2026-09-01"
    _wire(monkeypatch, set(), tmp_path, stamps=[older, AS_OF])
    _seed_ledger(misses={"TKR09": {"n": nightly.CHAIN_MISS_SKIP_AFTER - 1,
                                   "last": AS_OF}})
    f = _Fetcher(fail=lambda p: RuntimeError("orats hist/strikes returned HTTP 404")
                 if p["tradeDate"] == older else None)

    out = nightly.refresh_forward_chains(TICKERS, AS_OF, fetcher=f, sessions=2)

    assert f.asked == [AS_OF, older]
    assert out["missing"] == sorted(TICKERS), "the older session's 404 is still filed"
    assert nightly._read_chain_ledger()[1] == {}, \
        "data anywhere in the run wins: nothing is counted, the streak clears"
    assert out["miss_counts"] == {"tracked": 0, "at_threshold": 0, "reset": 1}
