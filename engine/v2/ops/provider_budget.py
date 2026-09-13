"""Account-wide leases, conservative call accounting and durable source backoff."""
from __future__ import annotations

from datetime import timedelta

from engine.v2.foundation import format_timestamp
from engine.v2.ops.catalog import transaction
from engine.v2.ops.errors import fail
from engine.v2.ops.lifecycle import verify_fence


def configure_account(conn, account, generation, remaining, live_reserve):
    if min(remaining, live_reserve) < 0:
        raise fail("INVALID_REQUEST", "negative provider budget")
    with transaction(conn):
        conn.execute("INSERT INTO provider_accounts(account,generation,remaining,live_reserve) "
                     "VALUES (?,?,?,?)", (account, generation, remaining, live_reserve))


def reserve(conn, claim, account, calls, *, clock):
    if calls <= 0:
        raise fail("INVALID_REQUEST", "provider reservation must include every planned retry")
    with transaction(conn):
        verify_fence(conn, claim.attempt_id, claim.fence, clock.now())
        row = conn.execute("SELECT * FROM provider_accounts WHERE account = ?", (account,)).fetchone()
        if row is None or row["blocked_code"]:
            raise fail("CREDENTIAL_INVALID", "provider account needs operator action")
        if row["next_eligible_at"] and row["next_eligible_at"] > format_timestamp(clock.now()):
            raise fail("RATE_LIMITED", "provider account is in backoff")
        active = conn.execute("SELECT 1 FROM provider_reservations WHERE account = ? "
                              "AND released_at IS NULL", (account,)).fetchone()
        if active or calls > row["remaining"] - row["live_reserve"]:
            raise fail("RESOURCE_UNAVAILABLE", "provider lease or call budget unavailable")
        conn.execute("INSERT INTO provider_reservations(account,attempt_id,fence,reserved_calls) "
                     "VALUES (?,?,?,?)", (account, claim.attempt_id, claim.fence, calls))


def before_request(conn, claim, account, *, clock):
    with transaction(conn):
        verify_fence(conn, claim.attempt_id, claim.fence, clock.now())
        row = conn.execute("SELECT r.*,a.blocked_code,a.next_eligible_at FROM provider_reservations r "
                           "JOIN provider_accounts a USING(account) WHERE account = ? "
                           "AND attempt_id = ? AND released_at IS NULL",
                           (account, claim.attempt_id)).fetchone()
        if row is None or row["blocked_code"]:
            raise fail("CREDENTIAL_INVALID", "provider lease is unavailable")
        if row["next_eligible_at"] and row["next_eligible_at"] > format_timestamp(clock.now()):
            raise fail("RATE_LIMITED", "account backoff has not elapsed")
        if row["used_calls"] >= row["reserved_calls"]:
            raise fail("RESOURCE_UNAVAILABLE", "provider retry budget exhausted")
        conn.execute("UPDATE provider_reservations SET used_calls = used_calls + 1 "
                     "WHERE account = ? AND attempt_id = ?", (account, claim.attempt_id))
        conn.execute("UPDATE provider_accounts SET remaining = MAX(0,remaining-1), uncertain = 1 "
                     "WHERE account = ?", (account,))


def record_response(conn, account, status, *, clock, remaining=None, empty=False, final=True):
    code = {401: "CREDENTIAL_INVALID", 403: "CREDENTIAL_INVALID",
            404: "SOURCE_NOT_FOUND", 429: "RATE_LIMITED"}.get(status)
    if status >= 500:
        code = "TRANSIENT_SOURCE"
    if 200 <= status < 300:
        code = "SOURCE_NOT_FINAL" if not final else ("SOURCE_EMPTY" if empty else None)
    with transaction(conn):
        if remaining is not None:
            if remaining < 0:
                raise fail("INVALID_REQUEST", "negative quota header")
            conn.execute("UPDATE provider_accounts SET remaining = ?, uncertain = 0 WHERE account = ?",
                         (remaining, account))
        if code == "CREDENTIAL_INVALID":
            conn.execute("UPDATE provider_accounts SET blocked_code = ? WHERE account = ?", (code, account))
        if code == "RATE_LIMITED":
            stamp = format_timestamp(clock.now() + timedelta(seconds=65))
            conn.execute("UPDATE provider_accounts SET next_eligible_at = ? WHERE account = ?", (stamp, account))
    return code
