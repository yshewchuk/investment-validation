"""Private user-act/spend primitive; no production caller or holdout-read capability."""
import json
import os
import pwd
import sqlite3
import sys

from engine.v2.foundation import (
    canonical_json,
    content_hash,
    format_timestamp,
    parse_timestamp,
    to_document,
)
from engine.v2.foundation.final_holdout import request_from_document, request_from_registration
from engine.v2.ops.catalog import transaction
from engine.v2.ops.errors import fail
from engine.v2.ops.native_registration import verified_registration_bytes


def _registered(conn, store, request):
    binding = verified_registration_bytes(conn, store, request.registration_run_id,
                                          expected_ref=request.registration_ref)
    if binding is None or request_from_registration(
            json.loads(binding), request.registration_run_id, request.registration_ref) != request:
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "final-read request differs from preregistration")


def authorize_final_holdout(conn, store, request, *, clock):
    """Record a fresh interactive user act and permanent spend, never a read grant."""
    try:
        return _authorize(conn, store, request, clock)
    except (ValueError, TypeError, KeyError):
        raise fail("VALIDATION_FAILED", "invalid final-read authorization metadata") from None
    except (OSError, sqlite3.Error, KeyboardInterrupt):
        raise fail("RESOURCE_UNAVAILABLE", "final-read authorization storage or terminal unavailable") from None


def _authorize(conn, store, request, clock):
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise fail("HOLDOUT_ACCESS_DENIED", "final-read authorization requires an interactive user")
    request = request_from_document(to_document(request))
    _registered(conn, store, request)
    encoded, digest = canonical_json(to_document(request)), content_hash(to_document(request))
    uid = os.getuid()
    try:
        username = pwd.getpwuid(uid).pw_name
    except KeyError:
        raise fail("RESOURCE_UNAVAILABLE", "current OS user record is unavailable") from None
    actor = f"{uid}:{username}"
    confirmation = f"AUTHORIZE {json.dumps(request.experiment_id)} {digest} AS {json.dumps(actor)}"
    sys.stdout.write(encoded + "\nThis permanently spends one final read; no expiry or interrupted-read continuation.\n")
    sys.stdout.flush()
    try:
        answer = input(f"Type exactly: {confirmation}\n")
    except (EOFError, KeyboardInterrupt):
        raise fail("HOLDOUT_ACCESS_DENIED", "final-read authorization was not confirmed") from None
    if answer != confirmation:
        raise fail("HOLDOUT_ACCESS_DENIED", "final-read authorization was not confirmed") from None
    with transaction(conn):
        _registered(conn, store, request)
        row = conn.execute("SELECT * FROM final_holdout_authorizations WHERE experiment_id=?",
                           (request.experiment_id,)).fetchone()
        if row is not None:
            if row["request_digest"] != digest or row["request_json"] != encoded:
                raise fail("IDEMPOTENCY_CONFLICT", "experiment final read is already spent for another request")
            if row["actor"].split(":", 1)[0] != str(uid):
                raise fail("HOLDOUT_ACCESS_DENIED", "only the original authorizing user may acknowledge this act")
            original_confirmation = (
                f"AUTHORIZE {json.dumps(request.experiment_id)} {digest} AS {json.dumps(row['actor'])}"
            )
            if (row["schema_version"] != "final_holdout_authorization.v1.0"
                    or row["registration_run_id"] != request.registration_run_id
                    or row["confirmation"] != original_confirmation):
                raise ValueError
            parse_timestamp(row["authorized_at"])
            return dict(row)
        stamp = format_timestamp(clock.now())
        conn.execute(
            "INSERT INTO final_holdout_authorizations "
            "(experiment_id, schema_version, registration_run_id, request_digest, request_json, actor, authorized_at, confirmation) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (request.experiment_id, "final_holdout_authorization.v1.0",
             request.registration_run_id, digest, encoded, actor, stamp, confirmation),
        )
        row = conn.execute("SELECT * FROM final_holdout_authorizations WHERE experiment_id=?",
                           (request.experiment_id,)).fetchone()
    return dict(row)
