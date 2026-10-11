"""Private user-act/spend primitive; no production caller or holdout-read capability."""
import json
import os
import pwd
import sqlite3
import sys

from engine.v2.contracts import ProcessIdentity
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
from engine.v2.ops.executor_watchdog import is_alive, process_info
from engine.v2.ops.native_registration import verified_registration_bytes
from engine.v2.ops.recovery import read_boot_id


def _registered(conn, store, request):
    binding = verified_registration_bytes(conn, store, request.registration_run_id, expected_ref=request.registration_ref)
    if binding is None or request_from_registration(json.loads(binding),
                                                    request.registration_run_id, request.registration_ref) != request:
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "final-read request differs from preregistration")


def _validate_authorization_evidence(row, request, digest, encoded):
    actor = row["actor"]
    if not isinstance(actor, str) or not actor or ":" not in actor:
        raise ValueError
    confirmation = (
        f"AUTHORIZE {json.dumps(request.experiment_id)} {digest} AS {json.dumps(actor)}"
    )
    if (row["schema_version"] != "final_holdout_authorization.v1.0"
            or row["registration_run_id"] != request.registration_run_id
            or row["request_digest"] != digest or row["request_json"] != encoded
            or row["confirmation"] != confirmation):
        raise ValueError
    parse_timestamp(row["authorized_at"])


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
            if (not isinstance(row["actor"], str)
                    or row["actor"].split(":", 1)[0] != str(uid)):
                raise fail("HOLDOUT_ACCESS_DENIED", "only the original authorizing user may acknowledge this act")
            _validate_authorization_evidence(row, request, digest, encoded)
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


def _validate_experiment_id(experiment_id):
    if (type(experiment_id) is not str or not experiment_id
            or experiment_id != experiment_id.strip()):
        raise fail("VALIDATION_FAILED", "invalid final-read experiment ID")


def _validate_publication_fence_inputs(experiment_id, request_digest, fence):
    _validate_experiment_id(experiment_id)
    prefix = "sha256:"
    if type(request_digest) is not str or not request_digest.startswith(prefix):
        raise fail("VALIDATION_FAILED", "invalid final-read request digest")
    digest_hex = request_digest[len(prefix):]
    if (len(digest_hex) != 64
            or any(character not in "0123456789abcdef" for character in digest_hex)):
        raise fail("VALIDATION_FAILED", "invalid final-read request digest")
    if type(fence) is not int or not 0 < fence <= (2**63 - 1):
        raise fail("VALIDATION_FAILED", "invalid final-read publication fence")


def _latest_event(conn, experiment_id):
    return conn.execute("SELECT * FROM final_holdout_state_events WHERE experiment_id=? "
                        "ORDER BY event_id DESC LIMIT 1", (experiment_id,)).fetchone()


def _owner_identity(event):
    values = (event["owner_boot_id"], event["owner_pid"],
              event["owner_start_ticks"], event["owner_process_group"])
    if any((not isinstance(values[0], str), not values[0],
            any(type(value) is not int or value <= 0 for value in values[1:]))):
        raise ValueError
    return ProcessIdentity(boot_id=values[0], pid=values[1], start_ticks=values[2],
                           process_group=values[3])


def _state_action_context(conn, store, request, action):
    request = request_from_document(to_document(request))
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise fail("HOLDOUT_ACCESS_DENIED", "final-read state change requires the authorizing user")
    authorization = conn.execute("SELECT * FROM final_holdout_authorizations WHERE experiment_id=?",
                                 (request.experiment_id,)).fetchone()
    if authorization is None:
        raise fail("HOLDOUT_ACCESS_DENIED", "final-read authorization is required")
    uid = os.getuid()
    if (not isinstance(authorization["actor"], str)
            or authorization["actor"].split(":", 1)[0] != str(uid)):
        raise fail("HOLDOUT_ACCESS_DENIED", "only the authorizing user may change final-read state")
    _registered(conn, store, request)
    encoded, digest = canonical_json(to_document(request)), content_hash(to_document(request))
    _validate_authorization_evidence(authorization, request, digest, encoded)
    try:
        username = pwd.getpwuid(uid).pw_name
    except KeyError:
        raise fail("RESOURCE_UNAVAILABLE", "current OS user record is unavailable") from None
    actor = f"{uid}:{username}"
    phrase = f'{action} {json.dumps(request.experiment_id)} {digest} AS {json.dumps(actor)}'
    sys.stdout.write(encoded + "\n")
    sys.stdout.flush()
    try:
        answer = input(f"Type exactly: {phrase}\n")
    except (EOFError, KeyboardInterrupt):
        raise fail("HOLDOUT_ACCESS_DENIED", "final-read state change was not confirmed") from None
    if answer != phrase:
        raise fail("HOLDOUT_ACCESS_DENIED", "final-read state change was not confirmed")
    return request, authorization, digest, actor, phrase


def start_final_holdout_read(conn, store, request, *, clock):
    """Atomically consume the one read fence before any outcome access."""
    try:
        request, authorization, digest, actor, phrase = _state_action_context(conn, store, request, "START")
        identity, *_ = process_info(os.getpid(), read_boot_id())
        if not isinstance(identity.boot_id, str) or not identity.boot_id or any(
                type(value) is not int or value <= 0 for value in
                (identity.pid, identity.start_ticks, identity.process_group)):
            raise fail("VALIDATION_FAILED", "final-read owner identity is malformed")
        with transaction(conn):
            _registered(conn, store, request)
            current = conn.execute("SELECT * FROM final_holdout_authorizations WHERE experiment_id=?",
                                   (request.experiment_id,)).fetchone()
            if current is None:
                raise fail("VALIDATION_FAILED", "final-read authorization changed before start")
            if any((current["request_digest"] != digest,
                    current["request_json"] != authorization["request_json"])):
                raise fail("VALIDATION_FAILED", "final-read authorization changed before start")
            latest = _latest_event(conn, request.experiment_id)
            if latest is not None:
                if latest["request_digest"] != digest or latest["state"] not in (
                    "read_started", "revoked", "indeterminate"
                ):
                    raise fail("VALIDATION_FAILED", "final-read state history is inconsistent")
                if latest["state"] == "read_started":
                    raise fail("HOLDOUT_ACCESS_DENIED", "the final read is already in progress")
                raise fail("HOLDOUT_ACCESS_DENIED", "the final read is already spent or revoked")
            stamp = format_timestamp(clock.now())
            conn.execute(
                "INSERT INTO final_holdout_state_events "
                "(experiment_id,request_digest,state,fence,owner_boot_id,owner_pid,owner_start_ticks,owner_process_group,actor,occurred_at,confirmation) "
                "VALUES (?,?, 'read_started',1,?,?,?,?,?,?,?)",
                (request.experiment_id, digest, identity.boot_id, identity.pid,
                 identity.start_ticks, identity.process_group, actor, stamp, phrase),
            )
            return dict(_latest_event(conn, request.experiment_id))
    except (ValueError, TypeError, KeyError, IndexError):
        raise fail("VALIDATION_FAILED", "invalid final-read state metadata") from None
    except (OSError, sqlite3.Error, KeyboardInterrupt):
        raise fail("RESOURCE_UNAVAILABLE", "final-read state storage or process identity unavailable") from None


def revoke_final_holdout(conn, store, request, *, clock):
    """Append a user revocation; it never refunds the permanent experiment spend."""
    try:
        request, authorization, digest, actor, phrase = _state_action_context(conn, store, request, "REVOKE")
        with transaction(conn):
            _registered(conn, store, request)
            current = conn.execute("SELECT * FROM final_holdout_authorizations WHERE experiment_id=?",
                                   (request.experiment_id,)).fetchone()
            if (current is None or current["request_digest"] != digest
                    or current["request_json"] != authorization["request_json"]):
                raise fail("VALIDATION_FAILED", "final-read authorization changed before revocation")
            latest = _latest_event(conn, request.experiment_id)
            if latest is not None:
                if latest["request_digest"] != digest:
                    raise fail("VALIDATION_FAILED", "final-read state history is inconsistent")
                if latest["state"] == "revoked":
                    return dict(latest)
                if latest["state"] == "indeterminate":
                    raise fail("HOLDOUT_ACCESS_DENIED", "the final read is already spent")
                if latest["state"] != "read_started":
                    raise fail("VALIDATION_FAILED", "final-read state history is inconsistent")
            fence = 1
            if latest is not None:
                if latest["fence"] >= 2**63 - 1:
                    raise fail("VALIDATION_FAILED", "final-read revocation fence is exhausted")
                fence = int(latest["fence"]) + 1
            stamp = format_timestamp(clock.now())
            conn.execute("INSERT INTO final_holdout_state_events (experiment_id,request_digest,state,fence,actor,occurred_at,confirmation) "
                         "VALUES (?,?, 'revoked',?,?,?,?)", (request.experiment_id, digest, fence, actor, stamp, phrase))
            return dict(_latest_event(conn, request.experiment_id))
    except (ValueError, TypeError, KeyError, IndexError):
        raise fail("VALIDATION_FAILED", "invalid final-read state metadata") from None
    except (OSError, sqlite3.Error, KeyboardInterrupt):
        raise fail("RESOURCE_UNAVAILABLE", "final-read state storage or terminal unavailable") from None


def final_holdout_fence_current(conn, experiment_id, request_digest, fence):
    """Whether a started read still owns the current publication fence."""
    _validate_publication_fence_inputs(experiment_id, request_digest, fence)
    try:
        latest = _latest_event(conn, experiment_id)
        if latest is None or latest["state"] != "read_started":
            return False
        _owner_identity(latest)
        return bool(latest["request_digest"] == request_digest and latest["fence"] == fence)
    except (ValueError, TypeError, KeyError, IndexError):
        raise fail("VALIDATION_FAILED", "final-read owner identity is malformed") from None
    except (OSError, sqlite3.Error):
        raise fail("RESOURCE_UNAVAILABLE", "final-read fence storage unavailable") from None


def recover_interrupted_final_holdout_read(conn, experiment_id, *, clock):
    """Mark a read indeterminate only after its exact owner process is proven gone."""
    _validate_experiment_id(experiment_id)
    try:
        latest = _latest_event(conn, experiment_id)
        if latest is None:
            raise fail("STALE_EXPECTATION", "no started final read awaits recovery")
        if latest["state"] != "read_started":
            raise fail("STALE_EXPECTATION", "no started final read awaits recovery")
        authorization = conn.execute("SELECT * FROM final_holdout_authorizations WHERE experiment_id=?",
                                     (experiment_id,)).fetchone()
        if authorization is None:
            raise fail("VALIDATION_FAILED", "final-read recovery evidence is inconsistent")
        request = request_from_document(json.loads(authorization["request_json"]))
        encoded = canonical_json(to_document(request))
        digest = content_hash(to_document(request))
        _validate_authorization_evidence(authorization, request, digest, encoded)
        if request.experiment_id != experiment_id or digest != latest["request_digest"]:
            raise fail("VALIDATION_FAILED", "final-read recovery evidence is inconsistent")
        identity = _owner_identity(latest)
        boot_id = read_boot_id()
        proven_dead = identity.boot_id != boot_id or not is_alive(identity, boot_id)
        if not proven_dead:
            raise fail("HOLDOUT_ACCESS_DENIED", "the final-read owner is still alive")
        with transaction(conn):
            current = _latest_event(conn, experiment_id)
            if current is None or (current["state"], current["event_id"], current["fence"]) != (
                "read_started", latest["event_id"], latest["fence"]
            ):
                raise fail("STALE_EXPECTATION", "the final-read fence changed during recovery")
            stamp = format_timestamp(clock.now())
            conn.execute("INSERT INTO final_holdout_state_events (experiment_id,request_digest,state,fence,occurred_at) "
                         "VALUES (?,?,'indeterminate',?,?)", (experiment_id, latest["request_digest"], latest["fence"], stamp))
            return dict(_latest_event(conn, experiment_id))
    except (ValueError, TypeError, KeyError, IndexError):
        raise fail("VALIDATION_FAILED", "invalid final-read recovery metadata") from None
    except (OSError, sqlite3.Error, KeyboardInterrupt):
        raise fail("RESOURCE_UNAVAILABLE", "final-read recovery evidence unavailable") from None
