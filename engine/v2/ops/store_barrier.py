"""Cooperative legacy read/write barrier with input mutation detection — §9.2.

The barrier is catalog state, not filesystem MVCC. A writer takes the domain's
exclusive lease inside the claim transaction and marks the domain dirty; only a
verified successful write clears it, so a partially failed legacy rebuild keeps
readers out until recovery and its existing validation succeed. Readers hold
shared leases over their declared read set, pinned by content hash while the
lease is held: inputs that change between pin and commit refuse reuse and block
the commit instead of mixing versions.

Functions suffixed ``_in`` run inside the CALLER'S transaction (the claim or
commit transaction is the authority); the unsuffixed wrappers open one and
verify the attempt fence first.
"""
from __future__ import annotations

import json

from engine.v2.contracts import LegacyInputManifest, QueueReason
from engine.v2.foundation import from_document, safe_relative_path
from engine.v2.ops.catalog import dumps, transaction
from engine.v2.ops.errors import fail
from engine.v2.ops.fingerprints import file_hash
from engine.v2.ops.lifecycle import verify_fence

__all__ = ["acquire", "acquire_in", "confirm_read_set", "domains_of", "lease_reason",
           "pin_files", "pin_read_set", "read_set_complete", "verified_write",
           "verified_write_in", "verify_files"]


def domains_of(registry, kind_name):
    """The declared store domains of a registered kind; () without a registry."""
    if registry is None:
        return ()
    return registry.get(kind_name).store_domains


def lease_reason(conn, domains):
    """Read-only claim-loop precheck: why these domains cannot be leased now."""
    for domain, mode in domains:
        row = conn.execute("SELECT dirty FROM store_domains WHERE domain=?", (domain,)).fetchone()
        if mode == "read" and row is not None and row[0]:
            return QueueReason(code="STORE_RECOVERY", reconsider="recovery_validation")
        active = conn.execute("SELECT mode FROM store_leases WHERE domain=? AND released_at IS NULL",
                              (domain,)).fetchall()
        if active and (mode == "write" or any(item[0] == "write" for item in active)):
            return QueueReason(code="STORE_LEASE_HELD", needed={"store_slots": 1},
                               available={"store_slots": 0},
                               reconsider="store_lease_release:" + domain)
    return None


def acquire_in(conn, attempt_id, domains):
    """Insert leases inside the caller's (claim) transaction; recheck conflicts."""
    for domain, mode in domains:
        if mode not in ("read", "write"):
            raise fail("INVALID_REQUEST", "unknown legacy lease mode")
        active = conn.execute("SELECT mode FROM store_leases WHERE domain=? AND released_at IS NULL",
                              (domain,)).fetchall()
        if active and (mode == "write" or any(item[0] == "write" for item in active)):
            raise fail("RESOURCE_UNAVAILABLE", "legacy input domain is leased")
        conn.execute("INSERT INTO store_leases(domain,attempt_id,mode) VALUES (?,?,?)",
                     (domain, attempt_id, mode))
        if mode == "write":
            conn.execute("INSERT INTO store_domains VALUES (?,1) ON CONFLICT(domain) "
                         "DO UPDATE SET dirty=1", (domain,))


def acquire(conn, claim, domain, mode, *, clock):
    """Standalone fenced acquisition for coordinator-side manual paths."""
    with transaction(conn):
        verify_fence(conn, claim.attempt_id, claim.fence, clock.now())
        acquire_in(conn, claim.attempt_id, ((domain, mode),))


def verified_write_in(conn, attempt_id, domain):
    """Inside the commit transaction: a validated write clears the dirty flag."""
    lease = conn.execute("SELECT mode FROM store_leases WHERE domain=? AND attempt_id=? "
                         "AND released_at IS NULL", (domain, attempt_id)).fetchone()
    if not lease or lease[0] != "write":
        raise fail("LEASE_LOST", "no exclusive legacy lease")
    conn.execute("UPDATE store_domains SET dirty=0 WHERE domain=?", (domain,))


def verified_write(conn, claim, domain, *, clock):
    with transaction(conn):
        verify_fence(conn, claim.attempt_id, claim.fence, clock.now())
        verified_write_in(conn, claim.attempt_id, domain)


def pin_files(root, paths):
    result = {}
    for rel in paths:
        safe_relative_path(rel)
        path = root / rel
        if path.is_symlink() or not path.is_file():
            raise fail("INPUT_CHANGED", "legacy input is missing or indirect")
        result[rel] = {"content_hash": file_hash(path), "byte_size": path.stat().st_size}
    return result


def verify_files(root, manifest):
    if pin_files(root, manifest) != manifest:
        raise fail("INPUT_CHANGED", "pinned legacy inputs changed")


def pin_read_set(conn, attempt_id, manifest, root):
    """Pin the declared legacy read set under this attempt; bytes must match."""
    parsed = from_document(LegacyInputManifest, manifest)
    expected = {ref.path: {"content_hash": ref.content_hash, "byte_size": ref.byte_size}
                for ref in parsed.file_refs}
    if len(expected) != len(parsed.file_refs):
        raise fail("INVALID_REQUEST", "manifest names one path twice")
    if pin_files(root, sorted(expected)) != expected:
        raise fail("INPUT_CHANGED", "legacy inputs differ from the declared manifest")
    with transaction(conn):
        conn.execute("INSERT INTO store_read_pins(attempt_id,manifest_json,read_set_complete) "
                     "VALUES (?,?,?)",
                     (attempt_id, dumps(expected), int(bool(parsed.read_set_complete))))


def read_set_complete(conn, attempt_id):
    """True only for a pinned, fully declared read set; None when nothing is pinned."""
    row = conn.execute("SELECT read_set_complete FROM store_read_pins WHERE attempt_id=?",
                       (attempt_id,)).fetchone()
    return None if row is None else bool(row[0])


def confirm_read_set(conn, attempt_id, root):
    """Before any commit: the pinned inputs still hold the pinned bytes."""
    row = conn.execute("SELECT manifest_json FROM store_read_pins WHERE attempt_id=?",
                       (attempt_id,)).fetchone()
    if row is None:
        return
    verify_files(root, json.loads(row[0]))
