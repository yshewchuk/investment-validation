#!/usr/bin/env python3
"""P7-1 check: validate one Phase 7 candidate/authority manifest. Stdlib only.

The guide (``guides/rearchitecture_phase7_cutover.md``) makes P7-1 a manifest
of exact identities — code/config, data snapshot, model deployment, native
job graph, consumer inventory, old and proposed schedule/writer/credential
ownership, and retained old deployment refs — whose acceptance bar is *no
implicit latest dependency and no unaccounted scheduled writer or background
job*. This checker validates one such manifest against the referenced files
it can reach. It is read-only: it never writes authority, production
pointers, credentials or release state, and a green result validates the
manifest's internal consistency and referenced evidence, **not** Phase 7
readiness, which requires real dated evidence this repo does not ship. The
checker never queries a live scheduler or deployment system, so it can never
claim the referenced job graph or consumer inventory is a *complete* real-world
inventory — only that the manifest agrees with the files it references.

Input manifest schema ``phase7_candidate_manifest.v1.0``
--------------------------------------------------------
Every key listed is required; an unknown key at any strict level is a
finding (a typo must not silently disable a control). All identity strings
must be exact: an empty value, or one of ``latest``/``current``/``newest``/
``head``/``main``/``master``/``trunk``/``stable``/``prod``/``production``/
``live``/``default`` — alone, as a ``:``-suffix (``deploy:current``,
``image:stable``), or as the last path segment — is rejected; trailing
``/`` and ``:`` separators are normalized away first, so ``refs/heads/main/``
is caught exactly like ``refs/heads/main``.

``candidate``: ``{"commit": "<40- or 64-hex git sha>", "config": [{"name", "path",
"sha256"}]}`` — at least one config identity; every referenced file must
exist, hash to its recorded ``sha256`` (64-hex) and parse.

``data_snapshot``: ``{"snapshot_id", "sha256", "path"}`` — one pinned
snapshot, never a floating pointer. The referenced snapshot document must
parse to a JSON object that declares the same ``snapshot_id``; a
non-object, a document that omits the key or names a different snapshot
fails closed.

``model_deployment``: ``{"release_id", "path", "sha256"}`` — the referenced
release manifest must parse to a JSON object that declares exactly the same
``release_id``; an object omitting ``release_id`` is a finding, not a pass.

``job_graph``: ``{"path", "sha256", "jobs": [{"id", "schedule", "kind",
"owner"}]}`` — the accounting of every job. ``kind`` is one of
``scheduled_writer``, ``background`` or ``reader``, and each ``schedule``
must be a string. The referenced graph document must parse to
``{"jobs": [{"id", "schedule", "writes_official": bool}]}`` where every row
carries a string ``id``, a string ``schedule`` and a real JSON boolean
``writes_official`` — missing, null or string-coerced values (``"false"``)
are findings, never silently converted. The two job sets must agree exactly
(id and schedule per id), so a scheduled job that exists but is not
accounted for, or an accounted job the graph does not run, both fail.

``consumer_inventory``: ``{"path", "sha256", "consumers": ["<name>", ...]}``
— a non-empty list of name strings; a dict, list or other nonstring entry
is a named finding. The referenced document must contain every declared
consumer name (as an ``id``/``name``/``consumer`` value anywhere in it).

``authority``: ``{"old": {"schedule", "writer", "credential_owner"},
"proposed": {same}, "retained_old_deployment": {"ref", "rollback_owner"}}``
— both sides of the switch must be owned explicitly, and the old deployment
must be named for retention with a rollback owner. Each side's
``schedule``/``writer`` pair is validated together against that side's own
job — the checker does not compare the two sides' schedules as a union, so
swapping the old and proposed schedule values fails. At most one
``scheduled_writer`` job per side (a duplicate official writer fails);
every ``scheduled_writer`` job's owner must be one declared writer; a
``background``/``reader`` job the graph marks ``writes_official`` is a
hidden competing writer; two non-reader jobs on one schedule is a
duplicate schedule; the old and proposed sides must differ on all three
roles (a manifest that changes nothing cannot be a switch).

``phase_evidence``: ``[{"phase", "schema_version", "status", "path",
"sha256"}]`` — one row per required phase ``3B``, ``4``, ``5``, ``6``
(the 3A preview cannot substitute: a row's ``schema_version`` must start
with ``phase<normalized phase>``). Each referenced evidence file must
exist, hash correctly, parse as a JSON *object* (a null, list or scalar
document is a finding, not a skipped check), carry the declared
``schema_version``, report exactly the declared ``status`` (a truthy
boolean in the file is never consulted), bind the manifest's ``commit``:
at least one of the commit keys must be present and every commit key
that is present must be a string equal to that commit — evidence naming
no binding, a null or non-string binding, or one alias contradicting
another fails closed — and — for Phase 5 — bind the manifest's
``release_id``, likewise required.

Usage::

    python3 checks/phase7_candidate_manifest.py MANIFEST.json \\
        [--root REFERENCE_ROOT] [--json]

Exit 0 = no findings, 1 = findings, 2 = the manifest itself cannot be read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]

MANIFEST_SCHEMA = "phase7_candidate_manifest.v1.0"
JOB_KINDS = ("scheduled_writer", "background", "reader")
REQUIRED_PHASES = ("3B", "4", "5", "6")

#: Identity values that mean "whatever is live now" — exactly what P7-1
#: forbids ("no implicit latest dependency").
IMPLICIT_REFS = frozenset({"latest", "current", "newest", "head", "main", "master",
                           "trunk", "stable", "prod", "production", "live", "default"})

SCHEMA_VERSION = "P7_SCHEMA_VERSION"
UNKNOWN_KEY = "P7_UNKNOWN_KEY"
MISSING_FIELD = "P7_MISSING_FIELD"
BAD_LIST = "P7_BAD_LIST"
BAD_CONSUMER = "P7_BAD_CONSUMER"
BAD_SCHEDULE = "P7_BAD_SCHEDULE"
BAD_JOB_KIND = "P7_BAD_JOB_KIND"
BAD_PHASE = "P7_BAD_PHASE"
LATEST_REF = "P7_LATEST_REF"
BAD_HASH = "P7_BAD_HASH"
BAD_CODE_IDENTITY = "P7_BAD_CODE_IDENTITY"
DUPLICATE = "P7_DUPLICATE"
REF_MISSING = "P7_REF_MISSING"
REF_HASH_MISMATCH = "P7_REF_HASH_MISMATCH"
REF_MALFORMED = "P7_REF_MALFORMED"
REF_MISMATCH = "P7_REF_MISMATCH"
UNACCOUNTED_JOB = "P7_UNACCOUNTED_JOB"
PHANTOM_JOB = "P7_PHANTOM_JOB"
WRITER_CONFLICT = "P7_WRITER_CONFLICT"
DUPLICATE_OFFICIAL_WRITER = "P7_DUPLICATE_OFFICIAL_WRITER"
WRITER_NO_JOB = "P7_WRITER_NO_JOB"
WRITER_NO_SWITCH = "P7_WRITER_NO_SWITCH"
CONSUMER_UNACCOUNTED = "P7_CONSUMER_UNACCOUNTED"
MISSING_PHASE_EVIDENCE = "P7_MISSING_PHASE_EVIDENCE"
EVIDENCE_STALE = "P7_EVIDENCE_STALE"
OWNERSHIP_INCOMPLETE = "P7_OWNERSHIP_INCOMPLETE"
OLD_DEPLOYMENT_MISSING = "P7_OLD_DEPLOYMENT_MISSING"

FINDING_CODES = (
    SCHEMA_VERSION, UNKNOWN_KEY, MISSING_FIELD, BAD_LIST, BAD_CONSUMER, BAD_SCHEDULE,
    BAD_JOB_KIND, BAD_PHASE,
    LATEST_REF, BAD_HASH, BAD_CODE_IDENTITY, DUPLICATE, REF_MISSING, REF_HASH_MISMATCH,
    REF_MALFORMED, REF_MISMATCH, UNACCOUNTED_JOB, PHANTOM_JOB, WRITER_CONFLICT,
    DUPLICATE_OFFICIAL_WRITER, WRITER_NO_JOB, WRITER_NO_SWITCH, CONSUMER_UNACCOUNTED,
    MISSING_PHASE_EVIDENCE, EVIDENCE_STALE, OWNERSHIP_INCOMPLETE, OLD_DEPLOYMENT_MISSING,
)

_TOP_KEYS = ("schema_version", "candidate", "data_snapshot", "model_deployment",
             "job_graph", "consumer_inventory", "authority", "phase_evidence")
_CANDIDATE_KEYS = ("commit", "config")
_CONFIG_KEYS = ("name", "path", "sha256")
_SNAPSHOT_KEYS = ("snapshot_id", "sha256", "path")
_DEPLOYMENT_KEYS = ("release_id", "path", "sha256")
_GRAPH_KEYS = ("path", "sha256", "jobs")
_JOB_KEYS = ("id", "schedule", "kind", "owner")
_INVENTORY_KEYS = ("path", "sha256", "consumers")
_SIDE_KEYS = ("schedule", "writer", "credential_owner")
_AUTHORITY_KEYS = ("old", "proposed", "retained_old_deployment")
_RETAINED_KEYS = ("ref", "rollback_owner")
_EVIDENCE_KEYS = ("phase", "schema_version", "status", "path", "sha256")


# --------------------------------------------------------------------------
# findings
# --------------------------------------------------------------------------


class ManifestUnreadable(Exception):
    """The manifest file itself cannot be read or parsed."""


def _finding(code: str, subject: str, detail: str) -> dict:
    return {"code": code, "subject": subject, "detail": detail}


class _Findings:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, code: str, subject: str, detail: str) -> None:
        self.rows.append(_finding(code, subject, detail))

    def sorted_rows(self) -> list[dict]:
        return sorted(self.rows, key=lambda r: (r["code"], r["subject"], r["detail"]))


_COMMIT_KEYS = ("commit", "candidate_commit", "code_commit", "git_commit")


# --------------------------------------------------------------------------
# shape + identity helpers
# --------------------------------------------------------------------------


def _is_implicit(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return True
    text = value.strip().lower()
    while text and text[-1] in "/: ":
        text = text[:-1].rstrip()
    if not text or text in IMPLICIT_REFS:
        return True
    tail = text.rsplit(":", 1)[-1].rstrip("/").rstrip(":")
    if tail in IMPLICIT_REFS:
        return True
    if bool(text.rsplit("/", 1)[-1]) and text.rsplit("/", 1)[-1] in IMPLICIT_REFS:
        return True
    return "/latest" in text


def _identity(findings: _Findings, subject: str, field: str, value: Any) -> Any:
    if _is_implicit(value):
        findings.add(LATEST_REF, subject, f"{field}={value!r} is missing or an implicit "
                                          "latest/current ref; P7-1 requires exact identity")
        return None
    return value


def _sha256(findings: _Findings, subject: str, field: str, value: Any) -> Any:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef"
                                                             for c in value.lower()):
        findings.add(BAD_HASH, subject, f"{field}={value!r} is not a 64-hex sha256")
        return None
    return value.lower()


def _strict(findings: _Findings, obj: Mapping, subject: str, required: tuple[str, ...],
            allowed: tuple[str, ...]) -> dict:
    if not isinstance(obj, Mapping):
        findings.add(MISSING_FIELD, subject, "must be an object")
        return {}
    for key in required:
        if key not in obj:
            findings.add(MISSING_FIELD, subject, f"required key {key!r} is absent")
    for key in obj:
        if key not in allowed:
            findings.add(UNKNOWN_KEY, subject, f"unknown key {key!r} (typos must not "
                                               "silently disable a control)")
    return dict(obj)


def _obj_list(findings: _Findings, value: Any, subject: str) -> list:
    if not isinstance(value, list) or not value:
        findings.add(BAD_LIST, subject, "must be a non-empty list")
        return []
    return value


# --------------------------------------------------------------------------
# referenced-file verification (hash first, then parse; booleans are never
# trusted — every check reads bytes)
# --------------------------------------------------------------------------


def _resolve(root: Path, path: Any, findings: _Findings, subject: str) -> Path | None:
    raw = _identity(findings, subject, "path", path)
    if raw is None:
        return None
    target = Path(raw) if Path(raw).is_absolute() else root / raw
    if not target.is_file():
        findings.add(REF_MISSING, subject, f"referenced file {str(target)!s} does not exist")
        return None
    return target.resolve()


def _read_verified(findings: _Findings, subject: str, path: Any, digest: Any,
                   root: Path) -> bytes | None:
    want = _sha256(findings, subject, "sha256", digest)
    target = _resolve(root, path, findings, subject)
    if target is None or want is None:
        return None
    try:
        data = target.read_bytes()
    except OSError as exc:
        findings.add(REF_MISSING, subject, f"{target}: {type(exc).__name__}")
        return None
    if hashlib.sha256(data).hexdigest() != want:
        findings.add(REF_HASH_MISMATCH, subject,
                     f"bytes of {target} do not hash to the recorded sha256")
        return None
    return data


_UNREADABLE = object()  # distinct from a document that parsed to JSON null


def _parse_json(findings: _Findings, subject: str, data: bytes | None) -> Any:
    if data is None:
        return _UNREADABLE
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        findings.add(REF_MALFORMED, subject, "verified bytes are not valid UTF-8 JSON")
        return _UNREADABLE


def _verify_ref(findings: _Findings, subject: str, obj: Mapping, root: Path) -> Any:
    """Verify a {path, sha256} ref end to end; returns the parsed document."""
    data = _read_verified(findings, subject, obj.get("path"), obj.get("sha256"), root)
    return _parse_json(findings, subject, data)


def _names_in(doc: Any) -> set:
    """Every id/name/consumer value anywhere in a parsed document."""
    found: set = set()
    if isinstance(doc, Mapping):
        for key, value in doc.items():
            if key in ("id", "name", "consumer") and isinstance(value, str):
                found.add(value)
            else:
                found |= _names_in(value)
    elif isinstance(doc, list):
        for item in doc:
            found |= _names_in(item)
    return found


# --------------------------------------------------------------------------
# section validators
# --------------------------------------------------------------------------


def _check_candidate(cand: Mapping, findings: _Findings, root: Path) -> None:
    commit = cand.get("commit")
    if isinstance(commit, str) and commit.strip() and not _is_implicit(commit):
        text = commit.strip()
        if len(text) not in (40, 64) or any(c not in "0123456789abcdef" for c in text.lower()):
            findings.add(BAD_CODE_IDENTITY, "candidate.commit",
                         f"{text!r} is not an exact commit hash")
    else:
        _identity(findings, "candidate.commit", "commit", commit)
    rows = _obj_list(findings, cand.get("config"), "candidate.config")
    seen = set()
    for i, item in enumerate(rows):
        row = _strict(findings, item, f"candidate.config[{i}]", _CONFIG_KEYS, _CONFIG_KEYS)
        subject = str(row.get("name") or f"candidate.config[{i}]")
        if subject in seen:
            findings.add(DUPLICATE, f"candidate.config.name={subject}", "identity repeated")
        seen.add(subject)
        _identity(findings, subject, "name", row.get("name"))
        _verify_ref(findings, f"config:{subject}", row, root)


def _check_data_snapshot(snap: Mapping, findings: _Findings, root: Path) -> None:
    declared_id = _identity(findings, "data_snapshot", "snapshot_id", snap.get("snapshot_id"))
    doc = _verify_ref(findings, "data_snapshot", snap, root)
    if doc is _UNREADABLE:
        return
    if not isinstance(doc, Mapping):
        findings.add(REF_MALFORMED, "data_snapshot",
                     "referenced snapshot document is not a JSON object")
        return
    named = doc.get("snapshot_id")
    if named is None:
        findings.add(REF_MISMATCH, "data_snapshot",
                     "referenced snapshot document declares no snapshot_id (fail closed)")
    elif named != declared_id:
        findings.add(REF_MISMATCH, "data_snapshot",
                     f"snapshot document names {named!r}, not the pinned {declared_id!r}")


def _check_model_deployment(dep: Mapping, findings: _Findings, root: Path) -> Any:
    release = _identity(findings, "model_deployment", "release_id", dep.get("release_id"))
    doc = _verify_ref(findings, "model_deployment", dep, root)
    if doc is not _UNREADABLE:
        if not isinstance(doc, Mapping):
            findings.add(REF_MALFORMED, "model_deployment",
                         "referenced release document is not a JSON object")
        else:
            named = doc.get("release_id")
            if named is None:
                findings.add(REF_MISMATCH, "model_deployment",
                             "release manifest declares no release_id (fail closed)")
            elif named != release:
                findings.add(REF_MISMATCH, "model_deployment",
                             f"release manifest names {named!r}, not {release!r}")
    return release


def _check_job_accounting(job: Mapping, findings: _Findings) -> None:
    subject = str(job.get("id") or "?")
    schedule = job.get("schedule")
    if isinstance(schedule, str):
        _identity(findings, f"job:{subject}", "schedule", schedule)
    else:
        findings.add(BAD_SCHEDULE, f"job:{subject}",
                     f"schedule={schedule!r} is not a string schedule")
    _identity(findings, f"job:{subject}", "owner", job.get("owner"))
    if job.get("kind") not in JOB_KINDS:
        findings.add(BAD_JOB_KIND, f"job:{subject}", f"kind={job.get('kind')!r} not in {JOB_KINDS}")


def _check_job_graph(graph: Mapping, findings: _Findings, root: Path) -> list[dict]:
    """Returns the well-formed declared jobs; graph file and accounting must agree."""
    jobs = _obj_list(findings, graph.get("jobs"), "job_graph.jobs")
    declared: dict[str, dict] = {}
    for i, item in enumerate(jobs):
        job = _strict(findings, item, f"job_graph.jobs[{i}]", _JOB_KEYS, _JOB_KEYS)
        _check_job_accounting(job, findings)
        jid = str(job.get("id") or f"job_graph.jobs[{i}]")
        if jid in declared:
            findings.add(DUPLICATE, f"job:{jid}", "job id repeated in the accounting")
            continue
        declared[jid] = job
    doc = _verify_ref(findings, "job_graph", graph, root)
    if doc is _UNREADABLE:
        return list(declared.values())
    graph_rows = doc.get("jobs") if isinstance(doc, Mapping) else None
    if not isinstance(graph_rows, list):
        findings.add(REF_MALFORMED, "job_graph",
                     "referenced graph is not a JSON object with a 'jobs' list")
        return list(declared.values())
    graph_jobs: dict[str, dict] = {}
    for i, row in enumerate(graph_rows):
        if not isinstance(row, Mapping):
            findings.add(REF_MALFORMED, "job_graph", f"graph jobs[{i}] is not a mapped row")
            continue
        jid = row.get("id")
        if not isinstance(jid, str) or not jid.strip():
            findings.add(REF_MALFORMED, "job_graph", f"graph jobs[{i}] has no string id")
            continue
        sched = row.get("schedule")
        if not isinstance(sched, str) or not sched.strip():
            findings.add(REF_MALFORMED, "job_graph",
                         f"graph jobs[{i}] (id {jid!r}) schedule={sched!r} is not a string")
        official = row.get("writes_official")
        if not isinstance(official, bool):
            findings.add(REF_MALFORMED, "job_graph",
                         f"graph jobs[{i}] (id {jid!r}) writes_official={official!r} "
                         "must be a JSON boolean, not missing, null or a string")
        if jid in graph_jobs:
            findings.add(DUPLICATE, f"job:{jid}", "job id repeated in the referenced graph")
            continue
        graph_jobs[jid] = row
    for jid, job in declared.items():
        if jid not in graph_jobs:
            findings.add(PHANTOM_JOB, f"job:{jid}", "accounted for but the graph does not run it")
    for jid, row in graph_jobs.items():
        if jid not in declared:
            findings.add(UNACCOUNTED_JOB, f"job:{jid}", "scheduled/background job with no "
                                                        "accounting row (P7-1 negative control)")
            continue
        job = declared[jid]
        if row.get("schedule") != job.get("schedule"):
            findings.add(REF_MISMATCH, f"job:{jid}", f"graph schedule {row.get('schedule')!r} "
                                                     f"!= accounted {job.get('schedule')!r}")
        official = row.get("writes_official")
        kind = job.get("kind")
        if official is True and kind != "scheduled_writer":
            findings.add(WRITER_CONFLICT, f"job:{jid}",
                         f"graph marks it an official writer but it is accounted {kind!r}")
        if kind == "scheduled_writer" and official is not True:
            findings.add(REF_MISMATCH, f"job:{jid}",
                         "accounted scheduled_writer but the graph does not confirm "
                         f"writes_official=true (found {official!r})")
    return list(declared.values())


def _check_authority(authority: Mapping, jobs: list[dict], findings: _Findings) -> None:
    sides = {}
    for side in ("old", "proposed"):
        rows = _strict(findings, authority.get(side) or {}, f"authority.{side}",
                       _SIDE_KEYS, _SIDE_KEYS)
        incomplete = False
        for key in _SIDE_KEYS:
            if _identity(findings, f"authority.{side}", key, rows.get(key)) is None:
                incomplete = True
        sides[side] = rows
        if incomplete:
            findings.add(OWNERSHIP_INCOMPLETE, f"authority.{side}",
                         "schedule, writer and credential_owner must all be explicit")
    if all(k in sides[side] and sides[side][k] for side in sides for k in _SIDE_KEYS):
        same = [k for k in _SIDE_KEYS if sides["old"][k] == sides["proposed"][k]]
        if same:
            findings.add(WRITER_NO_SWITCH, "authority",
                         f"old and proposed are identical on {same}; that is not a switch")
    retained = _strict(findings, authority.get("retained_old_deployment") or {},
                       "authority.retained_old_deployment", _RETAINED_KEYS, _RETAINED_KEYS)
    ref_ok = _identity(findings, "authority.retained_old_deployment", "ref",
                       retained.get("ref")) is not None
    owner_ok = _identity(findings, "authority.retained_old_deployment", "rollback_owner",
                         retained.get("rollback_owner")) is not None
    if not (ref_ok and owner_ok):
        findings.add(OLD_DEPLOYMENT_MISSING, "authority.retained_old_deployment",
                     "P7-1 requires the retained old deployment ref and its rollback owner")

    writers = {side: sides[side].get("writer") for side in sides}
    scheduled = [j for j in jobs if j.get("kind") == "scheduled_writer"]
    for job in scheduled:
        owner = job.get("owner")
        if owner not in list(writers.values()):
            findings.add(WRITER_CONFLICT, f"job:{job.get('id')}",
                         f"official writer {owner!r} is not a declared old/proposed writer")
            continue
    for side, writer in writers.items():
        count = sum(1 for j in scheduled if j.get("owner") == writer)
        if count > 1:
            findings.add(DUPLICATE_OFFICIAL_WRITER, f"authority.{side}.writer",
                          f"{writer!r} owns {count} scheduled_writer jobs; "
                          "one official writer only")
        if count == 0 and writer:
            findings.add(WRITER_NO_JOB, f"authority.{side}.writer",
                         f"{writer!r} owns no scheduled_writer job in the graph")
    by_schedule: dict = {}
    for job in jobs:
        sched = job.get("schedule")
        if job.get("kind") in ("scheduled_writer", "background") and isinstance(sched, str):
            by_schedule.setdefault(sched, []).append(job.get("id"))
    for schedule, ids in sorted(by_schedule.items(), key=lambda kv: str(kv[0])):
        if schedule and len(ids) > 1:
            findings.add(DUPLICATE, f"schedule:{schedule}",
                         f"non-reader jobs {sorted(map(str, ids))} share one schedule")
    for side_name, side in sides.items():
        schedule = side.get("schedule")
        writer = side.get("writer")
        owned = [j.get("schedule") for j in scheduled
                 if writer and j.get("owner") == writer and isinstance(j.get("schedule"), str)]
        if isinstance(schedule, str) and owned and schedule not in owned:
            findings.add(REF_MISMATCH, f"authority.{side_name}.schedule",
                         f"declared schedule {schedule!r} is not carried by {writer!r}'s own "
                         "scheduled_writer job; schedule and writer must be validated as one "
                         "pair per side and cannot be swapped between sides")


def _check_consumer_inventory(inv: Mapping, findings: _Findings, root: Path) -> None:
    names = _obj_list(findings, inv.get("consumers"), "consumer_inventory.consumers")
    seen = set()
    for i, name in enumerate(names):
        if not isinstance(name, str):
            findings.add(BAD_CONSUMER, f"consumer_inventory.consumers[{i}]",
                         f"entry {name!r} is not a consumer name string")
            continue
        _identity(findings, "consumer_inventory", "consumer", name)
        if name in seen:
            findings.add(DUPLICATE, f"consumer:{name}", "consumer listed twice")
        seen.add(name)
    doc = _verify_ref(findings, "consumer_inventory", inv, root)
    if doc is _UNREADABLE:
        return
    if not isinstance(doc, (Mapping, list)):
        findings.add(REF_MALFORMED, "consumer_inventory",
                     "referenced inventory is neither a JSON object nor array")
        return
    present = _names_in(doc)
    for name in sorted(seen):
        if name not in present:
            findings.add(CONSUMER_UNACCOUNTED, f"consumer:{name}",
                         "declared consumer has no row in the referenced inventory")


def _phase_token(phase: Any) -> str:
    return f"phase{str(phase).strip().lower()}"


def _check_phase_evidence(rows: Any, findings: _Findings, root: Path,
                          commit: Any, release: Any) -> None:
    required = {p.upper() for p in REQUIRED_PHASES}
    evidence, seen = [], set()
    for i, item in enumerate(rows if isinstance(rows, list) else []):
        row = _strict(findings, item, f"phase_evidence[{i}]", _EVIDENCE_KEYS, _EVIDENCE_KEYS)
        phase = str(row.get("phase", "")).strip().upper()
        if phase not in required:
            findings.add(BAD_PHASE, f"phase_evidence[{i}]",
                         f"phase {row.get('phase')!r} is not one of the required "
                         f"{REQUIRED_PHASES}")
        if phase in seen:
            findings.add(DUPLICATE, f"phase:{phase}", "evidence row repeated for one phase")
            continue
        seen.add(phase)
        evidence.append((phase, row))
    if not isinstance(rows, list):
        findings.add(BAD_LIST, "phase_evidence", "must be a list with one row per required phase")
    for phase in REQUIRED_PHASES:
        if phase.upper() not in seen:
            findings.add(MISSING_PHASE_EVIDENCE, f"phase:{phase}",
                         "required phase evidence has no row (3A preview evidence cannot "
                         "substitute for 3B/4/5/6)")
    for phase, row in evidence:
        subject = f"phase:{phase}"
        declared_schema = _identity(findings, subject, "schema_version", row.get("schema_version"))
        if not row.get("status"):
            findings.add(MISSING_FIELD, subject, "status the evidence must report is required")
        token = _phase_token(phase)
        if declared_schema and not declared_schema.lower().startswith(token):
            findings.add(SCHEMA_VERSION, subject,
                         f"{declared_schema!r} does not belong to phase {phase!r} ({token}...)")
        data = _read_verified(findings, subject, row.get("path"), row.get("sha256"), root)
        doc = _parse_json(findings, subject, data)
        if doc is _UNREADABLE:
            continue
        if not isinstance(doc, Mapping):
            findings.add(REF_MALFORMED, subject,
                         "evidence document is not a JSON object "
                         f"(found {type(doc).__name__.lower()})")
            continue
        if doc.get("schema_version") != declared_schema:
            findings.add(EVIDENCE_STALE, subject,
                         f"evidence schema_version {doc.get('schema_version')!r} != "
                         f"declared {declared_schema!r}")
        status = row.get("status")
        if status and doc.get("status") != status:
            findings.add(EVIDENCE_STALE, subject,
                         f"evidence status {doc.get('status')!r} != required {status!r}")
        bindings = {k: doc[k] for k in _COMMIT_KEYS if k in doc}
        if not bindings:
            findings.add(EVIDENCE_STALE, subject,
                         f"evidence names no candidate commit binding (one of "
                         f"{list(_COMMIT_KEYS)}); missing bindings fail closed")
        for key, bound in sorted(bindings.items()):
            if not isinstance(bound, str):
                findings.add(EVIDENCE_STALE, subject,
                             f"evidence binding {key}={bound!r} is not a commit string")
            elif bound != commit:
                findings.add(EVIDENCE_STALE, subject,
                             f"evidence {key}={bound!r}, "
                             f"not this candidate's commit {commit!r}")
        if phase == "5":
            if "release_id" not in doc:
                findings.add(REF_MISMATCH, subject,
                             "Phase 5 evidence names no release_id binding "
                             "(fail closed)")
            elif doc.get("release_id") != release:
                findings.add(REF_MISMATCH, subject,
                             f"evidence release {doc.get('release_id')!r} != deployed {release!r}")


# --------------------------------------------------------------------------
# the validation entry point
# --------------------------------------------------------------------------


def validate(manifest: Any, root: Path | str = ROOT) -> dict:
    """Validate one parsed manifest against reachable referenced files.

    Read-only: opens files for reading only, writes nothing anywhere.
    ``root`` anchors relative ref paths.
    """
    root = Path(root)
    findings = _Findings()
    top = _strict(findings, manifest if isinstance(manifest, Mapping) else {},
                  "manifest", _TOP_KEYS, _TOP_KEYS)
    if isinstance(manifest, Mapping) and top.get("schema_version") != MANIFEST_SCHEMA:
        findings.add(SCHEMA_VERSION, "manifest",
                     f"{top.get('schema_version')!r} != required {MANIFEST_SCHEMA!r}")
    candidate = _strict(findings, top.get("candidate") or {}, "candidate",
                        _CANDIDATE_KEYS, _CANDIDATE_KEYS)
    snapshot = _strict(findings, top.get("data_snapshot") or {}, "data_snapshot",
                       _SNAPSHOT_KEYS, _SNAPSHOT_KEYS)
    deployment = _strict(findings, top.get("model_deployment") or {}, "model_deployment",
                         _DEPLOYMENT_KEYS, _DEPLOYMENT_KEYS)
    graph = _strict(findings, top.get("job_graph") or {}, "job_graph", _GRAPH_KEYS, _GRAPH_KEYS)
    inventory = _strict(findings, top.get("consumer_inventory") or {}, "consumer_inventory",
                        _INVENTORY_KEYS, _INVENTORY_KEYS)
    authority = _strict(findings, top.get("authority") or {}, "authority",
                        _AUTHORITY_KEYS, _AUTHORITY_KEYS)

    _check_candidate(candidate, findings, root)
    commit = candidate.get("commit")
    _check_data_snapshot(snapshot, findings, root)
    release = _check_model_deployment(deployment, findings, root)
    jobs = _check_job_graph(graph, findings, root)
    _check_authority(authority, jobs, findings)
    _check_consumer_inventory(inventory, findings, root)
    _check_phase_evidence(top.get("phase_evidence"), findings, root, commit, release)

    rows = findings.sorted_rows()
    return {"schema_version": MANIFEST_SCHEMA,
            "manifest_root": str(root),
            "ok": not rows,
            "status": "MANIFEST_VALID" if not rows else "MANIFEST_INVALID",
            "findings": rows,
            "finding_codes": sorted({r["code"] for r in rows}),
            "note": "manifest-consistency validation only: it never queries a live "
                    "scheduler and cannot prove referenced inventories are complete; "
                    "not a Phase 7 readiness claim"}


def load_manifest(path: Path | str) -> Any:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise ManifestUnreadable(f"{path}: {exc}") from exc


def _summary(evidence: dict) -> str:
    lines = [f"{f['code']}: {f['subject']}: {f['detail']}" for f in evidence["findings"]]
    lines.append(f"phase7 candidate manifest: {evidence['status']} "
                 f"({len(evidence['findings'])} findings)")
    lines.append(evidence["note"])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P7-1 candidate/authority manifest validation")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="base directory for relative ref paths (default: the repo)")
    parser.add_argument("--json", action="store_true", help="machine-readable findings")
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
    except ManifestUnreadable as exc:
        print(f"cannot read manifest: {exc}", file=sys.stderr)
        return 2
    evidence = validate(manifest, args.root)
    print(json.dumps(evidence, indent=2, sort_keys=True) if args.json else _summary(evidence))
    return 0 if evidence["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
