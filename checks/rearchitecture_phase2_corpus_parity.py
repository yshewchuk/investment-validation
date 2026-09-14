#!/usr/bin/env python3
"""D14: Phase 0 tier-0 corpus scored through the Phase 2 snapshot adapter.

    python3 checks/rearchitecture_phase2_corpus_parity.py import \\
        --root /root/phase2-corpus-ops --source-root . --scope corpus --corpus fixtures/tier0
    python3 checks/rearchitecture_phase2_corpus_parity.py run \\
        --root /root/phase2-corpus-ops --store-root . --scope corpus \\
        --corpus fixtures/tier0 --out /root/phase2-corpus-ops/corpus_run.json
    python3 checks/rearchitecture_phase2_corpus_parity.py compare \\
        --corpus fixtures/tier0 --rows /root/phase2-corpus-ops/corpus_run.json \\
        --artifact-root /root/phase2-corpus-ops/evidence
    python3 checks/rearchitecture_phase2_corpus_parity.py control-missing-analogs \\
        --root /root/phase2-corpus-ops --source-root . --store-root . \\
        --corpus fixtures/tier0 --drop-ticker XYZ --artifact-root /root/phase2-corpus-ops/evidence

``--root``/``--artifact-root`` are required with no default, and must never
resolve inside ``--source-root``/``--store-root`` (the legacy checkout) --
refused (``INVALID_REQUEST``) before anything is written, so an operator
cannot point ops state or evidence at the legacy tree by typo. Use a private
directory outside this checkout (``/root/phase2-corpus-ops`` above), never
``data/operations`` or a path under this repo's own ``data/``.

D14 (quoted): "Phase 0 corpus requests scored through snapshot adapter match
expected IDs, contracts, null masks, flags, forecasts, decisions, and
full-precision values. A missing broad analog slice is caught as a
stage-localized finding." Guide note: "D14 must invoke the real legacy
scoring public entrypoint in a fresh supervised process."

Four stages, matching Phase 1/2's existing snapshot machinery exactly rather
than inventing a parallel path:

* ``import`` -- import a legacy root (the tier-0 corpus's own dependency: the
  live store at the corpus's frozen ``SNAPSHOT`` hash, verified in Step 1 to
  equal the current one, via :func:`legacy_store_snapshot_hash` -- the SAME
  relative path/key ``checks.replay_identity.store_snapshot`` reads from the
  live tree, parameterized on an arbitrary root instead of the process-wide
  ``engine.paths.ROOT``; a mismatch refuses ``INPUT_CHANGED`` naming both
  hashes before anything is submitted, since a store that moved since the
  corpus was captured would otherwise produce a disagree receipt blaming the
  adapter for inputs that simply changed) into an ops snapshot scope, through
  the real §7.1 coordinator (``engine.v2.data.import_snapshot.plan_import`` /
  ``engine.v2.ops.snapshot_import``) -- the SAME mechanism Phase 2's
  snapshot-backed ``legacy_score``/``legacy_score_requests`` already use.
* ``run`` -- materialize that scope (``legacy_materialize``) and submit one
  snapshot-mode ``legacy_score_requests`` job (task 6b) over every corpus
  request the action can reach, with the full corpus context, in a fresh
  worker process.
* ``compare`` -- ``engine.v2.diagnosis.compare_records`` each returned record
  against its frozen expected record under tier 0's own declared tolerance
  (``score_record.exact.v1``), and emit one ``comparison_receipt.v1.1`` of
  kind ``corpus_score_parity`` (``rearchitecture_phase2_evidence.
  CORPUS_PARITY_KIND``), bound to code/environment/snapshot identity exactly
  as D15 (``rearchitecture_phase2_parity.py``) already does.
* ``control-missing-analogs`` -- the same run against an import whose
  ``trades`` table has one ticker's rows removed, and assert every finding
  traces to the ``analogs`` stage (``engine.v2.diagnosis.stage_plan.SCORER_V1``
  already attributes a cascading diff to its earliest causal stage, so a
  perturbation that only touches the analog pool is expected to localize
  there even where it also moves ``simulation``/``gate``/``chooser`` fields).

**Scope of what ``legacy_score_requests`` can replay.** The action
(``engine.v2.ops.legacy_adapter._action_score_requests``) only builds a
``ScoreRequest`` and calls ``Scorer.score`` -- it has no entrypoint for
``engine.replay.replay_one`` (the corpus's ``research_replay`` fixtures, the
two disabled-structure records) or for re-running the DYN-SV chooser over a
frame (a ``dyn_sv_choice`` fixture's own top-level record). Those are counted
in ``expected`` and dropped to ``supported`` with a named reason. A
``dyn_sv_choice`` fixture's frozen FRAME rows are themselves plain score
requests (the same flattening ``rearchitecture_phase1_canary.prepare`` already
performs) and so are fully supported and compared.

Findings never carry a value (task brief): every finding is redacted and
prefixed ``<canary_id>::<field_path>``, exactly as D15
(``rearchitecture_phase2_parity.py``) already redacts its own.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks.rearchitecture_phase1_gate import source_files, source_hash  # noqa: E402
from checks.rearchitecture_phase2_evidence import (  # noqa: E402
    CORPUS_PARITY_KIND,
    CORPUS_SNAPSHOT_BINDING_V1,
)
from checks.rearchitecture_phase2_gate import environment_hash as _environment_hash  # noqa: E402
from checks.tier0_corpus import DEFAULT_CORPUS, Corpus, resolve_corpus  # noqa: E402
from checks.tier0_corpus import load as load_corpus  # noqa: E402
from engine.v2.contracts import JobSpec, SubmitRequest  # noqa: E402
from engine.v2.data import reference_inputs  # noqa: E402
from engine.v2.data.import_snapshot import plan_import  # noqa: E402
from engine.v2.data.repository import Repository  # noqa: E402
from engine.v2.diagnosis import (  # noqa: E402
    AGREE,
    DIFFER,
    INCOMPARABLE,
    SCORE_RECORD_V1,
    SCORER_V1,
    ComparisonReceipt,
    Envelope,
    Finding,
    Population,
    compare_records,
    content_hash,
)
from engine.v2.foundation import ArtifactStore, SystemClock, to_document  # noqa: E402
from engine.v2.ops.bootstrap import open_catalog  # noqa: E402
from engine.v2.ops.catalog import transaction  # noqa: E402
from engine.v2.ops.checkpoints import artifact as load_artifact  # noqa: E402
from engine.v2.ops.checkpoints import register_artifact  # noqa: E402
from engine.v2.ops.errors import OpsError, fail  # noqa: E402
from engine.v2.ops.fingerprints import environment_identity, worker_source_manifest  # noqa: E402
from engine.v2.ops.profiles import DEFAULT_POLICY, profile_named  # noqa: E402
from engine.v2.ops.snapshot_import import save_import_plan, submit_import  # noqa: E402
from engine.v2.ops.snapshot_planning import pin_snapshot_inputs  # noqa: E402
from engine.v2.ops.snapshot_stages import MANIFEST_OUTPUT  # noqa: E402
from engine.v2.ops.stages import registry  # noqa: E402
from engine.v2.ops.submission import NamespacePolicy, submit  # noqa: E402
from engine.v2.ops.supervisor import Service  # noqa: E402

__all__ = ["corpus_population", "legacy_store_snapshot_hash", "import_corpus", "run_corpus",
           "build_receipt", "control_localized_to_analogs", "publish",
           "publish_corpus_snapshot_binding", "main"]

DEFAULT_SCOPE = "corpus"
_TERMINAL = ("succeeded", "failed", "blocked", "cancelled")


# --------------------------------------------------------------------------
# legacy SNAPSHOT identity, and keeping ops/evidence state out of legacy data/
# --------------------------------------------------------------------------


def legacy_store_snapshot_hash(root: Path) -> str | None:
    """The legacy ``SNAPSHOT`` compatibility object's own ``snapshot`` id,
    read from ``root`` -- the SAME relative path
    (``reference_inputs.LEGACY_SNAPSHOT_PATH``, the v2-reviewed mirror of
    ``engine.paths.SNAPSHOT_FILE``) and JSON key
    ``checks.replay_identity.store_snapshot`` reads from the live tree,
    parameterized on an arbitrary root instead of the process-wide
    ``engine.paths.ROOT`` so a ``--source-root``/``--store-root`` can be
    checked before it is ever imported. No new hasher: the value is a
    precomputed id already written by ``engine.data.manifest.write_snapshot``,
    only read back here."""
    path = Path(root) / reference_inputs.LEGACY_SNAPSHOT_PATH
    try:
        return json.loads(path.read_text()).get("snapshot")
    except (ValueError, OSError):
        return None


def _check_corpus_snapshot(source_root: Path, corpus_root: Path) -> str | None:
    """Refuse ``INPUT_CHANGED`` (details ``reason=CORPUS_SNAPSHOT_MISMATCH``)
    before anything is submitted when ``source_root``'s legacy SNAPSHOT hash
    does not equal the tier-0 corpus's own frozen ``snapshot`` id -- a store
    that has moved since the corpus was captured must be refused here, not
    produce a disagree receipt that blames the adapter for inputs that simply
    changed. Returns the matched hash (possibly ``None`` if neither side
    declares one -- an un-hashed synthetic fixture, never the real corpus)."""
    corpus = load_corpus(resolve_corpus(Path(corpus_root)))
    corpus_snapshot = corpus.index.get("snapshot")
    source_snapshot = legacy_store_snapshot_hash(source_root)
    if source_snapshot != corpus_snapshot:
        raise fail("INPUT_CHANGED",
                  "legacy source_root's SNAPSHOT hash does not match the tier-0 "
                  "corpus's frozen snapshot",
                  details={"reason": "CORPUS_SNAPSHOT_MISMATCH",
                          "corpus_snapshot": corpus_snapshot, "source_snapshot": source_snapshot})
    return source_snapshot


def _refuse_if_inside(label: str, path: Path, boundary_label: str, boundary: Path) -> None:
    """Refuse ``INVALID_REQUEST`` when ``--<label>`` resolves inside
    ``--<boundary_label>`` -- ops/evidence state must never land inside the
    legacy checkout it reads from (``--source-root``/``--store-root``)."""
    resolved_path, resolved_boundary = Path(path).resolve(), Path(boundary).resolve()
    if resolved_path == resolved_boundary or resolved_path.is_relative_to(resolved_boundary):
        raise fail("INVALID_REQUEST", f"--{label} must not resolve inside --{boundary_label}",
                  details={"reason": "ROOT_INSIDE_LEGACY_TREE", label: str(resolved_path),
                          boundary_label: str(resolved_boundary)})


# --------------------------------------------------------------------------
# what a corpus fixture can be replayed through legacy_score_requests
# --------------------------------------------------------------------------


def corpus_population(corpus: Corpus) -> tuple[list[tuple[str, dict, dict]], list[dict]]:
    """Every corpus request ``legacy_score_requests`` can reach, and every
    drop from the declared corpus, explained (task P2-C01 decision 4 shape).

    Returns ``(supported, excluded)``: ``supported`` is
    ``[(canary_id, request, expected_record), ...]``; ``excluded`` is
    ``[{"stage": "expected_to_supported", "key": ..., "reason": ...}, ...]``.
    """
    supported: list[tuple[str, dict, dict]] = []
    excluded: list[dict] = []
    for fid in corpus.ordered_ids:
        payload = corpus.pairs[fid]["payload"]
        kind, request, record = payload["record_kind"], payload["request"], payload["record"]
        if kind == "score_result":
            supported.append((fid, request, record))
        elif kind == "research_replay":
            excluded.append({"stage": "expected_to_supported", "key": fid,
                             "reason": "research_replay uses engine.replay.replay_one, "
                                       "which legacy_score_requests does not call"})
        elif kind == "dyn_sv_choice":
            excluded.append({"stage": "expected_to_supported", "key": fid,
                             "reason": "dyn_sv_choice's chosen record has no ScoreRequest; "
                                       "only its frame rows replay"})
            for i, row in enumerate(request.get("frame_rows") or []):
                supported.append((f"{fid}#frame-{i}", row["request"], row["record"]))
    return supported, excluded


def _context(supported: list[tuple[str, dict, dict]]) -> tuple[list[str], int, int, tuple[str, ...]]:
    """The full corpus context: every ticker/year the supported requests need,
    and one ``ticker|strategy|event_date`` key per request for
    ``pin_snapshot_inputs``'s direct-scope derivation."""
    tickers = sorted({str(req["ticker"]) for _, req, _ in supported})
    years = sorted({int(str(req["event_date"])[:4]) for _, req, _ in supported})
    keys = tuple(sorted({f"{req['ticker']}|{req['strategy']}|{req['event_date']}"
                        for _, req, _ in supported}))
    return tickers, years[0], years[-1], keys


# --------------------------------------------------------------------------
# ops plumbing shared by import and run
# --------------------------------------------------------------------------


def _open(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    clock = SystemClock()
    return open_catalog(root / "catalog.sqlite", clock=clock), clock, ArtifactStore(root)


def _run_to_terminal(service, conn, job_id, timeout=1800) -> str:
    deadline = time.monotonic() + timeout
    state = "queued"
    while time.monotonic() < deadline:
        service.tick()
        state = conn.execute("SELECT state FROM jobs WHERE job_id=?", (job_id,)).fetchone()[0]
        if state in _TERMINAL:
            return state
        time.sleep(0.05)
    return state


def _require_succeeded(conn, job_id, state, what) -> None:
    if state != "succeeded":
        failure = conn.execute("SELECT failure_json FROM jobs WHERE job_id=?",
                               (job_id,)).fetchone()[0]
        raise RuntimeError(f"{what} did not succeed ({state}): {failure}")


def _job_output_artifact(conn, job_id, name=None) -> str:
    """One job output's artifact id. ``name=None``: the checkpoint-cache path
    (``supervisor.Service._checkpoint_refs``) names a non-coordinator-effect
    kind's output by its POSITIONAL index ("0", ...), never the worker's own
    declared name -- ``legacy_score_requests`` is exactly this case, so the
    only reliable lookup is "the attempt's one output", as
    ``rearchitecture_phase1_adapter.run_via_supervisor`` already does."""
    attempt = conn.execute(
        "SELECT attempt_id FROM attempts WHERE job_id=? AND state='succeeded' "
        "ORDER BY attempt_number DESC LIMIT 1", (job_id,)).fetchone()[0]
    if name is None:
        row = conn.execute("SELECT artifact_id FROM attempt_outputs WHERE attempt_id=?",
                           (attempt,)).fetchone()
    else:
        row = conn.execute("SELECT artifact_id FROM attempt_outputs WHERE attempt_id=? AND name=?",
                           (attempt, name)).fetchone()
    if row is None:
        raise RuntimeError(f"job {job_id} has no output" + (f" named {name!r}" if name else ""))
    return row[0]


def _submit(conn, policy, clock, *, kind, parameters, input_refs, resource,
           checkpoint_contract, key, deps=()) -> str:
    profile = profile_named(DEFAULT_POLICY, resource)
    job = JobSpec(
        kind=kind, implementation_ref=content_hash(worker_source_manifest(ROOT)), spec_hash=None,
        environment_ref=content_hash(environment_identity(profile.thread_count or profile.cpu_count)),
        parameters=parameters, input_refs=tuple(input_refs), dependency_job_ids=tuple(deps),
        output_namespace="shadow", resource_class=resource, retry_policy_ref="bounded",
        checkpoint_contract_ref=checkpoint_contract)
    return submit(conn, registry(), policy, SubmitRequest(
        namespace="shadow", idempotency_key=key, principal="operator", job=job),
        clock=clock).job_id


# --------------------------------------------------------------------------
# import: the corpus's frozen legacy root -> an ops snapshot scope
# --------------------------------------------------------------------------


def import_corpus(root: Path, source_root: Path, scope: str, corpus_root: Path, *, policy,
                  idempotency_key: str | None = None, resource_policy=DEFAULT_POLICY) -> dict:
    """Import ``source_root`` (a legacy checkout) into ops ``scope``, through
    the real §7.1 coordinator, run to completion in this process.

    Refuses before touching ``root`` at all: ``root`` resolving inside
    ``source_root`` (``INVALID_REQUEST``/``ROOT_INSIDE_LEGACY_TREE``, ops
    state landing inside the legacy checkout), then ``source_root``'s legacy
    SNAPSHOT hash not matching ``corpus_root``'s frozen ``snapshot`` id
    (``INPUT_CHANGED``/``CORPUS_SNAPSHOT_MISMATCH``, see
    :func:`_check_corpus_snapshot`).

    ``resource_policy`` is the ``Service``'s own capacity policy (production:
    ``DEFAULT_POLICY``; tests: ``tests.ops_support.TEST_POLICY``, which the
    real worker still runs under, only with smaller declared resource needs)
    -- distinct from ``policy``, the ``NamespacePolicy`` governing what this
    operator may submit."""
    _refuse_if_inside("root", root, "source-root", source_root)
    _check_corpus_snapshot(source_root, corpus_root)
    conn, clock, store = _open(root)
    try:
        plan = plan_import(source_root, scope=scope, expected_head_snapshot_id=None,
                           expected_head_generation=0)
        plan_ref = save_import_plan(conn, store, plan, clock=clock)
        receipt = submit_import(conn, store, plan_ref.artifact_id, registry=registry(),
                                policy=policy, clock=clock,
                                idempotency_key=idempotency_key or f"{scope}-import",
                                repo_root=ROOT)
        service = Service(conn, root, registry(), resource_policy, clock=clock,
                          code_source=ROOT, store_root=Path(source_root))
        try:
            service.start()
            state = _run_to_terminal(service, conn, receipt.job_id)
        finally:
            service.close()
        _require_succeeded(conn, receipt.job_id, state, "snapshot_import")
        head = conn.execute("SELECT snapshot_id, generation FROM data_snapshot_heads "
                            "WHERE scope=?", (scope,)).fetchone()
        snapshot = Repository(conn, store).resolve(head["snapshot_id"])
        return {"scope": scope, "snapshot_id": head["snapshot_id"], "generation": head["generation"],
               "snapshot_manifest_hash": snapshot.manifest_hash}
    finally:
        conn.close()


# --------------------------------------------------------------------------
# run: materialize the scope, submit one snapshot-mode legacy_score_requests
# --------------------------------------------------------------------------


def _submit_materialize(conn, pinned, *, policy, clock, key) -> str:
    bindings = {"snapshot_ref.json": pinned["snapshot_ref_artifact_id"],
               "materialization_request.json": pinned["materialization_request_ref"]}
    return _submit(conn, policy, clock, kind="legacy_materialize",
                   parameters={"expected_ids": ["legacy_materialize"], "input_bindings": bindings,
                              "scratch_estimate_bytes": pinned["scratch_estimate_bytes"]},
                   input_refs=tuple(bindings.values()), resource="materialize",
                   checkpoint_contract="legacy_materialization_manifest.v1.0", key=key)


def _submit_score_requests(conn, store, clock, pinned, manifest_id, supported, year_start, year_end,
                           *, policy, key, deps) -> str:
    entries = [{"canary_id": cid, "request": request} for cid, request, _ in supported]
    requests_ref = store.publish_bytes(json.dumps(entries, sort_keys=True, default=str).encode(),
                                       schema_ref="corpus_score_requests.v1.0")
    with transaction(conn):
        register_artifact(conn, requests_ref, None, clock)
    bindings = {"snapshot_ref.json": pinned["snapshot_ref_artifact_id"],
               "materialization_request.json": pinned["materialization_request_ref"],
               "materialization_manifest.json": manifest_id,
               "score_requests.json": requests_ref.artifact_id}
    return _submit(conn, policy, clock, kind="legacy_score_requests",
                   parameters={"expected_ids": ["legacy_score_requests"],
                              "requests_path": "score_requests.json", "year_start": year_start,
                              "year_end": year_end, "input_mode": "snapshot",
                              "input_bindings": bindings},
                   input_refs=tuple(bindings.values()), resource="legacy_score",
                   checkpoint_contract="legacy_action.v1.0", key=key, deps=deps)


def run_corpus(root: Path, store_root: Path, scope: str, corpus_root: Path, *, policy,
               idempotency_key: str | None = None, resource_policy=DEFAULT_POLICY) -> dict:
    """Materialize ``scope`` and submit a snapshot-mode ``legacy_score_requests``
    job over the corpus's full supported population, with the full corpus
    context, in a fresh worker process. ``resource_policy``: see
    :func:`import_corpus`. Refuses ``root`` resolving inside ``store_root``
    before anything is opened, same as :func:`import_corpus`."""
    _refuse_if_inside("root", root, "store-root", store_root)
    corpus = load_corpus(resolve_corpus(Path(corpus_root)))
    supported, _excluded = corpus_population(corpus)
    if not supported:
        raise RuntimeError("no supported corpus requests to score")
    _tickers, year_start, year_end, population = _context(supported)
    conn, clock, store = _open(root)
    key = idempotency_key or scope
    try:
        pinned = pin_snapshot_inputs(conn, store, scope, tickers=_tickers, year_start=year_start,
                                     year_end=year_end, expected_population=population, clock=clock)
        service = Service(conn, root, registry(), resource_policy, clock=clock,
                          code_source=ROOT, store_root=Path(store_root))
        try:
            service.start()
            materialize_id = _submit_materialize(conn, pinned, policy=policy, clock=clock,
                                                 key=f"{key}-materialize")
            state = _run_to_terminal(service, conn, materialize_id)
            _require_succeeded(conn, materialize_id, state, "legacy_materialize")
            manifest_id = _job_output_artifact(conn, materialize_id, MANIFEST_OUTPUT)
            score_id = _submit_score_requests(conn, store, clock, pinned, manifest_id, supported,
                                              year_start, year_end, policy=policy,
                                              key=f"{key}-score", deps=(materialize_id,))
            state = _run_to_terminal(service, conn, score_id)
            _require_succeeded(conn, score_id, state, "legacy_score_requests")
            rows_id = _job_output_artifact(conn, score_id)
        finally:
            service.close()
        document = json.loads(store.read_verified(load_artifact(conn, store, rows_id)))
        return {"rows": document["rows"], "snapshot_id": pinned["snapshot_id"],
               "snapshot_manifest_hash": pinned["snapshot_manifest_hash"],
               "legacy_snapshot_hash": legacy_store_snapshot_hash(store_root)}
    finally:
        conn.close()


# --------------------------------------------------------------------------
# compare: the real ComparisonReceipt, redacted and bound
# --------------------------------------------------------------------------


def _redact(finding: Finding, canary_id: str) -> Finding:
    """Never a value (task brief): prefix ``<canary_id>::<field_path>`` and
    drop everything that could carry licensed quote data, exactly as D15
    (``rearchitecture_phase2_parity._redact``) already does for its own rows."""
    return dataclasses.replace(finding, field_path=f"{canary_id}::{finding.field_path}",
                               left_value=None, right_value=None, delta=None,
                               exceeded_by=None, not_downstream_of=())


def _request_findings(canary_id: str, expected: dict, actual: dict) -> list[Finding]:
    receipt = compare_records(expected, actual, comparison_kind=CORPUS_PARITY_KIND, tier=0,
                              left_ref="corpus", right_ref="snapshot_adapter",
                              tolerance_policy=SCORE_RECORD_V1)
    return [_redact(f, canary_id) for f in receipt.findings]


def _corpus_version_name(corpus_root: Path, resolved: Path) -> str:
    """The identifier :func:`_corpus_index_snapshot` (the evidence validator's
    own mirror) can re-derive an INDEX.json path from: empty for the bare/
    unversioned layout (``corpus_root/INDEX.json`` IS the corpus -- every
    fixture this module's own tests build), otherwise the published version
    directory's own name, exactly what ``CURRENT`` names it
    (:func:`checks.tier0_corpus.resolve_corpus`'s other case)."""
    return "" if Path(resolved) == Path(corpus_root).resolve() else Path(resolved).name


def _corpus_snapshot_binding(corpus, corpus_root: Path, run_result: dict, *,
                             control_drop_ticker: str | None) -> dict:
    """The ``corpus_snapshot_binding.v1.0`` document (task D14 review): binds
    the legacy SNAPSHOT hash the run actually scored against to the corpus's
    own frozen ``snapshot`` id, machine-checkably -- the structured
    replacement for the old free-text ``diagnostic_ref`` stuffing. A control
    run's filtered copy only rewrites ``trades`` rows -- the SNAPSHOT
    compatibility object itself is hard-linked, unperturbed, so its hash
    cannot itself reveal the perturbation; ``control``/``control_drop_ticker``
    record that fact explicitly rather than leaving a matching hash to imply
    nothing changed, and the evidence validator refuses a control receipt
    offered in D14's own place outright."""
    resolved = resolve_corpus(Path(corpus_root))
    return {
        "schema_version": CORPUS_SNAPSHOT_BINDING_V1,
        "corpus_version": _corpus_version_name(corpus_root, resolved),
        "corpus_snapshot_hash": corpus.index.get("snapshot"),
        "source_snapshot_hash": run_result.get("legacy_snapshot_hash"),
        "control": control_drop_ticker is not None,
        "control_drop_ticker": control_drop_ticker,
    }


def publish_corpus_snapshot_binding(binding: dict, artifact_root: Path) -> dict:
    """Publish the binding document next to the receipt, content-addressed
    the same way :func:`publish` addresses the receipt itself. Returns the
    reference dict (``{"path": ..., "content_hash": ...}``) the receipt's
    ``envelope.diagnostic_ref`` is set to (JSON-encoded, since ``diagnostic_ref``
    is a plain ``str`` field -- see :class:`engine.v2.diagnosis.receipt.Envelope`)."""
    artifact_root = Path(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    data = json.dumps(binding, indent=2, sort_keys=True).encode()
    digest = hashlib.sha256(data).hexdigest()
    path = artifact_root / f"corpus_snapshot_binding_{digest[:16]}.json"
    path.write_bytes(data)
    return {"path": path.name, "content_hash": "sha256:" + digest}


def build_receipt(corpus_root: Path, run_result: dict, *, code_hash: str,
                  environment_hash: str, control_drop_ticker: str | None = None,
                  artifact_root: Path | None = None) -> ComparisonReceipt:
    """One receipt over the whole corpus: ``expected`` = every constituent
    request the 20 declared pairs imply; ``supported`` = what
    ``legacy_score_requests`` can reach; ``compared`` = what it actually
    returned a row for. Every drop between them is itemized (task P2-C01
    decision 4) rather than folded into a count. ``control_drop_ticker``:
    set only by ``run_control_missing_analogs``, see
    :func:`_corpus_snapshot_binding`. ``artifact_root``: when given, the
    corpus-snapshot binding is published there (next to where the caller
    will ``publish()`` this receipt) and ``envelope.diagnostic_ref`` is set
    to point at it; ``None`` leaves ``diagnostic_ref`` unset (a receipt built
    only to inspect population/findings, not for evidence submission)."""
    corpus = load_corpus(resolve_corpus(Path(corpus_root)))
    supported, excluded = corpus_population(corpus)
    actual_by_id = {row["request_id"]: row["record"] for row in run_result["rows"]
                    if row.get("request_id")}
    findings: list[Finding] = []
    drops = list(excluded)
    compared_ids: list[str] = []
    for cid, _request, expected_record in supported:
        if cid not in actual_by_id:
            drops.append({"stage": "supported_to_compared", "key": cid,
                         "reason": "legacy_score_requests returned no row for this request"})
            continue
        compared_ids.append(cid)
        findings.extend(_request_findings(cid, expected_record, actual_by_id[cid]))
    population = Population(expected=len(supported) + len(excluded), supported=len(supported),
                            compared=len(compared_ids), excluded=tuple(drops))
    diagnostic_ref = None
    if artifact_root is not None:
        binding = _corpus_snapshot_binding(corpus, corpus_root, run_result,
                                           control_drop_ticker=control_drop_ticker)
        ref = publish_corpus_snapshot_binding(binding, artifact_root)
        diagnostic_ref = json.dumps(ref, sort_keys=True)
    envelope = Envelope(code_hash=code_hash, environment_hash=environment_hash,
                        snapshot_id=run_result.get("snapshot_id"),
                        snapshot_manifest_hash=run_result.get("snapshot_manifest_hash"),
                        diagnostic_ref=diagnostic_ref)
    verdict = (INCOMPARABLE
              if population.expected <= 0 or population.supported <= 0 or population.compared <= 0
              else DIFFER if findings else AGREE)
    receipt_id = content_hash(["corpus_score_parity", corpus.index.get("corpus_hash"),
                               sorted(compared_ids), [f.finding_id for f in findings]])[7:23]
    return ComparisonReceipt(
        receipt_id=receipt_id, comparison_kind=CORPUS_PARITY_KIND, tier=0,
        left_ref="tier0_corpus", right_ref="snapshot_adapter", stage_plan_ref=SCORER_V1.plan_id,
        tolerance_policy_ref=SCORE_RECORD_V1.policy_id, verdict=verdict, findings=tuple(findings),
        population=population, envelope=envelope)


def code_and_environment_hash() -> tuple[str, str]:
    """Computed the SAME way ``rearchitecture_phase2_parity.build`` and the
    evidence builder compute them, over THIS repo checkout -- so a corpus
    receipt binds to exactly what ``rearchitecture_phase2_evidence.py``'s
    ``_check_bindings`` will check it against."""
    environment, _source = _environment_hash(ROOT)
    return source_hash(source_files(ROOT)), environment


def publish(receipt: ComparisonReceipt, artifact_root: Path, *, prefix="corpus_parity") -> dict:
    artifact_root.mkdir(parents=True, exist_ok=True)
    data = json.dumps(to_document(receipt), indent=2, sort_keys=True).encode()
    path = artifact_root / f"{prefix}_{receipt.receipt_id}.json"
    path.write_bytes(data)
    return {"path": path.name, "content_hash": "sha256:" + hashlib.sha256(data).hexdigest()}


# --------------------------------------------------------------------------
# control-missing-analogs
# --------------------------------------------------------------------------


def filtered_legacy_root(source_root: Path, scratch: Path, drop_ticker: str) -> Path:
    """A private copy of ``source_root`` with ``drop_ticker``'s ``trades``
    rows removed -- hard-linked everywhere else, so a multi-GB curated tree
    costs no extra bytes for the parts that do not change."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    source_root, scratch = Path(source_root), Path(scratch)
    trades_dir = f"{reference_inputs.DATA_DIR}/curated/trades/"
    for path in sorted(p for p in source_root.rglob("*") if p.is_file() and not p.is_symlink()):
        relative = path.relative_to(source_root).as_posix()
        target = scratch / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative.startswith(trades_dir) and path.suffix == ".parquet":
            table = pq.read_table(path)
            pq.write_table(table.filter(pc.not_equal(table.column("ticker"), drop_ticker)), target)
        else:
            os.link(path, target)
    return scratch


def control_localized_to_analogs(receipt: ComparisonReceipt) -> tuple[bool, list[str]]:
    """Every finding must trace to the ``analogs`` stage, and the receipt must
    actually disagree -- an unchanged corpus proves the control planted
    nothing, not that the adapter is correct."""
    stages = {f.first_differing_stage for f in receipt.findings}
    if receipt.verdict != DIFFER:
        return False, [f"expected a disagree verdict, got {receipt.verdict}"]
    stray = sorted(stages - {"analogs"})
    if stray:
        return False, [f"findings outside the analogs stage: {stray}"]
    return True, []


def run_control_missing_analogs(args, policy) -> tuple[ComparisonReceipt, bool, list[str]]:
    """Same refusals as ``import``/``run`` (``root``/``artifact_root`` must
    resolve outside ``source_root``/``store_root``), plus the snapshot check
    against the UNFILTERED ``source_root`` -- the perturbation only rewrites
    ``trades`` rows, so checking it here (before ``filtered_legacy_root``
    ever runs) is both cheaper and the honest comparison: the filtered copy's
    hard-linked SNAPSHOT file would trivially match regardless."""
    _refuse_if_inside("root", args.root, "source-root", args.source_root)
    _refuse_if_inside("root", args.root, "store-root", args.store_root)
    _refuse_if_inside("artifact-root", args.artifact_root, "source-root", args.source_root)
    _refuse_if_inside("artifact-root", args.artifact_root, "store-root", args.store_root)
    _check_corpus_snapshot(args.source_root, args.corpus)
    scratch = Path(args.root) / "control_source"
    filtered_legacy_root(args.source_root, scratch, args.drop_ticker)
    import_corpus(args.root, scratch, args.scope, args.corpus, policy=policy)
    run_result = run_corpus(args.root, args.store_root, args.scope, args.corpus, policy=policy)
    code, environment = code_and_environment_hash()
    receipt = build_receipt(args.corpus, run_result, code_hash=code, environment_hash=environment,
                            control_drop_ticker=args.drop_ticker, artifact_root=args.artifact_root)
    ok, problems = control_localized_to_analogs(receipt)
    return receipt, ok, problems


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _emit(document: dict[str, Any], out: Path | None) -> None:
    text = json.dumps(document, indent=2, sort_keys=True, default=str)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    print(text)


def _add_common(sub, *, source_root=False, store_root=False, corpus=False, scope=True):
    sub.add_argument("--root", type=Path, required=True)
    if source_root:
        sub.add_argument("--source-root", type=Path, required=True)
    if store_root:
        sub.add_argument("--store-root", type=Path, required=True)
    if corpus:
        sub.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    if scope:
        sub.add_argument("--scope", default=DEFAULT_SCOPE)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    imp = sub.add_parser("import")
    _add_common(imp, source_root=True, corpus=True)
    imp.add_argument("--out", type=Path)

    run_p = sub.add_parser("run")
    _add_common(run_p, store_root=True, corpus=True)
    run_p.add_argument("--out", type=Path, required=True)

    cmp_p = sub.add_parser("compare")
    cmp_p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    cmp_p.add_argument("--rows", type=Path, required=True)
    cmp_p.add_argument("--artifact-root", type=Path, required=True)

    ctl = sub.add_parser("control-missing-analogs")
    _add_common(ctl, source_root=True, store_root=True, corpus=True, scope=False)
    ctl.add_argument("--scope", default=DEFAULT_SCOPE + "-control")
    ctl.add_argument("--drop-ticker", required=True)
    ctl.add_argument("--artifact-root", type=Path, required=True)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    policy = NamespacePolicy({"operator": frozenset({"shadow", "smoke"})})
    try:
        return _dispatch(args, policy)
    except OpsError as exc:
        print(json.dumps({"refused": exc.code, "message": str(exc),
                          "details": exc.problem.details}, indent=2))
        return 1


def _dispatch(args, policy) -> int:
    if args.command == "import":
        _emit(import_corpus(args.root, args.source_root, args.scope, args.corpus, policy=policy),
             args.out)
        return 0
    if args.command == "run":
        _emit(run_corpus(args.root, args.store_root, args.scope, args.corpus, policy=policy),
             args.out)
        return 0
    if args.command == "compare":
        run_result = json.loads(args.rows.read_text())
        code, environment = code_and_environment_hash()
        receipt = build_receipt(args.corpus, run_result, code_hash=code, environment_hash=environment,
                                artifact_root=args.artifact_root)
        ref = publish(receipt, args.artifact_root)
        print(json.dumps({**ref, "verdict": receipt.verdict}, indent=2))
        return 0 if receipt.verdict == AGREE else 1
    receipt, ok, problems = run_control_missing_analogs(args, policy)
    ref = publish(receipt, args.artifact_root, prefix="corpus_parity_control")
    print(json.dumps({**ref, "verdict": receipt.verdict, "control_ok": ok, "problems": problems},
                     indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
