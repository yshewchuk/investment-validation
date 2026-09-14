"""D19: the real render-parity comparator -- synthetic, tier 2.

The catalog rows (job, attempt, reservation, input bindings, committed
``bundle.tar`` output) are produced by the REAL scheduler/catalog machinery
(``submit``, ``claim_next``, ``resolve_and_record``, ``ArtifactStore.
publish_candidate``, ``record_measurement``, ``commit_attempt``) -- not hand
-written SQL -- against the real ``engine.v2.ops.stages.registry()`` kind.
The v2 bundle itself comes from ``_action_render`` called in-process (the
same synthetic pattern ``tests/test_v2_ops_render_parity.py`` already
established for D19). The legacy-way rebuild and the selfcheck genuinely run
in bounded subprocesses (``PHASE2_RENDER_PARITY_TEST_PATCH`` monkeypatches
the heavy ``FeatureContext.load``/``Scorer``/``selfcheck`` loaders inside
them -- see ``tests/render_parity_support.py``). No real market data, no
network, no `INVESTING_PLAN_ROOT` pointed at a real store.
"""
from __future__ import annotations

import importlib
import json
import os

import pytest

import engine.features as features_module
import engine.score as score_module
from checks.rearchitecture_phase2_evidence import PHASE2_EVIDENCE_V1, validate_evidence
from checks.rearchitecture_phase2_render_parity import (
    build_render_comparison_receipt,
    main,
    publish_receipt,
)
from engine import paths
from engine.v2.contracts import JobSpec, SubmitRequest
from engine.v2.foundation import ArtifactStore
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.catalog import transaction
from engine.v2.ops.checkpoints import register_artifact
from engine.v2.ops.input_bindings import resolve_and_record
from engine.v2.ops.legacy_adapter import _action_render
from engine.v2.ops.lifecycle import Outcome, commit_attempt, record_measurement
from engine.v2.ops.profiles import DEFAULT_POLICY, GIB
from engine.v2.ops.recovery import begin_epoch
from engine.v2.ops.scheduler import Supervisor, claim_next
from engine.v2.ops.stages import registry as stages_registry
from engine.v2.ops.submission import NamespacePolicy, submit
from tests.ops_support import FakeClock, sample
from tests.render_parity_support import (
    AS_OF,
    TICKER,
    build_ledger_generation_tar,
    finality_doc,
    model_evidence_doc,
    patch_scorer,
    score_document,
)

_PARAMETERS = {"expected_ids": ("legacy_render",), "session": str(AS_OF.date()),
               "tickers": [TICKER], "year_start": 2025, "year_end": 2026,
               "horizon_days": 35, "alt_strikes": 1}


def _publish(store, conn, clock, data, *, schema_ref):
    ref = store.publish_bytes(data, schema_ref=schema_ref)
    with transaction(conn):
        register_artifact(conn, ref, None, clock)
    return ref


def _submit_and_claim(conn, clock, ops_root, *, input_bindings, input_refs):
    parameters = dict(_PARAMETERS, input_bindings=dict(input_bindings))
    spec = JobSpec(kind="legacy_render", implementation_ref="code", spec_hash=None,
                   environment_ref="env", parameters=parameters, input_refs=tuple(input_refs),
                   output_namespace="shadow", resource_class="projection",
                   retry_policy_ref="bounded", checkpoint_contract_ref="legacy_action.v1.0")
    request = SubmitRequest(namespace="shadow", idempotency_key="render1", principal="operator",
                            job=spec)
    policy = NamespacePolicy({"operator": frozenset({"shadow"})})
    receipt = submit(conn, stages_registry(), policy, request, clock=clock)
    epoch = begin_epoch(conn, clock=clock, boot_id="boot", pid=1)
    claim = claim_next(conn, policy=DEFAULT_POLICY, sample=sample(clock),
                       supervisor=Supervisor(epoch, "boot"), clock=clock,
                       registry=stages_registry())
    assert claim is not None and claim.job_id == receipt.job_id
    return receipt.job_id, claim


def _point_legacy_root(monkeypatch, legacy_root):
    monkeypatch.setenv("INVESTING_PLAN_ROOT", str(legacy_root))
    importlib.reload(paths)


@pytest.fixture
def seeded_render_job(tmp_path, monkeypatch):
    """A real ``legacy_render`` job/attempt with a committed ``bundle.tar``.

    Yields ``(ops_root, job_id)``; ``INVESTING_PLAN_ROOT`` is restored on
    teardown, matching ``tests/test_v2_ops_render_parity.py``'s own
    ``_pointed_paths`` fixture.
    """
    patch_scorer(monkeypatch, features_module, score_module)
    ops_root = tmp_path / "ops_root"
    ops_root.mkdir()
    clock = FakeClock()
    conn = open_catalog(ops_root / "catalog.sqlite", clock=clock)
    store = ArtifactStore(ops_root)

    score_bytes = json.dumps(score_document()).encode()
    finality_bytes = json.dumps(finality_doc()).encode()
    evidence_bytes = json.dumps(model_evidence_doc()).encode()
    ledger_bytes = build_ledger_generation_tar(tmp_path)
    refs = {name: _publish(store, conn, clock, data, schema_ref="legacy_action.v1.0")
           for name, data in (("score.json", score_bytes), ("finality.json", finality_bytes),
                              ("model_evidence.json", evidence_bytes),
                              ("ledger_generation.tar", ledger_bytes))}
    bindings = {name: ref.artifact_id for name, ref in refs.items()}
    job_id, claim = _submit_and_claim(conn, clock, ops_root, input_bindings=bindings,
                                      input_refs=[r.artifact_id for r in refs.values()])
    resolve_and_record(conn, store, claim)

    staging = store.staging_dir(claim.attempt_id)
    for name, ref in refs.items():
        (staging / name).write_bytes(store.read_verified(ref))
    _point_legacy_root(monkeypatch, staging / "legacy")
    _action_render(dict(_PARAMETERS, tickers=[TICKER]), staging)

    bundle_ref = store.publish_candidate(claim.attempt_id, "bundle.tar",
                                         schema_ref="legacy_action.v1.0",
                                         max_bytes=claim.resources.scratch_limit_bytes)
    with transaction(conn):
        register_artifact(conn, bundle_ref, claim.attempt_id, clock)
        conn.execute("INSERT INTO attempt_outputs VALUES (?,?,?)",
                     (claim.attempt_id, "legacy_render", bundle_ref.artifact_id))
    record_measurement(conn, claim.attempt_id, current_bytes=int(1.2 * GIB),
                       peak_bytes=int(1.2 * GIB), clock=clock)
    commit_attempt(conn, claim.attempt_id, claim.fence, Outcome(True, "verified_dead", 0),
                   clock=clock)
    conn.close()
    # The facts are explicit: the legacy-way rebuild runs against "the private
    # root the render job used ... never the live store" -- its OWN staging
    # copy, not an unrelated fresh directory. A fresh directory here would
    # embed a DIFFERENT (and therefore spuriously differing) path into any
    # flag whose detail string names a missing file under INVESTING_PLAN_ROOT
    # (e.g. panel-coverage-unmeasurable), which is an artifact of the test
    # picking two different roots, not a real D19 disagreement.
    yield ops_root, job_id, staging / "legacy"


@pytest.fixture
def patched_env(monkeypatch):
    env = dict(os.environ, PHASE2_RENDER_PARITY_TEST_PATCH="tests.render_parity_support")
    monkeypatch.setenv("PHASE2_RENDER_PARITY_TEST_PATCH", "tests.render_parity_support")
    return env


def test_identical_inputs_agree(seeded_render_job, patched_env, tmp_path):
    ops_root, job_id, legacy_root = seeded_render_job
    receipt = build_render_comparison_receipt(
        root=ops_root, render_job=job_id, legacy_root=legacy_root, max_rss_gb=4.0)
    assert receipt.verdict == "agree", receipt.summary()
    assert receipt.comparison_kind == "render_bundle_parity"
    assert receipt.population.compared > 0
    assert receipt.population.compared == receipt.population.expected


def test_selfcheck_failing_makes_the_receipt_disagree(seeded_render_job, monkeypatch, tmp_path):
    monkeypatch.setenv("PHASE2_RENDER_PARITY_TEST_PATCH", "tests.render_parity_support")
    monkeypatch.setenv("PHASE2_RENDER_PARITY_TEST_SELFCHECK_OK", "0")
    ops_root, job_id, legacy_root = seeded_render_job
    receipt = build_render_comparison_receipt(
        root=ops_root, render_job=job_id, legacy_root=legacy_root, max_rss_gb=4.0)
    assert receipt.verdict == "differ"
    assert any(f.field_path == "__selfcheck__.ok" for f in receipt.findings)
    assert any(p["code"] == "SELFCHECK_FAILED" for p in receipt.problems)


def test_cli_publishes_and_exits_zero_on_agree(seeded_render_job, patched_env, tmp_path, capsys):
    ops_root, job_id, legacy_root = seeded_render_job
    artifact_root = tmp_path / "evidence"
    code = main(["--root", str(ops_root), "--render-job", job_id,
                "--legacy-root", str(legacy_root), "--artifact-root", str(artifact_root),
                "--max-rss-gb", "4.0"])
    assert code == 0
    printed = json.loads(capsys.readouterr().out.strip())
    assert printed["verdict"] == "agree"
    published = artifact_root / printed["path"]
    assert published.is_file()
    import hashlib
    assert "sha256:" + hashlib.sha256(published.read_bytes()).hexdigest() == printed["content_hash"]


def test_receipt_passes_the_evidence_validators_render_kind_and_binding_checks(
        seeded_render_job, patched_env, tmp_path):
    ops_root, job_id, legacy_root = seeded_render_job
    artifact_root = tmp_path / "evidence"
    receipt = build_render_comparison_receipt(
        root=ops_root, render_job=job_id, legacy_root=legacy_root, max_rss_gb=4.0)
    ref = publish_receipt(receipt, artifact_root)
    evidence = {"schema_version": PHASE2_EVIDENCE_V1, "code_hash": receipt.envelope.code_hash,
               "environment_hash": receipt.envelope.environment_hash,
               "authority_mode": "shadow", "render_comparison_receipt_ref": ref}
    findings, field_ok, document_ok = validate_evidence(
        evidence, artifact_root=artifact_root, code_hash=receipt.envelope.code_hash,
        environment_hash=receipt.envelope.environment_hash)
    assert document_ok
    assert field_ok.get("render_comparison_receipt_ref") is True
    assert not any(f["field"] == "render_comparison_receipt_ref" for f in findings
                  if "field" in f)
