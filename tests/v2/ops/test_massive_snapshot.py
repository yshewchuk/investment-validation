"""Real catalog/input-binding/fenced native snapshots from synthetic raw pages."""
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from engine.v2.contracts import DataQuery, JobSpec, KeyPredicate, SubmitRequest
from engine.v2.data import errors, incremental, massive_daily
from engine.v2.data.price_history_table import PRICE_HISTORY_CONTRACT
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactStore, canonical_json, content_hash, to_document
from engine.v2.ops import checkpoints, input_bindings
from engine.v2.ops import snapshots as snapshot_commit
from engine.v2.ops.errors import OpsError
from engine.v2.ops.lifecycle import Outcome, commit_attempt, request_cancel
from engine.v2.ops.stages import registry
from engine.v2.ops.stores.massive import acquisition, coverage, pinned, snapshot
from engine.v2.ops.submission import submit
from tests.ops_support import POLICY
from tests.v2.ops import test_massive_coverage as support
from tests.v2.ops import test_massive_pagination as pagination
from tests.v2.ops.test_massive_submission import alternate_source as alternate_source

harness = support.harness
forbid_network = support.forbid_network


def snapshot_worker_policy():
    from tests.ops_support import TEST_POLICY

    # Generated incremental.py compilation alone peaks near 674 MiB. Identify
    # actual mutmut wrappers, not an environment flag or a directory name.
    trampoline = sys.modules.get("mutmut.mutation.trampoline")
    wrapper = getattr(trampoline, "wrap_in_trampoline", None)
    if wrapper is None or getattr(incremental, "_mutmut_mutated", None) is not wrapper:
        return TEST_POLICY
    return replace(TEST_POLICY, profiles=tuple(
        replace(profile, memory_bytes=1 << 30) if profile.name == "io_fetch" else profile
        for profile in TEST_POLICY.profiles))


def committed_output(h, attempt, document):
    ref = h.store.publish_bytes(canonical_json(document).encode(), schema_ref=acquisition.HISTORY_SCHEMA)
    def effects(conn):
        checkpoints.register_artifact(conn, ref, attempt.attempt_id, h.clock)
        conn.execute("INSERT INTO attempt_outputs VALUES (?,?,?)", (attempt.attempt_id, acquisition.KIND, ref.artifact_id))
    assert commit_attempt(h.conn, attempt.attempt_id, attempt.fence, Outcome(True, "exited", 0),
                          clock=h.clock, effects=effects) == "succeeded"
    h.clock.advance(13)
    return ref


def acquired(h, *, scope=None, key="source", payload=None, mutate=None):
    params = support.parameters(scope)
    if scope is not None and scope["dataset"] == "daily":
        params = replace(params, adjustment=scope["adjustment_basis"])
    support.enqueue(h, key, params)
    attempt = support.claim(h)
    _, document = support.run(h, attempt, support.fetching([], payload), params)
    if mutate is not None:
        mutate(document)
    return attempt.job_id, committed_output(h, attempt, document), document


def prepare(h, source_job, *, key="snapshot", **changes):
    params = snapshot.Parameters(source_job, {"acquisition.json": source_job + "#massive_acquisition"}, (source_job,))
    job = JobSpec(kind=snapshot.KIND, implementation_ref="synthetic-code", environment_ref="synthetic-env",
        spec_hash=None, parameters=to_document(params), output_namespace="shadow", resource_class="io_fetch",
        retry_policy_ref="bounded", checkpoint_contract_ref=snapshot.SCHEMA, dependency_job_ids=(source_job,))
    result = submit(h.conn, registry(), POLICY, SubmitRequest(namespace="shadow", principal="operator",
                    idempotency_key=key, job=replace(job, **changes)), clock=h.clock)
    attempt = support.claim(h)
    assert attempt is not None and attempt.job_id == result.job_id
    input_bindings.resolve_and_record(h.conn, h.store, attempt)
    return attempt, params


def execute(h, attempt, params, **kwargs):
    root = h.root / attempt.attempt_id
    root.mkdir(exist_ok=True)
    protocol = snapshot.run(to_document(params), root,
        {"attempt_id": attempt.attempt_id, "fence": attempt.fence,
         "catalog_path": str(h.root / "ops.sqlite"), "objects_root": str(h.store.root)}, clock=h.clock, **kwargs)
    doc = json.loads((root / "snapshot.json").read_bytes())
    return protocol, Repository(h.conn, h.store).resolve(doc["snapshot_id"])


def read(h, result, *, bound=1000):
    query = DataQuery(snapshot_id=result.snapshot_id,
        table_contract_ref=result.table_versions[massive_daily.TABLE].table_contract_ref,
        columns=tuple(c.name for c in massive_daily.CONTRACT.columns),
        key_filter=(KeyPredicate(column="ticker", operator="eq", values=("SYNTH",)),),
        order_by=massive_daily.CONTRACT.primary_key, max_batch_rows=1, max_result_rows=bound)
    return [r for batch in Repository(h.conn, h.store).scan(query, table_name=massive_daily.TABLE) for r in batch.to_pylist()]


def assert_fragment_provenance(h, result, artifact_id, original_receipts):
    records = Repository(h.conn, h.store).fragment_records(result, "massive_daily")
    assert len(records) == 1
    assert records[0].input_receipt_refs == tuple(sorted(set(original_receipts)))
    assert records[0].import_request_hash == content_hash([
        artifact_id, "massive_daily_normalizer.v1", massive_daily.CONTRACT.definition_hash])


@pytest.mark.parametrize("basis", ["raw", "split_adjusted"])
def test_saved_acquisition_becomes_atomic_pinned_daily_snapshot(harness, monkeypatch, basis):
    h = harness
    scope = support.request()
    scope.update(adjustment_basis=basis, params=scope["params"] | {"adjusted": basis == "split_adjusted"})
    payload = json.loads(support.body(scope)) | {"adjusted": basis == "split_adjusted"}
    payload["results"][0].update(o=13.25, h=18.5, l=11.75, c=17.125, v=137.5)
    source, ref, document = acquired(h, scope=scope, payload=canonical_json(payload).encode())
    attempt, params = prepare(h, source)
    before_budget = [tuple(r) for r in h.conn.execute("SELECT * FROM provider_reservations")]
    monkeypatch.setattr(coverage, "plan", support.no_fetch)
    monkeypatch.setattr(acquisition, "run", support.no_fetch)
    protocol, result = execute(h, attempt, params)
    rows = read(h, result, bound=1)
    original = document["coverage"]["bindings"][0]
    assert rows == [{"ticker": "SYNTH", "date": "2026-08-01", "adjustment_basis": basis,
        "source_timestamp_ms": payload["results"][0]["t"], "open": 13.25, "high": 18.5, "low": 11.75,
        "close": 17.125, "volume": 137.5, "occurrence_id": original["occurrence_id"],
        **{k: original["evidence"][k] for k in ("raw_receipt_id", "raw_hash", "first_received_at", "observed_at")}}]
    assert protocol["outputs"] == [{"name": "snapshot", "path": "snapshot.json", "schema": "snapshot_ref.v1.0"}]
    assert set(result.table_versions) == {"massive_daily"}
    assert result.table_versions["massive_daily"].table_contract_ref.contract_id == "massive_daily.v1"
    assert result.calendar_version == "massive_eastern_session_date.v1" and result.source_priority_version == "massive_only.v1"
    assert result.knowledge_mode_by_table == {"massive_daily": "reconstructed"}
    assert result.finality_receipt_refs == () and result.parent_snapshot_id is None
    resolved = Repository(h.conn, h.store).resolve_full(result.snapshot_id)
    assert resolved.table_manifests["massive_daily"].coverage_receipt_refs == (ref.artifact_id,)
    assert resolved.table_manifests["massive_daily"].availability_evidence_refs == ()
    assert_fragment_provenance(h, result, ref.artifact_id, [original["evidence"]["raw_receipt_id"]])
    assert PRICE_HISTORY_CONTRACT.contract_id == "price_history.v3"
    assert h.conn.execute("SELECT count(*) FROM data_contracts").fetchone()[0] == 1
    assert h.conn.execute("SELECT generation FROM data_snapshot_heads").fetchone()[0] == 1
    assert [tuple(r) for r in h.conn.execute("SELECT * FROM provider_reservations")] == before_budget
    before = tuple(h.conn.iterdump())
    assert execute(h, attempt, params)[1] == result
    assert tuple(h.conn.iterdump()) == before


@pytest.mark.parametrize("values", [
    {"o": 7.25, "h": 7.25, "l": 7.25, "c": 7.25, "v": 0},
    {"o": 17.5, "h": 19., "l": 16., "c": 17.5, "v": 23.5},
    {"o": 30., "h": 32., "l": 30., "c": 32., "v": 41.25},
], ids=["flat-zero-volume", "doji", "equal-ohlc-bounds"])
def test_valid_equal_prices_and_zero_volume_publish(harness, values):
    h = harness
    payload = json.loads(support.body(support.request()))
    payload["results"][0].update(values)
    source, _, _ = acquired(h, payload=canonical_json(payload).encode())
    attempt, params = prepare(h, source)
    _, result = execute(h, attempt, params)
    row = read(h, result, bound=1)[0]
    assert {key: row[key] for key in ("open", "high", "low", "close", "volume")} == {
        "open": values["o"], "high": values["h"], "low": values["l"],
        "close": values["c"], "volume": values["v"]}


@pytest.mark.parametrize("field,value", [("o", True), ("h", float("inf")), ("l", 0), ("c", -1),
    ("v", -1), ("o", "10"), ("h", 9), ("l", 12), ("c", None), ("v", 2**53 + 1), ("v", float("nan"))])
def test_bad_numeric_source_refuses_without_native_visibility(harness, field, value):
    h = harness
    payload = json.loads(support.body(support.request()))
    payload["results"][0][field] = value
    source, _, _ = acquired(h, payload=json.dumps(payload).encode())
    attempt, params = prepare(h, source)
    with pytest.raises(OpsError) as error:
        execute(h, attempt, params)
    assert error.value.code == "INTEGRITY_FAILED" and not error.value.problem.retryable
    assert h.conn.execute("SELECT count(*) FROM data_snapshots").fetchone()[0] == 0
    assert h.conn.execute("SELECT count(*) FROM data_snapshot_heads").fetchone()[0] == 0
    assert not (h.root / attempt.attempt_id / "snapshot.json").exists()


def test_postcommit_crash_recovers_on_new_attempt_under_live_fence(harness, monkeypatch):
    h = harness
    source, _, _ = acquired(h)
    attempt, params = prepare(h, source)
    original = Path.write_text
    def crash(path, *args, **kwargs):
        if path.name == "snapshot.json":
            raise OSError("synthetic postcommit interruption")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "write_text", crash)
    with pytest.raises(OpsError) as error:
        execute(h, attempt, params)
    assert error.value.code == "RESOURCE_UNAVAILABLE"
    first = dict(h.conn.execute("SELECT * FROM data_import_receipts").fetchone())
    assert support.retry(h, attempt, error.value) == "retry_wait"
    next_attempt = support.claim(h)
    assert next_attempt is not None and next_attempt.attempt_id != attempt.attempt_id
    input_bindings.resolve_and_record(h.conn, h.store, next_attempt)
    monkeypatch.setattr(Path, "write_text", original)
    monkeypatch.setattr(pinned, "normalized_rows", support.no_fetch)
    _, result = execute(h, next_attempt, params)
    assert result.snapshot_id == first["result_snapshot_id"]
    assert dict(h.conn.execute("SELECT * FROM data_import_receipts").fetchone()) == first
    assert h.conn.execute("SELECT generation FROM data_snapshot_heads").fetchone()[0] == 1


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("refusal", ["cancel", "expired"])
def test_snapshot_initial_and_replay_require_live_fence(harness, existing, refusal):
    h = harness
    source, _, _ = acquired(h)
    attempt, params = prepare(h, source)
    if existing:
        execute(h, attempt, params)
    if refusal == "cancel":
        request_cancel(h.conn, attempt.job_id, attempt.attempt_id, clock=h.clock)
    else:
        h.clock.advance(1000)
    before = tuple(h.conn.iterdump())
    with pytest.raises(OpsError) as error:
        execute(h, attempt, params)
    assert error.value.code == ("CANCELLED" if refusal == "cancel" else "LEASE_LOST")
    assert tuple(h.conn.iterdump()) == before


def test_paginated_rows_retain_each_original_page_clock_and_receipt(harness):
    h = harness
    scope, calls = support.request(), []
    params = support.parameters(cap=2)
    support.enqueue(h, params=params)
    first = support.claim(h)
    current = pagination.continuation(scope, 1, day="2026-08-02")
    payload = json.loads(daily_payload([("2026-08-01", 20)])) | {"next_url": pagination.next_url(current)}
    error = pagination.refused(h, first, params,
        pagination.fetcher(calls, scope, canonical_json(payload).encode()), "SOURCE_NOT_FINAL")
    assert support.retry(h, first, error) == "retry_wait"
    last = support.claim(h)
    _, document = support.run(h, last, pagination.fetcher(calls, current,
        daily_payload([("2026-08-02", 40)])), params)
    artifact = committed_output(h, last, document)
    pages = [json.loads(row["document_json"]) for row in pagination.pages(h)]
    assert pages[0]["observed_at"] != pages[1]["observed_at"]
    attempt, params = prepare(h, last.job_id)
    _, result = execute(h, attempt, params)
    rows = read(h, result, bound=2)
    assert [{k: r[k] for k in ("date", "open", "high", "low", "close", "volume")} for r in rows] == [
        {"date": "2026-08-01", "open": 19.5, "high": 21.5, "low": 18.5, "close": 20., "volume": 200.25},
        {"date": "2026-08-02", "open": 39.5, "high": 41.5, "low": 38.5, "close": 40., "volume": 400.25}]
    for row, page in zip(rows, pages, strict=True):
        assert {k: row[k] for k in ("raw_receipt_id", "raw_hash", "observed_at", "first_received_at")} == {
            k: page[k] for k in ("raw_receipt_id", "raw_hash", "observed_at", "first_received_at")}
    assert len({r["occurrence_id"] for r in rows}) == 1
    receipts = [page["raw_receipt_id"] for page in pages]
    assert len(set(receipts)) == 2
    assert_fragment_provenance(h, result, artifact.artifact_id, receipts)


def daily_payload(days):
    rows = []
    for day, close in days:
        row = json.loads(support.body(support.request(start=day)))["results"][0]
        rows.append(row | {"o": close - .5, "c": close, "l": close - 1.5, "h": close + 1.5,
                           "v": close * 10 + .25})
    return canonical_json({"status": "OK", "ticker": "SYNTH", "adjusted": False,
                           "resultsCount": len(rows), "results": rows}).encode()


def test_saved_clipped_mosaic_ignores_later_coverage_and_preserves_old_pin(harness, monkeypatch):
    h = harness
    acquired(h, key="wide", payload=daily_payload([("2026-08-01", 10), ("2026-08-10", 20), ("2026-08-20", 30)]))
    support.seed_overlap(h, "middle", support.request("2026-08-05", "2026-08-15"),
                         "2026-09-12T00:01:00.000000Z", payload=daily_payload([("2026-08-10", 40)]))
    scope = support.request("2026-08-01", "2026-08-22")
    source, artifact, saved = acquired(h, scope=scope, key="saved-mosaic")
    support.seed_overlap(h, "later", support.request(), "2026-09-12T00:02:00.000000Z",
                         payload=daily_payload([("2026-08-10", 100), ("2026-08-20", 100)]))
    assert coverage.plan(h.conn, h.store, scope, "initial") != saved["coverage"]
    attempt, params = prepare(h, source)
    monkeypatch.setattr(coverage, "plan", support.no_fetch)
    _, result = execute(h, attempt, params)
    assert [{k: r[k] for k in ("date", "open", "high", "low", "close", "volume")}
            for r in read(h, result, bound=3)] == [
        {"date": "2026-08-01", "open": 9.5, "high": 11.5, "low": 8.5, "close": 10., "volume": 100.25},
        {"date": "2026-08-10", "open": 39.5, "high": 41.5, "low": 38.5, "close": 40., "volume": 400.25},
        {"date": "2026-08-20", "open": 29.5, "high": 31.5, "low": 28.5, "close": 30., "volume": 300.25}]
    receipts = [binding["evidence"]["raw_receipt_id"] for binding in saved["coverage"]["bindings"]]
    assert len(receipts) == 3 and receipts[0] == receipts[2] != receipts[1]
    assert_fragment_provenance(h, result, artifact.artifact_id, receipts)
    before = tuple(h.conn.iterdump())
    with pytest.raises(errors.DataError) as error:
        read(h, result, bound=1)
    assert error.value.code == "RESULT_LIMIT_EXCEEDED"
    assert tuple(h.conn.iterdump()) == before


@pytest.mark.parametrize("mode", ["duplicate", "empty_clip"])
def test_duplicate_sessions_and_empty_selection_never_publish(harness, mode):
    h = harness
    if mode == "duplicate":
        source, _, _ = acquired(h, payload=daily_payload([("2026-08-01", 10), ("2026-08-01", 10)]))
    else:
        acquired(h, key="wide")
        source, _, _ = acquired(h, scope=support.request("2026-08-10", "2026-08-20"), key="empty-view")
    attempt, params = prepare(h, source)
    before = tuple(h.conn.iterdump())
    with pytest.raises(OpsError) as error:
        execute(h, attempt, params)
    assert error.value.code == ("INTEGRITY_FAILED" if mode == "duplicate" else "SOURCE_EMPTY")
    assert tuple(h.conn.iterdump()) == before
    assert h.conn.execute("SELECT count(*) FROM data_snapshots").fetchone()[0] == 0


@pytest.mark.parametrize("point", ["before_transaction", "after_contracts", "after_objects", "after_fragments",
    "after_dataset_versions", "after_memberships", "after_snapshot", "after_snapshot_tables", "before_head_update", "before_commit"])
def test_every_native_commit_boundary_rolls_back(harness, point):
    h = harness
    source, _, _ = acquired(h)
    attempt, params = prepare(h, source)
    before = tuple(h.conn.iterdump())
    reached = []
    def fault(where):
        if where == point:
            reached.append(where)
            raise RuntimeError("synthetic native commit interruption")
    with pytest.raises(RuntimeError, match="^synthetic native commit interruption$"):
        execute(h, attempt, params, fault=fault)
    assert reached == [point]
    assert tuple(h.conn.iterdump()) == before
    assert not (h.root / attempt.attempt_id / "snapshot.json").exists()
    assert execute(h, attempt, params)[1].table_versions


def test_interleaved_identical_builds_reconcile_only_exact_committed_receipt(harness, monkeypatch):
    h = harness
    source, _, _ = acquired(h)
    first, params = prepare(h, source, key="first-builder")
    second, second_params = prepare(h, source, key="second-builder")
    original = snapshot_commit.commit_snapshot_for_attempt
    winner, entered = [], []
    def interleave(*args, **kwargs):
        if not entered:
            entered.append(True)
            winner.append(execute(h, second, second_params)[1])
        return original(*args, **kwargs)
    monkeypatch.setattr(snapshot_commit, "commit_snapshot_for_attempt", interleave)
    _, result = execute(h, first, params)
    assert winner == [result]
    assert h.conn.execute("SELECT count(*) FROM data_import_receipts").fetchone()[0] == 1
    assert h.conn.execute("SELECT attempt_id FROM data_import_receipts").fetchone()[0] == second.attempt_id
    assert h.conn.execute("SELECT generation FROM data_snapshot_heads").fetchone()[0] == 1


@pytest.mark.parametrize("mode", ["schema", "attempt", "fence", "plan_hash", "gaps", "bound",
    "candidate_bool", "truncated", "wrong_evidence", "missing_occurrence"])
def test_malformed_saved_artifact_cannot_publish_or_replan(harness, monkeypatch, mode):
    h = harness
    def mutate(document):
        plan = document["coverage"]
        if mode == "schema":
            document["schema_version"] = "unsupported.synthetic"
        elif mode == "attempt":
            document["attempt_id"] = "att_synthetic"
        elif mode == "fence":
            document["fence"] += 1
        elif mode == "plan_hash":
            plan["plan_hash"] = "sha256:" + "0" * 64
        elif mode == "gaps":
            plan["gaps"] = [support.request()]
        elif mode == "bound":
            plan["candidate_limit"] = 999
        elif mode == "candidate_bool":
            plan["candidate_count"] = True
        elif mode == "truncated":
            plan["bindings"][0]["end"] = "2026-08-30"
        elif mode == "wrong_evidence":
            plan["bindings"][0]["evidence"]["observed_at"] = "2026-09-12T01:00:00.000000Z"
        elif mode == "missing_occurrence":
            plan["bindings"][0]["occurrence_id"] = "sha256:" + "0" * 64
        if mode != "plan_hash":
            plan["plan_hash"] = content_hash({k: v for k, v in plan.items() if k != "plan_hash"})
    source, _, _ = acquired(h, mutate=mutate)
    attempt, params = prepare(h, source)
    before = tuple(h.conn.iterdump())
    monkeypatch.setattr(coverage, "plan", support.no_fetch)
    with pytest.raises(OpsError) as error:
        execute(h, attempt, params)
    assert error.value.code == "INTEGRITY_FAILED" and not error.value.problem.retryable
    assert tuple(h.conn.iterdump()) == before
    assert not (h.root / attempt.attempt_id / "snapshot.json").exists()


@pytest.mark.parametrize("mode", ["scope", "generation"])
def test_valid_saved_plan_must_match_producer_inputs_before_normalization(harness, monkeypatch, mode):
    h = harness
    alternate_scope = support.request("2026-08-02", "2026-08-31") if mode == "scope" else support.request()
    generation = "refresh-guard" if mode == "generation" else "initial"
    _, alternate = support.acquire(h, "alternate", alternate_scope, generation=generation)
    source, _, saved = acquired(h, mutate=lambda doc: doc.update(coverage=alternate["coverage"]))
    plan = saved["coverage"]
    producer = json.loads(h.conn.execute("SELECT spec_json FROM jobs WHERE job_id=?", (source,)).fetchone()[0])
    declared = producer["parameters"]
    assert (declared["expected_ids"] != [content_hash(plan["request"])]) == (mode == "scope")
    assert (declared["generation"] != plan["generation"]) == (mode == "generation")
    # Prove all later binding/evidence/normalization guards accept this genuine plan.
    rows = pinned.normalized_rows(h.conn, h.store, plan)
    assert [(row["ticker"], row["date"], row["close"]) for row in rows] == [
        ("SYNTH", alternate_scope["start"], 11.0)]
    attempt, params = prepare(h, source)
    before = tuple(h.conn.iterdump())
    objects = {p.relative_to(h.store.root): p.read_bytes() for p in h.store.root.rglob("*") if p.is_file()}
    def unexpected_normalization(*_):
        pytest.fail("producer mismatch reached normalization")
    monkeypatch.setattr(pinned, "normalized_rows", unexpected_normalization)
    monkeypatch.setattr(coverage, "plan", support.no_fetch)
    with pytest.raises(OpsError) as error:
        execute(h, attempt, params)
    assert error.value.code == "INTEGRITY_FAILED" and not error.value.problem.retryable
    assert error.value.problem.message == "Massive saved scope differs from its producer"
    assert tuple(h.conn.iterdump()) == before
    assert {p.relative_to(h.store.root): p.read_bytes() for p in h.store.root.rglob("*") if p.is_file()} == objects
    assert not (h.root / attempt.attempt_id / "snapshot.json").exists()


@pytest.mark.parametrize("location", ["saved_binding", "catalog_occurrence"])
def test_unsupported_saved_dataset_is_integrity_failure(harness, monkeypatch, location):
    h = harness
    def mutate(document):
        if location == "saved_binding":
            plan = document["coverage"]
            plan["bindings"][0]["evidence"]["request"]["dataset"] = "unsupported-synthetic"
            plan["plan_hash"] = content_hash({k: v for k, v in plan.items() if k != "plan_hash"})
    source, _, saved = acquired(h, mutate=mutate)
    if location == "catalog_occurrence":
        binding = saved["coverage"]["bindings"][0]
        binding["evidence"]["request"]["dataset"] = "unsupported-synthetic"
        h.conn.execute("DROP TRIGGER massive_occurrences_no_update")
        h.conn.execute("UPDATE ops_massive_occurrences SET document_json=? WHERE occurrence_id=?",
                       (canonical_json(binding["evidence"]), binding["occurrence_id"]))
    attempt, params = prepare(h, source)
    before = tuple(h.conn.iterdump())
    objects = sorted(p.relative_to(h.store.root) for p in h.store.root.rglob("*") if p.is_file())
    monkeypatch.setattr(coverage, "plan", support.no_fetch)
    with pytest.raises(OpsError) as error:
        execute(h, attempt, params)
    assert error.value.code == "INTEGRITY_FAILED" and not error.value.problem.retryable
    assert error.value.problem.message == "Massive saved evidence scope is invalid"
    assert tuple(h.conn.iterdump()) == before
    assert sorted(p.relative_to(h.store.root) for p in h.store.root.rglob("*") if p.is_file()) == objects
    assert not (h.root / attempt.attempt_id / "snapshot.json").exists()


def test_saved_selected_occurrences_cannot_exceed_declared_candidates(harness, monkeypatch):
    h = harness
    support.seed_overlap(h, "wide", support.request(), "2026-09-10T12:00:00.000000Z")
    support.seed_overlap(h, "middle", support.request("2026-08-05", "2026-08-15"),
                         "2026-09-11T12:00:00.000000Z")
    def mutate(document):
        plan = document["coverage"]
        assert plan["candidate_count"] == 2 and len(plan["bindings"]) == 3
        plan["candidate_count"] = 1
        plan["plan_hash"] = content_hash({k: v for k, v in plan.items() if k != "plan_hash"})
    source, _, _ = acquired(h, mutate=mutate)
    attempt, params = prepare(h, source)
    before = tuple(h.conn.iterdump())
    def unexpected_verification(*_, **__):
        pytest.fail("selected occurrence overflow reached raw verification")
    monkeypatch.setattr(coverage, "_verified", unexpected_verification)
    with pytest.raises(OpsError) as error:
        execute(h, attempt, params)
    assert error.value.code == "INTEGRITY_FAILED" and not error.value.problem.retryable
    assert tuple(h.conn.iterdump()) == before
    assert not (h.root / attempt.attempt_id / "snapshot.json").exists()


def test_distinct_acquisition_pins_share_original_fragment_provenance(harness):
    h = harness
    results, artifacts, records = [], [], []
    for index in range(2):
        source, artifact, saved = acquired(h, key=f"source-{index}")
        attempt, params = prepare(h, source, key=f"snapshot-{index}")
        _, result = execute(h, attempt, params)
        resolved = Repository(h.conn, h.store).resolve_full(result.snapshot_id)
        assert resolved.table_manifests[massive_daily.TABLE].coverage_receipt_refs == (artifact.artifact_id,)
        records.append(Repository(h.conn, h.store).fragment_records(result, massive_daily.TABLE))
        results.append(result)
        artifacts.append(artifact.artifact_id)
        support.finish(h, attempt)
    assert artifacts[0] != artifacts[1] and results[0].snapshot_id != results[1].snapshot_id
    assert h.conn.execute("SELECT count(*) FROM data_snapshot_heads").fetchone()[0] == 2
    assert records[0] == records[1]
    original = saved["coverage"]["bindings"][0]["evidence"]["raw_receipt_id"]
    assert records[1][0].input_receipt_refs == (original,)
    assert records[1][0].import_request_hash == content_hash([
        artifacts[0], massive_daily.NORMALIZER, massive_daily.CONTRACT.definition_hash])
    assert records[1][0].import_request_hash != content_hash([
        artifacts[1], massive_daily.NORMALIZER, massive_daily.CONTRACT.definition_hash])


@pytest.mark.parametrize("mode", ["hole", "overlap", "lower_containment", "upper_containment", "family", "generation"])
def test_rehashed_binding_guards_refuse_independently_valid_evidence(harness, monkeypatch, mode):
    h = harness
    alternate = None
    if mode == "family":
        identity, evidence = support.seed_overlap(h, "other-family", support.request(ticker="OTHER"),
                                                 "2026-09-11T12:00:00.000000Z")
        alternate = {"occurrence_id": identity, "evidence": evidence}
    elif mode == "generation":
        _, record = support.acquire(h, "other-generation", generation="refresh-guard")
        alternate = record["coverage"]["bindings"][0]
    else:
        acquired(h, key="wide", payload=daily_payload([("2026-08-01", 10), ("2026-08-10", 20), ("2026-08-20", 30)]))
        support.seed_overlap(h, "middle", support.request("2026-08-05", "2026-08-15"),
                             "2026-09-12T00:01:00.000000Z", payload=daily_payload([("2026-08-10", 40)]))
    def mutate(document):
        plan = document["coverage"]
        bindings = plan["bindings"]
        if alternate is not None:
            bindings[0].update({k: alternate[k] for k in ("occurrence_id", "evidence")})
        elif mode in ("hole", "overlap"):
            bindings[0]["end"] = "2026-08-03" if mode == "hole" else "2026-08-05"
        elif mode == "lower_containment":
            bindings[0]["end"], bindings[1]["start"] = "2026-08-03", "2026-08-04"
        else:
            bindings[1]["end"], bindings[2]["start"] = "2026-08-16", "2026-08-17"
        plan["plan_hash"] = content_hash({k: v for k, v in plan.items() if k != "plan_hash"})
    source, _, saved = acquired(h, key="saved", mutate=mutate)
    plan = saved["coverage"]
    assert pinned._plan_identity(plan) == support.request() and plan["generation"] == "initial"
    for binding in plan["bindings"]:
        row = h.conn.execute("SELECT * FROM ops_massive_occurrences WHERE occurrence_id=?",
                             (binding["occurrence_id"],)).fetchone()
        assert coverage._verified(h.conn, h.store, row, [coverage.MAX_VERIFIED_BYTES]) == binding["evidence"]
    attempt, params = prepare(h, source)
    before = tuple(h.conn.iterdump())
    objects = sorted(p.relative_to(h.store.root) for p in h.store.root.rglob("*") if p.is_file())
    monkeypatch.setattr(coverage, "plan", support.no_fetch)
    with pytest.raises(OpsError) as error:
        execute(h, attempt, params)
    assert error.value.code == "INTEGRITY_FAILED" and not error.value.problem.retryable
    assert tuple(h.conn.iterdump()) == before
    assert not (h.root / attempt.attempt_id / "snapshot.json").exists()
    assert sorted(p.relative_to(h.store.root) for p in h.store.root.rglob("*") if p.is_file()) == objects


def test_artifact_byte_limit_precedes_worker_read(harness, monkeypatch):
    h = harness
    source, ref, _ = acquired(h, mutate=lambda doc: doc.update(synthetic_padding="x" * (4 << 20)))
    assert ref.byte_size > 4 << 20
    attempt, params = prepare(h, source)
    monkeypatch.setattr(ArtifactStore, "read_verified", support.no_fetch)
    before = tuple(h.conn.iterdump())
    with pytest.raises(OpsError) as error:
        execute(h, attempt, params)
    assert error.value.code == "RESOURCE_LIMIT_EXCEEDED"
    assert tuple(h.conn.iterdump()) == before


@pytest.mark.parametrize("kind", ["raw_object", "native_object", "success_slot"])
def test_corrupt_evidence_or_replay_slot_refuses_without_repair(harness, kind):
    h = harness
    source, _, saved = acquired(h)
    attempt, params = prepare(h, source)
    if kind in ("native_object", "success_slot"):
        _, result = execute(h, attempt, params)
    if kind == "success_slot":
        h.conn.execute("DROP TRIGGER data_import_receipts_no_update")
        h.conn.execute("UPDATE data_import_receipts SET source_manifest_hash=?", ("sha256:" + "0" * 64,))
    else:
        digest = (saved["coverage"]["bindings"][0]["evidence"]["raw_hash"] if kind == "raw_object" else
                  Repository(h.conn, h.store).resolve_full(result.snapshot_id).objects[0].content_hash)
        key = digest.removeprefix("sha256:")
        alternate = h.root / "corrupt-copy"
        for path in h.store.root.rglob("*"):
            if path.is_file():
                copy = alternate / path.relative_to(h.store.root)
                copy.parent.mkdir(parents=True, exist_ok=True)
                copy.write_bytes(b"synthetic corrupt bytes" if path.name == key else path.read_bytes())
        h.store = ArtifactStore(alternate)
    before = tuple(h.conn.iterdump())
    with pytest.raises(OpsError) as error:
        execute(h, attempt, params)
    assert error.value.code == "INTEGRITY_FAILED"
    assert tuple(h.conn.iterdump()) == before


@pytest.mark.parametrize("source_mode", ["default", "override", "mismatch"])
def test_real_submission_service_worker_and_bounded_pinned_read(
        tmp_path, monkeypatch, capsys, alternate_source, source_mode):
    from types import SimpleNamespace

    from engine import paths
    from engine.v2.data.incremental import RawPayload, cache_raw_receipt
    from engine.v2.foundation import SystemClock
    from engine.v2.ops import executor, massive_acquire, massive_snapshot, providers
    from engine.v2.ops.bootstrap import open_catalog
    from engine.v2.ops.provider_budget import configure_account
    from engine.v2.ops.supervisor import Service
    from tests.ops_support import run_until
    from tests.v2.ops.test_massive_submission import arguments

    checkout, selected, _, _ = alternate_source
    monkeypatch.setattr(paths, "ROOT", checkout if source_mode == "default" else selected)
    code_source = selected if source_mode == "override" else checkout
    clock, store = SystemClock(), ArtifactStore(tmp_path)
    conn = open_catalog(tmp_path / "catalog.sqlite", clock=clock)
    configure_account(conn, support.SOURCE, "synthetic", 10, 0)
    scope = support.request()
    cache_raw_receipt(conn, store, RawPayload(payload=support.body(scope), response_kind="complete", response_meta={}),
        source=support.SOURCE, endpoint=scope["endpoint"], request=scope, received_at="2026-09-12T00:00:00.000000Z")
    monkeypatch.setattr(providers, "os", SimpleNamespace(environ={"MASSIVE_API_KEY": "synthetic-not-a-key"}))
    original, launched = executor.subprocess.Popen, []
    def launch(command, **kwargs):
        environment = dict(kwargs["env"])
        launched.append(environment.pop("MASSIVE_API_KEY", None))
        # Poison actual child sockets before the unchanged worker entrypoint.
        assert command[-2:] == ["-m", "engine.v2.ops.worker"]
        guarded = command[:-2] + ["-c", "import socket, runpy; "
            "socket.socket.connect=lambda *a: (_ for _ in ()).throw(AssertionError('synthetic network guard')); "
            "runpy.run_module('engine.v2.ops.worker', run_name='__main__')"]
        return original(guarded, **(kwargs | {"env": environment}))
    monkeypatch.setattr(executor.subprocess, "Popen", launch)
    service = Service(conn, tmp_path, registry(), snapshot_worker_policy(), clock=clock,
                      code_source=code_source, store_root=tmp_path / "synthetic-legacy")
    service.start()
    try:
        assert massive_acquire.main(arguments(tmp_path) + ["--generation", "initial"], clock=clock) == 0
        source = json.loads(capsys.readouterr().out)["job_id"]
        state = run_until(service, conn, source, timeout=30)
        if source_mode == "mismatch":
            assert state == "failed"
            failure = conn.execute("SELECT failure_json FROM jobs WHERE job_id=?", (source,)).fetchone()[0]
            assert json.loads(failure)["code"] == "INPUT_CHANGED"
            assert launched == []
            assert conn.execute("SELECT SUM(used_calls) FROM provider_reservations").fetchone()[0] == 0
            assert conn.execute("SELECT count(*) FROM data_snapshots").fetchone()[0] == 0
            return
        assert state == "succeeded"
        capsys.readouterr()  # Supervisor diagnostics are separate from the submit CLI's response.
        monkeypatch.setattr(providers, "provider_credentials", support.no_fetch)
        argv = ["--root", str(tmp_path), "--acquisition-job", source, "--idempotency-key", "daily-native"]
        assert massive_snapshot.main(argv, clock=clock) == 0
        target = json.loads(capsys.readouterr().out)["job_id"]
        assert massive_snapshot.main(argv, clock=clock) == 0
        assert json.loads(capsys.readouterr().out)["job_id"] == target
        assert run_until(service, conn, target, timeout=30) == "succeeded"
        row = conn.execute("SELECT artifact_id FROM attempt_outputs JOIN attempts USING(attempt_id) "
                           "WHERE job_id=? AND name='snapshot'", (target,)).fetchone()
        ref = checkpoints.artifact(conn, store, row[0])
        result = Repository(conn, store).resolve(json.loads(store.read_verified(ref))["snapshot_id"])
        h = SimpleNamespace(conn=conn, store=store)
        assert read(h, result, bound=1)[0]["close"] == 11
        assert launched == ["synthetic-not-a-key", None]
        assert conn.execute("SELECT count(*) FROM checkpoints").fetchone()[0] == 0
        assert conn.execute("SELECT SUM(used_calls) FROM provider_reservations").fetchone()[0] == 0
        assert len(conn.execute("SELECT * FROM attempt_input_bindings").fetchall()) == 1
    finally:
        service.close()
        conn.close()


def test_replay_rechecks_cancellation_after_reading_prior_success(harness, monkeypatch):
    h = harness
    source, _, _ = acquired(h)
    attempt, params = prepare(h, source)
    _, expected = execute(h, attempt, params)
    original = snapshot._existing
    calls = []
    def cancelled(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result == expected
        calls.append(True)
        request_cancel(h.conn, attempt.job_id, attempt.attempt_id, clock=h.clock)
        return result
    monkeypatch.setattr(snapshot, "_existing", cancelled)
    with pytest.raises(OpsError) as error:
        execute(h, attempt, params)
    assert error.value.code == "CANCELLED" and calls == [True]
    assert h.conn.execute("SELECT generation FROM data_snapshot_heads").fetchone()[0] == 1


def test_unrelated_existing_head_cannot_be_adopted_or_replaced(harness):
    h = harness
    first_source, _, _ = acquired(h)
    first, first_params = prepare(h, first_source)
    _, original = execute(h, first, first_params)
    support.finish(h, first)
    source, artifact, _ = acquired(h, key="other-acquisition")
    identity = content_hash([artifact.artifact_id, massive_daily.NORMALIZER, massive_daily.CONTRACT.definition_hash])
    h.conn.execute("INSERT INTO data_snapshot_heads SELECT ?,snapshot_id,generation,updated_at,update_receipt_ref "
                   "FROM data_snapshot_heads", (identity,))
    attempt, params = prepare(h, source, key="other-snapshot")
    before = tuple(h.conn.iterdump())
    with pytest.raises(OpsError) as error:
        execute(h, attempt, params)
    assert error.value.code == "RESOURCE_UNAVAILABLE"
    assert tuple(h.conn.iterdump()) == before
    assert read(h, original, bound=1)[0]["close"] == 11


@pytest.mark.parametrize("stamp", ["malformed-synthetic-clock", "2026-09-12T00:00:00.0Z"])
def test_fresh_response_with_bad_retained_first_receipt_clock_refuses(harness, stamp):
    from engine.v2.data.incremental import RawPayload, cache_raw_receipt

    h, scope = harness, support.request()
    cache_raw_receipt(h.conn, h.store, RawPayload(payload=support.body(scope), response_kind="complete", response_meta={}),
        source=support.SOURCE, endpoint=scope["endpoint"], request=scope, received_at=stamp)
    params = support.parameters(generation="refresh")
    support.enqueue(h, "refresh", params)
    producer = support.claim(h)
    _, document = support.run(h, producer, support.fetching([]), params)
    assert document["coverage"]["bindings"][0]["evidence"]["first_received_at"] == stamp
    committed_output(h, producer, document)
    attempt, params = prepare(h, producer.job_id)
    with pytest.raises(OpsError) as error:
        execute(h, attempt, params)
    assert error.value.code == "INTEGRITY_FAILED"
    assert h.conn.execute("SELECT count(*) FROM data_snapshots").fetchone()[0] == 0


def test_source_timestamp_uses_eastern_session_date(harness):
    from datetime import datetime

    h = harness
    payload = json.loads(support.body(support.request()))
    stamp = int(datetime.fromisoformat("2026-08-02T02:00:00+00:00").timestamp() * 1000)
    payload["results"][0]["t"] = stamp
    source, _, _ = acquired(h, payload=canonical_json(payload).encode())
    attempt, params = prepare(h, source)
    _, result = execute(h, attempt, params)
    row = read(h, result, bound=1)[0]
    assert (row["date"], row["source_timestamp_ms"]) == ("2026-08-01", stamp)


@pytest.mark.parametrize("mode", ["rows", "raw_bytes"])
def test_explicit_normalization_limits_refuse_before_native_publication(harness, monkeypatch, mode):
    h = harness
    payload = daily_payload([("2026-08-01", 10)] * (1001 if mode == "rows" else 1))
    source, _, _ = acquired(h, payload=payload)
    attempt, params = prepare(h, source)
    if mode == "raw_bytes":
        monkeypatch.setattr(coverage, "MAX_VERIFIED_BYTES", len(payload) - 1)
    before = tuple(h.conn.iterdump())
    with pytest.raises(OpsError) as error:
        execute(h, attempt, params)
    assert error.value.code == "RESOURCE_LIMIT_EXCEEDED"
    assert tuple(h.conn.iterdump()) == before


def test_repeated_occurrence_decoding_exceeds_bound_after_verification_succeeds(harness, monkeypatch):
    h = harness
    wide = daily_payload([("2026-08-01", 10), ("2026-08-10", 20), ("2026-08-20", 30)])
    middle = daily_payload([("2026-08-10", 40)])
    acquired(h, key="wide", payload=wide)
    support.seed_overlap(h, "middle", support.request("2026-08-05", "2026-08-15"),
                         "2026-09-12T00:01:00.000000Z", payload=middle)
    source, _, saved = acquired(h, key="repeated-mosaic")
    bindings = saved["coverage"]["bindings"]
    assert [(b["start"], b["end"]) for b in bindings] == [
        ("2026-08-01", "2026-08-04"), ("2026-08-05", "2026-08-15"), ("2026-08-16", "2026-08-31")]
    identities = [b["occurrence_id"] for b in bindings]
    assert identities[0] == identities[2] != identities[1]
    attempt, params = prepare(h, source)
    limit = len(wide) + len(middle)
    monkeypatch.setattr(coverage, "MAX_VERIFIED_BYTES", limit)
    original_verify, original_load = coverage._verified, pinned.load_raw_receipt
    verified, decoded = [], []
    def verify(conn, store, row, remaining):
        evidence = original_verify(conn, store, row, remaining)
        verified.append((row["occurrence_id"], remaining[0]))
        return evidence
    def load(conn, store, receipt_id):
        body = original_load(conn, store, receipt_id)
        decoded.append((receipt_id, len(body)))
        return body
    monkeypatch.setattr(coverage, "_verified", verify)
    monkeypatch.setattr(pinned, "load_raw_receipt", load)
    before = tuple(h.conn.iterdump())
    objects = sorted(p.relative_to(h.store.root) for p in h.store.root.rglob("*") if p.is_file())
    with pytest.raises(OpsError) as error:
        execute(h, attempt, params)
    assert error.value.code == "RESOURCE_LIMIT_EXCEEDED"
    assert error.value.problem.message == "Massive normalization exceeds raw byte bound"
    assert verified == [(identities[0], len(middle)), (identities[1], 0)]
    assert decoded == [(b["evidence"]["raw_receipt_id"], size)
                       for b, size in zip(bindings, [len(wide), len(middle), len(wide)], strict=True)]
    assert sum(size for _, size in decoded[:2]) == limit < sum(size for _, size in decoded)
    assert tuple(h.conn.iterdump()) == before
    assert h.conn.execute("SELECT count(*) FROM data_snapshots").fetchone()[0] == 0
    assert not (h.root / attempt.attempt_id / "snapshot.json").exists()
    assert sorted(p.relative_to(h.store.root) for p in h.store.root.rglob("*") if p.is_file()) == objects


@pytest.mark.parametrize("flags", [[], ["--acquisition-job", "invalid"], ["--unexpected", "synthetic"]])
def test_snapshot_cli_invalid_arguments_precede_fingerprints_and_storage(tmp_path, monkeypatch, capsys, flags):
    from engine.v2.ops import massive_snapshot as cli

    monkeypatch.setattr(cli.fingerprints, "worker_source_manifest", support.no_fetch)
    monkeypatch.setattr(cli.catalog, "connect", support.no_fetch)
    args = [] if not flags else ["--root", str(tmp_path), "--idempotency-key", "test", *flags]
    assert cli.main(args) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"refused": "INVALID_REQUEST"} and captured.err == ""
    assert not (tmp_path / "catalog.sqlite").exists()


@pytest.mark.parametrize("changed", ["bindings", "expected", "input_ref", "dependency", "provider"])
def test_snapshot_allowlist_rejects_unbound_parameters(harness, changed):
    h = harness
    source, _, _ = acquired(h)
    params = snapshot.Parameters(source, {"acquisition.json": source + "#massive_acquisition"}, (source,))
    overrides = {}
    if changed == "bindings":
        params = replace(params, input_bindings={})
    elif changed == "expected":
        params = replace(params, expected_ids=())
    elif changed == "input_ref":
        overrides["input_refs"] = ("artifact_synthetic",)
    elif changed == "dependency":
        overrides["dependency_job_ids"] = ()
    else:
        overrides["provider_budget_ref"] = support.SOURCE
    before = tuple(h.conn.iterdump())
    with pytest.raises(OpsError) as error:
        prepare(h, source, parameters=to_document(params), **overrides)
    assert error.value.code == "INVALID_REQUEST"
    assert tuple(h.conn.iterdump()) == before
