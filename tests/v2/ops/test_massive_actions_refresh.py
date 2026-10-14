"""Real pinned action reconstruction and parent-bound atomic interval replacement."""
import json
from pathlib import Path

import pytest

from engine.v2.contracts import DataQuery, JobSpec, KeyPredicate, SubmitRequest
from engine.v2.data import massive_actions, massive_daily, massive_refresh
from engine.v2.data.repository import Repository
from engine.v2.foundation import canonical_json, content_hash, to_document
from engine.v2.ops import input_bindings, massive_snapshot
from engine.v2.ops.errors import OpsError
from engine.v2.ops.stages import registry
from engine.v2.ops.stores.massive import pinned, snapshot
from engine.v2.ops.submission import submit
from tests.ops_support import POLICY
from tests.v2.ops import test_massive_coverage as support
from tests.v2.ops import test_massive_pagination as pagination
from tests.v2.ops import test_massive_snapshot as daily

harness = support.harness
forbid_network = support.forbid_network


def action(dataset, identity="event-1", day="2026-08-10", ticker="SYNTH"):
    if dataset == "splits":
        return {"id": identity, "ticker": ticker, "execution_date": day,
            "split_from": 3., "split_to": 7., "adjustment_type": "stock_dividend",
            "historical_adjustment_factor": 0.125}
    return {"id": identity, "ticker": ticker, "ex_dividend_date": day,
        "cash_amount": 1.25, "currency": "CAD", "declaration_date": "2026-07-22",
        "record_date": "2026-08-11", "pay_date": "2026-09-17", "frequency": 4,
        "distribution_type": "supplemental", "split_adjusted_cash_amount": 0.625,
        "historical_adjustment_factor": 0.9375}


def acquired(h, dataset, *, key, rows=None, start="2026-08-01", end="2026-08-31", ticker="SYNTH"):
    scope = support.request(start, end, dataset, ticker)
    return daily.acquired(h, scope=scope, key=key,
        payload=canonical_json({"status": "OK", "results": rows or [action(dataset)]}).encode())


def prepare(h, source, *, key, parent=None, scope=None, generation=None):
    params = snapshot.Parameters(source, {"acquisition.json": source + "#massive_acquisition"},
        (source,), parent.snapshot_id if parent else None, scope, generation)
    job = JobSpec(kind=snapshot.KIND, implementation_ref="synthetic-code", environment_ref="synthetic-env",
        spec_hash=None, parameters=to_document(params), output_namespace="shadow", resource_class="io_fetch",
        retry_policy_ref="bounded", checkpoint_contract_ref=snapshot.SCHEMA, dependency_job_ids=(source,))
    result = submit(h.conn, registry(), POLICY, SubmitRequest(namespace="shadow", principal="operator",
                    idempotency_key=key, job=job), clock=h.clock)
    attempt = support.claim(h)
    assert attempt is not None and attempt.job_id == result.job_id
    input_bindings.resolve_and_record(h.conn, h.store, attempt)
    return attempt, params


def finish_snapshot(h, attempt, params):
    result = daily.execute(h, attempt, params)[1]
    support.finish(h, attempt)
    h.clock.advance(13)
    return result


def read(h, result, contract, ticker="SYNTH"):
    repo = Repository(h.conn, h.store)
    predicate = (KeyPredicate(column="ticker", operator="eq", values=(ticker,)),)
    ref = result.table_versions[contract.table_name].table_contract_ref
    bound = repo.scan_population_bound(result.snapshot_id, table_name=contract.table_name,
                                       table_contract_ref=ref, key_filter=predicate)
    query = DataQuery(snapshot_id=result.snapshot_id, table_contract_ref=ref,
        columns=tuple(c.name for c in contract.columns), key_filter=predicate,
        order_by=contract.primary_key, max_batch_rows=1, max_result_rows=bound)
    return [r for b in repo.scan(query, table_name=contract.table_name) for r in b.to_pylist()]


def bootstrap(h):
    source, ref, _ = daily.acquired(h)
    attempt, params = prepare(h, source, key="bootstrap")
    result = finish_snapshot(h, attempt, params)
    scope = h.conn.execute("SELECT scope FROM data_snapshot_heads").fetchone()[0]
    return result, scope, ref


@pytest.mark.parametrize("dataset", ["splits", "dividends"])
def test_action_values_ids_clocks_pins_and_same_day_events(harness, dataset):
    h = harness
    values = [action(dataset, "event-2"), action(dataset)]
    source, ref, document = acquired(h, dataset, key="actions", rows=values)
    before_calls = h.conn.execute("SELECT SUM(used_calls) FROM provider_reservations").fetchone()[0]
    attempt, params = prepare(h, source, key="publish")
    protocol, result = daily.execute(h, attempt, params)
    contract = massive_actions.CONTRACTS[dataset]
    rows = read(h, result, contract)
    evidence = document["coverage"]["bindings"][0]
    expected = [{**value, "date": "2026-08-10", "adjustment_basis": "corporate_action",
        "occurrence_id": evidence["occurrence_id"], **{k: evidence["evidence"][k] for k in
        ("raw_receipt_id", "raw_hash", "first_received_at", "observed_at")}} for value in reversed(values)]
    assert rows == expected
    resolved = Repository(h.conn, h.store).resolve_full(result.snapshot_id)
    assert result.knowledge_mode_by_table == {contract.table_name: "reconstructed"}
    assert result.finality_receipt_refs == () and result.parent_snapshot_id is None
    manifest = resolved.table_manifests[contract.table_name]
    assert manifest.coverage_receipt_refs == (ref.artifact_id,) and manifest.availability_evidence_refs == ()
    assert resolved.records[0].input_receipt_refs == (evidence["evidence"]["raw_receipt_id"],)
    assert resolved.records[0].import_request_hash == content_hash([
        ref.artifact_id, "massive_actions_normalizer.v1", contract.definition_hash])
    assert protocol["completed_ids"] == [source] and protocol["no_work"] is False
    assert h.conn.execute("SELECT SUM(used_calls) FROM provider_reservations").fetchone()[0] == before_calls
    before = tuple(h.conn.iterdump())
    assert daily.execute(h, attempt, params)[1] == result
    assert tuple(h.conn.iterdump()) == before


@pytest.mark.parametrize("dataset,required", [
    ("splits", {"id", "ticker", "execution_date", "split_from", "split_to"}),
    ("dividends", {"id", "ticker", "ex_dividend_date", "cash_amount", "currency"}),
])
@pytest.mark.parametrize("explicit_null", [False, True])
def test_optional_action_metadata_remains_unknown(harness, dataset, required, explicit_null):
    h = harness
    row = {k: v if k in required else None for k, v in action(dataset).items()}
    if not explicit_null:
        row = {k: v for k, v in row.items() if k in required}
    source, _, _ = acquired(h, dataset, key="source", rows=[row])
    attempt, params = prepare(h, source, key="publish")
    result = daily.execute(h, attempt, params)[1]
    actual = read(h, result, massive_actions.CONTRACTS[dataset])[0]
    optional = set(action(dataset)) - required
    assert all(actual[k] is None for k in optional)
    assert {k: actual[k] for k in required} == {k: action(dataset)[k] for k in required}


@pytest.mark.parametrize("field", ["split_from", "split_to"])
@pytest.mark.parametrize("explicit_null", [False, True], ids=["absent", "null"])
def test_missing_split_economics_preserves_native_visibility(harness, monkeypatch, field, explicit_null):
    h = harness
    parent, scope, _ = bootstrap(h)
    parent_rows = read(h, parent, massive_daily.CONTRACT)
    row = action("splits")
    if explicit_null:
        row[field] = None
    else:
        del row[field]
    source, _, _ = acquired(h, "splits", key="invalid-split-source", rows=[row])
    attempt, params = prepare(h, source, key="publish", parent=parent, scope=scope, generation=1)
    # Admission must refuse before a nullable schema or Parquet validation can mask it.
    monkeypatch.setattr(massive_daily, "build_candidate",
                        lambda *_a, **_k: pytest.fail("missing split economics reached publication"))
    before = tuple(h.conn.iterdump())
    with pytest.raises(OpsError) as error:
        daily.execute(h, attempt, params)
    assert error.value.code == "INTEGRITY_FAILED" and not error.value.problem.retryable
    assert tuple(h.conn.iterdump()) == before
    assert read(h, parent, massive_daily.CONTRACT) == parent_rows
    assert not (h.root / attempt.attempt_id / "snapshot.json").exists()


@pytest.mark.parametrize("dataset,field,value", [
    ("splits", "id", None), ("splits", "id", ""), ("splits", "id", 17),
    ("splits", "id", "x" * 257), ("splits", "adjustment_type", " "), ("splits", "split_from", 0),
    ("splits", "split_to", -1), ("splits", "split_from", True), ("splits", "split_to", "7"),
    ("splits", "split_to", 2**53 + 1), ("splits", "historical_adjustment_factor", 0),
    ("splits", "split_to", float("inf")), ("dividends", "cash_amount", float("nan")),
    ("dividends", "cash_amount", None), ("dividends", "cash_amount", -1),
    ("dividends", "currency", ""), ("dividends", "currency", None),
    ("dividends", "frequency", 2.5), ("dividends", "frequency", True),
    ("dividends", "frequency", 2**63), ("dividends", "record_date", "20260811"),
    ("dividends", "pay_date", "not-a-date"), ("dividends", "split_adjusted_cash_amount", -1),
])
def test_invalid_selected_action_fails_without_native_visibility(harness, monkeypatch, dataset, field, value):
    h = harness
    source, _, _ = daily.acquired(h, scope=support.request(dataset=dataset), key="source",
        payload=json.dumps({"status": "OK", "results": [action(dataset) | {field: value}]}).encode())
    attempt, params = prepare(h, source, key="publish")
    monkeypatch.setattr(massive_daily, "build_candidate", lambda *_a, **_k: pytest.fail("invalid row reached publication"))
    before = tuple(h.conn.iterdump())
    with pytest.raises(OpsError) as error:
        daily.execute(h, attempt, params)
    assert error.value.code == "INTEGRITY_FAILED" and not error.value.problem.retryable
    assert tuple(h.conn.iterdump()) == before
    assert not (h.root / attempt.attempt_id / "snapshot.json").exists()


@pytest.mark.parametrize("dataset", ["splits", "dividends"])
def test_duplicate_event_id_refuses_even_with_different_dates(harness, dataset):
    h = harness
    source, _, _ = acquired(h, dataset, key="source", rows=[action(dataset), action(dataset, day="2026-08-11")])
    attempt, params = prepare(h, source, key="publish")
    with pytest.raises(OpsError) as error:
        daily.execute(h, attempt, params)
    assert error.value.code == "INTEGRITY_FAILED"
    assert h.conn.execute("SELECT count(*) FROM data_snapshots").fetchone()[0] == 0


def test_parent_action_addition_and_interval_correction_preserve_native_membership(harness, monkeypatch):
    h = harness
    parent, scope, daily_ref = bootstrap(h)
    original_daily = read(h, parent, massive_daily.CONTRACT)
    contract = massive_actions.CONTRACTS["dividends"]
    initial = [action("dividends", "left", "2026-08-03"), action("dividends", "inside"),
               action("dividends", "old-start", "2026-08-09"), action("dividends", "old-end", "2026-08-12"),
               action("dividends", "right", "2026-08-25")]
    source, first_ref, first_doc = acquired(h, "dividends", key="actions", rows=initial)
    attempt, params = prepare(h, source, key="add", parent=parent, scope=scope, generation=1)
    added = finish_snapshot(h, attempt, params)
    assert added.table_versions["massive_daily"] == parent.table_versions["massive_daily"]
    assert added.parent_snapshot_id == parent.snapshot_id
    corrections = [action("dividends", identity, day) | {"cash_amount": amount} for identity, day, amount in
                   [("new-start", "2026-08-09", 8.5), ("replacement", "2026-08-10", 9.5),
                    ("new-end", "2026-08-12", 10.5)]]
    # New generation ensures a real independent provider observation, not older exact reuse.
    request = support.request("2026-08-09", "2026-08-12", "dividends")
    p = support.parameters(request, generation="correction")
    support.enqueue(h, "correction-source", p)
    source_attempt = support.claim(h)
    _, doc = support.run(h, source_attempt, support.fetching([], canonical_json(
        {"status": "OK", "results": corrections}).encode()), p)
    second_ref = daily.committed_output(h, source_attempt, doc)
    attempt, params = prepare(h, source_attempt.job_id, key="replace", parent=added, scope=scope, generation=2)
    monkeypatch.setattr(support.coverage, "plan", lambda *_: pytest.fail("snapshot replanned"))
    updated = finish_snapshot(h, attempt, params)
    rows = read(h, updated, contract)
    assert [(r["id"], r["date"], r["cash_amount"]) for r in rows] == [
        ("left", "2026-08-03", 1.25), ("new-end", "2026-08-12", 10.5),
        ("new-start", "2026-08-09", 8.5), ("replacement", "2026-08-10", 9.5), ("right", "2026-08-25", 1.25)]
    assert rows[0]["raw_receipt_id"] == first_doc["coverage"]["bindings"][0]["evidence"]["raw_receipt_id"]
    assert all(r["raw_receipt_id"] == doc["coverage"]["bindings"][0]["evidence"]["raw_receipt_id"]
               for r in rows[1:4])
    assert read(h, parent, massive_daily.CONTRACT) == original_daily == read(h, updated, massive_daily.CONTRACT)
    assert [(r["id"], r["date"], r["cash_amount"]) for r in read(h, added, contract)] == [
        ("inside", "2026-08-10", 1.25), ("left", "2026-08-03", 1.25),
        ("old-end", "2026-08-12", 1.25), ("old-start", "2026-08-09", 1.25), ("right", "2026-08-25", 1.25)]
    resolved = Repository(h.conn, h.store).resolve_full(updated.snapshot_id)
    manifest = resolved.table_manifests[contract.table_name]
    assert manifest.coverage_receipt_refs == tuple(sorted([first_ref.artifact_id, second_ref.artifact_id]))
    assert manifest.parent_dataset_version_id == added.table_versions[contract.table_name].dataset_version_id
    assert resolved.table_manifests["massive_daily"].coverage_receipt_refs == (daily_ref.artifact_id,)
    assert h.conn.execute("SELECT generation FROM data_snapshot_heads WHERE scope=?", (scope,)).fetchone()[0] == 3


def test_refresh_preserves_distinct_parent_policies_and_finality_on_replay(harness, monkeypatch):
    from engine.v2.data import manifests
    from engine.v2.ops import snapshots
    h = harness
    source, _, _ = daily.acquired(h)
    attempt, params = prepare(h, source, key="bootstrap")
    original = daily.execute(h, attempt, params)[1]
    repo = Repository(h.conn, h.store)
    resolved = repo.resolve_full(original.snapshot_id)
    # Commit an independent valid parent; do not construct it with the refresh composer.
    candidate = manifests.snapshot_ref(resolved.table_manifests, calendar_version="synthetic_calendar.v7",
        source_priority_version="synthetic_priority.v9",
        finality_receipt_refs=("synthetic-finality-z", "synthetic-finality-a"),
        parent_snapshot_id=original.snapshot_id)
    receipt = snapshots.commit_snapshot_for_attempt(h.conn, h.store, scope="policy-parent",
        request_hash=content_hash(["policy-parent"]), contracts=resolved.contracts, objects=resolved.objects,
        records=resolved.records, manifests=tuple(resolved.table_manifests.values()), snapshot=candidate,
        expected_head_snapshot_id=None, expected_head_generation=0, receipt_id="policy-parent",
        attempt_id=attempt.attempt_id, fence=attempt.fence, clock=h.clock)
    parent = repo.resolve(receipt.snapshot_ref.snapshot_id)
    expected = ("synthetic_calendar.v7", "synthetic_priority.v9",
                ("synthetic-finality-z", "synthetic-finality-a"))
    assert (parent.calendar_version, parent.source_priority_version, parent.finality_receipt_refs) == expected
    assert parent.table_versions == original.table_versions
    support.finish(h, attempt)
    h.clock.advance(13)
    source, _, _ = acquired(h, "splits", key="actions")
    attempt, params = prepare(h, source, key="refresh", parent=parent, scope="policy-parent", generation=1)
    fresh = daily.execute(h, attempt, params)[1]
    assert (fresh.calendar_version, fresh.source_priority_version, fresh.finality_receipt_refs) == expected
    assert fresh.parent_snapshot_id == parent.snapshot_id
    assert fresh.table_versions["massive_daily"] == parent.table_versions["massive_daily"]
    assert set(fresh.table_versions) == {"massive_daily", "massive_splits"}
    before = tuple(h.conn.iterdump())
    monkeypatch.setattr(pinned, "normalized_rows", lambda *_: pytest.fail("replay normalized again"))
    replay = daily.execute(h, attempt, params)[1]
    assert (replay.calendar_version, replay.source_priority_version, replay.finality_receipt_refs) == expected
    assert replay == fresh and tuple(h.conn.iterdump()) == before
    assert repo.resolve(parent.snapshot_id) == parent


@pytest.mark.parametrize("wrong_generation", [1, 3])
def test_stale_parent_generation_refuses_atomic_publication(harness, wrong_generation):
    h = harness
    parent, scope, _ = bootstrap(h)
    source, _, _ = acquired(h, "splits", key="actions")
    attempt, params = prepare(h, source, key="add", parent=parent, scope=scope, generation=1)
    added = finish_snapshot(h, attempt, params)
    attempt, params = prepare(h, source, key="stale", parent=added, scope=scope, generation=wrong_generation)
    before = tuple(h.conn.iterdump())
    with pytest.raises(OpsError) as error:
        daily.execute(h, attempt, params)
    assert error.value.code == "RESOURCE_UNAVAILABLE"
    assert tuple(h.conn.iterdump()) == before
    assert not (h.root / attempt.attempt_id / "snapshot.json").exists()


@pytest.mark.parametrize("args", [["--parent-snapshot", "snap_" + "a" * 32],
    ["--scope", "x"], ["--head-generation", "1"],
    ["--parent-snapshot", "bad", "--scope", "x", "--head-generation", "1"],
    ["--parent-snapshot", "snap_" + "a" * 32, "--scope", "x", "--head-generation", "0"]])
def test_partial_or_invalid_parent_submission_refuses_before_storage(tmp_path, capsys, args):
    root = tmp_path / "absent"
    assert massive_snapshot.main(["--root", str(root), "--acquisition-job", "job_" + "a" * 32,
                                  "--idempotency-key", "test", *args]) == 2
    assert json.loads(capsys.readouterr().out) == {"refused": "INVALID_REQUEST"}
    assert not root.exists()


def test_daily_refresh_preserves_other_ticker_basis_and_outside_dates(harness):
    h = harness
    source, _, _ = daily.acquired(h, payload=daily.daily_payload([
        ("2026-08-01", 10), ("2026-08-10", 20), ("2026-08-31", 30)]))
    attempt, params = prepare(h, source, key="bootstrap")
    parent = finish_snapshot(h, attempt, params)
    scope_name = h.conn.execute("SELECT scope FROM data_snapshot_heads").fetchone()[0]
    originals = read(h, parent, massive_daily.CONTRACT)
    for ticker, basis, close, key, generation in [
            ("OTHER", "raw", 400, "ticker", 1), ("SYNTH", "split_adjusted", 70, "basis", 2)]:
        request = support.request(ticker=ticker)
        request.update(adjustment_basis=basis, params=request["params"] | {"adjusted": basis == "split_adjusted"})
        payload = json.loads(daily.daily_payload([("2026-08-10", close)]))
        payload.update(ticker=ticker, adjusted=basis == "split_adjusted")
        source, _, _ = daily.acquired(h, key=key, scope=request, payload=canonical_json(payload).encode())
        attempt, params = prepare(h, source, key=key + "-snapshot", parent=parent,
                                  scope=scope_name, generation=generation)
        parent = finish_snapshot(h, attempt, params)
    request = support.request("2026-08-09", "2026-08-11")
    p = support.parameters(request, generation="correction")
    support.enqueue(h, "correction", p)
    source_attempt = support.claim(h)
    _, document = support.run(h, source_attempt, support.fetching([], daily.daily_payload([("2026-08-10", 90)])), p)
    daily.committed_output(h, source_attempt, document)
    attempt, params = prepare(h, source_attempt.job_id, key="refresh", parent=parent, scope=scope_name, generation=3)
    updated = finish_snapshot(h, attempt, params)
    actual = read(h, updated, massive_daily.CONTRACT)
    assert [(r["adjustment_basis"], r["date"], r["open"], r["high"], r["low"], r["close"], r["volume"])
            for r in actual] == [
        ("raw", "2026-08-01", 9.5, 11.5, 8.5, 10., 100.25),
        ("raw", "2026-08-10", 89.5, 91.5, 88.5, 90., 900.25),
        ("raw", "2026-08-31", 29.5, 31.5, 28.5, 30., 300.25),
        ("split_adjusted", "2026-08-10", 69.5, 71.5, 68.5, 70., 700.25)]
    assert actual[0] == originals[0] and actual[2] == originals[2]
    assert read(h, updated, massive_daily.CONTRACT, "OTHER") == read(h, parent, massive_daily.CONTRACT, "OTHER")
    old_records = Repository(h.conn).fragment_records(parent, "massive_daily")
    new_records = Repository(h.conn).fragment_records(updated, "massive_daily")
    assert [r for r in old_records if r.partition_key == "OTHER"] == [r for r in new_records if r.partition_key == "OTHER"]


@pytest.mark.parametrize("point", ["after_contracts", "after_objects", "after_fragments", "after_dataset_versions",
    "after_memberships", "after_snapshot", "after_snapshot_tables", "before_head_update", "before_commit"])
def test_refresh_transaction_fault_keeps_parent_and_all_membership(harness, point):
    h = harness
    parent, scope, _ = bootstrap(h)
    source, _, _ = acquired(h, "splits", key="actions")
    attempt, params = prepare(h, source, key="add", parent=parent, scope=scope, generation=1)
    before, reached = tuple(h.conn.iterdump()), []
    def fault(where):
        if where == point:
            reached.append(where)
            raise RuntimeError("synthetic refresh rollback")
    with pytest.raises(RuntimeError, match="^synthetic refresh rollback$"):
        daily.execute(h, attempt, params, fault=fault)
    assert reached == [point] and tuple(h.conn.iterdump()) == before
    assert not (h.root / attempt.attempt_id / "snapshot.json").exists()


def test_refresh_postcommit_crash_new_attempt_and_replay_after_later_head(harness, monkeypatch):
    h = harness
    parent, scope, _ = bootstrap(h)
    source, _, _ = acquired(h, "splits", key="actions")
    attempt, params = prepare(h, source, key="add", parent=parent, scope=scope, generation=1)
    original = Path.write_text
    def crash(path, *args, **kwargs):
        if path.name == "snapshot.json":
            raise OSError("synthetic committed output crash")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "write_text", crash)
    with pytest.raises(OpsError) as error:
        daily.execute(h, attempt, params)
    assert error.value.code == "RESOURCE_UNAVAILABLE"
    committed = dict(h.conn.execute("SELECT * FROM data_snapshot_heads WHERE scope=?", (scope,)).fetchone())
    assert committed["generation"] == 2
    assert support.retry(h, attempt, error.value) == "retry_wait"
    next_attempt = support.claim(h)
    input_bindings.resolve_and_record(h.conn, h.store, next_attempt)
    monkeypatch.setattr(Path, "write_text", original)
    with monkeypatch.context() as context:
        context.setattr(pinned, "normalized_rows", support.no_fetch)
        added = finish_snapshot(h, next_attempt, params)
    assert added.snapshot_id == committed["snapshot_id"]
    other, _, _ = acquired(h, "dividends", key="dividends")
    later_attempt, later_params = prepare(h, other, key="later", parent=added, scope=scope, generation=2)
    newer = finish_snapshot(h, later_attempt, later_params)
    replay_attempt, replay_params = prepare(h, source, key="replay", parent=parent, scope=scope, generation=1)
    before = tuple(h.conn.iterdump())
    monkeypatch.setattr(pinned, "normalized_rows", support.no_fetch)
    assert daily.execute(h, replay_attempt, replay_params)[1] == added
    assert tuple(h.conn.iterdump()) == before
    assert h.conn.execute("SELECT snapshot_id,generation FROM data_snapshot_heads WHERE scope=?", (scope,)).fetchone()[:] == (newer.snapshot_id, 3)


@pytest.mark.parametrize("limit", ["bytes", "rows"])
def test_parent_admission_bounds_precede_parquet_read_and_publication(harness, monkeypatch, limit):
    h = harness
    parent, scope, _ = bootstrap(h)
    source, _, _ = daily.acquired(h, key="new-pin")
    attempt, params = prepare(h, source, key="refresh", parent=parent, scope=scope, generation=1)
    selected = Repository(h.conn).fragment_records(parent, "massive_daily")[0]
    assert massive_refresh.MAX_PARENT_BYTES == 64 << 20 and massive_refresh.MAX_PARENT_ROWS == 1000
    if limit == "bytes":
        monkeypatch.setattr(massive_refresh, "MAX_PARENT_BYTES", selected.object_ref.byte_size - 1)
    else:
        monkeypatch.setattr(massive_refresh, "MAX_PARENT_ROWS", 0)
    monkeypatch.setattr(Repository, "scan", lambda *_a, **_k: pytest.fail("refused parent was read"))
    monkeypatch.setattr(massive_daily, "build_candidate", lambda *_a, **_k: pytest.fail("refused parent was published"))
    before = tuple(h.conn.iterdump())
    with pytest.raises(OpsError) as error:
        daily.execute(h, attempt, params)
    assert error.value.code == "RESOURCE_LIMIT_EXCEEDED"
    assert tuple(h.conn.iterdump()) == before


def test_rewritten_partition_bound_after_preserving_outside_rows(harness):
    h = harness
    rows = [action("splits", f"old-{i:04d}", "2026-08-01") for i in range(1000)]
    source, _, _ = acquired(h, "splits", key="source", rows=rows)
    attempt, params = prepare(h, source, key="bootstrap")
    parent = finish_snapshot(h, attempt, params)
    scope = h.conn.execute("SELECT scope FROM data_snapshot_heads").fetchone()[0]
    # A different generation and a disjoint interval have one real new row.
    request = support.request("2026-09-01", "2026-09-01", "splits")
    source, _, _ = daily.acquired(h, scope=request, key="new-source", payload=canonical_json({
        "status": "OK", "results": [action("splits", "new", "2026-09-01")]}).encode())
    attempt, params = prepare(h, source, key="refresh", parent=parent, scope=scope, generation=1)
    before = tuple(h.conn.iterdump())
    with pytest.raises(OpsError) as error:
        daily.execute(h, attempt, params)
    assert error.value.code == "RESOURCE_LIMIT_EXCEEDED"
    assert tuple(h.conn.iterdump()) == before
    assert len(read(h, parent, massive_actions.CONTRACTS["splits"])) == 1000


def test_cross_window_action_id_collision_refuses_instead_of_deleting_other_date(harness):
    h = harness
    source, _, _ = acquired(h, "splits", key="source", rows=[action("splits", "same", "2026-08-01")])
    attempt, params = prepare(h, source, key="bootstrap")
    parent = finish_snapshot(h, attempt, params)
    scope = h.conn.execute("SELECT scope FROM data_snapshot_heads").fetchone()[0]
    source, _, _ = acquired(h, "splits", key="new-source", start="2026-09-01", end="2026-09-01",
                           rows=[action("splits", "same", "2026-09-01")])
    attempt, params = prepare(h, source, key="refresh", parent=parent, scope=scope, generation=1)
    before = tuple(h.conn.iterdump())
    with pytest.raises(OpsError) as error:
        daily.execute(h, attempt, params)
    assert error.value.code == "INTEGRITY_FAILED"
    assert tuple(h.conn.iterdump()) == before


def test_cli_parent_fields_are_immutable_job_parameters(harness, monkeypatch, capsys):
    from engine.v2.ops import fingerprints
    h = harness
    parent, scope, _ = bootstrap(h)
    source, _, _ = acquired(h, "splits", key="actions")
    # Isolate only source fingerprint cost; the real CLI and submission/catalog path run.
    monkeypatch.setattr(fingerprints, "worker_source_manifest", lambda _: {"synthetic": "code"})
    monkeypatch.setattr(fingerprints, "environment_identity", lambda _: {"synthetic": "environment"})
    # The CLI's conventional catalog filename is independent of the fixture's chosen name.
    (h.root / "catalog.sqlite").symlink_to(h.root / "ops.sqlite")
    assert massive_snapshot.main(["--root", str(h.root), "--acquisition-job", source,
        "--idempotency-key", "parent-cli", "--parent-snapshot", parent.snapshot_id,
        "--scope", scope, "--head-generation", "1"], clock=h.clock) == 0
    job_id = json.loads(capsys.readouterr().out)["job_id"]
    spec = json.loads(h.conn.execute("SELECT spec_json FROM jobs WHERE job_id=?", (job_id,)).fetchone()[0])
    assert {k: spec["parameters"][k] for k in ("parent_snapshot_id", "scope", "head_generation")} == {
        "parent_snapshot_id": parent.snapshot_id, "scope": scope, "head_generation": 1}
    attempt = support.claim(h)
    assert attempt.job_id == job_id
    input_bindings.resolve_and_record(h.conn, h.store, attempt)
    params = snapshot.Parameters(source, {"acquisition.json": source + "#massive_acquisition"},
                                 (source,), parent.snapshot_id, scope, 1)
    result = daily.execute(h, attempt, params)[1]
    assert set(result.table_versions) == {"massive_daily", "massive_splits"}


@pytest.mark.parametrize("dataset", ["splits", "dividends"])
def test_paginated_action_pins_clip_without_replacing_original_page_receipts(harness, dataset):
    h = harness
    scope, calls = support.request(dataset=dataset), []
    params = support.parameters(scope, cap=2)
    support.enqueue(h, "pages", params)
    first = support.claim(h)
    continuation = pagination.continuation(scope, 1)
    payload = canonical_json({"status": "OK", "results": [action(dataset, "a", "2026-08-02"),
        action(dataset, "b", "2026-08-10")], "next_url": pagination.next_url(continuation)}).encode()
    error = pagination.refused(h, first, params, pagination.fetcher(calls, scope, payload), "SOURCE_NOT_FINAL")
    assert support.retry(h, first, error) == "retry_wait"
    last = support.claim(h)
    _, document = support.run(h, last, pagination.fetcher(calls, continuation,
        canonical_json({"status": "OK", "results": [action(dataset, "c", "2026-08-20")]}).encode()), params)
    daily.committed_output(h, last, document)
    original_pages = [json.loads(p["document_json"]) for p in pagination.pages(h)]
    # A separate successful acquisition pins a narrower view over the same original pages.
    source, ref, saved = daily.acquired(h, key="clipped", scope=support.request("2026-08-09", "2026-08-25", dataset))
    assert saved["coverage"]["bindings"][0]["evidence"] == document["coverage"]["bindings"][0]["evidence"]
    attempt, params = prepare(h, source, key="publish")
    result = daily.execute(h, attempt, params)[1]
    contract = massive_actions.CONTRACTS[dataset]
    rows = read(h, result, contract)
    assert [(r["id"], r["date"]) for r in rows] == [("b", "2026-08-10"), ("c", "2026-08-20")]
    for row, expected, page in zip(rows, [action(dataset, "b"), action(dataset, "c", "2026-08-20")], original_pages):
        assert {k: row[k] for k in expected} == expected
        assert {k: row[k] for k in ("raw_receipt_id", "raw_hash", "first_received_at", "observed_at")} == {
            k: page[k] for k in ("raw_receipt_id", "raw_hash", "first_received_at", "observed_at")}
    assert original_pages[0]["observed_at"] != original_pages[1]["observed_at"]
    resolved = Repository(h.conn, h.store).resolve_full(result.snapshot_id)
    assert resolved.records[0].input_receipt_refs == tuple(sorted(p["raw_receipt_id"] for p in original_pages))
    assert resolved.table_manifests[contract.table_name].coverage_receipt_refs == (ref.artifact_id,)
    assert calls == [scope, continuation]


@pytest.mark.parametrize("dataset", ["splits", "dividends"])
def test_empty_clipped_action_view_cannot_erase_parent(harness, dataset):
    h = harness
    source, _, _ = acquired(h, dataset, key="source")
    attempt, params = prepare(h, source, key="bootstrap")
    parent = finish_snapshot(h, attempt, params)
    scope = h.conn.execute("SELECT scope FROM data_snapshot_heads").fetchone()[0]
    source, _, _ = daily.acquired(h, key="empty-clip", scope=support.request("2026-08-20", "2026-08-25", dataset))
    attempt, params = prepare(h, source, key="replace", parent=parent, scope=scope, generation=1)
    before = tuple(h.conn.iterdump())
    with pytest.raises(OpsError) as error:
        daily.execute(h, attempt, params)
    assert error.value.code == "SOURCE_EMPTY"
    assert tuple(h.conn.iterdump()) == before


def test_parent_byte_limit_equality_is_admitted(harness, monkeypatch):
    h = harness
    parent, scope, _ = bootstrap(h)
    source, _, _ = daily.acquired(h, key="new-pin")
    record = Repository(h.conn).fragment_records(parent, "massive_daily")[0]
    monkeypatch.setattr(massive_refresh, "MAX_PARENT_BYTES", record.object_ref.byte_size)
    monkeypatch.setattr(massive_refresh, "MAX_PARENT_ROWS", 1)
    attempt, params = prepare(h, source, key="refresh", parent=parent, scope=scope, generation=1)
    result = daily.execute(h, attempt, params)[1]
    assert read(h, result, massive_daily.CONTRACT) == read(h, parent, massive_daily.CONTRACT)


@pytest.mark.parametrize("dataset", ["splits", "dividends"])
def test_action_symbol_case_matches_acquisition_canonicalization(harness, dataset):
    h = harness
    source, _, _ = acquired(h, dataset, key="source", rows=[action(dataset, ticker="synth")])
    attempt, params = prepare(h, source, key="publish")
    row = read(h, daily.execute(h, attempt, params)[1], massive_actions.CONTRACTS[dataset])[0]
    assert row["ticker"] == "SYNTH" and row["id"] == "event-1"


def test_same_acquisition_refresh_reconciles_original_snapshot_lineage(harness, monkeypatch):
    h = harness
    source, _, _ = daily.acquired(h)
    attempt, params = prepare(h, source, key="bootstrap")
    parent = finish_snapshot(h, attempt, params)
    scope = h.conn.execute("SELECT scope FROM data_snapshot_heads").fetchone()[0]
    original = Repository(h.conn).resolve_full(parent.snapshot_id)
    attempt, params = prepare(h, source, key="refresh", parent=parent, scope=scope, generation=1)
    result = daily.execute(h, attempt, params)[1]
    assert result == parent and result.parent_snapshot_id is None
    assert Repository(h.conn).resolve_full(result.snapshot_id) == original
    assert h.conn.execute("SELECT generation FROM data_snapshot_heads WHERE scope=?", (scope,)).fetchone()[0] == 1
    assert h.conn.execute("SELECT count(*) FROM data_import_receipts").fetchone()[0] == 2
    monkeypatch.setattr(pinned, "normalized_rows", support.no_fetch)
    before = tuple(h.conn.iterdump())
    assert daily.execute(h, attempt, params)[1] == parent
    assert tuple(h.conn.iterdump()) == before


def test_zero_dividend_values_are_reported_values_not_missing_defaults(harness):
    h = harness
    row = action("dividends") | {"cash_amount": 0, "frequency": 0, "split_adjusted_cash_amount": 0}
    source, _, _ = acquired(h, "dividends", key="source", rows=[row])
    attempt, params = prepare(h, source, key="publish")
    actual = read(h, daily.execute(h, attempt, params)[1], massive_actions.CONTRACTS["dividends"])[0]
    assert {k: actual[k] for k in row} == row


@pytest.mark.parametrize("field,value", [("parent_snapshot_id", "bad"), ("scope", None), ("head_generation", 0)])
def test_worker_rejects_unusable_parent_parameters_before_storage(harness, monkeypatch, field, value):
    h = harness
    parent, scope, _ = bootstrap(h)
    source, _, _ = acquired(h, "splits", key="actions")
    attempt, params = prepare(h, source, key="refresh", parent=parent, scope=scope, generation=1)
    from dataclasses import replace

    from engine.v2.ops import catalog
    monkeypatch.setattr(catalog, "connect", lambda *_a, **_k: pytest.fail("invalid parent opened catalog"))
    with pytest.raises(OpsError) as error:
        daily.execute(h, attempt, replace(params, **{field: value}))
    assert error.value.code == "INVALID_REQUEST"


def test_stale_parent_id_with_current_generation_cannot_remove_new_table(harness):
    h = harness
    parent, scope, _ = bootstrap(h)
    source, _, _ = acquired(h, "splits", key="actions")
    attempt, params = prepare(h, source, key="add", parent=parent, scope=scope, generation=1)
    added = finish_snapshot(h, attempt, params)
    source, _, _ = acquired(h, "dividends", key="dividends")
    attempt, params = prepare(h, source, key="stale", parent=parent, scope=scope, generation=2)
    before = tuple(h.conn.iterdump())
    with pytest.raises(OpsError) as error:
        daily.execute(h, attempt, params)
    assert error.value.code == "RESOURCE_UNAVAILABLE" and tuple(h.conn.iterdump()) == before
    assert h.conn.execute("SELECT snapshot_id FROM data_snapshot_heads WHERE scope=?", (scope,)).fetchone()[0] == added.snapshot_id


@pytest.mark.parametrize("mode", ["knowledge", "calendar_version", "source_priority_version", "finality_receipt_refs"])
def test_refresh_replay_refuses_valid_but_substituted_native_output(harness, monkeypatch, mode):
    from engine.v2.data import manifests
    from engine.v2.ops import snapshots
    h = harness
    parent, scope, _ = bootstrap(h)
    source, _, _ = acquired(h, "splits", key="actions")
    attempt, params = prepare(h, source, key="refresh", parent=parent, scope=scope, generation=1)
    result = daily.execute(h, attempt, params)[1]
    resolved = Repository(h.conn).resolve_full(result.snapshot_id)
    tables = dict(resolved.table_manifests)
    if mode == "knowledge":
        old = tables["massive_splits"]
        records = [r for r in resolved.records if r.table_contract_ref == old.dataset_version_ref.table_contract_ref]
        tables["massive_splits"] = manifests.dataset_manifest(old.dataset_version_ref.table_contract_ref,
            records, knowledge_mode="observed", coverage_receipt_refs=old.coverage_receipt_refs,
            availability_evidence_refs=("synthetic-availability",))
    policies = {k: getattr(parent, k) for k in ("calendar_version", "source_priority_version", "finality_receipt_refs")}
    if mode != "knowledge":
        policies[mode] = ("synthetic-proof",) if mode == "finality_receipt_refs" else "synthetic-policy"
    alternate = manifests.snapshot_ref(tables, parent_snapshot_id=parent.snapshot_id, **policies)
    receipt = snapshots.commit_snapshot_for_attempt(h.conn, h.store, scope="synthetic-alternate",
        request_hash=content_hash([mode]), contracts=resolved.contracts, objects=resolved.objects,
        records=resolved.records, manifests=tuple(tables.values()), snapshot=alternate,
        expected_head_snapshot_id=None, expected_head_generation=0, receipt_id="synthetic-" + mode,
        attempt_id=attempt.attempt_id, fence=attempt.fence, clock=h.clock)
    assert Repository(h.conn).resolve(receipt.snapshot_ref.snapshot_id) == receipt.snapshot_ref
    original_receipt = h.conn.execute("SELECT update_receipt_ref FROM data_snapshot_heads WHERE scope=?", (scope,)).fetchone()[0]
    h.conn.execute("DROP TRIGGER data_import_receipts_no_update")
    h.conn.execute("UPDATE data_import_receipts SET result_snapshot_id=? WHERE receipt_id=?",
                   (alternate.snapshot_id, original_receipt))
    before = tuple(h.conn.iterdump())
    monkeypatch.setattr(pinned, "normalized_rows", support.no_fetch)
    with pytest.raises(OpsError) as error:
        daily.execute(h, attempt, params)
    assert error.value.code == "INTEGRITY_FAILED" and tuple(h.conn.iterdump()) == before


def test_maximum_action_text_length_is_admitted(harness):
    h = harness
    source, _, _ = acquired(h, "splits", key="source", rows=[action("splits", "x" * 256)])
    attempt, params = prepare(h, source, key="publish")
    result = daily.execute(h, attempt, params)[1]
    assert read(h, result, massive_actions.CONTRACTS["splits"])[0]["id"] == "x" * 256


def test_valid_incompatible_parent_contract_refuses_before_read(harness, monkeypatch):
    from dataclasses import replace

    from engine.v2.data import manifests
    h = harness
    expected = massive_actions.CONTRACTS["splits"]
    other = replace(expected, contract_id="massive_splits.v2", semantic_version="2.0.0")
    other = replace(other, definition_hash=manifests.table_contract_hash(other))
    source, _, _ = acquired(h, "splits", key="source")
    with monkeypatch.context() as context:
        context.setitem(massive_actions.CONTRACTS, "splits", other)
        attempt, params = prepare(h, source, key="bootstrap")
        parent = finish_snapshot(h, attempt, params)
    assert Repository(h.conn).resolve_full(parent.snapshot_id).contracts == (other,)
    scope = h.conn.execute("SELECT scope FROM data_snapshot_heads").fetchone()[0]
    attempt, params = prepare(h, source, key="refresh", parent=parent, scope=scope, generation=1)
    monkeypatch.setattr(Repository, "scan", lambda *_a, **_k: pytest.fail("incompatible parent was read"))
    before = tuple(h.conn.iterdump())
    with pytest.raises(OpsError) as error:
        daily.execute(h, attempt, params)
    assert error.value.code == "INTEGRITY_FAILED" and tuple(h.conn.iterdump()) == before
