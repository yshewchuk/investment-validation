"""Bounded interval replacement over one explicit immutable parent snapshot."""
from engine.v2.contracts import DataQuery, KeyPredicate
from engine.v2.data import errors, manifests, massive_daily
from engine.v2.data.repository import Repository

MAX_PARENT_BYTES = 64 << 20
MAX_PARENT_ROWS = 1000


def _selection(parent, contract, ticker):
    table = contract.table_name
    prior = parent.table_manifests.get(table)
    members = set()
    selected = ()
    if prior is not None:
        if prior.dataset_version_ref.table_contract_ref.definition_hash != contract.definition_hash:
            raise errors.fail("CONTRACT_MISMATCH", "Massive parent table contract differs")
        members = {f.fragment_id for f in prior.fragment_refs}
        selected = tuple(r for r in parent.records if r.fragment_id in members and r.partition_key == ticker)
        if (sum(r.object_ref.byte_size for r in selected) > MAX_PARENT_BYTES
                or sum(r.row_count for r in selected) > MAX_PARENT_ROWS):
            raise errors.fail("RESULT_LIMIT_EXCEEDED", "Massive parent partition exceeds bound")
    return prior, selected, members


def _retained(conn, store, parent, contract, request, prior, selected):
    if prior is None:
        return []
    repo = Repository(conn, store)
    predicate = (KeyPredicate(column="ticker", operator="eq", values=(request["ticker"],)),)
    ref = prior.dataset_version_ref.table_contract_ref
    bound = repo.scan_population_bound(parent.snapshot.snapshot_id, table_name=contract.table_name,
                                      table_contract_ref=ref, key_filter=predicate)
    if bound != sum(r.row_count for r in selected):
        raise errors.fail("CONTRACT_MISMATCH", "Massive parent partition layout differs")
    if not bound:
        return []
    query = DataQuery(snapshot_id=parent.snapshot.snapshot_id, table_contract_ref=ref,
        columns=tuple(c.name for c in contract.columns), key_filter=predicate,
        order_by=contract.primary_key, max_batch_rows=bound, max_result_rows=bound)
    old = [r for batch in repo.scan(query, table_name=contract.table_name) for r in batch.to_pylist()]
    return [r for r in old if not (r["adjustment_basis"] == request["adjustment_basis"]
                                  and request["start"] <= r["date"] <= request["end"])]


def _compose(parent, contract, ticker, record, prior, selected, members, acquisition_id):
    table = contract.table_name
    removed = {r.fragment_id for r in selected}
    records = tuple(r for r in parent.records if r.fragment_id not in removed) + (record,)
    members = members - removed
    table_records = sorted((r for r in records if r.fragment_id in members or r == record),
                           key=lambda r: (r.partition_key, r.primary_key_min))
    coverage = tuple(sorted(set((*prior.coverage_receipt_refs, acquisition_id)))) if prior else (acquisition_id,)
    hashes = {k: v for k, v in prior.partition_logical_hashes.items() if k != ticker} if prior else {}
    manifest = manifests.dataset_manifest(record.table_contract_ref, table_records,
        knowledge_mode="reconstructed", coverage_receipt_refs=coverage, availability_evidence_refs=(),
        parent_dataset_version_id=prior.dataset_version_ref.dataset_version_id if prior else None,
        partition_logical_hashes=hashes)
    tables = parent.table_manifests | {table: manifest}
    snapshot = manifests.snapshot_ref(tables, calendar_version=parent.snapshot.calendar_version,
        source_priority_version=parent.snapshot.source_priority_version,
        finality_receipt_refs=parent.snapshot.finality_receipt_refs,
        parent_snapshot_id=parent.snapshot.snapshot_id)
    contracts = tuple(c for c in parent.contracts if c.table_name != table) + (contract,)
    objects = {r.object_ref.object_id: r.object_ref for r in records}
    return manifests.ResolvedSnapshot(contracts=contracts, objects=tuple(objects.values()),
        records=records, table_manifests=tables, snapshot=snapshot)


def build_candidate(conn, store, parent, contract, request, rows, acquisition_id, identity):
    prior, selected, members = _selection(parent, contract, request["ticker"])
    rows = rows + _retained(conn, store, parent, contract, request, prior, selected)
    if len(rows) > 1000:
        raise errors.fail("RESULT_LIMIT_EXCEEDED", "Massive rewritten partition exceeds row bound")
    record, _, _ = massive_daily.build_candidate(store, rows, acquisition_id, identity, contract=contract)
    return _compose(parent, contract, request["ticker"], record, prior, selected, members, acquisition_id)
