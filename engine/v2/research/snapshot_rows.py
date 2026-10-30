"""Bounded native row projections without lossy pandas conversion or read authority."""
from contextlib import closing

from engine.v2.contracts.data import DataQuery
from engine.v2.data import errors
from engine.v2.research.progress import Progress, stage
from engine.v2.research.snapshot_inputs import admit_snapshot


def read_snapshot_rows(repository, snapshot, table, columns, *, max_rows,
                       max_decoded_bytes, max_object_bytes, key_filter=(), max_batch_rows=1024):
    """Collect native rows atomically; a late refusal returns no partial list."""
    return list(iter_snapshot_rows(repository, snapshot, table, columns, max_rows=max_rows,
        max_decoded_bytes=max_decoded_bytes, max_object_bytes=max_object_bytes,
        key_filter=key_filter, max_batch_rows=max_batch_rows))


def iter_snapshot_rows(repository, snapshot, table, columns, *, max_rows,
                       max_decoded_bytes, max_object_bytes, key_filter=(), max_batch_rows=1024):
    """Yield native rows provisionally; caller owns outcome/holdout admission.

    Row cap bounds candidate fragments before reading; byte cap bounds cumulative
    Arrow batches, not Python heap size. Native per-object caps also apply.
    Rows are provisional until exhaustion; close an abandoned stream. No retry.
    """
    if any(type(v) is not int or v <= 0 for v in (max_rows, max_decoded_bytes, max_object_bytes, max_batch_rows)):
        raise errors.fail("CONTRACT_MISMATCH", "row projections require positive resource caps")
    admit_snapshot(repository, snapshot)
    contract = repository.table_contract(snapshot, table)
    ref = snapshot.table_versions[table].table_contract_ref
    predicates = tuple(key_filter)
    bound = repository.scan_population_bound(snapshot.snapshot_id, table_name=table,
        table_contract_ref=ref, key_filter=predicates)
    if bound > max_rows:
        raise errors.fail("RESULT_LIMIT_EXCEEDED", "row projection exceeds candidate row cap")
    query = DataQuery(snapshot_id=snapshot.snapshot_id, table_contract_ref=ref,
        columns=tuple(columns), key_filter=predicates, order_by=contract.primary_key,
        max_batch_rows=min(max_batch_rows, contract.maximum_batch_rows, max(1, bound)), max_result_rows=bound)
    total = 0
    progress = Progress("native projected rows", bound)
    with stage("native row projection", report_refusals=True), closing(
            repository.scan(query, table_name=table, max_object_bytes=max_object_bytes)) as batches:
        for batch in batches:
            total += batch.nbytes
            if total > max_decoded_bytes:
                raise errors.fail("RESULT_LIMIT_EXCEEDED", "row projection exceeds decoded byte cap")
            decoded = batch.to_pylist()
            del batch
            progress.tick(len(decoded))
            yield from decoded
            del decoded
