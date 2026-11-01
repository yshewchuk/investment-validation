"""Selection-only IV30 target preparation from a pinned research workspace."""
from collections import Counter, defaultdict
from contextlib import closing
from itertools import groupby

from engine.v2.contracts.data import KeyPredicate
from engine.v2.data import errors
from engine.v2.data.iv_crush import build_rows
from engine.v2.data.iv_crush_table import IV_CRUSH_CONTRACT
from engine.v2.data.repository import Repository
from engine.v2.ops.workspace.overlay import ResearchWorkspace
from engine.v2.research.event_target_store import publish_event_targets
from engine.v2.research.experiment_population import load_population
from engine.v2.research.progress import Progress, decision, stage
from engine.v2.research.snapshot_rows import iter_snapshot_rows, read_snapshot_rows


def _day(value):
    try:
        if value.isoformat() == value.date().isoformat() + "T00:00:00":
            return value.date().isoformat()
    except (AttributeError, TypeError, ValueError):
        pass
    raise errors.fail("CONTRACT_MISMATCH", "IV preparation requires canonical midnight source dates")


def prepare_iv_crush_targets(workspace, *, as_of_month, max_rows, max_decoded_bytes,
                             max_object_bytes, max_total_bytes, event_ids=None):
    """Build admitted targets, retain all metadata, and write only to workspace.

    Return distinct source/target snapshots and aggregate target/skip coverage.
    This reconstructs labels, not actual publication availability or features.
    No providers, fits, search registration or new holdout policy are involved.
    """
    if not isinstance(workspace, ResearchWorkspace):
        raise errors.fail("CONTRACT_MISMATCH", "IV preparation requires a research workspace")
    if type(max_total_bytes) is not int or max_total_bytes <= 0:
        raise errors.fail("CONTRACT_MISMATCH", "IV preparation requires a positive output byte cap")
    source, snapshot = workspace.source, workspace.source_snapshot()
    caps = dict(max_rows=max_rows, max_decoded_bytes=max_decoded_bytes, max_object_bytes=max_object_bytes)
    with stage("IV target preparation", report_refusals=True):
        event_contract = source.table_contract(snapshot, "earnings_events")
        if event_contract.partition_columns != ("year",):
            raise errors.fail("CONTRACT_MISMATCH", "IV preparation requires year-partitioned event metadata")
        years = tuple(sorted({int(r.partition_key) for r in source.fragment_records(snapshot, "earnings_events")}))
        metadata = read_snapshot_rows(source, snapshot, "earnings_events",
            [c.name for c in event_contract.columns],
            key_filter=(KeyPredicate(column="year", operator="in", values=years),), **caps)
        selected = load_population(source, snapshot, as_of_month=as_of_month,
                                   purpose="selection", event_ids=event_ids)
        daily_contract = source.table_contract(snapshot, "daily_market")
        iv = next((c for c in daily_contract.columns if c.name == "iv30"), None)
        if iv is None or iv.unit != "percent" or iv.physical_type != "float64":
            raise errors.fail("CONTRACT_MISMATCH", "IV preparation requires IV30 in percent units")
        tickers = tuple(sorted(set(selected["ticker"])))
        daily = iter_snapshot_rows(source, snapshot, "daily_market", ("ticker", "date", "iv30"),
            key_filter=(KeyPredicate(column="ticker", operator="in", values=tickers),), **caps)
        events = defaultdict(list)
        for row in selected.to_dict("records"):
            events[row["ticker"]].append(dict(event_date=_day(row["event_date"]), session=row["session"]))
        targets = []
        progress = Progress("IV target tickers", len(tickers))
        with closing(daily):
            for ticker, rows in groupby(daily, key=lambda row: row["ticker"]):
                quotes = (dict(date=_day(row["date"]), iv30=row["iv30"]) for row in rows)
                targets.extend(build_rows(ticker, events.pop(ticker), quotes,
                                          source_id=snapshot.snapshot_id, max_rows=max_rows))
                progress.tick()
        for ticker, pending in events.items():
            targets.extend(build_rows(ticker, pending, (), source_id=snapshot.snapshot_id, max_rows=max_rows))
            progress.tick()
        skipped = Counter(row["skip_reason"] for row in targets if row["skipped"])
        coverage = dict(selected_events=len(targets), available=sum(not r["skipped"] for r in targets),
                        skipped_by_reason=dict(sorted(skipped.items())))
        decision("IV target coverage", f"selected={len(targets)}, available={coverage['available']}, skipped={dict(skipped)}")
        target_snapshot = publish_event_targets(Repository(workspace.conn, workspace.store, pin_snapshots=True),
            source_snapshot=snapshot, provenance=dict(producer="iv_crush_prepare.v1",
                as_of_month=as_of_month, coverage=coverage), events_contract=event_contract,
            events=metadata, target_contract=IV_CRUSH_CONTRACT, targets=targets,
            max_rows=max_rows, max_object_bytes=max_object_bytes, max_total_bytes=max_total_bytes)
    return dict(source_snapshot=snapshot, target_snapshot=target_snapshot, coverage=coverage)
