"""Protected signed IV-crush reads from exact native, event-isolated targets."""
from __future__ import annotations

import math
from datetime import timedelta

from engine.v2.data import errors
from engine.v2.data.iv_crush_table import IV_CRUSH_CONTRACT, IV_CRUSH_TABLE_NAME
from engine.v2.foundation import canonical_json, content_hash, to_document
from engine.v2.foundation.experiment_holdouts import calendar_date
from engine.v2.research import event_history, experiment_population, progress

MAX_TARGET_OBJECT_BYTES = 1 << 20


def _candidates(repository, snapshot, as_of_month, purpose, event_ids, cutoff=None):
    """Complete population and singleton-object admission precedes any target read."""
    progress.decision("IV population", "selection=" + ("bulk" if event_ids is None else "explicit"))
    try:
        population = experiment_population.load_population(repository, snapshot, as_of_month=as_of_month,
                                     purpose=purpose, event_ids=event_ids)
    except errors.DataError as exc:
        if "holdout_exclusions" in exc.problem.details:
            progress.decision("IV population", f"holdout_or_ambiguous_refused={len(exc.problem.details['holdout_exclusions'])}")
        raise
    eligible = population.event_date.dt.date < cutoff if cutoff else population.event_id.notna()
    progress.decision("IV population", f"admitted={len(population)}; "
        f"holdout_or_ambiguous_excluded={len(population.attrs.get('holdout_exclusions', []))}; "
        f"cutoff_excluded={int((~eligible).sum())}; cutoff={cutoff.isoformat() if cutoff else 'none (preflight)'}")
    if event_ids is not None and not eligible.all():
        raise errors.fail("CONTRACT_MISMATCH", "requested IV-crush events must precede label cutoff")
    rows = population[eligible].sort_values(["event_date", "event_id"]).reset_index(drop=True)
    if rows.empty:
        raise errors.fail("POPULATION_COLLAPSED", "no IV-crush events precede label cutoff")
    contract = repository.table_contract(snapshot, IV_CRUSH_TABLE_NAME)
    if contract != IV_CRUSH_CONTRACT:
        raise errors.fail("CONTRACT_MISMATCH", "IV-crush labels require the exact native contract")
    plans = event_history.plan_event_targets(repository, snapshot, rows, contract,
                               columns=tuple(column.name for column in contract.columns))
    if any(record.object_ref.byte_size > MAX_TARGET_OBJECT_BYTES for _, _, record in plans):
        raise errors.fail("RESULT_LIMIT_EXCEEDED", "IV-crush target object exceeds the 1 MiB byte bound")
    progress.decision("IV targets", f"planned={len(plans)}; max_batch_rows={plans[0][1].max_batch_rows}; "
                      f"max_object_bytes={MAX_TARGET_OBJECT_BYTES}")
    return rows, plans, sorted(population.loc[~eligible, "event_id"].tolist())


def _object_input(event, record):
    """Native object identity is distinct from the builder's caller-declared source."""
    return {"event_id": event.event_id, "fragment_id": record.fragment_id,
            "object_id": record.object_ref.object_id, "content_hash": record.object_ref.content_hash}


def _identity(rows, snapshot, purpose):
    version = snapshot.table_versions[IV_CRUSH_TABLE_NAME]
    return {**{name: rows.iloc[0][name] for name in event_history.POPULATION_CONTEXT_COLUMNS}, "purpose": purpose,
            "target_version": to_document(version),
            "availability_basis": "modeled_post_close_plus_one_calendar_day",
            "source_publication": "unverified", "source_finality": "unverified"}


def preflight_iv_crush_targets(repository, snapshot, *, as_of_month, event_ids):
    """Prove explicit selection participants without opening target payloads."""
    with progress.stage("IV target admission", report_refusals=True):
        if event_ids is None:
            raise errors.fail("CONTRACT_MISMATCH", "IV-crush preflight requires explicit participants")
        rows, plans, _ = _candidates(repository, snapshot, as_of_month, "selection", event_ids)
    identity = {"schema_version": "research_iv_crush_preflight.v1",
        **_identity(rows, snapshot, "selection"), "event_ids": rows.event_id.tolist(),
        "planned_inputs": [_object_input(event, record) for event, _, record in plans]}
    return {**identity, "preflight_id": content_hash(identity)}


def _label_before(rows, event, cutoff):
    """Check the recorded modeled clock before consuming the signed numeric label."""
    try:
        if len(rows) != 1:
            raise ValueError
        row = rows[0]
        day = event.event_date.date()
        if (row["event_date"] != day.isoformat() or row["ticker"] != event.ticker
                or row["session"] != event.session or row["skipped"] is not False
                or row["skip_reason"] is not None):
            raise ValueError
        available = calendar_date(row["available_as_of_date"])
        post = _anchors(row, day)
        if available != post + timedelta(days=1) or available <= day or available >= cutoff:
            raise ValueError
    except (KeyError, TypeError, ValueError, OverflowError):
        raise errors.fail("EXPERIMENT_VARIANT_FAILED", "IV-crush label identity or availability is invalid") from None
    value, source, input_hash = _value(row)
    return value, available.isoformat(), source, input_hash


def _anchors(row, day):
    """Enforce the native session/gap rule without claiming quoted-row adjacency."""
    pre, post = calendar_date(row["pre_date"]), calendar_date(row["post_date"])
    if (not 0 < (post - pre).days <= 5
            or (row["session"] == "AMC" and not pre <= day < post)
            or (row["session"] == "BMO" and not pre < day <= post)):
        raise ValueError
    return post


def _value(row):
    """Validate eligible numeric content and unverified caller provenance syntax."""
    try:
        value, source, input_hash = row["crush_pct_iv30"], row["source_id"], row["input_hash"]
        if (type(value) not in (int, float) or not math.isfinite(value)
                or not isinstance(source, str) or not source.strip() or source != source.strip()
                or len(source) > 256 or not isinstance(input_hash, str)
                or len(input_hash) != 71 or not input_hash.startswith("sha256:")
                or any(c not in "0123456789abcdef" for c in input_hash[7:])):
            raise ValueError
    except (KeyError, TypeError, ValueError, OverflowError):
        raise errors.fail("EXPERIMENT_VARIANT_FAILED", "IV-crush label value or provenance is invalid") from None
    return value, source, input_hash


def load_iv_crush_targets(repository, snapshot, *, as_of_month, labels_before,
                          purpose="selection", event_ids=None):
    """Read complete admitted signed labels; skips/late labels refuse, never impute."""
    with progress.stage("IV target admission", report_refusals=True):
        try:
            cutoff = calendar_date(labels_before)
        except (TypeError, ValueError, OverflowError):
            raise errors.fail("CONTRACT_MISMATCH", "an explicit IV-crush label cutoff is required") from None
        rows, plans, future = _candidates(repository, snapshot, as_of_month, purpose, event_ids, cutoff)
    targets, consumed = [], []
    reporter = progress.Progress("IV targets", len(plans))
    with progress.stage("IV target reads", report_refusals=True):
        for event, request, record in plans:
            decoded = [row for batch in repository.scan(request, table_name=IV_CRUSH_TABLE_NAME,
                                                        max_object_bytes=MAX_TARGET_OBJECT_BYTES)
                       for row in batch.to_pylist()]
            target = _label_before(decoded, event, cutoff)
            targets.append(target)
            consumed.append({**_object_input(event, record), "source_id": target[2], "input_hash": target[3]})
            reporter.tick()
    rows["crush_pct_iv30"] = [target[0] for target in targets]
    rows["target_available_on"] = [target[1] for target in targets]
    rows["target_object_id"] = [item["object_id"] for item in consumed]
    identity = {"schema_version": "research_iv_crush_targets.v1", **_identity(rows, snapshot, purpose),
        "labels_before": cutoff.isoformat(), "consumed_inputs": consumed, "excluded_future_events": future}
    rows.attrs.update(identity, inputs_id=content_hash(identity))
    return rows


def validate_iv_crush_receipt(preflight, fold, receipt, targets):
    """Check stored evaluation evidence against admitted objects; never reopen labels."""
    events = set(fold["evaluation_event_ids"])
    planned = [row for row in preflight["planned_inputs"] if row["event_id"] in events]
    consumed = receipt["consumed_inputs"]
    if (len(planned) != len(events) or {row["event_id"] for row in planned} != events
            or not isinstance(consumed, list) or not isinstance(targets, list)
            or len(consumed) != len(planned) or len(targets) != len(planned)):
        raise ValueError
    for row, used, source in zip(targets, consumed, planned):
        _receipt_target(row, used, source)
    context = {key: value for key, value in preflight.items()
               if key not in {"schema_version", "event_ids", "planned_inputs", "preflight_id"}}
    identity = {**context, "schema_version": "research_iv_crush_targets.v1",
        "labels_before": fold["evaluation_labels_before"], "consumed_inputs": consumed,
        "excluded_future_events": []}
    expected = {**identity, "inputs_id": content_hash(identity), "holdout_exclusions": []}
    if canonical_json(receipt) != canonical_json(expected):
        raise ValueError


def _receipt_target(row, used, source):
    """Require one finite signed label and syntactically valid caller provenance per object."""
    if (set(row) != {"event_id", "crush_pct_iv30"} or row["event_id"] != source["event_id"]
            or set(used) != set(source) | {"source_id", "input_hash"}
            or {key: used[key] for key in source} != source):
        raise ValueError
    try:
        _value({**used, **row})
    except errors.DataError:
        raise ValueError from None
