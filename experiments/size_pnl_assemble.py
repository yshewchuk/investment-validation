"""EXP-188 S9c: assemble REPORT.md from the outer stage's receipts and decisions.parquet. Research-only: reached from `run.py --stage report`. Read-only over receipts; never rescores."""
import contextlib
import hashlib
import os
import resource
import stat
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from experiments import size_pnl_decisions as decisions, size_pnl_tuning as tuning, size_pnl_report as report, size_pnl_contrast as contrast
from experiments.size_pnl_inputs import PAIRS
from experiments.size_pnl_shortvol import Refusal
from experiments.size_pnl_straddle import RULES

_CONFLICT = "EXPERIMENT_IDENTITY_CONFLICT"
COLUMNS = ["model_id", "book", "arm_or_rule", "structure", "event_id", "event_date", "status", "rung", "pnl_dollars"]
ARMS = ("as_is", "retuned")
LONG_RULE = "R_L"
# name -> (predicted sign, arm -> legs, arms). A leg is (book, arm_or_rule value, sign); d = sum(sign * (median - mean)).
# Judgement call: H3 crosses each short-vol arm with the long rule R_L only; R_L2 is not in H3.
HYPOTHESES = {
    "H1": (1, lambda arm: [("short_vol", arm, 1)], ARMS),
    "H2": (-1, lambda rule: [("straddle", rule, 1)], RULES),
    "H3": (1, lambda arm: [("short_vol", arm, 1), ("straddle", LONG_RULE, -1)], ARMS),
}


def _log(message):
    print(f"[assemble] {message}", file=sys.stderr)


def _rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _refuse(reason, message):
    return Refusal(_CONFLICT, message, {"reason": reason})


@contextlib.contextmanager
def _admit(reason, message):
    """Turn a missing or mistyped field into a refusal; a Refusal (a ValueError) passes through unchanged."""
    try:
        yield
    except Refusal:
        raise
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise _refuse(reason, message) from exc


def _not_testable(title, reason):
    return f"### {title}\n\nnot_testable: {reason}"


def _rung_key(rung):
    return str(int(rung)) if isinstance(rung, (int, float, np.integer, np.floating)) else str(rung)


def _pnl(frame, label, book, value, ev):
    """One model's per-event P&L in one book for one arm or rule: taken pnl summed per event over all structures, 0 for no taken row."""
    rows = frame[(frame["model_id"] == label) & (frame["book"] == book) & (frame["arm_or_rule"] == value) & frame["taken"]]
    return rows.groupby("event_id")["pnl_dollars"].sum().reindex(ev, fill_value=0.0)


def _pair_reason(roster_pairs, pid, pair, outer_models):
    if roster_pairs.get(pid) == "not_testable" or None in pair:
        return "pair_member_excluded"
    if any(mid not in outer_models for mid in pair):
        return "pair_member_not_ready"
    return None


def _contrast_cell(frame, events, pair, legs, quarter_of, tick):
    """The contrast summary, or the reason it is not testable. ``pair`` is (median, mean) as labels."""
    median, mean = pair
    books = {book for book, _value, _sign in legs}
    ev = sorted({e for label in pair for book in books for e in events.get((label, book), ())})
    d = sum(sign * (_pnl(frame, median, book, value, ev) - _pnl(frame, mean, book, value, ev)) for book, value, sign in legs)
    quarters = np.array([quarter_of[e] for e in ev])
    tickers = np.array([tick[e] for e in ev])
    if len(set(quarters.tolist())) < 2 or len(set(tickers.tolist())) < 2:
        return "too_few_quarters_or_tickers"
    return contrast.contrast(d.to_numpy(dtype=float), quarters, tickers)


def _contrasts(frame, events, roster, role_model, label_of, outer_models, quarter_of, tick):
    """shown: {(hyp, pair, arm): summary or reason}; classify_in: {hyp: (sign, {pair: {arm: summary or None}})}.
    A summary missing a year of contrast.FULL_YEARS is shown but enters classify_in as None (not_testable)."""
    shown, classify_in = {}, {}
    for hyp, (sign, legs_for, arms) in HYPOTHESES.items():
        pairs = {}
        for pid, (mean_role, median_role) in PAIRS.items():
            raw = (role_model.get(median_role), role_model.get(mean_role))
            reason = _pair_reason(roster["pairs"], pid, raw, outer_models)
            pair = None if reason else tuple(label_of[mid] for mid in raw)
            pairs[pid] = {}
            for arm in arms:
                cell = reason or _contrast_cell(frame, events, pair, legs_for(arm), quarter_of, tick)
                shown[(hyp, pid, arm)] = cell
                full = isinstance(cell, dict) and set(contrast.FULL_YEARS) <= set(cell["year_signs"])
                pairs[pid][arm] = cell if full else None
        classify_in[hyp] = (sign, pairs)
    return shown, classify_in


def _shares(frame):
    """{label: {rung: share}} over each model's taken rows with a non-null rung, across books and arms."""
    taken = frame[frame["taken"] & frame["rung"].notna()]
    shares = {}
    for label, group in taken.groupby("model_id"):
        counts = group["rung"].map(_rung_key).value_counts()
        shares[label] = {rung: float(n) / len(group) for rung, n in counts.items()}
    return shares


def _run_section(outer):
    scope = "full" if outer["scope"] is None else f"SMOKE scope {outer['scope']}, not a full-population result"
    items = [("content_hash", outer["content_hash"]), ("decisions_sha256", outer["decisions_sha256"]), ("rows", outer["rows"]),
             ("scope", scope), ("common_dropped", outer["common_dropped"]),
             ("scoreability_dropped", outer["scoreability_dropped"]), ("ladder_skipped", outer["ladder_skipped"])]
    return "### Run\n\n" + "\n".join(f"- {key}: {value}" for key, value in items)


def _roster_rows(roster, outer_models, inputs, inner, label_of):
    rows = []
    for m in roster["models"]:
        mid = m["model_id"]
        reason = None if mid in outer_models else (inputs["models"].get(mid, {}).get("reason")
                                                   or inner.get("excluded", {}).get(mid) or "not_ready_in_outer")
        rows.append({"id": label_of[mid], "model": f"{m['kind']} {m['source']}/{m['candidate_id']}/{m['target_transform']}",
                     "type": m["kind"], "roles": m["roles"], "excluded_reason": reason})
    rows += [{"id": item["role"], "model": "-", "type": "-", "roles": [item["role"]], "excluded_reason": item["reason"]}
             for item in roster["not_testable"]]
    return rows


def _load_receipts(rdir):
    outer = tuning.load_receipt(rdir / "outer.json")
    inputs = tuning.load_receipt(rdir / "inputs_outer.json")
    inner = tuning.load_receipt(rdir / "inner.json")
    with _admit("report_receipt_malformed", "a receipt lacks a required field"):
        tuned = {mid: tuning.load_receipt(rdir / f"tuning_{mid}.json") for mid in outer["models"]}
        if outer.get("stage") != "outer":
            raise _refuse("report_receipt_stage", "outer.json is not the outer stage's receipt")
        broken = outer.get("inputs") != inputs["content_hash"]
        for mid, receipt in tuned.items():
            digest = receipt["content_hash"]
            broken |= outer.get("tuning", {}).get(mid) != digest or inner.get("models", {}).get(mid) != digest
            broken |= mid not in inputs.get("models", {})
    if broken:
        raise _refuse("report_receipt_chain_mismatch", "outer, inputs_outer, inner and the tuning receipts do not chain")
    return outer, inputs, inner, tuned


_HASH_BLOCK = 1 << 20


def _load_decisions(out, outer):
    path = out / "decisions.parquet"
    if not path.is_file():
        raise _refuse("report_decisions_missing", "decisions.parquet is missing")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        raise _refuse("report_decisions_missing", f"decisions.parquet cannot be opened safely: {exc}") from exc
    # One descriptor for the hash, the schema check and the read: the bytes hashed are the bytes parsed.
    with os.fdopen(fd, "rb") as handle:
        st = os.fstat(handle.fileno())
        if not stat.S_ISREG(st.st_mode) or st.st_size > decisions.MAX_DECISIONS_BYTES:
            raise _refuse("report_decisions_mismatch", "decisions.parquet must be a regular file of at most MAX_DECISIONS_BYTES")
        digest, total = hashlib.sha256(), 0
        while True:  # bounded by the cap, whatever fstat reported: never reads past MAX_DECISIONS_BYTES + 1 bytes
            block = handle.read(min(_HASH_BLOCK, decisions.MAX_DECISIONS_BYTES + 1 - total))
            if not block:
                break
            total += len(block)
            if total > decisions.MAX_DECISIONS_BYTES:
                raise _refuse("report_decisions_mismatch", "decisions.parquet is larger than MAX_DECISIONS_BYTES")
            digest.update(block)
        if digest.hexdigest() != outer.get("decisions_sha256"):
            raise _refuse("report_decisions_mismatch", "decisions.parquet does not match the outer receipt's decisions_sha256")
        handle.seek(0)
        parquet = pq.ParquetFile(handle)
        if not set(COLUMNS) <= set(parquet.schema_arrow.names):
            raise _refuse("report_decisions_columns", "decisions.parquet lacks a required column")
        frame = parquet.read(columns=COLUMNS).to_pandas()
    if len(frame) != outer["rows"]:
        raise _refuse("report_decisions_mismatch", "decisions.parquet row count differs from the outer receipt")
    if frame.empty:
        raise _refuse("report_decisions_empty", "decisions.parquet holds no rows")
    with _admit("report_decisions_invalid", "decisions.parquet has a non-date event_date or a non-numeric pnl_dollars"):
        frame = frame.assign(event_id=frame["event_id"].astype(str), event_date=pd.to_datetime(frame["event_date"]),
                             pnl_dollars=pd.to_numeric(frame["pnl_dollars"], errors="raise"))
    if frame["event_date"].isna().any():
        raise _refuse("report_decisions_invalid", "decisions.parquet has a blank event_date")
    return frame


def _tickers_ok(tickers):
    return isinstance(tickers, Mapping) and all(isinstance(k, str) and k.strip() and isinstance(v, str) and v.strip() for k, v in tickers.items())


def _sections(out, outer, inputs, inner, tuned, tickers):
    """The report's sections in order. Reads receipt fields throughout; the caller maps a missing or mistyped field to a refusal."""
    frame = _load_decisions(out, outer)
    tick = {str(e): str(t) for e, t in tickers.items()}
    if not set(frame["event_id"]) <= set(tick):
        raise _refuse("report_ticker_missing", "an event in decisions.parquet has no ticker")
    frame = frame.assign(taken=frame["status"] == "taken")
    if not np.isfinite(frame.loc[frame["taken"], "pnl_dollars"].to_numpy(dtype=float)).all():
        raise _refuse("report_nonfinite_pnl", "a taken row has a non-finite pnl_dollars")
    frame = frame.assign(quarter=frame["event_date"].dt.to_period("Q").astype(str))
    missing_years = sorted(set(contrast.FULL_YEARS) - {int(y) for y in frame["event_date"].dt.year})
    roster = inputs["roster"]
    label_of = {m["model_id"]: "+".join(m["roles"]) for m in roster["models"]}
    role_model = {role: m["model_id"] for m in roster["models"] for role in m["roles"]}
    outer_models = set(outer["models"])
    if (set(frame["model_id"]) | outer_models) - set(label_of):
        raise _refuse("report_receipt_chain_mismatch", "a model in the decisions has no roster entry")
    frame = frame.assign(model_id=frame["model_id"].map(label_of))  # tables show labels such as M0 or H-sq+H-abs
    quarter_of = frame.drop_duplicates("event_id").set_index("event_id")["quarter"].to_dict()
    events = {key: sorted(set(group)) for key, group in frame.groupby(["model_id", "book"])["event_id"]}
    labels = sorted({label_of[m] for m in outer_models} | set(frame["model_id"]))
    n_short = {lab: len(events.get((lab, "short_vol"), ())) for lab in labels}
    n_long = {lab: len(events.get((lab, "straddle"), ())) for lab in labels}
    tuning_by_label = {label_of[mid]: {k: v for k, v in receipt.items() if k != "content_hash"} for mid, receipt in tuned.items()}
    shown, classify_in = _contrasts(frame, events, roster, role_model, label_of, outer_models, quarter_of, tick)
    short = frame[frame["book"] == "short_vol"].rename(columns={"arm_or_rule": "arm"})
    long_rows = frame[frame["book"] == "straddle"].rename(columns={"arm_or_rule": "rule"})
    note = (f"per-quarter candidates and dropped counts are not recorded in decisions.parquet (not_testable); run totals from the receipt: "
            f"common_dropped={outer['common_dropped']}, scoreability_dropped={outer['scoreability_dropped']}.")
    verdicts = report.t10_verdicts(classify_in)
    if missing_years:
        verdicts += "\n\n" + (f"not_testable because the required full year(s) {missing_years} are absent from the outer decisions; "
                              "the inner window publishes no decisions, so this verdict needs inner evidence (design question, see PR).")
    return [
        _run_section(outer),
        report.t1_roster(_roster_rows(roster, outer_models, inputs, inner, label_of)),
        report.t2_in_e(frame[["event_id", "quarter"]].drop_duplicates(), note),
        _not_testable("T3 Forecast calibration", "labels_not_in_decisions"),
        report.t4_t5_shortvol(short, n_short, tuning_by_label, arm="as_is"),
        report.t4_t5_shortvol(short, n_short, tuning_by_label, arm="retuned"),
        report.t6_straddle(long_rows[long_rows["taken"]], long_rows, n_long),
        report.t7_contrasts(shown),
        report.t8_rungs(_shares(frame), models=[label_of[m] for m in outer["models"]]),
        _not_testable("T9 Pool sensitivity", "native_pool_arm_not_produced"),
        verdicts,
    ]


def assemble(out_dir, tickers, *, expected=None, spec_hash=None, snapshot=None):
    """Write ``out_dir/REPORT.md`` from the outer receipts and decisions.parquet; returns a JSON-safe summary.
    Each of expected, spec_hash and snapshot that is given must equal the matching field of inputs_outer.json."""
    if not _tickers_ok(tickers):
        raise _refuse("report_tickers_invalid", "tickers must map non-blank event ids to non-blank tickers")
    started, out = time.monotonic(), Path(out_dir)
    _log(f"start rss={_rss_mb():.0f}MB")
    outer, inputs, inner, tuned = _load_receipts(out / "receipts")
    requested = {"identity": expected, "spec_hash": spec_hash, "snapshot": snapshot}
    with _admit("report_receipt_malformed", "inputs_outer.json lacks a bound identity field"):
        mismatched = [key for key, want in requested.items() if want is not None and inputs[key] != want]
    if mismatched:
        raise _refuse("report_inputs_mismatch", f"inputs_outer.json differs from the requested {', '.join(mismatched)}")
    with _admit("report_receipt_malformed", "a receipt or decisions row lacks a required field"):
        sections = _sections(out, outer, inputs, inner, tuned, tickers)
    text = report.build_report(sections)
    path = out / "REPORT.md"
    handle = tempfile.NamedTemporaryFile("w", dir=out, delete=False, suffix=".tmp", encoding="utf-8")
    tmp = Path(handle.name)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        tuning._fsync_dir(out)
    finally:
        tmp.unlink(missing_ok=True)
    _log(f"end elapsed={time.monotonic() - started:.1f}s rss={_rss_mb():.0f}MB sections={len(sections)}")
    return {"report": str(path), "sections": len(sections), "rows": outer["rows"]}
