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
from experiments import size_pnl_driver, size_pnl_ladder
from experiments import size_pnl_synthetic_eval as synthetic_eval
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


def _run_section(outer, undetermined_excluded):
    items = [("content_hash", outer["content_hash"]), ("decisions_sha256", outer["decisions_sha256"]), ("rows", outer["rows"]),
             ("scope", "full"), ("common_dropped", outer["common_dropped"]),
             ("scoreability_dropped", outer["scoreability_dropped"]), ("ladder_skipped", outer["ladder_skipped"]),
             ("undetermined_events_excluded", undetermined_excluded)]
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
MAX_DECODED_BYTES = 1 << 31  # the uncompressed size decisions.parquet may declare in its row groups, checked before any decode


def _load_decisions(out, receipt, name="decisions.parquet"):
    path = out / name
    if not path.is_file():
        raise _refuse("report_decisions_missing", f"{name} is missing")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        raise _refuse("report_decisions_missing", f"{name} cannot be opened safely: {exc}") from exc
    # One descriptor for the hash, the schema check and the read: the bytes hashed are the bytes parsed.
    with os.fdopen(fd, "rb") as handle:
        st = os.fstat(handle.fileno())
        if not stat.S_ISREG(st.st_mode) or st.st_size > decisions.MAX_DECISIONS_BYTES:
            raise _refuse("report_decisions_mismatch", f"{name} must be a regular file of at most MAX_DECISIONS_BYTES")
        digest, total = hashlib.sha256(), 0
        while True:  # bounded by the cap, whatever fstat reported: never reads past MAX_DECISIONS_BYTES + 1 bytes
            block = handle.read(min(_HASH_BLOCK, decisions.MAX_DECISIONS_BYTES + 1 - total))
            if not block:
                break
            total += len(block)
            if total > decisions.MAX_DECISIONS_BYTES:
                raise _refuse("report_decisions_mismatch", f"{name} is larger than MAX_DECISIONS_BYTES")
            digest.update(block)
        if digest.hexdigest() != receipt["decisions_sha256"]:
            raise _refuse("report_decisions_mismatch", f"{name} does not match its receipt's decisions_sha256")
        handle.seek(0)
        parquet = pq.ParquetFile(handle)
        if not set(COLUMNS) <= set(parquet.schema_arrow.names):
            raise _refuse("report_decisions_columns", f"{name} lacks a required column")
        # Metadata only, before any decode: a highly compressed artifact must not exhaust memory on read.
        meta = parquet.metadata
        if meta.num_rows != receipt["rows"]:
            raise _refuse("report_decisions_mismatch", f"{name} row count differs from its receipt")
        decoded = sum(meta.row_group(i).total_byte_size for i in range(meta.num_row_groups))
        if decoded > MAX_DECODED_BYTES:
            raise _refuse("report_decisions_invalid", f"{name} decodes to more than MAX_DECODED_BYTES")
        # Optional columns, read only when the schema has them. "reason": population() exempts short-vol non-trades (published
        # decisions always carry it). "pnl_dollars_mid": the Amendment 1 sensitivity marks. "in_sample": the inner retuned arm.
        optional = [c for c in ("reason", "pnl_dollars_mid", "in_sample") if c in parquet.schema_arrow.names]
        frame = parquet.read(columns=COLUMNS + optional).to_pandas()
    if frame.empty:
        raise _refuse("report_decisions_empty", f"{name} holds no rows")
    mid = "pnl_dollars_mid" in optional
    with _admit("report_decisions_invalid", f"{name} has a non-date event_date or a non-numeric pnl_dollars or pnl_dollars_mid"):
        frame = frame.assign(event_id=frame["event_id"].astype(str), event_date=pd.to_datetime(frame["event_date"]),
                             pnl_dollars=pd.to_numeric(frame["pnl_dollars"], errors="raise"))
        if mid: frame = frame.assign(pnl_dollars_mid=pd.to_numeric(frame["pnl_dollars_mid"], errors="raise"))
    if frame["event_date"].isna().any():
        raise _refuse("report_decisions_invalid", f"{name} has a blank event_date")
    if not frame["status"].isin(report.STATUSES).all():
        raise _refuse("report_decisions_invalid", f"{name} has a status outside report.STATUSES")
    return frame


def _load_inner(out, outer, inner, tuned):
    """The inner stage's decisions, or None when the run has no inner_decisions receipt (older runs)."""
    path = out / "receipts" / "inner_decisions.json"
    if not path.exists():
        return None
    with _admit("report_receipt_malformed", "inner_decisions.json lacks a required field"):
        receipt = tuning.load_receipt(path)
        if receipt["stage"] != "inner_decisions":
            raise _refuse("report_receipt_stage", "inner_decisions.json is not the inner decisions stage's receipt")
        if not (isinstance(receipt["models"], list) and isinstance(receipt["tuning"], dict) and isinstance(receipt["c"], dict)):
            raise _refuse("report_receipt_malformed", "inner_decisions.json's models, tuning or c has the wrong type")
        # Subset semantics: the inner stage may have tuned models the outer excluded, so the receipt may cover a superset.
        if not (receipt["inputs"] == inner["inputs"] and receipt["scope"] == inner["scope"]
                and set(outer["models"]) <= set(receipt["models"])):
            raise _refuse("report_receipt_chain_mismatch", "inner_decisions.json is not bound to inner.json's inputs, scope or models")
        if any(receipt["tuning"].get(mid) != outer["tuning"][mid] or receipt["c"].get(mid) != tuned[mid]["c"] for mid in outer["models"]):
            raise _refuse("report_receipt_chain_mismatch", "inner_decisions.json's tuning or c does not match outer.json's or the tuning receipts'")
    return _load_decisions(out, receipt, "decisions_inner.parquet")  # the FULL inner frame: the caller checks every row before filtering to the outer's models


def _tickers_ok(tickers):
    return isinstance(tickers, Mapping) and all(isinstance(k, str) and k.strip() and isinstance(v, str) and v.strip() for k, v in tickers.items())


def _sens(text):
    """A sensitivity section, titled as Amendment 1 and never as the pre-registered verdict."""
    return "### Sensitivity (Amendment 1), not the pre-registered verdict: " + text[len("### "):]


def _inner_title(text):
    """An inner-window section, titled as the inner window and never as the headline."""
    return "### Inner window (2023H2-2024), not the headline: " + text[len("### "):]


def _insample_lines(shown):
    """One bullet per retuned (hyp, pair) cell that is a summary; 2024 is marked * because the retuned c was chosen on it."""
    lines = []
    for (hyp, pid, arm), cell in shown.items():
        if arm != "retuned" or not isinstance(cell, dict):
            continue
        signs = ", ".join(f"{year}{'*' if year == 2024 else ''} {int(np.sign(v)):+d}" for year, v in sorted(cell["year_signs"].items()))
        lines.append(f"- {hyp} {pid} retuned: year_signs {signs}")
    return "\n".join(lines)


def _verdicts(classify_in, missing_years, inner_present=False, shown_c=None):
    """The T10 section, with the note that a missing full year makes its verdicts not testable. With inner decisions it pools
    them with the outer ones and notes the in-sample retuned rows."""
    verdicts = report.t10_verdicts(classify_in)
    if missing_years and inner_present:
        verdicts += "\n\n" + f"not_testable because the required full year(s) {missing_years} are absent from the inner and outer decisions."
    elif missing_years:
        verdicts += "\n\n" + (f"not_testable because the required full year(s) {missing_years} are absent from the outer decisions; "
                              "the inner window publishes no decisions, so this verdict needs inner evidence (design question, see PR).")
    if inner_present:
        verdicts += ("\n\nT10 pools the inner (2023H2-2024) and outer (2025) decisions by the registered rule: predicted sign in every full year "
                     "(2024, 2025; 2023 partial). The retuned arm's 2023H2 and 2024 rows are in-sample (each c was chosen on those years), "
                     "so its 2024 sign is in-sample.\n" + _insample_lines(shown_c))
    return verdicts


def _sections(out, outer, inputs, inner, tuned, tickers):
    """The report's sections in order. Reads receipt fields throughout; the caller maps a missing or mistyped field to a refusal.
    Returns (sections, inner row count)."""
    raw = _load_decisions(out, outer)
    inner_raw = _load_inner(out, outer, inner, tuned)
    tick = {str(e): str(t) for e, t in tickers.items()}
    roster = inputs["roster"]
    label_of = {m["model_id"]: "+".join(m["roles"]) for m in roster["models"]}
    outer_models = set(outer["models"])
    if outer_models - set(label_of):
        raise _refuse("report_receipt_chain_mismatch", "a model in the decisions has no roster entry")
    role_model = {role: m["model_id"] for m in roster["models"] for role in m["roles"]}

    def _check(raw):
        """The raw-row checks: ticker present, finite taken P&L, model ids in the roster. They run before population drops any event."""
        if not set(raw["event_id"]) <= set(tick):
            raise _refuse("report_ticker_missing", "an event in decisions.parquet has no ticker")
        if not np.isfinite(raw.loc[raw["status"] == "taken", "pnl_dollars"].to_numpy(dtype=float)).all():
            raise _refuse("report_nonfinite_pnl", "a taken row has a non-finite pnl_dollars")
        if "pnl_dollars_mid" in raw.columns and not np.isfinite(raw.loc[raw["status"] == "taken", "pnl_dollars_mid"].to_numpy(dtype=float)).all():
            raise _refuse("report_nonfinite_pnl", "a taken row has a non-finite pnl_dollars_mid")
        if set(raw["model_id"]) - set(label_of):
            raise _refuse("report_receipt_chain_mismatch", "a model in the decisions has no roster entry")

    def _window(raw):
        """One window's evaluated frame (E) with its undetermined count; the checks run on the raw rows."""
        # Checked on the raw rows, before population drops the events with an undetermined row that leaves E: a bad row refuses, it is never skipped.
        _check(raw)
        frame = synthetic_eval.population(raw)  # E: no undetermined row that leaves E under any model; short-vol non-trades stay in E at P&L 0
        if frame.empty:
            raise _refuse("report_decisions_empty", "no event in decisions.parquet is in the evaluated population")
        undetermined = raw["event_id"].nunique() - frame["event_id"].nunique()
        frame = frame.assign(taken=frame["status"] == "taken", quarter=frame["event_date"].dt.to_period("Q").astype(str))
        return frame.assign(model_id=frame["model_id"].map(label_of)), undetermined  # tables show labels such as M0 or H-sq+H-abs

    def _context(frame):
        """The per-window inputs of the contrasts and the rung counts: events by (model, book), quarter of each event, and the counts."""
        quarter_of = frame.drop_duplicates("event_id").set_index("event_id")["quarter"].to_dict()
        events = {key: sorted(set(group)) for key, group in frame.groupby(["model_id", "book"])["event_id"]}
        labels = sorted({label_of[m] for m in outer_models} | set(frame["model_id"]))
        n_short = {lab: len(events.get((lab, "short_vol"), ())) for lab in labels}
        n_long = {lab: len(events.get((lab, "straddle"), ())) for lab in labels}
        return {"events": events, "quarter_of": quarter_of, "n_short": n_short, "n_long": n_long}

    tuning_by_label = {label_of[mid]: {k: v for k, v in receipt.items() if k != "content_hash"} for mid, receipt in tuned.items()}

    def tables(scored, ctx):
        """T4, T5, T6 and T7 of ``scored`` (its ``pnl_dollars`` is the P&L under test) in the window's context ``ctx``,
        with the contrast summaries ``shown`` and the T10 input ``classify`` that the same call computed."""
        shown, classify = _contrasts(scored, ctx["events"], roster, role_model, label_of, outer_models, ctx["quarter_of"], tick)
        short = scored[scored["book"] == "short_vol"].rename(columns={"arm_or_rule": "arm"})
        long_rows = scored[scored["book"] == "straddle"].rename(columns={"arm_or_rule": "rule"})
        sections = [report.t4_t5_shortvol(short, ctx["n_short"], tuning_by_label, arm="as_is"),
                    report.t4_t5_shortvol(short, ctx["n_short"], tuning_by_label, arm="retuned"),
                    report.t6_straddle(long_rows[long_rows["taken"]], long_rows, ctx["n_long"]),
                    report.t7_contrasts(shown)]
        return sections, shown, classify

    frame, undetermined_events = _window(raw)
    ctx_o = _context(frame)
    if inner_raw is not None:
        _check(inner_raw)  # every inner row, the outer-excluded models' included, is checked before the filter
        inner_raw = inner_raw[inner_raw["model_id"].isin(outer["models"])]  # the inner window of the models the outer ran; an empty result refuses in _window
    inner_frame, inner_undetermined = _window(inner_raw) if inner_raw is not None else (None, 0)
    inner_present = inner_frame is not None
    combined = pd.concat([inner_frame, frame]) if inner_present else frame  # T10 pools the windows by the registered rule
    ctx_c = _context(combined)
    missing_years = sorted(set(contrast.FULL_YEARS) - {int(y) for y in combined["event_date"].dt.year})
    o_tables, shown_o, classify_o = tables(frame, ctx_o)
    # Without an inner window the pooled frame is the outer frame: reuse its contrasts rather than bootstrapping them again.
    shown_c, classify_c = (_contrasts(combined, ctx_c["events"], roster, role_model, label_of, outer_models, ctx_c["quarter_of"], tick)
                           if inner_present else (shown_o, classify_o))
    note = (f"events with an undetermined burn-in or missing-data row (any book) or an undetermined straddle row are excluded from E: {undetermined_events}. "
            f"short-vol non-trades (e.g. width_declined) stay in E at P&L 0. "
            f"per-quarter candidates and dropped counts are not recorded in decisions.parquet (not_testable); run totals from the receipt: "
            f"common_dropped={outer['common_dropped']}, scoreability_dropped={outer['scoreability_dropped']}.")
    sections = [
        _run_section(outer, undetermined_events),
        report.t1_roster(_roster_rows(roster, outer_models, inputs, inner, label_of)),
        report.t2_in_e(frame[["event_id", "quarter"]].drop_duplicates(), note),
        _not_testable("T3 Forecast calibration", "labels_not_in_decisions"),
        *o_tables,
        report.t8_rungs(_shares(frame), models=[label_of[m] for m in outer["models"]]),
        _not_testable("T9 Pool sensitivity", "native_pool_arm_not_produced"),
        _verdicts(classify_c, missing_years, inner_present, shown_c),
    ]
    if inner_present:  # the inner window, after the outer tables: never the headline
        ctx_i = _context(inner_frame)
        inner_note = (f"events with an undetermined burn-in or missing-data row (any book) or an undetermined straddle row are excluded from E: "
                      f"{inner_undetermined}. Inner window only; run totals are the outer receipt's.")
        (t4_as_is, t5_retuned, t6, t7), _shown_i, _classify_i = tables(inner_frame, ctx_i)
        t2_inner = report.t2_in_e(inner_frame[["event_id", "quarter"]].drop_duplicates(), inner_note)
        t5_retuned += "\n\nretuned rows are IN-SAMPLE: each model's c was chosen on these same years."
        sections += [_inner_title(text) for text in (t2_inner, t4_as_is, t5_retuned, t6, t7)]
    inner_rows = len(inner_raw) if inner_present else 0
    if "pnl_dollars_mid" not in frame.columns:
        return sections + [_not_testable("Sensitivity (Amendment 1)", "pnl_dollars_mid_not_in_decisions")], inner_rows
    # The Amendment 1 marks: T4-T7 on the outer window, and T10 pooled over the same windows as the primary verdict.
    mid, shown_mo, classify_mo = tables(frame.assign(pnl_dollars=frame["pnl_dollars_mid"]), ctx_o)
    if inner_present:
        combined_mid = pd.concat([w.assign(pnl_dollars=w["pnl_dollars_mid"]) for w in (inner_frame, frame)])
        shown_m, classify_mid = _contrasts(combined_mid, ctx_c["events"], roster, role_model, label_of, outer_models, ctx_c["quarter_of"], tick)
    else:  # no inner window: the pooled mid frame is the outer one, so reuse its contrasts
        shown_m, classify_mid = shown_mo, classify_mo
    return sections + [_sens(text) for text in (*mid, _verdicts(classify_mid, missing_years, inner_present, shown_m))], inner_rows


def assemble(out_dir, plan, *, admitted, expected=None, spec_hash=None, snapshot=None):
    """Write ``out_dir/REPORT.md`` from the outer receipts and decisions.parquet; returns a JSON-safe summary.
    Each of expected, spec_hash and snapshot that is given must equal the matching field of inputs_outer.json.
    ``plan`` is the plan frame (sources.plan_frame) and ``admitted`` the sources' admitted events; the shared plan
    admission (size_pnl_ladder.check_plan) runs here, so every caller gets the same refusals as the scoring path.
    The ticker map and the plan binding (the receipted plan hash that a full run must match) are both derived from
    this one frame, so a changed ticker changes the binding and refuses."""
    if not isinstance(out_dir, (str, os.PathLike)):
        raise _refuse("report_out_dir_invalid", "out_dir must be a path")
    if not isinstance(plan, pd.DataFrame) or plan.columns.empty:
        raise _refuse("report_tickers_invalid", "plan must be a DataFrame with columns")
    try:
        size_pnl_ladder.check_plan(admitted, plan)  # raises its own typed refusals (OpsError, Refusal); those pass through
    except (KeyError, TypeError, AttributeError) as exc:
        raise _refuse("report_tickers_invalid", "the plan frame cannot be admitted") from exc
    if not {"event_id", "ticker"} <= set(plan.columns):
        raise _refuse("report_tickers_invalid", "plan must be a DataFrame with event_id and ticker columns")
    with _admit("report_tickers_invalid", "the plan's event ids or tickers cannot be read"):
        tickers = dict(zip(plan["event_id"].astype(str), plan["ticker"]))
        event_ids_unique = plan["event_id"].astype(str).is_unique
    if not (event_ids_unique and _tickers_ok(tickers)):
        raise _refuse("report_tickers_invalid", "tickers must map unique non-blank event ids to non-blank tickers")
    started, out = time.monotonic(), Path(out_dir)
    _log(f"start rss={_rss_mb():.0f}MB")
    outer, inputs, inner, tuned = _load_receipts(out / "receipts")
    with _admit("report_receipt_malformed", "outer.json lacks a scope field"):
        if outer["scope"] is not None:  # a smoke receipt binds a restricted plan that this report cannot recover
            raise _refuse("report_scope_unsupported", "the report covers full-population outer runs only; this receipt is a smoke scope")
    requested = {"identity": expected, "spec_hash": spec_hash, "snapshot": snapshot}
    with _admit("report_receipt_malformed", "inputs_outer.json lacks a bound identity field"):
        mismatched = [key for key, want in requested.items() if want is not None and inputs[key] != want]
        with _admit("report_tickers_invalid", "the plan frame cannot be bound"):
            binding = size_pnl_driver.plan_binding(plan)
        plan_bad = inputs.get("plan") != binding
    if mismatched:
        raise _refuse("report_inputs_mismatch", f"inputs_outer.json differs from the requested {', '.join(mismatched)}")
    if plan_bad:
        raise _refuse("report_inputs_mismatch", "the plan's tickers do not match the receipted plan")
    with _admit("report_receipt_malformed", "a receipt or decisions row lacks a required field"):
        sections, inner_rows = _sections(out, outer, inputs, inner, tuned, tickers)
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
    _log(f"end elapsed={time.monotonic() - started:.1f}s rss={_rss_mb():.0f}MB sections={len(sections)} inner_rows={inner_rows}")
    return {"report": str(path), "sections": len(sections), "rows": outer["rows"]}
