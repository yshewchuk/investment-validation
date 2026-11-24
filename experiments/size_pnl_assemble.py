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

from experiments import size_pnl_decisions as decisions, size_pnl_receipts as receipts, size_pnl_report as report, size_pnl_contrast as contrast
from experiments import size_pnl_account, size_pnl_driver, size_pnl_ladder
from experiments import size_pnl_synthetic_eval as synthetic_eval
from experiments import size_pnl_gprod_report as gprod_report
from experiments import size_pnl_board_match as board_match
from experiments.size_pnl_inputs import PAIRS
from experiments.size_pnl_shortvol import Refusal
from experiments.size_pnl_straddle import RULES, GATE_RULE

_CONFLICT = "EXPERIMENT_IDENTITY_CONFLICT"
COLUMNS = ["model_id", "book", "arm_or_rule", "structure", "event_id", "event_date", "status", "rung", "pnl_dollars"]
# Amendment 2 account columns: optional, read only when the run stored them.
ACCOUNT_COLS = ["entry_date", "exit_date", "entry_cost", "exit_value", "exit_value_mid", "secured_per_contract"]
ARMS = ("as_is",)
LONG_RULE = "R_L"
AMENDMENT_5_NOTE = ("Amendment 5 (2026-11-23): the straddle universe was changed to production's (market cap at or above engine.score "
                    "GATE_MCAP_FLOOR, no spread term) after the 2025 outputs of the pre-registered run were inspected; the H2 verdict, "
                    "T6 and the straddle T7 rows are computed on E_str and are labelled as run under Amendment 5.")
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
    # The paired events: for each book, the union over the pair's labels of that book's events; then the intersection over the books.
    # One book gives that book's paired population (H1, H2); H3 crosses both books, so it uses E intersect E_str (Amendment 5).
    per_book = [set().union(*(set(events.get((label, book), ())) for label in pair)) for book in books]
    ev = sorted(set.intersection(*per_book))
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


def _run_section(outer, undetermined_excluded, undetermined_straddle_excluded):
    items = [("content_hash", outer["content_hash"]), ("decisions_sha256", outer["decisions_sha256"]), ("rows", outer["rows"]),
             ("scope", "full"), ("common_dropped", outer["common_dropped"]),
             ("scoreability_dropped", outer["scoreability_dropped"]), ("ladder_skipped", outer["ladder_skipped"]),
             ("undetermined_events_excluded", undetermined_excluded),
             ("undetermined_straddle_events_excluded", undetermined_straddle_excluded)]
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
    outer = receipts.load_receipt(rdir / "outer.json")
    inputs = receipts.load_receipt(rdir / "inputs_outer.json")
    inner = receipts.load_receipt(rdir / "inner.json")
    with _admit("report_receipt_malformed", "a receipt lacks a required field"):
        if outer.get("stage") != "outer":
            raise _refuse("report_receipt_stage", "outer.json is not the outer stage's receipt")
        broken = outer.get("inputs") != inputs["content_hash"]
        broken |= not set(outer["models"]) <= set(inner.get("models", []))  # inner.json lists every outer model
        broken |= any(mid not in inputs.get("models", {}) for mid in outer["models"])
    if broken:
        raise _refuse("report_receipt_chain_mismatch", "outer, inputs_outer and inner do not chain")
    return outer, inputs, inner


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
        if "secured_per_contract" in parquet.schema_arrow.names and not {"entry_date", "exit_date", "entry_cost", "exit_value"} <= set(parquet.schema_arrow.names):
            raise _refuse("report_decisions_columns", f"{name} has secured_per_contract but lacks entry_date, exit_date, entry_cost or exit_value")
        # Metadata only, before any decode: a highly compressed artifact must not exhaust memory on read.
        meta = parquet.metadata
        if meta.num_rows != receipt["rows"]:
            raise _refuse("report_decisions_mismatch", f"{name} row count differs from its receipt")
        decoded = sum(meta.row_group(i).total_byte_size for i in range(meta.num_row_groups))
        if decoded > MAX_DECODED_BYTES:
            raise _refuse("report_decisions_invalid", f"{name} decodes to more than MAX_DECODED_BYTES")
        # Optional columns, read only when the schema has them. "reason": population() exempts short-vol non-trades (published
        # decisions always carry it). "pnl_dollars_mid": the Amendment 1 sensitivity marks.
        optional = [c for c in ("reason", "pnl_dollars_mid") if c in parquet.schema_arrow.names]
        account = [c for c in ACCOUNT_COLS if c in parquet.schema_arrow.names]  # Amendment 2 account columns, when the run stored them
        frame = parquet.read(columns=COLUMNS + optional + account).to_pandas()
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


def _load_inner(out, outer, inner):
    """The inner stage's decisions, or None when the run has no inner_decisions receipt (older runs)."""
    path = out / "receipts" / "inner_decisions.json"
    if not path.exists():
        if (out / "decisions_inner.parquet").exists():
            raise _refuse("report_receipt_chain_mismatch", "decisions_inner.parquet has no inner_decisions.json receipt")
        return None
    with _admit("report_receipt_malformed", "inner_decisions.json lacks a required field"):
        receipt = receipts.load_receipt(path)
        if receipt["stage"] != "inner_decisions":
            raise _refuse("report_receipt_stage", "inner_decisions.json is not the inner decisions stage's receipt")
        if not isinstance(receipt["models"], list):
            raise _refuse("report_receipt_malformed", "inner_decisions.json's models has the wrong type")
        # Subset semantics: the inner stage may have scored models the outer excluded, so the receipt may cover a superset.
        if not (receipt["inputs"] == inner["inputs"] and receipt["scope"] == inner["scope"]
                and set(outer["models"]) <= set(receipt["models"])):
            raise _refuse("report_receipt_chain_mismatch", "inner_decisions.json is not bound to inner.json's inputs, scope or models")
    return _load_decisions(out, receipt, "decisions_inner.parquet")  # the FULL inner frame: the caller checks every row before filtering to the outer's models


def _tickers_ok(tickers):
    return isinstance(tickers, Mapping) and all(isinstance(k, str) and k.strip() and isinstance(v, str) and v.strip() for k, v in tickers.items())


def _sens(text):
    """A sensitivity section, titled as Amendment 1 and never as the pre-registered verdict."""
    return "### Sensitivity (Amendment 1), not the pre-registered verdict: " + text[len("### "):]


def _inner_title(text):
    """An inner-window section, titled as the inner window and never as the headline."""
    return "### Inner window (2023H2-2024), not the headline: " + text[len("### "):]


def _verdicts(classify_in, missing_years, inner_present=False):
    """The T10 section, with the note that a missing full year makes its verdicts not testable. With inner decisions it pools
    them with the outer ones."""
    verdicts = report.t10_verdicts(classify_in)
    if missing_years and inner_present:
        verdicts += "\n\n" + f"not_testable because the required full year(s) {missing_years} are absent from the inner and outer decisions."
    elif missing_years:
        verdicts += "\n\n" + (f"not_testable because the required full year(s) {missing_years} are absent from the outer decisions; "
                              "the inner window publishes no decisions, so this verdict needs inner evidence (design question, see PR).")
    if inner_present:
        verdicts += ("\n\nT10 pools the inner (2023H2-2024) and outer (2025) decisions by the registered rule: predicted sign in every full year "
                     "(2024, 2025; 2023 partial).")
    verdicts += "\n\n" + AMENDMENT_5_NOTE
    return verdicts


def _account_sections(frame, role_model, label_of, universe):
    """The Amendment 2 sensitivity sections over ``frame``, which holds ALL taken rows of decisions.parquet (model_id already
    labelled), or not_testable when the frame has no secured_per_contract column, or when the frame is non-empty and no taken
    row carries a non-null secured_per_contract. An empty frame that has the column is testable: every account in ``universe``
    is reported at its empty-account metrics ($100,000). A straddle-only run is testable: its taken straddle rows carry 0.0
    secured cash, which is non-null. ``universe`` holds every decision row (model_id labelled) and supplies the account keys,
    so a model that takes no trade is still reported."""
    title = "Sensitivity (Amendment 2)"
    if "secured_per_contract" not in frame.columns or (not frame.empty and not frame["secured_per_contract"].notna().any()):
        return [_not_testable(title, "account_columns_not_in_decisions")]
    results = size_pnl_account.accounts(frame, universe)
    _log(f"accounts n={len(results)}")
    pairs = {pid: (label_of[role_model[mean]], label_of[role_model[median]]) for pid, (mean, median) in PAIRS.items()
             if mean in role_model and median in role_model}  # (mean, median) labels; a pair with an absent member is skipped downstream
    diffs = size_pnl_account.differences(results, pairs, sorted(set(universe["model_id"])))
    return size_pnl_account.tables(results, diffs)


def _sections(out, outer, inputs, inner, tickers, stored_gate=None, board_ledger=None):
    """The report's sections in order. Reads receipt fields throughout; the caller maps a missing or mistyped field to a refusal.
    Returns (sections, inner row count)."""
    raw = _load_decisions(out, outer)
    gate_rows = raw["arm_or_rule"] == GATE_RULE  # Amendment 3: G_PROD rows never enter E, E_str, T1-T10 or the verdicts; only the Amendment 2 accounts read them
    raw, gate_raw = raw[~gate_rows], raw[gate_rows]
    inner_raw = _load_inner(out, outer, inner)
    tick = {str(e): str(t) for e, t in tickers.items()}
    roster = inputs["roster"]
    label_of = {m["model_id"]: "+".join(m["roles"]) for m in roster["models"]}
    outer_models = set(outer["models"])
    if outer_models - set(label_of):
        raise _refuse("report_receipt_chain_mismatch", "a model in the decisions has no roster entry")
    role_model = {role: m["model_id"] for m in roster["models"] for role in m["roles"]}

    def _check(raw):
        """The raw-row checks: ticker present, finite taken P&L and account cash, valid account dates, model ids in the roster. They run before population drops any event."""
        if not raw["book"].isin(["short_vol", "straddle"]).all():  # null is not in the list, so a null book refuses too; an empty frame passes
            raise _refuse("report_decisions_invalid", "a row in decisions.parquet has a book other than short_vol or straddle")
        if not set(raw["event_id"]) <= set(tick):
            raise _refuse("report_ticker_missing", "an event in decisions.parquet has no ticker")
        if not np.isfinite(raw.loc[raw["status"] == "taken", "pnl_dollars"].to_numpy(dtype=float)).all():
            raise _refuse("report_nonfinite_pnl", "a taken row has a non-finite pnl_dollars")
        if "pnl_dollars_mid" in raw.columns and not np.isfinite(raw.loc[raw["status"] == "taken", "pnl_dollars_mid"].to_numpy(dtype=float)).all():
            raise _refuse("report_nonfinite_pnl", "a taken row has a non-finite pnl_dollars_mid")
        if any(not np.isfinite(raw.loc[raw["status"] == "taken", col].to_numpy(dtype=float)).all() for col in ("secured_per_contract", "entry_cost", "exit_value", "exit_value_mid") if col in raw.columns):
            raise _refuse("report_nonfinite_pnl", "a taken row has a non-finite secured_per_contract, entry_cost, exit_value or exit_value_mid")
        for col in ("entry_date", "exit_date"):  # Amendment 2 account dates: a taken row with a null or unparseable date refuses
            if col in raw.columns and pd.to_datetime(raw.loc[raw["status"] == "taken", col], errors="coerce").isna().any():
                raise _refuse("report_decisions_invalid", "a taken row has a missing or invalid entry_date or exit_date")
        if set(raw["model_id"]) - set(label_of):
            raise _refuse("report_receipt_chain_mismatch", "a model in the decisions has no roster entry")

    _check(gate_raw)

    def _window(raw):
        """One window's evaluated frame and its two undetermined counts: short-vol E and straddle E_str (Amendment 5).
        Each population is computed from its own book's rows, so a straddle row never removes an event from E and a short-vol row never
        removes one from E_str. The checks run on the raw rows."""
        # Checked on the raw rows, before population drops the events with an undetermined row that leaves its population: a bad row refuses, it is never skipped.
        _check(raw)
        short = synthetic_eval.population(raw[raw["book"] == "short_vol"])  # E: no undetermined short-vol row that leaves E; short-vol non-trades stay in E at P&L 0
        long = synthetic_eval.population(raw[raw["book"] == "straddle"])  # E_str: no undetermined straddle row that leaves E_str under R_L or R_L2 (G_PROD rows are already stripped)
        frame = pd.concat([short, long])
        if frame.empty:
            raise _refuse("report_decisions_empty", "no event in decisions.parquet is in the evaluated population")
        undet_e = raw.loc[raw["book"] == "short_vol", "event_id"].nunique() - short["event_id"].nunique()
        undet_str = raw.loc[raw["book"] == "straddle", "event_id"].nunique() - long["event_id"].nunique()
        frame = frame.assign(taken=frame["status"] == "taken", quarter=frame["event_date"].dt.to_period("Q").astype(str))
        return frame.assign(model_id=frame["model_id"].map(label_of)), undet_e, undet_str  # tables show labels such as M0 or H-sq+H-abs

    def _context(frame):
        """The per-window inputs of the contrasts and the rung counts: events by (model, book), quarter of each event, and the counts."""
        quarter_of = frame.drop_duplicates("event_id").set_index("event_id")["quarter"].to_dict()
        events = {key: sorted(set(group)) for key, group in frame.groupby(["model_id", "book"])["event_id"]}
        labels = sorted({label_of[m] for m in outer_models} | set(frame["model_id"]))
        n_short = {lab: len(events.get((lab, "short_vol"), ())) for lab in labels}
        n_long = {lab: len(events.get((lab, "straddle"), ())) for lab in labels}
        return {"events": events, "quarter_of": quarter_of, "n_short": n_short, "n_long": n_long}

    def tables(scored, ctx):
        """T4, T6 and T7 of ``scored`` (its ``pnl_dollars`` is the P&L under test) in the window's context ``ctx``,
        with the contrast summaries ``shown`` and the T10 input ``classify`` that the same call computed."""
        shown, classify = _contrasts(scored, ctx["events"], roster, role_model, label_of, outer_models, ctx["quarter_of"], tick)
        short = scored[scored["book"] == "short_vol"].rename(columns={"arm_or_rule": "arm"})
        long_rows = scored[scored["book"] == "straddle"].rename(columns={"arm_or_rule": "rule"})
        sections = [report.t4_shortvol(short, ctx["n_short"]),
                    report.t6_straddle(long_rows[long_rows["taken"]], long_rows, ctx["n_long"]),
                    report.t7_contrasts(shown)]
        return sections, shown, classify

    frame, undet_e, undet_str = _window(raw)
    ctx_o = _context(frame)
    if inner_raw is not None:
        _check(inner_raw)  # every inner row, the outer-excluded models' included, is checked before the filter
        inner_raw = inner_raw[inner_raw["model_id"].isin(outer["models"])]  # the inner window of the models the outer ran; an empty result refuses in _window
    inner_frame, inner_undet_e, inner_undet_str = _window(inner_raw) if inner_raw is not None else (None, 0, 0)
    inner_present = inner_frame is not None
    combined = pd.concat([inner_frame, frame]) if inner_present else frame  # T10 pools the windows by the registered rule
    ctx_c = _context(combined)
    missing_years = sorted(set(contrast.FULL_YEARS) - {int(y) for y in combined["event_date"].dt.year})
    o_tables, shown_o, classify_o = tables(frame, ctx_o)
    # Without an inner window the pooled frame is the outer frame: reuse its contrasts rather than bootstrapping them again.
    _shown_c, classify_c = (_contrasts(combined, ctx_c["events"], roster, role_model, label_of, outer_models, ctx_c["quarter_of"], tick)
                            if inner_present else (shown_o, classify_o))
    note = (f"short-vol undetermined rows exclude events from E: {undet_e}; "
            f"straddle undetermined rows exclude events from E_str (H2, T6, straddle T7, G_PROD, the H2 verdict): {undet_str}; "
            f"H3 uses E intersect E_str. "
            f"short-vol non-trades (e.g. width_declined) stay in E at P&L 0. "
            f"per-quarter candidates and dropped counts are not recorded in decisions.parquet (not_testable); run totals from the receipt: "
            f"common_dropped={outer['common_dropped']}, scoreability_dropped={outer['scoreability_dropped']}.")
    sections = [
        _run_section(outer, undet_e, undet_str),
        report.t1_roster(_roster_rows(roster, outer_models, inputs, inner, label_of)),
        report.t2_in_e(frame[frame["book"] == "short_vol"][["event_id", "quarter"]].drop_duplicates(), note),
        _not_testable("T3 Forecast calibration", "labels_not_in_decisions"),
        *o_tables,
        report.t8_rungs(_shares(frame), models=[label_of[m] for m in outer["models"]]),
        _not_testable("T9 Pool sensitivity", "native_pool_arm_not_produced"),
        _verdicts(classify_c, missing_years, inner_present),
    ]
    if inner_present:  # the inner window, after the outer tables: never the headline
        ctx_i = _context(inner_frame)
        inner_note = (f"short-vol undetermined rows exclude events from E: {inner_undet_e}; "
                      f"straddle undetermined rows exclude events from E_str (H2, T6, straddle T7, G_PROD, the H2 verdict): {inner_undet_str}; "
                      f"H3 uses E intersect E_str. Inner window only; run totals are the outer receipt's.")
        (t4_as_is, t6, t7), _shown_i, _classify_i = tables(inner_frame, ctx_i)
        t2_inner = report.t2_in_e(inner_frame[inner_frame["book"] == "short_vol"][["event_id", "quarter"]].drop_duplicates(), inner_note)
        sections += [_inner_title(text) for text in (t2_inner, t4_as_is, t6, t7)]
    inner_rows = len(inner_raw) if inner_present else 0
    if "pnl_dollars_mid" not in frame.columns:
        sections += [_not_testable("Sensitivity (Amendment 1)", "pnl_dollars_mid_not_in_decisions")]
    else:
        # The Amendment 1 marks: T4-T7 on the outer window, and T10 pooled over the same windows as the primary verdict.
        mid, _shown_mo, classify_mo = tables(frame.assign(pnl_dollars=frame["pnl_dollars_mid"]), ctx_o)
        if inner_present:
            combined_mid = pd.concat([w.assign(pnl_dollars=w["pnl_dollars_mid"]) for w in (inner_frame, frame)])
            _shown_m, classify_mid = _contrasts(combined_mid, ctx_c["events"], roster, role_model, label_of, outer_models, ctx_c["quarter_of"], tick)
        else:  # no inner window: the pooled mid frame is the outer one, so reuse its contrasts
            classify_mid = classify_mo
        sections += [_sens(text) for text in (*mid, _verdicts(classify_mid, missing_years, inner_present))]
    if not gate_raw.empty:  # Amendment 3: the G_PROD sections, after the Amendment 1 ones and before the accounts
        sections += gprod_report.sections(gate_raw, frame[frame["book"] == "straddle"], label_of=label_of, role_model=role_model, roster=roster, outer_models=outer_models,
                                          quarter_of=ctx_o["quarter_of"], tick=tick, stored_gate=stored_gate)
    m0 = raw[(raw["model_id"] == role_model.get("M0")) & (raw["book"] == "short_vol") & (raw["arm_or_rule"] == "as_is")][["event_id", "structure", "event_date"]]
    sections.append(board_match.section(board_ledger, m0, None))  # Amendment 6: the board-match check, research-only
    every = pd.concat([raw, gate_raw])  # Amendment 2 accounts read every row, G_PROD included
    taken_all = every[every["status"] == "taken"].assign(model_id=every["model_id"].map(label_of), taken=True)  # Amendment 2: every taken row, not only E
    universe = every.assign(model_id=every["model_id"].map(label_of))  # Amendment 2: every decision row, of any status, discovers the account keys
    sections += _account_sections(taken_all, role_model, label_of, universe)  # Amendment 2, on both paths
    return sections, inner_rows


def assemble(out_dir, plan, *, admitted, expected=None, spec_hash=None, snapshot=None, stored_gate=None, board_ledger=None):
    """Write ``out_dir/REPORT.md`` from the outer receipts and decisions.parquet; returns a JSON-safe summary.
    Each of expected, spec_hash and snapshot that is given must equal the matching field of inputs_outer.json.
    ``stored_gate`` is optional {'scored': ids, 'selected': ids} from production's stored walk-forward 2025 gate decisions; None reports the M0 agreement as not_available.
    ``plan`` is the plan frame (sources.plan_frame) and ``admitted`` the sources' admitted events; the shared plan
    admission (size_pnl_ladder.check_plan) runs here, so every caller gets the same refusals as the scoring path.
    The ticker map and the plan binding (the receipted plan hash that a full run must match) are both derived from
    this one frame, so a changed ticker changes the binding and refuses."""
    if not isinstance(out_dir, (str, os.PathLike)):
        raise _refuse("report_out_dir_invalid", "out_dir must be a path")
    if board_ledger is not None and (not isinstance(board_ledger, (str, os.PathLike)) or not os.fspath(board_ledger).strip()):  # blank path refuses too, as in size_pnl_sources._config
        raise _refuse("report_board_ledger_invalid", "board_ledger must be a non-blank path or None")
    if stored_gate is not None and not gprod_report.stored_gate_valid(stored_gate):
        raise _refuse("report_stored_gate_invalid", "stored_gate must be {scored, selected} lists of event id strings, selected within scored")
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
    outer, inputs, inner = _load_receipts(out / "receipts")
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
    if (out / "receipts" / "inner_decisions.json").exists():  # the inner window's tickers feed the pooled T10 bootstrap: bind them too
        with _admit("report_receipt_malformed", "inputs_inner.json is missing or malformed"):
            inner_path = out / "receipts" / "inputs_inner.json"
            if not inner_path.exists():
                raise _refuse("report_receipt_malformed", "inputs_inner.json is missing")
            inner_inputs = receipts.load_receipt(inner_path)
            bound_inputs = inner.get("inputs")
            inner_unchained = bound_inputs is None or inner_inputs.get("content_hash") != bound_inputs
            with _admit("report_tickers_invalid", "the plan frame cannot be bound"):
                inner_binding = size_pnl_driver.inner_plan_binding(plan)
            inner_bad = inner_inputs.get("plan") != inner_binding
        if inner_unchained:
            raise _refuse("report_receipt_chain_mismatch", "inputs_inner.json is not the receipt inner.json binds")
        if inner_bad:
            raise _refuse("report_inputs_mismatch", "the plan's inner-window rows do not match inputs_inner.json")
    with _admit("report_receipt_malformed", "a receipt or decisions row lacks a required field"):
        sections, inner_rows = _sections(out, outer, inputs, inner, tickers, stored_gate, board_ledger)
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
        receipts._fsync_dir(out)
    finally:
        tmp.unlink(missing_ok=True)
    _log(f"end elapsed={time.monotonic() - started:.1f}s rss={_rss_mb():.0f}MB sections={len(sections)} inner_rows={inner_rows}")
    return {"report": str(path), "sections": len(sections), "rows": outer["rows"]}
