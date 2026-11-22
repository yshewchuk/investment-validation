"""EXP-188 real-store Sources adapter, slice 2: synthetic in-memory frames only (no store, no network)."""
import datetime as dt
import re
import types

import numpy as np
import pandas as pd
import pytest

from engine.v2.contracts import KeyPredicate
from engine.v2.data.errors import DataError
from engine.v2.data.errors import fail as data_fail
from engine.v2.ops.errors import OpsError
from experiments import size_pnl_ladder as ladder
from experiments import size_pnl_readers as readers
from experiments import size_pnl_tables as tbl

MAX = 1 << 20
HASH = "tier4-manifest-9c2e41"  # literal manifest hash; deliberately unlike any other fake value
CRUSH_HASH = "iv-crush-manifest-4b7d10"  # the crush snapshot's iv_crush_targets manifest; also unlike any other fake value
SNAP = types.SimpleNamespace(table_versions={"tier4_forecasts": types.SimpleNamespace(manifest_hash=HASH)})
CRUSH = types.SimpleNamespace(table_versions={"iv_crush_targets": types.SimpleNamespace(manifest_hash=CRUSH_HASH)})
CRUSH_TABLES = ("earnings_events", "iv_crush_targets")
D1, D2 = pd.Timestamp("2025-01-02"), pd.Timestamp("2025-02-03")
DAY = pd.Timedelta(days=1)
PROD = tuple(readers.PRODUCTION_COLUMNS)
LABEL = tuple(readers.LABEL_FETCH)
AVAILABLE = readers.AVAILABLE
ADMITTED = readers.Admitted({"events": pd.DataFrame({"event_id": ["e1", "e2"], "event_date": [D1, D2]}), "excluded": {}})
# Three admitted events: e1 AAA/D1 (AMC, forecast present), e2 BBB/D2 (BMO, NaN forecast), e3 CCC/D1 (AMC, no forecast row).
E3 = pd.DataFrame({"event_id": ["e1", "e2", "e3"], "ticker": ["AAA", "BBB", "CCC"], "event_date": [D1, D2, D1],
                   "session": ["AMC", "BMO", "AMC"]})
FC3 = pd.DataFrame({"ticker": ["AAA", "BBB"], "event_date": [D1, D2], "pred_abs_move": [4.0, np.nan], "pred_iv_crush_30": [-1.0, -2.0]})
ADMITTED3 = readers.Admitted({"events": E3[["event_id", "event_date"]], "excluded": {}})


def _refusal(exc):
    return exc.value.problem.code, exc.value.problem.details.get("reason")


def _events(sessions=("AMC", "BMO")):
    return pd.DataFrame({"event_id": ["e1", "e2"], "ticker": ["AAA", "BBB"], "event_date": [D1, D2], "session": list(sessions)})


def _frames(**over):
    frames = {
        "earnings_events": _events(),
        "tier4_forecasts": pd.DataFrame({"ticker": ["AAA", "BBB", "CCC"], "event_date": [D1, D2, D1],
                                         "pred_abs_move": [4.0, 5.0, 6.0], "pred_iv_crush_30": [-1.0, -2.0, -3.0]}),
        "computed_moves": pd.DataFrame({"ticker": ["AAA", "BBB"], "event_date": [D1, D2],
                                        "realized_move_pct": [-3.0, 2.0], "skipped": [False, False],
                                        "available_as_of_date": ["2025-01-03", "2025-02-04"]}),
        "iv_crush_targets": pd.DataFrame({"ticker": ["AAA", "BBB"], "event_date": [D1, D2],
                                          "crush_pct_iv30": [-10.0, -20.0], "skipped": [False, True],
                                          "available_as_of_date": ["2025-01-03", "2025-02-04"]}),
        "feature_panel": pd.DataFrame({"ticker": ["AAA", "BBB"], "date": [D1, D2], "pre_iv30": [0.3, 0.4],
                                       "mcap_usd": [5e9, 2e9], "or_implied": [4.0, 5.0]}),
    }
    frames.update(over)
    return frames


def _served(snapshot, table):
    """Two-snapshot split: the crush snapshot serves only its own tables, and iv_crush_targets is never served by the main snapshot."""
    if snapshot is CRUSH:
        return table in CRUSH_TABLES
    return table != "iv_crush_targets"


def _crush_of(frames, **over):
    """The crush snapshot's frames for a test: the same earnings_events and iv_crush_targets unless overridden.
    Its earnings_events carries the crush-context columns (date_conflict defaults to False, event_cluster_id to event_id)."""
    crush = {k: frames[k] for k in ("earnings_events", "iv_crush_targets") if k in frames}
    crush.update(over)
    if "earnings_events" in crush:
        events = crush["earnings_events"]
        crush["earnings_events"] = events.assign(date_conflict=events.get("date_conflict", False),
                                                 event_cluster_id=events.get("event_cluster_id", events["event_id"]))
    return crush


def _crush_keys(crush_frames):
    frame = crush_frames.get("iv_crush_targets")
    return [] if frame is None else [(t, d.date().isoformat()) for t, d in zip(frame["ticker"], frame["event_date"])]


ALL_KEYS = [("AAA", "2025-01-02"), ("BBB", "2025-02-03")]


class _Repo:
    """Fake repository: fragment_records lists one fragment per (ticker, date text) key, for iv_crush_targets (`keys`) and
    computed_moves (`moves`; every admitted event unless given)."""
    def __init__(self, keys=ALL_KEYS, moves=ALL_KEYS):
        self.keys = list(keys)
        self.moves = list(moves)

    def fragment_records(self, snapshot, table_name):
        keys = {"iv_crush_targets": self.keys, "computed_moves": self.moves}[table_name]
        return [types.SimpleNamespace(partition_key=t, time_min=d) for t, d in keys]


REPO = _Repo()


class _Read:
    """In-memory stand-in for read_table. Takes no partition_keys (passing one raises TypeError), applies ticker `in`
    key_filters, then batch_filter, and records what the caller kept. The crush snapshot reads from `crush_frames`."""
    def __init__(self, frames, crush_frames=None):
        self.frames = frames
        self.crush_frames = _crush_of(frames) if crush_frames is None else crush_frames
        self.kept = []
        self.calls = []
        self.labels = None

    def __call__(self, repository, snapshot, table, columns, *, key_filter=(), batch_filter=None, max_object_bytes=None):
        self.calls.append({"table": table, "key_filter": key_filter, "max_object_bytes": max_object_bytes, "crush": snapshot is CRUSH})
        if not _served(snapshot, table):
            raise data_fail("CONTRACT_MISMATCH", "table is not part of this snapshot", details={"table_name": table})
        source = self.crush_frames if snapshot is CRUSH else self.frames
        if table not in source:
            raise data_fail("CONTRACT_MISMATCH", "table is not part of this snapshot", details={"table_name": table})
        if any(c not in source[table].columns for c in columns):
            raise data_fail("CONTRACT_MISMATCH", "a requested column is not in the table", details={"table_name": table})
        frame = source[table]
        for pred in key_filter:
            assert pred.operator == "in" and pred.column == "ticker"
            frame = frame[frame["ticker"].isin(pred.values).to_numpy()]
        frame = frame[list(columns)].copy()
        if batch_filter is not None:
            frame = batch_filter(frame)
        self.kept.append((table, frame.copy()))
        return frame.reset_index(drop=True)


class _Labels:
    """Stand-in for label_rows: returns only the requested events' rows of the table frame, and records each call."""
    def __init__(self, frames, crush_frames=None):
        self.frames = frames
        self.crush_frames = frames if crush_frames is None else crush_frames
        self.calls = []

    def __call__(self, repository, snapshot, table, columns, want, *, max_object_bytes):
        self.calls.append({"table": table, "events": list(want["event_id"]), "max_object_bytes": max_object_bytes,
                           "crush": snapshot is CRUSH})
        if not _served(snapshot, table):
            raise data_fail("CONTRACT_MISMATCH", "table is not part of this snapshot", details={"table_name": table})
        source = self.crush_frames if snapshot is CRUSH else self.frames
        if table not in source:
            raise data_fail("CONTRACT_MISMATCH", "table is not part of this snapshot", details={"table_name": table})
        frame = source[table]
        if any(c not in frame.columns for c in columns):
            raise data_fail("CONTRACT_MISMATCH", "a requested column is not in the table", details={"table_name": table})
        keys = set(zip(want["ticker"], want["event_date"]))
        hit = np.asarray([key in keys for key in zip(frame["ticker"], frame["event_date"])], dtype=bool)
        return frame.loc[hit, list(columns)].to_dict("records")


class _Cal:
    def __init__(self, missing=()):
        self.missing = set(missing)
        self.calls = []

    def last_pre_print(self, day, session):
        self.calls.append(("last_pre_print", day, session))
        if day in self.missing:
            raise KeyError(f"no trading day before {day.date()}")
        return day - DAY

    def first_post_print(self, day, session):
        self.calls.append(("first_post_print", day, session))
        return day + DAY


def _build(frames=None, *, cal=None, admitted=ADMITTED, snapshot=SNAP, expected_hash=HASH, crush=CRUSH, crush_frames=None,
           repository=None):
    frames = _frames() if frames is None else frames
    crush_frames = _crush_of(frames) if crush_frames is None else crush_frames
    read, labels = _Read(frames, crush_frames), _Labels(frames, crush_frames)
    cal = _Cal() if cal is None else cal
    repo = _Repo(_crush_keys(crush_frames)) if repository is None else repository
    ns = tbl.tables(repo, snapshot, admitted, crush_snapshot=crush, max_object_bytes=MAX, expected_hash=expected_hash, read=read,
                    label_rows=labels, calendar=lambda repository, snap, *, max_object_bytes: cal)
    read.labels = labels
    return ns, read


def _set(frame, column, value):
    out = frame.copy()
    out[column] = pd.Series([value, *out[column].iloc[1:]], dtype=object)
    return out


# --- table_hash and missing table --------------------------------------------

def test_table_hash_is_forecast_manifest_hash():
    ns, _ = _build()
    assert ns.table_hash == HASH


def test_hash_mismatch_refused_before_any_read():
    read = _Read(_frames())
    with pytest.raises(OpsError) as exc:
        tbl.tables(REPO, SNAP, ADMITTED, crush_snapshot=CRUSH, max_object_bytes=MAX, expected_hash="OTHER", read=read,
                   calendar=lambda r, s, *, max_object_bytes: _Cal())
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "production_hash_mismatch")
    assert read.calls == []


def test_calendar_receives_max_object_bytes():
    seen = {}

    def calendar(repository, snap, *, max_object_bytes):
        seen["max_object_bytes"] = max_object_bytes
        return _Cal()

    tbl.tables(REPO, SNAP, ADMITTED, crush_snapshot=CRUSH, max_object_bytes=MAX, expected_hash=HASH, read=_Read(_frames()), calendar=calendar)
    assert seen == {"max_object_bytes": MAX}


def test_calendar_contract_mismatch_refused_as_missing():
    def calendar(repository, snap, *, max_object_bytes):
        raise data_fail("CONTRACT_MISMATCH", "calendar table is absent")

    with pytest.raises(OpsError) as exc:
        tbl.tables(REPO, SNAP, ADMITTED, crush_snapshot=CRUSH, max_object_bytes=MAX, expected_hash=HASH, read=_Read(_frames()),
                   label_rows=_Labels(_frames()), calendar=calendar)
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "source_table_missing")


def test_missing_forecast_table_refused_before_any_read():
    read = _Read(_frames())
    snap = types.SimpleNamespace(table_versions={})
    with pytest.raises(OpsError) as exc:
        tbl.tables(REPO, snap, ADMITTED, crush_snapshot=CRUSH, max_object_bytes=MAX, expected_hash=HASH, read=read,
                   calendar=lambda r, s, *, max_object_bytes: _Cal())
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "source_table_missing")
    assert read.kept == [] and read.calls == []


def test_missing_earnings_events_refused_at_construction():
    frames = _frames()
    del frames["earnings_events"]
    with pytest.raises(OpsError) as exc:
        _build(frames)
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "source_table_missing")


@pytest.mark.parametrize("events", [
    pd.concat([_events(), _events().iloc[[0]]], ignore_index=True),
], ids=["duplicate_event_id"])
def test_duplicate_event_id_refused(events):
    with pytest.raises(OpsError) as exc:
        _build(_frames(earnings_events=events))
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "source_value_invalid")


def test_admitted_event_absent_from_earnings_events_refused():
    admitted = readers.Admitted({"events": pd.DataFrame({"event_id": ["e1", "e2", "e3"], "event_date": [D1, D2, D1]}), "excluded": {}})
    with pytest.raises(OpsError) as exc:
        _build(admitted=admitted)
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "source_table_missing")


# --- production fetch --------------------------------------------------------

def test_production_fetch_joins_to_event_id_and_restricts_to_requested():
    ns, _ = _build()
    out = ns.fetch(PROD, ["e1"]).to_dict("records")
    assert out == [{"event_id": "e1", "pred_abs_move": 4.0, "pred_iv_crush_30": -1.0, "horizon_sessions": 1}]
    both = ns.fetch(PROD, ["e1", "e2"]).set_index("event_id")
    assert both.loc["e1"].tolist() == [4.0, -1.0, 1]
    assert both.loc["e2"].tolist() == [5.0, -2.0, 1]


def test_production_absent_and_nan_forecasts_are_excluded_not_refused(capsys):
    frames = _frames(earnings_events=E3, tier4_forecasts=FC3)  # e1 forecast; e2 NaN; e3 no forecast row
    ns, _ = _build(frames, admitted=ADMITTED3)
    out = ns.fetch(PROD, ["e1", "e2", "e3"])
    assert out.to_dict("list") == {"event_id": ["e1"], "pred_abs_move": [4.0], "pred_iv_crush_30": [-1.0], "horizon_sessions": [1]}
    assert "[pnl_tables] excluded production_forecast_missing=2 of 3" in capsys.readouterr().err


def test_production_no_requested_event_with_a_forecast_refuses_empty_population():
    ns, _ = _build(_frames(earnings_events=E3, tier4_forecasts=FC3), admitted=ADMITTED3)
    with pytest.raises(OpsError) as exc:
        ns.fetch(PROD, ["e2", "e3"])
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "empty_population")


def test_readers_drop_absent_and_nan_forecast_events_end_to_end(capsys):
    ns, _ = _build(_frames(earnings_events=E3, tier4_forecasts=FC3), admitted=ADMITTED3)
    prod = readers.read_production(ns.fetch, ADMITTED3, expected_hash=HASH, table_hash=ns.table_hash)
    assert list(prod["frame"].index) == ["e1"]
    assert prod["frame"].loc["e1", "pred_move"] == 4.0 and prod["frame"].loc["e1", "pred_crush"] == -1.0
    assert "[pnl_readers] production rows=1 production_missing=2" in capsys.readouterr().err


def test_production_duplicate_key_refused():
    dup = pd.DataFrame({"ticker": ["AAA", "AAA"], "event_date": [D1, D1],
                        "pred_abs_move": [4.0, 5.0], "pred_iv_crush_30": [-1.0, -2.0]})
    ns, _ = _build(_frames(tier4_forecasts=dup))
    with pytest.raises(OpsError) as exc:
        ns.fetch(PROD, ["e1"])
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "source_value_invalid")


@pytest.mark.parametrize("column, value", [
    ("pred_abs_move", np.inf),
    ("pred_abs_move", 3.4028235e38),
    ("pred_iv_crush_30", 3.0e38),
], ids=["inf", "flt_max", "at_bound"])
def test_production_sentinel_or_nonfinite_prediction_refused(column, value):
    frames = _frames()
    frames["tier4_forecasts"] = _set(frames["tier4_forecasts"], column, value)  # row 0 is AAA/D1, event e1
    ns, _ = _build(frames)
    with pytest.raises(OpsError) as exc:
        ns.fetch(PROD, ["e1"])
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "source_value_invalid")


# --- label fetch -------------------------------------------------------------

def test_label_fetch_takes_abs_move_and_nans_skipped_crush():
    ns, _ = _build()
    out = ns.fetch(LABEL, ["e1", "e2"]).set_index("event_id")
    assert out.loc["e1", "abs_move"] == 3.0 and out.loc["e1", "crush_pct_iv30"] == -10.0
    assert out.loc["e2", "abs_move"] == 2.0 and np.isnan(out.loc["e2", "crush_pct_iv30"])


def test_label_skipped_blanks_move_too():
    frames = _frames()
    frames["computed_moves"] = _set(frames["computed_moves"], "skipped", True)
    ns, _ = _build(frames)
    out = ns.fetch(LABEL, ["e1"]).set_index("event_id")
    assert np.isnan(out.loc["e1", "abs_move"]) and out.loc["e1", "crush_pct_iv30"] == -10.0


def test_label_nan_value_allowed_and_comes_back_nan():
    frames = _frames()
    frames["computed_moves"] = _set(frames["computed_moves"], "realized_move_pct", np.nan)
    ns, _ = _build(frames)
    out = ns.fetch(LABEL, ["e1"]).set_index("event_id")
    assert np.isnan(out.loc["e1", "abs_move"])


def test_label_with_no_returned_row_is_excluded_not_refused(capsys):
    # e2 has a move fragment but its label read returns no row: NaN value, None availability, counted on stderr.
    frames = _frames(
        computed_moves=pd.DataFrame({"ticker": ["AAA"], "event_date": [D1], "realized_move_pct": [-3.0], "skipped": [False],
                                     "available_as_of_date": ["2025-01-03"]}),
        iv_crush_targets=pd.DataFrame({"ticker": ["AAA"], "event_date": [D1], "crush_pct_iv30": [-10.0], "skipped": [False],
                                       "available_as_of_date": ["2025-01-03"]}))
    ns, _ = _build(frames)
    out = ns.fetch(LABEL, ["e2"]).set_index("event_id")
    assert np.isnan(out.loc["e2", "abs_move"]) and np.isnan(out.loc["e2", "crush_pct_iv30"]) and out.loc["e2", AVAILABLE] is None
    assert "[pnl_tables] excluded move_label_missing=1 crush_label_missing=1 of 1" in capsys.readouterr().err


def test_event_without_move_fragment_is_excluded_and_never_planned(capsys):
    ns, read = _build(repository=_Repo(moves=[("AAA", "2025-01-02")]))
    out = ns.fetch(LABEL, ["e1", "e2"]).set_index("event_id")
    assert out.loc["e1", "abs_move"] == 3.0
    assert np.isnan(out.loc["e2", "abs_move"]) and pd.isna(out.loc["e2", AVAILABLE])
    assert [c["events"] for c in read.labels.calls if c["table"] == "computed_moves"] == [["e1"]]
    assert "[pnl_tables] excluded move_label_missing=1 crush_label_missing=1 of 2" in capsys.readouterr().err


def test_readers_drop_move_label_missing_event_end_to_end(capsys):
    ns, _ = _build(repository=_Repo(moves=[("AAA", "2025-01-02")]))
    labels = readers.read_labels(ns.fetch, ADMITTED)
    assert list(labels.index) == ["e1"] and labels.loc["e1", "abs_move"] == 3.0
    err = capsys.readouterr().err
    assert "[pnl_tables] excluded move_label_missing=1 crush_label_missing=1 of 2" in err
    assert "[pnl_readers] labels requested=2 label_missing=1" in err


def test_label_two_rows_for_one_event_refused():
    frames = _frames(computed_moves=pd.DataFrame({
        "ticker": ["AAA", "AAA"], "event_date": [D1, D1], "realized_move_pct": [-3.0, -4.0], "skipped": [False, False],
        "available_as_of_date": ["2025-01-03", "2025-01-03"]}))
    ns, _ = _build(frames)
    with pytest.raises(OpsError) as exc:
        ns.fetch(LABEL, ["e1"])
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "source_value_invalid")


@pytest.mark.parametrize("table, column, value", [
    ("computed_moves", "realized_move_pct", np.inf),
    ("computed_moves", "realized_move_pct", 3.4028235e38),
    ("iv_crush_targets", "crush_pct_iv30", -np.inf),
    ("iv_crush_targets", "crush_pct_iv30", 3.0e38),
], ids=["move_inf", "move_flt_max", "crush_inf", "crush_at_bound"])
def test_label_sentinel_or_inf_refused(table, column, value):
    frames = _frames()
    frames[table] = _set(frames[table], column, value)  # row 0 is AAA/D1, event e1
    ns, _ = _build(frames)
    with pytest.raises(OpsError) as exc:
        ns.fetch(LABEL, ["e1"])
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "source_value_invalid")


def test_label_read_is_called_only_with_requested_events_once_per_table():
    frames = _frames(computed_moves=pd.DataFrame({
        "ticker": ["AAA", "AAA", "BBB"], "event_date": [D1, pd.Timestamp("2024-06-03"), D2],
        "realized_move_pct": [-3.0, 9.0, 2.0], "skipped": [False, False, False],
        "available_as_of_date": ["2025-01-03", "2024-06-04", "2025-02-04"]}))
    ns, read = _build(frames)
    out = ns.fetch(LABEL, ["e1"])
    assert [c["table"] for c in read.labels.calls] == ["computed_moves", "iv_crush_targets"]
    assert all(c["events"] == ["e1"] for c in read.labels.calls)
    assert out.set_index("event_id").loc["e1", "abs_move"] == 3.0


def test_plan_refusal_from_label_rows_propagates_unchanged():
    def refuse(repository, snapshot, table, columns, want, *, max_object_bytes):
        raise data_fail("HOLDOUT_ACCESS_DENIED", "event is not isolated")

    ns = tbl.tables(REPO, SNAP, ADMITTED, crush_snapshot=CRUSH, max_object_bytes=MAX, expected_hash=HASH, read=_Read(_frames()),
                    label_rows=refuse, calendar=lambda r, s, *, max_object_bytes: _Cal())
    with pytest.raises(DataError) as exc:
        ns.fetch(LABEL, ["e1"])
    assert exc.value.code == "HOLDOUT_ACCESS_DENIED"


def test_contract_mismatch_from_label_read_maps_to_source_table_missing():
    frames = _frames()
    del frames["computed_moves"]
    ns, _ = _build(frames)
    with pytest.raises(OpsError) as exc:
        ns.fetch(LABEL, ["e1"])
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "source_table_missing")


class _UnlistedRepo(_Repo):
    """Fake repository whose fragment listing for one table raises CONTRACT_MISMATCH, as a snapshot without that table does."""
    def __init__(self, table):
        super().__init__()
        self.table = table

    def fragment_records(self, snapshot, table_name):
        if table_name == self.table:
            raise data_fail("CONTRACT_MISMATCH", "table is not part of this snapshot", details={"table_name": table_name})
        return super().fragment_records(snapshot, table_name)


@pytest.mark.parametrize("table", ["computed_moves", "iv_crush_targets"])
def test_fragment_listing_contract_mismatch_refused_as_source_table_missing(table):
    with pytest.raises(OpsError) as exc:
        _build(repository=_UnlistedRepo(table))
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "source_table_missing")


class _Batch:
    def __init__(self, rows):
        self._rows = rows

    def to_pylist(self):
        return list(self._rows)


class _ScanRepo:
    def table_contract(self, snapshot, table):
        return ("contract", table)

    def scan(self, request, *, table_name, max_object_bytes):
        return [_Batch([{"ticker": request[0], "event_date": request[1]}])]


LABEL_LOG = re.compile(r"^\[pnl_tables\] read (\w+) events=(\d+) rows=(\d+) elapsed=\d+\.\d+s rss=\d+MiB$", re.M)


def test_label_rows_plans_only_requested_events_and_logs_one_line(monkeypatch, capsys):
    seen = {}

    def plan(repository, snapshot, candidates, contract, *, columns):
        seen.update(event_ids=list(candidates["event_id"]), contract=contract, columns=columns)
        return [(eid, (t, d), None) for eid, t, d in zip(candidates["event_id"], candidates["ticker"], candidates["event_date"])]

    monkeypatch.setattr(tbl, "plan_event_targets", plan)
    want = pd.DataFrame({"event_id": ["e1"], "ticker": ["AAA"], "event_date": [D1]})
    rows = tbl.label_rows(_ScanRepo(), SNAP, "computed_moves", ["ticker", "event_date"], want, max_object_bytes=MAX)
    assert rows == [{"ticker": "AAA", "event_date": D1}]
    assert seen == {"event_ids": ["e1"], "contract": ("contract", "computed_moves"), "columns": ("ticker", "event_date")}
    assert LABEL_LOG.findall(capsys.readouterr().err) == [("computed_moves", "1", "1")]


class _PlanRepo:
    """Fake repository for the real label_rows: per-table contract, and one scan per request returning one batch."""
    def __init__(self):
        self.scans = []

    def table_contract(self, snapshot, table):
        return ("contract", table)

    def fragment_records(self, snapshot, table_name):
        return [types.SimpleNamespace(partition_key=t, time_min=d) for t, d in ALL_KEYS]

    def scan(self, request, *, table_name, max_object_bytes):
        self.scans.append((request, table_name, max_object_bytes))
        return [_Batch([{"ticker": request[0], "event_date": request[1]}])]


def test_label_rows_real_plan_passes_contract_tuple_columns_and_scans_each_event(monkeypatch):
    seen = []

    def plan(repository, snapshot, candidates, contract, *, columns):
        seen.append({"repository": repository, "snapshot": snapshot, "contract": contract, "columns": columns})
        return [("e1", ("AAA", D1), "rec1"), ("e2", ("BBB", D2), "rec2")]

    monkeypatch.setattr(tbl, "plan_event_targets", plan)
    repo = _PlanRepo()
    want = pd.DataFrame({"event_id": ["e1", "e2"], "ticker": ["AAA", "BBB"], "event_date": [D1, D2]})
    rows = tbl.label_rows(repo, SNAP, "iv_crush_targets", ["ticker", "event_date"], want, max_object_bytes=MAX)
    assert len(seen) == 1
    assert seen[0]["repository"] is repo and seen[0]["snapshot"] is SNAP
    assert seen[0]["contract"] == repo.table_contract(SNAP, "iv_crush_targets") == ("contract", "iv_crush_targets")
    assert isinstance(seen[0]["columns"], tuple) and seen[0]["columns"] == ("ticker", "event_date")
    assert repo.scans == [(("AAA", D1), "iv_crush_targets", MAX), (("BBB", D2), "iv_crush_targets", MAX)]
    assert rows == [{"ticker": "AAA", "event_date": D1}, {"ticker": "BBB", "event_date": D2}]


def test_label_rows_real_plan_refusal_propagates_unchanged_and_scans_nothing(monkeypatch):
    refusal = data_fail("HOLDOUT_ACCESS_DENIED", "event is not isolated")

    def plan(repository, snapshot, candidates, contract, *, columns):
        raise refusal

    monkeypatch.setattr(tbl, "plan_event_targets", plan)
    repo = _PlanRepo()
    want = pd.DataFrame({"event_id": ["e1"], "ticker": ["AAA"], "event_date": [D1]})
    with pytest.raises(DataError) as exc:
        tbl.label_rows(repo, SNAP, "computed_moves", ["ticker"], want, max_object_bytes=MAX)
    assert exc.value is refusal
    assert exc.value.code == "HOLDOUT_ACCESS_DENIED"
    assert repo.scans == []


def test_empty_keys_read_nothing():
    ns, read = _build()
    before = len(read.calls)
    assert ns.fetch(LABEL, []).empty
    assert ns.fetch(PROD, []).empty
    assert read.calls[before:] == []
    assert read.labels.calls == []


def test_reads_never_take_partition_keys_and_logical_tables_filter_by_ticker():
    ns, read = _build()
    ns.fetch(PROD, ["e1"])
    ns.facts(["e1"])
    logical = [c for c in read.calls if c["table"] in ("tier4_forecasts", "feature_panel")]
    assert {c["table"] for c in logical} == {"tier4_forecasts", "feature_panel"}
    for call in logical:
        assert call["key_filter"] == (KeyPredicate(column="ticker", operator="in", values=("AAA",)),)


READ_LOG = re.compile(r"^\[pnl_tables\] read (\w+) keys=(\d+) rows=(\d+) elapsed=\d+\.\d+s rss=\d+MiB$", re.M)


def test_read_progress_logged_once_per_read(capsys):
    ns, _ = _build()
    ns.fetch(PROD, ["e1"])
    logged = READ_LOG.findall(capsys.readouterr().err)
    assert logged == [("earnings_events", "2", "2"), ("tier4_forecasts", "1", "1")]


def test_unsupported_column_set_refused():
    ns, _ = _build()
    with pytest.raises(OpsError) as exc:
        ns.fetch(("abs_move",), ["e1"])
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "source_value_invalid")


# --- facts -------------------------------------------------------------------

def test_facts_returns_indexed_sizing_inputs_with_or_implied_as_implied_move():
    ns, _ = _build()
    out = ns.facts(["e1", "e2"])
    assert list(out.index) == ["e1", "e2"]
    assert list(out.columns) == ["pre_iv30", "mcap_usd", "implied_move"]
    assert out["pre_iv30"].tolist() == [0.3, 0.4]
    assert out["mcap_usd"].tolist() == [5e9, 2e9]
    assert out["implied_move"].tolist() == [4.0, 5.0]


def test_facts_never_reads_labels_or_computed_moves():
    ns, read = _build()
    ns.facts(["e1", "e2"])
    assert all(c["table"] != "computed_moves" for c in read.calls)
    assert read.labels.calls == []


def test_facts_missing_feature_panel_row_refused():
    frames = _frames(feature_panel=pd.DataFrame({"ticker": ["AAA"], "date": [D1], "pre_iv30": [0.3],
                                                 "mcap_usd": [5e9], "or_implied": [4.0]}))
    with pytest.raises(OpsError) as exc:
        _build(frames)[0].facts(["e1", "e2"])
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "source_table_missing")


@pytest.mark.parametrize("column, value", [
    ("pre_iv30", np.nan),
    ("mcap_usd", np.inf),
    ("pre_iv30", 3.4028235e38),
    ("mcap_usd", 5e3),
    ("pre_iv30", "abc"),
    ("or_implied", 0.0),
], ids=["nan", "inf", "flt_max", "mcap_thousands", "non_numeric", "implied_not_positive"])
def test_facts_invalid_values_refused(column, value):
    frames = _frames()
    frames["feature_panel"] = _set(frames["feature_panel"], column, value)
    ns, _ = _build(frames)
    with pytest.raises(OpsError) as exc:
        ns.facts(["e1", "e2"])
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "source_value_invalid")


@pytest.mark.parametrize("column, value, passes", [
    ("mcap_usd", 1e6, True),
    ("mcap_usd", 999999.99, False),
    ("pre_iv30", 2.9e38, True),
    ("pre_iv30", 3.0e38, False),
    ("pre_iv30", 0.0, False),
], ids=["mcap_exactly_1e6", "mcap_just_below_1e6", "pre_iv30_below_flt_max", "pre_iv30_at_flt_max", "pre_iv30_zero"])
def test_facts_boundaries(column, value, passes):
    frames = _frames()
    frames["feature_panel"] = _set(frames["feature_panel"], column, value)
    ns, _ = _build(frames)
    if passes:
        assert ns.facts(["e1", "e2"]).loc["e1", column] == value
    else:
        with pytest.raises(OpsError) as exc:
            ns.facts(["e1", "e2"])
        assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "source_value_invalid")


# --- plan_frame --------------------------------------------------------------

def test_plan_frame_uses_calendar_windows_per_session_and_normalises_session():
    cal = _Cal()
    ns, _ = _build(_frames(earnings_events=_events(sessions=("amc", "bmo"))), cal=cal)
    plan = ns.plan_frame
    assert list(plan.columns) == ["event_id", "ticker", "event_date", "session", "entry_date", "exit_date"]
    assert plan["event_id"].tolist() == ["e1", "e2"]
    assert plan["session"].tolist() == ["AMC", "BMO"]
    assert plan["entry_date"].tolist() == [D1 - DAY, D2 - DAY]
    assert plan["exit_date"].tolist() == [D1 + DAY, D2 + DAY]
    assert ("last_pre_print", D1, "AMC") in cal.calls and ("first_post_print", D2, "BMO") in cal.calls


def test_plan_frame_drops_bad_session_and_logs_counts(capsys):
    ns, _ = _build(_frames(earnings_events=_events(sessions=("AMC", "XYZ"))))
    assert ns.plan_frame["event_id"].tolist() == ["e1"]
    assert "[pnl_tables] plan_frame events=2 dropped=1" in capsys.readouterr().err


def test_plan_frame_drops_calendar_key_error():
    ns, _ = _build(cal=_Cal(missing=[D1]))
    assert ns.plan_frame["event_id"].tolist() == ["e2"]


def test_plan_frame_refuses_when_no_event_survives():
    with pytest.raises(OpsError) as exc:
        _build(_frames(earnings_events=_events(sessions=("XYZ", "XYZ"))))
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "plan_frame_empty")


def test_plan_frame_passes_check_plan():
    ns, _ = _build()
    plan = ns.plan_frame
    assert list(plan.columns) == ["event_id", "ticker", "event_date", "session", "entry_date", "exit_date"]
    assert plan.to_dict("records") == [
        {"event_id": "e1", "ticker": "AAA", "event_date": D1, "session": "AMC", "entry_date": D1 - DAY, "exit_date": D1 + DAY},
        {"event_id": "e2", "ticker": "BBB", "event_date": D2, "session": "BMO", "entry_date": D2 - DAY, "exit_date": D2 + DAY},
    ]
    assert ladder.check_plan(ADMITTED, plan) is None


class _SessionCal:
    """Distinct windows per session: AMC enters on the event day and exits the next day; BMO enters the day before and exits on the event day."""
    def last_pre_print(self, day, session):
        return day if session == "AMC" else day - DAY

    def first_post_print(self, day, session):
        return day + DAY if session == "AMC" else day


def test_plan_frame_windows_are_session_specific_for_amc_and_bmo():
    ns, _ = _build(cal=_SessionCal())
    plan = ns.plan_frame.set_index("event_id")
    assert plan.loc["e1", "session"] == "AMC"
    assert plan.loc["e1", "entry_date"] == pd.Timestamp("2025-01-02") and plan.loc["e1", "exit_date"] == pd.Timestamp("2025-01-03")
    assert plan.loc["e2", "session"] == "BMO"
    assert plan.loc["e2", "entry_date"] == pd.Timestamp("2025-02-02") and plan.loc["e2", "exit_date"] == pd.Timestamp("2025-02-03")


# --- availability conversion -------------------------------------------------

def test_iso_day_converts_dates_keeps_text_and_null_and_passes_other_types_through():
    assert tbl._iso_day(dt.date(2025, 1, 3)) == "2025-01-03"
    assert tbl._iso_day(pd.Timestamp("2025-01-03 15:30")) == "2025-01-03"
    assert tbl._iso_day("2025-01-03") == "2025-01-03"
    assert tbl._iso_day(None) is None and tbl._iso_day(pd.NaT) is None and tbl._iso_day(np.nan) is None
    assert tbl._iso_day(3) == 3


# --- read errors -------------------------------------------------------------

def test_contract_mismatch_from_forecast_read_maps_to_source_table_missing():
    frames = _frames()
    del frames["tier4_forecasts"]
    ns, _ = _build(frames)
    with pytest.raises(OpsError) as exc:
        ns.fetch(PROD, ["e1"])
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "source_table_missing")


# --- end to end through the readers ------------------------------------------

def test_readers_accept_production_and_label_frames():
    ns, _ = _build()
    assert ns.table_hash == HASH
    prod = readers.read_production(ns.fetch, ADMITTED, expected_hash=HASH, table_hash=ns.table_hash)
    assert prod["table_hash"] == HASH
    assert prod["binding"] == readers._production_binding(HASH, prod["frame"])
    assert list(prod["frame"].index) == ["e1", "e2"]
    labels = readers.read_labels(ns.fetch, ADMITTED)
    assert list(labels.index) == ["e1"] and labels.loc["e1", "abs_move"] == 3.0


# --- label availability, zero rows, planner denial and scan progress ---------------

def test_label_availability_is_the_later_of_the_two_tables():
    frames = _frames()
    frames["iv_crush_targets"] = _set(frames["iv_crush_targets"], AVAILABLE, "2025-01-09")  # row 0 is AAA/D1, event e1
    out = _build(frames)[0].fetch(LABEL, ["e1"]).set_index("event_id")
    assert out.loc["e1", AVAILABLE] == "2025-01-09"  # crush later than moves ("2025-01-03")
    frames["computed_moves"] = _set(frames["computed_moves"], AVAILABLE, "2025-01-12")
    out = _build(frames)[0].fetch(LABEL, ["e1"]).set_index("event_id")
    assert out.loc["e1", AVAILABLE] == "2025-01-12"  # moves later than crush ("2025-01-09")


def test_label_availability_is_none_when_either_table_has_none():
    frames = _frames()
    frames["iv_crush_targets"] = _set(frames["iv_crush_targets"], AVAILABLE, None)
    out = _build(frames)[0].fetch(LABEL, ["e1"]).set_index("event_id")
    assert out.loc["e1", AVAILABLE] is None


@pytest.mark.parametrize("bad_table", ["iv_crush_targets", "computed_moves"])
def test_a_malformed_availability_date_is_not_masked_by_a_later_valid_date(bad_table):
    frames = _frames()
    frames[bad_table] = _set(frames[bad_table], AVAILABLE, "2024-12-1")  # malformed, sorts before the valid date
    good = "computed_moves" if bad_table == "iv_crush_targets" else "iv_crush_targets"
    frames[good] = _set(frames[good], AVAILABLE, "2025-01-12")
    ns, _ = _build(frames)
    with pytest.raises(OpsError) as exc:
        ns.fetch(LABEL, ["e1"])
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "source_value_invalid")


def test_label_fetch_planner_denial_propagates_unchanged(monkeypatch):
    refusal = data_fail("HOLDOUT_ACCESS_DENIED", "event is not isolated")

    def plan(repository, snapshot, candidates, contract, *, columns):
        raise refusal

    monkeypatch.setattr(tbl, "plan_event_targets", plan)
    ns = tbl.tables(_PlanRepo(), SNAP, ADMITTED, crush_snapshot=CRUSH, max_object_bytes=MAX, expected_hash=HASH, read=_Read(_frames()),
                    calendar=lambda r, s, *, max_object_bytes: _Cal())
    with pytest.raises(DataError) as exc:
        ns.fetch(LABEL, ["e1"])
    assert exc.value is refusal
    assert exc.value.code == "HOLDOUT_ACCESS_DENIED"


def test_label_rows_prints_one_scan_line_per_2000_planned_events(monkeypatch, capsys):
    n = 2001

    def plan(repository, snapshot, candidates, contract, *, columns):
        return [(f"e{i}", ("AAA", D1), None) for i in range(n)]

    monkeypatch.setattr(tbl, "plan_event_targets", plan)
    want = pd.DataFrame({"event_id": [f"e{i}" for i in range(n)], "ticker": ["AAA"] * n, "event_date": [D1] * n})
    rows = tbl.label_rows(_PlanRepo(), SNAP, "computed_moves", ["ticker", "event_date"], want, max_object_bytes=MAX)
    assert len(rows) == n
    lines = [line for line in capsys.readouterr().err.splitlines() if line.startswith("[pnl_tables] scan ")]
    assert len(lines) == 2  # the stage-start line, then one progress line
    assert re.fullmatch(r"\[pnl_tables\] scan computed_moves start events=2001 rss=\d+MiB", lines[0])
    assert re.fullmatch(r"\[pnl_tables\] scan computed_moves 2000/2001 events, \d+\.\d+/s, ETA \d+s", lines[1])


@pytest.mark.parametrize("bad", [None, 0, True, 1.0], ids=["none", "zero", "bool", "float"])
def test_tables_refuses_an_invalid_cap_before_any_read(bad):
    read, labels = _Read(_frames()), _Labels(_frames())
    with pytest.raises(OpsError) as exc:
        tbl.tables(REPO, SNAP, ADMITTED, crush_snapshot=CRUSH, max_object_bytes=bad, expected_hash=HASH, read=read,
                   label_rows=labels, calendar=lambda r, s, *, max_object_bytes: _Cal())
    assert _refusal(exc) == ("INVALID_EXPERIMENT_SPEC", "max_object_bytes_invalid")
    assert read.calls == [] and labels.calls == []


@pytest.mark.parametrize("bad", [None, 0, True, 1.0], ids=["none", "zero", "bool", "float"])
def test_label_rows_refuses_an_invalid_cap_before_any_plan_or_scan(monkeypatch, bad):
    planned = []
    monkeypatch.setattr(tbl, "plan_event_targets", lambda *a, **k: planned.append(True) or [])
    repo = _PlanRepo()
    want = pd.DataFrame({"event_id": ["e1"], "ticker": ["AAA"], "event_date": [D1]})
    with pytest.raises(OpsError) as exc:
        tbl.label_rows(repo, SNAP, "computed_moves", ["ticker", "event_date"], want, max_object_bytes=bad)
    assert _refusal(exc) == ("INVALID_EXPERIMENT_SPEC", "max_object_bytes_invalid")
    assert planned == [] and repo.scans == []


@pytest.mark.parametrize("ids", [[], ["nope"]], ids=["no_ids", "no_matching_rows"])
def test_empty_label_and_production_fetch_keep_the_expected_columns(ids):
    ns, _ = _build()
    labels, prod = ns.fetch(LABEL, ids), ns.fetch(PROD, ids)
    assert list(labels.columns) == ["event_id", "abs_move", "crush_pct_iv30", AVAILABLE]
    assert list(prod.columns) == ["event_id", "pred_abs_move", "pred_iv_crush_30", "horizon_sessions"]
    assert len(labels) == 0 and len(prod) == 0
    assert labels["abs_move"].dtype == float and labels["crush_pct_iv30"].dtype == float


# --- two snapshots: iv_crush_targets comes only from the crush snapshot --------------

def test_iv_crush_targets_read_only_from_crush_snapshot():
    ns, read = _build()
    out = ns.fetch(LABEL, ["e1", "e2"]).set_index("event_id")
    assert out.loc["e1", "crush_pct_iv30"] == -10.0
    assert [(c["table"], c["crush"]) for c in read.labels.calls] == [("computed_moves", False), ("iv_crush_targets", True)]
    assert {c["table"] for c in read.calls if c["crush"]} == {"earnings_events"}


def test_crush_table_hash_is_the_crush_manifest_hash():
    ns, _ = _build()
    assert ns.crush_table_hash == CRUSH_HASH
    assert ns.table_hash == HASH


def test_crush_snapshot_without_table_refused_before_any_read():
    read = _Read(_frames())
    with pytest.raises(OpsError) as exc:
        tbl.tables(REPO, SNAP, ADMITTED, crush_snapshot=types.SimpleNamespace(table_versions={}), max_object_bytes=MAX,
                   expected_hash=HASH, read=read, calendar=lambda r, s, *, max_object_bytes: _Cal())
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "crush_snapshot_invalid")
    assert read.calls == []


def test_all_usable_counts_zero_missing(capsys):
    _build()
    assert "[pnl_tables] crush_label_missing=0 of 2" in capsys.readouterr().err


@pytest.mark.parametrize("crush_events", [
    _events().assign(event_date=[D2, D2]),  # e1's crush row carries a different date
    _events().iloc[[1]],                    # e1 is absent from the crush earnings_events
], ids=["date_differs", "absent_from_crush_events"])
def test_crush_event_mismatch_excludes_the_event(crush_events, capsys):
    ns, _ = _build(crush_frames=_crush_of(_frames(), earnings_events=crush_events))
    out = ns.fetch(LABEL, ["e1"]).set_index("event_id")
    assert np.isnan(out.loc["e1", "crush_pct_iv30"]) and out.loc["e1", AVAILABLE] is None
    assert out.loc["e1", "abs_move"] == 3.0
    assert "[pnl_tables] crush_label_missing=1 of 2" in capsys.readouterr().err


def test_event_without_iv_crush_fragment_is_excluded_not_imputed(capsys):
    ns, _ = _build(repository=_Repo([("BBB", "2025-02-03")]))
    out = ns.fetch(LABEL, ["e1"]).set_index("event_id")
    assert np.isnan(out.loc["e1", "crush_pct_iv30"]) and out.loc["e1", AVAILABLE] is None
    assert "[pnl_tables] crush_label_missing=1 of 2" in capsys.readouterr().err


def test_excluded_crush_event_never_reads_main_snapshot_for_crush():
    ns, read = _build(repository=_Repo([]))
    out = ns.fetch(LABEL, ["e1", "e2"]).set_index("event_id")
    assert out["crush_pct_iv30"].isna().all() and out[AVAILABLE].isna().all()
    assert all(c["table"] != "iv_crush_targets" for c in read.labels.calls if not c["crush"])
    assert [c for c in read.labels.calls if c["table"] == "iv_crush_targets"] == []


CRUSH_CONTEXT_CASES = {
    "session_mismatch": _events(sessions=("BMO", "BMO")),                                              # e1 is AMC in the main snapshot
    "crush_date_conflict": _events().assign(date_conflict=[True, False]),                              # e1 flagged on the crush side
    "crush_duplicate_key": pd.concat([_events(), _events().iloc[[0]].assign(event_id="e9")], ignore_index=True),  # e1 and e9 share AAA/D1
}


@pytest.mark.parametrize("case", list(CRUSH_CONTEXT_CASES))
def test_crush_context_exclusions_leave_crush_nan_and_never_read_the_main_snapshot(case, capsys):
    ns, read = _build(crush_frames=_crush_of(_frames(), earnings_events=CRUSH_CONTEXT_CASES[case]))
    out = ns.fetch(LABEL, ["e1", "e2"]).set_index("event_id")
    assert np.isnan(out.loc["e1", "crush_pct_iv30"]) and pd.isna(out.loc["e1", AVAILABLE])
    assert out.loc["e1", "abs_move"] == 3.0
    assert "[pnl_tables] crush_label_missing=1 of 2" in capsys.readouterr().err
    assert [c["crush"] for c in read.labels.calls if c["table"] == "iv_crush_targets"] == [True]


# --- pricing pass-through ----------------------------------------------------------

def test_trading_calendar_forwards_max_object_bytes(monkeypatch):
    import engine.v2.research._scan as scan
    from engine.v2.research._pricing import trading_calendar_from_snapshot

    seen = {}

    def fake_read_table(repository, snapshot_ref, table, columns, *, max_object_bytes=None, **kwargs):
        seen.update(table=table, columns=list(columns), max_object_bytes=max_object_bytes)
        return pd.DataFrame({"date": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"])})

    monkeypatch.setattr(scan, "read_table", fake_read_table)
    trading_calendar_from_snapshot(REPO, SNAP, max_object_bytes=4096)
    assert seen == {"table": "daily_market", "columns": ["date"], "max_object_bytes": 4096}
