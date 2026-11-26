"""EXP-189 W7a1b: the option_chains reader and --f-quotes factory (experiments/size_width_f_chains.py), on a real committed
snapshot with a 2023 and a 2024 partition. Synthetic; run.py is loaded by path."""
import importlib.util
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from engine.v2.data.errors import DataError
from engine.v2.data.repository import Repository
from engine.v2.ops.errors import OpsError
from engine.v2.research import _scan
from experiments import fair_value_inputs as fvi
from experiments import size_pnl_sources as src
from experiments import size_width_f_chains as chains
from experiments import size_width_f_quotes as wf
from tests.data_scan_support import catalog_and_store, commit_tables, publish_and_inspect
from tests.v2.ops.chain_support import CHAINS, CHAINS_NAME, CHAINS_REF, chain_row
from tests.v2.research import test_size_width_f_quotes as fq_t

RUN = Path(__file__).resolve().parents[3] / "experiments" / "EXP-189_short_vol_width_tuning" / "run.py"
SPEC = "experiments.size_width_f_chains:build_f_quotes"
KEY_COLS = ["ticker", "obs_date", "expiry", "right", "strike"]


def _key(ticker, obs, expiry, right, strike):
    return (ticker, pd.Timestamp(obs), pd.Timestamp(expiry), right, strike)


# (ticker, obs_date, expiry, right, strike) rows per year. Requested: BBB 2023-03-01 and AAA 2024-01-05, two rows each.
PARTITIONS = {
    "2023": [("BBB", "2023-03-01", "2023-03-17", "C", 100.0), ("BBB", "2023-03-01", "2023-03-17", "P", 105.0),
             ("BBB", "2023-03-01", "2023-04-21", "C", 100.0),  # same ticker and date, other expiry
             ("AAA", "2023-03-01", "2023-03-17", "C", 100.0),  # AAA on BBB's date, with BBB's expiry (cross)
             ("BBB", "2023-03-02", "2023-03-17", "C", 100.0)],  # unrequested date
    "2024": [("AAA", "2024-01-05", "2024-01-19", "C", 100.0), ("AAA", "2024-01-05", "2024-01-19", "P", 105.0),
             ("AAA", "2024-01-05", "2024-02-16", "C", 100.0),  # same ticker and date, other expiry
             ("BBB", "2024-01-05", "2024-01-19", "C", 100.0),  # BBB on AAA's date, with AAA's expiry (cross)
             ("CCC", "2024-01-05", "2024-01-19", "C", 100.0)],  # unrequested ticker
}
# Hand-listed: the 2023 partition first, then 2024. Within a partition the order is the primary-key order.
WANTED = [_key("BBB", "2023-03-01", "2023-03-17", "C", 100.0), _key("BBB", "2023-03-01", "2023-03-17", "P", 105.0),
          _key("AAA", "2024-01-05", "2024-01-19", "C", 100.0), _key("AAA", "2024-01-05", "2024-01-19", "P", 105.0)]
EXPIRIES = pd.DataFrame({"ticker": ["AAA", "BBB"], "obs_date": pd.to_datetime(["2024-01-05", "2023-03-01"]),
                         "expiry": pd.to_datetime(["2024-01-19", "2023-03-17"])})  # AAA/2024 listed first
# Distinct hand-typed (bid, ask) per WANTED row, so a swapped or altered column is caught.
QUOTES = {_key("BBB", "2023-03-01", "2023-03-17", "C", 100.0): (1.1, 1.3),
          _key("BBB", "2023-03-01", "2023-03-17", "P", 105.0): (2.1, 2.3),
          _key("AAA", "2024-01-05", "2024-01-19", "C", 100.0): (3.1, 3.3),
          _key("AAA", "2024-01-05", "2024-01-19", "P", 105.0): (4.1, 4.3)}


def _contract_row(ticker, obs, expiry, right, strike, **quote):
    return chain_row(ticker, pd.Timestamp(obs).to_pydatetime(), pd.Timestamp(expiry).to_pydatetime(), strike, right=right,
                     **quote)


def _quoted_row(row):
    """A contract row carrying its QUOTES pair when it has one; decoy rows keep chain_row's defaults."""
    key = _key(*row)
    return _contract_row(*row, **dict(zip(("bid", "ask"), QUOTES[key]))) if key in QUOTES else _contract_row(*row)


def _commit(tmp_path, partitions):
    """Commit one fragment per year (`partitions` maps year -> contract rows) into a catalog and store under tmp_path."""
    conn, clock, store = catalog_and_store(tmp_path)
    records = []
    for year, rows in sorted(partitions.items()):
        ordered = sorted(rows, key=lambda row: tuple(row[f] for f in CHAINS.primary_key))
        records.append(publish_and_inspect(store, CHAINS, CHAINS_REF, ordered, year))
    snapshot = commit_tables(conn, clock, {CHAINS_NAME: records}, {CHAINS_NAME: CHAINS})
    conn.commit()
    return snapshot.snapshot_id


@pytest.fixture
def committed(tmp_path):
    return _commit(tmp_path, {y: [_quoted_row(r) for r in rows] for y, rows in PARTITIONS.items()})


def _run():
    spec = importlib.util.spec_from_file_location("exp189_run_f_chains", RUN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config(tmp_path, snapshot_id):
    path = tmp_path / "sources.json"
    doc = {**dict.fromkeys(src._KEYS, "x"), "catalog": str(tmp_path / "catalog.sqlite"),
           "store_root": str(tmp_path / "store"), "snapshot_id": snapshot_id}
    path.write_text(json.dumps(doc), encoding="utf-8")
    return str(path)


def _factory(tmp_path, snapshot_id, *extra):
    """The run.py --f-quotes factory, parsed through run.py's own parser with a real sources config."""
    module = _run()
    args = module._parser().parse_args(["--stage", "inner", "--out-dir", str(tmp_path), "--expected", "x", "--sources", "x:y",
                                        "--no-ledger", "--sources-config", _config(tmp_path, snapshot_id), *extra])
    return module._factory(SPEC, args, "--f-quotes", "f_quotes_invalid")


def test_factory_reads_the_pinned_snapshot_in_manifest_order(tmp_path, capsys, committed):
    read, sid = _factory(tmp_path, committed)
    assert callable(read) and sid == committed
    capsys.readouterr()
    frame = read(EXPIRIES)
    assert capsys.readouterr().out == ""
    assert list(frame.columns) == fvi.QUOTE_COLUMNS
    got = [(t, pd.Timestamp(o), pd.Timestamp(e), r, k) for t, o, e, r, k in frame[KEY_COLS].itertuples(index=False, name=None)]
    assert got == WANTED  # manifest order: 2023 partition before 2024
    pairs = list(frame[["bid", "ask"]].itertuples(index=False, name=None))
    assert pairs == [QUOTES[key] for key in WANTED]
    assert (frame["iv"] == 30.0).all() and (frame["spot"] == 100.0).all()
    assert not frame["quote_repaired"].any() and (frame["chain_kind"] == "entry").all()


@pytest.mark.parametrize("extra,cap", [(["--f-max-object-bytes", "123456789"], 123456789), ([], 1 << 28)],
                         ids=["explicit", "default"])
def test_scans_are_one_per_year_with_the_pushed_down_filter_and_the_cap(tmp_path, monkeypatch, committed, extra, cap):
    scans = []
    real = Repository.scan

    def scan(self, query, **kwargs):
        scans.append((query.key_filter, kwargs.get("max_object_bytes")))
        return real(self, query, **kwargs)

    monkeypatch.setattr(Repository, "scan", scan)
    read, _ = _factory(tmp_path, committed, *extra)
    read(EXPIRIES)
    assert len(scans) == 2
    for (key_filter, scan_cap), year in zip(scans, (2023, 2024)):
        preds = {p.column: p for p in key_filter}
        assert preds["ticker"].operator == "in" and set(preds["ticker"].values) == {"AAA", "BBB"}
        assert preds["obs_date"].operator == "in" and set(preds["obs_date"].values) == {"2023-03-01", "2024-01-05"}
        assert preds["year"].operator == "eq" and preds["year"].values == (year,)
        assert scan_cap == cap


def test_batch_filter_keeps_only_the_requested_triples(tmp_path, monkeypatch, committed):
    calls = []

    def fake(repository, snapshot, table, columns, **kwargs):
        calls.append({"table": table, "columns": columns, **kwargs})
        return pd.DataFrame(columns=list(columns))

    monkeypatch.setattr(_scan, "read_table", fake)
    read, _ = _factory(tmp_path, committed)
    assert read(EXPIRIES).empty
    call = calls[0]
    assert call["table"] == "option_chains" and call["columns"] == fvi.QUOTE_COLUMNS
    assert call["partition_keys"] == ["2023", "2024"] and call["max_object_bytes"] == 1 << 28
    batch = pd.DataFrame([
        {"ticker": "AAA", "obs_date": pd.Timestamp("2024-01-05"), "expiry": pd.Timestamp("2024-01-19"), "strike": 100.0},
        {"ticker": "BBB", "obs_date": pd.Timestamp("2023-03-01"), "expiry": pd.Timestamp("2023-03-17"), "strike": 105.0},
        {"ticker": "AAA", "obs_date": pd.Timestamp("2024-01-05"), "expiry": pd.Timestamp("2024-02-16"), "strike": 100.0},  # wrong expiry
        {"ticker": "AAA", "obs_date": pd.Timestamp("2023-03-01"), "expiry": pd.Timestamp("2023-03-17"), "strike": 100.0},  # cross
        {"ticker": "BBB", "obs_date": pd.Timestamp("2024-01-05"), "expiry": pd.Timestamp("2024-01-19"), "strike": 100.0},  # cross
    ])
    assert call["batch_filter"](batch).index.tolist() == [0, 1]
    empty = batch.iloc[:0]
    assert call["batch_filter"](empty) is empty


def test_a_one_byte_cap_refuses_the_scan(tmp_path, committed):
    read, _ = _factory(tmp_path, committed, "--f-max-object-bytes", "1")
    with pytest.raises(DataError) as exc:
        read(EXPIRIES)
    assert exc.value.code == "RESULT_LIMIT_EXCEEDED"


@pytest.mark.parametrize("cap", [0, -1])
def test_a_non_positive_cap_is_refused_before_anything_opens(cap):
    with pytest.raises(DataError) as exc:
        chains.build_f_quotes(SimpleNamespace(f_max_object_bytes=cap, sources_config=None))
    assert exc.value.code == "QUERY_NOT_BOUNDED"
    with pytest.raises(DataError) as exc:
        chains.option_chain_reader(None, None, max_object_bytes=cap)
    assert exc.value.code == "QUERY_NOT_BOUNDED"


def test_the_reader_requires_its_cap_keyword():
    with pytest.raises(TypeError):
        chains.option_chain_reader(None, None)


def test_a_contract_without_the_f_columns_refuses_before_any_scan(monkeypatch):
    columns = [SimpleNamespace(name=c.name) for c in CHAINS.columns if c.name != "chain_kind"]
    repository = SimpleNamespace(table_contract=lambda snapshot, table: SimpleNamespace(columns=columns))
    scans = []
    monkeypatch.setattr(_scan, "read_table", lambda *args, **kwargs: scans.append(args))
    with pytest.raises(fvi.FairValueRefusal) as exc:
        chains.option_chain_reader(repository, None, max_object_bytes=1 << 28)
    assert exc.value.details == {"reason": "required_column", "columns": ["chain_kind"]}
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert scans == []


def test_no_sources_config_is_refused():
    with pytest.raises(OpsError) as exc:
        chains.build_f_quotes(SimpleNamespace(f_max_object_bytes=None, sources_config=None))
    assert exc.value.problem.details["reason"] == "sources_config_invalid"
    assert exc.value.problem.code == "INVALID_EXPERIMENT_SPEC"


def test_an_unknown_snapshot_is_refused_and_its_connection_closed(tmp_path, monkeypatch, committed):
    opened, real_open = [], src._open

    def capture(cfg):
        conn, store = real_open(cfg)
        opened.append(conn)
        return conn, store

    monkeypatch.setattr(src, "_open", capture)
    with pytest.raises(OpsError) as exc:
        _factory(tmp_path, "snap-missing")
    assert exc.value.problem.details["reason"] == "snapshot_missing"
    assert exc.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"
    with pytest.raises(sqlite3.ProgrammingError):
        opened[0].execute("SELECT 1")


def test_no_matching_ticker_gives_an_empty_frame_with_the_quote_columns(tmp_path, committed):
    read, _ = _factory(tmp_path, committed)
    assert len(read(EXPIRIES)) == len(WANTED)
    frame = read(EXPIRIES.assign(ticker="ZZZ"))
    assert frame.empty and list(frame.columns) == fvi.QUOTE_COLUMNS


def _quote_row(quote):
    """A `_frames()` quote as a contract row: chain_row's keys, the frame's QUOTE_COLUMNS values, and the year and dte."""
    obs, expiry = (pd.Timestamp(quote[c]).to_pydatetime() for c in ("obs_date", "expiry"))
    row = chain_row(quote["ticker"], obs, expiry, quote["strike"], right=quote["right"], bid=quote["bid"], ask=quote["ask"])
    row.update({c: quote[c] for c in fvi.QUOTE_COLUMNS})
    row.update(obs_date=obs, expiry=expiry, year=obs.year, dte=quote["dte"])
    return row


def test_the_factory_reader_feeds_fquotes_like_the_in_memory_reader(tmp_path):
    frames = fq_t._frames()
    base = fq_t._base(frames)
    sid = _commit(tmp_path, {"2024": [_quote_row(q) for q in base.to_dict("records")]})
    read, got_sid = _factory(tmp_path, sid)
    assert got_sid == sid
    root = fq_t._write(tmp_path / "f", base)  # the same pinned tables for both FQuotes
    real = wf.FQuotes(fq_t.DEFAULT_PLAN, read, tables_root=root, pins=fq_t._pins(root), snapshot_id=sid,
                      expected_snapshot_id=sid, vol_scale=0.01, rate_scale=0.01)
    ref, chunk = fq_t._setup(tmp_path / "ref", frames, fq_t.DEFAULT_PLAN)
    for exit_ in (False, True):
        got = real.table(chunk, exit=exit_).sort_index()
        want = ref.table(chunk, exit=exit_).sort_index()
        assert len(got) > 0
        pd.testing.assert_index_equal(got.index, want.index)
        for col in ("V", "fill_buy", "status"):
            pd.testing.assert_series_equal(got[col], want[col])
