"""EXP-189 Amendment 1, slice W7d2: THE end-to-end test of an F run: driver.run for inner and outer, then the report.

The F spec is a fixture fill dict (the spec's fill_model shape); its pricing copy is real for resolve, score_events, score,
secured, entry and the exit marks. Only what cannot run on synthetic chains is stubbed: simulate and decide (the decision
of the sweep), and the residual pool. Synthetic only: tmp_path pinned tables and fixture events, never data/.

Independent oracle: the 2025 F rows (debit, proceeds, mid pair, ladder returns, secured cash) and the 2025 production
account are checked against values computed IN THIS TEST from the stored synthetic quotes and the F design formulas
(guides/fair_value_fill_design.md sections 5-6). Nothing from the experiments fair_value_*, size_width_f_* or
size_width_pricing modules computes an expected value here; only the legs (strike resolution) come from resolve_priced."""
import json

import numpy as np
import pandas as pd
import pytest

from engine import paths
from engine.fills import FillModel
from experiments import fair_value_inputs as fvi
from experiments import fair_value_run as fvr
from experiments import size_pnl_driver as base
from experiments import size_pnl_readers as readers
from experiments import size_pnl_receipts as receipts
from experiments import size_pnl_shortvol as sv
from experiments import size_width_account as account
from experiments import size_width_driver as driver
from experiments import size_width_f_fills as ff
from experiments import size_width_f_quotes as wf
from experiments import size_width_f_select as sel
from experiments import size_width_fake as fake
from experiments import size_width_pricing as seam
from experiments import size_width_report as report
from experiments.size_pnl_shortvol import Refusal
from tests.v2.research import test_size_pnl_driver as pnl
from tests.v2.research import test_size_width_f_quotes as qt
from tests.v2.research.test_size_width_run import OK

BINDING_KEYS = {"fill_model_id", "fvf_id", "reingest_manifest_sha256", "every_table_file_sha256", "quote_snapshot_id", "units",
                "rate_out_of_range", "rate_empty_row", "carry_dividend_mode"}
LABEL = "POST-HOC (Amendment 1, F fills)"
# Inner events are in 2024 (the inner window), outer events in 2025; entry and exit are one trading week apart.
PLAN = qt._plan(("E1", "AAA", "2024-02-05", "2024-02-08"), ("E2", "BBB", "2024-02-05", "2024-02-08"),
                ("E3", "AAA", "2025-02-05", "2025-02-08"), ("E4", "BBB", "2025-02-05", "2025-02-08"))
IDS = list(PLAN["event_id"])
EXPIRY = {2024: "2024-03-15", 2025: "2025-03-16"}  # 39 days after entry in both years, as the quote fixtures' expiry
TIGHT = {(right, k): (qt._mid(k) - 0.005, qt._mid(k) + 0.005) for k in qt.STRIKES for right in ("P", "C")}
CARRY = {"borrow30": 0.0, "borrow2yr": 0.0, "divAmt": 0.0, "divDate": pd.NaT, "annIdiv": 0.0, "annActDiv": 0.0, "assetType": 1.0}
SIMS_GRID = np.arange(60.0, 141.0)
ADMITTED = readers.Admitted({"events": pd.DataFrame({"event_id": IDS, "event_date": PLAN["event_date"].to_numpy()}), "excluded": {}})
LABELS = pd.DataFrame({"event_id": IDS, "pred_abs_move": 6.0, "pred_iv_crush_30": -12.0, "horizon_sessions": 1, "abs_move": 4.0,
                       "crush_pct_iv30": -10.0,
                       "available_as_of_date": (PLAN["event_date"] + pd.Timedelta(days=1)).dt.date.astype(str).to_numpy()})
FACTS = pd.DataFrame({"pre_iv30": 40.0, "mcap_usd": 2e10, "implied_move": 5.0}, index=pd.Index(IDS, name="event_id"))
# Oracle constants: the ladder alphas and their columns, the model-versus-quote margin (covers rate 5% and early exercise),
# the 2025 expiry, and each event's (entry, exit) dates.
ALPHAS = (0.0, 0.5, 1.0)
RET_COLS = ("ret_a0.00", "ret_a0.50", "ret_a1.00")
MARGIN = 0.05
EXP_2025 = pd.Timestamp(EXPIRY[2025])
DATES = {str(e): (en, ex) for e, en, ex in zip(PLAN["event_id"], PLAN["entry_date"], PLAN["exit_date"])}


@pytest.fixture(autouse=True)
def bounded(monkeypatch):
    monkeypatch.setenv("BOUNDED_RUN_NESTED", "1")


def _frames():
    """Tight listed chains for AAA and BBB on the entry and exit dates, of the expiry of each event's year."""
    return {(t, pd.Timestamp(d)): qt._listing(t, d, expiry=EXPIRY[int(d[:4])], quotes=TIGHT)
            for t in ("AAA", "BBB") for d in ("2024-02-05", "2024-02-08", "2025-02-05", "2025-02-08")}


def _tables(tmp_path, frames):
    """Pinned tables under tmp_path/root: quotes for each obs month, rates for each obs month and its lookback, a manifest."""
    base_frame = qt._base(frames)
    root = tmp_path / "root"
    vols = base_frame[fvi.VOL_KEYS].drop_duplicates().assign(smvVol=25.0)
    dates = base_frame[["ticker", "obs_date"]].drop_duplicates()
    lagged = dates.assign(obs_date=dates["obs_date"] - pd.Timedelta(days=fvi.RATE_LOOKBACK_DAYS))
    rates = pd.concat([dates, lagged]).drop_duplicates().assign(iRate5wk=5.0, iRateLt=4.0, **CARRY)
    for kind, frame in (("quotes", vols), ("rates", rates)):
        for (year, month), part in frame.groupby([frame["obs_date"].dt.year, frame["obs_date"].dt.month]):
            path = root / kind / f"year={year}" / f"month={month:02d}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            part.to_parquet(path, index=False)
    (root / "manifest.json").write_bytes(json.dumps({"version": "3", "tables": {"quotes": qt.OK, "rates": qt.OK},
                                                     "units": qt.UNITS}).encode())
    return root


def _add_rate_file(root):
    """A pinned rates file no event needs (March 2024): a changed pin set, the same tables."""
    path = root / "rates" / "year=2024" / "month=03.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    rate_row = pd.read_parquet(root / "rates" / "year=2024" / "month=02.parquet").iloc[:1]
    rate_row.assign(obs_date=pd.Timestamp("2024-03-01")).to_parquet(path, index=False)


def _fill(pins):
    """The spec's fill_model shape for the pinned tables under the fill's reingest path (root under engine.paths.ROOT)."""
    return {"id": "F", "fvf_id": fvr.fvf_id(), "carry_dividend_mode": sel.MODE,
            "rate_rules": {"rate_out_of_range": "missing", "rate_empty_row": "rate_unknown"}, "vol_scale": 0.01, "rate_scale": 0.01,
            "reingest": {"path": "root", "tool_version": "4", "manifest_sha256": pins["manifest_sha256"], "manifest_units": qt.UNITS}}


def _sources():
    def fetch(columns, ids):
        return LABELS[LABELS["event_id"].isin(list(ids))].copy()
    return pnl.sources(admitted=ADMITTED, fetch=fetch, plan_frame=PLAN, facts=lambda ids: FACTS.reindex(list(ids)))


def _chunks(frames):
    """A ladder over the frames: one NEW chunk object per quarter on every call, as the real ladder builds them."""
    def ladder_chunks(admitted, plan, load_index):
        for label, part in plan.groupby(plan["event_date"].dt.to_period("Q")):
            yield str(label), qt._chunk(frames, part)
    return ladder_chunks


def _setup(tmp_path, monkeypatch):
    """The synthetic world of every test: pinned tables under tmp_path, the root, the ladder and the sources."""
    frames = _frames()
    root = _tables(tmp_path, frames)
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    monkeypatch.setattr(base.ladder, "ladder_chunks", _chunks(frames))
    return frames, root, qt._pins(root), _sources()


def _receipt(fill):
    return {**OK(), "fill_model": fill}


def _sim(chunk_, events, pool, **kw):
    return {str(e): (SIMS_GRID, 0.001 * (SIMS_GRID - 100.0) ** 2) for e in events["event_id"]}


def _decide(scored):
    ok = (scored["status"] == "scored").to_numpy()
    return scored.assign(status=np.where(ok, "taken", "undetermined"), reason=np.where(ok, None, scored["reason"]),
                         alpha=np.where(ok, 0.5, np.nan))


@pytest.fixture
def m0(monkeypatch):
    """The M0 table (width 6.0, crush -12.0) for the four events: the only ready model."""
    table = pd.DataFrame({"event_date": PLAN["event_date"].to_numpy(), "pred_move": 6.0, "pred_crush": -12.0},
                         index=pd.Index(IDS, name="event_id"))

    def tables(bound, roster, admitted, production, **kw):
        key = next(m["model_id"] for m in roster["models"] if "M0" in m["roles"])
        return {key: {"status": "ready", "table": table, "binding": {"model": "M0"}}}
    monkeypatch.setattr(readers, "model_tables", tables)
    return table


@pytest.fixture
def stubbed(monkeypatch):
    """Simulate and decide stubbed on every pricing copy (the F copies an F spec opens, and the current-marks seam); the pool is None.
    The F copy keeps its real fills, resolve, score_events, score and secured."""
    real_load = seam.load

    def load(fills=None):
        copy = real_load(fills)
        copy.simulate, copy.decide = _sim, _decide
        return copy
    monkeypatch.setattr(seam, "load", load)
    monkeypatch.setattr(seam.PRICING, "simulate", _sim)
    monkeypatch.setattr(seam.PRICING, "decide", _decide)
    monkeypatch.setattr(driver, "build_pool", lambda table, labels: None)
    monkeypatch.setattr(fake, "build_pool", lambda table, labels: None)


def _f_run(src, out, stage, fill, inputs):
    return driver.run(src, stage, out, event_ids=IDS, prereg=lambda: _receipt(fill), f_inputs=inputs)


def _outer_chunk():
    """The 2025 ladder chunk of the outer events, and each event's index in it (the chunk the outer run prices)."""
    chunk = qt._chunk(_frames(), PLAN[PLAN["event_date"].dt.year == 2025])
    return chunk, {str(e): i for i, e in enumerate(chunk["event_id"])}


def _traded(chunk, pos, eid, structure):
    """The traded legs (quantity > 0) of ``structure`` for event ``eid``, strikes resolved by resolve_priced (no mark)."""
    _priced, legs, reason = sv.resolve_priced(chunk, pos[eid], structure, qt.PRED, FillModel(0.5))
    assert reason is None, reason
    traded = [leg for leg in legs if leg["qty"] > 0]
    assert all(chunk["right"][leg["idx"]] == 0 for leg in traded)  # every leg is a put, as the fixture's oracle assumes
    return traded


def _quote(k):
    """(mid, bid, ask) of the TIGHT quote at strike ``k``, identical on entry and exit."""
    mid = qt._mid(k)
    return mid, mid - 0.005, mid + 0.005


def _fair(k, obs):
    """F's fair value V of a put on ``obs``: the zero-rate closed form clamped to the quote. The margin rule decides it:
    model above ask + MARGIN gives the ask, below bid - MARGIN gives the bid; otherwise intrinsic above the ask gives the ask."""
    _, bid, ask = _quote(k)
    model = qt._bs_put(k, (EXP_2025 - obs).days / 365, s=qt.SPOT, v=0.25)
    if model > ask + MARGIN:
        return ask
    if model < bid - MARGIN:
        return bid
    assert k - qt.SPOT > ask, f"strike {k}: model {model} is inside the margin and intrinsic does not decide V"
    return ask


def _fills(k, obs):
    """F's buy and sell fills of a put on ``obs``: V moved toward the quote by c = min(1, 0.25 + 0.50 * min(s / 0.50, 1))."""
    mid, bid, ask = _quote(k)
    v = _fair(k, obs)
    c = min(1.0, 0.25 + 0.50 * min(((ask - bid) / mid) / 0.50, 1.0))
    return {"buy": v + c * (ask - v), "sell": v - c * (v - bid)}


def _oracle(legs, entry, exit_):
    """The oracle of one structure's traded ``legs``: F debit and proceeds, the mid pair, the ladder returns at alphas 0, 0.5
    and 1 (NaN where that alpha's debit is not positive), and the secured cash of one contract (100 times the short strikes)."""
    o = {"entry_date": entry, "exit_date": exit_, "debit_f": 0.0, "proceeds_f": 0.0, "debit_mid": 0.0, "exit_mid": 0.0,
         "secured": 0.0}
    debit, closes = dict.fromkeys(ALPHAS, 0.0), dict.fromkeys(ALPHAS, 0.0)
    for leg in legs:
        q, k, buy = float(leg["qty"]), float(leg["strike"]), leg["side"] == "buy"
        sign = 1.0 if buy else -1.0
        mid, bid, ask = _quote(k)
        fe, fx = _fills(k, entry), _fills(k, exit_)
        o["debit_f"] += sign * q * (fe["buy"] if buy else fe["sell"])
        o["proceeds_f"] += q * (fx["sell"] if buy else -fx["buy"])
        o["debit_mid"] += sign * q * mid
        o["exit_mid"] += sign * q * mid
        if not buy:
            o["secured"] += 100.0 * q * k
        for a in ALPHAS:  # alpha 0: buys at the ask, sells at the bid, longs sold at the bid, shorts bought at the ask
            spread = a * (ask - bid)
            debit[a] += sign * q * ((ask - spread) if buy else (bid + spread))
            closes[a] += q * ((bid + spread) if buy else -(ask - spread))
    o["ret"] = {a: (closes[a] - debit[a]) / debit[a] if debit[a] > 0 else float("nan") for a in ALPHAS}
    return o


def _oracle_rows(taken, chunk, pos):
    """{(event_id, structure): oracle} for every frozen taken row."""
    out = {}
    for rec in taken.to_dict("records"):
        eid = str(rec["event_id"])
        entry, exit_ = DATES[eid]
        out[(eid, rec["structure"])] = _oracle(_traded(chunk, pos, eid, rec["structure"]), entry, exit_)
    return out


def _account(oracle, *, mid):
    """The account frame of the oracle's rows: the F pair, or the mid pair when ``mid``; secured as the production account
    defines it."""
    return pd.DataFrame([{"entry_date": o["entry_date"], "exit_date": o["exit_date"],
                          "entry_cost": o["debit_mid" if mid else "debit_f"], "exit_value": o["exit_mid" if mid else "proceeds_f"],
                          "secured_per_contract": o["secured"]} for o in oracle.values()])


def test_an_f_run_writes_the_binding_and_the_post_hoc_report(tmp_path, monkeypatch, m0, stubbed):
    frames, root, pins, src = _setup(tmp_path, monkeypatch)
    out = tmp_path / "out"
    inputs = sel.FInputs(pins, qt._reader(qt._base(frames)), "snap-1")
    _f_run(src, out, "inner", _fill(pins), inputs)
    _f_run(src, out, "outer", _fill(pins), inputs)
    binding = {"fill_model_id": "F", "fvf_id": _fill(pins)["fvf_id"], "reingest_manifest_sha256": pins["manifest_sha256"],
               "every_table_file_sha256": dict(sorted(pins["files"].items())), "quote_snapshot_id": "snap-1",
               "units": {"vol_scale": 0.01, "rate_scale": 0.01, "manifest_units": qt.UNITS},
               "rate_out_of_range": "missing", "rate_empty_row": "rate_unknown", "carry_dividend_mode": sel.MODE}
    assert set(binding) == BINDING_KEYS
    for stage in ("inner", "outer"):
        receipt = receipts.load_receipt(out / "receipts" / f"inputs_{stage}.json")
        assert receipt["fill_model"] == binding
    summary = report.assemble(out, expected=src.expected)
    assert summary["recommendation"]
    assert LABEL in (out / "REPORT.md").read_text(encoding="utf-8")
    assert json.loads((out / "proposed_rule.json").read_text(encoding="utf-8"))["label"] == LABEL


def test_the_taken_rows_pair_f_debits_with_f_proceeds_and_the_account_is_the_f_account(tmp_path, monkeypatch, m0, stubbed):
    frames, root, pins, src = _setup(tmp_path, monkeypatch)
    out = tmp_path / "out"
    _f_run(src, out, "inner", _fill(pins), sel.FInputs(pins, qt._reader(qt._base(frames)), "snap-1"))
    _f_run(src, out, "outer", _fill(pins), sel.FInputs(pins, qt._reader(qt._base(frames)), "snap-1"))
    frozen = pd.read_parquet(out / "months" / "2025-02.parquet")
    taken = frozen[(frozen["status"] == "taken") & (frozen["arm_or_rule"] == "1.0")]
    assert len(taken) >= 1
    assert (taken["entry_cost"] != taken["entry_cost_mid"]).all()
    assert (taken["exit_value"] != taken["exit_value_mid"]).all()
    chunk, pos = _outer_chunk()
    oracle = _oracle_rows(taken, chunk, pos)
    for rec in taken.to_dict("records"):  # every taken 1.0 row, every structure present
        o = oracle[(str(rec["event_id"]), rec["structure"])]
        assert rec["entry_cost"] == pytest.approx(o["debit_f"], rel=1e-9, abs=1e-12)
        assert rec["exit_value"] == pytest.approx(o["proceeds_f"], rel=1e-9, abs=1e-12)
        assert rec["entry_cost_mid"] == pytest.approx(o["debit_mid"], rel=1e-9, abs=1e-12)
        assert rec["exit_value_mid"] == pytest.approx(o["exit_mid"], rel=1e-9, abs=1e-12)
        assert rec["secured_per_contract"] == pytest.approx(o["secured"], rel=1e-9, abs=1e-12)
    primary = account.simulate_account(_account(oracle, mid=False), "primary")
    mid = account.simulate_account(_account(oracle, mid=True), "primary")
    outer = receipts.load_receipt(out / "receipts" / "outer_summary.json")
    assert outer["results"]["production"]["2025"]["account"]["total_return"] == pytest.approx(primary["total_return"], rel=1e-9, abs=1e-12)
    assert primary["total_return"] != 0.0
    assert primary["total_return"] != pytest.approx(mid["total_return"])


def test_the_alpha_ladder_is_not_flat_and_matches_the_oracle_and_the_current_marks(tmp_path, monkeypatch, m0, stubbed):
    frames, root, pins, src = _setup(tmp_path, monkeypatch)
    out, current = tmp_path / "out", tmp_path / "current"
    _f_run(src, out, "inner", _fill(pins), sel.FInputs(pins, qt._reader(qt._base(frames)), "snap-1"))
    _f_run(src, out, "outer", _fill(pins), sel.FInputs(pins, qt._reader(qt._base(frames)), "snap-1"))
    driver.run(src, "inner", current, event_ids=IDS, prereg=OK)  # current marks: the seam's own CurrentFills copy
    driver.run(src, "outer", current, event_ids=IDS, prereg=OK)
    f_rows = pd.read_parquet(out / "months" / "2025-02.parquet")
    f_rows = f_rows[(f_rows["status"] == "taken") & (f_rows["arm_or_rule"] == "1.0")]
    chunk, pos = _outer_chunk()
    oracle = _oracle_rows(f_rows, chunk, pos)
    for rec in f_rows.to_dict("records"):  # every taken 1.0 row: NaN exactly where the oracle's debit is not positive
        o = oracle[(str(rec["event_id"]), rec["structure"])]
        for a, col in zip(ALPHAS, RET_COLS):
            if np.isnan(o["ret"][a]):
                assert np.isnan(rec[col]), col
            else:
                assert rec[col] == pytest.approx(o["ret"][a], rel=1e-9, abs=1e-12), col
    first = f_rows.iloc[0]
    o = oracle[(str(first["event_id"]), first["structure"])]
    if o["ret"][0.0] < o["ret"][0.5] < o["ret"][1.0]:  # asserted only where the oracle itself shows the strict ordering
        assert first["ret_a0.00"] < first["ret_a0.50"] < first["ret_a1.00"]
    cur_rows = pd.read_parquet(current / "months" / "2025-02.parquet")
    cur_rows = cur_rows[(cur_rows["status"] == "taken") & (cur_rows["arm_or_rule"] == "1.0")]
    merged = f_rows.merge(cur_rows, on=["event_id", "structure"], suffixes=("_f", "_c"))
    assert len(merged) == len(f_rows) > 0
    for col in RET_COLS:
        np.testing.assert_allclose(merged[f"{col}_f"].to_numpy(float), merged[f"{col}_c"].to_numpy(float), rtol=1e-12)


def test_the_quote_reader_is_called_once_per_side_for_one_quarter_chunk(tmp_path, monkeypatch, m0, stubbed):
    frames, root, pins, src = _setup(tmp_path, monkeypatch)
    calls = []
    _f_run(src, tmp_path / "out", "inner", _fill(pins), sel.FInputs(pins, qt._reader(qt._base(frames), calls), "snap-1"))
    # one quarter chunk (2024Q1, both inner events) and two sides; the anchor and the fallback pass rebuild the chunk objects
    assert len(calls) == 2


def test_two_draws_over_two_quarter_chunks_read_each_chunk_side_at_most_once(tmp_path, monkeypatch, m0, stubbed):
    frames, root, pins, src = _setup(tmp_path, monkeypatch)
    calls = []
    quotes = wf.FQuotes(PLAN, qt._reader(qt._base(frames), calls), tables_root=root, pins=pins, snapshot_id="snap-1",
                        expected_snapshot_id="snap-1", vol_scale=0.01, rate_scale=0.01)
    pricing = seam.load(fills=ff.FFills(quotes))
    table = m0
    facts = base._load_facts(src, set(IDS))
    exits, sessions = base._exits(src), base._sessions(src)
    labels = readers.read_labels(src.fetch, src.admitted, start=base.INNER[0], end=base.OUTER[0], event_ids=None)
    printed = base._printed(labels, base.OUTER[0])
    model = {"L": 1.0, "b": 0.8, "s": 0.2, "mean_log_y": 1.0, "n": 300}

    def months():  # fresh chunk objects for every draw, as the driver's months are rebuilt
        return list(driver._months(src, PLAN, table, facts, exits, base.INNER_EMIT_FROM))
    fake.run_draw(months(), table, printed, model, 0, sessions, draws=1, pricing=pricing)
    after_first = len(calls)
    assert after_first >= 1
    fake.run_draw(months(), table, printed, model, 1, sessions, draws=1, pricing=pricing)
    assert len(calls) == after_first <= 4  # two quarter chunks, two sides each


def test_rerunning_inner_with_changed_pins_refuses_inputs_receipt_changed(tmp_path, monkeypatch, m0, stubbed):
    frames, root, pins, src = _setup(tmp_path, monkeypatch)
    out = tmp_path / "out"
    _f_run(src, out, "inner", _fill(pins), sel.FInputs(pins, qt._reader(qt._base(frames)), "snap-1"))
    _add_rate_file(root)
    changed = qt._pins(root)
    with pytest.raises(Refusal) as caught:
        _f_run(src, out, "inner", _fill(changed), sel.FInputs(changed, qt._reader(qt._base(frames)), "snap-1"))
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.details["reason"] == "inputs_receipt_changed"


def test_outer_with_an_f_binding_other_than_the_inners_refuses_tuning_receipt_changed(tmp_path, monkeypatch, m0, stubbed):
    frames, root, pins, src = _setup(tmp_path, monkeypatch)
    out = tmp_path / "out"
    _f_run(src, out, "inner", _fill(pins), sel.FInputs(pins, qt._reader(qt._base(frames)), "snap-1"))
    _add_rate_file(root)
    changed = qt._pins(root)
    with pytest.raises(Refusal) as caught:
        _f_run(src, out, "outer", _fill(changed), sel.FInputs(changed, qt._reader(qt._base(frames)), "snap-1"))
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.details["reason"] == "tuning_receipt_changed"
    assert not (out / "receipts" / "outer_summary.json").exists()
