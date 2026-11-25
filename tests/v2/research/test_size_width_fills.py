"""EXP-189 fill-model seam: the default marks are today's, the seamed sweep reproduces the unseamed one, and a custom model is live."""
import numpy as np
import pandas as pd

from experiments import size_pnl_decisions as decisions
from experiments import size_pnl_shortvol as sv
from experiments import size_width_fills as fills
from experiments import size_width_pricing as seam
from experiments import size_width_tuning as tuning
from experiments.size_pnl_pool import build_pool
from experiments.size_pnl_straddle import EXIT_UNQUOTED
from experiments.size_width_sweep import Sweep
from tests.v2.research.test_size_pnl_shortvol import _tables, chunk_of, events_for
from tests.v2.research.test_size_width_sweep import DRAW, NAME, _two_months

PERIOD = pd.Period("2024-06", "M")
SIMS_GRID = np.arange(60.0, 141.0)


def _small_on(monkeypatch, module):
    """``_small_window`` for any module holding ``trailing_cutoff``: the real module, or a copy whose name is looked up in its own globals."""
    real = module.trailing_cutoff
    monkeypatch.setattr(module, "trailing_cutoff", lambda history, as_of, **kw: real(history, as_of, **{**kw, "min_window": 1}))


def _files(directory):
    return sorted(p.name for p in directory.iterdir() if p.is_file())


def _run(pricing, out, **kwargs):
    """The two-month chain on a ladder sweep over ``pricing``, the output month frozen and the inner tuning written under ``out``."""
    sweep = Sweep(ladder=True, pricing=pricing, **kwargs)
    pool = build_pool(*_tables(), floor=None)
    chunk, prior, cur, sessions = _two_months()
    sweep.add_month(chunk, prior, pool, output=False, sessions=sessions, draws=DRAW)
    fresh = sweep.add_month(chunk, cur, pool, output=True, sessions=sessions, draws=DRAW)
    taken = fresh.assign(model_id="M0", book="short_vol", arm_or_rule=fresh["arm"], status="taken", reason=None)
    dec = sweep.declined_rows().assign(model_id="M0", book="short_vol", arm_or_rule=lambda f: f["arm"])
    out.mkdir()
    decisions.freeze_month(out, PERIOD, [taken, dec])
    tuning.run_inner(sweep, out, "H")
    return sweep, fresh


def test_the_seamed_default_reproduces_the_unseamed_module_exactly(tmp_path, monkeypatch):
    seamed = seam.load()
    _small_on(monkeypatch, sv)
    _small_on(monkeypatch, seamed)
    plain, plain_fresh = _run(sv, tmp_path / "unseamed")
    seam_sweep, seam_fresh = _run(seamed, tmp_path / "seamed", choices=True)
    assert len(plain_fresh) > 0  # at least one taken row

    pd.testing.assert_frame_equal(plain_fresh, seam_fresh, check_exact=True)
    pd.testing.assert_frame_equal(plain.declined_rows(), seam_sweep.declined_rows(), check_exact=True)
    assert plain.taken.keys() == seam_sweep.taken.keys() and plain.scored.keys() == seam_sweep.scored.keys()
    for store in ("taken", "scored"):
        for key, frames in getattr(plain, store).items():
            other = getattr(seam_sweep, store)[key]
            assert len(frames) == len(other)
            for got, want in zip(other, frames):
                pd.testing.assert_frame_equal(got, want, check_exact=True)

    rel = f"{PERIOD}.parquet"
    assert (tmp_path / "unseamed" / "months" / rel).read_bytes() == (tmp_path / "seamed" / "months" / rel).read_bytes()
    left, right = tmp_path / "unseamed" / "receipts", tmp_path / "seamed" / "receipts"
    assert _files(left) == _files(right) and _files(left)
    for name in _files(left):
        assert (left / name).read_bytes() == (right / name).read_bytes(), name


def _sims(ids):
    return {e: (SIMS_GRID, 0.001 * (SIMS_GRID - 100.0) ** 2) for e in ids}


def test_a_custom_fill_model_moves_the_seamed_marks_and_leaves_the_real_module_alone():
    real_entry, real_exit = sv.entry, sv._exit_value

    class Custom(fills.CurrentFills):
        def entry(self, priced):
            got = super().entry(priced)
            return None if got is None else (got[0] + 0.10, got[1])

        def exit_value(self, chunk, traded, alpha, mid=False):
            return super().exit_value(chunk, traded, alpha, mid=mid) + 0.05

    base, moved = seam.load(), seam.load(fills=Custom())
    # R's exit rung is 3 (exit unquoted): score marks it at zero, the defined-risk floor, without reading the quotes
    chunk = chunk_of(("A", {}), ("B", {}), ("R", {k: (1.1, 0.9, 3) for k in range(80, 121)}))
    events = events_for("A", "B", "R")
    sims = _sims(["A", "B", "R"])
    b_scores = base.score_events(chunk, events, sims, NAME, alpha=0.5)
    m_scores = moved.score_events(chunk, events, sims, NAME, alpha=0.5)
    ok = (b_scores["status"] == "scored").to_numpy()
    assert ok.any()
    np.testing.assert_allclose((m_scores["entry_cost"] - b_scores["entry_cost"])[ok].to_numpy(), 0.10)
    assert (m_scores["exp_pnl_sim"][ok].to_numpy() != b_scores["exp_pnl_sim"][ok].to_numpy()).all()
    base_exp, base_cost = b_scores["exp_pnl_sim"][ok].to_numpy(), b_scores["entry_cost"][ok].to_numpy()
    value = base_exp * base_cost + base_cost  # the sims' expected payoff, independent of the fill model
    np.testing.assert_allclose(m_scores["exp_pnl_sim"][ok].to_numpy(), (value - (base_cost + 0.10)) / (base_cost + 0.10))

    taken = b_scores[ok].assign(status="taken")
    b_rows = base.score(chunk, events, taken, NAME, 0.5, repriced=True)
    m_rows = moved.score(chunk, events, taken, NAME, 0.5, repriced=True)
    np.testing.assert_allclose((m_rows["entry_cost"] - b_rows["entry_cost"]).to_numpy(), 0.10)
    rung3 = (b_rows["rung"] == EXIT_UNQUOTED).to_numpy()
    assert rung3.any() and (~rung3).any()
    assert (m_rows.loc[rung3, "exit_value"] == 0.0).all() and (b_rows.loc[rung3, "exit_value"] == 0.0).all()
    np.testing.assert_allclose((m_rows["exit_value"] - b_rows["exit_value"])[~rung3].to_numpy(), 0.05)
    pnl_per_share = lambda rows: (rows["exit_value"] - rows["entry_cost"]).to_numpy()
    change = pnl_per_share(m_rows) - pnl_per_share(b_rows)
    np.testing.assert_allclose(change[~rung3], -0.05)  # +0.05 exit - 0.10 entry
    np.testing.assert_allclose(change[rung3], -0.10)   # rung 3 exits at zero: only the entry moves

    assert sv.entry is real_entry and sv._exit_value is real_exit
    assert type(seam.PRICING.fills) is fills.CurrentFills and type(moved.fills) is Custom
    pd.testing.assert_frame_equal(seam.PRICING.score_events(chunk, events, sims, NAME, alpha=0.5), b_scores, check_exact=True)
