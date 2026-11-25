"""EXP-189 flex: the curves, the factor target, the 2025 stage's pre-stated rules (with stubs), and the fallback's pricing seam.
Synthetic fixtures only."""
import numpy as np
import pandas as pd
import pytest

from engine.structures import BUY, SELL
from experiments import size_pnl_shortvol as sv
from experiments import size_width_band
from experiments import size_width_driver as driver
from experiments import size_width_flex as flex
from experiments.size_pnl_pool import build_pool
from experiments.size_pnl_shortvol import NAMES, Refusal
from experiments.size_width_band import PHIS, write_bands
from experiments.size_width_fallback import FallbackSweep
from experiments.size_width_sweep import E_ARM, Sweep
from tests.v2.research.test_size_pnl_shortvol import _tables, chunk_of, events_for

CONFLICT = "EXPERIMENT_IDENTITY_CONFLICT"
NAME = "CND-PS"
SESSIONS = {}
POOL = build_pool(*_tables(), floor=None)


def _factors(k=1.0):
    return {n: k for n in NAMES}


class Fake:
    """A stand-in FallbackSweep: only the phi and the factors that flex reads."""
    def __init__(self, phi, factors):
        self.phi, self.factors = phi, factors


def _inner_stage(choice_phi, flags=()):
    """A flex_inner summary for production: 2023-24 accounts that the fallback beats (0.5 against 0.4)."""
    acc = {"2023H2": {"fallback": 0.5, "base": 0.4}, "2024": {"fallback": 0.5, "base": 0.4}}
    cell = {"recovery": {}, "accounts": acc, "t9": []}
    return {"production": {"factors": _factors(), "choice": {"phi": choice_phi, "flags": list(flags)},
                           "phis": {str(p): cell for p in PHIS}}}


@pytest.fixture
def stubs(monkeypatch):
    """_run, accounts and t9 are stubs. Every phi but 0.95 gives 2025 accounts that beat base; 0.95 looks best of all."""
    seen = []

    def run(factor_sets, months, total, out, inputs_hash, sessions, pricing, ladder, tag):
        seen.append(sorted(factor_sets))
        return {label: {phi: Fake(phi, factors) for phi in PHIS} for label, factors in factor_sets.items()}

    monkeypatch.setattr(flex, "_run", run)
    monkeypatch.setattr(flex, "accounts", lambda fb, base, windows: {w: {"fallback": 0.9 if fb.phi == 0.95 else 0.5, "base": 0.4}
                                                                     for w in windows})
    monkeypatch.setattr(flex, "t9", lambda fb, base, phi, window: [])
    return seen


def _stats(sweep, by, window):
    return {"x": 0.3}  # breakeven alpha equal for fallback and base: alpha_ok


def _outer(flex_inner, rec, frozen=None, months=()):
    frozen = frozen or {"production": _factors()}
    return flex.outer(frozen, flex_inner, rec, months, 0, object(), None, "H", SESSIONS, None, _stats)


# ---- curves and the factor target -------------------------------------------

def test_curves_marks_the_anchor_not_scored_from_min_trades():
    tuned_b = {n: {"scores": {"1.0": {"ret": 0.1, "se": 0.01, "trades": 300, "not_scored": False}}} for n in NAMES}
    anchor = {n: {"0.5": {"ret": 0.2, "se": 0.0, "trades": 299}, "1.0": {"ret": 0.3, "se": 0.0, "trades": 300}} for n in NAMES}
    got = flex.curves(tuned_b, None, anchor, 300)
    assert set(got[NAME]) == {"B", "anchor"}
    assert got[NAME]["anchor"]["0.5"]["not_scored"] is True and got[NAME]["anchor"]["1.0"]["not_scored"] is False
    tuned_c = {n: {"scores": {}} for n in NAMES}
    assert "C" in flex.curves(tuned_b, tuned_c, anchor, 300)[NAME]


def test_target_maps_the_recommendation_to_its_factors():
    assert flex.target(None) == "production"
    assert flex.target("B") == "B"
    assert flex.target("E") is None


# ---- the 2025 stage's pre-stated rules ---------------------------------------

def test_adoption_uses_the_inner_phi_even_when_another_phi_looks_better_in_2025(stubs):
    got = _outer(_inner_stage(0.8), None)
    assert got["sets"]["production"]["choice"]["phi"] == 0.8
    assert got["adoption"]["phi"] == 0.8 and got["adoption"]["adopt"] is True
    assert stubs == [["production"]]


def test_recommending_arm_e_runs_only_production_and_adopts_nothing(stubs):
    frozen = {"production": _factors(), "E": {n: E_ARM for n in NAMES}}
    got = _outer(_inner_stage(0.8), "E", frozen=frozen)
    assert got["adoption"] == {"adopt": False, "phi": None, "reason": "recommended_arm_e_has_no_fixed_factors"}
    assert got["factors_label"] is None
    assert stubs == [["production"]]


def _admission():
    """A valid (inner, bands, frozen, outer body) for one fixed-factor label, production: admit passes on it."""
    bands = {n: {"primary": "B", "curves": {}} for n in NAMES}
    accounts = {w: {"fallback": 0.5, "base": 0.4} for w in ("2023H2", "2024", "2025")}
    sets = {"production": {"factors": _factors(), "choice": {"phi": 0.8, "flags": []},
                           "phis": {str(p): {"t9": [], "alpha": {"fallback": 0.84, "base": 0.80}, "accounts": accounts} for p in PHIS}}}
    t8 = {"quantiles": [], "buckets": [], "declines": [], "deviation_unavailable": 0}
    return {"inner": {"bands": bands, "t8": dict(t8), "flex": _inner_stage(0.8)}, "bands": bands, "frozen": {"production": _factors()},
            "outer": {"t8": dict(t8), "flex": {"recommendation": None, "factors_label": "production", "adoption": {"adopt": True}, "sets": sets}}}


MUTATIONS = {
    "bands differ": lambda s: s.update(bands={n: {"primary": "C", "curves": {}} for n in NAMES}),
    "flex empty": lambda s: s["inner"].update(flex=[]),
    "missing factors": lambda s: s["inner"]["flex"]["production"].pop("factors"),
    "changed factor": lambda s: s["inner"]["flex"]["production"].update(factors=_factors(0.9)),
    "missing phi cell": lambda s: s["inner"]["flex"]["production"]["phis"].pop("0.9"),
    "missing inner account window": lambda s: s["inner"]["flex"]["production"]["phis"]["0.8"]["accounts"].pop("2024"),
    "non-numeric account value": lambda s: s["inner"]["flex"]["production"]["phis"]["0.8"]["accounts"]["2024"].update(fallback="x"),
    "missing t8": lambda s: s["inner"].pop("t8"),
    "outer flex missing": lambda s: s["outer"].pop("flex"),
    "outer set factors differ": lambda s: s["outer"]["flex"]["sets"]["production"].update(factors=_factors(0.9)),
    "outer choice differs": lambda s: s["outer"]["flex"]["sets"]["production"].update(choice={"phi": 0.9, "flags": []}),
    "outer missing the 2025 account": lambda s: s["outer"]["flex"]["sets"]["production"]["phis"]["0.8"]["accounts"].pop("2025"),
}


def test_mismatched_inner_factors_refuse_at_admission():
    s = _admission()
    s["inner"]["flex"]["production"]["factors"] = _factors(0.9)
    with pytest.raises(Refusal) as caught:
        flex.admit(s["inner"], s["bands"], s["frozen"], s["outer"])
    assert caught.value.code == CONFLICT and caught.value.details["reason"] == "tuning_receipt_changed"


def test_a_valid_admission_passes():
    s = _admission()
    assert flex.admit(s["inner"], s["bands"], s["frozen"]) is None
    assert flex.admit(s["inner"], s["bands"], s["frozen"], s["outer"]) is None


@pytest.mark.parametrize("name", list(MUTATIONS))
def test_admit_refuses_each_mutation_of_a_valid_admission(name):
    s = _admission()
    MUTATIONS[name](s)
    with pytest.raises(Refusal) as caught:
        flex.admit(s["inner"], s["bands"], s["frozen"], s["outer"])
    assert caught.value.code == CONFLICT and caught.value.details["reason"] == "tuning_receipt_changed"


def test_insufficient_recovery_adopts_nothing_with_that_reason(stubs):
    got = _outer(_inner_stage(None, ["insufficient_recovery"]), None)
    assert got["adoption"] == {"adopt": False, "phi": None, "beats": False, "alpha_ok": False, "reason": "insufficient_recovery"}
    assert set(got["sets"]["production"]["phis"]) == {"0.8", "0.9", "0.95"}


# ---- the fallback routes its pricing through the seam ------------------------

class Seam:
    """A pricing stub: the shortvol functions, with score_events and resolve recorded as the seam calls."""
    simulate = staticmethod(sv.simulate)
    decide = staticmethod(sv.decide)
    score = staticmethod(sv.score)
    secured = staticmethod(sv.secured)

    def __init__(self):
        self.calls = []

    def score_events(self, *args, **kw):
        self.calls.append("score_events")
        return sv.score_events(*args, **kw)

    def resolve(self, *args):
        self.calls.append("resolve")
        return sv.resolve(*args)


def test_fallback_scores_and_recovers_through_the_pricing_it_is_given(tmp_path):
    none = {"lo": None, "hi": None, "peak_k": None, "flags": ["no_positive_peak"]}
    wide = {"lo": 0.5, "hi": 3.0, "peak_k": 1.0, "flags": []}
    body = {n: {"primary": "B", "curves": {"B": {str(p): none for p in PHIS}}} for n in NAMES}
    body[NAME] = {"primary": "B", "curves": {"B": {str(p): wide for p in PHIS}}}
    write_bands(tmp_path, body, "h1")
    seam = Seam()
    fb = FallbackSweep(_factors(), tmp_path, "h1", 0.9, pricing=seam)
    chunk = chunk_of(("A", {}), strikes=range(60, 141, 10))  # coarse: CND-PS at k 1 is ladder_too_coarse, recovered at rho 2.55
    fb.add_month(chunk, events_for("A"), POOL, output=False, sessions=SESSIONS, draws=200)
    assert "score_events" in seam.calls and "resolve" in seam.calls
    assert [t["rho"] for t in fb.trials] == [pytest.approx(2.55)]


# ---- scored values by hand, through the fallback stages -----------------------

class OracleStub:
    """The pricing stub of the scored-values oracle. Only NAME trades: T events are taken at production width; R events are
    declined ladder_too_coarse there and recovered at the first rung. Per-share P&L is edge - alpha, edge 0.8 (T) or 0.88 (R)."""
    @staticmethod
    def simulate(chunk, events, pool, **kw):
        return {str(e): "sim" for e in events["event_id"]}

    @staticmethod
    def score_events(chunk, events, sims, name, *, alpha=0.5):
        rows = []
        for ev in events.to_dict("records"):
            eid, status, reason = str(ev["event_id"]), "scored", None
            if name != NAME:
                status, reason = "undetermined", "structure_unresolved"
            elif eid.startswith("R") and abs(ev["pred_move"] - 5.0) < 1e-9:
                status, reason = "undetermined", "ladder_too_coarse"
            rows.append({"event_id": eid, "event_date": ev["event_date"], "structure": name, "width": np.nan,
                         "entry_cost": 1.0, "rel_spread": 0.0, "mcap_usd": ev["mcap_usd"], "exp_pnl_sim": 0.01,
                         "status": status, "reason": reason, "alpha": alpha})
        return pd.DataFrame(rows, columns=sv.SCORE_COLS)

    @staticmethod
    def resolve(chunk, i, name, pred):
        return [{"name": "long", "strike": 100.0, "side": BUY, "qty": 1.0},
                {"name": "short", "strike": 90.0, "side": SELL, "qty": 1.0}], None

    @staticmethod
    def decide(scored):
        out = scored.copy()
        scored_rows = out["status"] == "scored"
        out.loc[scored_rows, "status"] = "taken"
        out.loc[scored_rows, "reason"] = None
        return out

    @staticmethod
    def score(chunk, events, decisions, name, alpha, *, drop_unquoted=False, mid_marks=False, repriced=False):
        mine = decisions[(decisions["status"] == "taken") & (decisions["structure"] == name)]
        ids = mine["event_id"].tolist()
        pnl = [(0.8 if eid.startswith("T") else 0.88) - alpha for eid in ids]
        return pd.DataFrame({"event_id": pd.Series(ids, dtype=object), "structure": [name] * len(ids),
                             "entry_cost": [1.0] * len(ids), "exit_value": [1.0 + p for p in pnl],
                             "pnl_per_share": pnl, "pnl_dollars": [100 * p for p in pnl]},
                            columns=["event_id", "structure", "entry_cost", "exit_value", "pnl_per_share", "pnl_dollars"])

    @staticmethod
    def secured(chunk, events, decisions, name):
        mine = decisions[(decisions["status"] == "taken") & (decisions["structure"] == name)]
        return pd.DataFrame({"event_id": pd.Series(mine["event_id"].tolist(), dtype=object),
                             "secured_per_contract": pd.Series([5000.0] * len(mine), dtype=float)})


def _assert_t9(rows, n, rho, lo, hi):
    """One NAME row in tier ge_10B: n declined events, all recovered at the one rung ``rho`` of the phi's band (lo, hi); the n taken base trades are the T events."""
    assert len(rows) == 1
    row = rows[0]
    assert (row["structure"], row["tier"]) == (NAME, "ge_10B")
    assert (row["lo"], row["hi"]) == (pytest.approx(lo), pytest.approx(hi))
    assert (row["declined"], row["recovered"], row["fallback_infeasible"]) == (n, n, 0)
    assert row["share"] == pytest.approx(1.0) and row["rho"] == {rho: n}
    assert row["recovered_trades"] == n and row["recovered_ret"] == pytest.approx(0.38) and row["recovered_pnl"] == pytest.approx(38.0)
    assert row["taken_trades"] == n and row["taken_ret"] == pytest.approx(0.30) and row["taken_pnl"] == pytest.approx(30.0)


def test_fallback_inner_and_outer_scored_values_match_the_hand_computation(tmp_path, monkeypatch):
    # Hand computation. Per-share P&L is edge - alpha: at mid (alpha 0.5) T is 0.80 - 0.5 = 0.30 and R is 0.88 - 0.5 = 0.38.
    # Account 200,000; headroom 0.66 * 200,000 = 132,000; want 0.25 * 132,000 / 5,000 = 6.6 -> 6 contracts (still 6.6 after T's gain).
    # T gain = 6 * 100 * 0.30 = 180; R gain = 6 * 100 * 0.38 = 228.
    # Per window: base total_return = 180 / 200,000 = 0.0009; fallback = 408 / 200,000 = 0.00204.
    # First rho per phi: rungs are multiples of 0.05 inside the phi's band [lo, hi] without 1.0, nearest by |log rho| first.
    #   phi 0.8  band [0.5, 3.0]: 1.05 is nearer than 0.95, so 1.05 scores first.
    #   phi 0.9  band [0.5, 1.0]: rungs below 1 only, so 0.95 is first.
    #   phi 0.95 band [1.1, 3.0]: 1.1 is the nearest rung in band, so 1.1 is first.
    # The stub declines R only at pred_move 5.0 and every rung moves pred_move, so each first rung recovers R.
    # Breakeven alpha of mean return per trade: base is T only, 0.80 - a -> 0.80; fallback mean of T and R is 0.84 - a -> 0.84.
    # adopt: 0.84 <= 0.80 + 0.05, so alpha_ok.
    monkeypatch.setattr(size_width_band, "MIN_RECOVERED", 1)
    spans = {"T1": ("2023-08-01", "2023-08-02"), "R1": ("2023-09-05", "2023-09-06"),
             "T2": ("2024-02-01", "2024-02-02"), "R2": ("2024-03-05", "2024-03-06"),
             "T3": ("2025-02-01", "2025-02-02"), "R3": ("2025-03-05", "2025-03-06")}
    sessions = {eid: (pd.Timestamp(a), pd.Timestamp(b)) for eid, (a, b) in spans.items()}
    months = [({"event_id": np.array([eid])},
               pd.DataFrame({"event_id": [eid], "event_date": [pd.Timestamp(a)], "pred_move": [5.0], "mcap_usd": [20e9]}),
               True, None) for eid, (a, _b) in spans.items()]
    none = {"lo": None, "hi": None, "peak_k": None, "flags": ["no_positive_peak"]}
    band = {"0.8": {"lo": 0.5, "hi": 3.0, "peak_k": 1.0, "flags": []},
            "0.9": {"lo": 0.5, "hi": 1.0, "peak_k": 1.0, "flags": []},
            "0.95": {"lo": 1.1, "hi": 3.0, "peak_k": 1.5, "flags": []}}
    body = {n: {"primary": "B", "curves": {"B": {str(p): none for p in PHIS}}} for n in NAMES}
    body[NAME] = {"primary": "B", "curves": {"B": band}}
    write_bands(tmp_path, body, "H")
    stub, factors = OracleStub(), {n: 1.0 for n in NAMES}
    base = Sweep(ladder=True, pricing=stub)
    for chunk, events, _output, pool in months:
        base.add_month(chunk, events, pool, output=True, sessions=sessions)
    inner = flex.inner({"production": factors}, iter(months[:4]), 4, base, tmp_path, "H", sessions, stub)
    got = flex.outer({"production": factors}, inner, None, iter(months[4:]), 2, base, tmp_path, "H", sessions, stub, driver._stats)

    # phi -> (first rung that recovers R, band lo, band hi)
    per_phi = {"0.8": ("1.05", 0.5, 3.0), "0.9": ("0.95", 0.5, 1.0), "0.95": ("1.1", 1.1, 3.0)}
    for p in ("0.8", "0.9", "0.95"):
        cell = inner["production"]["phis"][p]
        assert cell["recovery"]["recovered"] == 2
        assert cell["recovery"]["recovered_ret"] == pytest.approx(0.38) and cell["recovery"]["taken_ret"] == pytest.approx(0.30)
        for w in ("2023H2", "2024"):
            assert cell["accounts"][w]["fallback"] == pytest.approx(0.00204)
            assert cell["accounts"][w]["base"] == pytest.approx(0.0009)
        _assert_t9(cell["t9"], 2, *per_phi[p])
    assert inner["production"]["choice"] == {"phi": 0.8, "flags": []}

    for p in ("0.8", "0.9", "0.95"):
        cell = got["sets"]["production"]["phis"][p]
        for w in ("2023H2", "2024", "2025"):
            assert cell["accounts"][w]["fallback"] == pytest.approx(0.00204)
            assert cell["accounts"][w]["base"] == pytest.approx(0.0009)
        assert cell["alpha"]["fallback"] == pytest.approx(0.84) and cell["alpha"]["base"] == pytest.approx(0.80)
        _assert_t9(cell["t9"], 1, *per_phi[p])
    adoption = got["adoption"]
    assert adoption["adopt"] is True and adoption["phi"] == 0.8
    assert adoption["beats"] is True and adoption["alpha_ok"] is True
