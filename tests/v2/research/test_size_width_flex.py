"""EXP-189 flex: the curves, the factor target, the 2025 stage's pre-stated rules (with stubs), and the fallback's pricing seam.
Synthetic fixtures only."""
import pytest

from experiments import size_pnl_shortvol as sv
from experiments import size_width_flex as flex
from experiments.size_pnl_pool import build_pool
from experiments.size_pnl_shortvol import NAMES, Refusal
from experiments.size_width_band import PHIS, write_bands
from experiments.size_width_fallback import FallbackSweep
from experiments.size_width_sweep import E_ARM
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


def test_mismatched_inner_factors_refuse_before_any_month_is_consumed(stubs):
    inner = _inner_stage(0.8)
    inner["production"]["factors"] = _factors(0.9)
    pulled = []

    def months():
        pulled.append(1)  # runs only when the generator is advanced
        yield from ()

    with pytest.raises(Refusal) as caught:
        flex.outer({"production": _factors()}, inner, None, months(), 0, object(), None, "H", SESSIONS, None, _stats)
    assert caught.value.code == CONFLICT and caught.value.details["reason"] == "tuning_receipt_changed"
    assert pulled == [] and stubs == []


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
