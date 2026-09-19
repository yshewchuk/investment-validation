"""Native entry-rule gate and the frozen trailing pnl_sim cutoff vs legacy.

Legacy side: the REAL ``Scorer._apply_entry_rule`` and ``Scorer._simulated_pnl``
(with ``pnl_sim.load_history`` pointed at a synthetic history and a stub
``_expectation``), recorded through a real ``Phase4TraceCollector``. Native
side: the capture converter's ``entry_rule_gate_block`` over that recording,
run by ``native_entry_rule.execute_entry_rule`` on the same priced legs and
simulated expectation. Synthetic only.
"""
from __future__ import annotations

import copy
import inspect
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from engine import entry_rules, pnl_sim
from engine import score as score_mod
from engine.v2.foundation import content_hash
from engine.v2.models.frozen_state import (
    FrozenStateLoader,
    FrozenStateRef,
    serialize_frozen_state,
)
from engine.v2.models.trailing_cutoff_artifact import (
    TRAILING_MIN_WINDOW,
    TRAILING_QUANTILE,
    TRAILING_WINDOW_MONTHS,
    TrailingCutoffError,
    make_trailing_cutoff_artifact,
    trailing_cutoff_from_document,
)
from engine.v2.models.training.trailing_cutoff import (
    build_trailing_cutoff_artifact,
    trailing_cutoff_lineage,
)
from engine.v2.scoring import native_entry_rule as native
from tools.capture_tier0_corpus import entry_rule_gate_block

STRATEGIES = tuple(entry_rules.ENTRY_RULES)
NAN, INF = float("nan"), float("inf")


def _history(seed: int = 7) -> pd.DataFrame:
    """~13 events a week from 2025-01 to 2026-06: thick windows from mid-2025,
    a thin one at 2025-02 (fewer than 100 prior events)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-02", "2026-06-30", freq="13h")
    values = rng.normal(0.02, 0.1, len(dates))
    values[::97] = np.nan  # missing values are dropped, as legacy's dropna
    return pd.DataFrame({"event_date": dates, "exp_pnl_sim": values})


def _legs(spread: float | None):
    """Two priced legs whose mean relative spread is ``spread`` (``None``:
    zero mids, which legacy's ``_mean_relative_spread`` skips -> no spread)."""
    if spread is None:
        return [{"bid": 0.0, "ask": 0.0}, {"bid": 0.0, "ask": 0.0}]
    mid = 2.0
    return [{"bid": mid - spread, "ask": mid + spread},
            {"bid": mid - spread, "ask": mid + spread}]


def _legacy(monkeypatch, strategy, *, history, event_date, sim, legs, mcap):
    """The real legacy rule on one synthetic row: ``(result, candidate)``."""
    monkeypatch.setattr(pnl_sim, "load_history", lambda path=None: history)
    stub = SimpleNamespace(_expectation=lambda request, result, features=None: (
        None if sim is None else {"exp_pnl_sim": sim, "win_sim": 0.5}))
    stub._simulated_pnl = lambda request, result, features=None: (
        score_mod.Scorer._simulated_pnl(stub, request, result, features))
    result = score_mod.ScoreResult(ticker="AAA", strategy=strategy,
                                   as_of=pd.Timestamp(event_date),
                                   event_date=pd.Timestamp(event_date))
    result.entry_cost = 1.0
    result.rel_spread = score_mod._mean_relative_spread(
        SimpleNamespace(legs=[SimpleNamespace(**leg) for leg in legs]))
    collector = score_mod.Phase4TraceCollector(retain_full_trace=False,
                                               content_hasher=content_hash)
    result._phase4_checkpoint_collector = collector
    features = pd.DataFrame({"mcap_usd": [np.nan if mcap is None else mcap]})
    score_mod.Scorer._apply_entry_rule(stub, SimpleNamespace(strategy=strategy), result, features)
    return result, {"legacy_trace": collector.diagnostic_checkpoint()}


def _native(candidate, strategy, event_date, legs, sim):
    flags: list[str] = []
    block = entry_rule_gate_block(candidate, strategy, str(event_date)[:10])
    output = native.execute_entry_rule(block, strategy=strategy, event_date=event_date,
                                       legs=legs, exp_pnl_sim=sim, flags=flags)
    return output, flags, block


# -- constants and the rule table --------------------------------------------


def test_constants_and_rule_table_match_legacy():
    assert native.ENTRY_RULE_STRATEGIES == STRATEGIES
    assert (native.MAX_REL_SPREAD, native.MCAP_FLOOR) == (
        entry_rules.MAX_REL_SPREAD, entry_rules.MCAP_FLOOR)
    for rule in entry_rules.ENTRY_RULES.values():
        assert tuple((term.name, term.needs) for term in rule.terms) == native.ENTRY_RULE_TERMS
    default = inspect.signature(pnl_sim.trailing_cutoff).parameters["min_window"].default
    assert (TRAILING_WINDOW_MONTHS, TRAILING_QUANTILE, TRAILING_MIN_WINDOW) == (
        pnl_sim.WINDOW_MONTHS, pnl_sim.QUANTILE, default)


_FACT_VALUES = {
    "exp_pnl_sim": (None, NAN, INF, -0.2, 0.05, 0.06, "x"),
    "pnl_cutoff": (None, NAN, 0.05, 0.06),
    "rel_spread": (None, NAN, 0.25, 0.2500001, 0.1),
    "mcap_usd": (None, NAN, 1e10, 9.999e9, 5e10),
}


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_evaluate_equals_legacy_rule_on_every_fact_combination(strategy):
    rule = entry_rules.ENTRY_RULES[strategy]
    grid = [dict(zip(_FACT_VALUES, combo)) for combo in
            np.array(np.meshgrid(*[range(len(v)) for v in _FACT_VALUES.values()])).T.reshape(-1, 4)]
    for index in grid:
        facts = {name: _FACT_VALUES[name][int(i)] for name, i in index.items()}
        legacy = rule.evaluate(facts)
        passed, terms = native.evaluate_entry_rule(facts)
        assert (passed, terms) == (legacy.passed, legacy.terms), facts


# -- the real legacy _apply_entry_rule, recorded and converted ----------------

_SCENARIOS = (
    # (event_date, sim, spread, mcap, history?)
    ("2026-03-16", "at_bar", 0.1, 5e10, True),
    ("2026-03-16", "below_bar", 0.1, 5e10, True),
    ("2026-03-16", 0.9, 0.1, 5e10, True),
    ("2026-03-16", 0.9, 0.3, 5e10, True),
    ("2026-03-16", 0.9, 0.1, 9e9, True),
    ("2026-03-16", 0.9, 0.1, 1e10, True),
    ("2026-03-16", 0.9, None, 5e10, True),
    ("2026-03-16", 0.9, 0.1, None, True),
    ("2026-03-16", None, 0.1, 5e10, True),
    ("2026-03-16", NAN, 0.1, 5e10, True),
    ("2025-02-10", 0.9, 0.1, 5e10, True),   # thin window: no bar
    ("2026-03-16", 0.9, 0.1, 5e10, False),  # no history file
    ("2026-03-16", -0.9, 0.3, 9e9, True),
)


def _bar(history, event_date):
    return pnl_sim.trailing_cutoff(history, event_date) if history is not None else None


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_native_gate_equals_legacy_apply_entry_rule(monkeypatch, strategy):
    base = _history()
    for event_date, sim, spread, mcap, has_history in _SCENARIOS:
        history = base if has_history else None
        bar = _bar(history, event_date)
        if sim == "at_bar":
            sim = bar
        elif sim == "below_bar":
            sim = float(np.nextafter(bar, -INF))
        legs = _legs(spread)
        result, candidate = _legacy(monkeypatch, strategy, history=history,
                                    event_date=event_date, sim=sim, legs=legs, mcap=mcap)
        output, flags, block = _native(candidate, strategy, event_date, legs,
                                       result.exp_pnl_sim)
        case = (strategy, event_date, sim, spread, mcap, has_history)
        assert output == {"gate_pass": result.gate_pass}, case
        assert ("MISSING_FEATURES" in flags) == ("MISSING_FEATURES" in result.flags), case
        assert set(flags) <= {"MISSING_FEATURES"}, case
        # The release builder freezes the same bar under the same content hash,
        # so a staged release artifact can be pinned in place of the recorded one.
        rows = [] if history is None else history.to_dict("records")
        built = build_trailing_cutoff_artifact(rows, month=event_date)
        assert built.cutoff == bar
        assert block["trailing_cutoff_key"]["content_hash"] == built.content_hash, case


def test_no_gate_inputs_means_no_entry_rule_block():
    assert entry_rule_gate_block({"legacy_trace": {"checkpoints": {}}}, "TWIN-P", "2026-03-16") is None


def test_a_rule_recorded_for_another_strategy_is_refused(monkeypatch):
    _result, candidate = _legacy(monkeypatch, "TWIN-P", history=_history(),
                                 event_date="2026-03-16", sim=0.9, legs=_legs(0.1), mcap=5e10)
    with pytest.raises(Exception, match="entry rule"):
        entry_rule_gate_block(candidate, "CTR5", "2026-03-16")


# -- the frozen artifact ------------------------------------------------------


def test_trailing_cutoff_artifact_round_trips_through_the_frozen_loader(tmp_path):
    artifact = build_trailing_cutoff_artifact(_history().to_dict("records"), month="2026-03-16")
    assert artifact.month == "2026-03-01" and artifact.cutoff is not None
    payload = serialize_frozen_state(artifact)
    (tmp_path / "cut.json").write_bytes(payload)
    loaded = FrozenStateLoader(tmp_path).load(FrozenStateRef(
        path="cut.json", content_hash="sha256:" + __import__("hashlib").sha256(payload).hexdigest()))
    assert loaded == artifact
    assert trailing_cutoff_from_document(artifact.payload()) == artifact


def test_artifact_refuses_non_finite_bars_and_foreign_constants():
    lineage = trailing_cutoff_lineage("2026-03-01")
    with pytest.raises(TrailingCutoffError):
        make_trailing_cutoff_artifact(month="2026-03-01", cutoff=INF, lineage=lineage)
    with pytest.raises(TrailingCutoffError):
        make_trailing_cutoff_artifact(month="2026-03-01", cutoff=0.1, lineage=lineage,
                                      quantile=0.25)


def test_builder_matches_legacy_trailing_cutoff_month_by_month():
    history = _history(11)
    rows = history.to_dict("records")
    for day in pd.date_range("2025-01-15", "2026-07-15", freq="MS") + pd.Timedelta(days=9):
        assert build_trailing_cutoff_artifact(rows, month=day).cutoff == (
            pnl_sim.trailing_cutoff(history, day)), day


# -- refusals and planted defects ---------------------------------------------


def _block(month_artifact, *, pin=True, mcap=5e10, month=None):
    return native.entry_rule_block("TWIN-P", mcap_usd=mcap, cutoff=month_artifact,
                                   month=month, pin=pin)


def _run(block, event_date="2026-03-16", sim=0.9, legs=None):
    flags: list[str] = []
    out = native.execute_entry_rule(block, strategy="TWIN-P", event_date=event_date,
                                    legs=_legs(0.1) if legs is None else legs,
                                    exp_pnl_sim=sim, flags=flags)
    return out, flags


def test_absent_or_mismatched_cutoff_is_model_not_ready():
    rows = _history().to_dict("records")
    march = build_trailing_cutoff_artifact(rows, month="2026-03-01")
    assert _run(_block(march)) == ({"gate_pass": True}, [])
    assert _run(_block(None, month="2026-03-01")) == ({}, ["MODEL_NOT_READY"])
    # Planted off-by-one month: February's bar served for a March event.
    february = build_trailing_cutoff_artifact(rows, month="2026-02-01")
    assert _run(_block(february)) == ({}, ["MODEL_NOT_READY"])
    relabelled = _block(march)
    relabelled["trailing_cutoff"] = february.payload()
    assert _run(relabelled) == ({}, ["MODEL_NOT_READY"])
    # A pinned hash that is not the served artifact's.
    other = make_trailing_cutoff_artifact(month="2026-03-01", cutoff=march.cutoff + 1.0,
                                          lineage=march.lineage)
    pinned = _block(march)
    pinned["trailing_cutoff"] = other.payload()
    assert _run(pinned) == ({}, ["MODEL_NOT_READY"])
    # Another strategy's rule, or an unknown field: never evaluated.
    assert _run({**_block(march), "rule": "CTR5"}) == ({}, ["UNSUPPORTED_GATE_RECIPE:entry_rule"])
    assert _run({**_block(march), "facts": {"mcap_usd": 5e10, "exp_pnl_sim": 1.0}}) == (
        {}, ["UNSUPPORTED_GATE_RECIPE:entry_rule"])


def test_verdicts_dimension_catches_an_off_by_one_cutoff(monkeypatch):
    """A bar one ulp above legacy's flips an at-the-bar row from pass to fail,
    and Phase 4's verdicts comparison reports it; the faithful bar agrees."""
    from checks.phase4_real import _compare_dimension

    history = _history()
    bar = pnl_sim.trailing_cutoff(history, "2026-03-16")
    result, candidate = _legacy(monkeypatch, "TWIN-P5", history=history,
                                event_date="2026-03-16", sim=bar, legs=_legs(0.1), mcap=5e10)
    assert result.gate_pass is True
    expected = {"gate_score": None, "gate_threshold": None, "gate_pass": result.gate_pass}

    def verdicts(block):
        flags: list[str] = []
        out = native.execute_entry_rule(block, strategy="TWIN-P5", event_date="2026-03-16",
                                        legs=_legs(0.1), exp_pnl_sim=bar, flags=flags)
        return {"gate_score": None, "gate_threshold": None, "gate_pass": out.get("gate_pass")}

    faithful = entry_rule_gate_block(candidate, "TWIN-P5", "2026-03-16")
    assert _compare_dimension(expected, verdicts(faithful), "verdicts")["agree"] is True
    nudged = make_trailing_cutoff_artifact(month="2026-03-01", cutoff=float(np.nextafter(bar, INF)),
                                           lineage=trailing_cutoff_lineage("2026-03-01"))
    planted = native.entry_rule_block("TWIN-P5", mcap_usd=5e10, cutoff=nudged)
    compared = _compare_dimension(expected, verdicts(planted), "verdicts")
    assert compared["agree"] is False
    assert "gate_pass" in " ".join(compared["finding_fields"])


# -- the training job and the preparer -----------------------------------------


def test_training_job_freezes_one_state_per_month_and_the_preparer_stages_it(tmp_path):
    from tools import phase5_prepare_release as prep
    from tools.phase5_training_job import run_trailing_cutoff_job

    history = _history()
    plan = run_trailing_cutoff_job(tmp_path, as_of=["2026-03-16", "2026-03-02", "2025-02-10"],
                                   plan_only=True, history=history)
    assert plan["months"] == ["2025-02-01", "2026-03-01"] and not list(tmp_path.glob("*__*"))
    summary = run_trailing_cutoff_job(tmp_path, as_of=["2026-03-16", "2025-02-10"],
                                      history=history)
    assert [f["has_bar"] for f in summary["files"]] == [False, True]
    again = run_trailing_cutoff_job(tmp_path, as_of=["2026-03-16"], history=history)
    assert again["files"][0]["status"] == "resumed"
    with pytest.raises(SystemExit, match="RESUME_MISMATCH"):
        run_trailing_cutoff_job(tmp_path, as_of=["2026-03-16"], history=_history(99))
    found = prep.frozen_state_payloads([tmp_path])
    assert sorted(found["trailing_pnl_cutoff"]) == [
        "features.pnl_sim_history|2025-02-01", "features.pnl_sim_history|2026-03-01"]
    rows = history.to_dict("records")
    staged = found["trailing_pnl_cutoff"]["features.pnl_sim_history|2026-03-01"]
    assert staged == serialize_frozen_state(build_trailing_cutoff_artifact(rows, month="2026-03-01"))


def test_score_one_runs_the_entry_rule_after_the_simulation():
    """End to end: the gate stage reads the simulation stage's own
    ``exp_pnl_sim`` and the priced legs; an undetermined verdict flags
    MISSING_FEATURES on the record, as legacy's row does."""
    from checks.phase5_consumers import _entry_rule_inputs, _score_inputs

    lineage = trailing_cutoff_lineage("2026-09-01")
    low = make_trailing_cutoff_artifact(month="2026-09-01", cutoff=-50.0, lineage=lineage)
    high = make_trailing_cutoff_artifact(month="2026-09-01", cutoff=50.0, lineage=lineage)
    inputs = _entry_rule_inputs("2026-09-15")
    passed = _score_inputs(inputs, native.entry_rule_block("TWIN-P", mcap_usd=5e10, cutoff=low))
    failed = _score_inputs(inputs, native.entry_rule_block("TWIN-P", mcap_usd=5e10, cutoff=high))
    unknown = _score_inputs(inputs, native.entry_rule_block("TWIN-P", mcap_usd=None, cutoff=low))
    assert passed.gate_terms == {"gate_score": None, "gate_threshold": None, "gate_pass": True}
    assert failed.gate_terms["gate_pass"] is False
    assert unknown.gate_terms["gate_pass"] is None
    assert "MISSING_FEATURES" in unknown.reason_codes
    assert "MISSING_FEATURES" not in passed.reason_codes + failed.reason_codes


# -- capture -> strict trace -> phase4_real verification and replay ------------

_DIVISOR = {"TWIN-P": 1.5, "TWIN-P5": 1.0, "CND-PS": 2.0, "BFLY-P": 1.0, "BFLY-P5": 3.0,
            "RAMP7": 3.0, "CTR5": 2.0}


def _entry_rule_candidate(tmp_path, strategy, pnl_cutoff):
    """A scored rule-gated row as the capture records it: a size binding
    (forecast = 2 x the strategy's width divisor, so every strike lands on the
    half-dollar put grid), a terminal simulation and ``gate_inputs`` kind
    entry_rule. No gate binding: legacy had no gate champion."""
    from tests.test_phase4_capture_strict import _artifact, _full_strict_candidate

    path, digest = _artifact(tmp_path)
    candidate = _full_strict_candidate(
        fixture_id="rule", ticker="AAA", strategy=strategy,
        driver_vector={"x": 2.0 * _DIVISOR[strategy]}, gate_vector={"x": 9.0},
        path=path, digest=digest)
    checkpoints = candidate["legacy_trace"]["checkpoints"]
    source = checkpoints["source_inputs"]["value"]
    source["model_bindings"] = [b for b in source["model_bindings"] if b["role"] != "gate"]
    for binding in source["model_bindings"]:
        binding.update(decision_clock="legacy.decision_offset.0", strategy=strategy,
                       role="size", output_names=["forecast_abs_move"])
    source["quote_domain"] = [{"right": "P", "strike": 80 + 0.5 * i, "expiry": "2026-09-18",
                               "bid": 1.0, "ask": 1.2} for i in range(81)]
    source["native_recipes"]["simulation"] = {
        "terminal_spots": [95.0, 105.0], "weights": [0.5, 0.5], "capital_at_risk": 1.0}
    checkpoints["source_inputs"]["content_hash"] = content_hash(source)
    features = checkpoints["features"]["value"]
    for key in ("feature_vector", "missing_mask", "model_identity"):
        features[key] = {"size": features[key]["abs_move"]}
    checkpoints["features"]["content_hash"] = content_hash(features)
    gate = {"kind": "entry_rule", "rule_identity": f"entry-rule:{strategy}",
            "facts": {"mcap_usd": 5e10, "pnl_cutoff": pnl_cutoff, "rel_spread": 0.1},
            "terms": {}}
    checkpoints["gate_inputs"] = {"value": gate, "content_hash": content_hash(gate)}
    return candidate


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_captured_rule_gate_traces_verifies_and_replays(tmp_path, monkeypatch, strategy):
    import shutil

    from checks import phase4_real
    from tools.capture_tier0_corpus import strict_trace_one

    verdicts = []
    for index, bar in enumerate((-50.0, 50.0, None)):
        root = tmp_path / f"case-{index}"
        root.mkdir()
        monkeypatch.setattr("engine.paths.ROOT", root)
        candidate = _entry_rule_candidate(root, strategy, bar)
        trace, native_record = strict_trace_one(candidate, "snapshot-1", root / "captured")
        assert trace["native_inputs"]["gate"]["mode"] == "entry_rule"
        shutil.copytree(root / "captured", root / "published")
        pair = {"payload": {"request": candidate["request"], "input_trace": trace,
                            "input_trace_hash": trace["trace_hash"],
                            "legacy_input_hash": trace["shared_input_hash"]}}
        verified = phase4_real._verified_trace_bundle(pair, root / "published")
        replayed, _receipts, _ids = phase4_real._replayed_member(verified)
        assert replayed.score_id == native_record.score_id
        verdicts.append((replayed.gate_terms["gate_pass"],
                         "MISSING_FEATURES" in replayed.reason_codes))
        # Planted defect: the recorded bar changed after capture no longer
        # verifies (the gate block is inside the hash-bound trace).
        tampered = copy.deepcopy(pair)
        block = tampered["payload"]["input_trace"]["native_inputs"]["gate"]
        block["trailing_cutoff"]["cutoff"] = 1.0
        with pytest.raises(phase4_real._TraceError):
            phase4_real._verified_trace_bundle(tampered, root / "published")
    assert verdicts == [(True, False), (False, False), (None, True)]


def test_phase5_replay_requires_the_pinned_cutoff_to_be_staged():
    from types import SimpleNamespace as NS

    from checks.phase5_phase4_replay import _entry_rule_cutoff, staged_model_objects

    cutoff = build_trailing_cutoff_artifact(_history().to_dict("records"), month="2026-03-01")
    manifest = {"members": [{"member_id": "trailing_pnl_cutoff", "objects": [
        {"name": "features.pnl_sim_history|2026-03-01", "content_hash": cutoff.content_hash}]}]}
    staged = staged_model_objects(NS(bindings=()), manifest)
    assert staged == {cutoff.content_hash: "trailing_pnl_cutoff"}
    rule = NS(gate=native.entry_rule_block("TWIN-P", mcap_usd=5e10, cutoff=cutoff))
    assert _entry_rule_cutoff(rule) in staged
    unpinned = NS(gate=native.entry_rule_block("TWIN-P", mcap_usd=5e10, cutoff=cutoff, pin=False))
    assert _entry_rule_cutoff(unpinned) not in staged
    assert _entry_rule_cutoff(NS(gate={"mode": "not_applicable"})) is None
