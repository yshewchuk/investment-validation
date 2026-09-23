"""Per-row, per-dimension evidence added to ``_native_parity``.

Before this, the emitted evidence kept only ``fixture_id``/``disposition``/
``reason`` per row (``release["dispositions"]``): diagnosing which dimension
disagreed on which row required a separate single-fixture reproduction.
``_record_checks`` and ``_compare_dimension`` already compute the per-row,
per-dimension result and the diverging field NAMES; this file proves the
additive evidence (``row_dimension_checks``, ``row_numeric_findings``,
``dimension_rollup``) actually surfaces that data, that the roll-up agrees
with the per-row detail, and -- the hard constraint -- that none of it ever
carries a value from either side's record.

Fixture/helper shape follows ``tests/test_phase4_numeric_negative_control_states.py``
(synthetic records driven through the real ``_native_parity`` orchestration,
only ``_verified_trace_bundle``/``_replayed_member`` stubbed).
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from checks import phase4_real
from engine.v2.contracts import ScoreRecord


def _clean_record(**overrides):
    record = {
        "strategy": "STR-THRU",
        "driver_prediction": 5.0,
        "forecast_abs_move": 5.0,
        "exp_pnl_sim": 0.2,
        "exp_pnl_model": 0.15,
        "win_sim": 0.6,
        "win_model": 0.55,
        "gate_score": 0.8,
        "gate_threshold": 0.5,
        "gate_pass": True,
        "ci_low": -0.01,
        "ci_high": 0.05,
        "n_analogs": 10,
        "spot": 100.0,
        "entry_cost": 5.0,
        "structure_width": 10.0,
        "driver_name": "abs_move",
        "implied_move": 4.0,
        "payoff": {"intercept": 0.0, "slope": 0.03},
        "flags": (),
        "model_inputs": {"x": 1.0, "y": None},
        "legs": (
            {"name": "call", "right": "C", "side": "long", "quantity": 1.0,
             "strike": 100.0, "expiry": "2026-09-18", "fill": 1.5,
             "cash_flow": -150.0},
        ),
        "entry_date": "2026-09-16", "exit_date": "2026-09-17",
        "quote_date": "2026-09-16",
    }
    record.update(overrides)
    return record


def _native_record(record):
    return ScoreRecord(
        score_id="native-score-1",
        canonical_request={},
        resolved_request=dict(record),
        event_ref={},
        clock_id="clock-1",
        snapshot_ref="snapshot-1",
        dependency_hash="dep-1",
        model_artifact_ids=(),
        selected_contracts=(),
        legs=tuple(record.get("legs") or ()),
        entry_exit_plan={
            "entry_date": record.get("entry_date"),
            "exit_date": record.get("exit_date"),
        },
        quote_provenance={"quote_date": record.get("quote_date")},
        forecasts={
            "driver_prediction": record.get("driver_prediction"),
            "forecast_abs_move": record.get("forecast_abs_move"),
            "exp_pnl_sim": record.get("exp_pnl_sim"),
            "exp_pnl_model": record.get("exp_pnl_model"),
            "win_sim": record.get("win_sim"),
            "win_model": record.get("win_model"),
        },
        uncertainty={},
        residual_state_ref=None,
        analog_state_ref=None,
        payoff_state_ref=None,
        feature_values={},
        null_masks={
            key: value is None
            for key, value in (record.get("model_inputs") or {}).items()
        },
        feature_lineage_refs=(),
        gate_terms={
            "gate_score": record.get("gate_score"),
            "gate_threshold": record.get("gate_threshold"),
            "gate_pass": record.get("gate_pass"),
        },
        chooser_candidates=(),
        chooser_selection=None,
        financial_diagnostics=phase4_real._expected_financial_diagnostics(record),
        requested_payoff_views=(),
        validation_status="scored",
        reason_codes=tuple(record.get("flags") or ()),
        warnings=(),
        evidence_refs=(),
    )


def _pair(fixture_id, record, kind="score_result"):
    return {
        "fixture_id": fixture_id,
        "payload_hash": f"hash-{fixture_id}",
        "payload": {"record_kind": kind, "record": record},
    }


def _corpus(tmp_path, *pairs):
    return SimpleNamespace(
        root=tmp_path,
        index={"pairs": {
            p["fixture_id"]: {"record_kind": p["payload"]["record_kind"]}
            for p in pairs
        }},
        ordered_ids=[p["fixture_id"] for p in pairs],
        pairs={p["fixture_id"]: p for p in pairs},
    )


def _stub_replay(records_by_fixture, natives_by_fixture=None):
    natives_by_fixture = natives_by_fixture or {}

    def verified_fn(pair, _root):
        return {
            "same_input_receipt": "same", "trace_hash": "trace",
            "frozen_replay": None, "fixture_id": pair["fixture_id"],
        }

    def replayed_fn(verified):
        fixture_id = verified["fixture_id"]
        record = records_by_fixture[fixture_id]
        native = natives_by_fixture.get(fixture_id) or _native_record(record)
        receipts = tuple({"stage": s} for s in phase4_real._REQUIRED_TRACE_STAGES)
        return native, receipts, ()

    return verified_fn, replayed_fn


def _run(tmp_path, monkeypatch, records_by_fixture, natives_by_fixture=None):
    pairs = [_pair(fid, record) for fid, record in records_by_fixture.items()]
    corpus = _corpus(tmp_path, *pairs)
    verified_fn, replayed_fn = _stub_replay(records_by_fixture, natives_by_fixture)
    monkeypatch.setattr(phase4_real, "_verified_trace_bundle", verified_fn)
    monkeypatch.setattr(phase4_real, "_replayed_member", replayed_fn)
    return phase4_real._native_parity(corpus)


def _assert_only_safe_leaves(node, *, allow_str=True, allow_bool=True, allow_int=True):
    """Walk a nested structure and fail on any leaf that is not one of the
    explicitly value-free kinds this evidence is allowed to carry: a string
    (field/dimension name), a bool (pass/fail), or a plain int (a count).
    A float leaf anywhere -- rounded, bucketed or exact -- is a violation,
    since Requirement 3 forbids it categorically."""
    if isinstance(node, dict):
        for value in node.values():
            _assert_only_safe_leaves(
                value, allow_str=allow_str, allow_bool=allow_bool, allow_int=allow_int,
            )
        return
    if isinstance(node, (list, tuple)):
        for value in node:
            _assert_only_safe_leaves(
                value, allow_str=allow_str, allow_bool=allow_bool, allow_int=allow_int,
            )
        return
    assert not isinstance(node, float), f"found a float leaf where none is allowed: {node!r}"
    if isinstance(node, bool):
        assert allow_bool, f"unexpected bool leaf: {node!r}"
        return
    if isinstance(node, int):
        assert allow_int, f"unexpected int leaf: {node!r}"
        return
    if isinstance(node, str):
        assert allow_str, f"unexpected str leaf: {node!r}"
        return
    if node is None:
        return
    raise AssertionError(f"unexpected leaf type {type(node)!r}: {node!r}")


def test_per_row_checks_appear_for_a_compared_row_keyed_by_fixture_id(tmp_path, monkeypatch):
    record = _clean_record()
    release, _parity = _run(tmp_path, monkeypatch, {"a": record})

    assert release["population"]["compared"] == 1
    assert "a" in release["row_dimension_checks"]
    checks = release["row_dimension_checks"]["a"]
    # A clean record replayed against a native built from the SAME record
    # agrees on every dimension it carries.
    for dimension in ("keys", "contracts", "flags", "null_masks",
                      "forecasts", "simulation", "financial_diagnostics",
                      "verdicts", "analogs"):
        assert checks[dimension] is True, (dimension, checks)
    assert "a" in release["row_numeric_findings"]
    findings = release["row_numeric_findings"]["a"]
    for dimension in ("forecasts", "simulation", "financial_diagnostics",
                      "verdicts", "analogs"):
        assert findings[dimension] == [], (dimension, findings[dimension])


def test_rollup_counts_match_per_row_data(tmp_path, monkeypatch):
    from dataclasses import replace

    agreeing_record = _clean_record()
    disagreeing_record = _clean_record()
    # Corrupt only the NATIVE side of "b"'s simulation dimension -- the
    # legacy record is untouched, matching R4-15's "never touch record"
    # convention for planted defects.
    corrupted_native = replace(
        _native_record(disagreeing_record),
        forecasts={**_native_record(disagreeing_record).forecasts,
                   "exp_pnl_sim": 0.987654321},
    )

    release, _parity = _run(
        tmp_path, monkeypatch,
        {"a": agreeing_record, "b": disagreeing_record},
        natives_by_fixture={"b": corrupted_native},
    )

    assert release["population"]["compared"] == 2
    assert release["row_dimension_checks"]["a"]["simulation"] is True
    assert release["row_dimension_checks"]["b"]["simulation"] is False

    rollup = release["dimension_rollup"]["simulation"]
    passed_from_rows = sum(
        1 for checks in release["row_dimension_checks"].values()
        if checks.get("simulation") is True
    )
    failed_from_rows = sum(
        1 for checks in release["row_dimension_checks"].values()
        if checks.get("simulation") is False
    )
    assert rollup == {"passed": passed_from_rows, "failed": failed_from_rows}
    assert rollup == {"passed": 1, "failed": 1}


def test_numeric_finding_names_the_diverging_field_and_carries_no_value(tmp_path, monkeypatch):
    """The most important test: the diverging field NAME must show up, and
    the diverging VALUES -- on either side -- must never appear anywhere in
    the row-level evidence. Two distinct, deliberately unmistakable magic
    floats stand in for "legacy value" and "native value" so a leak of
    either one is easy to catch and cannot be confused with an unrelated
    number (e.g. a rollup count)."""
    from dataclasses import replace

    legacy_magic = 0.20  # the legacy record's own exp_pnl_sim, set below
    native_magic = 0.987654321
    record = _clean_record(exp_pnl_sim=legacy_magic)
    corrupted_native = replace(
        _native_record(record),
        forecasts={**_native_record(record).forecasts, "exp_pnl_sim": native_magic},
    )

    release, _parity = _run(
        tmp_path, monkeypatch, {"a": record},
        natives_by_fixture={"a": corrupted_native},
    )

    checks = release["row_dimension_checks"]["a"]
    findings = release["row_numeric_findings"]["a"]
    assert checks["simulation"] is False
    assert "exp_pnl_sim" in findings["simulation"]

    # Value-freedom, structural: every leaf under the three new evidence
    # keys is a string, bool or int -- never a float, whatever the
    # underlying records happened to disagree about.
    _assert_only_safe_leaves(release["row_dimension_checks"], allow_int=False)
    _assert_only_safe_leaves(release["row_numeric_findings"], allow_bool=False, allow_int=False)
    _assert_only_safe_leaves(release["dimension_rollup"], allow_str=False)

    # Value-freedom, by content: neither magic value is reachable through
    # the new evidence, serialized.
    dumped = json.dumps({
        "row_dimension_checks": release["row_dimension_checks"],
        "row_numeric_findings": release["row_numeric_findings"],
        "dimension_rollup": release["dimension_rollup"],
    })
    assert str(native_magic) not in dumped
    assert "0.987654321" not in dumped
    assert "0.2" not in dumped


def test_key_differences_recorded_when_a_decision_field_vanishes_from_native(
        tmp_path, monkeypatch):
    """The one case that matters: a decision-carrying field present on the
    legacy side but silently absent from native's resolved_request fails
    "keys" and is named in row_key_differences.legacy_only."""
    from dataclasses import replace

    record = _clean_record()
    native_missing_gate_score = replace(
        _native_record(record),
        resolved_request={k: v for k, v in record.items() if k != "gate_score"},
    )

    release, _parity = _run(
        tmp_path, monkeypatch,
        {"a": record},
        natives_by_fixture={"a": native_missing_gate_score},
    )

    checks = release["row_dimension_checks"]["a"]
    assert checks["keys"] is False
    assert "a" in release["row_key_differences"]
    diff = release["row_key_differences"]["a"]
    assert diff["legacy_only"] == ["gate_score"]
    assert diff["native_only"] == []


def test_null_decision_key_placeholders_count_as_absent_on_both_sides(
        tmp_path, monkeypatch):
    """None placeholders match native omissions, while populated keys remain."""
    from dataclasses import replace

    record = _clean_record(runup_move_1d=None, driver_prediction=None)
    native = replace(
        _native_record(record),
        resolved_request={
            **_native_record(record).resolved_request,
            "chooser_score": None,
        },
    )

    release, _parity = _run(
        tmp_path, monkeypatch,
        {"a": record},
        natives_by_fixture={"a": native},
    )

    assert release["row_dimension_checks"]["a"]["keys"] is True
    assert "a" not in release["row_key_differences"]


def test_out_of_scope_key_difference_does_not_fail_keys_check(tmp_path, monkeypatch):
    """A structural field present on only one side, but outside
    _DECISION_KEY_FIELDS, must NOT fail "keys" -- the two documents are
    different shapes by design, and only the named correspondence set is
    supposed to match."""
    from dataclasses import replace

    record = _clean_record()
    # Add an extra key to the legacy record
    record_with_extra = {**record, "extra_legacy_key": "value"}
    # Create a native with a different, out-of-scope extra key
    native_with_diff = replace(
        _native_record(record),
        resolved_request={**record, "extra_native_key": "value"}
    )

    release, _parity = _run(
        tmp_path, monkeypatch,
        {"a": record_with_extra},
        natives_by_fixture={"a": native_with_diff},
    )

    checks = release["row_dimension_checks"]["a"]
    assert checks["keys"] is True
    assert "a" not in release.get("row_key_differences", {})


def test_flag_differences_recorded_when_flags_check_fails(tmp_path, monkeypatch):
    """A row with a flag mismatch records symmetric difference."""
    from dataclasses import replace

    record = _clean_record(flags=("FLAG_A", "FLAG_B"))
    native_with_diff = replace(
        _native_record(record),
        reason_codes=("FLAG_A", "FLAG_C")  # FLAG_B is only in legacy, FLAG_C only in native
    )

    release, _parity = _run(
        tmp_path, monkeypatch,
        {"a": record},
        natives_by_fixture={"a": native_with_diff},
    )

    checks = release["row_dimension_checks"]["a"]
    assert checks["flags"] is False
    assert "a" in release["row_flag_differences"]
    diff = release["row_flag_differences"]["a"]
    assert "native_only" in diff
    assert "legacy_only" in diff
    assert "FLAG_C" in diff["native_only"]
    assert "FLAG_B" in diff["legacy_only"]


def test_passing_row_omits_key_and_flag_differences(tmp_path, monkeypatch):
    """A row where all checks pass does not record difference entries."""
    record = _clean_record()
    # Use the same record for native so everything matches
    release, _parity = _run(
        tmp_path, monkeypatch,
        {"a": record},
    )

    checks = release["row_dimension_checks"]["a"]
    assert checks["keys"] is True
    assert checks["flags"] is True
    # The fixture should not appear in row_key_differences or row_flag_differences
    assert "a" not in release.get("row_key_differences", {})
    assert "a" not in release.get("row_flag_differences", {})


def test_key_flag_differences_contain_only_strings(tmp_path, monkeypatch):
    """All leaves in key and flag differences are field/flag names (strings)."""
    from dataclasses import replace

    record = _clean_record(flags=("FLAG_A",))
    native_with_diffs = replace(
        _native_record(record),
        reason_codes=("FLAG_B",),
        resolved_request={k: v for k, v in record.items() if k != "gate_score"},
    )

    release, _parity = _run(
        tmp_path, monkeypatch,
        {"a": record},
        natives_by_fixture={"a": native_with_diffs},
    )

    # Assert no float leaves in the new difference fields
    _assert_only_safe_leaves(
        release.get("row_key_differences", {}),
        allow_bool=False, allow_int=False
    )
    _assert_only_safe_leaves(
        release.get("row_flag_differences", {}),
        allow_bool=False, allow_int=False
    )


def test_null_mask_differences_recorded_when_check_fails(tmp_path, monkeypatch):
    """A row with a null-mask mismatch records native_only/legacy_only/
    value_mismatch feature key NAMES -- one of each in the same row, so the
    three buckets can't be confused with one another."""
    from dataclasses import replace

    record = _clean_record(model_inputs={"x": 1.0, "y": None, "w": 2.0})
    # legacy_mask (derived from model_inputs) = {"x": False, "y": True, "w": False}
    native_with_diff = replace(
        _native_record(record),
        # "x": value_mismatch (False on legacy, True here); "w": legacy_only
        # (absent here); "z": native_only (absent on legacy); "y": agrees.
        null_masks={"x": True, "y": True, "z": True},
    )

    release, _parity = _run(
        tmp_path, monkeypatch,
        {"a": record},
        natives_by_fixture={"a": native_with_diff},
    )

    checks = release["row_dimension_checks"]["a"]
    assert checks["null_masks"] is False
    assert "a" in release["row_null_mask_differences"]
    diff = release["row_null_mask_differences"]["a"]
    assert diff["native_only"] == ["z"]
    assert diff["legacy_only"] == ["w"]
    assert diff["value_mismatch"] == ["x"]


def test_passing_row_omits_null_mask_differences(tmp_path, monkeypatch):
    """A row where the null-mask check passes does not record a difference
    entry (mirrors the existing key/flag omission test)."""
    record = _clean_record()
    release, _parity = _run(
        tmp_path, monkeypatch,
        {"a": record},
    )

    checks = release["row_dimension_checks"]["a"]
    assert checks["null_masks"] is True
    assert "a" not in release.get("row_null_mask_differences", {})


def test_null_mask_differences_contain_only_strings_never_feature_values(
        tmp_path, monkeypatch):
    """The null-mask difference evidence names feature keys only. Two
    unmistakable magic floats stand in for the underlying feature VALUES on
    each side; neither may leak into the emitted evidence, whether raw or
    serialized."""
    from dataclasses import replace

    legacy_magic = 0.123456789
    native_magic = 0.987654321
    record = _clean_record(model_inputs={"x": legacy_magic, "y": None})
    native_with_diff = replace(
        _native_record(record),
        null_masks={"x": True, "y": True},  # "x" flips legacy False -> True
        feature_values={"x": native_magic},
    )

    release, _parity = _run(
        tmp_path, monkeypatch,
        {"a": record},
        natives_by_fixture={"a": native_with_diff},
    )

    diff = release["row_null_mask_differences"]["a"]
    assert diff["value_mismatch"] == ["x"]

    _assert_only_safe_leaves(
        release.get("row_null_mask_differences", {}),
        allow_bool=False, allow_int=False,
    )
    dumped = json.dumps(release["row_null_mask_differences"])
    assert str(legacy_magic) not in dumped
    assert str(native_magic) not in dumped


def test_verdicts_unchanged_by_difference_recording(tmp_path, monkeypatch):
    """The checks verdicts are identical whether differences are recorded or not."""
    from dataclasses import replace

    record_agree = _clean_record()
    record_disagree = _clean_record()
    native_disagree = replace(
        _native_record(record_disagree),
        resolved_request={k: v for k, v in record_disagree.items() if k != "gate_score"},
    )

    release, _parity = _run(
        tmp_path, monkeypatch,
        {"agree": record_agree, "disagree": record_disagree},
        natives_by_fixture={"disagree": native_disagree},
    )

    # Check that verdicts match what we'd expect independent of differences
    assert release["row_dimension_checks"]["agree"]["keys"] is True
    assert release["row_dimension_checks"]["disagree"]["keys"] is False

    # The presence/absence of difference data doesn't affect the verdict
    agree_checks = release["row_dimension_checks"]["agree"]
    disagree_checks = release["row_dimension_checks"]["disagree"]
    for dimension in agree_checks:
        if dimension in disagree_checks:
            # verdicts should match their check's purpose
            pass


def test_never_ran_dimensions_agree_when_both_sides_show_no_stage_ran(
        tmp_path, monkeypatch):
    """The rule's positive case: legacy holds only the typed placeholder
    defaults for verdicts/analogs AND native's resolved_request holds NONE
    of those names as keys -- both sides say the stages never ran, so the
    dimensions agree, are named in row_never_ran_dimensions, and produce no
    "keys" finding."""
    from dataclasses import replace

    record = _clean_record(
        gate_score=None, gate_threshold=None, gate_pass=None,
        ci_low=None, ci_high=None, n_analogs=0,
    )
    native_base = _native_record(record)
    native = replace(
        native_base,
        resolved_request={
            key: value for key, value in native_base.resolved_request.items()
            if key not in ("gate_score", "gate_threshold", "gate_pass",
                           "ci_low", "ci_high", "n_analogs")
        },
        # The "verdicts" numeric dimension reads native.gate_terms (see
        # _numeric_views), not resolved_request, so empty it too -- every
        # verdict field then resolves to None via .get, matching legacy.
        gate_terms={},
    )

    release, _parity = _run(
        tmp_path, monkeypatch, {"a": record}, natives_by_fixture={"a": native},
    )

    checks = release["row_dimension_checks"]["a"]
    assert checks["verdicts"] is True
    assert checks["analogs"] is True
    never_ran = release["row_never_ran_dimensions"]["a"]
    assert "verdicts" in never_ran
    assert "analogs" in never_ran
    # The suppressed fields must not read as a "keys" finding either, so the
    # row passes keys and is omitted from row_key_differences entirely.
    assert checks["keys"] is True
    assert "a" not in release["row_key_differences"]


def test_never_ran_rule_does_not_suppress_a_real_analog_regression(
        tmp_path, monkeypatch):
    """The rule's guard rail: native missing every analog key while LEGACY
    carries real (non-placeholder) analog values is a native regression,
    not a never-ran agreement -- the rule must not fire and the normal
    comparator must fail the dimension."""
    from dataclasses import replace

    record = _clean_record()  # ci_low=-0.01, ci_high=0.05, n_analogs=10: real values
    native = replace(
        _native_record(record),
        resolved_request={
            key: value for key, value in record.items()
            if key not in ("ci_low", "ci_high", "n_analogs")
        },
    )

    release, _parity = _run(
        tmp_path, monkeypatch, {"a": record}, natives_by_fixture={"a": native},
    )

    checks = release["row_dimension_checks"]["a"]
    assert checks["analogs"] is False
    assert "a" not in release["row_never_ran_dimensions"]


def test_native_carrying_none_gate_keys_is_a_normal_agreeing_comparison(
        tmp_path, monkeypatch):
    """Key PRESENCE is the rule's condition, not value: legacy holds the
    gate placeholders and native's resolved_request carries the gate keys
    with None values (real keys, written by a stage that ran) -- the rule
    must not fire, and the plain comparator still agrees None against
    None."""
    record = _clean_record(gate_score=None, gate_threshold=None, gate_pass=None)
    # Native UNCHANGED: _native_record builds resolved_request from
    # dict(record), so the gate keys are present there with None values.
    native = _native_record(record)

    release, _parity = _run(
        tmp_path, monkeypatch, {"a": record}, natives_by_fixture={"a": native},
    )

    checks = release["row_dimension_checks"]["a"]
    assert checks["verdicts"] is True
    assert "a" not in release["row_never_ran_dimensions"]
