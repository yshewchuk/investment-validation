"""Bounded, value-bearing ``row_diagnostics`` added to ``_native_parity``.

``row_dimension_checks`` / ``row_numeric_findings`` / ``row_*_differences``
are value-free by construction, which is exactly why one full gate run could
not previously root-cause the few fields that disagreed. ``row_diagnostics``
adds the missing values -- contract leg/timeline diffs through the checker's
own ``_contract_projection``, failed numeric field names paired with the
legacy/native scalars from ``_numeric_views`` (including ``None``), the
chooser selection triple both sides, and bounded flag context -- but ONLY as
a separate map in ``saved_release_comparison``, never inside the ``rows``
whose ``content_hash`` IS ``comparison_receipt``, and never where any check,
control or gate decision reads it.

Since this revision the flag context also carries the exact ORDERED
normalized flag sequences the comparator compared
(``_diagnostic_flag_sequences``), so an order-only or duplicate-count
mismatch -- invisible to the value-free symmetric set diff -- is explained;
the mirror against ``_record_checks`` is pinned by
``test_flag_sequence_diagnostics_match_the_comparator_semantics``.

Fixture/helper shape follows ``tests/test_phase4_real_row_evidence.py``
(synthetic records driven through the real ``_native_parity`` orchestration,
only the replay helpers stubbed), which also keeps every existing
value-freedom assertion in that file running against the same rows these
diagnostics accompany.
"""
from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

from checks import phase4_real
from tests.test_phase4_real_row_evidence import (
    _clean_record, _corpus, _native_record, _pair, _run,
)


def _member_entry(release, fixture_id, member_index=0):
    return release["row_diagnostics"][fixture_id]["members"][str(member_index)]


# -- (1) contract diagnostics ---------------------------------------------------------


def test_contract_diagnostics_name_drifted_leg_attribute_and_timeline_date_with_both_values(
        tmp_path, monkeypatch):
    record = _clean_record()
    native = _native_record(record)
    drifted = replace(
        native,
        legs=tuple({**leg, "strike": float(leg["strike"]) + 2.5}
                   for leg in native.legs),
        entry_exit_plan={**native.entry_exit_plan, "entry_date": "2026-09-15"},
    )
    release, _parity = _run(
        tmp_path, monkeypatch, {"a": record}, natives_by_fixture={"a": drifted},
    )
    assert release["row_dimension_checks"]["a"]["contracts"] is False

    entry = _member_entry(release, "a")
    assert "contracts" in entry["failed_checks"]
    contracts = entry["contracts"]
    assert {"leg": 0, "attribute": "strike", "native": 102.5, "legacy": 100.0} \
        in contracts["legs"]["entries"]
    assert {"attribute": "entry_date", "native": "2026-09-15",
            "legacy": "2026-09-16"} in contracts["timeline"]
    # Same leg count on both sides: no count difference is invented.
    assert "leg_count" not in contracts


def test_contract_diagnostics_report_a_leg_count_difference_as_scalars(tmp_path, monkeypatch):
    first = {"name": "call", "right": "C", "side": "long", "quantity": 1.0,
             "strike": 100.0, "expiry": "2026-09-18", "fill": 1.5,
             "cash_flow": -150.0}
    second = {"name": "put", "right": "P", "side": "long", "quantity": 1.0,
              "strike": 95.0, "expiry": "2026-09-18", "fill": 1.25,
              "cash_flow": -125.0}
    record = _clean_record(legs=(first, second))
    native = _native_record(record)
    dropped = replace(native, legs=(native.legs[0],))
    release, _parity = _run(
        tmp_path, monkeypatch, {"a": record}, natives_by_fixture={"a": dropped},
    )
    assert release["row_dimension_checks"]["a"]["contracts"] is False

    contracts = _member_entry(release, "a")["contracts"]
    assert contracts["leg_count"] == {"native": 1, "legacy": 2}
    # The one aligned leg agrees: only the count is named, never a leg dump.
    assert "legs" not in contracts


def test_contract_leg_diffs_are_bounded(tmp_path, monkeypatch):
    legacy_legs = tuple({
        "name": f"leg{i}", "right": "C", "side": "long", "quantity": 1.0,
        "strike": 100.0 + i, "expiry": "2026-09-18", "fill": 1.0,
        "cash_flow": -100.0,
    } for i in range(5))
    record = _clean_record(legs=legacy_legs)
    native = _native_record(record)
    # Every one of the 8 fixed attributes differs on all 5 legs: 40 possible
    # diff entries, capped to 12 with the omitted count named.
    native_legs = tuple({
        "name": f"nleg{i}", "right": "P", "side": "short", "quantity": 2.0,
        "strike": 101.0 + i, "expiry": "2026-09-25", "fill": 2.0,
        "cash_flow": 200.0,
    } for i in range(5))
    release, _parity = _run(
        tmp_path, monkeypatch, {"a": record},
        natives_by_fixture={"a": replace(native, legs=native_legs)},
    )
    legs = _member_entry(release, "a")["contracts"]["legs"]
    assert len(legs["entries"]) == phase4_real._DIAGNOSTIC_LEG_DIFF_CAP
    assert legs["legs_omitted"] == 40 - phase4_real._DIAGNOSTIC_LEG_DIFF_CAP


def test_contract_diagnostics_report_nan_values_the_check_compared_unequal(
        tmp_path, monkeypatch):
    """Regression (Astra review blocker 2): two DISTINCT NaN objects fail
    the real ``contracts`` projection equality (NaN != NaN) even though both
    sides encode to one identical ``__nonfinite__`` tag. The diff walk
    compares raw projected values first and encodes only for display, so a
    failing contracts check can never diagnose itself as an empty diff."""
    legacy_leg = {"name": "call", "right": "C", "side": "long",
                  "quantity": 1.0, "strike": float("nan"),
                  "expiry": "2026-09-18", "fill": 1.5, "cash_flow": -150.0}
    record = _clean_record(legs=(legacy_leg,))
    native = _native_record(record)
    native_leg = dict(legacy_leg, strike=float("nan"))  # distinct NaN object
    release, _parity = _run(
        tmp_path, monkeypatch, {"a": record},
        natives_by_fixture={"a": replace(native, legs=(native_leg,))},
    )
    assert release["row_dimension_checks"]["a"]["contracts"] is False

    contracts = _member_entry(release, "a")["contracts"]
    assert contracts["legs"]["entries"] == [{
        "leg": 0, "attribute": "strike",
        "native": {"__nonfinite__": "nan"}, "legacy": {"__nonfinite__": "nan"},
    }]


def test_diagnostic_scalar_bounds_arbitrary_strings_with_a_deterministic_marker():
    """Regression (Astra review blocker 1): strings are display material and
    must never pass through ``_diagnostic_scalar`` unbounded."""
    limit = phase4_real._DIAGNOSTIC_TEXT_CHARS
    marker = "...[truncated]"
    bounded = phase4_real._diagnostic_scalar("q" * 1_000_000)
    assert bounded == "q" * limit + marker
    assert len(bounded) == limit + len(marker)
    # Short strings -- dates, strategy ids, leg names -- are untouched.
    assert phase4_real._diagnostic_scalar("2026-09-18") == "2026-09-18"
    assert phase4_real._diagnostic_scalar(None) is None
    assert phase4_real._diagnostic_scalar(True) is True
    assert phase4_real._diagnostic_scalar(7) == 7


def test_contract_diagnostics_never_emit_unbounded_corpus_text(tmp_path, monkeypatch):
    first = {"name": "call", "right": "C", "side": "long", "quantity": 1.0,
             "strike": 100.0, "expiry": "2026-09-18", "fill": 1.5,
             "cash_flow": -150.0}
    record = _clean_record(legs=(dict(first, name="L" * 5000 + "-tail"),))
    native = _native_record(record)
    long_name = "N" * 5000 + "-tail"
    release, _parity = _run(
        tmp_path, monkeypatch, {"a": record},
        natives_by_fixture={"a": replace(
            native, legs=(dict(first, name=long_name),))},
    )
    assert release["row_dimension_checks"]["a"]["contracts"] is False

    entries = _member_entry(release, "a")["contracts"]["legs"]["entries"]
    limit = phase4_real._DIAGNOSTIC_TEXT_CHARS
    assert entries == [{
        "leg": 0, "attribute": "name",
        "native": "N" * limit + "...[truncated]",
        "legacy": "L" * limit + "...[truncated]",
    }]
    dumped = json.dumps(release["row_diagnostics"], allow_nan=False)
    assert "L" * 5000 not in dumped
    assert "N" * 5000 not in dumped


# -- (2) numeric scalar diagnostics ----------------------------------------------------


def test_numeric_diagnostics_pair_failed_fields_with_legacy_native_scalars(
        tmp_path, monkeypatch):
    """The few failed numeric fields now carry BOTH values -- including
    ``None`` (a native null versus a legacy number is itself a root cause)
    and non-finite floats (tagged, never a bare JSON-invalid ``NaN``)."""
    record = _clean_record()
    native = _native_record(record)
    corrupted = replace(
        native,
        forecasts={**native.forecasts, "exp_pnl_sim": float("inf")},
        gate_terms={"gate_score": 0.8, "gate_threshold": None,
                    "gate_pass": True},
    )
    release, _parity = _run(
        tmp_path, monkeypatch, {"a": record}, natives_by_fixture={"a": corrupted},
    )
    checks = release["row_dimension_checks"]["a"]
    assert checks["simulation"] is False and checks["verdicts"] is False

    numeric = _member_entry(release, "a")["numeric"]
    # Only the dimensions that produced findings appear -- agreeing
    # dimensions carry no diagnostic bulk.
    assert sorted(numeric) == ["simulation", "verdicts"]
    simulation = {item["field"]: item for item in numeric["simulation"]["fields"]}
    assert simulation["exp_pnl_sim"]["legacy"] == 0.2
    assert simulation["exp_pnl_sim"]["native"] == {"__nonfinite__": "inf"}
    verdicts = {item["field"]: item for item in numeric["verdicts"]["fields"]}
    assert verdicts["gate_threshold"]["legacy"] == 0.5
    assert verdicts["gate_threshold"]["native"] is None

    # Deterministic strict-JSON serialization of the whole diagnostics map.
    dumped = json.dumps(release["row_diagnostics"], allow_nan=False, sort_keys=True)
    assert json.loads(dumped) == release["row_diagnostics"]


# -- (3/4) flag context and chooser selection ------------------------------------------


def test_flag_failure_diagnostics_carry_bounded_detail_and_warnings(tmp_path, monkeypatch):
    record = _clean_record(flags=("BAD_QUOTE",), detail="refusal detail " * 40)
    native = replace(
        _native_record(record),
        reason_codes=("BAD_QUOTE", "NATIVE_ONLY"),
        warnings=tuple(f"WARN-{index}" for index in range(12)),
    )
    release, _parity = _run(
        tmp_path, monkeypatch, {"a": record}, natives_by_fixture={"a": native},
    )
    assert release["row_dimension_checks"]["a"]["flags"] is False

    flags = _member_entry(release, "a")["flags"]
    assert flags["legacy_detail"].endswith("...[truncated]")
    assert flags["legacy_detail_truncated"] is True
    assert len(flags["legacy_detail"]) <= (
        phase4_real._DIAGNOSTIC_TEXT_CHARS + len("...[truncated]"))
    assert flags["native_warnings"] == [f"WARN-{i}" for i in range(8)]
    assert flags["native_warnings_omitted"] == 4


def test_flag_failure_without_detail_or_warnings_reports_bounded_sequences(
        tmp_path, monkeypatch):
    """A failed flags check is now ALWAYS explainable: even with no
    ``detail`` and no ``warnings`` to quote, the flag section carries both
    compared ORDERED sequences (the old no-section behavior suppressed
    exactly the evidence fixtures like phase4-observational-13's 009/015
    needed -- failed check, empty symmetric set diffs)."""
    record = _clean_record(flags=("BAD_QUOTE",))
    native = replace(_native_record(record), reason_codes=("OTHER_REFUSAL",))
    release, _parity = _run(
        tmp_path, monkeypatch, {"a": record}, natives_by_fixture={"a": native},
    )
    entry = _member_entry(release, "a")
    assert "flags" in entry["failed_checks"]
    flags = entry["flags"]
    assert flags["legacy_flags"] == ["BAD_QUOTE"]
    assert flags["native_flags"] == ["OTHER_REFUSAL"]
    # Nothing is invented: the absent context fields simply do not appear.
    assert "legacy_detail" not in flags
    assert "native_warnings" not in flags


def test_flag_failure_without_detail_or_warnings_adds_no_flag_section(
        tmp_path, monkeypatch):
    """A PASSING flags check adds no flag section -- sequences included,
    nothing is invented where the check did not fail (this row fails only
    ``contracts``). The name is retained verbatim from the pre-sequences
    revision, when it proved a FAILED flags check without detail/warnings
    got no section at all; that scenario now earns the ordered-sequence
    section above it, and this line's assertion moved here to keep proving
    the section appears only alongside a failed flags check."""
    record = _clean_record()
    base = _native_record(record)
    drifted = replace(base, legs=tuple(
        {**leg, "strike": float(leg["strike"]) + 2.5} for leg in base.legs))
    release, _parity = _run(
        tmp_path, monkeypatch, {"a": record}, natives_by_fixture={"a": drifted},
    )
    entry = _member_entry(release, "a")
    assert "flags" not in entry["failed_checks"]
    assert "flags" not in entry  # nothing available to report, nothing invented


def test_flag_sequence_diagnostics_match_the_comparator_semantics():
    """The section's sequences come from ``_diagnostic_flag_sequences``, a
    diagnostics-only MIRROR of ``_record_checks``' normalization (kept a
    separate function only so the landed comparator source stays
    untouched). This pins the mirror against the REAL comparator across
    every translation branch: the LAYER_DISAGREE strip, order-only and
    multiplicity-only mismatch, CAL-P/CND-P append, and the NO_SCORE
    expectation fired, required-but-missing, suppressed by a real refusal,
    suppressed by the synthetic refusal (so append ORDER matters), and not
    fired on rows legacy scores -- where a native NO_SCORE stays a finding.
    The invariants proved per case: a reported sequence difference never
    accompanies a PASSED verdict, and with no truncation the verdict is
    exactly sequence equality (no reported difference hides a failure); the
    value-free symmetric set diff equals the set arithmetic of the reported
    sequences whenever the whole sequences are visible."""
    def case(record_overrides, native_reason_codes, *, expect):
        record = _clean_record(**record_overrides)
        native = replace(
            _native_record(record), reason_codes=native_reason_codes)
        checks, _numeric, differences = phase4_real._record_checks(record, native)
        sequences = phase4_real._diagnostic_flag_sequences(record, native)
        legacy_seq, native_seq = sequences["legacy_flags"], sequences["native_flags"]
        truncated = ("legacy_flags_omitted" in sequences
                     or "native_flags_omitted" in sequences)
        assert checks["flags"] is expect, record_overrides
        if legacy_seq != native_seq:
            assert checks["flags"] is False, record_overrides
        elif not truncated:
            assert checks["flags"] is True, record_overrides
        if checks["flags"]:
            assert differences["flag_differences"] == {}
        elif not truncated:
            assert differences["flag_differences"] == {
                "native_only": sorted(set(native_seq) - set(legacy_seq)),
                "legacy_only": sorted(set(legacy_seq) - set(native_seq)),
            }, record_overrides
        for side in ("legacy", "native"):
            key = f"{side}_flags"
            assert len(sequences[key]) <= phase4_real._DIAGNOSTIC_FLAG_SEQ_CAP
            assert (f"{key}_omitted" in sequences) is (
                len(sequences[key]) == phase4_real._DIAGNOSTIC_FLAG_SEQ_CAP)

    # Membership agreement and mismatch.
    case({"flags": ("FLAG_A",)}, ("FLAG_A",), expect=True)
    case({"flags": ("FLAG_A",)}, ("FLAG_B",), expect=False)
    # Order-only and multiplicity-only mismatch: sets agree, tuples differ
    # (the fixtures-009/015 blind spot) -- and the sequences show it.
    case({"flags": ("FLAG_A", "FLAG_B")}, ("FLAG_B", "FLAG_A"), expect=False)
    case({"flags": ("FLAG_A", "FLAG_A", "FLAG_B")}, ("FLAG_A", "FLAG_B"),
         expect=False)
    # The advisory strip applies to BOTH sides before comparison.
    case({"flags": ("LAYER_DISAGREE", "FLAG_A")},
         ("LAYER_DISAGREE", "FLAG_A"), expect=True)
    case({"exp_pnl_model": None, "flags": ("LAYER_DISAGREE",)},
         ("NO_SCORE",), expect=True)
    # CAL-P/CND-P synthetic UNVALIDATED_STRUCTURE: appended when missing,
    # never duplicated when already present.
    case({"strategy": "CAL-P"}, (), expect=False)
    case({"strategy": "CND-P"}, (), expect=False)
    case({"strategy": "CAL-P", "flags": ("UNVALIDATED_STRUCTURE",)},
         ("UNVALIDATED_STRUCTURE",), expect=True)
    # NO_SCORE: expected exactly once on unscored unrefusing legacy rows,
    # required even when native omits it, suppressed by a real refusal and
    # by the synthetic refusal alike, never excused on scored rows (0.0 IS
    # a score; an analog score is a score; exp_pnl_sim is neither).
    case({"exp_pnl_model": None}, ("NO_SCORE",), expect=True)
    case({"exp_pnl_model": None}, (), expect=False)
    case({"exp_pnl_model": None, "flags": ("BAD_QUOTE",)}, ("BAD_QUOTE",),
         expect=True)
    case({"strategy": "CAL-P", "exp_pnl_model": None},
         ("UNVALIDATED_STRUCTURE",), expect=True)
    case({}, ("NO_SCORE",), expect=False)
    case({"exp_pnl_model": 0.0}, ("NO_SCORE",), expect=False)
    case({"exp_pnl_model": None, "exp_pnl_analog": -0.05}, ("NO_SCORE",),
         expect=False)
    # Cap bookkeeping: sequences never exceed the cap, and the omitted tail
    # is counted on each side independently.
    long_flags = tuple(f"DIAG_FLAG_{index:02d}"
                       for index in range(phase4_real._DIAGNOSTIC_FLAG_SEQ_CAP + 4))
    case({"flags": long_flags}, long_flags, expect=True)
    case({"flags": long_flags}, long_flags[:-1], expect=False)


def test_flag_sequence_diagnostics_explain_a_same_set_different_order_failure(
        tmp_path, monkeypatch):
    """The observability gap this fixes: the comparator compares ORDERED
    tuples while ``row_flag_differences`` is a symmetric SET diff, so an
    order-only mismatch failed the check while explaining nothing (both
    difference lists empty). The flag section now carries the exact
    sequences that were compared, in order, so the mismatch is visible."""
    record = _clean_record(flags=("FLAG_A", "FLAG_B"))
    native = replace(_native_record(record), reason_codes=("FLAG_B", "FLAG_A"))
    release, _parity = _run(
        tmp_path, monkeypatch, {"a": record}, natives_by_fixture={"a": native},
    )
    assert release["row_dimension_checks"]["a"]["flags"] is False
    diff = release["row_flag_differences"]["a"]
    assert diff == {"native_only": [], "legacy_only": []}

    flags = _member_entry(release, "a")["flags"]
    assert flags["legacy_flags"] == ["FLAG_A", "FLAG_B"]
    assert flags["native_flags"] == ["FLAG_B", "FLAG_A"]


def test_flag_sequence_diagnostics_explain_a_duplicate_count_failure(
        tmp_path, monkeypatch):
    """Same set, same order of distinct names, different multiplicity: also
    invisible to the set diff, explicit in the compared sequences."""
    record = _clean_record(flags=("FLAG_A", "FLAG_A", "FLAG_B"))
    native = replace(
        _native_record(record), reason_codes=("FLAG_A", "FLAG_B"))
    release, _parity = _run(
        tmp_path, monkeypatch, {"a": record}, natives_by_fixture={"a": native},
    )
    assert release["row_dimension_checks"]["a"]["flags"] is False
    diff = release["row_flag_differences"]["a"]
    assert diff == {"native_only": [], "legacy_only": []}

    flags = _member_entry(release, "a")["flags"]
    assert flags["legacy_flags"] == ["FLAG_A", "FLAG_A", "FLAG_B"]
    assert flags["native_flags"] == ["FLAG_A", "FLAG_B"]


def test_flag_sequence_diagnostics_report_the_translated_tuples_not_raw_flags(
        tmp_path, monkeypatch):
    """The reported sequences are the comparator's NORMALIZED tuples: the
    CAL-P/CND-P UNVALIDATED_STRUCTURE and the unscored-row NO_SCORE
    translations appear even though the raw legacy flags omit them -- the
    diagnostic cannot contradict the verdict because it reuses the exact
    function the verdict was decided with."""
    cal_record = _clean_record(strategy="CAL-P", exp_pnl_model=None)
    cal_native = replace(_native_record(cal_record), reason_codes=())
    no_score_record = _clean_record(exp_pnl_model=None)
    no_score_native = replace(_native_record(no_score_record), reason_codes=())
    release, _parity = _run(
        tmp_path, monkeypatch,
        {"cal": cal_record, "nos": no_score_record},
        natives_by_fixture={"cal": cal_native, "nos": no_score_native},
    )
    assert cal_record["flags"] == ()
    assert no_score_record["flags"] == ()

    cal = _member_entry(release, "cal")["flags"]
    assert cal["legacy_flags"] == ["UNVALIDATED_STRUCTURE"]
    assert cal["native_flags"] == []
    assert release["row_flag_differences"]["cal"]["legacy_only"] == [
        "UNVALIDATED_STRUCTURE"]

    nos = _member_entry(release, "nos")["flags"]
    assert nos["legacy_flags"] == ["NO_SCORE"]
    assert nos["native_flags"] == []
    assert release["row_flag_differences"]["nos"]["legacy_only"] == ["NO_SCORE"]


def test_flag_sequence_diagnostics_are_bounded_with_omitted_counts(
        tmp_path, monkeypatch):
    """A pathological row cannot grow the report without limit: each side
    keeps its first ``_DIAGNOSTIC_FLAG_SEQ_CAP`` names IN ORDER and the
    truncated tail is counted, never silently dropped."""
    cap = phase4_real._DIAGNOSTIC_FLAG_SEQ_CAP
    legacy_flags = tuple(f"FLAG_A_{index:02d}" for index in range(cap + 5))
    native_flags = tuple(f"FLAG_B_{index:02d}" for index in range(cap + 9))
    record = _clean_record(flags=legacy_flags)
    native = replace(_native_record(record), reason_codes=native_flags)
    release, _parity = _run(
        tmp_path, monkeypatch, {"a": record}, natives_by_fixture={"a": native},
    )
    assert release["row_dimension_checks"]["a"]["flags"] is False

    flags = _member_entry(release, "a")["flags"]
    assert flags["legacy_flags"] == list(legacy_flags[:cap])
    assert flags["legacy_flags_omitted"] == 5
    assert flags["native_flags"] == list(native_flags[:cap])
    assert flags["native_flags_omitted"] == 9
    # Bounded serialization of the whole section.
    dumped = json.dumps(flags, allow_nan=False)
    assert json.loads(dumped) == flags


def _chooser_members_and_choice():
    """(members, choice, summary record) for a 3-member row where ONLY ranked
    members 0 and 2 disagree (sparse), plus a drifted chosen_margin."""
    member_zero = _clean_record()
    native_zero = _native_record(member_zero)
    drifted_zero = replace(
        native_zero,
        legs=tuple({**leg, "strike": float(leg["strike"]) + 1.0}
                   for leg in native_zero.legs))
    member_one = _clean_record()
    member_two = _clean_record()
    native_two = _native_record(member_two)
    corrupted_two = replace(
        native_two,
        forecasts={**native_two.forecasts, "exp_pnl_model": 7.777777},
    )
    members = [(member_zero, drifted_zero), (member_one, _native_record(member_one)),
               (member_two, corrupted_two)]
    choice = replace(_native_record(_clean_record()), chooser_selection={
        "strategy": "STR-THRU", "menu_size": 3, "margin": 0.1,
    })
    summary = {"strategy": "DYN-SV", "chosen_strategy": "STR-THRU",
               "menu_size": 3, "chosen_margin": 0.2, "flags": ()}
    return members, choice, summary


def _run_chooser(tmp_path, monkeypatch, members, choice, summary, fixture_id="dyn"):
    pair = {
        "fixture_id": fixture_id, "payload_hash": f"hash-{fixture_id}",
        "payload": {"record_kind": "dyn_sv_choice", "record": summary,
                    "input_trace_hash": "chooser-trace-hash"},
    }
    corpus = SimpleNamespace(
        root=tmp_path,
        index={"pairs": {fixture_id: {"record_kind": "dyn_sv_choice"}}},
        ordered_ids=[fixture_id], pairs={fixture_id: pair},
    )

    def replayed(_pair, _root):
        out = []
        for index, (member_record, member_native) in enumerate(members):
            verified = {"same_input_receipt": f"same-{index}",
                        "trace_hash": f"trace-{index}", "frozen_replay": None,
                        "request": SimpleNamespace(dependency_refs=())}
            receipts = tuple({"stage": stage}
                             for stage in phase4_real._REQUIRED_TRACE_STAGES)
            out.append((member_record, verified, member_native, receipts, ()))
        return out, choice

    monkeypatch.setattr(phase4_real, "_replayed_chooser", replayed)
    return phase4_real._native_parity(corpus)


def test_chooser_diagnostics_keep_original_member_indices_and_selection_values(
        tmp_path, monkeypatch):
    members, choice, summary = _chooser_members_and_choice()
    release, _parity = _run_chooser(tmp_path, monkeypatch, members, choice, summary)

    diagnostics = release["row_diagnostics"]["dyn"]
    # Sparse failures keep their ORIGINAL member indices (0 and 2 of 0,1,2),
    # never renumbered positions of a filtered list.
    assert sorted(diagnostics["members"]) == ["0", "2"]
    zero = diagnostics["members"]["0"]
    assert zero["failed_checks"] == ["contracts"]
    assert zero["contracts"]["legs"]["entries"][0]["native"] == 101.0
    two = diagnostics["members"]["2"]
    assert {item["field"] for item in two["numeric"]["simulation"]["fields"]} == {
        "exp_pnl_model"}
    assert two["numeric"]["simulation"]["fields"][0]["native"] == 7.777777
    assert two["numeric"]["simulation"]["fields"][0]["legacy"] == 0.15

    chooser = diagnostics["chooser"]
    assert chooser["failed"] == ["chosen_margin"]
    assert chooser["legacy"] == {"chosen_strategy": "STR-THRU", "menu_size": 3,
                                 "margin": 0.2}
    assert chooser["native"] == {"chosen_strategy": "STR-THRU", "menu_size": 3,
                                 "margin": 0.1}


def test_chooser_diagnostics_stay_bounded_with_many_failing_members(tmp_path, monkeypatch):
    members = []
    for index in range(14):
        member_record = _clean_record()
        member_native = _native_record(member_record)
        members.append((member_record, replace(
            member_native,
            legs=tuple({**leg, "strike": float(leg["strike"]) + 1.0}
                       for leg in member_native.legs))))
    choice = replace(_native_record(_clean_record()), chooser_selection={
        "strategy": "STR-THRU", "menu_size": 14, "margin": 0.1,
    })
    summary = {"strategy": "DYN-SV", "chosen_strategy": "STR-THRU",
               "menu_size": 14, "chosen_margin": 0.1, "flags": ()}
    release, _parity = _run_chooser(tmp_path, monkeypatch, members, choice, summary)

    diagnostics = release["row_diagnostics"]["dyn"]
    assert sorted(diagnostics["members"], key=int) == [
        str(index) for index in range(phase4_real._DIAGNOSTIC_MEMBER_CAP)]
    assert diagnostics["members_omitted"] == 14 - phase4_real._DIAGNOSTIC_MEMBER_CAP
    # All three selection checks agreed here: no chooser section is invented.
    assert "chooser" not in diagnostics


# -- invariance: diagnostics never touch decisions, receipts or reasons ---------------


def test_row_diagnostics_change_no_receipt_check_control_or_status(tmp_path, monkeypatch):
    """Run the same corpus twice: once with the real diagnostics, once with
    the diagnostics builder stubbed out. Every hashed, decided or controlled
    value must be byte-identical -- the diagnostics are evidence about the
    verdicts, never an input to them."""
    agreeing = _clean_record()
    disagreeing = _clean_record()
    native = _native_record(disagreeing)
    corrupted = replace(
        native, forecasts={**native.forecasts, "exp_pnl_sim": 1.23456789})
    records = {"ok": agreeing, "bad": disagreeing}
    natives = {"bad": corrupted}

    release_with, parity_with = _run(tmp_path, monkeypatch, records,
                                     natives_by_fixture=natives)
    assert release_with["row_diagnostics"]["bad"]["members"]["0"]["numeric"]
    assert "ok" not in release_with["row_diagnostics"]  # passing rows: no entry

    monkeypatch.setattr(phase4_real, "_row_diagnostics",
                        lambda *args, **kwargs: {"members": {}})
    release_without, parity_without = _run(tmp_path, monkeypatch, records,
                                           natives_by_fixture=natives)

    assert release_with["row_diagnostics"] != release_without["row_diagnostics"]
    for key in ("comparison_receipt", "native_execution_receipt",
                "legacy_execution_receipt", "row_dimension_checks",
                "row_numeric_findings", "row_key_differences",
                "row_flag_differences", "row_null_mask_differences",
                "row_never_ran_dimensions", "dimension_agreement",
                "dimension_rollup", "population", "numeric_coverage",
                "numeric_negative_controls", "complete"):
        assert release_with[key] == release_without[key], key
    for key in ("complete", "same_input_hashes", "dimension_agreement",
                "population", "stages", "stage_coverage"):
        assert parity_with[key] == parity_without[key], key
    assert (parity_with["planted_defect"]["receipt"]
            == parity_without["planted_defect"]["receipt"])


def test_incomparable_rows_keep_their_typed_reasons_and_are_never_diagnosed(
        tmp_path, monkeypatch):
    record = _clean_record()
    corpus = _corpus(tmp_path, _pair("a", record))

    def broken_verified(_pair, _root):
        raise ValueError("trace refuses to verify")

    monkeypatch.setattr(phase4_real, "_verified_trace_bundle", broken_verified)
    release, _parity = phase4_real._native_parity(corpus)

    row = next(row for row in release["dispositions"]
               if row["fixture_id"] == "a")
    assert row["disposition"] == "incomparable"
    assert row["reason"] == "ValueError: trace refuses to verify"
    assert "a" not in release["row_diagnostics"]
