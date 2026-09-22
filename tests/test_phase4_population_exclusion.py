"""Counted exclusion of ``research_replay`` pairs from the Phase 4 population.

User decision 2026-09-19: ``research_replay`` pairs have no v2 replay path and
are covered by the Tier-1 replay, so ``checks/phase4_real.py`` drops that one
named kind from ``expected`` and counts it under ``population.excluded``.
Every other kind stays expected; an unknown kind is still a gap.

Synthetic only: the per-pair replay is stubbed so a ``score_result`` pair
compares cleanly and anything without a stub trace is incomparable, as a
pair with no trace is in the real runner.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from checks import phase4_real

DIMENSIONS = ("keys", "contracts", "verdicts", "flags", "null_masks",
              "forecasts", "simulation", "financial_diagnostics", "analogs")


def _pair(fixture_id, kind):
    return {"fixture_id": fixture_id, "payload_hash": f"hash-{fixture_id}",
            "payload": {"record_kind": kind, "record": {"strategy": "STR-THRU"}}}


def _corpus(tmp_path, *pairs, manifest_kinds=None):
    manifest_kinds = manifest_kinds or {}
    return SimpleNamespace(
        root=tmp_path,
        index={"pairs": {
            p["fixture_id"]: {"record_kind": manifest_kinds.get(
                p["fixture_id"], p["payload"]["record_kind"])}
            for p in pairs}},
        ordered_ids=[p["fixture_id"] for p in pairs],
        pairs={p["fixture_id"]: p for p in pairs})


@pytest.fixture
def stub_replay(monkeypatch):
    """Only ``score_result`` pairs replay; every other kind raises, as an
    untraced pair does in the real verifier."""
    def verified(pair, _root):
        if pair["payload"]["record_kind"] != "score_result":
            raise phase4_real._TraceError("input_trace: missing")
        return {"same_input_receipt": "same", "trace_hash": "trace",
                "frozen_replay": None}

    def replayed(_verified):
        native = SimpleNamespace(reason_codes=(), payload_hash="native")
        receipts = tuple({"stage": s} for s in phase4_real._REQUIRED_TRACE_STAGES)
        return native, receipts, ()

    monkeypatch.setattr(phase4_real, "_verified_trace_bundle", verified)
    monkeypatch.setattr(phase4_real, "_replayed_member", replayed)
    monkeypatch.setattr(phase4_real, "_record_checks",
                        lambda _r, _n: ({d: True for d in DIMENSIONS}, {}, {}))
    empty_views = {d: {} for d in ("forecasts", "simulation", "financial_diagnostics",
                                   "verdicts", "analogs")}
    monkeypatch.setattr(phase4_real, "_numeric_views",
                        lambda _r, _n: (empty_views, empty_views))


def _dispositions(release):
    return {row["fixture_id"]: row for row in release["dispositions"]}


def test_the_exclusion_list_is_one_named_constant_with_research_replay_only():
    assert set(phase4_real.PHASE4_EXCLUDED_RECORD_KINDS) == {"research_replay"}
    reason = phase4_real.PHASE4_EXCLUDED_RECORD_KINDS["research_replay"]
    assert "no v2 replay path" in reason
    assert "2026-09-19" in reason
    assert "Tier-1" in reason


def test_research_replay_pairs_are_excluded_and_counted(tmp_path, stub_replay):
    corpus = _corpus(tmp_path, _pair("a", "score_result"),
                     _pair("r1", "research_replay"), _pair("r2", "research_replay"))
    release, parity = phase4_real._native_parity(corpus)
    for population in (release["population"], parity["population"]):
        assert population["expected"] == 1
        assert population["excluded"] == {"research_replay": 2}
        assert population["incomparable"] == 0
    rows = _dispositions(release)
    for fid in ("r1", "r2"):
        assert rows[fid]["disposition"] == "excluded"
        assert rows[fid]["reason"] == phase4_real.PHASE4_EXCLUDED_RECORD_KINDS[
            "research_replay"]
    assert rows["a"]["disposition"] == "compared"


def test_excluded_pairs_are_not_gaps_in_the_gate_verdict(
        tmp_path, stub_replay, monkeypatch):
    pairs = (_pair("a", "score_result"), _pair("r1", "research_replay"))
    release, parity = phase4_real._native_parity(_corpus(tmp_path, *pairs))
    assert release["complete"] is True and parity["complete"] is True
    assert release["population"]["compared"] == release["population"]["expected"]

    # Planted defect: without the exclusion the same pair is a gap.
    monkeypatch.setattr(phase4_real, "PHASE4_EXCLUDED_RECORD_KINDS", {})
    release, _parity = phase4_real._native_parity(_corpus(tmp_path, *pairs))
    assert release["complete"] is False
    assert release["population"]["expected"] == 2
    assert release["population"]["incomparable"] == 1
    assert release["population"]["excluded"] == {}


def test_an_unknown_kind_is_still_a_gap(tmp_path, stub_replay):
    corpus = _corpus(tmp_path, _pair("a", "score_result"), _pair("u", "mystery_kind"))
    release, _parity = phase4_real._native_parity(corpus)
    assert release["population"]["expected"] == 2
    assert release["population"]["excluded"] == {"research_replay": 0}
    assert _dispositions(release)["u"]["disposition"] == "incomparable"
    assert release["complete"] is False


def test_dyn_sv_choice_is_still_expected(tmp_path, stub_replay, monkeypatch):
    def no_chooser_trace(_pair, _root):
        raise phase4_real._TraceError("chooser trace missing")

    monkeypatch.setattr(phase4_real, "_replayed_chooser", no_chooser_trace)
    corpus = _corpus(tmp_path, _pair("a", "score_result"), _pair("d", "dyn_sv_choice"))
    release, _parity = phase4_real._native_parity(corpus)
    assert release["population"]["expected"] == 2
    assert _dispositions(release)["d"]["disposition"] == "incomparable"
    assert release["complete"] is False


def test_a_manifest_that_disagrees_on_the_kind_does_not_exclude(tmp_path, stub_replay):
    # The payload says research_replay but the bound manifest names another
    # kind: the pair is not excluded, so it stays expected (and is a gap).
    corpus = _corpus(tmp_path, _pair("a", "score_result"), _pair("r", "research_replay"),
                     manifest_kinds={"r": "score_result"})
    release, _parity = phase4_real._native_parity(corpus)
    assert release["population"]["expected"] == 2
    assert release["population"]["excluded"] == {"research_replay": 0}
    assert _dispositions(release)["r"]["disposition"] == "incomparable"
