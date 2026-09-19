"""P5-6 calibration fold keys: derived from a Phase 4 corpus or a nightly as-of.

Synthetic tmp_path corpora only; no real data.
"""
from __future__ import annotations

import json
from pathlib import Path

from tests.test_checks_phase5_acceptance import _release
from tools import phase5_calibration_keys as keys_tool


def _pair(fixture_id: str, *, strategy: str, alpha, cutoff=None, context=None,
          traced: bool = True) -> dict:
    # The tier-0 layout: payload.request is the legacy request; a traced
    # pair's canonical V2 request is input_trace.request.
    payload = {"request": {"strategy": strategy, "fill": {"alpha": alpha}},
               "record": {"strategy": strategy, "fill": alpha}}
    if cutoff is not None:
        payload["record"]["evidence_cutoff"] = cutoff
    if traced:
        payload["input_trace"] = {
            "request": {"strategy_version": strategy, "fill_model": {"alpha": alpha}},
            "native_inputs": {"context": context or {}},
        }
    return {"fixture_id": fixture_id, "payload": payload}


def _corpus(tmp_path: Path, pairs: list[dict]) -> Path:
    root = tmp_path / "corpus"
    (root / "pairs").mkdir(parents=True)
    for pair in pairs:
        (root / "pairs" / f"{pair['fixture_id']}.json").write_text(json.dumps(pair))
    (root / "INDEX.json").write_text("{}")
    return root


def test_corpus_keys_are_the_requests_own_causal_keys(tmp_path):
    corpus = _corpus(tmp_path, [
        _pair("a", strategy="STR-THRU", alpha=0.5, cutoff="2026-09-16"),
        _pair("b", strategy="STR-THRU", alpha=0.50000001, cutoff="2026-09-16"),
        _pair("c", strategy="STR-RUNUP", alpha=0.5,
              context={"as_of": "2026-09-16", "entry_date": "2026-08-27"}),
        _pair("d", strategy="CND-P", alpha=0.5, cutoff="2026-09-16"),
        _pair("e", strategy="STR-THRU", alpha=0.5, cutoff="2026-01-01", traced=False),
        _pair("f", strategy="STR-THRU", alpha=0.5),
    ])
    keys, underivable, traced = keys_tool.keys_from_corpus(corpus)

    assert traced == 5
    assert keys == {("STR-THRU", 0.5, "2026-09-16"), ("STR-RUNUP", 0.5, "2026-08-27"),
                    ("CND-P", 0.5, "2026-09-16")}
    assert underivable == {"f": "no cutoff"}
    by_member, uncatalogued = keys_tool.member_keys(keys)
    assert by_member["payoff_line:STR-THRU"] == [("STR-THRU", 0.5, "2026-09-16")]
    assert by_member["recalibration_map:STR-THRU"] == [("STR-THRU", 0.5, "2026-09-16")]
    assert by_member["payoff_surface:STR-RUNUP"] == [("STR-RUNUP", 0.5, "2026-08-27")]
    assert uncatalogued == [("CND-P", 0.5, "2026-09-16")]


def test_nightly_as_of_keys(tmp_path):
    keys = keys_tool.keys_from_as_of("2026-09-18", [0.5, 0.25], ["2026-08-29", "2026-09-01"])
    assert keys == {("STR-THRU", 0.5, "2026-09-18"), ("STR-THRU", 0.25, "2026-09-18"),
                    ("STR-RUNUP", 0.5, "2026-08-29"), ("STR-RUNUP", 0.5, "2026-09-01"),
                    ("STR-RUNUP", 0.25, "2026-08-29"), ("STR-RUNUP", 0.25, "2026-09-01")}


def test_commands_plan_only_first_one_job_per_recipe_and_alpha(tmp_path):
    keys = {("STR-THRU", 0.5, "2026-09-18"), ("STR-THRU", 0.5, "2026-09-17")}
    by_member, _ = keys_tool.member_keys(keys)
    commands = keys_tool.training_commands(by_member, tmp_path / "train")

    assert len(commands) == 4
    assert all(c.endswith("--plan-only") for c in commands[:2])
    assert not any(c.endswith("--plan-only") for c in commands[2:])
    line = commands[2]
    assert "--recipe payoff_line:STR-THRU:calibration --alpha 0.5" in line
    assert "--cutoff 2026-09-17 --cutoff 2026-09-18" in line
    assert "tools/bounded_run.py" in line


def test_release_coverage_names_the_missing_folds(tmp_path):
    # the fixture release stages the STR-THRU line at (0.5, None) and
    # (0.25, 2026-09-10), and the recalibration map at (0.5, None)
    root = _release(tmp_path)
    keys = {("STR-THRU", 0.25, "2026-09-10"), ("STR-THRU", 0.5, "2026-09-16")}
    result = keys_tool.plan(keys, train_root=tmp_path / "t", release_root=root)

    assert result["missing_from_release"]["payoff_line:STR-THRU"] == [
        ["STR-THRU", 0.5, "2026-09-16"]]
    assert result["missing_from_release"]["recalibration_map:STR-THRU"] == [
        ["STR-THRU", 0.25, "2026-09-10"], ["STR-THRU", 0.5, "2026-09-16"]]


def test_cli_writes_keys_outside_data(tmp_path, capsys):
    corpus = _corpus(tmp_path, [_pair("a", strategy="STR-THRU", alpha=0.5,
                                      cutoff="2026-09-16")])
    out = tmp_path / "keys.json"
    assert keys_tool.main(["--phase4-corpus", str(corpus), "--train-root",
                           str(tmp_path / "t"), "--out", str(out)]) == 0
    written = json.loads(out.read_text())
    assert written["keys"] == [["STR-THRU", 0.5, "2026-09-16"]]
    assert written["source"]["traced_pairs"] == 1
    assert "--plan-only" in capsys.readouterr().out


def test_traced_pair_key_reads_the_v2_request_from_the_input_trace():
    """payload.request is the legacy request; the key's strategy and alpha
    come from input_trace.request even when the record carries neither."""
    pair = _pair("p", strategy="STR-RUNUP", alpha=0.25, cutoff="2026-09-01")
    pair["payload"]["record"] = {"evidence_cutoff": "2026-09-01"}
    key, reason = keys_tool._pair_key(pair)
    assert (key, reason) == (keys_tool._key("STR-RUNUP", 0.25, "2026-09-01"), "")
    # Planted defect: the V2 request gone from its home, so the legacy
    # payload.request (no strategy_version/fill_model) cannot supply it.
    del pair["payload"]["input_trace"]["request"]
    assert keys_tool._pair_key(pair)[0] is None
