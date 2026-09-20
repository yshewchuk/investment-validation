"""The tier-0 corpus check, and its negative controls.

`component_contracts.md` §9.5 and §15.3 define what the check must prove; each
case in :mod:`checks.tier0_corpus` is proved here by corrupting a synthetic
corpus and asserting the case goes red, and each coverage axis has a crafted
positive and negative record.

A synthetic corpus rather than the real one, deliberately: the real corpus
carries licensed quotes and is not in this repository, so a test that needed
it would not run on a clean checkout. The real corpus is exercised by
``checks/rearchitecture_phase0_gate.py``, which reports honestly when it is
absent.
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks import tier0_corpus as t0  # noqa: E402
from engine.v2.diagnosis import AGREE, DIFFER, INCOMPARABLE, content_hash  # noqa: E402

MENU = ["TWIN-P5", "BFLY-P", "CND-PS"]
AXIS_INPUTS = {
    "structures": ["BFLY-P", "CAL-P", "CND-PS", "STR-RUNUP", "STR-THRU", "TWIN-P5"],
    "dynamic_strategy": "DYN-SV",
    "menu": MENU,
    "disabled": ["CAL-P"],
    "model_roles": ["size", "implied_t1", "runup_move", "iv_crush", "gate", "chooser"],
    "refusal_code_mapping": {code: code for code in (
        "UNVALIDATED_STRUCTURE", "NO_CHAIN", "BAD_QUOTE", "COARSE_LADDER", "NO_FORECAST")},
}
LEGS = [{"name": "dn1", "strike": 240.0, "qty": 1},
        {"name": "atm", "strike": 250.0, "qty": -2},
        {"name": "up1", "strike": 260.0, "qty": 1}]
WIDTH = 0.05123456789012


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------


def request(strategy: str, **fields) -> dict:
    base = {"ticker": "MTN", "strategy": strategy, "event_date": "2026-09-28",
            "session": "AMC", "structure_params": None, "strike": None,
            "fill": {"policy_id": "legacy.fill_alpha.v1", "alpha": 0.5}}
    base.update(fields)
    base["identity_key"] = f"MTN|{strategy}|{json.dumps(fields, sort_keys=True)}"
    return base


def priced(strategy: str, width: float = WIDTH, **fields) -> dict:
    record = {"ticker": "MTN", "strategy": strategy, "event_date": "2026-09-28",
              "session": "AMC", "entry_date": "2026-09-28", "exit_date": "2026-09-29",
              "legs": copy.deepcopy(LEGS), "entry_cost": 3.45, "spot": 249.88,
              "forecast_abs_move": 5.218743916, "forecast_p10": 2.91,
              "forecast_p90": 8.0, "forecast_sd": 1.99, "forecast_model": "size_v1_4",
              "forecast_fold": "2025-01-01", "ci_low": -0.01193117,
              "ci_high": 0.04217742, "structure_params": {"width_moneyness": width},
              "flags": []}
    record.update(fields)
    return record


def refusal(strategy: str, flag: str, **fields) -> dict:
    record = {"ticker": "MTN", "strategy": strategy, "event_date": "2026-09-28",
              "session": "BMO", "structure_params": {}, "legs": [], "flags": [flag]}
    record.update(fields)
    return record


def pair(fid: str, req: dict, rec: dict, kind: str = "score_result",
         relations: dict | None = None) -> dict:
    payload = {"request": req, "record": rec, "record_kind": kind}
    if relations:
        payload["relations"] = relations
    return {
        "schema_version": "tier0_pair.v1.1",
        "fixture_id": fid,
        "covers": t0.derive_covers(rec, req, kind, AXIS_INPUTS, relations),
        "notes": "",
        "payload": payload,
        "payload_hash": content_hash(payload),
        "request_hash": content_hash(req),
        "envelope": {"captured_at": "2026-09-12T00:00:00.000000+00:00",
                     "worker_ref": "test:1", "duration_seconds": 0.01},
    }


def standard_pairs() -> list[dict]:
    """Six pairs that together carry every seeded control and a pinned relation."""
    selector = request("TWIN-P5")
    return [
        pair("000_TWIN-P5", selector, priced("TWIN-P5")),
        pair("001_TWIN-P5-pinned", request("TWIN-P5", structure_params={"width_moneyness": WIDTH}),
             priced("TWIN-P5"), relations={"pinned_from": content_hash(selector)}),
        pair("002_STR-THRU", request("STR-THRU", strike=14.7615), refusal("STR-THRU", "NO_CHAIN")),
        pair("003_CAL-P", request("CAL-P"), refusal("CAL-P", "UNVALIDATED_STRUCTURE")),
        pair("004_BFLY-P", request("BFLY-P", strike=250.0), priced("BFLY-P", 0.061246010132397256)),
        pair("005_CND-PS", request("CND-PS"), priced("CND-PS", 0.0234567891234)),
    ]


def build(root: Path, pairs: list[dict]) -> Path:
    (root / "pairs").mkdir(parents=True, exist_ok=True)
    for old in (root / "pairs").glob("*.json"):
        old.unlink()
    coverage: dict[str, list[str]] = {}
    for p in pairs:
        (root / "pairs" / f"{p['fixture_id']}.json").write_text(
            json.dumps(p, indent=2, sort_keys=True) + "\n")
        for axis in p["covers"]:
            coverage.setdefault(axis, []).append(p["fixture_id"])
    (root / "INDEX.json").write_text(json.dumps({
        "schema_version": "tier0_corpus.v1.1",
        "pairs": {p["fixture_id"]: {"payload_hash": p["payload_hash"],
                                    "request_hash": p["request_hash"],
                                    "record_kind": p["payload"]["record_kind"],
                                    "covers": p["covers"]} for p in pairs},
        "axis_inputs": AXIS_INPUTS,
        "refusal_code_mapping": AXIS_INPUTS["refusal_code_mapping"],
        "coverage": coverage,
        "required_axes": sorted(coverage),
        "uncovered_axes": [],
        "corpus_hash": content_hash({p["fixture_id"]: p["payload_hash"] for p in pairs}),
    }, indent=2, sort_keys=True) + "\n")
    return root


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    return build(tmp_path / "tier0", standard_pairs())


def _pair_path(root: Path, fixture_id: str) -> Path:
    return root / "pairs" / f"{fixture_id}.json"


def _rewrite(root: Path, fixture_id: str, mutate) -> None:
    """Edit a pair file WITHOUT re-hashing it — a tampered or corrupted file."""
    path = _pair_path(root, fixture_id)
    doc = json.loads(path.read_text())
    mutate(doc)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")


def covers(record: dict, req: dict | None = None, kind: str = "score_result",
           relations: dict | None = None) -> set[str]:
    return set(t0.derive_covers(record, req or request(record["strategy"]), kind,
                                AXIS_INPUTS, relations))


# --------------------------------------------------------------------------
# a healthy corpus
# --------------------------------------------------------------------------


def test_a_healthy_corpus_agrees(corpus):
    merged, cases = t0.run(corpus)
    assert merged.verdict == AGREE, merged.summary()
    assert set(cases) == {"manifest", "corpus_replay", "coverage", "pinned_counterparts",
                          "seeded_controls", "batch_vs_single", "fresh_process"}
    assert all(r.verdict == AGREE for r in cases.values())


def test_it_runs_well_inside_the_ten_second_budget(corpus):
    started = time.monotonic()
    t0.run(corpus)
    assert time.monotonic() - started < t0.TIME_BUDGET_SECONDS


def test_it_loads_no_panel_and_opens_no_socket(corpus):
    """§7.2: no network, no panel load, no fitting on replay."""
    script = (
        "import sys, json; sys.path.insert(0, %r)\n"
        "from pathlib import Path\n"
        "from checks import tier0_corpus as t0\n"
        "merged, _ = t0.run(Path(%r))\n"
        "print(json.dumps({'verdict': merged.verdict,\n"
        "  'pandas': 'pandas' in sys.modules,\n"
        "  'score': 'engine.score' in sys.modules}))\n"
    ) % (str(ROOT), str(corpus))
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True,
                          text=True, cwd=ROOT, check=True)
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out == {"verdict": AGREE, "pandas": False, "score": False}


def test_the_network_guard_actually_refuses():
    script = (
        "import sys; sys.path.insert(0, %r)\n"
        "from checks import tier0_corpus as t0\n"
        "t0._forbid_network()\n"
        "import socket\n"
        "try:\n"
        "    socket.socket()\n"
        "except Exception as exc:\n"
        "    print(type(exc).__name__)\n"
    ) % str(ROOT)
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True,
                          text=True, cwd=ROOT, check=True)
    assert "_NetworkUsed" in proc.stdout


# --------------------------------------------------------------------------
# integrity: addressing, digest, manifest
# --------------------------------------------------------------------------


def test_a_rounded_request_stops_addressing_its_record(corpus):
    """`b33036c`: the lookup key is the hash of the FULL-PRECISION request."""
    _rewrite(corpus, "000_TWIN-P5", lambda d: d["payload"]["request"].update(
        {"identity_key": d["payload"]["request"]["identity_key"] + "-rounded"}))
    merged, cases = t0.run(corpus)
    assert merged.verdict == DIFFER
    assert cases["corpus_replay"].verdict == DIFFER
    assert {"request_hash", "resolves_to[0]"} & {f.field_path for f in merged.findings}


def test_a_file_that_disagrees_with_its_digest_fails(corpus):
    """`6b9d5cf`: rounding reapplied after the exemption."""
    _rewrite(corpus, "004_BFLY-P", lambda d: d["payload"]["record"]["structure_params"].update(
        {"width_moneyness": 0.061246}))
    merged, _ = t0.run(corpus)
    assert merged.verdict == DIFFER
    assert any(f.field_path == "payload_hash" for f in merged.findings)


def test_an_empty_corpus_is_incomparable_not_agreement(tmp_path):
    root = tmp_path / "tier0"
    (root / "pairs").mkdir(parents=True)
    (root / "INDEX.json").write_text(json.dumps(
        {"pairs": {}, "coverage": {}, "required_axes": [], "uncovered_axes": []}))
    merged, _ = t0.run(root)
    assert merged.verdict == INCOMPARABLE


def test_a_missing_corpus_exits_nonzero_rather_than_passing(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(ROOT / "checks" / "tier0_corpus.py"),
         "--corpus", str(tmp_path / "nothing")],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert proc.returncode == 1
    assert "CORPUS_MISSING" in proc.stderr or "INCOMPARABLE" in proc.stderr


def test_batch_and_single_agree_on_a_corrupted_corpus_too(corpus):
    """The batch must not hide a finding the singles would have reported."""
    _rewrite(corpus, "003_CAL-P", lambda d: d["payload"]["record"].update({"ci_low": -9.9}))
    merged, cases = t0.run(corpus)
    assert cases["batch_vs_single"].verdict == AGREE
    assert merged.verdict == DIFFER


def test_deleting_fixtures_and_keeping_the_index_is_not_a_pass(corpus):
    """The 2026-09-12 review probe: keep one file, leave the index alone."""
    keep = "000_TWIN-P5"
    deleted = []
    for path in (corpus / "pairs").glob("*.json"):
        if path.stem != keep:
            deleted.append(path.stem)
            path.unlink()
    merged, cases = t0.run(corpus)
    assert cases["manifest"].verdict == DIFFER
    paths = {f.field_path for f in cases["manifest"].findings}
    assert any(p.startswith("pair_ids") for p in paths)
    for fid in deleted:
        assert f"pairs.{fid}.payload_hash" in paths
    assert merged.verdict != AGREE
    assert cases["coverage"].verdict == DIFFER


def test_an_extra_undeclared_file_is_a_finding(corpus):
    extra = json.loads(_pair_path(corpus, "000_TWIN-P5").read_text())
    extra["fixture_id"] = "999_UNDECLARED"
    (corpus / "pairs" / "999_UNDECLARED.json").write_text(json.dumps(extra))
    _, cases = t0.run(corpus)
    assert cases["manifest"].verdict == DIFFER


def test_a_tampered_manifest_hash_is_a_finding(corpus):
    index = json.loads((corpus / "INDEX.json").read_text())
    index["pairs"]["003_CAL-P"]["payload_hash"] = "sha256:" + "0" * 64
    (corpus / "INDEX.json").write_text(json.dumps(index, indent=2, sort_keys=True))
    _, cases = t0.run(corpus)
    assert cases["manifest"].verdict == DIFFER


def test_a_tampered_corpus_hash_is_a_finding(corpus):
    index = json.loads((corpus / "INDEX.json").read_text())
    index["corpus_hash"] = "sha256:" + "0" * 64
    (corpus / "INDEX.json").write_text(json.dumps(index, indent=2, sort_keys=True))
    _, cases = t0.run(corpus)
    assert cases["manifest"].verdict == DIFFER


# --------------------------------------------------------------------------
# coverage
# --------------------------------------------------------------------------


def test_a_missing_strategy_axis_claim_fails_coverage(corpus):
    index = json.loads((corpus / "INDEX.json").read_text())
    index["coverage"].pop("strategy:CAL-P")
    (corpus / "INDEX.json").write_text(json.dumps(index, indent=2, sort_keys=True))
    _, cases = t0.run(corpus)
    assert cases["coverage"].verdict == DIFFER
    assert any(f.field_path.startswith("axes.strategy:CAL-P")
               for f in cases["coverage"].findings)


def test_a_missing_refusal_code_claim_fails_coverage(corpus):
    index = json.loads((corpus / "INDEX.json").read_text())
    index["coverage"].pop("refusal:NO_CHAIN")
    (corpus / "INDEX.json").write_text(json.dumps(index, indent=2, sort_keys=True))
    _, cases = t0.run(corpus)
    assert cases["coverage"].verdict == DIFFER


def test_deleting_the_only_priced_fixture_loses_its_axis(corpus):
    _pair_path(corpus, "004_BFLY-P").unlink()
    _, cases = t0.run(corpus)
    paths = {f.field_path for f in cases["coverage"].findings}
    assert any(p.startswith("axes.priced:BFLY-P") for p in paths)


def test_an_inflated_covers_list_is_a_finding(corpus):
    _rewrite(corpus, "002_STR-THRU", lambda d: d["covers"].append("geometry:round_listed_strike"))
    _, cases = t0.run(corpus)
    assert cases["manifest"].verdict == DIFFER
    assert cases["coverage"].verdict == DIFFER


def test_uncovered_axes_are_derived_from_the_records_not_read_from_the_index(corpus):
    """An index claiming no gaps cannot hide one: the gate reads the derivation."""
    index = json.loads((corpus / "INDEX.json").read_text())
    index["required_axes"].append("dyn_sv:tie")
    (corpus / "INDEX.json").write_text(json.dumps(index, indent=2, sort_keys=True))
    assert t0.derived_uncovered(t0.load(corpus)) == ["dyn_sv:tie"]


def test_a_refusal_with_a_computed_strike_does_not_cover_round_listed_strike():
    """The first corpus's round-listed-strike fixture was exactly this row."""
    got = covers(refusal("TWIN-P5", "NO_CHAIN"), request("TWIN-P5", strike=14.7615))
    assert "geometry:round_listed_strike" not in got


def test_a_priced_row_at_one_of_its_listed_strikes_covers_round_listed_strike():
    assert "geometry:round_listed_strike" in covers(
        priced("TWIN-P5"), request("TWIN-P5", strike=250.0))


def test_a_priced_row_whose_legs_snapped_away_from_the_request_does_not():
    assert "geometry:round_listed_strike" not in covers(
        priced("TWIN-P5"), request("TWIN-P5", strike=251.3))


def test_a_pinned_request_covers_pinned_only_beside_its_named_source():
    rec, req = priced("TWIN-P5"), request("TWIN-P5", structure_params={"width_moneyness": WIDTH})
    assert not {"geometry:pinned", "geometry:selector"} & covers(rec, req)
    assert "geometry:pinned" in covers(rec, req, relations={"pinned_from": "sha256:x"})


def test_a_selector_resolved_priced_row_covers_selector_and_computed_width():
    assert {"geometry:selector", "geometry:computed_width"} <= covers(priced("TWIN-P5"))


def test_geometry_axes_need_a_priced_row():
    rec = refusal("TWIN-P5", "NO_FORECAST", structure_params={"width_moneyness": WIDTH})
    assert not {axis for axis in covers(rec) if axis.startswith("geometry:")}


def test_the_coarse_ladder_refusal_covers_its_axis():
    assert "geometry:coarse_ladder" in covers(refusal("TWIN-P5", "COARSE_LADDER"))


def test_an_even_ladder_is_an_exact_mirror_and_an_uneven_one_is_not():
    assert "geometry:exact_mirror" in covers(priced("BFLY-P"))
    uneven = priced("BFLY-P", legs=[{"strike": 240.0}, {"strike": 250.0}, {"strike": 265.0}])
    assert "geometry:exact_mirror" not in covers(uneven)


def test_bad_quote_is_one_refusal_code_not_two():
    got = covers(refusal("STR-THRU", "BAD_QUOTE"))
    assert "refusal:BAD_QUOTE" in got
    assert not any("COST_PCT" in axis for axis in got)


def test_a_disabled_refusal_and_a_research_replay_cover_their_axes():
    assert "disabled:CAL-P:refused" in covers(refusal("CAL-P", "UNVALIDATED_STRUCTURE"))
    assert "disabled:CAL-P:research_replay" in covers(
        {"strategy": "CAL-P", "rows": []}, {"kind": "research_replay"}, kind="research_replay")


def test_boundaries_come_from_the_trade_window():
    across = priced("STR-RUNUP", entry_date="2025-12-22", exit_date="2026-01-06")
    assert {"boundary:year", "boundary:month"} <= covers(across)
    assert not {"boundary:year", "boundary:month"} & covers(priced("TWIN-P5"))


def dyn_record(**fields) -> dict:
    record = {"strategy": "DYN-SV", "chosen_strategy": "BFLY-P", "ticker": "MTN",
              "event_date": "2026-09-28", "session": "AMC", "menu_size": 2,
              "chosen_margin": 0.01, "chooser_score": 0.3,
              "legs": copy.deepcopy(LEGS), "entry_cost": 1.0, "flags": []}
    record.update(fields)
    return record


def dyn_request(strategies: list[str]) -> dict:
    return {"kind": "dyn_sv_resolution", "menu": MENU,
            "frame_rows": [{"request": request(s), "record": {"strategy": s}}
                           for s in strategies]}


def test_a_tie_between_two_structures_covers_tie():
    got = covers(dyn_record(chosen_margin=0.0), dyn_request(["BFLY-P", "TWIN-P5"]),
                 kind="dyn_sv_choice")
    assert {"dyn_sv:tie", "dyn_sv:partial_menu", "priced:DYN-SV"} <= got


def test_a_frame_carrying_one_structure_twice_covers_no_dyn_sv_axis():
    """The first corpus's only tie: BFLY-P against its own pinned re-score."""
    got = covers(dyn_record(chosen_margin=0.0), dyn_request(["BFLY-P", "BFLY-P"]),
                 kind="dyn_sv_choice")
    assert not {axis for axis in got if axis.startswith("dyn_sv:")}


def test_a_nonfinite_chooser_score_on_a_full_menu_is_the_fallback():
    got = covers(dyn_record(chooser_score={"__nonfinite__": "nan"}, menu_size=3),
                 dyn_request(MENU), kind="dyn_sv_choice")
    assert {"dyn_sv:fallback", "dyn_sv:full_menu"} <= got
    assert "dyn_sv:tie" not in got


# --------------------------------------------------------------------------
# pinned counterparts — e845f3e on real-shaped data
# --------------------------------------------------------------------------


def test_a_pinned_fixture_without_its_source_fails(tmp_path):
    pairs = [p for p in standard_pairs() if p["fixture_id"] != "000_TWIN-P5"]
    _, cases = t0.run(build(tmp_path / "tier0", pairs))
    assert cases["pinned_counterparts"].verdict == DIFFER


def test_a_pinned_fixture_that_lost_its_forecast_fails(tmp_path):
    pairs = standard_pairs()
    pinned = pairs[1]
    record = copy.deepcopy(pinned["payload"]["record"])
    record["forecast_abs_move"] = None
    pairs[1] = pair(pinned["fixture_id"], pinned["payload"]["request"], record,
                    relations=pinned["payload"]["relations"])
    _, cases = t0.run(build(tmp_path / "tier0", pairs))
    assert cases["pinned_counterparts"].verdict == DIFFER
    assert any(f.field_path.endswith("forecast_abs_move")
               for f in cases["pinned_counterparts"].findings)


# --------------------------------------------------------------------------
# seeded controls over the corpus
# --------------------------------------------------------------------------


def test_the_seeded_controls_detect_every_cause_on_distinct_pairs(corpus):
    summary = t0.seeded_controls(t0.load(corpus))
    targets = [control["target"] for control in summary["controls"].values()]
    assert None not in targets and len(set(targets)) == len(targets)
    assert all(control["problems"] == [] for control in summary["controls"].values()), summary
    assert summary["untargeted_findings"] == []
    assert summary["field_set_mismatches"] == []
    assert summary["one_pass_verdict"] == DIFFER


def test_the_seeded_controls_fail_when_no_pair_can_carry_a_cause(tmp_path):
    pairs = [p for p in standard_pairs() if p["fixture_id"] != "005_CND-PS"]
    _, cases = t0.run(build(tmp_path / "tier0", pairs))
    assert cases["seeded_controls"].verdict == DIFFER


# --------------------------------------------------------------------------
# the CLI
# --------------------------------------------------------------------------


def test_the_cli_reports_json(corpus):
    proc = subprocess.run(
        [sys.executable, str(ROOT / "checks" / "tier0_corpus.py"),
         "--corpus", str(corpus), "--json"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["verdict"] == AGREE
    assert out["uncovered_axes"] == []
    assert out["pairs"] == out["declared_pairs"] == 6
    assert all(c["problems"] == [] for c in out["seeded_controls"]["controls"].values())


# --------------------------------------------------------------------------
# shared frozen documents (a pair's payload may embed one by reference)
# --------------------------------------------------------------------------


def write_shared(root: Path, value, *, expanded=None) -> str:
    """Write ``value`` (the on-disk storage form, which may itself carry
    nested ``$shared`` references) under ``root/shared/<hex>.json``. The
    digest is the content hash of ``expanded`` (the fully expanded logical
    document, defaulting to ``value`` itself when it carries no nested
    reference) -- the same value a real writer computes, never the hash of
    the reference-shaped storage form.
    """
    digest = content_hash(expanded if expanded is not None else value)
    shared_dir = root / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    body = {"schema_version": t0.SHARED_DOCUMENT_SCHEMA_VERSION,
            "digest": digest, "value": value}
    (shared_dir / f"{digest.split(':', 1)[-1]}.json").write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n")
    return digest


def test_a_shared_reference_resolves_to_one_object_shared_by_every_occurrence(tmp_path):
    """Two DIFFERENT pairs, and two occurrences within one of them, all
    referencing the same fold pool: `load` must read `shared/<hex>.json`
    exactly once and hand every occurrence the SAME Python object back."""
    pool = {"predictions": [0.1, 0.2, 0.3], "tag": "pool"}
    digest = write_shared(tmp_path / "tier0", pool)
    ref = {t0.SHARED_REF_KEY: digest}
    p1 = pair("100_shared_a", request("STR-THRU"),
              priced("STR-THRU", extra_field={"a": ref, "b": ref}))
    p2 = pair("101_shared_b", request("BFLY-P"), priced("BFLY-P", extra_field=ref))
    root = build(tmp_path / "tier0", [p1, p2])
    corpus = t0.load(root)
    a = corpus.pairs["100_shared_a"]["payload"]["record"]["extra_field"]
    b = corpus.pairs["101_shared_b"]["payload"]["record"]["extra_field"]
    assert a["a"] is a["b"] is b
    assert a["a"] == pool


def test_an_unknown_reference_shape_refuses_loudly(tmp_path):
    pool = {"tag": "pool"}
    digest = write_shared(tmp_path / "tier0", pool)
    bad_ref = {t0.SHARED_REF_KEY: digest, "extra": 1}
    p = pair("102_bad_shape", request("STR-THRU"), priced("STR-THRU", extra_field=bad_ref))
    root = build(tmp_path / "tier0", [p])
    with pytest.raises(t0.CorpusFormatError):
        t0.load(root)


def test_a_missing_shared_file_refuses_loudly(tmp_path):
    ref = {t0.SHARED_REF_KEY: "sha256:" + "a" * 64}
    p = pair("103_missing", request("STR-THRU"), priced("STR-THRU", extra_field=ref))
    root = build(tmp_path / "tier0", [p])
    with pytest.raises(t0.CorpusFormatError):
        t0.load(root)


def test_a_digest_mismatched_shared_file_refuses_loudly(tmp_path):
    pool = {"tag": "pool"}
    digest = write_shared(tmp_path / "tier0", pool)
    # Tamper with the shared file's content after its digest was computed.
    shared_path = tmp_path / "tier0" / "shared" / f"{digest.split(':', 1)[-1]}.json"
    doc = json.loads(shared_path.read_text())
    doc["value"]["tag"] = "tampered"
    shared_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    ref = {t0.SHARED_REF_KEY: digest}
    p = pair("104_tampered", request("STR-THRU"), priced("STR-THRU", extra_field=ref))
    root = build(tmp_path / "tier0", [p])
    with pytest.raises(t0.CorpusFormatError):
        t0.load(root)


def test_a_shared_document_may_reference_another(tmp_path):
    """Nested sharing: one shared document embeds a reference to another."""
    inner = {"tag": "inner-pool"}
    inner_digest = write_shared(tmp_path / "tier0", inner)
    outer_stored = {"nested": {t0.SHARED_REF_KEY: inner_digest}, "tag": "outer"}
    outer_expanded = {"nested": inner, "tag": "outer"}
    outer_digest = write_shared(tmp_path / "tier0", outer_stored, expanded=outer_expanded)
    ref = {t0.SHARED_REF_KEY: outer_digest}
    p = pair("105_nested", request("STR-THRU"), priced("STR-THRU", extra_field=ref))
    root = build(tmp_path / "tier0", [p])
    corpus = t0.load(root)
    resolved = corpus.pairs["105_nested"]["payload"]["record"]["extra_field"]
    assert resolved["nested"] == inner
    assert resolved["tag"] == "outer"


def test_an_old_format_corpus_has_no_shared_directory_and_loads_unchanged(corpus):
    assert not (corpus / "shared").exists()
    loaded = t0.load(corpus)
    assert loaded.pairs["000_TWIN-P5"]["payload"]["record"]["strategy"] == "TWIN-P5"


# --------------------------------------------------------------------------
# shared translation row tables (input_translation.mappings by reference)
# --------------------------------------------------------------------------


def translation_row(shared_path: list, native_path: list, value) -> dict:
    return {"shared_path": shared_path, "native_path": native_path,
            "value_hash": content_hash(value)}


def write_translation_table(root: Path, rows: list) -> str:
    """Write ``rows`` under ``root/shared/translations/<hex>.json``, the
    format ``tools/capture_tier0_corpus.py``'s ``_TranslationTableWriter``
    writes. Returns the digest (``content_hash(rows)`` over the exact array
    stored, per ``SHARED_TRANSLATION_TABLE_SCHEMA_VERSION``).
    """
    digest = content_hash(rows)
    tables_dir = root / "shared" / "translations"
    tables_dir.mkdir(parents=True, exist_ok=True)
    body = {"schema_version": t0.SHARED_TRANSLATION_TABLE_SCHEMA_VERSION,
            "digest": digest, "rows": rows}
    (tables_dir / f"{digest.split(':', 1)[-1]}.json").write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n")
    return digest


def test_an_identity_reference_returns_the_table_rows_exactly(tmp_path):
    """The table itself may hold rows in ANY order (here deliberately
    REVERSED path order): an "identity" reference must return exactly that,
    unchanged -- never re-sorted."""
    table_rows = [translation_row(["p", i], ["p", i], i) for i in range(50, 0, -1)]
    digest = write_translation_table(tmp_path / "tier0", table_rows)
    ref = {t0.ROWS_REF_KEY: digest, "order": t0.ROWS_ORDER_IDENTITY}
    p = pair("199_identity", request("STR-THRU"), priced("STR-THRU", extra_field=ref))
    root = build(tmp_path / "tier0", [p])
    corpus = t0.load(root)
    resolved = corpus.pairs["199_identity"]["payload"]["record"]["extra_field"]
    assert resolved == table_rows
    assert content_hash(resolved) == content_hash(table_rows)


def test_a_positions_reference_reconstructs_realistically_unsorted_order(tmp_path):
    """2026-09-20: the real capture's mappings order turned out to be
    NEITHER sorted-by-path NOR consistent across members -- the probe on
    real data refused an earlier, sort-based design. This fixture is
    deliberately adversarial to that assumption: the table holds rows in
    REVERSED path order, and the member's OWN order interleaves an ascending
    even-path pass, one of its own rows not in the table at all, a
    descending odd-path pass, and a second own-only row -- proving
    reconstruction depends on nothing but the explicit ``sequence``, never
    on re-deriving order from content.
    """
    table_rows = [translation_row(["p", i], ["p", i], i) for i in range(199, -1, -1)]
    digest = write_translation_table(tmp_path / "tier0", table_rows)
    table_index = {content_hash(row): pos for pos, row in enumerate(table_rows)}

    own_row_a = translation_row(["p", 500], ["p", 500], "own-a")
    own_row_b = translation_row(["p", 501], ["p", 501], "own-b")
    member_rows = (
        [translation_row(["p", i], ["p", i], i) for i in range(0, 200, 2)]
        + [own_row_a]
        + [translation_row(["p", i], ["p", i], i) for i in range(199, 0, -2)]
        + [own_row_b]
    )
    tokens = []
    literals = []
    for row in member_rows:
        key = content_hash(row)
        position = table_index.get(key)
        if position is not None:
            tokens.append(str(position))
        else:
            tokens.append(f"L{len(literals)}")
            literals.append(row)
    ref = {t0.ROWS_REF_KEY: digest, "order": t0.ROWS_ORDER_POSITIONS,
           "sequence": ",".join(tokens), "literals": literals}
    p = pair("200_rows", request("STR-THRU"), priced("STR-THRU", extra_field=ref))
    root = build(tmp_path / "tier0", [p])
    corpus = t0.load(root)
    resolved = corpus.pairs["200_rows"]["payload"]["record"]["extra_field"]
    assert resolved == member_rows  # exact order, including the two literals
    assert content_hash(resolved) == content_hash(member_rows)  # hash-oracle equivalence


def test_an_unknown_rows_reference_shape_refuses_loudly(tmp_path):
    digest = write_translation_table(
        tmp_path / "tier0", [translation_row(["p", 0], ["p", 0], 0)])
    bad_ref = {t0.ROWS_REF_KEY: digest, "order": t0.ROWS_ORDER_IDENTITY, "surprise": 1}
    p = pair("201_bad_shape", request("STR-THRU"), priced("STR-THRU", extra_field=bad_ref))
    root = build(tmp_path / "tier0", [p])
    with pytest.raises(t0.CorpusFormatError):
        t0.load(root)


def test_a_missing_translation_table_refuses_loudly(tmp_path):
    ref = {t0.ROWS_REF_KEY: "sha256:" + "b" * 64, "order": t0.ROWS_ORDER_IDENTITY}
    p = pair("202_missing_table", request("STR-THRU"), priced("STR-THRU", extra_field=ref))
    root = build(tmp_path / "tier0", [p])
    with pytest.raises(t0.CorpusFormatError):
        t0.load(root)


def test_a_tampered_translation_table_refuses_loudly(tmp_path):
    table_rows = [translation_row(["p", i], ["p", i], i) for i in range(5)]
    digest = write_translation_table(tmp_path / "tier0", table_rows)
    table_path = (tmp_path / "tier0" / "shared" / "translations"
                  / f"{digest.split(':', 1)[-1]}.json")
    doc = json.loads(table_path.read_text())
    doc["rows"][0]["value_hash"] = "sha256:" + "c" * 64
    table_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    ref = {t0.ROWS_REF_KEY: digest, "order": t0.ROWS_ORDER_IDENTITY}
    p = pair("203_tampered_table", request("STR-THRU"), priced("STR-THRU", extra_field=ref))
    root = build(tmp_path / "tier0", [p])
    with pytest.raises(t0.CorpusFormatError):
        t0.load(root)


def test_an_unsupported_row_order_refuses_loudly(tmp_path):
    digest = write_translation_table(
        tmp_path / "tier0", [translation_row(["p", 0], ["p", 0], 0)])
    ref = {t0.ROWS_REF_KEY: digest, "order": "shared_path"}  # the retired v1 scheme
    p = pair("204_bad_order", request("STR-THRU"), priced("STR-THRU", extra_field=ref))
    root = build(tmp_path / "tier0", [p])
    with pytest.raises(t0.CorpusFormatError):
        t0.load(root)


def test_a_malformed_sequence_token_refuses_loudly(tmp_path):
    digest = write_translation_table(
        tmp_path / "tier0", [translation_row(["p", 0], ["p", 0], 0)])
    ref = {t0.ROWS_REF_KEY: digest, "order": t0.ROWS_ORDER_POSITIONS,
           "sequence": "0,not-a-token", "literals": []}
    p = pair("205_bad_token", request("STR-THRU"), priced("STR-THRU", extra_field=ref))
    root = build(tmp_path / "tier0", [p])
    with pytest.raises(t0.CorpusFormatError):
        t0.load(root)


def test_a_sequence_table_index_out_of_range_refuses_loudly(tmp_path):
    digest = write_translation_table(
        tmp_path / "tier0", [translation_row(["p", 0], ["p", 0], 0)])
    ref = {t0.ROWS_REF_KEY: digest, "order": t0.ROWS_ORDER_POSITIONS,
           "sequence": "1", "literals": []}
    p = pair("206_index_oob", request("STR-THRU"), priced("STR-THRU", extra_field=ref))
    root = build(tmp_path / "tier0", [p])
    with pytest.raises(t0.CorpusFormatError):
        t0.load(root)


def test_a_sequence_literal_index_out_of_range_refuses_loudly(tmp_path):
    digest = write_translation_table(
        tmp_path / "tier0", [translation_row(["p", 0], ["p", 0], 0)])
    ref = {t0.ROWS_REF_KEY: digest, "order": t0.ROWS_ORDER_POSITIONS,
           "sequence": "L0", "literals": []}
    p = pair("207_literal_oob", request("STR-THRU"), priced("STR-THRU", extra_field=ref))
    root = build(tmp_path / "tier0", [p])
    with pytest.raises(t0.CorpusFormatError):
        t0.load(root)


def test_an_unused_literal_refuses_loudly(tmp_path):
    digest = write_translation_table(
        tmp_path / "tier0", [translation_row(["p", 0], ["p", 0], 0)])
    unused = translation_row(["p", 1], ["p", 1], 1)
    ref = {t0.ROWS_REF_KEY: digest, "order": t0.ROWS_ORDER_POSITIONS,
           "sequence": "0", "literals": [unused]}
    p = pair("208_unused_literal", request("STR-THRU"), priced("STR-THRU", extra_field=ref))
    root = build(tmp_path / "tier0", [p])
    with pytest.raises(t0.CorpusFormatError):
        t0.load(root)


def test_a_duplicate_table_position_in_a_sequence_refuses_loudly(tmp_path):
    digest = write_translation_table(
        tmp_path / "tier0", [translation_row(["p", 0], ["p", 0], 0)])
    ref = {t0.ROWS_REF_KEY: digest, "order": t0.ROWS_ORDER_POSITIONS,
           "sequence": "0,0", "literals": []}
    p = pair("209_dup_position", request("STR-THRU"), priced("STR-THRU", extra_field=ref))
    root = build(tmp_path / "tier0", [p])
    with pytest.raises(t0.CorpusFormatError):
        t0.load(root)


def test_a_duplicate_literal_use_in_a_sequence_refuses_loudly(tmp_path):
    """Distinct from a duplicate TABLE position: here the same literal index
    is referenced twice by the sequence, which is just as ambiguous as
    replaying one table row twice.
    """
    digest = write_translation_table(
        tmp_path / "tier0", [translation_row(["p", 0], ["p", 0], 0)])
    literal = translation_row(["p", 1], ["p", 1], 1)
    ref = {t0.ROWS_REF_KEY: digest, "order": t0.ROWS_ORDER_POSITIONS,
           "sequence": "L0,L0", "literals": [literal]}
    p = pair("210_dup_literal", request("STR-THRU"), priced("STR-THRU", extra_field=ref))
    root = build(tmp_path / "tier0", [p])
    with pytest.raises(t0.CorpusFormatError):
        t0.load(root)


def test_a_duplicate_row_within_a_shared_table_refuses_loudly(tmp_path):
    """A stored table with a repeated row is ambiguous for both overlap
    matching and position reconstruction -- the writer never produces this,
    but a hand-edited or corrupted table must still be refused, not silently
    accepted.
    """
    row = translation_row(["p", 0], ["p", 0], 0)
    table_rows = [row, dict(row)]
    tables_dir = tmp_path / "tier0" / "shared" / "translations"
    tables_dir.mkdir(parents=True, exist_ok=True)
    # Write directly (bypassing `write_translation_table`'s digest-over-rows
    # convention isn't needed here) so the digest matches this exact,
    # duplicate-carrying row list.
    digest = content_hash(table_rows)
    body = {"schema_version": t0.SHARED_TRANSLATION_TABLE_SCHEMA_VERSION,
            "digest": digest, "rows": table_rows}
    (tables_dir / f"{digest.split(':', 1)[-1]}.json").write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n")
    ref = {t0.ROWS_REF_KEY: digest, "order": t0.ROWS_ORDER_IDENTITY}
    p = pair("211_dup_table_row", request("STR-THRU"), priced("STR-THRU", extra_field=ref))
    root = build(tmp_path / "tier0", [p])
    with pytest.raises(t0.CorpusFormatError):
        t0.load(root)


def test_an_unparseable_translation_table_file_refuses_loudly(tmp_path):
    digest = "sha256:" + "d" * 64
    tables_dir = tmp_path / "tier0" / "shared" / "translations"
    tables_dir.mkdir(parents=True, exist_ok=True)
    (tables_dir / f"{digest.split(':', 1)[-1]}.json").write_text("{not json")
    ref = {t0.ROWS_REF_KEY: digest, "order": t0.ROWS_ORDER_IDENTITY}
    p = pair("212_bad_json", request("STR-THRU"), priced("STR-THRU", extra_field=ref))
    root = build(tmp_path / "tier0", [p])
    with pytest.raises(t0.CorpusFormatError):
        t0.load(root)


def test_a_translation_table_with_the_wrong_schema_version_refuses_loudly(tmp_path):
    table_rows = [translation_row(["p", 0], ["p", 0], 0)]
    digest = content_hash(table_rows)
    tables_dir = tmp_path / "tier0" / "shared" / "translations"
    tables_dir.mkdir(parents=True, exist_ok=True)
    body = {"schema_version": "tier0_shared_translation_table.v9.9",
            "digest": digest, "rows": table_rows}
    (tables_dir / f"{digest.split(':', 1)[-1]}.json").write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n")
    ref = {t0.ROWS_REF_KEY: digest, "order": t0.ROWS_ORDER_IDENTITY}
    p = pair("213_bad_schema", request("STR-THRU"), priced("STR-THRU", extra_field=ref))
    root = build(tmp_path / "tier0", [p])
    with pytest.raises(t0.CorpusFormatError):
        t0.load(root)


def test_a_translation_table_whose_rows_is_not_a_list_refuses_loudly(tmp_path):
    digest = content_hash({"not": "a list"})
    tables_dir = tmp_path / "tier0" / "shared" / "translations"
    tables_dir.mkdir(parents=True, exist_ok=True)
    body = {"schema_version": t0.SHARED_TRANSLATION_TABLE_SCHEMA_VERSION,
            "digest": digest, "rows": {"not": "a list"}}
    (tables_dir / f"{digest.split(':', 1)[-1]}.json").write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n")
    ref = {t0.ROWS_REF_KEY: digest, "order": t0.ROWS_ORDER_IDENTITY}
    p = pair("214_rows_not_list", request("STR-THRU"), priced("STR-THRU", extra_field=ref))
    root = build(tmp_path / "tier0", [p])
    with pytest.raises(t0.CorpusFormatError):
        t0.load(root)


def test_an_old_format_plain_mappings_list_loads_unchanged(tmp_path):
    """A pair captured before this fix carries `mappings` as a plain list,
    with no `shared/translations/` directory anywhere -- CURRENT is still an
    old-format corpus. It must load byte-for-byte unchanged, order included
    (deliberately reversed here, not ascending-by-path).
    """
    mappings = [translation_row(["p", i], ["p", i], i) for i in range(4, -1, -1)]
    p = pair("206_plain_mappings", request("STR-THRU"),
             priced("STR-THRU", extra_field={"mappings": mappings}))
    root = build(tmp_path / "tier0", [p])
    assert not (root / "shared").exists()
    corpus = t0.load(root)
    loaded = corpus.pairs["206_plain_mappings"]["payload"]["record"]["extra_field"]["mappings"]
    assert loaded == mappings
