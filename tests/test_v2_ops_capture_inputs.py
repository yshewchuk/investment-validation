"""Completeness proof for ``engine.v2.ops.capture_inputs`` (task brief
deliverable 4): a barrier legacy action, staged from a root built ONLY from
a captured manifest, must succeed -- and an access audit proves nothing it
read escaped that staged root.

**Why only ``legacy_finality`` and ``legacy_decisions`` run here.**
``legacy_finality`` is the required case (the real 2026-09-14 defect). Every
legacy action must run in a FRESH process for ``engine.paths.ROOT`` to bind
correctly (``_rooted_import`` only sets an environment variable; nothing
re-reads it once ``engine.paths`` has been imported once in a process --
``tests/test_v2_ops_supervised_legacy.py``'s own docstring: "the gap that hid
every defect in this file's siblings"), so both cases below launch a real
subprocess rather than calling ``legacy_action`` in-process.
``legacy_decisions`` is the one other CHEAP case: ``engine.ledger._event_ids``
degrades to an unmatched (but non-crashing) join when ``earnings_events`` is
absent (the same fact ``test_v2_ops_supervised_legacy.py`` uses to make it
"the cheapest legacy kind to run for real: it needs no market data, no
scorer, no FeatureContext"), so it needs no trained model, no Scorer, no
multi-gigabyte panel.

``legacy_settlement``, ``legacy_model_evidence``, ``legacy_render`` and
``legacy_selfcheck`` are NOT exercised here. Settlement replays a recorded
structure through ``engine.replay`` against real option-chain quotes;
model_evidence/render/selfcheck all build a real
``Scorer(context=FeatureContext.load(...))`` over the full ``daily_market``/
``trades``/Tier-4 stack -- the AGENTS.md-recorded measurement is ~3 GiB
resident for one ``Scorer()`` alone, and a synthetic type-correct fixture
does not exercise the actual model artifacts those paths fit against. None
of that is "cheap" in the task brief's sense; it needs the real, licensed
data this task's read-only approval does not extend to running scoring over.
"""
from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.data import reference_inputs  # noqa: E402
from engine.v2.data.legacy_nightly_read_plan import (  # noqa: E402
    BARRIER_KINDS,
    NIGHTLY_CAPTURE_IMPLEMENTATION_REF,
    manifest_problems,
    required_families,
)
from engine.v2.foundation import to_document  # noqa: E402
from engine.v2.ops.capture_inputs import capture  # noqa: E402
from engine.v2.ops.errors import OpsError  # noqa: E402
from engine.v2.ops.legacy_adapter import copy_read_set  # noqa: E402
from tests.test_v2_data_import import build_legacy_store  # noqa: E402

SESSION = "2024-01-01"
TICKER = "fx_ticker"  # build_legacy_store's own fixed placeholder ticker


def _write_calendar(root: Path, dates: list[str]) -> None:
    path = root / reference_inputs.LEGACY_REFERENCE_INPUTS_V1["inputs"]["calendar"]["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Price,Close", "Ticker,^GSPC", "Date,"] + [f"{d},100.0" for d in dates]
    path.write_text("\n".join(lines) + "\n")


def _write_orats_cache(root: Path, trade_date: str) -> None:
    fetch_dir = root / "data" / "raw" / "fetch" / "orats" / "ab"
    fetch_dir.mkdir(parents=True, exist_ok=True)
    for i, endpoint in enumerate(("hist/summaries", "hist/cores")):
        key = f"{trade_date}-{i}"
        meta = {"key": key, "source": "orats", "endpoint": endpoint,
                "params": {"tradeDate": trade_date}, "url": "x", "status": 200,
                "bytes": 2, "elapsed_s": 0.1, "quota_remaining": 100,
                "fetched_at": trade_date + "T00:00:00Z", "note": ""}
        (fetch_dir / f"{key}.meta.json").write_text(json.dumps(meta))
        with gzip.open(fetch_dir / f"{key}.body.gz", "wb") as fh:
            fh.write(b"{}")


def _calendar_dates_ending_at(session: str, n: int) -> list[str]:
    """``n`` consecutive calendar dates ending at ``session`` -- enough
    trading-day history that ``resolve_final_session``'s 15-session walk-back
    never runs off the front of the calendar (``TradingCalendar.shift``
    raises ``KeyError`` past the earliest known day, which is a calendar
    fixture defect, not the ``SOURCE_NOT_FINAL`` outcome under test)."""
    import pandas as pd

    end = pd.Timestamp(session)
    return [str((end - pd.Timedelta(days=i)).date()) for i in range(n - 1, -1, -1)]


def _build_fixture(root: Path, *, year: int = 2024, with_orats: bool = True) -> None:
    """A synthetic legacy root: TIER2 tables + reference inputs (via the
    already-reviewed ``build_legacy_store``, ``tests/test_v2_data_import.py``),
    a calendar covering ``SESSION`` with 20 days of history, an empty ledger,
    and (optionally) cached ORATS market-wide files for ``SESSION``.
    """
    build_legacy_store(root, year=year)
    _write_calendar(root, _calendar_dates_ending_at(SESSION, 20))
    (root / "ledger" / "predictions").mkdir(parents=True)
    (root / "ledger" / "outcomes").mkdir(parents=True)
    if with_orats:
        _write_orats_cache(root, SESSION)


# --------------------------------------------------------------------------
# pure: legacy_nightly_read_plan.manifest_problems
# --------------------------------------------------------------------------


def test_barrier_kinds_are_the_structural_six():
    """BARRIER_KINDS must equal the real ``legacy_*`` ACTION_NAMES minus the
    three kinds that are ALWAYS snapshot-backed once a plan pins a snapshot
    and never fall back to the barrier (``legacy_score``, ``legacy_score_
    requests``, ``legacy_decision_replay``) -- derived from the registry/
    allowlist, never typed by hand (task brief deliverable 1).

    ``legacy_render``/``legacy_selfcheck`` (attempt-19 fix, 2026-09-15) are
    dual-mode: in ``SNAPSHOT_BACKED_KINDS`` (they run snapshot-backed when a
    plan pins one) AND in ``BARRIER_KINDS`` (they still need this module's
    declared read plan for a legacy-mode plan), so the two sets are no
    longer exact complements within ``ACTION_NAMES`` -- assert the sharper
    invariant instead: every barrier kind not exclusively snapshot-backed is
    declared here, and nothing declared here is snapshot-only.
    """
    from engine.v2.ops.legacy_actions import ACTION_NAMES
    from engine.v2.ops.stages import SNAPSHOT_BACKED_KINDS

    snapshot_only = SNAPSHOT_BACKED_KINDS - set(BARRIER_KINDS)
    assert snapshot_only == {"legacy_score", "legacy_score_requests", "legacy_decision_replay"}
    assert set(BARRIER_KINDS) == set(ACTION_NAMES) - snapshot_only
    assert set(BARRIER_KINDS) & snapshot_only == set()


def test_manifest_problems_rejects_wrong_capture_ref():
    """The real 2026-09-14 defect: a snapshot-import manifest passed as a
    barrier-mode nightly's --input-manifest must be refused before any
    family check even runs."""
    problems = manifest_problems({"capture_implementation_ref": "snapshot_import_plan.v1"})
    assert len(problems) == 1
    assert problems[0]["kind"] is None
    assert "snapshot_import_plan.v1" in problems[0]["reason"]


def test_cli_guard_refuses_a_snapshot_import_manifest():
    """End-to-end guard proof (task brief's verify step): the real
    2026-09-14 failure was the ``snapshot_import_plan.v1`` manifest --
    ``engine.v2.data.import_snapshot``'s own shape, with real curated
    file_refs and no ``data/raw/fetch/**`` at all -- reaching a barrier-mode
    nightly's ``--input-manifest``. ``test_manifest_problems_rejects_wrong_capture_ref``
    already proves the pure function; this proves the CLI-wired guard
    (``engine.v2.ops.cli._check_nightly_manifest``, called from both
    ``plan nightly`` and ``submit``) raises a typed ``OpsError`` for exactly
    that document shape, not just an empty stub.
    """
    from engine.v2.ops.cli import _check_nightly_manifest

    document = {
        "manifest_id": "snap1",
        "file_refs": [{"path": "data/curated/daily_market/year=2024/part-0000.parquet",
                       "content_hash": "sha256:" + "0" * 64, "byte_size": 1},
                      {"path": "data/curated/option_chains/year=2024/part-0000.parquet",
                       "content_hash": "sha256:" + "0" * 64, "byte_size": 1}],
        "table_contract_refs": [], "registry_and_model_refs": ["placeholder::sha256:" + "0" * 64],
        "calendar_ref": "placeholder::sha256:" + "0" * 64, "selected_session": SESSION,
        "finality_receipt_refs": [], "knowledge_mode_by_table": {},
        "availability_evidence_refs": [], "read_set_complete": True,
        "capture_implementation_ref": "snapshot_import_plan.v1",
    }
    with pytest.raises(OpsError) as excinfo:
        _check_nightly_manifest(json.dumps(document).encode())
    assert excinfo.value.code == "INPUT_CHANGED"
    problems = excinfo.value.problem.details["problems"]
    assert len(problems) == 1 and problems[0]["kind"] is None
    assert "snapshot_import_plan.v1" in problems[0]["reason"]


def test_manifest_problems_flags_missing_family():
    """An empty read set flags every required family for the kind -- exactly
    the real defect: the snapshot-import manifest had zero
    data/raw/fetch/orats entries, and legacy_finality needs that family."""
    document = {"capture_implementation_ref": NIGHTLY_CAPTURE_IMPLEMENTATION_REF, "file_refs": []}
    problems = manifest_problems(document, kinds=("legacy_finality",))
    families = {p["family"] for p in problems}
    assert families == set(required_families("legacy_finality", only_required=True))
    assert "finality_raw_fetch_orats" in families


def test_manifest_problems_empty_for_a_complete_capture(tmp_path):
    _build_fixture(tmp_path)
    manifest = capture(tmp_path, as_of=SESSION, tickers=[TICKER], year_start=2024, year_end=2024)
    assert manifest_problems(to_document(manifest)) == []


# --------------------------------------------------------------------------
# pure: manifest_problems' ledger session bound (2026-09-10 attempt 11)
# --------------------------------------------------------------------------


def test_manifest_problems_flags_session_own_predictions():
    """The real defect: a manifest carrying session S's own predictions
    file. v2 must produce and commit those itself; reading them
    pre-computed from the staged legacy ledger is what let
    ``legacy_settlement`` try to commit outcomes for predictions never
    committed in the catalog."""
    document = {
        "capture_implementation_ref": NIGHTLY_CAPTURE_IMPLEMENTATION_REF,
        "selected_session": SESSION,
        "file_refs": [{"path": f"ledger/predictions/{SESSION}.jsonl",
                       "content_hash": "sha256:" + "0" * 64, "byte_size": 1}],
    }
    problems = manifest_problems(document, kinds=())
    assert len(problems) == 1
    assert problems[0]["kind"] is None and problems[0]["family"] == "ledger_predictions"
    assert SESSION in problems[0]["reason"]


def test_manifest_problems_flags_session_own_outcomes():
    """Send-back on 7d235f8: ``ledger/outcomes/S.jsonl`` is S's own nightly
    settlement output (legacy writes it same-day, by ``resolved_at``), not
    only a file dated strictly after S -- staging it lets legacy's own
    ``_unresolved`` treat those rows as already settled and skip them, so
    v2 never produces or commits them itself."""
    document = {
        "capture_implementation_ref": NIGHTLY_CAPTURE_IMPLEMENTATION_REF,
        "selected_session": SESSION,
        "file_refs": [{"path": f"ledger/outcomes/{SESSION}.jsonl",
                       "content_hash": "sha256:" + "0" * 64, "byte_size": 1}],
    }
    problems = manifest_problems(document, kinds=())
    assert len(problems) == 1
    assert problems[0]["kind"] is None and problems[0]["family"] == "ledger_outcomes"
    assert SESSION in problems[0]["reason"]


def test_manifest_problems_flags_future_outcomes():
    """Outcomes dated strictly after the session are future data too
    (a superset of the session-own-outcomes bound above)."""
    import pandas as pd

    future = str((pd.Timestamp(SESSION) + pd.Timedelta(days=1)).date())
    document = {
        "capture_implementation_ref": NIGHTLY_CAPTURE_IMPLEMENTATION_REF,
        "selected_session": SESSION,
        "file_refs": [{"path": f"ledger/outcomes/{future}.jsonl",
                       "content_hash": "sha256:" + "0" * 64, "byte_size": 1}],
    }
    problems = manifest_problems(document, kinds=())
    assert len(problems) == 1
    assert problems[0]["kind"] is None and problems[0]["family"] == "ledger_outcomes"
    assert future in problems[0]["reason"]


def test_manifest_problems_allows_prior_predictions_and_prior_outcomes():
    """The in-bounds case: predictions and outcomes strictly before S are
    both legitimate and must not be flagged."""
    import pandas as pd

    prior = str((pd.Timestamp(SESSION) - pd.Timedelta(days=1)).date())
    document = {
        "capture_implementation_ref": NIGHTLY_CAPTURE_IMPLEMENTATION_REF,
        "selected_session": SESSION,
        "file_refs": [{"path": f"ledger/predictions/{prior}.jsonl",
                       "content_hash": "sha256:" + "0" * 64, "byte_size": 1},
                      {"path": f"ledger/outcomes/{prior}.jsonl",
                       "content_hash": "sha256:" + "0" * 64, "byte_size": 1}],
    }
    assert manifest_problems(document, kinds=()) == []


# --------------------------------------------------------------------------
# capture(): ledger glob bounded to the session (2026-09-10 attempt 11)
# --------------------------------------------------------------------------


def test_capture_bounds_ledger_predictions_and_outcomes_to_the_session(tmp_path):
    """End-to-end proof of the same real defect at the capture layer: a
    fixture with a ledger predictions/outcomes file for the day before,
    on, and after SESSION must only carry the day-before prediction and
    the day-before outcome in the captured manifest -- never SESSION's own
    predictions or outcomes (send-back on 7d235f8: same-day outcomes are
    S's own nightly settlement output, not v2's), nor anything after it.
    """
    import pandas as pd

    fixture = tmp_path / "fixture"
    fixture.mkdir()
    _build_fixture(fixture)
    prior = str((pd.Timestamp(SESSION) - pd.Timedelta(days=1)).date())
    future = str((pd.Timestamp(SESSION) + pd.Timedelta(days=1)).date())
    for date_str in (prior, SESSION, future):
        (fixture / "ledger" / "predictions" / f"{date_str}.jsonl").write_text('{"row_id": "x"}\n')
        (fixture / "ledger" / "outcomes" / f"{date_str}.jsonl").write_text('{"row_id": "x"}\n')

    manifest = capture(fixture, as_of=SESSION, tickers=[TICKER], year_start=2024, year_end=2024)
    paths = {ref.path for ref in manifest.file_refs}

    assert f"ledger/predictions/{prior}.jsonl" in paths
    assert f"ledger/predictions/{SESSION}.jsonl" not in paths
    assert f"ledger/predictions/{future}.jsonl" not in paths

    assert f"ledger/outcomes/{prior}.jsonl" in paths
    assert f"ledger/outcomes/{SESSION}.jsonl" not in paths
    assert f"ledger/outcomes/{future}.jsonl" not in paths

    assert manifest_problems(to_document(manifest)) == []


# --------------------------------------------------------------------------
# capture(): score_context_price_series (real 2026-09-10 defect --
# job_ab3df699bc040a0ad012312f04995ad9 omitted 6 planned DYN-SV rows because
# neither px_<T>.csv nor the Tier-1 yfinance fetch cache reached the staged
# legacy tree)
# --------------------------------------------------------------------------


def _write_px_csv(root: Path, ticker: str, days) -> None:
    import pandas as pd

    path = root / "earnings_predictions" / "data" / "raw" / "yfinance" / f"px_{ticker}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": days, "close_adj": 100.0 + np.arange(len(days), dtype=float)}).to_csv(
        path, index=False)


def _write_yfinance_tier1_cache(root: Path, ticker: str, days) -> None:
    """A real Tier-1 cache entry via the actual ``Fetcher``/``cache_key``
    machinery (not a hand-rolled path), so the test exercises the same
    key/body-path shape ``_yf_history_from_tier1`` and ``iter_raw_fetch_cache``
    read, matching ``_write_orats_cache``'s equivalent role for the ORATS
    family above."""
    import gzip

    from engine.data.fetch import Fetcher, cache_key

    key = cache_key("yfinance", "history", {"ticker": ticker, "period": "max"})
    fetcher = Fetcher(root=root / "data" / "raw" / "fetch")
    body_path = fetcher.body_path("yfinance", key)
    meta_path = fetcher.meta_path("yfinance", key)
    body_path.parent.mkdir(parents=True, exist_ok=True)
    body = "Date,Close\n" + "\n".join(f"{d.date()},{100.0 + i}" for i, d in enumerate(days))
    with gzip.open(body_path, "wb") as fh:
        fh.write(body.encode())
    meta = {"key": key, "source": "yfinance", "endpoint": "history",
            "params": {"ticker": ticker, "period": "max"}, "url": "x", "status": 200,
            "bytes": len(body), "elapsed_s": 0.1, "quota_remaining": None,
            "fetched_at": SESSION + "T00:00:00Z", "note": ""}
    meta_path.write_text(json.dumps(meta))


def test_capture_includes_px_and_tier1_price_sources_for_context_tickers(tmp_path):
    """The staged tree must carry BOTH price sources
    ``engine.data.features.panel.add_runup_features``/``_yf_history_from_tier1``
    actually read (panel.py:495,533,444-475): the legacy ``px_<T>.csv``
    default-path file for a ticker that has one, and the Tier-1 yfinance
    fetch-cache entry for a ticker whose only source is Tier-1 -- neither
    was captured before ``score_context_price_series`` existed."""
    import pandas as pd

    _build_fixture(tmp_path)
    px_ticker, tier1_ticker = "PXTICK", "T1TICK"
    days = pd.bdate_range("2023-01-02", periods=310)
    _write_px_csv(tmp_path, px_ticker, days)
    _write_yfinance_tier1_cache(tmp_path, tier1_ticker, days)

    manifest = capture(tmp_path, as_of=SESSION, tickers=[px_ticker, tier1_ticker],
                       year_start=2024, year_end=2024)
    paths = {ref.path for ref in manifest.file_refs}

    assert f"earnings_predictions/data/raw/yfinance/px_{px_ticker}.csv" in paths
    assert any(p.startswith("data/raw/fetch/yfinance/") for p in paths)
    assert manifest_problems(to_document(manifest)) == []


_RUNUP_SCRIPT = r"""
import json, os, sys
staging, tickers_json = sys.argv[1], sys.argv[2]
os.environ["INVESTING_PLAN_ROOT"] = staging
px_ticker, tier1_ticker = json.loads(tickers_json)

import pandas as pd
from engine.data.features import panel as panel_mod

days = pd.bdate_range("2023-01-02", periods=310)
event_date = days[305]
frame = pd.DataFrame({
    "ticker": [px_ticker, tier1_ticker],
    "date": [event_date, event_date],
    "move": [5.0, 5.0],
    "n_prior": [20, 20],
    "ema12_prior_abs_move": [4.0, 4.0],
    "mean_prior_abs_move": [3.0, 3.0],
})
out = panel_mod.add_runup_features(frame)
print(json.dumps({
    "px_nan": bool(pd.isna(out.loc[out["ticker"] == px_ticker, "dist_high"].iloc[0])),
    "tier1_nan": bool(pd.isna(out.loc[out["ticker"] == tier1_ticker, "dist_high"].iloc[0])),
}))
"""


def test_captured_price_sources_produce_runup_features_not_nan(tmp_path):
    """End-to-end proof for the real defect: stage a private root from ONLY
    ``capture()``'s declared files (no other write to the staged tree), then
    run ``add_runup_features`` (default ``px_dir``, no override -- exactly
    what a real legacy score batch calls) against it in a fresh process so
    ``engine.paths.ROOT`` binds to the staging root. Both the px-file ticker
    and the Tier-1-only ticker must come back with real ``dist_high`` values,
    not NaN -- NaN here is exactly what fed ``size_v1_4`` a NaN input and
    produced ``NO_FORECAST`` for the 6 omitted DYN-SV rows.
    """
    import pandas as pd

    fixture = tmp_path / "fixture"
    fixture.mkdir()
    _build_fixture(fixture)
    px_ticker, tier1_ticker = "PXTICK", "T1TICK"
    days = pd.bdate_range("2023-01-02", periods=310)
    _write_px_csv(fixture, px_ticker, days)
    _write_yfinance_tier1_cache(fixture, tier1_ticker, days)

    manifest = capture(fixture, as_of=SESSION, tickers=[px_ticker, tier1_ticker],
                       year_start=2024, year_end=2024)

    staging = tmp_path / "staging"
    staging.mkdir()
    copy_read_set(fixture, staging, [ref.path for ref in manifest.file_refs])

    result = subprocess.run(
        [sys.executable, "-c", _RUNUP_SCRIPT, str(staging),
         json.dumps([px_ticker, tier1_ticker])],
        cwd=str(ROOT), capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr[-4000:]
    outcome = json.loads(result.stdout.strip().splitlines()[-1])
    assert outcome["px_nan"] is False, "px_<T>.csv source did not produce runup features"
    assert outcome["tier1_nan"] is False, "Tier-1 yfinance fallback did not produce runup features"


# --------------------------------------------------------------------------
# capture(): refusal on a missing required family
# --------------------------------------------------------------------------


def test_capture_refuses_when_no_orats_cache_exists(tmp_path):
    _build_fixture(tmp_path, with_orats=False)
    with pytest.raises(OpsError) as excinfo:
        capture(tmp_path, as_of=SESSION, tickers=[TICKER], year_start=2024, year_end=2024)
    assert excinfo.value.code == "SOURCE_EMPTY"
    assert excinfo.value.problem.details["family"] == "finality_raw_fetch_orats"


# --------------------------------------------------------------------------
# completeness + access audit: legacy_finality in a real subprocess
# --------------------------------------------------------------------------

_FINALITY_SCRIPT = r"""
import builtins, json, os, sys
staging, session, tickers_json = sys.argv[1], sys.argv[2], sys.argv[3]
os.environ["INVESTING_PLAN_ROOT"] = os.path.join(staging, "legacy")
escaped = []
# The whole staging dir, not just staging/legacy: _action_finality WRITES
# finality.json/finality_coverage.json directly under staging (root, not
# root/legacy) -- those opens are legitimate output, not a read escape.
# What must never appear here is a path outside staging entirely (the
# fixture root this staged copy was built from, or the real repo).
staged_root = os.path.realpath(staging)
def _check(path):
    try:
        real = os.path.realpath(str(path))
    except OSError:
        return
    if real.startswith(sys.prefix) or "site-packages" in real or "/usr/lib/python" in real or real.startswith("/usr/share/zoneinfo"):
        return
    if not real.startswith(staged_root):
        escaped.append(real)

_orig_open = builtins.open
def _open(file, *a, **k):
    _check(file)
    return _orig_open(file, *a, **k)
builtins.open = _open

from pathlib import Path as _Path
_orig_path_open = _Path.open
def _path_open(self, *a, **k):
    _check(self)
    return _orig_path_open(self, *a, **k)
_Path.open = _path_open

import pandas as pd
_orig_read_parquet = pd.read_parquet
def _read_parquet(path, *a, **k):
    _check(path)
    return _orig_read_parquet(path, *a, **k)
pd.read_parquet = _read_parquet

import pyarrow.parquet as pq
_orig_parquet_file = pq.ParquetFile
class _ParquetFile(_orig_parquet_file):
    def __init__(self, path, *a, **k):
        _check(path)
        super().__init__(path, *a, **k)
pq.ParquetFile = _ParquetFile

from engine.v2.ops.legacy_adapter import legacy_action
try:
    result = legacy_action("legacy_finality",
                            {"session": session, "tickers": tuple(json.loads(tickers_json))},
                            staging)
    outcome = {"ok": True, "result": result}
except Exception as exc:  # noqa: BLE001 -- report, do not crash the harness
    from engine.v2.ops.errors import OpsError
    if isinstance(exc, OpsError):
        outcome = {"ok": False, "code": exc.code, "message": str(exc)}
    else:
        outcome = {"ok": False, "code": type(exc).__name__, "message": str(exc)}
outcome["escaped_reads"] = sorted(set(escaped))
finality_path = os.path.join(staging, "finality.json")
outcome["finality"] = json.load(open(finality_path)) if os.path.isfile(finality_path) else None
print(json.dumps(outcome))
"""


def _run_finality_subprocess(staging: Path, *, tickers=(TICKER,)):
    result = subprocess.run(
        [sys.executable, "-c", _FINALITY_SCRIPT, str(staging), SESSION, json.dumps(list(tickers))],
        cwd=str(ROOT), capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-4000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_legacy_finality_succeeds_against_a_captured_manifest_only_staged_root(tmp_path):
    """Deliverable 4, required case: stage a private root from ONLY the
    files ``capture()`` declared for a synthetic fixture, run the real
    ``legacy_finality`` action against it, and assert success. The access
    audit (wrapped ``open``/``Path.open``/``pandas.read_parquet``/
    ``pyarrow.parquet.ParquetFile``) proves every read the action performed
    landed inside the staged copy -- nothing escaped to the fixture root or
    anywhere else.
    """
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    _build_fixture(fixture)

    manifest = capture(fixture, as_of=SESSION, tickers=[TICKER], year_start=2024, year_end=2024)
    assert manifest_problems(to_document(manifest), kinds=("legacy_finality",)) == []

    staging = tmp_path / "staging"
    staging.mkdir()
    copy_read_set(fixture, staging / "legacy", [ref.path for ref in manifest.file_refs])

    outcome = _run_finality_subprocess(staging)
    assert outcome["ok"], outcome
    assert outcome["escaped_reads"] == []
    assert outcome["finality"]["is_final"] is True
    assert outcome["finality"]["date"] == SESSION


def test_legacy_finality_fails_without_the_raw_fetch_family(tmp_path):
    """Negative control: drop ``finality_raw_fetch_orats`` from the staged
    read set and the SAME action must fail -- proving the family is
    necessary, not merely present by coincidence in the positive case above.
    """
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    _build_fixture(fixture)
    manifest = capture(fixture, as_of=SESSION, tickers=[TICKER], year_start=2024, year_end=2024)

    staging = tmp_path / "staging"
    staging.mkdir()
    kept = [ref.path for ref in manifest.file_refs if "raw/fetch/orats" not in ref.path]
    assert len(kept) < len(manifest.file_refs)
    copy_read_set(fixture, staging / "legacy", kept)

    outcome = _run_finality_subprocess(staging)
    assert not outcome["ok"]
    assert outcome["code"] == "SOURCE_NOT_FINAL"


# --------------------------------------------------------------------------
# completeness: legacy_decisions (cheap -- degrades on an absent table)
# --------------------------------------------------------------------------

_DECISIONS_SCRIPT = r"""
import json, os, sys
staging, session = sys.argv[1], sys.argv[2]
os.environ["INVESTING_PLAN_ROOT"] = os.path.join(staging, "legacy")
from engine.v2.ops.legacy_adapter import legacy_action
try:
    result = legacy_action("legacy_decisions",
                            {"session": session, "tickers": ()}, staging)
    outcome = {"ok": True, "result": result}
except Exception as exc:  # noqa: BLE001
    outcome = {"ok": False, "code": type(exc).__name__, "message": str(exc)}
print(json.dumps(outcome))
"""


def test_legacy_decisions_succeeds_with_empty_legacy_read_set(tmp_path):
    """Cheap second case (deliverable 4: "others where cheap"). Mirrors
    ``tests/test_v2_ops_supervised_legacy.py``'s own finding that
    ``legacy_decisions`` needs no legacy data at all when there is nothing
    to decide on -- ``_event_ids`` (engine/ledger.py:302-310) degrades to an
    unmatched join against an absent ``earnings_events`` table. Pre-staged
    ``score.json``/``finality.json``/``decision_plan.json`` are the prior
    stages' committed OUTPUTS, never part of the legacy read set the
    manifest declares -- exactly ``_legacy_params``'s own ``input_bindings``
    shape (nightly.py ~161-169).
    """
    staging = tmp_path / "staging"
    (staging / "legacy").mkdir(parents=True)  # empty: earnings_events is simply absent
    score = {"ticker": "FAKE", "event_id": "event-1", "event_date": SESSION,
             "as_of": SESSION, "entry_date": SESSION, "evidence_cutoff": SESSION,
             "strategy": "TWIN-P", "strike": 100.0, "expiry": "2026-10-16",
             "session": "AMC", "snapshot_hash": "sha256:" + "a" * 64}
    (staging / "score.json").write_text(json.dumps({"rows": [score]}))
    (staging / "finality.json").write_text(json.dumps({
        "date": SESSION, "is_final": True, "market_wide": True,
        "daily_share": 1.0, "chain_share": 1.0, "tickers": 0, "covered": 0, "detail": "final"}))
    (staging / "decision_plan.json").write_text(json.dumps({
        "session": SESSION, "decision_clock": SESSION + "T21:00:00+00:00"}))

    result = subprocess.run([sys.executable, "-c", _DECISIONS_SCRIPT, str(staging), SESSION],
                            cwd=str(ROOT), capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr[-4000:]
    outcome = json.loads(result.stdout.strip().splitlines()[-1])
    assert outcome["ok"], outcome


# --------------------------------------------------------------------------
# Coverage-margin follow-up (2026-09-15): engine.v2.data.legacy_nightly_read_plan's
# presence-check dispatch table -- pure functions of (spec, paths, manifest),
# reached only through manifest_problems/_family_present in the tests above,
# which never happens to exercise every branch of every checker. Direct,
# real (no mocking) unit tests against the actual dispatch functions.
# --------------------------------------------------------------------------

from engine.v2.data.legacy_nightly_read_plan import (  # noqa: E402
    _present_ledger_glob,
    _present_price_series_bundle,
    _present_prefixed,
    _present_reference_calendar,
    _present_score_context_bundle,
    _present_single_file,
)


def test_present_score_context_bundle_requires_a_table_and_reference():
    spec = {}
    manifest_full = {"calendar_ref": {"path": "cal"}, "registry_and_model_refs": ({"path": "r"},)}
    assert _present_score_context_bundle(
        spec, ["data/curated/daily_market/year=2024/part-0.parquet"], manifest_full) is True
    # No matching curated table path at all: absent even with a full reference bundle.
    assert _present_score_context_bundle(spec, ["some/other/path.csv"], manifest_full) is False
    # A table path present, but no reference bundle: still absent.
    assert _present_score_context_bundle(
        spec, ["data/curated/daily_market/year=2024/part-0.parquet"], {}) is False


def test_present_reference_calendar_checks_the_manifest_field_only():
    assert _present_reference_calendar({}, [], {"calendar_ref": {"path": "cal"}}) is True
    assert _present_reference_calendar({}, [], {}) is False
    assert _present_reference_calendar({}, [], {"calendar_ref": None}) is False


def test_present_prefixed_matches_an_explicit_or_derived_prefix():
    explicit = {"path_prefix": "data/raw/fetch/orats/"}
    assert _present_prefixed(explicit, ["data/raw/fetch/orats/2026-09-10.meta.json"], {}) is True
    assert _present_prefixed(explicit, ["data/raw/fetch/polygon/x.json"], {}) is False
    derived = {"table": "daily_market"}
    assert _present_prefixed(derived, ["data/curated/daily_market/year=2024/part-0.parquet"], {}) is True
    assert _present_prefixed(derived, [], {}) is False


def test_present_ledger_glob_matches_the_declared_directory():
    spec = {"directory": "ledger/predictions"}
    assert _present_ledger_glob(spec, ["ledger/predictions/2026-09-10.jsonl"], {}) is True
    assert _present_ledger_glob(spec, ["ledger/outcomes/2026-09-10.jsonl"], {}) is False


def test_present_single_file_is_vacuously_true_when_not_required():
    optional = {"path": "data/features/model_evidence.json", "required": False}
    assert _present_single_file(optional, [], {}) is True
    required = {"path": "data/features/model_evidence.json"}
    assert _present_single_file(required, ["data/features/model_evidence.json"], {}) is True
    assert _present_single_file(required, ["some/other/file.json"], {}) is False


def test_present_price_series_bundle_accepts_either_source_when_required():
    forced_required = {"path_prefix": None, "required": True}
    assert _present_price_series_bundle(
        forced_required, ["earnings_predictions/data/raw/yfinance/px_AAPL.csv"], {}) is True
    assert _present_price_series_bundle(
        forced_required, ["data/raw/fetch/yfinance/history.body.gz"], {}) is True
    assert _present_price_series_bundle(forced_required, ["unrelated/path.csv"], {}) is False


def test_required_families_only_required_drops_optional_families():
    every_family = required_families("legacy_settlement")
    required_only = required_families("legacy_settlement", only_required=True)
    assert set(required_only) <= set(every_family)
    assert "ledger_outcomes" not in required_only  # required: False in FAMILIES
    assert "ledger_outcomes" in every_family

