"""The Phase 3 dashboard pipeline: renderer, self-check, publisher, nightly parts.

Built against a synthetic board — hand-made ``ScoreResult`` rows and a fake
scorer — so what is under test is the pipeline's own logic rather than the
state of the real store. The integration properties (that a real board renders,
that the real engine agrees with it) are the acceptance layer's job, in
``checks/phase3_checks.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import engine.dashboard.publish as publish_mod
from engine.dashboard.render import (
    BOARD_MAX_BYTES,
    compact_row,
    render_bundle,
    row_digest,
)
from engine.dashboard.selfcheck import reconstruct_request, selfcheck
from engine.score import ScoreResult, unscorable_result

AS_OF = pd.Timestamp("2026-08-10")
EVENT = pd.Timestamp("2026-08-12")


# --------------------------------------------------------------------------
# a synthetic board
# --------------------------------------------------------------------------


def _result(ticker="AAA", strategy="STR-THRU", **kwargs) -> ScoreResult:
    """One fully-populated score, with every field the board displays."""
    base = dict(
        as_of=AS_OF,
        event_date=EVENT,
        session="AMC",
        exp_pnl_model=0.031,
        win_model=0.56,
        model_p10=-0.22,
        model_p90=0.41,
        exp_pnl_analog=0.028,
        win_analog=0.54,
        ci_low=-0.01,
        ci_high=0.07,
        n_analogs=120,
        gate_score=0.71,
        gate_threshold=0.6,
        gate_pass=True,
        entry_date=EVENT,
        exit_date=EVENT + pd.Timedelta(days=1),
        strike=100.0,
        expiry=EVENT + pd.Timedelta(days=2),
        entry_cost=6.0,
        spot=100.0,
        dte_entry=2,
        payoff={"intercept": 0.01, "slope": 0.006, "n": 400},
        driver_name="abs_move",
        driver_prediction=7.5,
        model_versions={"size": "size_v13"},
        snapshot_hash="snap-test",
    )
    base.update(kwargs)
    return ScoreResult(ticker=ticker, strategy=strategy, **base)


@pytest.fixture
def scores() -> pd.DataFrame:
    """Three rows whose SHAPES differ — the case that broke the digest.

    One fully-scored row, one ATM row with no strike or gate, and one NO_CHAIN
    placeholder. A DataFrame holding all three turns the missing values into
    ``NaN``; a board of identically-shaped rows would never notice.
    """
    rows = [
        _result().as_dict() | {"strike_offset": None},
        _result(
            ticker="BBB", strike=None, expiry=None, gate_score=None,
            gate_threshold=None, gate_pass=None, exp_pnl_model=-0.02,
            n_analogs=8, flags=["THIN_ANALOGS"],
        ).as_dict() | {"strike_offset": None},
        ScoreResult(
            ticker="CCC", strategy="STR-RUNUP", as_of=AS_OF, event_date=EVENT,
            snapshot_hash="snap-test", detail=str(KeyError("no chain for CCC")),
            flags=["NO_CHAIN"],
        ).as_dict() | {"strike_offset": None},
    ]
    return pd.DataFrame(rows)


class FakeScorer:
    """Replays the same results the board was rendered from, by row identity."""

    def __init__(self, results, snapshot="snap-test"):
        self.snapshot = snapshot
        self._by_key = {(r.ticker, r.strategy): r for r in results}

    def score(self, request, chain_index=None):
        result = self._by_key.get((request.ticker, request.strategy))
        if result is None:
            raise KeyError(f"no chain for {request.ticker}")
        return result


# --------------------------------------------------------------------------
# digest canonicalization
# --------------------------------------------------------------------------


class TestRowDigest:
    def test_survives_the_dataframe_round_trip(self, scores):
        """None → NaN is not a change of score, and must not read as one.

        A board whose rows have different shapes gets ``NaN`` in place of the
        missing values the moment it becomes a DataFrame. Hashing that
        difference would fail the self-check every night on any real board.
        """
        results = [_result(), _result(ticker="BBB", strike=None, expiry=None,
                                      gate_score=None, gate_threshold=None,
                                      gate_pass=None, exp_pnl_model=-0.02,
                                      n_analogs=8, flags=["THIN_ANALOGS"])]
        records = scores.to_dict(orient="records")[:2]
        for record, result in zip(records, results):
            assert row_digest(record) == row_digest(result.as_dict())

    def test_still_catches_a_changed_number(self):
        a = _result()
        b = _result(exp_pnl_model=0.032)
        assert row_digest(a.as_dict()) != row_digest(b.as_dict())

    def test_ignores_strike_offset(self):
        record = _result().as_dict()
        assert row_digest(record | {"strike_offset": 0.025}) == row_digest(record)

    def test_numpy_scalars_hash_as_their_python_values(self):
        record = _result().as_dict()
        widened = record | {"n_analogs": np.int64(record["n_analogs"]),
                            "gate_pass": np.bool_(True)}
        assert row_digest(widened) == row_digest(record)


# --------------------------------------------------------------------------
# derived display values
# --------------------------------------------------------------------------


class TestCompactRow:
    def test_derives_the_premium_comparison(self):
        record = _result().as_dict()
        row = compact_row(record, rank=1)
        assert row["entry_cost_pct"] == pytest.approx(6.0)
        # exit_value/spot = 0.01 + 0.006 * 7.5 = 0.055 → 5.5% of spot
        assert row["model_fair_pct"] == pytest.approx(5.5)
        assert row["premium_vs_fair"] == pytest.approx(6.0 / 5.5)
        assert row["rank"] == 1

    def test_missing_payoff_leaves_the_comparison_empty(self):
        row = compact_row(_result(payoff={}).as_dict())
        assert row["model_fair_pct"] is None
        assert row["premium_vs_fair"] is None

    def test_carries_only_board_fields(self):
        row = compact_row(_result().as_dict())
        assert "payoff" not in row and "analog_buckets" not in row
        assert row["digest"] and row["row_id"].startswith("AAA|STR-THRU|")


# --------------------------------------------------------------------------
# the bundle
# --------------------------------------------------------------------------


class TestRenderBundle:
    def test_writes_a_complete_bundle(self, tmp_path, scores):
        summary = render_bundle(scores, tmp_path / "b", as_of=AS_OF)
        out = tmp_path / "b"
        for name in ("index.html", "assets/app.js", "assets/app.css",
                     "data/board.json", "data/meta.json", "data/health.json",
                     "data/flags.json"):
            assert (out / name).exists(), name
        assert summary["n_rows"] == 3
        assert summary["n_tickers"] == 3
        assert not summary["board_oversized"]
        assert summary["board_bytes"] < BOARD_MAX_BYTES

    def test_json_is_strict(self, tmp_path, scores):
        """No NaN in the bundle: a strict parser downstream must not choke."""
        render_bundle(scores, tmp_path / "b", as_of=AS_OF)
        for path in (tmp_path / "b" / "data").rglob("*.json"):
            text = path.read_text()
            assert "NaN" not in text and "Infinity" not in text
            json.loads(text)  # raises if the file is not strict JSON

    def test_js_wrapper_matches_its_json(self, tmp_path, scores):
        """The offline path and the API path carry the same bytes."""
        render_bundle(scores, tmp_path / "b", as_of=AS_OF)
        data = tmp_path / "b" / "data"
        for stem, var in (("board", "BOARD"), ("meta", "META"),
                          ("health", "HEALTH"), ("flags", "FLAGS")):
            js = (data / f"{stem}.js").read_text()
            assert js.startswith(f"window.{var} = ") and js.rstrip().endswith(";")
            payload = js[len(f"window.{var} = "):].rstrip().rstrip(";")
            assert json.loads(payload) == json.loads((data / f"{stem}.json").read_text())

    def test_per_ticker_files_hold_the_evidence(self, tmp_path, scores):
        panel = pd.DataFrame(
            {"ticker": ["AAA", "AAA"], "date": pd.to_datetime(["2026-02-10", "2026-05-11"]),
             "k": [30, 31], "implied_move": [6.0, 6.5], "or_implied": [6.1, 6.4],
             "move": [-4.0, 3.0], "abs_move": [4.0, 3.0]}
        )
        trades = pd.DataFrame(
            {"ticker": ["AAA"], "strategy": ["STR-THRU"],
             "event_date": pd.to_datetime(["2026-05-11"]),
             "entry_date": pd.to_datetime(["2026-05-11"]),
             "exit_date": pd.to_datetime(["2026-05-12"]),
             "fill_alpha": [0.5], "entry_cost": [5.0], "exit_value": [5.6], "ret": [0.12]}
        )
        render_bundle(scores, tmp_path / "b", as_of=AS_OF, panel=panel, trades=trades)
        payload = json.loads((tmp_path / "b" / "data" / "tickers" / "AAA.json").read_text())
        assert payload["ticker"] == "AAA"
        assert len(payload["events"]) == 1 and payload["events"][0]["rows"]
        assert [h["abs_move"] for h in payload["history"]] == [3.0, 4.0]  # newest first
        assert payload["analogs"][0]["ret"] == pytest.approx(0.12)

    def test_ranks_within_a_strategy(self, tmp_path):
        rows = [
            _result(ticker="LOW", gate_pass=False, gate_score=0.1).as_dict() | {"strike_offset": None},
            _result(ticker="HIGH", gate_pass=True, gate_score=0.9).as_dict() | {"strike_offset": None},
        ]
        render_bundle(pd.DataFrame(rows), tmp_path / "b", as_of=AS_OF)
        board = json.loads((tmp_path / "b" / "data" / "board.json").read_text())
        ranks = {r["ticker"]: r["rank"] for r in board["rows"]}
        assert ranks == {"HIGH": 1, "LOW": 2}

    def test_rerender_drops_stale_ticker_files(self, tmp_path, scores):
        render_bundle(scores, tmp_path / "b", as_of=AS_OF)
        assert (tmp_path / "b" / "data" / "tickers" / "CCC.json").exists()
        render_bundle(scores[scores["ticker"] != "CCC"], tmp_path / "b", as_of=AS_OF)
        assert not (tmp_path / "b" / "data" / "tickers" / "CCC.json").exists()


# --------------------------------------------------------------------------
# the self-check
# --------------------------------------------------------------------------


class TestSelfCheck:
    def test_green_when_the_bundle_matches_the_engine(self, tmp_path, scores):
        render_bundle(scores, tmp_path / "b", as_of=AS_OF,
                      meta={"as_of": str(AS_OF.date()), "snapshot_hash": "snap-test"})
        results = [_result(), _result(ticker="BBB", strike=None, expiry=None,
                                      gate_score=None, gate_threshold=None,
                                      gate_pass=None, exp_pnl_model=-0.02,
                                      n_analogs=8, flags=["THIN_ANALOGS"])]
        report = selfcheck(tmp_path / "b", scorer=FakeScorer(results))
        assert report.ok, report.mismatches
        assert report.n_checked == 3

    def test_a_poisoned_row_is_caught(self, tmp_path, scores):
        render_bundle(scores, tmp_path / "b", as_of=AS_OF,
                      meta={"as_of": str(AS_OF.date()), "snapshot_hash": "snap-test"})
        board_path = tmp_path / "b" / "data" / "board.json"
        board = json.loads(board_path.read_text())
        board["rows"][0]["exp_pnl_model"] = 0.99
        board_path.write_text(json.dumps(board))

        report = selfcheck(tmp_path / "b", scorer=FakeScorer([_result()]))
        assert not report.ok
        assert any(m["reason"] == "digest mismatch" for m in report.mismatches)

    def test_a_tamper_that_keeps_the_digest_is_still_caught(self, tmp_path, scores):
        """The digest travels in the row; only the field diff catches an edit
        that rewrites the digest to match its own lie."""
        render_bundle(scores, tmp_path / "b", as_of=AS_OF,
                      meta={"as_of": str(AS_OF.date()), "snapshot_hash": "snap-test"})
        board_path = tmp_path / "b" / "data" / "board.json"
        board = json.loads(board_path.read_text())
        row = board["rows"][0]
        row["win_model"] = 0.95
        board_path.write_text(json.dumps(board))

        # Re-score returns the true result; digest still matches (the edit did
        # not touch a digested copy), so the field diff must do the work.
        results = [_result()]
        report = selfcheck(tmp_path / "b", scorer=FakeScorer(results))
        assert not report.ok
        reasons = {m["reason"] for m in report.mismatches}
        assert reasons & {"digest mismatch", "display values diverge from the engine"}

    def test_snapshot_drift_is_red(self, tmp_path, scores):
        render_bundle(scores, tmp_path / "b", as_of=AS_OF,
                      meta={"as_of": str(AS_OF.date()), "snapshot_hash": "snap-test"})
        report = selfcheck(tmp_path / "b",
                           scorer=FakeScorer([_result()], snapshot="snap-other"))
        assert not report.ok and not report.snapshot_ok

    def test_no_chain_rows_reconcile_against_the_same_placeholder(self, tmp_path, scores):
        """An unpriceable row must verify, not read as a re-score failure."""
        render_bundle(scores, tmp_path / "b", as_of=AS_OF,
                      meta={"as_of": str(AS_OF.date()), "snapshot_hash": "snap-test"})
        board = json.loads((tmp_path / "b" / "data" / "board.json").read_text())
        row = next(r for r in board["rows"] if r["ticker"] == "CCC")
        request = reconstruct_request(row)
        placeholder = unscorable_result(
            request, as_of=AS_OF, snapshot="snap-test",
            exc=KeyError("no chain for CCC"),
        )
        assert row_digest(placeholder.as_dict()) == row["digest"]

    def test_missing_board_is_red(self, tmp_path):
        report = selfcheck(tmp_path / "nothing")
        assert not report.ok and report.n_checked == 0


# --------------------------------------------------------------------------
# the publisher
# --------------------------------------------------------------------------


@pytest.fixture
def bundle(tmp_path, scores) -> Path:
    render_bundle(scores, tmp_path / "b", as_of=AS_OF)
    return tmp_path / "b"


class TestPublish:
    def test_atomic_flip(self, tmp_path, bundle):
        target = tmp_path / "pub"
        result = publish_mod.LocalPublisher(target).publish(bundle)
        current = target / "current"
        assert current.is_symlink()
        assert json.loads((current / "data" / "meta.json").read_text())["as_of"] == "2026-08-10"
        assert result.as_of == "2026-08-10"

    def test_a_kill_mid_upload_leaves_the_previous_release_serving(self, tmp_path, bundle, scores):
        target = tmp_path / "pub"
        publish_mod.LocalPublisher(target).publish(bundle)
        first = (target / "current").resolve()

        render_bundle(scores, bundle, as_of=AS_OF + pd.Timedelta(days=1))

        def die(path):
            if path.name == "board.json":
                raise OSError("simulated kill mid-upload")

        with pytest.raises(OSError):
            publish_mod.LocalPublisher(target, copy_hook=die).publish(bundle)

        assert (target / "current").resolve() == first
        served = json.loads((target / "current" / "data" / "meta.json").read_text())
        assert served["as_of"] == "2026-08-10"  # unchanged

    def test_secret_scan_refuses(self, tmp_path, bundle):
        (bundle / "data" / "leak.json").write_text('{"api_key": "hunter2hunter2"}')
        with pytest.raises(publish_mod.PublishError, match="secret scan"):
            publish_mod.LocalPublisher(tmp_path / "pub").publish(bundle)

    def test_a_clean_bundle_scans_clean(self, bundle):
        assert publish_mod.secret_scan(bundle) == []

    def test_incomplete_bundle_refuses(self, tmp_path, bundle):
        (bundle / "data" / "board.json").unlink()
        with pytest.raises(publish_mod.PublishError, match="incomplete"):
            publish_mod.LocalPublisher(tmp_path / "pub").publish(bundle)

    def test_a_publicly_readable_target_refuses(self, bundle, monkeypatch):
        """The hard rule: nothing ships to a target without Access in front."""
        monkeypatch.setattr(
            publish_mod, "access_probe",
            lambda url, **kw: {"status": 200, "public": True, "error": None},
        )
        publisher = publish_mod.CommandPublisher(
            "true {bundle}", probe_url="https://example.invalid/board"
        )
        with pytest.raises(publish_mod.PublishError, match="publicly readable"):
            publisher.publish(bundle)

    def test_a_login_redirect_is_not_proof_of_publicness(self, bundle, monkeypatch):
        monkeypatch.setattr(
            publish_mod, "access_probe",
            lambda url, **kw: {"status": 302, "public": False, "error": None},
        )
        publisher = publish_mod.CommandPublisher(
            "true {bundle}", probe_url="https://example.invalid/board"
        )
        assert publisher.publish(bundle).probe["public"] is False

    def test_old_releases_are_pruned(self, tmp_path, bundle):
        target = tmp_path / "pub"
        for i in range(publish_mod.KEEP_RELEASES + 3):
            render_bundle(pd.DataFrame([_result().as_dict() | {"strike_offset": None}]),
                          bundle, as_of=AS_OF + pd.Timedelta(days=i))
            publish_mod.LocalPublisher(target).publish(bundle)
        releases = list((target / "releases").iterdir())
        assert len(releases) == publish_mod.KEEP_RELEASES
        assert (target / "current").resolve().exists()


# --------------------------------------------------------------------------
# nightly parts that need no store
# --------------------------------------------------------------------------


class TestNightlyPieces:
    def test_state_round_trips(self, tmp_path):
        from engine.dashboard import nightly

        nightly._write_state(tmp_path / "earnings", {"last_successful_as_of": "2026-08-10"})
        assert nightly._read_state(tmp_path / "earnings")["last_successful_as_of"] == "2026-08-10"

    def test_unreadable_state_is_empty_not_fatal(self, tmp_path):
        from engine.dashboard import nightly

        path = nightly._state_path(tmp_path / "earnings")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json")
        assert nightly._read_state(tmp_path / "earnings") == {}

    def test_date_change_flag_on_a_moved_print(self):
        from engine.dashboard import nightly

        previous = [{"ticker": "AAA", "event_date": "2026-08-12", "session": "AMC"}]
        current = pd.DataFrame(
            {"ticker": ["AAA"], "event_date": [pd.Timestamp("2026-08-19")], "session": ["AMC"]}
        )
        flag = nightly._date_change_flag(previous, current, as_of=AS_OF)
        assert flag and flag["kind"] == "earnings_date_changed"
        assert flag["changes"][0]["ticker"] == "AAA"

    def test_no_date_change_flag_when_nothing_moved(self):
        from engine.dashboard import nightly

        previous = [{"ticker": "AAA", "event_date": "2026-08-12", "session": "AMC"}]
        current = pd.DataFrame(
            {"ticker": ["AAA"], "event_date": [pd.Timestamp("2026-08-12")], "session": ["AMC"]}
        )
        assert nightly._date_change_flag(previous, current, as_of=AS_OF) is None

    def test_strike_ladder_only_prices_gate_passers(self):
        """The expensive rows go where someone might act, not everywhere."""
        from engine.dashboard import nightly

        board = pd.DataFrame([
            _result(ticker="PASS", gate_pass=True).as_dict() | {"strike_offset": None},
            _result(ticker="FAIL", gate_pass=False).as_dict() | {"strike_offset": None},
            _result(ticker="NONE", gate_pass=None).as_dict() | {"strike_offset": None},
        ])
        asked: list[tuple[str, float]] = []

        class Recorder:
            snapshot = "snap-test"

            def score(self, request, chain_index=None):
                asked.append((request.ticker, request.strike))
                return _result(ticker=request.ticker, strike=request.strike)

        rows = nightly.strike_ladder(board, scorer=Recorder(), alt_strikes=1, as_of=AS_OF)
        assert {t for t, _ in asked} == {"PASS"}
        assert sorted(round(s, 4) for _, s in asked) == [97.5, 102.5]
        assert [r["strike_offset"] for r in rows] == [-0.025, 0.025]

    def test_strike_ladder_is_off_at_zero(self):
        from engine.dashboard import nightly

        board = pd.DataFrame([_result(gate_pass=True).as_dict() | {"strike_offset": None}])

        class Boom:
            snapshot = "snap-test"

            def score(self, request, chain_index=None):
                raise AssertionError("must not score anything")

        assert nightly.strike_ladder(board, scorer=Boom(), alt_strikes=0, as_of=AS_OF) == []

    def test_strike_ladder_keeps_unpriceable_rows_as_placeholders(self):
        from engine.dashboard import nightly

        board = pd.DataFrame([_result(gate_pass=True).as_dict() | {"strike_offset": None}])

        class NoChain:
            snapshot = "snap-test"

            def score(self, request, chain_index=None):
                raise KeyError("no chain at that strike")

        rows = nightly.strike_ladder(board, scorer=NoChain(), alt_strikes=1, as_of=AS_OF)
        assert len(rows) == 2
        assert all(r["flags"] == ["NO_CHAIN"] for r in rows)


def test_no_package_export_shadows_a_submodule():
    """A submodule and an export of the same name is a trap, not a style point.

    `from engine.dashboard import publish` once returned the FUNCTION where the
    caller wanted the module — and the nightly job's `except publish.PublishError`
    then raised AttributeError at exactly the moment a publish failed.
    """
    import importlib
    import pkgutil

    import engine.dashboard as package

    submodules = {m.name for m in pkgutil.iter_modules(package.__path__)}
    clashes = sorted(submodules & set(package.__all__))
    assert not clashes, f"package exports shadow submodules: {clashes}"
    for name in submodules:
        module = importlib.import_module(f"engine.dashboard.{name}")
        assert getattr(package, name) is module, f"engine.dashboard.{name} is not the module"


class TestPublishTarget:
    def test_defaults_to_the_local_publisher(self, monkeypatch):
        from engine import paths
        from engine.dashboard import nightly

        monkeypatch.delenv("DASHBOARD_PUBLISH_CMD", raising=False)
        assert nightly._default_target() == paths.ROOT / "dashboard" / "published"

    def test_uses_the_configured_command(self, monkeypatch):
        from engine.dashboard import nightly

        monkeypatch.setenv("DASHBOARD_PUBLISH_CMD", "wrangler pages deploy {bundle}")
        assert nightly._default_target() == "wrangler pages deploy {bundle}"

    def test_a_command_without_the_placeholder_is_ignored(self, monkeypatch):
        """A malformed template must not silently become the target: the bundle
        would never reach the command, and the night would report a publish."""
        from engine import paths
        from engine.dashboard import nightly

        monkeypatch.setenv("DASHBOARD_PUBLISH_CMD", "wrangler pages deploy")
        assert nightly._default_target() == paths.ROOT / "dashboard" / "published"


class TestJsonSafe:
    """The one normalizer both the bundle and the ledger write through."""

    def test_missing_values_become_null(self):
        from engine.jsonio import json_safe

        assert json_safe(float("nan")) is None
        assert json_safe(float("inf")) is None
        assert json_safe(pd.NaT) is None
        assert json_safe(np.float64("nan")) is None

    def test_numpy_scalars_become_python(self):
        from engine.jsonio import json_safe

        out = json_safe({"i": np.int64(3), "b": np.bool_(True), "f": np.float64(1.5)})
        assert out == {"i": 3, "b": True, "f": 1.5}
        assert [type(v) for v in out.values()] == [int, bool, float]

    def test_rounding_is_opt_in(self):
        from engine.jsonio import json_safe

        assert json_safe(1 / 3) == pytest.approx(1 / 3)
        assert json_safe(1 / 3, round_to=6) == 0.333333

    def test_output_is_strict_json(self):
        from engine.jsonio import json_safe

        payload = {"a": [np.nan, pd.Timestamp("2026-08-30"), np.int64(2)], "b": {"c": np.inf}}
        text = json.dumps(json_safe(payload), allow_nan=False)
        assert json.loads(text) == {"a": [None, "2026-08-30", 2], "b": {"c": None}}


class TestForwardCalendarRefresh:
    """The pull that fills the board. ORATS /hist/earnings cannot: it is a
    history endpoint, so the upcoming calendar comes from Nasdaq (one call per
    date, whole market) plus yfinance (per ticker, for the session)."""

    class FakeFetcher:
        """Records every call and answers from a scripted map."""

        def __init__(self, nasdaq_by_date=None, yf_status=200):
            self.calls = []
            self.nasdaq_by_date = nasdaq_by_date or {}
            self.yf_status = yf_status

        def fetch(self, source, endpoint, params=None, *, live=False, note=""):
            self.calls.append((source, endpoint, dict(params or {})))
            params = params or {}
            if source == "nasdaq":
                rows = self.nasdaq_by_date.get(params.get("date"), [])
                body = {"data": {"rows": rows}}
                return type("R", (), {
                    "from_cache": False, "meta": {"status": 200},
                    "json": lambda self, b=body: b,
                })()
            return type("R", (), {
                "from_cache": False, "meta": {"status": self.yf_status},
                "json": lambda self: {},
            })()

    def _row(self, symbol, time="time-not-supplied"):
        return {"symbol": symbol, "time": time, "marketCap": "1000000000"}

    def test_one_nasdaq_call_per_trading_day(self, monkeypatch):
        """The whole market for a date in one call is the entire reason this
        source was chosen over a per-ticker one."""
        from engine.data.pulls import forward_calendar as fc

        fetcher = self.FakeFetcher()
        result = fc.refresh_forward_calendar(
            "2026-08-31", horizon_days=7, fetcher=fetcher,
            confirm_sessions=False, rebuild_events=False,
        )
        nasdaq_calls = [c for c in fetcher.calls if c[0] == "nasdaq"]
        assert len(nasdaq_calls) == len(result.dates)
        assert all(c[1] == "calendar/earnings" for c in nasdaq_calls)
        # Weekends are not trading days and must not cost a call.
        assert all(pd.Timestamp(d).weekday() < 5 for d in result.dates)

    def test_yfinance_runs_only_where_the_session_is_unknown(self):
        from engine.data.pulls import forward_calendar as fc

        fetcher = self.FakeFetcher(nasdaq_by_date={
            "2026-08-31": [self._row("KNOWN", "time-after-hours"), self._row("UNKNOWN")],
        })
        result = fc.refresh_forward_calendar(
            "2026-08-31", horizon_days=0, fetcher=fetcher, rebuild_events=False,
        )
        asked = [c[2]["ticker"] for c in fetcher.calls if c[0] == "yfinance"]
        assert asked == ["UNKNOWN"]
        assert result.session_from_nasdaq == 1 and result.session_missing == 1

    def test_confirmations_are_capped_and_soonest_first(self):
        """It is a time budget: ~0.9s a name. What gets dropped must be the
        events furthest out, which have the most nights left to be filled."""
        from engine.data.pulls import forward_calendar as fc

        fetcher = self.FakeFetcher(nasdaq_by_date={
            "2026-08-31": [self._row("SOON")],
            "2026-09-01": [self._row("LATER")],
        })
        result = fc.refresh_forward_calendar(
            "2026-08-31", horizon_days=2, fetcher=fetcher,
            max_confirmations=1, rebuild_events=False,
        )
        asked = [c[2]["ticker"] for c in fetcher.calls if c[0] == "yfinance"]
        assert asked == ["SOON"]
        assert result.truncated

    def test_a_failing_date_does_not_lose_the_others(self):
        """The calendar is additive: today's failure is retried tomorrow."""
        from engine.data.pulls import forward_calendar as fc

        class Flaky(self.FakeFetcher):
            def fetch(self, source, endpoint, params=None, **kw):
                if source == "nasdaq" and (params or {}).get("date") == "2026-09-01":
                    raise RuntimeError("nasdaq said no")
                return super().fetch(source, endpoint, params, **kw)

        fetcher = Flaky(nasdaq_by_date={"2026-08-31": [self._row("AAA", "time-pre-market")]})
        result = fc.refresh_forward_calendar(
            "2026-08-31", horizon_days=2, fetcher=fetcher,
            confirm_sessions=False, rebuild_events=False,
        )
        assert result.rows_seen == 1
        assert [f["date"] for f in result.nasdaq_failed] == ["2026-09-01"]

    def test_live_true_makes_a_same_day_rerun_free(self):
        """One cache entry per source per day: idempotent tonight, and fresh
        tomorrow when Nasdaq has firmed up more sessions."""
        from engine.data.pulls import forward_calendar as fc

        seen = {}

        class Recorder(self.FakeFetcher):
            def fetch(self, source, endpoint, params=None, *, live=False, note=""):
                seen[(source, endpoint)] = live
                return super().fetch(source, endpoint, params, live=live, note=note)

        fetcher = Recorder(nasdaq_by_date={"2026-08-31": [self._row("AAA")]})
        fc.refresh_forward_calendar("2026-08-31", horizon_days=0, fetcher=fetcher,
                                    rebuild_events=False)
        assert seen[("nasdaq", "calendar/earnings")] is True
        assert seen[("yfinance", "earnings")] is True


class TestNasdaqAdapter:
    def test_session_mapping_never_guesses(self):
        """`time-not-supplied` must stay NULL. The scorer skips events with no
        session, and a wrong one shifts every entry and exit by a day."""
        from engine.data.sources.nasdaq import SESSION_BY_TIME

        assert SESSION_BY_TIME["time-pre-market"] == "BMO"
        assert SESSION_BY_TIME["time-after-hours"] == "AMC"
        assert SESSION_BY_TIME.get("time-not-supplied") is None
        assert SESSION_BY_TIME.get("") is None

    def test_a_403_is_not_a_rotated_credential(self):
        """There is no key to rotate. Raising CredentialRotated would stop the
        nightly for what is actually a user-agent fix."""
        from engine.data.sources.base import Response
        from engine.data.sources.nasdaq import NasdaqAdapter

        adapter = NasdaqAdapter()
        assert adapter.is_auth_failure(Response(status=403, body=b"")) is False
        assert adapter.quota_from(Response(status=200, body=b"")) is None

    def test_url_shape(self):
        from engine.data.sources.nasdaq import NasdaqAdapter

        url = NasdaqAdapter().build_url("calendar/earnings", {"date": "2026-09-02"})
        assert url == "https://api.nasdaq.com/api/calendar/earnings?date=2026-09-02"


class TestSingleFile:
    """The one-file package. It must be the bundle, not a second renderer."""

    def test_it_inlines_everything_and_references_nothing(self, tmp_path, scores):
        import re

        from engine.dashboard.render import write_single_file

        render_bundle(scores, tmp_path / "b", as_of=AS_OF)
        out = write_single_file(tmp_path / "b", tmp_path / "one.html")
        text = out.read_text()
        refs = [
            u for u in re.findall(r'(?:src|href)="(?!#)([^"]+)"', text)
            if not u.startswith("data:")  # inlined content, not a reference
        ]
        assert not refs, f"a single file must reference nothing: {refs}"
        assert "<style>" in text and "window.BOARD" in text

    def test_the_data_is_the_bundle_byte_for_byte(self, tmp_path, scores):
        """If this can drift from the bundle it is a second rendering path,
        which is the one thing the whole design forbids."""
        from engine.dashboard.render import write_single_file

        render_bundle(scores, tmp_path / "b", as_of=AS_OF)
        text = write_single_file(tmp_path / "b", tmp_path / "one.html").read_text()
        for stem, var in (("board", "BOARD"), ("meta", "META")):
            payload = json.loads((tmp_path / "b" / "data" / f"{stem}.json").read_text())
            start = text.index(f"window.{var} = ") + len(f"window.{var} = ")
            end = text.index(";\n", start)
            assert json.loads(text[start:end]) == payload

    def test_every_ticker_is_inlined_so_the_explorer_works_offline(self, tmp_path, scores):
        from engine.dashboard.render import write_single_file

        render_bundle(scores, tmp_path / "b", as_of=AS_OF)
        text = write_single_file(tmp_path / "b", tmp_path / "one.html").read_text()
        expected = {p.stem for p in (tmp_path / "b" / "data" / "tickers").glob("*.json")}
        for ticker in expected:
            assert f'window.TICKER_DATA["{ticker}"]' in text


class TestDerivation:
    """The audit trail: what went in, which model, and how it became a number."""

    def test_model_inputs_are_recorded_even_when_the_row_declines(self):
        """The row that could not score is exactly the one where someone needs
        to see what went in — recording inputs only on success hides that."""
        from engine.score import ScoreResult

        result = ScoreResult(ticker="T", strategy="STR-THRU", as_of=AS_OF)
        assert result.model_inputs == {}
        assert result.model_input_as_of is None
        # The field exists on the dataclass, so a declining row can carry it.
        assert "model_inputs" in result.as_dict()

    def test_strategies_payload_describes_every_structure(self):
        from engine.dashboard.render import build_strategies
        from engine.structures import STRUCTURES

        out = build_strategies()
        assert set(out) == set(STRUCTURES)
        thru = out["STR-THRU"]
        assert thru["driver"] == "abs_move"
        assert thru["driver_note"]
        assert [leg["right"] for leg in thru["structure"]["legs"]] == ["call", "put"]
        assert thru["structure"]["entry_note"] and thru["structure"]["exit_note"]

    def test_a_disabled_strategy_says_why(self):
        from engine.dashboard.render import build_strategies

        calp = build_strategies()["CAL-P"]
        assert calp["enabled"] is False
        assert "EXP-046b" in (calp["disabled_reason"] or "") or calp["disabled_reason"]

    def test_offset_notes_are_plain_english(self):
        from engine.dashboard.render import _offset_note

        assert _offset_note(0) == "the last close before the announcement"
        assert _offset_note(1) == "the first close after the announcement"
        assert "14 trading days before" in _offset_note(-14)

    def test_the_bundle_carries_the_strategies_payload(self, tmp_path, scores):
        render_bundle(scores, tmp_path / "b", as_of=AS_OF)
        payload = json.loads((tmp_path / "b" / "data" / "strategies.json").read_text())
        assert payload, "the derivation view has nothing to render"
        js = (tmp_path / "b" / "data" / "strategies.js").read_text()
        assert js.startswith("window.STRATEGIES = ")

    def test_every_champion_feature_carries_a_note(self):
        """A bare `fexern90_30` explains nothing to anyone who did not build it."""
        from engine.dashboard.render import build_strategies

        for name, block in build_strategies().items():
            for role in ("model", "gate"):
                entry = block.get(role)
                if not entry:
                    continue
                unlabelled = [f["name"] for f in entry["features"] if not f["note"]]
                assert not unlabelled, f"{name}/{role} has undocumented inputs: {unlabelled}"


class TestFeatureNotes:
    def test_lag_and_ema_families_are_derived_not_listed(self):
        """So a new lag or window documents itself instead of arriving bare."""
        from engine.features import feature_note

        assert "5 observations" in feature_note("im_d5")
        assert "span 8" in feature_note("ema8_prior_abs_move")
        assert "absolute" in feature_note("ema8_prior_abs_move")
        assert "signed" in feature_note("ema4_prior_move")

    def test_an_undocumented_feature_gets_no_invented_explanation(self):
        from engine.features import feature_note

        assert feature_note("some_new_thing") == ""


class TestModelEvidence:
    """What each input does to the output, on the model's own training set."""

    def test_feature_stats_report_shape_not_just_a_coefficient(self):
        """A correlation cannot show whether a relationship is monotone or
        driven by one tail; the decile table is the part that can."""
        from engine.dashboard.model_evidence import DECILES, _feature_stats

        rng = np.random.default_rng(0)
        x = pd.Series(rng.normal(size=5000))
        y = pd.Series(x * 2 + rng.normal(size=5000) * 0.5)
        stats = _feature_stats(x, y)
        assert stats["usable"] and stats["n"] == 5000
        assert stats["pearson"] > 0.9 and stats["spearman"] > 0.9
        assert len(stats["deciles"]) == DECILES
        means = [d["y_mean"] for d in stats["deciles"]]
        assert means == sorted(means), "a monotone relationship must read monotone"
        assert stats["decile_spread"] > 0

    def test_a_curved_relationship_shows_in_spearman_not_pearson(self):
        """Why both are reported: reading only the linear one calls this noise."""
        from engine.dashboard.model_evidence import _feature_stats

        x = pd.Series(np.linspace(0.01, 1, 4000))
        y = pd.Series(np.exp(x * 6))
        stats = _feature_stats(x, y)
        assert stats["spearman"] > 0.99
        assert stats["spearman"] > stats["pearson"]

    def test_too_few_rows_is_reported_not_computed(self):
        from engine.dashboard.model_evidence import _feature_stats

        stats = _feature_stats(pd.Series([1.0, 2.0, 3.0]), pd.Series([1.0, 2.0, 3.0]))
        assert stats["usable"] is False and "rows" in stats["reason"]
        assert "pearson" not in stats

    def test_a_lumpy_feature_gets_fewer_honest_buckets(self):
        """n_prior and signed_streak cannot always be cut into ten distinct
        bins — fewer real buckets beat ten fabricated ones."""
        from engine.dashboard.model_evidence import _feature_stats

        x = pd.Series([1.0] * 900 + [2.0] * 900)
        y = pd.Series(np.arange(1800, dtype=float))
        stats = _feature_stats(x, y)
        assert stats["usable"]
        assert 0 < len(stats.get("deciles", [])) <= 10

    def test_missing_values_do_not_silently_shrink_the_claim(self):
        from engine.dashboard.model_evidence import _feature_stats

        x = pd.Series([1.0] * 500 + [np.nan] * 500)
        y = pd.Series(np.arange(1000, dtype=float))
        stats = _feature_stats(x, y)
        assert stats["coverage"] == 0.5
        assert stats["n"] == 500

    def test_the_bundle_carries_the_models_payload(self, tmp_path, scores):
        render_bundle(scores, tmp_path / "b", as_of=AS_OF)
        payload = json.loads((tmp_path / "b" / "data" / "models.json").read_text())
        assert "models" in payload
        js = (tmp_path / "b" / "data" / "models.js").read_text()
        assert js.startswith("window.MODELS = ")


class TestNonMonotoneEvidence:
    """A V-shaped input is a real relationship that both correlations score at
    zero. Ranking on correlation alone buries exactly the inputs that matter."""

    def _v_shape(self):
        x = pd.Series(np.linspace(-3, 3, 4000))
        y = x.abs() * 2 + 1  # high at both ends, low in the middle
        return x, y

    def test_a_v_shape_reads_zero_on_both_correlations(self):
        from engine.dashboard.model_evidence import _feature_stats

        stats = _feature_stats(*self._v_shape())
        assert abs(stats["pearson"]) < 0.05
        assert abs(stats["spearman"]) < 0.05

    def test_but_the_magnitude_reading_sees_it(self):
        from engine.dashboard.model_evidence import _feature_stats

        stats = _feature_stats(*self._v_shape())
        assert stats["magnitude_spearman"] > 0.95

    def test_and_the_decile_range_sees_it(self):
        """End-to-end is ~0 for a symmetric V; best-to-worst is the real span."""
        from engine.dashboard.model_evidence import _feature_stats

        stats = _feature_stats(*self._v_shape())
        assert abs(stats["decile_spread"]) < 0.5
        assert stats["decile_range"] > 4
        assert stats["monotone"] is False

    def test_a_monotone_input_is_marked_monotone(self):
        from engine.dashboard.model_evidence import _feature_stats

        x = pd.Series(np.linspace(0, 10, 4000))
        stats = _feature_stats(x, x * 3)
        assert stats["monotone"] is True
        assert stats["decile_range"] == pytest.approx(stats["decile_spread"], rel=0.01)

    def test_the_scatter_carries_the_fitted_line_with_it(self):
        """The line is drawn precisely so a reader can see it explain nothing
        where the decile means clearly do."""
        from engine.dashboard.model_evidence import SCATTER_POINTS, _feature_stats

        stats = _feature_stats(*self._v_shape())
        assert len(stats["scatter"]) == SCATTER_POINTS
        assert all(len(p) == 2 for p in stats["scatter"])
        assert abs(stats["ols"]["slope"]) < 0.05, "a symmetric V has no linear slope"

    def test_scatter_is_deterministic(self):
        from engine.dashboard.model_evidence import _feature_stats

        a = _feature_stats(*self._v_shape())["scatter"]
        b = _feature_stats(*self._v_shape())["scatter"]
        assert a == b, "two runs on the same store must agree"


class TestLazyModelsPayload:
    def test_models_js_is_not_on_the_first_paint_path(self, tmp_path, scores):
        """It carries the scatter samples and is ~1 MB; the board must not pay
        for it on every visit."""
        render_bundle(scores, tmp_path / "b", as_of=AS_OF)
        html = (tmp_path / "b" / "index.html").read_text()
        assert 'src="data/models.js"' not in html
        assert (tmp_path / "b" / "data" / "models.js").exists()

    def test_the_single_file_build_still_inlines_it(self, tmp_path, scores):
        """Inlining only what index.html references would ship a one-file board
        whose Models area is permanently empty."""
        from engine.dashboard.render import write_single_file

        render_bundle(scores, tmp_path / "b", as_of=AS_OF)
        text = write_single_file(tmp_path / "b", tmp_path / "one.html").read_text()
        assert "window.MODELS = " in text
        assert "window.STRATEGIES = " in text


class TestChainRefreshCoversTheGap:
    """The refresh acquired almost nothing for six nights, silently.

    ORATS publishes a session around midnight, so an evening run asking only
    for today gets an empty response — indistinguishable from a name that has
    no chain. Two sessions, and skipping what is already held, fixes both that
    and a night that did not run.
    """

    class _Fetcher:
        def __init__(self):
            self.asked = []

        def fetch(self, source, endpoint, params, note=""):
            self.asked.append(params["tradeDate"])

            class R:
                from_cache = False

                @staticmethod
                def json():
                    return {"data": [{"ticker": params["ticker"].split(",")[0]}]}

            return R()

    def test_it_asks_for_the_previous_session_too(self, monkeypatch):
        from engine.dashboard import nightly

        monkeypatch.setattr(nightly, "_unknown_symbols", lambda: set())
        f = self._Fetcher()
        out = nightly.refresh_forward_chains(
            ["AAA"], "2026-09-02", fetcher=f, skip_held=False
        )
        assert len(out["sessions"]) == 2
        assert out["sessions"][0] == "2026-09-02"
        assert out["sessions"][1] < "2026-09-02"
        assert set(f.asked) == set(out["sessions"])

    def test_a_pair_already_in_the_store_is_not_re_bought(self, monkeypatch):
        from engine.dashboard import nightly

        monkeypatch.setattr(nightly, "_unknown_symbols", lambda: set())
        monkeypatch.setattr(
            "engine.replay.available_chain_keys",
            lambda: {("AAA", pd.Timestamp("2026-09-02"))},
        )
        f = self._Fetcher()
        out = nightly.refresh_forward_chains(["AAA"], "2026-09-02", fetcher=f)
        assert out["skipped_held"] == 1
        assert "2026-09-02" not in f.asked, "already held — must not be re-bought"

    def test_sessions_can_be_widened_to_heal_a_longer_gap(self, monkeypatch):
        from engine.dashboard import nightly

        monkeypatch.setattr(nightly, "_unknown_symbols", lambda: set())
        f = self._Fetcher()
        out = nightly.refresh_forward_chains(
            ["AAA"], "2026-09-02", fetcher=f, sessions=5, skip_held=False
        )
        assert len(out["sessions"]) == 5
        assert out["sessions"] == sorted(out["sessions"], reverse=True)

    def test_weekends_are_stepped_over(self, monkeypatch):
        """Sessions, not calendar days — a Monday run must reach back to
        Friday, not to Sunday."""
        from engine.dashboard import nightly

        monkeypatch.setattr(nightly, "_unknown_symbols", lambda: set())
        f = self._Fetcher()
        out = nightly.refresh_forward_chains(
            ["AAA"], "2026-09-14", fetcher=f, sessions=2, skip_held=False
        )
        assert out["sessions"] == ["2026-09-14", "2026-09-11"], out["sessions"]

    def test_a_run_on_a_market_holiday_falls_back_to_real_sessions(self, monkeypatch):
        """2026-09-07 is Labor Day. A run stamped that date must not ask for a
        closed session — there is no chain to publish for one."""
        from engine.dashboard import nightly

        monkeypatch.setattr(nightly, "_unknown_symbols", lambda: set())
        f = self._Fetcher()
        out = nightly.refresh_forward_chains(
            ["AAA"], "2026-09-07", fetcher=f, sessions=2, skip_held=False
        )
        assert out["sessions"] == ["2026-09-04", "2026-09-03"], out["sessions"]
        assert "2026-09-07" not in f.asked


class TestBookPanel:
    """The ledger surfaced in the UI.

    The panel exists to be inspected, so the thing that matters is that it
    never takes the board down and never quietly drops a trade.
    """

    def test_it_asks_the_ledger_for_the_contrarian_rows_too(self, monkeypatch):
        """The client filters one dataset by `recommended` — book.json has to
        carry both populations or there is nothing for the filter to show."""
        from engine.dashboard import render

        captured = {}

        def fake(contracts=1, include_declined=False):
            captured["include_declined"] = include_declined
            return pd.DataFrame()

        monkeypatch.setattr("engine.portfolio.build_book", fake)
        render.build_book()
        assert captured["include_declined"] is True

    def test_it_carries_the_rows_and_the_summary(self, monkeypatch):
        from engine.dashboard import render

        book = pd.DataFrame([{
            "as_of": pd.Timestamp("2026-08-28"), "ticker": "AAA",
            "strategy": "STR-THRU", "event_date": pd.Timestamp("2026-08-31"),
            "state": "settled", "entry_cost": 6.0, "realized_pnl": 0.25,
            "pnl": 150.0, "capital": 600.0, "contracts": 1,
        }])
        monkeypatch.setattr("engine.portfolio.build_book", lambda contracts=1, include_declined=False: book)
        payload = render.build_book()
        assert payload["available"] is True
        assert len(payload["rows"]) == 1
        assert payload["rows"][0]["ticker"] == "AAA"
        assert payload["summary"]["n_settled"] == 1

    def test_a_broken_ledger_renders_empty_rather_than_failing(self, monkeypatch):
        """A dashboard that will not load is worse than one saying 'nothing yet'."""
        from engine.dashboard import render

        def boom(contracts=1, include_declined=False):
            raise RuntimeError("ledger unreadable")

        monkeypatch.setattr("engine.portfolio.build_book", boom)
        payload = render.build_book()
        assert payload["available"] is False
        assert payload["rows"] == []
        assert "ledger unreadable" in payload["error"]

    def test_an_empty_ledger_is_available_but_empty(self, monkeypatch):
        from engine.dashboard import render

        monkeypatch.setattr("engine.portfolio.build_book", lambda contracts=1, include_declined=False: pd.DataFrame())
        payload = render.build_book()
        assert payload["available"] is True and payload["rows"] == []

    def test_the_js_wrapper_is_named_so_the_page_can_read_it(self):
        from engine.dashboard.render import _js_name

        assert _js_name("book") == "BOOK"

    def test_the_page_actually_loads_the_book_script(self):
        """A wrapper nobody's <script> tag points at is dead on arrival.

        `data/book.js` shipped correctly for a full session — real rows,
        written by `_write_pair` alongside its `.json` twin — and the client
        never loaded it: `index.html` had no `<script src="data/book.js">`,
        and unlike `models.js` it had no on-demand loader either (`app.js`
        reads `window.BOOK` at module-parse time, not behind a click). The
        Book tab rendered its empty state against a real book, in every
        environment, for as long as the tag was missing — and nothing that
        inspects book.json's CONTENT (including the two tests above this one)
        would ever have caught it.
        """
        from pathlib import Path

        html = Path("engine/dashboard/static/index.html").read_text()
        assert '<script src="data/book.js">' in html

    def test_every_state_the_book_emits_has_a_ui_label(self):
        """A state with no label renders as a raw enum, and `unresolvable`
        rendering as a blank is how a dropped trade hides."""
        from pathlib import Path

        app = Path("engine/dashboard/static/assets/app.js").read_text()
        for state in ("settled", "open", "awaiting_exit", "unresolvable"):
            assert f"{state}:" in app or f'"{state}"' in app, state


class TestBoardOrderAndDomainFilter:
    """The board's default order and its out-of-domain switch.

    Both are client-side, so the assertions come in two layers: cheap text
    checks that pin the invariants a reader of `app.js` could otherwise break
    silently, and — where node is available — the real sort run against
    synthetic rows.
    """

    APP = Path("engine/dashboard/static/assets/app.js")

    def test_the_session_order_is_explicit_not_alphabetical(self):
        """`"AMC" < "BMO"` as strings, so sorting the label sorts the schedule
        backwards: a BMO print is decided at the PREVIOUS session's close and
        must come first."""
        app = self.APP.read_text()
        assert "SESSION_ORDER = { BMO: 0, AMC: 1 }" in app

    def test_the_out_of_domain_switch_defaults_to_off(self):
        app = self.APP.read_text()
        assert "outOfDomain: false," in app
        html = Path("engine/dashboard/static/index.html").read_text()
        assert 'id="f-ood" type="checkbox"' in html
        # No `checked` attribute: the markup must not re-enable what the state
        # turns off.
        assert 'id="f-ood" type="checkbox" checked' not in html

    # -- the real sort, when node is around ---------------------------------

    @staticmethod
    def _run(rows, state=None, tmp_path=None):
        import shutil
        import subprocess

        node = shutil.which("node")
        if node is None:
            pytest.skip("node not available")
        driver = tmp_path / "drive.mjs"
        driver.write_text(
            "import fs from 'node:fs';\n"
            "const src = fs.readFileSync(%r, 'utf8');\n"
            "const body = src.slice(0, src.indexOf('function init()'));\n"
            "globalThis.window = { BOARD: { rows: %s } };\n"
            "globalThis.document = { getElementById: () => ({}),"
            " querySelectorAll: () => [], querySelector: () => ({}),"
            " createElement: () => ({ appendChild(){} }) };\n"
            "const m = new Function(body + '\\nreturn {boardRows, state, get hidden(){return boardHidden;}};')();\n"
            "Object.assign(m.state, %s);\n"
            "const out = m.boardRows().map((r) => r.ticker + '/' + r.strategy);\n"
            "console.log(JSON.stringify({ order: out, hidden: m.hidden }));\n"
            % (str(TestBoardOrderAndDomainFilter.APP), json.dumps(rows), json.dumps(state or {}))
        )
        proc = subprocess.run(
            [node, str(driver)], capture_output=True, text=True, timeout=60
        )
        assert proc.returncode == 0, proc.stderr
        return json.loads(proc.stdout.strip().splitlines()[-1])

    @staticmethod
    def _row(ticker, session, event_date, *, gate_pass=None, flags=(), strategy="STR-THRU"):
        return {
            "row_id": f"{ticker}|{strategy}|{event_date}",
            "ticker": ticker, "strategy": strategy, "session": session,
            "event_date": event_date, "gate_pass": gate_pass, "gate_score": None,
            "flags": list(flags),
        }

    def test_bmo_precedes_amc_on_the_same_date(self, tmp_path):
        rows = [
            self._row("ZZZ", "AMC", "2026-09-02"),
            self._row("AAA", "BMO", "2026-09-02"),
            self._row("BBB", "BMO", "2026-09-01"),
        ]
        got = self._run(rows, tmp_path=tmp_path)
        assert got["order"] == ["BBB/STR-THRU", "AAA/STR-THRU", "ZZZ/STR-THRU"]

    def test_within_a_session_a_proposed_trade_comes_first(self, tmp_path):
        """Order inside a date/session is the gate verdict: pass, then fail,
        then rows carrying no decision at all."""
        rows = [
            self._row("CCC", "BMO", "2026-09-02", gate_pass=None),
            self._row("BBB", "BMO", "2026-09-02", gate_pass=False),
            self._row("AAA", "BMO", "2026-09-02", gate_pass=True),
        ]
        got = self._run(rows, tmp_path=tmp_path)
        assert got["order"] == ["AAA/STR-THRU", "BBB/STR-THRU", "CCC/STR-THRU"]

    def test_a_prints_structures_stay_together_under_its_best_verdict(self, tmp_path):
        """A name with one tradeable structure sorts above a name with none,
        and its other structures travel with it rather than scattering."""
        rows = [
            self._row("NOPE", "BMO", "2026-09-02", gate_pass=False),
            self._row("NOPE", "BMO", "2026-09-02", gate_pass=None, strategy="CAL-P"),
            self._row("YES", "BMO", "2026-09-02", gate_pass=None, strategy="CAL-P"),
            self._row("YES", "BMO", "2026-09-02", gate_pass=True),
        ]
        got = self._run(rows, tmp_path=tmp_path)
        assert got["order"] == [
            "YES/STR-THRU", "YES/CAL-P", "NOPE/STR-THRU", "NOPE/CAL-P",
        ]

    def test_reversing_the_date_keeps_the_passers_on_top(self, tmp_path):
        """Reading the calendar backwards is not a request to bury the trades."""
        rows = [
            self._row("AAA", "BMO", "2026-09-02", gate_pass=None),
            self._row("BBB", "BMO", "2026-09-02", gate_pass=True),
        ]
        got = self._run(rows, {"sortDir": -1}, tmp_path=tmp_path)
        assert got["order"] == ["BBB/STR-THRU", "AAA/STR-THRU"]

    def test_out_of_domain_rows_are_hidden_by_default_and_counted(self, tmp_path):
        rows = [
            self._row("AAA", "BMO", "2026-09-02", gate_pass=True),
            self._row("OOD", "BMO", "2026-09-02", flags=["OUT_OF_DOMAIN"]),
        ]
        got = self._run(rows, tmp_path=tmp_path)
        assert got["order"] == ["AAA/STR-THRU"]
        assert got["hidden"] == 1

        shown = self._run(rows, {"outOfDomain": True}, tmp_path=tmp_path)
        assert shown["order"] == ["AAA/STR-THRU", "OOD/STR-THRU"]
        assert shown["hidden"] == 0

    def test_the_switch_hides_only_the_out_of_domain_structure(self, tmp_path):
        """A print with one withheld structure and one scored keeps the scored
        one — the filter is per row, not per name."""
        rows = [
            self._row("AAA", "BMO", "2026-09-02", gate_pass=True),
            self._row("AAA", "BMO", "2026-09-02", flags=["OUT_OF_DOMAIN"], strategy="CAL-P"),
        ]
        got = self._run(rows, tmp_path=tmp_path)
        assert got["order"] == ["AAA/STR-THRU"]


class TestLadderRowsRoundTrip:
    """Two defects the ladder carried, both invisible until a self-check ran.

    A ±2.5% strike off a $369.59 spot is 360.35024999999996. The bundle stores
    floats at six places, so it comes back as 360.35025 — and `ScoreRequest.key()`
    renders the strike at FOUR, where those two floats disagree. The re-scored
    row then drew from a different bootstrap seed and failed the digest, on a
    row nothing was actually wrong with. Separately, neither ladder row resolved
    a strike, so both were keyed `…|atm`: one identity, two requests.
    """

    def test_a_ladder_strike_survives_the_bundles_rounding(self):
        from engine.dashboard.render import BUNDLE_PRECISION
        from engine.score import ladder_strike

        spot = 369.59
        for offset in (-0.025, 0.025, -0.05, 0.05):
            strike = ladder_strike(spot, offset)
            stored = round(strike, BUNDLE_PRECISION)
            assert stored == strike
            # The four-place rendering `ScoreRequest.key()` uses is what the
            # bootstrap seed is derived from.
            assert f"{stored:.4f}" == f"{strike:.4f}"

    def test_the_unrounded_strike_is_what_used_to_break(self):
        """Guards the diagnosis, not the fix: if this ever stops being true the
        test above is no longer testing anything."""
        from engine.dashboard.render import BUNDLE_PRECISION

        raw = 369.59 * (1 - 0.025)
        assert f"{raw:.4f}" != f"{round(raw, BUNDLE_PRECISION):.4f}"

    def test_two_unresolved_ladder_rows_are_two_rows(self):
        from engine.dashboard.render import _row_identity

        base = {"ticker": "AVGO", "strategy": "STR-THRU", "event_date": "2026-09-02"}
        low = _row_identity(base | {"strike": None, "requested_strike": 360.3502})
        high = _row_identity(base | {"strike": None, "requested_strike": 378.8297})
        assert low != high

    def test_a_resolved_row_still_keys_on_the_strike_it_got(self):
        from engine.dashboard.render import _row_identity

        base = {"ticker": "AVGO", "strategy": "STR-THRU", "event_date": "2026-09-02"}
        assert _row_identity(base | {"strike": 370.0}) == "AVGO|STR-THRU|2026-09-02|370.0000"
        # A plain ATM row that resolved nothing keeps the old identity: it is
        # the only row of its print with no requested strike.
        assert _row_identity(base | {"strike": None}) == "AVGO|STR-THRU|2026-09-02|atm"

    def test_a_nan_strike_reads_as_absent(self):
        """A ScoreResult travels through a DataFrame, where a missing strike
        becomes NaN — keying on `is None` alone would format it as `nan`."""
        from engine.dashboard.render import _row_identity

        base = {"ticker": "AVGO", "strategy": "STR-THRU", "event_date": "2026-09-02"}
        got = _row_identity(base | {"strike": float("nan"), "requested_strike": 360.3502})
        assert got == "AVGO|STR-THRU|2026-09-02|req360.3502"

    def test_compact_row_and_rank_agree_on_identity(self):
        """They used to build the id twice, in two places, from one rule."""
        from engine.dashboard.render import _row_identity, compact_row

        record = {
            "ticker": "AVGO", "strategy": "STR-THRU", "event_date": "2026-09-02",
            "strike": None, "requested_strike": 360.3502, "strike_offset": -0.025,
        }
        assert compact_row(record)["row_id"] == _row_identity(record)

    def test_a_rendered_bundle_has_no_duplicate_row_ids(self, tmp_path):
        """The property the acceptance check enforces on the real board, tested
        here on the ladder shape that actually broke it."""
        from engine.dashboard.render import compact_row

        base = {
            "ticker": "AVGO", "strategy": "STR-THRU", "event_date": "2026-09-02",
            "strike": None,
        }
        ids = [
            compact_row(base | {"requested_strike": s, "strike_offset": o})["row_id"]
            for s, o in ((360.3502, -0.025), (378.8297, 0.025))
        ]
        ids.append(compact_row(base | {"strike": 370.0})["row_id"])
        assert len(set(ids)) == len(ids)


class TestImpliedMoveSentinel:
    """ORATS writes `impliedMove == 0` for "no quote" (EXP-110).

    Left in, it tells a model the market expects no movement on precisely the
    events that move most. The normalizer masks it; `has_implied_quote`
    (EXP-111) keeps the absence as information.
    """

    def test_zero_is_nulled_but_a_real_quote_survives(self):
        import numpy as np
        from engine.data.normalize.common import clip_implausible

        # The range check alone does NOT catch it — the lower bound is
        # inclusive, which is why this needed its own step.
        frame = pd.DataFrame({"implied_move": [0.0, 4.2, -1.0, 120.0]})
        clipped, _ = clip_implausible(frame)
        assert clipped["implied_move"].iloc[0] == 0.0, "range check keeps zero"

        implied = pd.to_numeric(frame["implied_move"], errors="coerce")
        sentinel = implied.notna() & (implied <= 0)
        assert list(sentinel) == [True, False, True, False]

    def test_the_convention_is_recorded(self):
        from engine.data.schemas import CONVENTIONS

        assert "orats_implied_zero_sentinel" in CONVENTIONS
        assert "has_implied_quote" in CONVENTIONS["orats_implied_zero_sentinel"]

    def test_a_withheld_gate_says_why(self):
        """Both decline paths used to `return` in silence, which is
        indistinguishable on the board from never having been asked."""
        from pathlib import Path

        src = Path("engine/score.py").read_text()
        gate = src[src.index("def _score_gate"):]
        gate = gate[:gate.index("def _compare_layers")]
        # every `return` that is not the prediction path carries a flag
        assert gate.count("MISSING_FEATURES") == 2
        assert "non-finite" in gate

    def test_a_gate_note_does_not_run_into_the_analog_note(self):
        """Both layers append to `detail`, and the analog layer writes first.

        The first version put the separator INSIDE the appended fragment and
        then `.strip()`ed it off, producing "…dte_bandgate gate_midfill…" on
        255 live rows. The join has to live between the parts.
        """
        from pathlib import Path

        src = Path("engine/score.py").read_text()
        gate = src[src.index("def _score_gate"):]
        gate = gate[:gate.index("def _compare_layers")]
        assert 'f"{result.detail}; {note}"' in gate
        # Not `assert ".strip()" not in gate` — the comment above the fix names
        # `.strip()` as the cause, and a test that trips on its own explanation
        # is a test that punishes documenting the bug.
        code = "\n".join(l for l in gate.splitlines() if not l.lstrip().startswith("#"))
        assert ").strip()" not in code


class TestChainAnchorFollowsWhatIsPublished:
    """The chain pass must not buy a session ORATS has not published.

    Measured 2026-09-03: an `as_of` whose market had not closed produced 30 of
    30 `hist/strikes` 404s. The pass costs one call per TEN TICKERS per session,
    while the market-wide pass that runs immediately before it discovers the
    newest published date at one call per session — and used to throw that
    answer away. The prescribed 21:30 cron would have paid ~630 calls a month
    for nothing, against a 3,000-call live reserve.
    """

    def test_an_unpublished_today_falls_back_to_the_published_session(self):
        from engine.dashboard.nightly import _newest_published

        out = {"endpoints": {"hist/summaries": {"tradeDate": "2026-09-02"},
                             "hist/cores": {"tradeDate": "2026-09-02"}}}
        got = _newest_published(out, default=pd.Timestamp("2026-09-03"))
        assert got == pd.Timestamp("2026-09-02")

    def test_a_published_today_is_used_as_is(self):
        from engine.dashboard.nightly import _newest_published

        out = {"endpoints": {"hist/summaries": {"tradeDate": "2026-09-03"}}}
        assert _newest_published(out, default=pd.Timestamp("2026-09-03")) == pd.Timestamp("2026-09-03")

    def test_it_never_runs_ahead_of_as_of(self):
        """A backfill run for an old date must not be dragged forward to today
        by a summaries pass that happened to fetch something newer."""
        from engine.dashboard.nightly import _newest_published

        out = {"endpoints": {"hist/summaries": {"tradeDate": "2026-09-09"}}}
        assert _newest_published(out, default=pd.Timestamp("2026-09-03")) == pd.Timestamp("2026-09-03")

    def test_no_information_means_no_change(self):
        from engine.dashboard.nightly import _newest_published

        assert _newest_published({}, default=pd.Timestamp("2026-09-03")) == pd.Timestamp("2026-09-03")
        assert _newest_published({"endpoints": {"x": {}}},
                                 default=pd.Timestamp("2026-09-03")) == pd.Timestamp("2026-09-03")
