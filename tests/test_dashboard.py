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
