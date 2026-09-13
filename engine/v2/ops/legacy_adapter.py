"""Audited bridge to the frozen legacy tree."""
from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

from engine.v2.foundation import safe_relative_path
from engine.v2.ops.errors import fail

__all__ = ["copy_read_set", "invoke_evaluate", "invoke_nightly_helper",
           "invoke_score_calendar", "manifest_files", "run_legacy_script"]


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def manifest_files(root: Path | str, paths: tuple[str, ...] | list[str]) -> dict:
    """Return a complete, immutable read-set manifest and reject indirection."""
    base = Path(root).resolve()
    result = {}
    for relative in paths:
        safe_relative_path(relative)
        source = base / relative
        if source.is_symlink() or not source.is_file():
            raise fail("INPUT_CHANGED", "legacy read-set member is missing or indirect",
                       details={"path": relative})
        result[relative] = {"content_hash": _digest(source),
                            "byte_size": source.stat().st_size}
    return result


def copy_read_set(source_root: Path | str, private_root: Path | str,
                  paths: tuple[str, ...] | list[str]) -> dict:
    """Copy declared inputs privately, preserving bytes without links."""
    source_raw = Path(source_root)
    target_raw = Path(private_root)
    if source_raw.is_symlink() or target_raw.is_symlink():
        raise fail("INTEGRITY_FAILED", "private root may not be a symlink")
    source = source_raw.resolve()
    target = target_raw.resolve()
    manifest = manifest_files(source, paths)
    target.mkdir(parents=True, exist_ok=True)
    for relative, expected in manifest.items():
        current = source
        for component in Path(relative).parts[:-1]:
            current = current / component
            if current.is_symlink():
                raise fail("INPUT_CHANGED", "legacy read-set ancestor is indirect")
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        probe = target
        for component in Path(relative).parts[:-1]:
            probe = probe / component
            if probe.is_symlink():
                raise fail("INTEGRITY_FAILED", "private destination ancestor is a symlink")
        if destination.exists() and destination.is_symlink():
            raise fail("INTEGRITY_FAILED", "private destination is a symlink")
        shutil.copyfile(source / relative, destination)
        if destination.is_symlink() or _digest(destination) != expected["content_hash"]:
            raise fail("INTEGRITY_FAILED", "private legacy copy failed verification",
                       details={"path": relative})
        destination.chmod(0o444)
    return manifest


def _rooted_import(root: Path | str):
    path = Path(root).resolve()
    os.environ["INVESTING_PLAN_ROOT"] = str(path)


def invoke_score_calendar(root, as_of, *, scorer, tickers=None, horizon_days=35):
    """Call the existing scorer with its public legacy arguments."""
    _rooted_import(root)
    from engine.score import score_calendar
    return score_calendar(as_of, horizon_days=horizon_days, alt_strikes=0,
                          scorer=scorer, tickers=tickers, progress_every=10)


def _write_action(root, name, value):
    import json

    from engine.v2.foundation import content_hash

    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str,
                               allow_nan=False))
    return {"path": str(path.relative_to(root)), "hash": content_hash(value)}


def legacy_action(action, parameters, staging):
    """Execute one registered legacy stage inside a fresh worker process."""
    root = Path(staging).resolve()
    _rooted_import(root / "legacy")
    actions = {
        "legacy_finality": _action_finality,
        "legacy_score": _action_score,
        "legacy_decisions": _action_decisions,
        "legacy_settlement": _action_settlement,
        "legacy_model_evidence": _action_model_evidence,
        "legacy_render": _action_render,
        "legacy_selfcheck": _action_selfcheck,
        "legacy_score_requests": _action_score_requests,
    }
    if action not in actions:
        raise fail("INVALID_REQUEST", "legacy action is not allowlisted")
    return actions[action](parameters, root)


def _action_finality(parameters, root):
    from engine.calendar import trading_calendar
    from engine.data.finality import resolve_final_session

    result = resolve_final_session(parameters["session"], parameters["tickers"],
                                   calendar=trading_calendar())
    return _write_action(root, "finality.json", result.as_dict())


def _action_score(parameters, root):
    import pandas as pd

    from engine.dashboard.nightly import strike_ladder
    from engine.features import FeatureContext
    from engine.jsonio import json_safe
    from engine.score import Scorer, score_calendar

    tickers = sorted(set(parameters["tickers"]))
    years = range(int(parameters["year_start"]), int(parameters["year_end"]) + 1)
    scorer = Scorer(context=FeatureContext.load(tickers, years=years))
    frame = score_calendar(pd.Timestamp(parameters["session"]),
                          horizon_days=int(parameters.get("horizon_days", 35)),
                          alt_strikes=0, scorer=scorer, tickers=tickers,
                          progress_every=10)
    rows = json_safe(frame.to_dict(orient="records"), round_to=None)
    for row in rows:
        row["row_id"] = _score_row_id(row)
    expected = tuple(parameters.get("expected_population", ()))
    if not expected:
        raise fail("VALIDATION_FAILED", "score population must be planned before execution")
    if len(set(expected)) != len(expected):
        raise fail("VALIDATION_FAILED", "planned population has duplicate keys")
    observed_keys = {_population_key(row) for row in rows}
    missing = sorted(set(expected) - observed_keys)
    unplanned = sorted(observed_keys - set(expected))
    if missing or unplanned:
        raise fail("VALIDATION_FAILED", "score population differs from planned inputs",
                   details={"missing": missing, "unplanned": unplanned})
    ladder = strike_ladder(frame, scorer=scorer,
                           alt_strikes=int(parameters.get("alt_strikes", 1)),
                           as_of=pd.Timestamp(parameters["session"]))
    return _write_action(root, "score.json", {
        "rows": rows, "expected_population": list(expected),
        "observed_population": sorted(observed_keys), "ladder": json_safe(ladder, round_to=None),
        "tickers": tickers, "analog_entry_coverage": scorer.analog_entry_coverage})


def _action_score_requests(parameters, root):
    import dataclasses
    import json

    import pandas as pd

    from engine.features import FeatureContext
    from engine.jsonio import json_safe
    from engine.score import UNSCORABLE, FillModel, Scorer, ScoreRequest, unscorable_result

    entries = json.loads((root / parameters["requests_path"]).read_text())
    requests = [entry["request"] for entry in entries]
    tickers = sorted({str(row["ticker"]) for row in requests})
    years = range(int(parameters["year_start"]), int(parameters["year_end"]) + 1)
    scorer = Scorer(context=FeatureContext.load(tickers, years=years))
    rows = []
    fields = {field.name for field in dataclasses.fields(ScoreRequest)}
    dates = {"as_of", "event_date", "expiry", "chain_as_of"}
    for entry, request in zip(entries, requests):
        values = {key: value for key, value in request.items() if key in fields}
        for key in dates:
            if isinstance(values.get(key), str):
                values[key] = pd.Timestamp(values[key])
        if isinstance(values.get("fill"), dict):
            values["fill"] = FillModel(alpha=float(values["fill"]["alpha"]))
        score_request = ScoreRequest(**values)
        try:
            result = scorer.score(score_request)
        except UNSCORABLE as exc:
            result = unscorable_result(score_request, as_of=score_request.as_of,
                                       snapshot=scorer.snapshot, exc=exc)
        record = json_safe(result.as_dict(), round_to=None)
        rows.append({"request_id": entry["canary_id"],
                     "record": record})
    return _write_action(root, "score_requests.json", {"rows": rows,
                                                        "expected_population": len(requests)})


def _score_row_id(row):
    return "|".join(str(row.get(key, "")) for key in
                     ("ticker", "strategy", "event_date", "strike", "expiry"))


def _population_key(row):
    """A2: the planned population is keyed before strike/expiry are known —
    they only exist after scoring, so the plan cannot name them in advance."""
    return "|".join(str(row.get(key, "")) for key in ("ticker", "strategy", "event_date"))


def _load_action_frame(root):
    import json

    import pandas as pd

    path = root / "score.json"
    if not path.is_file():
        raise fail("INPUT_CHANGED", "score artifact is missing")
    return pd.DataFrame(json.loads(path.read_text())["rows"])


def _load_finality(root):
    import json

    path = root / "finality.json"
    if not path.is_file():
        raise fail("INPUT_CHANGED", "finality artifact is missing")
    return json.loads(path.read_text())


def _action_decisions(parameters, root):
    from engine.ledger import build_prediction_rows

    frame = _load_action_frame(root)
    finality = _load_finality(root)
    rows = build_prediction_rows(frame, as_of=parameters["session"],
                                 finality=finality, entry_dated_only=True)
    return _write_action(root, "decisions.json", {"rows": rows, "expected_rows": len(rows)})


def _action_settlement(parameters, root):
    from engine.ledger import score_outcomes

    return _write_action(root, "settlement.json",
                         score_outcomes(through=parameters["session"]))


def _action_model_evidence(parameters, root):
    from engine.dashboard.model_evidence import build_model_evidence

    result = build_model_evidence(force=bool(parameters.get("force", False)))
    return _write_action(root, "model_evidence.json", result)


def _action_render(parameters, root):
    import tarfile

    from engine.dashboard.render import render_bundle

    output = root / "bundle"
    result = render_bundle(_load_action_frame(root), output,
                           as_of=parameters["session"],
                           horizon_days=int(parameters.get("horizon_days", 35)))
    with tarfile.open(root / "bundle.tar", "w") as archive:
        archive.add(output, arcname="bundle")
    return _write_action(root, "render.json", {"bundle_archive": "bundle.tar",
                                                "result": result}) | {
                                                    "path": "bundle.tar"}


def _action_selfcheck(parameters, root):
    import tarfile

    from engine.dashboard.selfcheck import selfcheck
    from engine.features import FeatureContext
    from engine.score import Scorer

    tickers = sorted(set(parameters.get("tickers", ())))
    years = range(int(parameters.get("year_start", 0)),
                  int(parameters.get("year_end", 0)) + 1)
    scorer = Scorer(context=FeatureContext.load(tickers, years=years))
    archive = root / "bundle.tar"
    if archive.is_file() and not (root / "bundle").is_dir():
        with tarfile.open(archive) as stream:
            stream.extractall(root)
    result = selfcheck(root / "bundle", n=int(parameters.get("sample", 10)),
                       scorer=scorer)
    value = result.as_dict() if hasattr(result, "as_dict") else vars(result)
    if not value.get("ok"):
        raise fail("VALIDATION_FAILED", "serialized legacy bundle selfcheck failed")
    return _write_action(root, "selfcheck.json", value)


def invoke_nightly_helper(root, helper, *args, **kwargs):
    """Call one explicitly named read/compute helper, never ``run_nightly``."""
    _rooted_import(root)
    from engine.dashboard.nightly import refresh_calendar_data, strike_ladder, validate_refresh
    allowed = {"validate_refresh": validate_refresh,
               "strike_ladder": strike_ladder,
               "refresh_calendar_data": refresh_calendar_data}
    if helper not in allowed:
        raise fail("INVALID_REQUEST", "legacy nightly helper is not audited",
                   details={"helper": helper})
    return allowed[helper](*args, **kwargs)


def invoke_evaluate(root, spec, trades, *, run_dir, **kwargs):
    """Use the repository evaluator for a genuine experiment report."""
    _rooted_import(root)
    from engine.evaluate import evaluate
    return evaluate(spec, trades, run_dir=run_dir, **kwargs)


REGISTERED_RUNNERS = frozenset({
    "experiments/EXP-182_d_1_gated_execution_parity_registered/run.py",
})


def run_legacy_script(root, script, args=()):
    """Run a registered legacy runner in a private root with smoke protection."""
    _rooted_import(root)
    base = Path(root).resolve()
    relative = str(Path(script))
    script_path = (base / relative).resolve()
    if relative not in REGISTERED_RUNNERS or not script_path.is_relative_to(base):
        raise fail("INVALID_REQUEST", "legacy experiment runner is unaudited")
    if tuple(args):
        raise fail("INVALID_REQUEST", "legacy runner may not enable ledger writes")
    import subprocess
    command = ["/usr/bin/python3", "-u", str(script_path), "--no-ledger"]
    return subprocess.run(command, cwd=base, check=False,
                          capture_output=True, text=True, timeout=3600)
