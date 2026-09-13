"""Audited bridge to the frozen legacy tree."""
from __future__ import annotations

import hashlib
import json
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
        "legacy_decision_replay": _action_decision_replay,
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


def _load_score_document(root):
    path = root / "score.json"
    if not path.is_file():
        raise fail("INPUT_CHANGED", "score artifact is missing")
    return json.loads(path.read_text())


def _empty_replay(session):
    from engine.v2.foundation import content_hash
    return {"schema_version": "decision_replay.v1.0", "session": session,
            "population": [], "source_rows": [], "replayed_rows": [],
            "source_rows_hash": content_hash([]), "replayed_rows_hash": content_hash([]),
            "findings": []}


def _action_decision_replay(parameters, root):
    """B1b: re-score the decision-eligible board rows through the SAME public
    entrypoint the score stage used, in a fresh process.

    Never rebuilds a ``ScoreRequest`` by hand: a DYN-SV chooser row is not a
    request the engine can score on its own (``DYN-SV`` names no structure,
    only ``score_calendar``'s own menu step can produce one), so the only
    faithful replay is re-running ``score_calendar`` itself and letting the
    chooser rank the menu again.

    ``FeatureContext`` loads the score job's FULL ticker set — analog pools
    and the registered gate/chooser champions are read off that context, not
    off ``score_calendar``'s own ``tickers`` argument, so a narrower context
    would be a different computation. ``score_calendar``'s own ``tickers``
    argument, by contrast, only restricts which events are enumerated and
    which chains are pre-loaded (read ``engine.score.score_calendar``: the
    events table is filtered by ticker before anything cross-request is
    built, and the DYN-SV chooser groups by event, never across tickers) — so
    it is safe, and far cheaper, to restrict it to just the eligible rows'
    own tickers rather than rescoring the whole board.
    """
    from engine.jsonio import json_safe
    from engine.v2.foundation import content_hash
    from engine.v2.ops.decision_replay import compare_rows, decision_population, population_key

    session = parameters["session"]
    population = decision_population(_load_score_document(root), session)
    if not population:
        return _write_action(root, "replay.json", _empty_replay(session))

    import pandas as pd

    from engine.features import FeatureContext
    from engine.score import Scorer, score_calendar

    tickers = sorted(set(parameters["tickers"]))
    years = range(int(parameters["year_start"]), int(parameters["year_end"]) + 1)
    scorer = Scorer(context=FeatureContext.load(tickers, years=years))
    eligible_tickers = sorted({str(row["ticker"]) for row in population})
    frame = score_calendar(pd.Timestamp(session),
                          horizon_days=int(parameters.get("horizon_days", 35)),
                          alt_strikes=0, scorer=scorer, tickers=eligible_tickers)
    rows = json_safe(frame.to_dict(orient="records"), round_to=None)
    for row in rows:
        row["row_id"] = _score_row_id(row)
    eligible_keys = {population_key(row) for row in population}
    replayed = [row for row in rows if population_key(row) in eligible_keys]
    return _write_action(root, "replay.json", {
        "schema_version": "decision_replay.v1.0", "session": session,
        "population": [population_key(row) for row in population],
        "source_rows": population, "replayed_rows": replayed,
        "source_rows_hash": content_hash(population),
        "replayed_rows_hash": content_hash(replayed),
        "findings": compare_rows(population, replayed)})


def _action_decisions(parameters, root):
    from engine.ledger import build_prediction_rows

    frame = _load_action_frame(root)
    finality = _load_finality(root)
    plan_path = root / "decision_plan.json"
    if not plan_path.is_file():
        raise fail("VALIDATION_FAILED", "decision plan artifact is missing")
    plan = json.loads(plan_path.read_text())
    rows = build_prediction_rows(frame, as_of=parameters["session"],
                                 decision_ts=plan.get("decision_clock"),
                                 finality=finality, entry_dated_only=True)
    for row in rows:
        row["written_at"] = plan.get("decision_clock")
        row["decision_ts"] = plan.get("decision_clock")
        if not row.get("event_id"):
            row["event_id"] = (row.get("score") or {}).get("event_id")
    return _write_action(root, "decisions.json", {"rows": rows, "expected_rows": len(rows)})


def _action_settlement(parameters, root):
    import base64

    from engine.ledger import score_outcomes

    directory = root / "legacy" / "ledger" / "outcomes"
    before = {path: path.stat().st_size for path in directory.glob("*.jsonl")}
    result = score_outcomes(through=parameters["session"])
    captured = []
    for path in sorted(directory.glob("*.jsonl")):
        data = path.read_bytes()[before.get(path, 0):]
        for raw in data.splitlines(keepends=True):
            captured.append({"original_b64": base64.b64encode(raw).decode("ascii"),
                             "row": json.loads(raw)})
    return _write_action(root, "settlement.json", {"result": result, "rows": captured})


def _action_model_evidence(parameters, root):
    from engine.dashboard.model_evidence import build_model_evidence

    result = build_model_evidence(force=bool(parameters.get("force", False)))
    return _write_action(root, "model_evidence.json", result)


def _action_render(parameters, root):
    """P2-5/D19: render at parity with the legacy nightly's own render call.

    Carries the full legacy argument set (guide §9.4 item 2): the score
    artifact's ``rows`` + ``ladder`` concatenated exactly as the legacy
    nightly does (:func:`render_inputs.assemble_scores`), the model-evidence
    artifact placed at its legacy path, the bound ledger generation staged as
    ``legacy/ledger`` (never the mutable staged copy), and ``meta``/``health``
    built from those same legacy helpers — mirroring
    ``engine/dashboard/nightly.py:1580-1622``.
    """
    import tarfile

    import pandas as pd

    from engine.dashboard.render import (
        build_health,
        build_meta,
        freshness_summary,
        quota_state,
        render_bundle,
        size_model_mae_from_ledger,
    )
    from engine.features import FeatureContext
    from engine.score import Scorer
    from engine.v2.ops.render_inputs import (
        ABSENT_STAGES,
        absent_stage_flags,
        assemble_scores,
        bundle_content_hash,
        stage_ledger_generation,
        stage_model_evidence,
    )

    score_document = _load_score_document(root)
    scores = assemble_scores(score_document)
    finality = _load_finality(root)

    evidence_path = root / "model_evidence.json"
    if not evidence_path.is_file():
        raise fail("VALIDATION_FAILED", "model evidence artifact is missing")
    stage_model_evidence(evidence_path, root / "legacy")

    ledger_tar = root / "ledger_generation.tar"
    if not ledger_tar.is_file():
        raise fail("VALIDATION_FAILED", "ledger generation not bound")
    stage_ledger_generation(ledger_tar, root / "legacy")
    tickers = sorted(set(parameters["tickers"]))
    years = range(int(parameters["year_start"]), int(parameters["year_end"]) + 1)
    scorer = Scorer(context=FeatureContext.load(tickers, years=years))

    as_of = parameters["session"]
    horizon_days = int(parameters.get("horizon_days", 35))
    alt_strikes = int(parameters.get("alt_strikes", 1))
    board = pd.DataFrame(score_document.get("rows") or [])
    fill_alpha = float(board["fill"].iloc[0]) if len(board) and "fill" in board else 0.5

    meta = build_meta(scores, as_of=as_of, horizon_days=horizon_days,
                      fill_alpha=fill_alpha, alt_strikes=alt_strikes,
                      freshness=freshness_summary(as_of), quota=quota_state(),
                      registry=scorer.registry)
    meta["execution_clock"] = {"requested_as_of": str(pd.Timestamp(as_of).date()),
                               "resolved_as_of": str(pd.Timestamp(as_of).date()),
                               "finality": finality}
    health = build_health(as_of=as_of,
                          size_mae=size_model_mae_from_ledger(panel=scorer.context.panel))

    output = root / "bundle"
    result = render_bundle(scores, output, as_of=as_of, horizon_days=horizon_days,
                           fill_alpha=fill_alpha, alt_strikes=alt_strikes,
                           panel=scorer.context.panel, trades=scorer.trades,
                           meta=meta, health=health, flags=absent_stage_flags(),
                           registry=scorer.registry)
    _write_action(root, "meta.json", meta)
    _write_action(root, "health.json", health)
    with tarfile.open(root / "bundle.tar", "w") as archive:
        archive.add(output, arcname="bundle")
    return _write_action(root, "render.json", {
        "bundle_archive": "bundle.tar", "bundle_content_hash": bundle_content_hash(output),
        "absent_stages": list(ABSENT_STAGES), "result": result}) | {"path": "bundle.tar"}


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
