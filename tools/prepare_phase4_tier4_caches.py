#!/usr/bin/env python3
"""Prepare old Tier-4 serving caches for bounded Phase 4 capture.

The Phase 4 corpus touches several historical Tier-4 folds, for every
producer ``Scorer._serving`` and ``Scorer._crush_forecast`` reach:
``implied_t1`` (``pred_im_t1_d14``), ``size`` (``pred_abs_move``),
``runup_move`` (``pred_runup_abs_move_d14``) and ``iv_crush``
(``pred_iv_crush_30``) — see ``MODEL_CHOICES``. Caches written before the
residual-pool embedding change contain the fitted estimator but not
``pool_pred``/``pool_res``.  Serving one of those files recomputes the pool
while a fully populated ``Scorer`` is resident, which exceeds this host's
memory budget.

This utility upgrades only existing old-format caches that match all of:

* the current registered champion for the requested ``--model``
  (``implied_t1`` by default, matching pre-``--model`` behaviour);
* the current Tier-3 panel snapshot and feature order; and
* a fold selected by the Phase 4 capture workload (or an explicit ``--fold``,
  which applies to every requested model — see ``phase4_required_folds``).

Each cache is rebuilt in its own subprocess.  The child calls Tier 4's existing
``_pool_before`` implementation, verifies the result and the unchanged model
payload, writes a temporary sibling, fsyncs it, and atomically replaces the
old file.  Missing caches are not created: ``serving_model`` already writes
new caches with embedded pools.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import paths  # noqa: E402
from engine.data import store  # noqa: E402
from engine.data.features import tier4  # noqa: E402

MAX_CACHE_BYTES = 64 * 1024 * 1024
HEARTBEAT_SECONDS = 30.0
POOL_KEYS = frozenset({"pool_pred", "pool_res"})

#: ``--model`` names -> the Tier-4 ``produces`` column each resolves to,
#: exactly as ``Scorer._serving`` resolves them:
#:
#: - ``implied_t1`` -> ``pred_im_t1_d14``: ``_chooser_frame``'s
#:   ``self._serving(fold, produces="pred_im_t1_d14")``
#:   (engine/score.py:3438-3442) -> ``tier4.feature_model("pred_im_t1_d14")``
#:   -> ``tier4.im_t1_feature_model`` (engine/data/features/tier4.py:273).
#: - ``runup_move`` -> ``pred_runup_abs_move_d14``: the same
#:   ``_chooser_frame`` loop, second entry (engine/score.py:3438-3442) ->
#:   ``tier4.feature_model("pred_runup_abs_move_d14")`` ->
#:   ``tier4.runup_move_feature_model`` (tier4.py:347).
#: - ``size`` -> ``pred_abs_move``: ``_chooser_frame``'s own forecast call,
#:   ``self._serving(tier4.serving_fold(...))`` with no ``produces`` kwarg
#:   (engine/score.py:3363-3365), so ``_serving`` takes its
#:   ``produces="pred_abs_move"`` default (engine/score.py:2569) and passes
#:   ``model=None`` to ``tier4.serving_model`` (engine/score.py:2577-2578),
#:   which defaults ``model`` to ``size_feature_model()`` itself
#:   (tier4.py:1291). ``tier4.feature_model("pred_abs_move")`` reaches the
#:   identical factory through ``FEATURE_MODELS["pred_abs_move"]`` (tier4.py:
#:   446), so resolving it that way here matches the live default exactly.
#: - ``iv_crush`` -> ``pred_iv_crush_30``: ``Scorer._crush_forecast``'s
#:   ``self._serving(fold, produces="pred_iv_crush_30")``
#:   (engine/score.py:2908-2913) -> ``tier4.feature_model("pred_iv_crush_30")``
#:   -> ``tier4.iv_crush_feature_model`` (tier4.py:401).
MODEL_CHOICES: dict[str, str] = {
    "implied_t1": "pred_im_t1_d14",
    "size": "pred_abs_move",
    "runup_move": "pred_runup_abs_move_d14",
    "iv_crush": "pred_iv_crush_30",
}

#: Kept as the sole default so an unflagged run upgrades exactly what it
#: upgraded before ``--model`` existed.
DEFAULT_MODEL = "implied_t1"


def _resolve_model(name: str, registry=None) -> "tier4.FeatureModel":
    """The ``FeatureModel`` for one ``--model`` name, resolved as ``Scorer._serving`` does."""
    try:
        produces = MODEL_CHOICES[name]
    except KeyError:
        raise CachePreparationError(
            f"{name!r} is not a known --model; choices: {sorted(MODEL_CHOICES)} or 'all'"
        ) from None
    return tier4.feature_model(produces, registry)


def _selected_models(requested: list[str]) -> list[str]:
    """``--model`` values -> the ordered, de-duplicated model names to upgrade.

    No flag at all means ``[DEFAULT_MODEL]`` (unchanged pre-``--model``
    behaviour). Any occurrence of ``all`` expands to every ``MODEL_CHOICES``
    name in its declared order, regardless of what else was passed.
    """
    values = requested or [DEFAULT_MODEL]
    if "all" in values:
        return list(MODEL_CHOICES)
    ordered: list[str] = []
    for name in values:
        if name not in ordered:
            ordered.append(name)
    return ordered


class CachePreparationError(RuntimeError):
    """A cache cannot be upgraded without changing or guessing its meaning."""


@dataclass(frozen=True)
class CacheTarget:
    path: Path
    fold: pd.Timestamp
    original_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _fold(value: object) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except Exception as exc:
        raise CachePreparationError(f"invalid fold {value!r}: {exc}") from exc
    if pd.isna(stamp):
        raise CachePreparationError("fold must not be missing")
    stamp = stamp.normalize()
    expected = pd.Timestamp(tier4.fold_start_of([stamp]).iloc[0])
    if stamp != expected:
        raise CachePreparationError(f"fold {stamp.date()} is not a {tier4.CADENCE} fold boundary")
    return stamp


def _load_bounded(path: Path, expected_sha256: str | None = None) -> Mapping:
    if path.is_symlink() or not path.is_file():
        raise CachePreparationError(f"cache must be a regular file: {path}")
    size = path.stat().st_size
    if size > MAX_CACHE_BYTES:
        raise CachePreparationError(
            f"cache is {size:,} bytes, above the {MAX_CACHE_BYTES:,}-byte limit: {path}"
        )
    actual = _sha256(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise CachePreparationError(
            f"cache changed after discovery: {path} expected {expected_sha256}, got {actual}"
        )
    try:
        stored = joblib.load(path)
    except Exception as exc:
        raise CachePreparationError(f"cannot load serving cache {path}: {exc}") from exc
    if not isinstance(stored, Mapping):
        raise CachePreparationError(f"serving cache is not a mapping: {path}")
    return stored


def _validate_identity(
    stored: Mapping,
    *,
    path: Path,
    model: tier4.FeatureModel,
    snapshot: str,
    fold: pd.Timestamp,
) -> None:
    expected_path = tier4._serving_path(model.model_id, fold, snapshot)
    if path.resolve() != expected_path.resolve():
        raise CachePreparationError(
            f"cache path does not match its current model/fold/snapshot identity: {path}"
        )
    expected = {
        "model_id": model.model_id,
        "fold_start": fold,
        "tier3_snapshot": snapshot,
        "features": tuple(model.features),
    }
    try:
        stored_fold = pd.Timestamp(stored.get("fold_start")).normalize()
    except Exception as exc:
        raise CachePreparationError(
            f"cache has an invalid fold_start in {path}: {stored.get('fold_start')!r}"
        ) from exc
    actual = {
        "model_id": stored.get("model_id"),
        "fold_start": stored_fold,
        "tier3_snapshot": stored.get("tier3_snapshot"),
        "features": tuple(stored.get("features", ())),
    }
    if actual != expected:
        raise CachePreparationError(
            f"cache identity mismatch for {path}: expected {expected}, got {actual}"
        )
    if "estimator" not in stored:
        raise CachePreparationError(f"cache has no estimator: {path}")


def _validated_pools(stored: Mapping, path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    present = POOL_KEYS.intersection(stored)
    if not present:
        return None
    if present != POOL_KEYS:
        raise CachePreparationError(
            f"cache has a partial residual pool ({sorted(present)}): {path}"
        )
    pred = np.asarray(stored["pool_pred"], dtype=float)
    res = np.asarray(stored["pool_res"], dtype=float)
    if pred.ndim != 1 or res.ndim != 1 or pred.shape != res.shape:
        raise CachePreparationError(f"cache residual pools are not paired 1-D arrays: {path}")
    if not np.isfinite(pred).all() or not np.isfinite(res).all():
        raise CachePreparationError(f"cache residual pools contain non-finite values: {path}")
    return pred, res


def discover_targets(
    folds: Iterable[pd.Timestamp],
    *,
    cache_dir: Path | None = None,
    model: tier4.FeatureModel | None = None,
    snapshot: str | None = None,
) -> tuple[list[CacheTarget], list[pd.Timestamp]]:
    """Return old matching caches and requested folds with no existing cache."""
    selected_model = model or tier4.im_t1_feature_model()
    selected_snapshot = snapshot or store.file_sha256(paths.PANEL)
    directory = Path(cache_dir) if cache_dir is not None else tier4.SERVING_DIR
    targets: list[CacheTarget] = []
    missing: list[pd.Timestamp] = []
    for selected_fold in sorted({_fold(value) for value in folds}):
        path = (
            directory
            / tier4._serving_path(selected_model.model_id, selected_fold, selected_snapshot).name
        )
        if path.is_symlink():
            raise CachePreparationError(f"cache must not be a symlink: {path}")
        if not path.exists():
            missing.append(selected_fold)
            continue
        digest = _sha256(path)
        stored = _load_bounded(path, digest)
        _validate_identity(
            stored,
            path=path,
            model=selected_model,
            snapshot=selected_snapshot,
            fold=selected_fold,
        )
        if _validated_pools(stored, path) is None:
            targets.append(CacheTarget(path, selected_fold, digest))
    return targets, missing


def phase4_required_folds(
    as_of: pd.Timestamp,
    *,
    forward_days: int,
    max_events: int,
    boundary_events: int,
) -> tuple[pd.Timestamp, ...]:
    """Plan folds from the same events and strategy windows as corpus capture.

    Model-independent: every fold comes from ``tier4.serving_fold(event_date,
    decision_date)``, which takes no model argument, so this same fold set is
    what every Tier-4 producer needs for a given capture window — it is
    planned once in ``main`` and reused across every ``--model`` selection
    rather than recomputed per model.
    """
    from engine import replay as replay_mod
    from engine import score as score_mod
    from engine.calendar import trading_calendar
    from engine.structures import STRUCTURES
    from tools.capture_tier0_corpus import _boundary_events, _events

    calendar = trading_calendar()
    event_sets = (
        _events(as_of, forward_days, max_events),
        _boundary_events(as_of, boundary_events, calendar),
    )
    folds: set[pd.Timestamp] = set()
    for events in event_sets:
        if events.empty:
            continue
        for strategy, factory in STRUCTURES.items():
            if strategy in score_mod.DISABLED_STRATEGIES:
                continue
            if score_mod.superseded_by(strategy) is not None:
                continue
            plan = replay_mod.plan_events(factory(), events, calendar=calendar)
            for row in plan.frame.to_dict("records"):
                folds.add(tier4.serving_fold(row["event_date"], row["decision_date"]))
    return tuple(sorted(folds))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def upgrade_one(
    target: CacheTarget,
    *,
    model: tier4.FeatureModel | None = None,
    snapshot: str | None = None,
    panel_loader: Callable[[], pd.DataFrame] | None = None,
    pool_builder: Callable[[pd.Timestamp, tier4.FeatureModel, pd.DataFrame], tuple] | None = None,
) -> int:
    """Upgrade one old cache atomically; return the stored residual count."""
    selected_model = model or tier4.im_t1_feature_model()
    selected_snapshot = snapshot or store.file_sha256(paths.PANEL)
    original = _load_bounded(target.path, target.original_sha256)
    _validate_identity(
        original,
        path=target.path,
        model=selected_model,
        snapshot=selected_snapshot,
        fold=target.fold,
    )
    if _validated_pools(original, target.path) is not None:
        return int(np.asarray(original["pool_pred"]).size)

    if panel_loader is None:
        from engine.features import load_panel

        panel_loader = load_panel
    panel = panel_loader()
    builder = pool_builder or tier4._pool_before
    pool_pred, pool_res = builder(target.fold, selected_model, panel)
    pool_pred = np.asarray(pool_pred, dtype=float)
    pool_res = np.asarray(pool_res, dtype=float)
    candidate = dict(original)
    candidate["pool_pred"] = pool_pred
    candidate["pool_res"] = pool_res
    _validated_pools(candidate, target.path)

    original_estimator_hash = joblib.hash(original["estimator"])
    original_metadata_hash = joblib.hash(
        {key: value for key, value in original.items() if key not in POOL_KEYS | {"estimator"}}
    )
    temporary = target.path.parent / ("." + target.path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        joblib.dump(candidate, temporary)
        os.chmod(temporary, target.path.stat().st_mode & 0o7777)
        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

        verified = _load_bounded(temporary)
        _validate_identity(
            verified,
            path=target.path,
            model=selected_model,
            snapshot=selected_snapshot,
            fold=target.fold,
        )
        verified_pools = _validated_pools(verified, target.path)
        assert verified_pools is not None
        if not np.array_equal(verified_pools[0], pool_pred) or not np.array_equal(
            verified_pools[1], pool_res
        ):
            raise CachePreparationError(f"temporary pool verification failed: {target.path}")
        if joblib.hash(verified["estimator"]) != original_estimator_hash:
            raise CachePreparationError(f"temporary estimator changed: {target.path}")
        verified_metadata_hash = joblib.hash(
            {key: value for key, value in verified.items() if key not in POOL_KEYS | {"estimator"}}
        )
        if verified_metadata_hash != original_metadata_hash:
            raise CachePreparationError(f"temporary cache metadata changed: {target.path}")
        if _sha256(target.path) != target.original_sha256:
            raise CachePreparationError(f"cache changed before replacement: {target.path}")
        os.replace(temporary, target.path)
        _fsync_directory(target.path.parent)
    finally:
        temporary.unlink(missing_ok=True)

    final = _load_bounded(target.path)
    _validate_identity(
        final,
        path=target.path,
        model=selected_model,
        snapshot=selected_snapshot,
        fold=target.fold,
    )
    final_pools = _validated_pools(final, target.path)
    if (
        final_pools is None
        or not np.array_equal(final_pools[0], pool_pred)
        or not np.array_equal(final_pools[1], pool_res)
    ):
        raise CachePreparationError(f"installed cache verification failed: {target.path}")
    return int(pool_pred.size)


def _run_child(target: CacheTarget, produces: str, snapshot: str) -> None:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--upgrade-one",
        str(target.path),
        "--expected-sha256",
        target.original_sha256,
        "--expected-fold",
        str(target.fold.date()),
        "--expected-snapshot",
        snapshot,
        "--expected-produces",
        produces,
    ]
    process = subprocess.Popen(command, cwd=str(ROOT))
    started = time.monotonic()
    while True:
        try:
            code = process.wait(timeout=HEARTBEAT_SECONDS)
            break
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - started
            print(
                f"[phase4-cache] fold {target.fold:%Y-%m} ({produces}) still rebuilding "
                f"({elapsed:.0f}s elapsed)",
                flush=True,
            )
    if code != 0:
        raise CachePreparationError(
            f"fold {target.fold:%Y-%m} ({produces}) child exited with status {code}"
        )


def _worker(args: argparse.Namespace) -> int:
    # Resolved exactly as `discover_targets`/`main` resolved it when planning
    # this target: `tier4.feature_model(produces)` covers all four producers,
    # `pred_abs_move` included (see MODEL_CHOICES above for why that matches
    # `Scorer._serving`'s own default).
    model = tier4.feature_model(args.expected_produces)
    snapshot = store.file_sha256(paths.PANEL)
    if snapshot != args.expected_snapshot:
        raise CachePreparationError(
            "Tier-3 panel changed between planning and child execution: "
            f"expected {args.expected_snapshot}, got {snapshot}"
        )
    target = CacheTarget(
        Path(args.upgrade_one).resolve(),
        _fold(args.expected_fold),
        args.expected_sha256,
    )
    print(
        f"[phase4-cache] fold {target.fold:%Y-%m} model={model.model_id} "
        f"({args.expected_produces}) loading panel",
        flush=True,
    )
    started = time.monotonic()
    count = upgrade_one(target, model=model, snapshot=snapshot)
    elapsed = time.monotonic() - started
    print(
        f"[phase4-cache] fold {target.fold:%Y-%m} ({args.expected_produces}) "
        f"installed {count:,} residuals in {elapsed:.0f}s",
        flush=True,
    )
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--forward-days", type=int, default=35)
    parser.add_argument("--max-events", type=int, default=40)
    parser.add_argument("--boundary-events", type=int, default=4)
    parser.add_argument(
        "--fold",
        action="append",
        default=[],
        help="upgrade this YYYY-MM-01 fold; repeat to bypass capture planning",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        choices=[*MODEL_CHOICES, "all"],
        help=(
            "Tier-4 producer to upgrade (repeatable); one of "
            f"{sorted(MODEL_CHOICES)} or 'all' for every producer "
            "Scorer._serving serves. Folds are the same for every model "
            "(phase4_required_folds keys them on event_date/decision_date, "
            "not on the producer). Default: "
            f"{DEFAULT_MODEL!r} (unchanged behaviour)."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--upgrade-one", help=argparse.SUPPRESS)
    parser.add_argument("--expected-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--expected-fold", help=argparse.SUPPRESS)
    parser.add_argument("--expected-snapshot", help=argparse.SUPPRESS)
    parser.add_argument("--expected-produces", help=argparse.SUPPRESS)
    args = parser.parse_args(list(argv) if argv is not None else None)

    worker_values = (
        args.upgrade_one,
        args.expected_sha256,
        args.expected_fold,
        args.expected_snapshot,
        args.expected_produces,
    )
    if any(worker_values):
        if not all(worker_values):
            parser.error("internal worker mode requires all expected values")
        return _worker(args)

    as_of = (
        pd.Timestamp(args.as_of).normalize()
        if args.as_of is not None
        else pd.Timestamp.today().normalize()
    )
    # phase4_required_folds keys every fold on (event_date, decision_date)
    # via tier4.serving_fold, with no model in that computation at all — so
    # the same fold set is required by every producer, and it is planned
    # once and reused across --model selections rather than per model.
    folds = (
        tuple(_fold(value) for value in args.fold)
        if args.fold
        else phase4_required_folds(
            as_of,
            forward_days=args.forward_days,
            max_events=args.max_events,
            boundary_events=args.boundary_events,
        )
    )
    model_names = _selected_models(args.model)

    snapshot = store.file_sha256(paths.PANEL)
    plan: list[tuple[str, str, "tier4.FeatureModel", list[CacheTarget], list[pd.Timestamp]]] = []
    total_targets = 0
    total_missing = 0
    for name in model_names:
        produces = MODEL_CHOICES[name]
        model = _resolve_model(name)
        targets, missing = discover_targets(folds, model=model, snapshot=snapshot)
        plan.append((name, produces, model, targets, missing))
        total_targets += len(targets)
        total_missing += len(missing)

    print(
        f"[phase4-cache] planned {len(folds)} fold(s) x {len(model_names)} model(s) "
        f"({', '.join(model_names)}); "
        f"{total_targets} old cache(s), {total_missing} missing cache(s)",
        flush=True,
    )
    for name, produces, model, targets, missing in plan:
        print(
            f"[phase4-cache] {name} ({produces}, model_id={model.model_id}): "
            f"{len(targets)} old cache(s), {len(missing)} missing cache(s)",
            flush=True,
        )
        for target in targets:
            print(
                f"[phase4-cache] old {name} {target.fold:%Y-%m} {target.path.name} "
                f"sha256={target.original_sha256}",
                flush=True,
            )
        for fold in missing:
            print(
                f"[phase4-cache] missing {name} {fold:%Y-%m}; "
                "left untouched because no old cache exists",
                flush=True,
            )
    if args.dry_run:
        return 0

    total = total_targets
    index = 0
    for name, produces, model, targets, missing in plan:
        for target in targets:
            index += 1
            print(
                f"[phase4-cache] upgrading {index}/{total}: {name} fold {target.fold:%Y-%m}",
                flush=True,
            )
            _run_child(target, produces, snapshot)
    print(f"[phase4-cache] complete: upgraded {total} cache(s)", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CachePreparationError as exc:
        print(f"[phase4-cache] ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2) from exc
