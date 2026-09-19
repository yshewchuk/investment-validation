#!/usr/bin/env python3
"""P5-6: the calibration fold keys a set of score requests will ask for.

Payoff and recalibration artifacts are keyed per ``(strategy, alpha,
cutoff)`` (``payoff_artifact_key``: alpha at 4dp, cutoff as an ISO date).
Legacy's cutoff is each request's ``evidence_cutoff`` = min(decision date,
entry date) (``engine/score.py``), so STR-THRU asks for the decision date and
STR-RUNUP for its entry date, fourteen trading days earlier. A release that
must replay or serve those requests needs exactly one fold per distinct key.

Two sources:

* ``--phase4-corpus`` -- every pair carrying an ``input_trace``: strategy and
  alpha from the saved request (``strategy_version``, ``fill_model.alpha``),
  cutoff from the pair's legacy record ``evidence_cutoff`` (legacy's own
  causal key for that request; no value is read), else min(``as_of``,
  ``entry_date``) from the trace's native context. A pair with no derivable
  key is listed, never guessed.
* ``--as-of`` (a nightly) -- STR-THRU at the as-of date, and STR-RUNUP at each
  ``--runup-entry-date`` (the board's STR-RUNUP entry dates; the as-of alone
  does not determine them). ``--alpha`` is repeatable.

Each key maps to the catalog members its strategy reads (STR-THRU: payoff
line + recalibration map; STR-RUNUP: payoff surface) plus the catalog-only
STR-RUNUP line and map (no v2 consumer, but the catalog stages them; my
judgement call to build them at the same keys so the release is complete).
Strategies with no catalog calibration member are listed as uncatalogued.

Output: a keys JSON (``--out``) and the ordered training-job commands,
``--plan-only`` first, one per (recipe, alpha) with a ``--cutoff`` per fold.
With ``--release-root`` it also reports which keys a staged release already
holds. Light: reads pair JSON files and (optionally) the staged calibration
objects; writes only ``--out``, which may not be inside ``data/``.

Usage::

    python3 tools/phase5_calibration_keys.py --phase4-corpus <corpus> \\
        --train-root /root/p5-3-runs --out /root/p5-6/calibration-keys.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: strategy -> the catalog calibration members a request of it reads (first)
#: and the catalog-only members staged at the same keys (after).
MEMBERS_BY_STRATEGY = {
    "STR-THRU": ("payoff_line:STR-THRU", "recalibration_map:STR-THRU"),
    "STR-RUNUP": ("payoff_surface:STR-RUNUP", "payoff_line:STR-RUNUP",
                  "recalibration_map:STR-RUNUP"),
}
CONSUMED = {"payoff_line:STR-THRU", "recalibration_map:STR-THRU", "payoff_surface:STR-RUNUP"}

Key = tuple  # (strategy, alpha, cutoff)


def _key(strategy: str, alpha: Any, cutoff: Any) -> Key:
    from engine.v2.models.payoff_artifact import payoff_artifact_key

    return payoff_artifact_key(strategy, alpha, cutoff)


def _pair_key(pair: Mapping[str, Any]) -> tuple[Key | None, str]:
    payload = pair.get("payload") or {}
    request = payload.get("request") or {}
    record = payload.get("record") or {}
    strategy = request.get("strategy_version") or record.get("strategy")
    alpha = (request.get("fill_model") or {}).get("alpha", record.get("fill"))
    cutoff = record.get("evidence_cutoff")
    if cutoff is None:
        context = ((payload.get("input_trace") or {}).get("native_inputs") or {}).get(
            "context") or {}
        dates = [str(context[k])[:10] for k in ("as_of", "entry_date")
                 if isinstance(context.get(k), str)]
        cutoff = min(dates) if dates else None
    missing = [name for name, value in (("strategy", strategy), ("alpha", alpha),
                                        ("cutoff", cutoff)) if value is None]
    if missing:
        return None, "no " + "/".join(missing)
    return _key(strategy, alpha, cutoff), ""


def keys_from_corpus(corpus: Path) -> tuple[set[Key], dict[str, str], int]:
    """``(keys, {fixture_id: reason} for traced pairs without one, traced count)``."""
    from checks.tier0_corpus import load, resolve_corpus

    loaded = load(resolve_corpus(Path(corpus)))
    keys, underivable, traced = set(), {}, 0
    for fixture_id in loaded.ordered_ids:
        pair = loaded.pairs[fixture_id]
        if not (pair.get("payload") or {}).get("input_trace"):
            continue
        traced += 1
        key, reason = _pair_key(pair)
        if key is None:
            underivable[fixture_id] = reason
        else:
            keys.add(key)
    return keys, underivable, traced


def keys_from_as_of(as_of: str, alphas: Iterable[float],
                    runup_entry_dates: Iterable[str] = ()) -> set[Key]:
    keys = set()
    for alpha in alphas:
        keys.add(_key("STR-THRU", alpha, as_of))
        for entry in runup_entry_dates:
            keys.add(_key("STR-RUNUP", alpha, min(str(entry)[:10], str(as_of)[:10])))
    return keys


def member_keys(keys: Iterable[Key]) -> tuple[dict[str, list[Key]], list[Key]]:
    """``({member_id: sorted keys}, uncatalogued keys)``."""
    by_member: dict[str, set[Key]] = {}
    uncatalogued = []
    for key in keys:
        members = MEMBERS_BY_STRATEGY.get(key[0])
        if not members:
            uncatalogued.append(key)
            continue
        for member in members:
            by_member.setdefault(member, set()).add(key)
    return ({m: sorted(v, key=_sort) for m, v in sorted(by_member.items())},
            sorted(uncatalogued, key=_sort))


def _sort(key: Key):
    return (key[0], key[1], key[2] or "")


def training_commands(by_member: Mapping[str, list[Key]], train_root: Path, *,
                      max_rss_gb: float = 2.5) -> list[str]:
    """Ordered job commands: every ``--plan-only`` first, then the real runs."""
    jobs = []
    for member, keys in by_member.items():
        by_alpha: dict[float, list[str]] = {}
        for _strategy, alpha, cutoff in keys:
            by_alpha.setdefault(alpha, []).append(cutoff)
        for alpha, cutoffs in sorted(by_alpha.items()):
            out = Path(train_root) / f"{member.replace(':', '__')}__a{alpha}"
            cuts = " ".join(f"--cutoff {c}" for c in sorted(c for c in cutoffs if c))
            jobs.append(
                f"python3 tools/bounded_run.py --max-rss-gb {max_rss_gb} -- python3 -u "
                f"tools/phase5_training_job.py --recipe {member}:calibration "
                f"--alpha {alpha} {cuts} --out {out}")
    return [f"{job} --plan-only" for job in jobs] + jobs


def staged_keys(release_root: Path) -> dict[str, set[Key]]:
    """The ``(strategy, alpha, cutoff)`` keys a staged release holds, per member."""
    from checks.phase5_release import deployment_root, read_manifest
    from engine.v2.models.payoff_artifact import PayoffArtifactLoader, PayoffArtifactRef
    from engine.v2.models.recalibration_artifact import (
        RecalibrationArtifactLoader,
        RecalibrationArtifactRef,
    )

    manifest = read_manifest(Path(release_root))
    base = deployment_root(Path(release_root))
    held: dict[str, set[Key]] = {}
    for row in manifest.get("members", ()):
        member = row["member_id"]
        if member.split(":")[0] not in ("payoff_line", "payoff_surface", "recalibration_map"):
            continue
        for obj in row.get("objects", ()):
            if member.startswith("recalibration_map"):
                artifact = RecalibrationArtifactLoader(base).load(RecalibrationArtifactRef(
                    path=obj["path"], content_hash=obj["content_hash"]))
            else:
                artifact = PayoffArtifactLoader(base).load(PayoffArtifactRef(
                    path=obj["path"], content_hash=obj["content_hash"]))
            held.setdefault(member, set()).add(tuple(artifact.key))
    return held


def plan(keys: set[Key], *, train_root: Path, release_root: Path | None = None,
         source: dict | None = None) -> dict:
    by_member, uncatalogued = member_keys(keys)
    out = {"source": source or {}, "keys": sorted((list(k) for k in keys),
                                                   key=lambda k: _sort(tuple(k))),
           "members": {m: [list(k) for k in v] for m, v in by_member.items()},
           "consumed_members": sorted(m for m in by_member if m in CONSUMED),
           "uncatalogued": [list(k) for k in uncatalogued],
           "commands": training_commands(by_member, train_root)}
    if release_root is not None:
        held = staged_keys(release_root)
        out["missing_from_release"] = {
            m: [list(k) for k in v if tuple(k) not in held.get(m, set())]
            for m, v in by_member.items()}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--phase4-corpus", type=Path)
    source.add_argument("--as-of", help="nightly decision date YYYY-MM-DD")
    parser.add_argument("--alpha", type=float, action="append",
                        help="fill alpha (with --as-of; repeatable)")
    parser.add_argument("--runup-entry-date", action="append", default=[],
                        help="a board STR-RUNUP entry date (with --as-of; repeatable)")
    parser.add_argument("--train-root", type=Path, required=True,
                        help="parent dir for the training-job --out dirs")
    parser.add_argument("--release-root", type=Path,
                        help="report which keys this staged release already holds")
    parser.add_argument("--out", type=Path, help="write the keys JSON here (not in data/)")
    args = parser.parse_args(argv)
    from engine import paths

    for target in (args.out, args.train_root):
        if target is not None:
            resolved = target.resolve()
            if resolved == paths.DATA.resolve() or paths.DATA.resolve() in resolved.parents:
                parser.error(f"{target} may not be inside data/")
    if args.phase4_corpus is not None:
        keys, underivable, traced = keys_from_corpus(args.phase4_corpus)
        src = {"phase4_corpus": str(args.phase4_corpus), "traced_pairs": traced,
               "underivable": underivable}
    else:
        if not args.alpha:
            parser.error("--as-of needs at least one --alpha")
        keys = keys_from_as_of(args.as_of, args.alpha, args.runup_entry_date)
        src = {"as_of": args.as_of, "runup_entry_dates": sorted(args.runup_entry_date)}
    result = plan(keys, train_root=args.train_root, release_root=args.release_root,
                  source=src)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(f"keys: {len(result['keys'])}  members: "
          + ", ".join(f"{m}={len(v)}" for m, v in result["members"].items())
          + f"  uncatalogued: {len(result['uncatalogued'])}")
    if src.get("underivable"):
        print(f"underivable traced pairs: {len(src['underivable'])}")
    for member, missing in (result.get("missing_from_release") or {}).items():
        print(f"missing from release {member}: {len(missing)}")
    print("\n".join(result["commands"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
