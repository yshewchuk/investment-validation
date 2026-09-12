#!/usr/bin/env python3
"""Export the frozen baseline compatibility package (`§3.2`, phase 0 step 3).

    python3 tools/baseline_export.py                    # baseline/<today>/
    python3 tools/baseline_export.py --out /tmp/b2      # anywhere, for a diff
    python3 tools/baseline_export.py --verify /tmp/b2   # byte-compare two exports

§12 deletes legacy whole at phase 8 and trusts the replacement because it
produces numbers identical to the old one. That rests entirely on parity, and
parity needs an oracle. The tier-0 corpus is the oracle's *answers*; this
package is what the answers are answers **about** — without it, a fixture that
disagrees cannot be attributed to a strategy, a model, a library or a clock.

Five parts, per §3.2 and the guide's §6:

1. ``environment_lock.json`` + ``requirements.txt`` — Python, platform, commit,
   and the exact version of every third-party package the code imports. Comes
   first: a fixture captured in an unpinned environment cannot later distinguish
   a code change from a library upgrade.
2. ``definitions/`` — the **resolved** structures, offsets, gates and DYN-SV
   menu. Resolved, not referenced: a definition that has to be re-derived from
   code is not frozen.
3. ``artifacts/`` — every registered model with its fingerprint, the feature
   recipes, the role dependency graph, the fitted-state inventory and the named
   legacy adapters.
4. ``conventions/`` — calendar version, quote and fill conventions, the data
   snapshot, and the display/precision rules a replay has to survive.
5. The tier-0 corpus itself, written by ``tools/capture_tier0_corpus.py``.

**Nothing here contains a wall clock.** A second export from the same commit
and snapshot is byte-identical, which is the §6 acceptance criterion, and a
timestamp inside the payload would break it on the first re-run. The date is
the directory name, and it is an argument.

**It is private.** The definitions carry no quotes, but the corpus beside them
does, and splitting the package across two remotes is how half of it goes
missing. ``checks/repo_hygiene.py`` blocks ``baseline/`` from the public repo
and ``tools/private_mirror.py`` carries it to the private one.
"""
from __future__ import annotations

import argparse
import ast
import json
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import date
from importlib import metadata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import entry_rules, forecast_sizing, paths, score, structures  # noqa: E402
from engine.models import registry as model_registry  # noqa: E402
from engine import structure_registry  # noqa: E402
from engine.v2.diagnosis import content_hash  # noqa: E402

SCHEMA_VERSION = "baseline_package.v1.0"

#: §5.5 knowledge modes. Nothing captured today is `observed`: that mode is
#: reserved for decisions made under a live clock with availability evidence,
#: and it is never granted retroactively.
ATTESTED = "attested_stable"
RECONSTRUCTED = "reconstructed"

#: Trees whose imports define the environment. `experiments/` is deliberately
#: absent: its runners import each other by bare module name, which would put
#: a dozen local modules into a dependency lock.
_IMPORT_ROOTS = ("engine", "checks", "tools", "dashboard", "tests")

#: §6.1 of the guide: `_serving_path` carries a panel hash but neither seed nor
#: runtime, and this is the gap that closes.
_LOCK_KEYS = ("python", "platform", "commit", "packages")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def plain(value: Any) -> Any:
    """Make a value JSON-writable without losing precision or inventing one."""
    if is_dataclass(value) and not isinstance(value, type):
        return plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        seq = sorted(value) if isinstance(value, (set, frozenset)) else value
        return [plain(v) for v in seq]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if callable(value):
        return getattr(value, "__name__", repr(value))
    return str(value)


def _git(*args: str) -> str:
    proc = subprocess.run(["git", "-C", str(ROOT), *args],
                          capture_output=True, text=True, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else ""


def part(payload: Any, *, knowledge_mode: str, describes: str) -> dict:
    """One export part: a payload, its knowledge mode, and what it is."""
    return {
        "schema_version": SCHEMA_VERSION,
        "describes": describes,
        "knowledge_mode": knowledge_mode,
        "payload": plain(payload),
    }


# --------------------------------------------------------------------------
# 1. environment lock
# --------------------------------------------------------------------------


def _imported_roots() -> set[str]:
    std = set(sys.stdlib_module_names)
    local = set(_IMPORT_ROOTS) | {"experiments", "bt", "earnings_predictions"}
    out: set[str] = set()
    for root in _IMPORT_ROOTS:
        for path in sorted((ROOT / root).rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(errors="replace"))
            except (SyntaxError, OSError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    out.update(a.name.split(".")[0] for a in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                    out.add(node.module.split(".")[0])
    return {m for m in out if m not in std and m not in local}


def _distributions(roots: set[str]) -> dict[str, str]:
    """``{distribution: version}`` for every importable third-party root.

    Resolved through the installed metadata rather than a hand-written list,
    and a root with no distribution is dropped rather than guessed: a lock that
    names a package that is not installed cannot reproduce anything.
    """
    mapping = metadata.packages_distributions()
    out: dict[str, str] = {}
    for root in sorted(roots):
        for dist in mapping.get(root, []):
            try:
                out[dist] = metadata.version(dist)
            except metadata.PackageNotFoundError:  # pragma: no cover - defensive
                continue
    return out


def environment_lock() -> dict:
    packages = _distributions(_imported_roots())
    return part(
        {
            "python": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
                # Not `sys.executable`: a venv path is a property of this
                # machine, not of the baseline.
                "api_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            },
            "platform": {
                "system": platform.system(),
                "machine": platform.machine(),
                "libc": "-".join(x for x in platform.libc_ver() if x),
            },
            "commit": {
                "sha": _git("rev-parse", "HEAD"),
                "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": bool(_git("status", "--porcelain")),
            },
            "packages": packages,
        },
        knowledge_mode=ATTESTED,
        describes="the environment this baseline was captured in (§6.1)",
    )


def requirements_txt(lock: dict) -> str:
    lines = [
        "# Environment lock for the phase-0 baseline package.",
        "# Generated by tools/baseline_export.py — pinned, not ranged: a range",
        "# reintroduces exactly the ambiguity the lock exists to remove.",
        f"# python {lock['payload']['python']['version']} "
        f"on {lock['payload']['platform']['system']}/"
        f"{lock['payload']['platform']['machine']}",
        "",
    ]
    lines += [f"{name}=={ver}" for name, ver in sorted(lock["payload"]["packages"].items())]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# 2. resolved definitions
# --------------------------------------------------------------------------


def resolved_structures() -> dict:
    """Every structure in ``STRUCTURES``, built and serialized, not referenced."""
    out = {}
    for name, factory in structures.STRUCTURES.items():
        built = factory()
        out[name] = {
            "factory": factory.__name__,
            "structure": built.to_dict(),
            "decided_at": built.decided_at,
            "decided_early": built.decided_early,
            "holds_through_print": built.holds_through_print,
            "has_short_leg": built.has_short_leg,
            "execution_variant": structures.execution_variant_label(built),
        }
    return part(
        {
            "structures": out,
            "ladder": {
                "step": score.LADDER_STEP,
                "strike_decimals": score.LADDER_STRIKE_DP,
            },
            "forecast_sizing": {
                "sized_strategies": list(forecast_sizing.sized_strategies()),
                "width_bounds": [forecast_sizing.WIDTH_MIN, forecast_sizing.WIDTH_MAX],
                "twin_p_plateau_centre": forecast_sizing.PLATEAU_CENTRE,
                "twin_p5_peak": forecast_sizing.FIVE_STRIKE_PEAK,
                "short_vol_outer": forecast_sizing.SHORT_VOL_OUTER,
            },
            "d1": {
                "decision_offset": structures.D1_DECISION_OFFSET,
                "strategies": list(structures.D1_STRATEGIES),
                "offsets": structures.D1_DECISION_OFFSETS,
            },
        },
        knowledge_mode=ATTESTED,
        describes="resolved structure definitions, selectors, offsets and "
                  "forecast width rules (§6.2)",
    )


def _rule(rule: entry_rules.EntryRule) -> dict:
    return {
        "strategy": rule.strategy,
        "evidence": rule.evidence,
        "terms": [
            {"name": t.name, "describes": t.describes, "test": t.test.__name__,
             "needs": list(t.needs)}
            for t in rule.terms
        ],
    }


def resolved_gates() -> dict:
    """Arithmetic rules with their exact constants, and the fitted gates.

    §3.1 is explicit that these differ from the learned-gate domain floor and
    must not be consolidated, so the two floors are exported side by side with
    their different values rather than folded into one number.
    """
    fitted = [
        {
            "id": entry.id, "strategy": entry.strategy, "role": entry.role,
            "threshold": getattr(entry, "threshold", None),
            "champion": getattr(entry, "champion", None),
        }
        for entry in model_registry.load_registry().entries
        if entry.role == "gate"
    ]
    return part(
        {
            "arithmetic_rules": {name: _rule(rule)
                                 for name, rule in entry_rules.ENTRY_RULES.items()},
            "historical_rules": {"TWIN-P-legacy": _rule(entry_rules.TWIN_P_LEGACY_RULE)},
            "constants": {
                "max_rel_spread": entry_rules.MAX_REL_SPREAD,
                "mcap_floor_usd": entry_rules.MCAP_FLOOR,
                "trailing_bar": "trailing six-month top-20% simulated return",
            },
            "learned_gate_domain": {
                "mcap_floor_usd": score.GATE_MCAP_FLOOR,
                "note": "DIFFERENT from the arithmetic rules' floor above "
                        "(§3.1: do not consolidate apparently similar constants)",
                "thin_history_events": score.THIN_HISTORY_EVENTS,
                "atm_tolerance_pct": score.ATM_TOLERANCE_PCT,
                "min_quoted_implied_move": score.MIN_QUOTED_IMPLIED_MOVE,
            },
            "fitted_gates": fitted,
        },
        knowledge_mode=ATTESTED,
        describes="gate definitions: fitted gates with thresholds, and the "
                  "arithmetic rules with their exact constants (§6.2)",
    )


def resolved_dyn_sv() -> dict:
    return part(
        {
            "strategy": score.DYNAMIC_STRATEGY,
            "menu": list(score.DYNAMIC_MENU),
            "menu_is_ordered": True,
            "ranking_rule": "highest chooser_score within an event; "
                            "stable sort, na_position=last",
            "partial_menu": "ranked on the scores that exist; unscored "
                            "candidates cannot compete, matching the training "
                            "folds' np.isfinite mask",
            "missing_score_fallback": "rank by exp_pnl_sim (the pre-champion "
                                      "resolver) when NO candidate is scored",
            "tie_behaviour": "stable sort keeps the earlier menu member; an "
                             "event where no member produced either ranking "
                             "key yields no chooser row at all",
            "re_gating": "none — the winner keeps its own entry-rule verdict",
            "event_key": list(score._EVENT_KEY),
            "ladder_rows_excluded": True,
            "n_admissible_surrogate": {
                "training_median": score._N_ADMISSIBLE_MEDIAN,
                "note": "a documented serving approximation (§11): the live "
                        "value is a chain-depth-conditioned surrogate for a "
                        "grid-derived training input, registered rather than "
                        "declared equal",
            },
        },
        knowledge_mode=ATTESTED,
        describes="the DYN-SV menu, ranking rule, tie and fallback behaviour (§6.2)",
    )


#: How ``validation_status`` decides between promoted and tracked. Stated here
#: and exported beside the answer, because it is a DERIVATION and not a stored
#: field: nothing in the codebase records "promoted" as a per-structure
#: attribute, so any answer to that question is read off the evidence. Anyone
#: who disagrees with the reading can see exactly what it was applied to.
STATUS_RULE = (
    "disabled if the strategy is in engine.score.DISABLED_STRATEGIES; "
    "superseded if engine.structure_registry names a later champion; "
    "promoted if a champion gate is registered for it, or its arithmetic "
    "entry rule's evidence says 'promoted' without saying 'NOT promoted'; "
    "tracked otherwise."
)


def _status_of(name: str, gated_strategies: set[str]) -> tuple[str, str | None]:
    if name in score.DISABLED_STRATEGIES:
        return "disabled", "UNVALIDATED_STRUCTURE"
    if structure_registry.superseded_by(name) is not None:
        return "superseded", "SUPERSEDED"
    rule = entry_rules.rule_for(name)
    if rule is not None:
        evidence = rule.evidence.lower()
        if "not promoted" in evidence:
            return "tracked", None
        return ("promoted", None) if "promoted" in evidence else ("tracked", None)
    return ("promoted", None) if name in gated_strategies else ("tracked", None)


def validation_status() -> dict:
    live = set(structure_registry.live_strategies(list(structures.STRUCTURES)))
    gated = {
        e.strategy for e in model_registry.load_registry().entries
        if e.role == "gate" and getattr(e, "champion", False)
    }
    rows = {}
    for name in structures.STRUCTURES:
        status, reason = _status_of(name, gated)
        superseded = structure_registry.superseded_by(name)
        rule = entry_rules.rule_for(name)
        rows[name] = {
            "status": status,
            "refusal_code": reason,
            "refusal_detail": score.DISABLED_STRATEGIES.get(name),
            "superseded_by": superseded.strategy if superseded else None,
            "live": name in live,
            "family": structure_registry.family_of(name),
            "gate": "registered champion" if name in gated else (
                "arithmetic rule" if rule is not None else "none"),
            "evidence": rule.evidence if rule is not None else None,
        }
    return part(
        {"per_structure": rows, "families": structure_registry.FAMILIES,
         "flags": list(score.FLAGS), "status_derivation": STATUS_RULE,
         "champion_gated_strategies": sorted(gated)},
        knowledge_mode=ATTESTED,
        describes="validation status per structure: promoted, tracked, or "
                  "disabled with its refusal code (§6.2)",
    )


# --------------------------------------------------------------------------
# 3. artifacts and state
# --------------------------------------------------------------------------


def _artifact_row(entry) -> dict:
    row = plain(entry)
    artifact = ROOT / str(row.get("artifact", ""))
    row["artifact_present"] = artifact.exists()
    row["artifact_bytes"] = artifact.stat().st_size if artifact.exists() else None
    return row


def registered_models() -> dict:
    """All nine registered models, champions and not.

    §3.1 requires historical definitions preserved, so the two non-champions
    (`size_v1_3`, `gate_midfill_str_thru`) are exported beside the seven
    champions rather than filtered out.
    """
    registry = model_registry.load_registry()
    rows = [_artifact_row(e) for e in registry.entries]
    champions = [r["id"] for r in rows if r.get("champion")]
    return part(
        {
            "roles": list(model_registry.ROLES),
            "role_tier": dict(model_registry.ROLE_TIER),
            "model_tiers": list(model_registry.MODEL_TIERS),
            "tier4_columns": list(model_registry.TIER4_COLUMNS),
            "champion_key": "(strategy, role, decision_offset)",
            "models": rows,
            "champions": sorted(champions),
            "non_champions": sorted(r["id"] for r in rows if not r.get("champion")),
        },
        knowledge_mode=ATTESTED,
        describes="every registered model with its fingerprint, feature order "
                  "and role dependency graph (§6.3)",
    )


#: The named legacy adapters of `structure_generation_and_simulation.md` §6.1.
#: Their content is defined by this export rather than by their names — which
#: is why each one records the module and symbols it actually resolves to.
LEGACY_ADAPTERS = {
    "legacy.structure_selectors.v1": {
        "module": "engine.structures",
        "symbols": ["ExpirySelector", "StrikeSelector", "LegSpec", "Structure",
                    "price_structure", "ladder_strike"],
    },
    "legacy.paired_move_crush.v1": {
        "module": "engine.pnl_sim",
        "symbols": ["ResidualPool", "expected_pnl"],
    },
    "legacy.put_exit_bs.v1": {
        "module": "engine.pnl_sim",
        "symbols": ["black_scholes_put"],
    },
    "legacy.pnl_return.v1": {
        "module": "engine.structures",
        "symbols": ["structure_return"],
    },
    "legacy.payoff_map.v1": {
        "module": "engine.payoff",
        "symbols": ["PayoffMap", "RunupPayoffSurface"],
    },
    "legacy.runup_surface.v1": {
        "module": "engine.payoff",
        "symbols": ["RunupPayoffSurface"],
    },
}


def _symbol_digest(module_name: str, symbols: list[str]) -> dict:
    """Source digest of each named symbol, so the adapter's CONTENT is frozen."""
    path = ROOT / (module_name.replace(".", "/") + ".py")
    if not path.exists():
        return {"module_present": False, "symbols": {}}
    tree = ast.parse(path.read_text())
    text = path.read_text().splitlines()
    found: dict[str, Any] = {}
    for node in ast.walk(tree):
        name = getattr(node, "name", None)
        if name in symbols and isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            body = "\n".join(text[node.lineno - 1: node.end_lineno])
            found[name] = {"lines": node.end_lineno - node.lineno + 1,
                           "source_hash": content_hash(body)}
    return {
        "module_present": True,
        "module_hash": content_hash(path.read_text()),
        "symbols": {s: found.get(s, {"present": False}) for s in symbols},
    }


def legacy_adapters() -> dict:
    rows = {
        name: {**spec, **_symbol_digest(spec["module"], spec["symbols"])}
        for name, spec in LEGACY_ADAPTERS.items()
    }
    return part(
        rows,
        knowledge_mode=ATTESTED,
        describes="the named legacy adapters, whose content is defined by this "
                  "export rather than by their names (§6.3)",
    )


def fitted_state() -> dict:
    """What is fitted in-process today, and where a replay must pin it instead.

    §7 of the design records that fold artifacts, residual pools, calibration
    and payoff state are currently built per process. That is the honest state
    of the baseline: this part names each one, where it comes from and what a
    tier-0 fixture has to carry so a replay does not refit.
    """
    return part(
        {
            "fitted_in_process": {
                "tier4_fold_models": {
                    "where": "engine.score.Scorer._serving_models",
                    "keyed_by": "fold timestamp",
                    "pinned_by_fixture_as": "forecast_model + forecast_fold",
                },
                "residual_pool": {
                    "where": "engine.pnl_sim.ResidualPool",
                    "note": "context-dependent: exp_pnl_sim shifts with the "
                            "loaded ticker set (measured +/-0.5pp cohort bias)",
                    "pinned_by_fixture_as": "exp_pnl_sim, win_sim",
                },
                "payoff_maps": {
                    "where": "engine.score.Scorer._payoffs / _runup_payoffs",
                    "pinned_by_fixture_as": "payoff",
                },
                "recalibration": {
                    "where": "engine.score.Scorer._recalibrations",
                    "pinned_by_fixture_as": "win_model vs win_model_raw",
                },
                "analog_matcher": {
                    "where": "engine.analogs.AnalogMatcher",
                    "pinned_by_fixture_as": "analog_buckets, n_analogs, "
                                            "ci_low, ci_high",
                },
            },
            "chooser_analog_pool": score.CHOOSER_ANALOG_POOL,
            "panel_market_block": list(score._PANEL_MARKET_BLOCK),
            "model_draws": score.MODEL_DRAWS,
        },
        knowledge_mode=RECONSTRUCTED,
        describes="fold artifacts, residual pools, calibration and payoff "
                  "state — currently fitted in-process (§6.3)",
    )


# --------------------------------------------------------------------------
# 4. conventions
# --------------------------------------------------------------------------


def conventions() -> dict:
    from engine.data import manifest  # local: it reads the store

    snapshot = manifest.read_snapshot() or {}
    return part(
        {
            "calendar": {
                "source": "engine.calendar.trading_calendar",
                "offset_semantics": "trading days relative to the print: -1 is "
                                    "one session before the last pre-print "
                                    "close, 0 is that close, +1 the first "
                                    "post-print close",
                "sessions": ["BMO", "AMC"],
            },
            "quotes": {
                "fill_default_alpha": 0.5,
                "fill_policy_id": "legacy.fill_alpha.v1",
                "stale_quote_fallback": "quote_max_age_sessions, anchored on "
                                        "the ENTRY date; chain_as_of caps it",
                "min_quoted_implied_move": score.MIN_QUOTED_IMPLIED_MOVE,
            },
            "money": {
                "entry_cost": "positive debit (legacy mapping, documented)",
                "entry_cost_estimate_vs_actual": "SEPARATE fields, never "
                                                 "reconciled into one (§6.4)",
                "expected_return_denominator": "declared per field; no generic "
                                               "expected_pnl switching between "
                                               "dollars and return fraction",
            },
            "precision": {
                "board_display_round_to": 6,
                "replay_inputs_round_to": None,
                "ladder_strike_decimals": score.LADDER_STRIKE_DP,
                "note": "the board rounds to six places for display ONLY. "
                        "b33036c and 6b9d5cf were both a replay input rounded "
                        "on the way out; a corpus captured through the display "
                        "path would freeze that as the baseline.",
            },
            "snapshot": {"snapshot": snapshot.get("snapshot", ""),
                         "as_of": snapshot.get("as_of", "")},
            "store_root": str(paths.ROOT.name),
        },
        knowledge_mode=ATTESTED,
        describes="calendar version, quote and fill conventions, data snapshot "
                  "references and precision rules (§6.4)",
    )


def worked_examples() -> dict:
    """Where the existing examples live, with their provenance.

    Pointers rather than copies: the ledger is append-only evidence and the
    reports are generated from it, so duplicating them into a dated directory
    would create a second copy that can drift from the authoritative one. What
    is frozen here is which file, at which commit, was the example.
    """
    rows = {}
    for name, path in (("ledger", paths.ROOT / "ledger"),
                       ("reports", paths.ROOT / "reports")):
        rows[name] = {
            "path": str(path.relative_to(paths.ROOT)),
            "present": path.exists(),
            "files": sorted(p.name for p in path.glob("*")) if path.exists() else [],
        }
    rows["tier0_corpus"] = {
        "path": "fixtures/tier0",
        "note": "the exact score requests and results behind each example, "
                "written by tools/capture_tier0_corpus.py",
    }
    return part(rows, knowledge_mode=RECONSTRUCTED,
                describes="report, book, prediction and settlement examples "
                          "with their provenance (§6.4)")


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------


PARTS = {
    "environment_lock.json": environment_lock,
    "definitions/structures.json": resolved_structures,
    "definitions/gates.json": resolved_gates,
    "definitions/dyn_sv.json": resolved_dyn_sv,
    "definitions/validation_status.json": validation_status,
    "artifacts/models.json": registered_models,
    "artifacts/legacy_adapters.json": legacy_adapters,
    "artifacts/state.json": fitted_state,
    "conventions/conventions.json": conventions,
    "conventions/worked_examples.json": worked_examples,
}


def build() -> dict[str, str]:
    """Every part, rendered to canonical JSON text. No wall clock anywhere."""
    out: dict[str, str] = {}
    lock = None
    for name, builder in PARTS.items():
        payload = builder()
        if name == "environment_lock.json":
            lock = payload
        out[name] = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    out["requirements.txt"] = requirements_txt(lock)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "parts": {name: content_hash(text) for name, text in sorted(out.items())},
        "package_hash": content_hash(
            {name: content_hash(text) for name, text in sorted(out.items())}
        ),
    }
    out["MANIFEST.json"] = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    return out


def write(out_dir: Path) -> dict[str, str]:
    files = build()
    for name, text in files.items():
        target = out_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    return files


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--version", default=date.today().isoformat(),
                    help="directory name under baseline/ (default: today)")
    ap.add_argument("--out", default=None, help="write here instead")
    ap.add_argument("--verify", default=None,
                    help="build and byte-compare against an existing export")
    args = ap.parse_args(argv)

    if args.verify:
        existing = Path(args.verify)
        built = build()
        bad = [
            name for name, text in built.items()
            if not (existing / name).exists() or (existing / name).read_text() != text
        ]
        missing = sorted(
            str(p.relative_to(existing))
            for p in existing.rglob("*") if p.is_file()
            and str(p.relative_to(existing)) not in built
        )
        if bad or missing:
            print(f"BASELINE DIFFERS — {len(bad)} changed, {len(missing)} extra",
                  file=sys.stderr)
            for name in sorted(bad) + missing:
                print(f"  {name}", file=sys.stderr)
            return 1
        print(f"BASELINE REPRODUCIBLE — {len(built)} files byte-identical")
        return 0

    out_dir = Path(args.out) if args.out else ROOT / "baseline" / args.version
    files = write(out_dir)
    manifest = json.loads(files["MANIFEST.json"])
    print(f"baseline package -> {out_dir}")
    for name in sorted(manifest["parts"]):
        print(f"  {manifest['parts'][name][7:19]}  {name}")
    print(f"package hash: {manifest['package_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
