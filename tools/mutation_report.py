"""Query mutation-testing results: which mutants survive, where, and the trend.

Reads both artifact schemas without fabricating anything: schema 1 rows from
the mutmut backend (``tools/mutation_results.py``) and schema 2 rows from
pytest-gremlins (``tools/gremlin_results.py``), whose ``diff``/``triage``/
``mutmut_status``/``retested_this_run`` are always null, and whose statuses add
``excluded`` (pardoned) while never carrying ``no_tests``/``skipped``.

By default reads the merged ``mutation-report`` artifact of the latest
completed run of ``.github/workflows/mutation.yml`` on main, fetched with
``gh`` into a cache directory outside the repo
(``$MUTATION_REPORT_CACHE``, default ``~/.cache/investing-plan-mutation-report``).
Each run's artifact holds every mutant, not only the ones it re-tested, so the
latest run is a complete picture.

Sources (pick one; default: latest main run)::

    --run ID            that workflow run
    --sha SHA           the latest run for that commit
    --dir PATH          a directory with results.jsonl (an artifact, or
                        ``mutation_results.py export`` output); repeatable
    --local [MODULE..]  offline: build rows from the pilot's own work copies
                        ($MUTATION_PILOT_HOME), without writing to them

Filters (combine freely): --module, --file, --function (exact, or a glob),
--status (comma list of killed, survived, no_tests, timeout, suspicious,
skipped, excluded), --untriaged (survived/no_tests with no current triage
entry),
--changed-since SHA (functions changed between SHA and the run's commit, by
local git).

Output: --format table|jsonl|csv, --diff to show mutation diffs in the table,
--limit N. ``--history N`` prints the score trend over the last N runs
instead of rows. The guide (tests/README.md, "Mutation CI") has examples.
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import gremlin_results as gr  # noqa: E402
import mutation_results as mr  # noqa: E402

WORKFLOW = "mutation.yml"
ARTIFACT = "mutation-report"
CSV_FIELDS = [f for f in mr.ROW_FIELDS if f != "triage"] + ["triage_verdict", "triage_note",
                                                           "triage_stale"]


# -- sources -----------------------------------------------------------------------

def cache_dir() -> Path:
    raw = os.environ.get("MUTATION_REPORT_CACHE")
    base = (Path(raw) if raw else Path.home() / ".cache" / "investing-plan-mutation-report")
    base = base.expanduser().resolve()
    if base == REPO or REPO in base.parents:
        sys.exit(f"MUTATION_REPORT_CACHE must be outside the repo; got {base}")
    return base


def gh_json(args: list[str], repo: str | None) -> list | dict:
    cmd = ["gh", *args] + (["-R", repo] if repo else [])
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"gh failed ({' '.join(cmd[:3])}): {proc.stderr.strip()}")
    return json.loads(proc.stdout)


_RUN_FIELDS = "databaseId,headSha,headBranch,event,createdAt,conclusion,status"


def list_runs(limit: int, repo: str | None, *, sha: str | None = None,
              branch: str | None = "main") -> list[dict]:
    """Completed runs of the mutation workflow, newest first."""
    args = ["run", "list", "--workflow", WORKFLOW, "--status", "completed",
            "--limit", str(limit), "--json", _RUN_FIELDS]
    if sha:
        args += ["--commit", sha]
    elif branch:
        args += ["--branch", branch]
    return gh_json(args, repo)


def fetch_run(run_id: str, repo: str | None) -> Path | None:
    """The run's merged artifact, downloaded once and cached by run id; None
    when the run has none (e.g. its plan job failed, or retention expired)."""
    dest = cache_dir() / str(run_id)
    if (dest / "summary.json").exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["gh", "run", "download", str(run_id), "-n", ARTIFACT, "-D", str(dest)]
    cmd += ["-R", repo] if repo else []
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if proc.returncode != 0 or not (dest / "summary.json").exists():
        print(f"note: no {ARTIFACT} artifact for run {run_id}: {proc.stderr.strip()[:200]}",
              file=sys.stderr)
        return None
    return dest


def local_rows(modules: list[str]) -> list[dict]:
    """Rows built straight from the pilot's work copies (read-only)."""
    import mutation_pilot as pilot

    cfg = pilot.load_config()
    names = modules or [n for n in pilot.enabled_modules(cfg)
                        if (pilot.home() / n / "mutants").is_dir()]
    info = mr.run_info("local")
    triage = mr.load_triage()
    rows: list[dict] = []
    for name in names:
        rows += mr.build_rows(pilot.home() / name, name, pilot.mutate_files(cfg, name), info,
                              triage=triage)
    return rows


def resolve_dirs(args) -> list[Path]:
    if args.dir:
        return args.dir
    if args.run:
        found = fetch_run(args.run, args.repo)
        if found is None:
            sys.exit(f"run {args.run} has no {ARTIFACT} artifact")
        return [found]
    # Newest first; skip a run whose merged artifact is missing.
    for meta in list_runs(5, args.repo, sha=args.sha):
        found = fetch_run(meta["databaseId"], args.repo)
        if found is not None:
            return [found]
    sys.exit("no completed mutation run with a report found"
             + (f" for {args.sha}" if args.sha else " on main"))


# -- filters -----------------------------------------------------------------------

def _match(value: str | None, patterns: list[str]) -> bool:
    return value is not None and any(
        fnmatch.fnmatchcase(value, p) if any(ch in p for ch in "*?[") else value == p
        for p in patterns)


def _split(values: list[str] | None) -> list[str]:
    return [v.strip() for item in values or [] for v in item.split(",") if v.strip()]


def filter_rows(rows: list[dict], *, modules=(), files=(), functions=(), statuses=(),
                untriaged: bool = False, changed: set[tuple[str, str]] | None = None) -> list[dict]:
    known = set(mr.STATUSES) | set(gr.STATUSES)
    bad = [s for s in statuses if s not in known]
    if bad:
        raise ValueError(f"unknown status {bad}; known: {', '.join(sorted(known))}")
    out = []
    for r in rows:
        if modules and r["module"] not in modules:
            continue
        if files and not _match(r["file"], list(files)):
            continue
        if functions and not _match(r["function"], list(functions)):
            continue
        if statuses and r["status"] not in statuses:
            continue
        if untriaged and (r["status"] not in ("survived", "no_tests") or mr.is_triaged(r)):
            continue
        if changed is not None and (r["file"], r["function"]) not in changed:
            continue
        out.append(r)
    return out


def changed_since(rows: list[dict], base: str) -> set[tuple[str, str]]:
    heads = {r["sha"] for r in rows if r.get("sha")}
    if len(heads) != 1:
        sys.exit(f"--changed-since needs rows from one commit; got {len(heads)}")
    head = heads.pop()
    files = sorted({r["file"] for r in rows})
    try:
        return mr.changed_functions(base, head, files)
    except subprocess.CalledProcessError as exc:
        sys.exit(f"git diff {base}..{head[:12]} failed (fetch both commits first): "
                 f"{exc.stderr.strip()}")


# -- history -----------------------------------------------------------------------

def run_identity(summary: dict) -> tuple:
    """The comparison identity of one artifact: (backend, backend_version,
    scoring-policy identity). A schema-1 mutmut summary states none of them, so
    it reads as the historical mutmut identity ``(None, None, None)``; a
    schema-2 gremlins one carries all three. ``policy_identity`` ignores the
    operator set *observed* in a run (data, not policy), so two gremlins runs
    that happened to fire different operators still share an identity -- but a
    different backend, pinned version or scoring policy does not."""
    return (summary.get("backend"), summary.get("backend_version"),
            gr.policy_identity(summary.get("policy")))


def merge_history(runs: list[tuple[dict, dict]], modules=()) -> list[dict]:
    """(run metadata, merged summary.json) pairs -> one row per run and module,
    oldest first, with the score change against the module's previous run.

    mutmut and pytest-gremlins measure different things (operator sets,
    error/pardon semantics, score denominators), so a numeric delta across two
    identities is meaningless -- a mutmut 50% followed by a gremlins 80% is NOT
    a +30pp improvement and must not be shown as one. Each row keeps its
    backend/version/policy identity, and ``delta`` is computed only against the
    module's previous run on the SAME identity; across a mismatch it is ``None``
    (shown as ``--``), exactly as when there is no previous run at all."""
    rows, last = [], {}
    ordered = sorted(runs, key=lambda pair: (pair[0].get("createdAt") or "",
                                            str(pair[0].get("databaseId"))))
    for meta, summary in ordered:
        identity = run_identity(summary)
        blocks = dict(summary.get("modules", {}))
        blocks["ALL"] = summary
        for name, b in blocks.items():
            if modules and name not in modules:
                continue
            prev = last.get(name)
            score = b.get("score")
            comparable = prev is not None and prev[1] == identity
            rows.append({
                "created": meta.get("createdAt"), "run_id": meta.get("databaseId", summary.get("run_id")),
                "sha": summary.get("sha") or meta.get("headSha"), "mode": summary.get("mode"),
                "module": name, "backend": identity[0], "backend_version": identity[1],
                "policy_id": identity[2],
                "total": b.get("total"), "killed": b.get("killed"),
                "survived": b.get("survived"), "no_tests": b.get("no_tests"),
                "skipped": b.get("skipped"), "survived_untriaged": b.get("survived_untriaged"),
                "score": score,
                "delta": None if score is None or not comparable else round(score - prev[0], 4),
            })
            if score is not None:
                last[name] = (score, identity)
    return rows


def history(args, modules) -> list[dict]:
    if args.dir:
        pairs = [({"databaseId": json.loads((d / "summary.json").read_text()).get("run_id"),
                   "createdAt": None}, json.loads((d / "summary.json").read_text()))
                 for d in args.dir]
        # Directories carry no timestamp: keep the order they were given in.
        for i, (meta, _) in enumerate(pairs):
            meta["createdAt"] = f"{i:06d}"
    else:
        pairs = []
        for meta in list_runs(args.history, args.repo):
            found = fetch_run(meta["databaseId"], args.repo)
            if found is not None:
                pairs.append((meta, json.loads((found / "summary.json").read_text())))
    return merge_history(pairs, modules)


# -- output ------------------------------------------------------------------------

def _pct(v) -> str:
    return "--" if v is None else f"{100 * v:.1f}%"


def _pad(v, width: int) -> str:
    """History columns for a gremlins block: no_tests/skipped are not its vocabulary."""
    return f"{'--' if v is None else v:>{width}}"


def emit(rows: list[dict], fmt: str, *, show_diff: bool, history_rows: bool, out=None) -> None:
    out = out or sys.stdout
    if fmt == "jsonl":
        for r in rows:
            out.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")
        return
    if fmt == "csv":
        fields = list(rows[0]) if history_rows and rows else CSV_FIELDS
        if not history_rows and any(r.get("schema_version") == gr.SCHEMA_VERSION for r in rows):
            fields = fields + [f for f in gr.CSV_EXTRA_FIELDS if f not in fields]
        w = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            flat = dict(r)
            if not history_rows:
                t = r.get("triage") or {}
                flat.update(triage_verdict=t.get("verdict"), triage_note=t.get("note"),
                            triage_stale=t.get("stale"))
            w.writerow(flat)
        return
    if history_rows:
        out.write(f"{'created':20s} {'run':>12s} {'sha':10s} {'mode':11s} {'module':22s} "
                  f"{'backend':>12s} "
                  f"{'total':>6s} {'surv':>5s} {'notest':>6s} {'untri':>5s} {'score':>7s} {'delta':>7s}\n")
        for r in rows:
            # A delta across incompatible backend/version/policy is None (below),
            # never a fabricated percentage-point jump; the backend column shows
            # why a run's delta reads as "--" when its identity changed.
            delta = "--" if r["delta"] is None else f"{100 * r['delta']:+.1f}"
            backend = r.get("backend") or "mutmut"
            out.write(f"{str(r['created'] or '')[:19]:20s} {str(r['run_id']):>12s} "
                      f"{str(r['sha'] or '')[:10]:10s} {str(r['mode']):11s} {r['module']:22s} "
                      f"{backend:>12s} "
                      f"{_pad(r['total'], 6)} {_pad(r['survived'], 5)} {_pad(r['no_tests'], 6)} "
                      f"{_pad(r['survived_untriaged'], 5)} {_pct(r['score']):>7s} {delta:>7s}\n")
        return
    for r in rows:
        t = r.get("triage")
        tri = "" if not t else f"  [{t['verdict']}{' STALE' if t['stale'] else ''}]"
        if r.get("backend"):  # schema-2 gremlins row: operator, not a mutmut ordinal
            ident = f"{r.get('operator')} [{r.get('backend_status')}]"
        else:
            ident = "#" + r["mutant_name"].rpartition("__mutmut_")[2]
        out.write(f"{r['status']:10s} {r['module']:18s} {r['file']}:{r['line']}  "
                  f"{r['function']} {ident}{tri}\n")
        if show_diff and r.get("diff"):
            body = [ln for ln in r["diff"].splitlines() if not ln.startswith(("---", "+++"))]
            out.write("".join(f"    {ln}\n" for ln in body))
    out.write(f"-- {len(rows)} mutants\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__.split("\n\n", 1)[1])
    src = p.add_mutually_exclusive_group()
    src.add_argument("--run", help="workflow run id")
    src.add_argument("--sha", help="latest run for this commit")
    src.add_argument("--dir", type=Path, action="append", help="local results directory")
    src.add_argument("--local", nargs="*", metavar="MODULE", default=None,
                     help="offline: rows from the pilot's work copies")
    p.add_argument("--repo", default=None, help="OWNER/REPO for gh (default: this checkout's)")
    p.add_argument("--module", action="append")
    p.add_argument("--file", action="append")
    p.add_argument("--function", action="append")
    p.add_argument("--status", action="append")
    p.add_argument("--untriaged", action="store_true")
    p.add_argument("--changed-since", metavar="SHA")
    p.add_argument("--history", type=int, metavar="N", help="score trend over the last N runs")
    p.add_argument("--format", choices=("table", "jsonl", "csv"), default="table")
    p.add_argument("--diff", action="store_true", help="table: print each mutation diff")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)
    modules = _split(args.module)

    if args.history:
        rows = history(args, modules)
        emit(rows[-args.limit:] if args.limit else rows, args.format, show_diff=False,
             history_rows=True)
        return 0
    if args.local is not None:
        rows = local_rows(args.local)
    else:
        rows = [row for d in resolve_dirs(args) for row in mr.read_jsonl(d / "results.jsonl")]
    changed = changed_since(rows, args.changed_since) if args.changed_since else None
    try:
        rows = filter_rows(rows, modules=modules, files=_split(args.file),
                           functions=_split(args.function), statuses=_split(args.status),
                           untriaged=args.untriaged, changed=changed)
    except ValueError as exc:
        p.error(str(exc))
    emit(rows[: args.limit] if args.limit else rows, args.format, show_diff=args.diff,
         history_rows=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
