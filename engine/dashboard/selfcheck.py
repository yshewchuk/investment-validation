"""Bundle vs engine: no silent divergence between what the UI shows and the scorer.

The nightly job renders the bundle from ``score_calendar`` output and then
re-scores a sample of board rows DIRECTLY through :func:`engine.score.score`
— a second code path into the same numbers. Two comparisons run per sampled
row, because they catch different failures:

* **Digest.** The stored row digest must equal the digest of a fresh
  :class:`~engine.score.ScoreResult`. This proves the request identity and the
  engine state (snapshot, models, data) behind the bundle are the ones in force
  now. Both sides hash through :func:`engine.dashboard.render.row_digest`, so
  the question asked is "does the engine still produce this score", not "did a
  value travel through pandas on its way into the bundle".
* **Displayed values.** Every ScoreResult field the board shows must equal the
  fresh result at display precision. A bundle hand-edited after rendering
  keeps its digest — only the field diff catches the tamper.

Any mismatch means the bundle and the engine disagree, and the publish is
refused. The report never raises on a mismatch; the caller decides.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from engine.dashboard.render import row_digest

__all__ = ["SelfCheckReport", "selfcheck", "reconstruct_request"]

#: The guide's sample size for the nightly re-score.
DEFAULT_N = 20


@dataclass
class SelfCheckReport:
    ok: bool
    n_checked: int
    n_board_rows: int
    mismatches: list[dict] = field(default_factory=list)
    snapshot_ok: bool = True
    as_of: str | None = None
    seed: int = 0
    elapsed_s: float = 0.0
    detail: str = ""

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "n_checked": self.n_checked,
            "n_board_rows": self.n_board_rows,
            "mismatches": self.mismatches[:10],
            "snapshot_ok": self.snapshot_ok,
            "as_of": self.as_of,
            "seed": self.seed,
            "elapsed_s": round(self.elapsed_s, 2),
            "detail": self.detail,
        }


def reconstruct_request(row: dict):
    """Board row → the exact ``ScoreRequest`` that produced it.

    ``score_calendar`` scores with ``as_of=None`` (the scorer defaults it to
    the entry date) and an optional fixed strike, so those are the only
    ingredients needed to reproduce the call.
    """
    from engine.fills import FillModel
    from engine.score import ScoreRequest

    offset = row.get("strike_offset")
    return ScoreRequest(
        ticker=str(row["ticker"]),
        strategy=str(row["strategy"]),
        as_of=None,
        event_date=pd.Timestamp(row["event_date"]),
        session=row.get("session"),
        # `requested_strike` first: `strike` is what RESOLVED, and a ladder row
        # that failed to resolve has none — reconstructing it from `strike`
        # alone silently turns it back into an ATM request.
        strike=(
            float(row["requested_strike"])
            if row.get("requested_strike") is not None
            else (float(row["strike"]) if offset is not None and row.get("strike") is not None else None)
        ),
        fill=FillModel(float(row.get("fill", 0.5))),
        # Part of the request, so part of what has to be reproduced. A row
        # priced off an older chain re-scores to NO_CHAIN without it, and the
        # digest would flag every forward row as a mismatch.
        quote_max_age_sessions=(
            int(row["quote_max_age_sessions"])
            if row.get("quote_max_age_sessions") is not None else None
        ),
    )


def _sample_rows(rows: Sequence[dict], n: int, seed: int) -> list[dict]:
    if len(rows) <= n:
        return list(rows)
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(rows), size=n, replace=False).tolist())
    return [rows[i] for i in idx]


#: ScoreResult fields the board displays. The digest proves the REQUEST and the
#: engine state match; comparing these values proves the NUMBERS ON SHOW match
#: too — a bundle hand-edited after rendering keeps its digest and still fails.
_COMPARED_FIELDS = (
    "ticker", "strategy", "event_date", "session", "entry_date", "exit_date",
    "strike", "expiry", "dte_entry", "spot", "entry_cost", "extrapolated",
    "flags", "exp_pnl_model", "win_model", "model_p10", "model_p90",
    "exp_pnl_analog", "win_analog", "ci_low", "ci_high", "n_analogs",
    "analog_widened", "gate_score", "gate_threshold", "gate_pass",
    "model_versions", "driver_name", "driver_prediction", "detail",
    "chain_last_obs", "chain_age_days",
)


def _norm(value: Any) -> Any:
    """Normalize both sides to the board's display precision before comparing.

    The renderer rounds floats to six places when it writes the bundle, so the
    fresh engine value must be reduced the same way before equality is a
    meaningful question.
    """
    if value is None:
        return None
    if isinstance(value, float):
        return round(value, 6) if np.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [_norm(v) for v in value]
    if isinstance(value, dict):
        return {k: _norm(v) for k, v in sorted(value.items())}
    return value


def _diff_fields(stored: dict, fresh: dict) -> list[str]:
    diffs = []
    for field in _COMPARED_FIELDS:
        if field not in stored or field not in fresh:
            continue
        if _norm(stored[field]) != _norm(fresh[field]):
            diffs.append(field)
    return diffs


def selfcheck(
    bundle: Path | str,
    *,
    n: int = DEFAULT_N,
    seed: int = 0,
    scorer=None,
) -> SelfCheckReport:
    """Diff ``n`` random board rows against fresh :mod:`engine.score` calls.

    ``scorer`` may be injected (the nightly job reuses the one it scored the
    board with); otherwise a fresh one is built. Returns a report — never
    raises on a mismatch; the caller decides whether the publish stops.
    """
    started = time.time()
    bundle = Path(bundle)
    board_path = bundle / "data" / "board.json"
    meta_path = bundle / "data" / "meta.json"
    if not board_path.exists():
        return SelfCheckReport(
            ok=False, n_checked=0, n_board_rows=0,
            detail=f"no board at {board_path}", elapsed_s=time.time() - started,
        )
    board = json.loads(board_path.read_text())
    rows = board.get("rows", [])
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    board_as_of = board.get("as_of") or meta.get("as_of")

    from engine.score import UNSCORABLE, Scorer, unscorable_result

    if scorer is None:
        scorer = Scorer()

    snapshot_ok = True
    bundle_snapshot = meta.get("snapshot_hash") or ""
    if bundle_snapshot and scorer.snapshot != bundle_snapshot:
        snapshot_ok = False

    mismatches: list[dict] = []
    sample = _sample_rows(rows, n, seed)
    for row in sample:
        digest = row.get("digest")
        if not digest:
            mismatches.append({"row_id": row.get("row_id"), "reason": "missing digest"})
            continue
        try:
            request = reconstruct_request(row)
            result = scorer.score(request)
        except UNSCORABLE as exc:
            # The board carries NO_CHAIN placeholders for events whose chains
            # were never pulled. Re-scoring one raises the same exception the
            # calendar caught, so the honest comparison is against the same
            # placeholder — not a mismatch, and not a silent pass either: the
            # digest below still has to agree.
            result = unscorable_result(
                request, as_of=board_as_of, snapshot=scorer.snapshot, exc=exc
            )
        except Exception as exc:
            mismatches.append(
                {
                    "row_id": row.get("row_id"),
                    "reason": f"re-score raised {type(exc).__name__}: {exc}"[:300],
                }
            )
            continue
        fresh_digest = row_digest(result.as_dict())
        if fresh_digest != digest:
            mismatches.append(
                {
                    "row_id": row.get("row_id"),
                    "reason": "digest mismatch",
                    "stored": digest[:16],
                    "fresh": fresh_digest[:16],
                }
            )
            continue
        field_diffs = _diff_fields(row, result.as_dict())
        if field_diffs:
            mismatches.append(
                {
                    "row_id": row.get("row_id"),
                    "reason": "display values diverge from the engine",
                    "fields": field_diffs[:12],
                }
            )

    ok = not mismatches and snapshot_ok
    detail = (
        f"{len(sample)}/{len(rows)} rows re-scored, 0 mismatches"
        if ok
        else f"{len(mismatches)} mismatch(es) in {len(sample)} re-scored rows"
    )
    if not snapshot_ok:
        detail += (
            f"; snapshot drift: bundle {bundle_snapshot[:12]}… vs engine "
            f"{scorer.snapshot[:12]}…"
        )
    return SelfCheckReport(
        ok=ok,
        n_checked=len(sample),
        n_board_rows=len(rows),
        mismatches=mismatches,
        snapshot_ok=snapshot_ok,
        as_of=board.get("as_of"),
        seed=seed,
        elapsed_s=time.time() - started,
        detail=detail,
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", default="dashboard/earnings")
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    report = selfcheck(args.bundle, n=args.n, seed=args.seed)
    print(json.dumps(report.as_dict(), indent=1, default=str))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
