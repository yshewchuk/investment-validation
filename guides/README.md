# Build Guides — Earnings-Vol Trading Program

One guide per phase of `../EARNINGS_VOL_PROGRAM_PLAN.md` (v1.1). Each guide
tells the implementing agent: build order, architecture, contracts/schemas,
hard constraints, acceptance tests ("how we know it works"), and known failure
modes. Read the plan first; read this file before any guide.

| Guide | Phase | Depends on |
|---|---|---|
| `phase0_data_foundations.md` | Data tiers, engine core, Sep-1 pulls | — |
| `phase1_scoring_engine.md` | Model registry + score API | Phase 0 |
| `phase2_experiment_framework.md` | Experiment harness + evaluation suite | Phase 0; report format from Phase 4 |
| `phase3_dashboard.md` | Monitoring dashboard + remote snapshot | Phases 1, 4 |
| `phase4_verification_reporting.md` | Report generator, ledger, leak auditor | Phase 0 (build alongside Phase 2) |
| `phase5_forward_test.md` | Paper trading, fill-quality measurement | Phases 1–4 |
| `phase6_thesis_overlay.md` | AI-correction overlay | Phases 2, 3 |

## Environment (verified 2026-08-29)

- Python 3.14, system dist-packages: numpy, pandas, scipy, scikit-learn,
  matplotlib, fastapi, uvicorn, requests, playwright, yfinance, joblib,
  pydantic, lxml, bs4. A `.venv` exists at repo root.
- **Missing:** pyarrow (Parquet), jinja2, pytest. Try `pip install pyarrow`
  first; if the environment can't install, every guide specifies a
  `csv.gz` fallback for storage and string-template fallback for HTML/MD.
  Do not add other new dependencies without need.
- Platform: WSL2. Consequences: the machine sleeps with the host (long jobs
  must be resumable; the published snapshot, not the live server, is the
  always-up surface), and cron must be verified running (`service cron
  status`) before relying on it.
- Credentials in `/root/investing-plan/.env` (source, never echo). Operational
  rules in `/root/investing-plan/AGENTS.md` are **binding**: curl for Polygon,
  Playwright token dance for oquants, throttle playbook, progress logging
  ≥1 line/min on any job >3–4s, check-ins every 5 min on long runs.

## Cross-cutting conventions (every phase)

1. **Real prices only.** ORATS chain bid/ask (validated ±2–3%) and Polygon
   bars are the only P&L sources. oquants model-fitted marks are banned from
   P&L (standing rule, `bt/straddle/VERDICT_2026-08-27.md`).
2. **FillModel everywhere.** No function computes P&L without an explicit
   `FillModel(alpha)`; results are reported at worst(0)/mid(0.5)/best(1) plus
   the breakeven alpha. Hardcoding a fill convention is a bug.
3. **Leak discipline.** Every feature value carries an as-of timestamp;
   `engine/audit.py` (Phase 4) asserts as_of < decision time on every scoring
   and backtest path. Headline numbers are walk-forward out-of-sample only.
4. **Cache-first, quota-guarded.** No network call for data that exists in
   Tier 1. Any script spending >500 ORATS calls requires `--dry-run` output
   first and an explicit `--confirm`. Never run two Polygon processes at once.
5. **Determinism.** Same inputs (snapshot hash + seed) → identical outputs.
   All randomness (bootstrap, MC, NN seeds) is seeded and recorded.
6. **Definition of done** for a phase = the plan's exit criteria + the guide's
   acceptance tests green + a generated report (Phase 4 format) documenting
   the evidence. Acceptance tests live in `checks/phaseN_*.py`, plain scripts
   with asserts, runnable as `python3 checks/phaseN_checks.py` (use pytest
   only if installable). A phase without green checks is not done.
7. **Code style:** plain Python scripts + pandas, matching the existing repo.
   No databases, no message queues, no npm/build toolchains, no Docker. Files,
   cron, and one FastAPI app are the whole runtime.
8. **Don't break the running research.** Never move or rewrite
   `earnings_predictions/` or `bt/` content; the new engine wraps and reads,
   the migration test proves equivalence, and old paths keep working until
   the plan retires them explicitly.
9. **Reports are the interface.** Anything an agent concludes must exist as a
   generated report with a provenance block — chat summaries are not records.
10. **Source control.** All code and these guides live in the PUBLIC GitHub
    repo (allowlist `.gitignore` — everything ignored unless explicitly
    allowed; see the Phase 0 guide §10). Commit and push at every green
    acceptance milestone and at end of session. NEVER commit: `.env` or any
    credential, any data tier or cache, results/reports/figures, research
    verdict docs, `ledger/`, live watchlist configs. Run
    `python3 checks/repo_hygiene.py` before every push (also wired as a
    pre-commit hook). A secret that reaches the public remote is compromised
    no matter how fast it's removed — rotate it immediately. Irreplaceable
    non-code artifacts (ledger, reports, findings, thesis YAML) go to the
    PRIVATE mirror, synced by the nightly job.
