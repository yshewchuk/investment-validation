# Build Guides — Earnings-Vol Trading Program

**For the current rearchitecture, start with the
[delivery plan](rearchitecture_delivery_plan.md).** The research-program
sequence immediately below is historical context with separate phase numbers;
it is not the rearchitecture assignment queue.

One guide per phase of `../EARNINGS_VOL_PROGRAM_PLAN.md` (v1.3). Each guide
tells the implementing agent: build order, architecture, contracts/schemas,
hard constraints, acceptance tests ("how we know it works"), and known failure
modes. Read the plan first; read this file before any guide.

**Build order is 0 → 1 → 2 → 4 → 3 → 5 → 6** (the plan's sequencing table):
phases are numbered in the order they were specified, not built. Phase 3's
nightly job writes the Phase 4 ledger, so Phase 4 comes first.
**Historical research-program position: phases 0, 1, 2 and 4 complete. Phase 3
(dashboard) was the active phase — it reads `engine/score.py`, the report generator, and the
ledger's `health.json`, all of which now exist.**

| Guide | Phase | Depends on |
|---|---|---|
| `phase0_data_foundations.md` | Data tiers, engine core, Sep-1 pulls | — |
| `phase1_scoring_engine.md` | Model registry + score API | Phase 0 |
| `phase2_experiment_framework.md` | Experiment harness + evaluation suite | Phase 0; report format from Phase 4 |
| `phase3_dashboard.md` | Monitoring dashboard + remote snapshot | Phases 1, 4 |
| `phase4_verification_reporting.md` | Report generator, ledger, leak auditor (the spec) | Phase 0 (build alongside Phase 2) |
| `phase4_implementation.md` | **Phase 4 completion pass — report legibility, ledger, audit receipts (the how; wins where it disagrees with the spec guide)** | Phases 0–2 |
| `phase5_forward_test.md` | Paper trading, fill-quality measurement | Phases 1–4 |
| `phase6_thesis_overlay.md` | AI-correction overlay | Phases 2, 3 |
| `tier4_feature_models.md` | **Tier 4 — a model's forecast as a feature, causally** | Phase 0 (data tiers); table built, scorer wiring pending |

## Architecture upgrade proposals (2026-09-12)

- [System rearchitecture](system_rearchitecture.md): source review, preserved
  strategy inventory, incremental storage, shared scoring/model recipes,
  supervised jobs, lazy UI, live scoring and staged migration. §4.1-4.3 map
  each owner to a package, declare the import direction and its enforcement,
  and set module size/fan-out budgets.
- [Component contracts](component_contracts.md): proposed schemas, interfaces,
  time/version semantics, failure behavior and transactional boundaries for
  review before implementation. §2.5 groups every type by kind and lifecycle;
  §15 specifies the ComparisonReceipt and the tier-0/1/2 latency commitment
  that keeps confirming a fix off the nightly path.
- [Data model diagrams](rearchitecture_data_model.md): entity identities,
  cardinalities, immutable snapshot membership, model/score lineage and ledger
  relationships.
- [Structure generation and PnL simulation](structure_generation_and_simulation.md):
  exhaustive placement contracts, historical/synthetic scenarios, reusable
  valuation and accounting, time/IV what-ifs and compatibility tests.

Rearchitecture implementation plans (separate from the older research-program
phase numbers above). Start with the
[delivery plan and current assessment](rearchitecture_delivery_plan.md), aligned
with [system rearchitecture §12](system_rearchitecture.md#12-migration-sequence-and-rollback).
It owns sequencing; phase guides own tasks, status files own evidence, and
runbooks own tested commands.

- [Phase 0 — Baseline](rearchitecture_phase0_baseline.md): the tier-0 corpus,
  the ComparisonReceipt, the enforced layer map and the empty `engine/v2/`
  skeleton. Writes no production logic and changes no board number.
- [Phase 1 — Operations](rearchitecture_phase1_operations.md): durable jobs,
  automatic resource admission, stage checkpoints, legacy-compatible nightly
  and experiment workflows, decision/publication safety, and a sequenced
  implementation plan with fault-injection acceptance tests.
- [Phase 2 — Data Access](rearchitecture_phase2_data_access.md): immutable
  dataset/snapshot manifests, bounded Arrow reads, exact event/chain access,
  legacy scoring adapters, and atomic rebuild/rollback acceptance gates.
- [Phase 3A — Preview closeout](rearchitecture_phase3_parity_launch.md): finish
  the merged saved-score board/detail, frozen replay, acceptance evidence and
  repeatable update/rollback. Existing Phase 3 gate names cover only this part.
- [Phase 3B — Incremental data](rearchitecture_phase3_incremental_data.md):
  no-op, append/correction/deletion, coverage/finality and complete invalidation.
- [Phase 4 — Native scoring](rearchitecture_phase4_scoring.md): current
  strategies/refusals, shared canonical kernel and financial diagnostics.
- [Phase 5 — Frozen models](rearchitecture_phase5_models.md): inference without
  fitting, current training recipes, fold/residual artifacts and deployment rollback.
- [Phase 6 — Consumer parity](rearchitecture_phase6_consumer_parity.md): native
  nightly, research, ledger/book, all current views and phone/offline access.
- [Phase 7 — Cutover](rearchitecture_phase7_cutover.md): ten qualified sessions,
  controlled official-writer switch and tested rollback.
- [Phase 8 — Post-cutover](rearchitecture_phase8_post_cutover.md): physical
  cleanup/rename, UI enhancements, efficiency/generalization and live shadow.
- [Rearchitecture tech debt](rearchitecture_tech_debt.md): nice-to-haves
  owned by Phase 8C unless a measured prerequisite moves into an earlier phase.

These are design proposals. They describe explicit future changes to the
runtime choices in convention 7 below, while retaining the research and safety
requirements. They do not change current strategy definitions or deployments.

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
