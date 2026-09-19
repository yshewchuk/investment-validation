# Rearchitecture Phase 6 — Consumer parity and operational rehearsal

Status: implementation plan, 2026-09-16. Authority:
[delivery plan](rearchitecture_delivery_plan.md), system design §§8–9, 11–13.
This phase completes existing workflows; it does not require rewriting every
screen in React.

## Outcome and entry

Every current consumer obtains native canonical outputs from the v2 pipeline.
The full current feature inventory works through new screens, compatibility
views, CLI or exports as appropriate to the existing access path. A local
board/detail screenshot is not sufficient.

Start the inventory now. Final integration uses accepted 3B data, Phase 4
scoring and Phase 5 deployments. Preserve the 3A API, immutable release binding,
pagination, authentication and lazy detail; do not build another serving stack.

## Required inventory

For every current capability record its old entrypoint, new entrypoint,
producer, snapshot/release identity, test and disposition. Required entries
include nightly refresh/score/selfcheck/publish/backup; predictions and
settlement; hypothetical/contrarian books and funding; replay/research and
generated reports; board filters/deep links; strategy/refusal details; model
evidence; history/analogs; flags/finality/health/quotas; and phone/offline export.
Use the Phase 0 screen/artifact inventory and actual source, not memory.

An existing interactive action needs an equivalent callable path. A frozen
screenshot cannot replace it. A CLI is suitable only where it preserves the
existing workflow or the user explicitly accepts that access change. New job
buttons, a strike explorer and new research views are post-cutover enhancements.

## Ordered implementation assignments

| Task | Deliverable | Acceptance / negative control |
|---|---|---|
| P6-1 Capability and adapter inventory | Complete mapping above; assign every adapter edge to a consumer/removal phase. | No unowned missing capability. Distinguish dormant historical readers from active official-path dependencies. |
| P6-2 Native nightly and command wiring | Existing supervisor graph consumes incremental snapshots, frozen deployment and native scorer; research/replay/evaluation use the same kernel. | Cold preparation handled by jobs; retry/cancel/resume idempotent; no request-time fitting; original generated report/ledger rules retained. |
| P6-3 Decision and accounting consumers | Prediction import/commit, settlement, portfolio/funding and compatible export against canonical records. | Duplicate/retry/same-session supersession tests; original decisions immutable; missing finality blocks settlement; totals agree with existing accounting. |
| P6-4 Serving and compatibility views | Complete Models/book/flags/operations/history/analog and other inventoried capabilities through existing/new views. | Same release end to end; current native outputs visible; null/refusal/stale semantics retained; old pinned legacy pages alone do not prove current capability. |
| P6-5 Existing access and recovery | Existing phone/offline workflow, private delivery/auth, consistent SQLite + referenced-object backup and restoration. | Offline opens without network; no secret/data in public assets; restored deployment replays a score and reconciles ledger/report. |
| P6-6 Full rehearsal and handoff | Real integrated EOD candidate, full population comparisons, runbook and Phase 7 session evidence collection. | Controlled failures keep last good release; measured runtime/memory with contention; every inventory row has evidence. |

## Compatibility is a presentation choice

Compatibility pages may remain after cutover, but they must consume the current
native release through an explicit maintained adapter. They cannot depend on
an old legacy nightly continuing to score or on mutable legacy forecasts.
Financial arithmetic moves to Phase 4; model evidence to Phase 5; accounting
to the existing semantics extracted into the evaluation/ledger boundary.
Presentation adapters may format values and render charts.

The Phase 3 bridge is temporary input plumbing. Replace its legacy-score input
with a canonical projection while retaining release IDs, provenance and tested
read contracts. Do not write financial formulas into React to fill missing fields.

All official-path dependencies must be inventoried. Compatibility rendering or
file-format adapters may remain if their behavior and read sets are frozen and
tested; their existence is not permission for a second authoritative scorer,
registry or ledger. Full physical adapter deletion is Phase 8A.

## Acceptance and qualified sessions

Create a Phase 6 capability matrix and acceptance entrypoint with subjects:
native nightly integration, research/replay/report parity, prediction/settlement
idempotency, book reconciliation, full screen inventory, browser release
consistency, offline/mobile access and restore faults. Reuse L01–L14 checks
for unchanged routes. A test suite for new components alone does not close
existing-consumer parity.

Measure a full current workload, not a six-ticker preview. Record workload,
provider/queue/compute time, cache state, available RAM, contention and deadline.
Agree the operating deadline from those measurements; do not fabricate a
latency gate from a historical contended run.

Start Phase 7 qualification when the integrated candidate satisfies its entry
requirements, even before writing the Phase 6 final report. Ten distinct
completed-session runs are required; old adapter-only nights are not eligible.
Missing sessions remain unknown and repeated attempts count once.

Handoff: capability matrix with evidence, active adapter/read-set inventory,
native job graph/deployment refs, backup/restore and authority-switch runbook,
measured resource policy, qualified session records and remaining Phase 8 work.
No production switch is performed merely by completing this phase.

## P6-1 inventory results

Status: 2026-09-19, from source at `4352b77`. No data, ledger or model value
was read; only source files, file names and declarations.

**How it is produced.** `tools/phase6_inventory.py` discovers every consumer
entrypoint from source with `ast`/regex (no `engine.*` import): legacy
FastAPI routes (`dashboard/earnings_app.py`), the v2 API
(`engine/v2/serving/api.py`) and operations server
(`engine/v2/serving/operations.py`), the legacy SPA views and controls
(`engine/dashboard/static/`), the React routes (`ui/src/routes.ts`), every
`__main__` CLI under `engine/`, `dashboard/`, `tools/` plus the
`engine.ledger` and `ops` subcommands, the legacy nightly steps, the v2
nightly DAG stages, the `legacy_*` actions and coordinator effects, and the
legacy ledger/report/export/backup writers. Adapter edges come from
`checks/legacy_adapters.json` plus every v2-side module named
`*legacy*`/`*compat*`/`*bridge*`/`*adapter*` or importing legacy `engine.*`.
The judgement (capability, disposition, owner) lives in
`tools/phase6_capabilities.toml`. `checks/phase6_inventory.py` regenerates
the document and fails when a discovered entrypoint has no row, a row lacks a
valid disposition or owner, a claimed v2 entrypoint or test does not exist,
an interactive action is replaced by a CLI without user acceptance, an
adapter edge or adapter module is unowned, or the section below is stale.
Refresh the section with `python3 tools/phase6_inventory.py --write-guide`.

**Reading the matrix.** "compatibility-adapter" means the capability works
through v2 today, but its values still come from the legacy scorer or
renderer through a declared adapter. Most rows are in this state: the v2
nightly is a supervised wrapper around legacy actions, and the compatibility
shell serves legacy-rendered bundles. P6-2..P6-4 replace those inputs with
Phase 4/5 native outputs. They keep the entrypoints. "native" rows already
run v2 code end to end. Dormant rows are historical one-shot readers, not
official-path dependencies. Adapter edges marked inactive in the JSON
document (`--out`) are comparison-only, report-only or one-time bootstrap
readers.

**Owner labels on adapter edges.** Ledger labels for open phases keep their
owner (`phase-4 …` → P4, `phase-5 …` → P5). Labels naming closed phases were
re-owned: `phase-1 ledger authority migration` → P6-3, and
`phase-2 data access` → P3B, per system design §12.1, which puts finality
migration in 3B. The generated edge table shows both labels. Refresh and
calendar-validation edges were re-owned to P3B, which is my judgement call.

**User decisions.** UD-1..UD-4 below are open questions for the user. The
matrix records each one; the decision itself has not been made.

<!-- phase6-inventory:begin (generated by tools/phase6_inventory.py --write-guide; do not edit) -->

Schema `phase6_capability_matrix.v1.0`: 199 discovered entrypoints, 46 capability rows, 14 excluded tooling entrypoints, 87 adapter edges.

Rows per disposition: native 9, compatibility-adapter 29, CLI-with-user-acceptance-needed 0, missing 6, dormant-historical 2.

Rows per owner: 8A 2, P6-2 15, P6-3 10, P6-4 15, P6-5 4.

Adapter edges per owner: 8A 2, P3B 12, P4 36, P5 9, P6-2 4, P6-3 9, P6-4 15.

#### Capability matrix

| Row | Capability | Old entrypoint | New v2 entrypoint | Producer | Snapshot/release identity | Tests | Disposition | Owner |
|---|---|---|---|---|---|---|---|---|
| `nightly-orchestration` | Nightly orchestration: plan, universe, single-run lock, submit and supervise the EOD graph | `cli:engine/dashboard/nightly.py`<br>`nightly:legacy universe` | `cli:ops plan`<br>`cli:ops submit`<br>`cli:ops serve`<br>`py:engine/v2/ops/nightly.py::build_legacy_job_requests`<br>`cli:engine/v2/ops/__main__.py` | engine/v2/ops supervisor (Service) over the shadow nightly DAG; every stage a legacy_* job kind | plan implementation_ref + decision_clock (generation pin); pinned snapshot head with --input-mode snapshot | `tests/test_v2_ops_nightly_completion.py`<br>`tests/test_v2_ops_supervised_legacy.py`<br>`tests/test_dashboard.py` | compatibility-adapter | P6-2 |
| `nightly-refresh` | Nightly data refresh: calendar, chains, prices, history backfill, computed moves | `nightly:legacy refresh`<br>`nightly:legacy backfill`<br>`nightly:legacy moves`<br>`cli:engine/data/pulls/forward_calendar.py`<br>`cli:engine/data/pulls/computed_moves.py` | `nightly:v2 refresh`<br>`nightly:v2 materialize`<br>`cli:ops price-refresh`<br>`cli:ops price-history`<br>`cli:ops price-history capture`<br>`cli:ops snapshot`<br>`cli:ops snapshot plan-import`<br>`cli:ops snapshot submit`<br>`cli:ops snapshot promote`<br>`cli:ops snapshot rollback`<br>`cli:ops capture-inputs` | incremental_refresh job (opt-in --refresh-mode native) and snapshot import/promote; default refresh_mode stays legacy | snapshot manifest id / promoted snapshot head; RefreshPlan document hash | `tests/test_v2_ops_incremental_refresh_wiring.py`<br>`tests/test_v2_ops_snapshot_stages.py`<br>`tests/test_v2_ops_price_history.py`<br>`tests/test_data_price_refresh.py`<br>`tests/test_computed_moves.py` | compatibility-adapter | P6-2 |
| `nightly-features` | Tier-3 panel and Tier-4 forecast rebuild before scoring | `nightly:legacy tiers`<br>`cli:engine/data/rebuild.py`<br>`cli:engine/data/features/tier4.py` | `MISSING`<br>`nightly:v2 features` | legacy engine.data.rebuild / engine.data.features.tier4 only | none: Tier 4 is whatever was last written (no snapshot binding) | `tests/test_tier4.py`<br>`tests/test_panel.py` | missing | P6-2 |
| `nightly-score` | Score the calendar, validate refreshed inputs, and build the explorer strike ladder | `nightly:legacy score`<br>`nightly:legacy validate`<br>`nightly:legacy ladder` | `nightly:v2 score`<br>`action:legacy_score`<br>`action:legacy_score_requests`<br>`py:engine/v2/scoring/application.py::score_batch` | legacy_score action (pinned engine.score.score_calendar) inside a supervised worker; native score_batch not wired into the nightly | score.json artifact bound to job id, implementation_ref and legacy input manifest | `tests/test_v2_ops_supervised_legacy.py`<br>`tests/test_v2_scoring_application.py`<br>`tests/test_score.py` | compatibility-adapter | P6-2 |
| `nightly-selfcheck` | Bundle self-check: re-score sampled board rows directly through the engine before publish | `nightly:legacy selfcheck`<br>`cli:engine/dashboard/selfcheck.py` | `nightly:v2 selfcheck`<br>`action:legacy_selfcheck` | legacy_selfcheck action (engine.dashboard.selfcheck) over the staged bundle | selfcheck report bound to the projection job output | `tests/test_v2_ops_supervised_legacy.py`<br>`tests/test_dashboard.py` | compatibility-adapter | P6-2 |
| `nightly-publish` | Atomic publication of the rendered bundle with secret scan and gates | `nightly:legacy publish`<br>`writer:engine/dashboard/publish.py::publish_bundle`<br>`writer:engine/dashboard/publish.py::LocalPublisher.publish` | `nightly:v2 publication`<br>`effect:publication_effect`<br>`py:engine/v2/ops/publication.py::publish_local`<br>`cli:tools/v2_dashboard_publish.py` | publication coordinator effect: fenced local release plus monotone CURRENT pointer | release_id bound to decision, projection, engineering and security gates | `tests/test_v2_ops_effects_graph.py`<br>`tests/test_v2_dashboard_publish.py`<br>`tests/test_v2_serving_publication_binding.py` | native | P6-2 |
| `nightly-backup` | Nightly backup after decisions commit | `nightly:legacy backup`<br>`cli:tools/private_mirror.py`<br>`writer:tools/private_mirror.py::sync` | `nightly:v2 backup`<br>`effect:backup_effect`<br>`py:engine/v2/ops/backup.py::run_backup` | backup coordinator effect: consistent SQLite backup plus referenced objects, local only | backup receipt bound to the committed decision generation | `tests/test_v2_ops_effects_backup.py`<br>`tests/test_private_mirror.py` | native | P6-5 |
| `nightly-engineering-gate` | Code-budget/engineering gate that withholds publication and counts the failure streak | - | `nightly:v2 engineering`<br>`nightly:v2 engineering_gate`<br>`effect:engineering_gate_effect` | engineering_gate coordinator effect running checks/ over HEAD | engineering receipt bound into the publication gates | `tests/test_v2_ops_engineering.py`<br>`tests/test_v2_ops_engineering_history.py` | native | P6-2 |
| `decisions-predictions` | Freeze predictions before rendering (idempotent row ids, late backfilled nights flagged) | `nightly:legacy ledger`<br>`cli:engine/ledger.py`<br>`cli:engine.ledger snapshot`<br>`writer:engine/ledger.py::snapshot`<br>`writer:engine/ledger.py::write_predictions` | `nightly:v2 decision_commit`<br>`action:legacy_decisions`<br>`py:engine/v2/ops/decision_commit.py::commit_decisions` | decision_commit: catalog decisions table; rows built by legacy build_prediction_rows | decision_id + generation_ref + bound score artifact | `tests/test_v2_ops_authority.py`<br>`tests/test_v2_ops_same_session_replan.py`<br>`tests/test_ledger.py` | compatibility-adapter | P6-3 |
| `decisions-validation` | Validate decisions by replaying them from frozen inputs before commit | - | `nightly:v2 decision_validation`<br>`nightly:v2 decision_replay`<br>`nightly:v2 decision_evidence`<br>`action:legacy_decision_replay`<br>`py:engine/v2/ops/decision_validation.py` | legacy_decision_replay action (legacy scorer in a fresh process) plus decision_evidence | decision evidence bound to score and finality outputs | `tests/test_v2_ops_decision_validation_order.py`<br>`tests/test_v2_ops_replay_scope.py` | compatibility-adapter | P6-3 |
| `decisions-supersede` | Reasoned superseding entry for an immutable bad decision | `writer:engine/ledger.py::supersede` | `py:engine/v2/ledger/decisions.py::insert` | decisions.insert with supersedes + supersede_reason | decision_id chain | `tests/test_v2_ops_authority.py`<br>`tests/test_ledger.py` | native | P6-3 |
| `decisions-settlement` | Settle predictions whose events passed; missing finality blocks settlement | `nightly:legacy settle`<br>`cli:engine.ledger score`<br>`writer:engine/ledger.py::score_outcomes` | `nightly:v2 settlement`<br>`action:legacy_settlement`<br>`py:engine/v2/ops/decision_commit.py::import_settlement_candidates_in_transaction` | legacy_settlement action (engine.ledger.score_outcomes) imported into the catalog | outcome generation_ref + settlement watermark | `tests/test_v2_ops_outcome_session_backfill_wiring.py`<br>`tests/test_v2_ops_critical_review.py`<br>`tests/test_ledger.py` | compatibility-adapter | P6-3 |
| `decisions-ledger-export` | Export canonical decisions/outcomes in the legacy ledger file format for legacy readers | `py:engine/ledger.py::read_predictions`<br>`py:engine/ledger.py::read_outcomes` | `nightly:v2 ledger_export`<br>`nightly:v2 export`<br>`effect:ledger_export_effect`<br>`py:engine/v2/ledger/export.py::export_generation` | ledger_export coordinator effect (purposes legacy_import + shadow) | ledger generation tarball bound to the decision generation | `tests/test_v2_ops_effects_graph.py`<br>`tests/test_v2_ops_effects.py` | compatibility-adapter | P6-3 |
| `decisions-history-import` | Bootstrap a catalog with the historical legacy prediction/outcome ledger | `py:engine/ledger.py::read_predictions` | `cli:ops ledger`<br>`cli:ops ledger import-history`<br>`py:engine/v2/ops/ledger_history_import.py::import_history` | ledger_history_import into decisions/decision_imports | import digest per legacy ledger file | `tests/test_v2_ops_ledger_history_import.py` | compatibility-adapter | P6-3 |
| `decisions-calibration` | Regenerate the ledger calibration report and health.json from settled predictions | `cli:engine.ledger calibrate`<br>`writer:engine/ledger.py::calibrate`<br>`writer:engine/ledger.py::write_health` | `cli:ops ledger calibrate`<br>`py:engine/v2/ledger/calibration.py::calibrate` | engine.v2.ledger.calibration.calibrate over catalog decisions/outcomes, publishing health.json/report as immutable catalog artifacts; reuses engine.ledger.scored_pairs/settlement_summary/_strategy_calibration through engine/v2/ledger/legacy_adapter.py | ledger_calibration_state singleton row (n_scored_at_last_report) plus health/report artifact content hash | `tests/test_ledger.py`<br>`tests/test_v2_ledger_calibration.py` | compatibility-adapter | P6-3 |
| `decisions-status` | Ledger status summary (counts, duplicates, pending settlement) | `cli:engine.ledger status` | `cli:ops ledger status`<br>`py:engine/v2/ledger/status.py::status` | engine.v2.ledger.status.status over catalog decisions/outcomes (engine/v2/ledger/catalog_reader.py), reusing engine.ledger.settlement_summary via engine/v2/ledger/legacy_adapter.py | computed on demand from the decisions table's append-only sequence; no persisted identity | `tests/test_ledger.py`<br>`tests/test_v2_ledger_status.py` | compatibility-adapter | P6-3 |
| `books-view` | Hypothetical book and its contrarian (declined) counterpart, with strategy/recommended filters | `cli:engine/portfolio.py`<br>`view:legacy #/trades/book`<br>`control:legacy bk-f-strategy`<br>`control:legacy bk-f-recommended` | `route:v2-ops GET /book`<br>`route:v2-ops GET /release/*` | legacy render.build_book inside legacy_render over the exported ledger generation | compatibility bundle inside the published release_id | `tests/test_portfolio.py`<br>`tests/test_dashboard.py`<br>`tests/test_v2_ops_serving.py` | compatibility-adapter | P6-3 |
| `books-funding` | Book totals, capital per trade and funding computed against canonical records | `cli:engine/portfolio.py` | `cli:ops ledger book`<br>`py:engine/v2/ledger/portfolio.py::build_book` | engine.v2.ledger.portfolio.build_book/summarize over catalog decisions/outcomes, reusing engine.portfolio.build_book/summarize through engine/v2/ledger/legacy_adapter.py | computed on demand from the decisions table's append-only sequence; no persisted identity | `tests/test_portfolio.py`<br>`tests/test_v2_ledger_portfolio.py` | compatibility-adapter | P6-3 |
| `research-experiments` | Preregistered experiment runs through evaluate(): run log, testing ledger, --no-ledger smoke | `cli-family:experiments/*/run.py`<br>`writer:engine/evaluate.py::evaluate`<br>`writer:engine/evaluate.py::append_run_log` | `cli:ops plan`<br>`py:engine/v2/ops/experiments.py::run_experiment` | supervised experiment job: legacy runner via the declared command adapter (EXP-182 only) | spec_hash + input manifest + runner manifest | `tests/test_experiments.py`<br>`tests/test_v2_ops_runner_onboarding.py`<br>`tests/test_evaluate.py` | compatibility-adapter | P6-2 |
| `research-reports` | Generated REPORT.md with figures, fill sensitivity, sample funnel and accuracy checklist | `writer:engine/report.py::Report.write`<br>`writer:engine/report.py::fig_equity`<br>`writer:engine/report.py::fig_by_year`<br>`writer:engine/report.py::fig_mc_fan`<br>`writer:engine/report.py::fig_mc_fan_paths`<br>`writer:engine/report.py::fig_alpha_curve`<br>`writer:engine/report.py::fig_stress_grid`<br>`writer:engine/report.py::fig_reliability` | `py:engine/v2/ops/experiments.py::run_experiment` | legacy engine.report inside the runner; the v2 job records the report as evidence | report bytes hash in the experiment receipt | `tests/test_report.py`<br>`tests/test_experiments.py` | compatibility-adapter | P6-2 |
| `research-replay` | Replay selected trades on real prices through the same kernel as serving | `py:engine/replay.py::replay` | `MISSING`<br>`py:engine/v2/scoring/application.py::replay` | legacy engine.replay | none | `tests/test_replay.py`<br>`tests/test_v2_scoring_application.py` | missing | P6-2 |
| `research-tools` | Research data tools: real-fill pull and quality, trade table build/prune, signal screen, log diagnostics | `cli:engine/data/pulls/polygon_fills.py`<br>`cli:tools/fill_quality.py`<br>`cli:engine/build_trades.py`<br>`cli:tools/reconcile_trades.py`<br>`cli:tools/signal_screen.py`<br>`cli:tools/log_diagnostics.py` | `MISSING` | legacy Tier-2 store readers/writers (engine.data.store) | none: read the mutable legacy store | `tests/test_polygon_fills.py`<br>`tests/test_fill_quality.py`<br>`tests/test_build_trades.py`<br>`tests/test_log_diagnostics.py` | missing | P6-2 |
| `dormant-sep2026-pull` | One-time Sep-2026 ORATS pull plan and T-2 decision-chain arms | `cli:engine/data/pulls/sep2026_plan.py` | - | legacy pull planner | none | `tests/test_pull_plan.py` | dormant-historical (dormant reader) | 8A |
| `dormant-orats-cache-migration` | One-shot import of old ORATS strike pulls into the Tier-1 cache | `cli:tools/migrate_orats_cache.py` | - | legacy Tier-1 fetch store | none | - | dormant-historical (dormant reader) | 8A |
| `board-list` | Upcoming-prints board: strategy/gate/ticker filters, out-of-domain and disabled toggles, ranking | `route:legacy-app GET /api/board`<br>`route:legacy-app GET /api/meta`<br>`view:legacy #/trades/board`<br>`control:legacy f-strategy`<br>`control:legacy f-gate`<br>`control:legacy f-ticker`<br>`control:legacy f-ood`<br>`control:legacy f-disabled` | `route:v2-api GET /api/v1/events`<br>`route:v2-api GET /api/v1/releases/current`<br>`route:v2-api GET /api/v1/releases/{release_id}`<br>`view:ui #board`<br>`route:v2-ops GET /board`<br>`cli:engine/v2/serving/api.py` | serving projections built by the Phase 3 bridge from the legacy score.json + rendered bundle | release_id (projection binding), cursor bound to release and filters | `tests/test_v2_serving_api.py`<br>`tests/test_v2_serving_projections.py`<br>`tests/test_checks_phase3_api_pagination.py`<br>`tests/test_v2_dashboard_browser.py` | compatibility-adapter | P6-4 |
| `board-deep-links` | Deep links and back/forward that reopen a view (and ticker) in a pinned release | `view:legacy #/trades/explorer`<br>`view:legacy #/models/modelx` | `view:ui #event`<br>`view:ui #score`<br>`route:v2-ops GET /release/current.json`<br>`route:v2-ops GET /` | React hash routes carry release_id; the shell pins one release and forwards legacy hashes into its frame | release_id in the URL / pinned once per shell session | `tests/test_v2_dashboard_browser.py`<br>`tests/test_v2_ops_serving_browser.py` | compatibility-adapter | P6-4 |
| `board-compat-shell` | Serve the dashboard bundle (desk server on 8711) | `route:legacy-app MOUNT /`<br>`cli:dashboard/earnings_app.py` | `route:v2-ops GET /`<br>`route:v2-ops GET /release/current`<br>`route:v2-ops GET /release/*`<br>`route:v2-ops GET /legacy/*`<br>`cli:engine/v2/dashboard/preview.py` | operations server serving immutable compatibility bundles under a health banner | release_id resolved once from CURRENT | `tests/test_v2_ops_serving.py`<br>`tests/test_v2_dashboard_preview.py`<br>`tests/test_checks_phase3_preview.py` | compatibility-adapter | P6-4 |
| `board-refresh-action` | Desk action: re-run the nightly without publishing | `route:legacy-app POST /api/refresh` | `cli:ops plan`<br>`cli:ops submit`<br>`route:v2-ops POST /actions/refresh` | supervised nightly plan/submit | job ids of the submitted plan | `tests/test_v2_ops_nightly_completion.py`<br>`tests/test_v2_ops_cli_refresh_action.py`<br>`tests/test_v2_ops_serving.py` | native | P6-4 |
| `board-adhoc-rescore` | Desk action: ad-hoc re-score of one ticker/strategy at an off-ladder strike or expiry | `route:legacy-app GET /api/score` | `cli:ops rescore`<br>`route:v2-ops POST /actions/whatif`<br>`route:v2-ops GET /actions/whatif/*`<br>`py:engine/v2/scoring/application.py::score_one` | adhoc_rescore supervised job (engine/v2/ops/worker.py::_dispatch_adhoc_rescore) under the v2 no-fit guard | job id from the what-if submission; result route binds to the server's pinned current release_id | `tests/test_v2_ops_cli_rescore.py`<br>`tests/test_v2_ops_cli_whatif.py`<br>`tests/test_v2_ops_worker_adhoc_rescore.py`<br>`tests/test_v2_ops_serving.py` | native | P6-4 |
| `strategy-derivation` | Strategy definitions and per-row derivation (gate terms, thresholds, worked example) | `view:legacy #/models/derivation`<br>`control:legacy d-strategy`<br>`control:legacy d-row` | `route:v2-ops GET /derivation`<br>`py:engine/v2/registry/strategies.py::default_registry` | legacy render.build_strategies inside legacy_render; native registry not surfaced | compatibility bundle inside the published release_id | `tests/test_dashboard.py`<br>`tests/test_v2_registry_scoring.py` | compatibility-adapter | P6-4 |
| `strategy-score-detail` | Per-event strategy scores, refusal codes, legs/order ticket and payoff | `route:legacy-app GET /api/ticker/{sym}` | `route:v2-api GET /api/v1/events/{event_id}/scores`<br>`route:v2-api GET /api/v1/scores/{score_id}`<br>`view:ui #event`<br>`view:ui #score` | LegacyScoreBridge rows (Phase 3 bridge) published as immutable objects | release_id + score_id; lazy detail with ETag | `tests/test_v2_serving_api.py`<br>`tests/test_v2_serving_bridge.py`<br>`tests/test_checks_phase3_browser.py` | compatibility-adapter | P6-4 |
| `strategy-explorer` | Explorer: strike grid for a ticker, detail and order ticket | `view:legacy #/trades/explorer`<br>`control:legacy x-ticker` | `route:v2-ops GET /explorer`<br>`route:v2-ops GET /release/*` | legacy ticker payload (strike_ladder rows) inside the compatibility bundle | compatibility bundle inside the published release_id | `tests/test_dashboard.py`<br>`tests/test_v2_ops_serving.py` | compatibility-adapter | P6-4 |
| `models-evidence-stage` | Build the model-evidence payload (per-feature stats, scatter samples) nightly | `nightly:legacy model_evidence`<br>`cli:engine/dashboard/model_evidence.py`<br>`writer:engine/dashboard/model_evidence.py::build_model_evidence` | `nightly:v2 model_evidence`<br>`action:legacy_model_evidence` | legacy_model_evidence action (engine.dashboard.model_evidence) | model_evidence artifact bound to the job; not to a Phase 5 release | `tests/test_dashboard_model_evidence.py`<br>`tests/test_v2_ops_render_parity.py` | compatibility-adapter | P6-2 |
| `models-explorer` | Model explorer: model/feature pickers, inputs, shape and caveats | `view:legacy #/models/modelx`<br>`control:legacy m-model`<br>`control:legacy m-feature` | `route:v2-ops GET /models`<br>`route:v2-ops GET /release/*` | legacy data/models.js inside the compatibility bundle | compatibility bundle inside the published release_id | `tests/test_dashboard_model_evidence.py`<br>`tests/test_v2_ops_serving.py` | compatibility-adapter | P6-4 |
| `models-release-page` | Show the deployed model release (ids, folds, dependencies) matching the scores on the board | - | `route:v2-ops GET /models/release.json`<br>`route:v2-ops GET /models/release`<br>`py:engine/v2/models/deployment.py::current_pointer`<br>`py:engine/v2/models/deployment.py::resolve_release` | Phase 5 deployment pointer (engine/v2/models/deployment.py), read once per request by the ops server | ModelRelease release_id, resolved from the SAME current_pointer() read as its dependencies | `tests/test_v2_ops_serving_model_release.py`<br>`tests/test_v2_models_deployment.py` | native | P6-4 |
| `models-training` | Retrain current model roles and register candidates | `cli:engine/models/training/train_all.py`<br>`cli:tools/build_chooser_pool.py` | `cli:tools/phase5_training_job.py`<br>`py:engine/v2/models/deployment.py::promote` | P5-3 supervised training job over legacy dataset builders; P5-5 staged deployment | training receipt + staged ModelRelease id | `tests/test_training.py`<br>`tests/test_v2_models_deployment.py`<br>`tests/test_v2_models_phase5_datasets.py` | compatibility-adapter | P6-2 |
| `history-analogs` | Historical prints and analog list for a ticker | `route:legacy-app GET /api/ticker/{sym}`<br>`view:legacy #/trades/explorer` | `route:v2-ops GET /explorer`<br>`route:v2-ops GET /release/*` | legacy ticker payload inside the compatibility bundle | compatibility bundle inside the published release_id | `tests/test_dashboard.py`<br>`tests/test_analogs.py` | compatibility-adapter | P6-4 |
| `operations-flags` | Flags: gate triggers, date changes/conflicts, calibration drift, staleness, quota reserve | `nightly:legacy flags`<br>`route:legacy-app GET /api/flags` | `route:v2-ops GET /flags`<br>`route:v2-api GET /api/v1/operations`<br>`py:engine/v2/ops/render_inputs.py::render_flags` | render_inputs.render_flags plus legacy flag helpers (_panel_staleness_flags, _date_conflict_flag) via adapters | flags.json in the release bundle; operations document names its release_id | `tests/test_v2_ops_render_parity.py`<br>`tests/test_v2_serving_api.py` | compatibility-adapter | P6-4 |
| `operations-finality` | Resolve the final session and covered tickers before scoring/settlement | `nightly:legacy finality` | `nightly:v2 finality`<br>`action:legacy_finality`<br>`py:engine/v2/ops/finality.py::resolve_final_session` | native engine/v2/ops/finality.py inside the legacy_finality job kind | finality receipt bound to the materialization root | `tests/test_v2_ops_finality.py`<br>`tests/test_finality.py` | native | P6-2 |
| `operations-health` | Health: model/calibration health view, selfcheck result, job and publication health | `route:legacy-app GET /api/health`<br>`view:legacy #/models/health` | `cli:ops health`<br>`cli:ops doctor`<br>`cli:ops init`<br>`route:v2-ops GET /health.json`<br>`route:v2-api GET /api/v1/operations`<br>`route:v2-ops GET /flags` | ops health document (operations_health.v1.0) and the legacy health view in the bundle | health generated_at + release_id; scope-current, not release-pinned | `tests/test_v2_ops_health.py`<br>`tests/test_v2_ops_serving.py` | compatibility-adapter | P6-4 |
| `operations-quotas` | Provider quota state and reserve | `py:engine/dashboard/render.py::quota_state` | `py:engine/v2/ops/provider_budget.py::reserve`<br>`route:v2-ops GET /release/*` | legacy quota_state adapter inside legacy_render; v2 provider_budget enforces but does not publish | meta.json in the release bundle | `tests/test_v2_ops_provider_admission.py`<br>`tests/test_ledger.py` | compatibility-adapter | P6-4 |
| `operations-jobs` | Job inspection and control: get, logs, explain, cancel, resume, reconcile | `py:engine/dashboard/nightly.py::single_run_lock` | `cli:ops get`<br>`cli:ops logs`<br>`cli:ops explain`<br>`cli:ops cancel`<br>`cli:ops resume`<br>`cli:ops reconcile`<br>`cli:engine/v2/ops/worker.py` | catalog jobs/attempts tables | job_id / attempt_id / fence | `tests/test_v2_ops_explain_from_logs.py`<br>`tests/test_v2_ops_recovery_ownership.py`<br>`tests/test_dashboard.py` | native | P6-2 |
| `export-bundle` | Render the self-contained bundle (.json + .js wrappers, opens from file://) | `nightly:legacy render`<br>`writer:engine/dashboard/render.py::render_bundle` | `nightly:v2 projection`<br>`action:legacy_render`<br>`cli:tools/v2_dashboard_project.py`<br>`cli:tools/v2_dashboard_verified_input.py` | legacy_render action (engine.dashboard.render.render_bundle) then the Phase 3 projection | bundle content hash inside the release | `tests/test_v2_ops_render_parity.py`<br>`tests/test_v2_serving_projections.py`<br>`tests/test_v2_serving_legacy_bundle.py` | compatibility-adapter | P6-4 |
| `export-offline-file` | Offline single-file export of a pinned release (phone without network) | `writer:engine/dashboard/render.py::write_single_file` | `MISSING` | legacy write_single_file over the local bundle | bundle as-of only | `tests/test_dashboard.py` | missing | P6-5 |
| `export-remote-delivery` | Authenticated remote snapshot for phone access (Cloudflare Access), with access probe | `nightly:legacy publish`<br>`doc:dashboard/README.md` | `MISSING`<br>`nightly:v2 delivery` | legacy CommandPublisher + access_probe | remote release stamp | `tests/test_dashboard.py` | missing | P6-5 |
| `export-restore-drill` | Restore a deployment from backup, replay an original score offline, reconcile ledger/report | `doc:RECOVERY.md` | `MISSING`<br>`py:engine/v2/ops/backup.py::restore_backup` | ops/backup.restore_backup (restore only) | backup receipt | `tests/test_v2_ops_effects_backup.py` | missing | P6-5 |

#### MISSING capabilities (the P6-2..P6-5 work)

| Row | Missing capability | Owner | What is missing |
|---|---|---|---|
| `nightly-features` | Tier-3 panel and Tier-4 forecast rebuild before scoring | P6-2 | The bounded GRAPH names a features stage, but no job kind implements it and _DAG_STAGES omits it; the v2 nightly scores against the last legacy Tier-4 write. |
| `research-replay` | Replay selected trades on real prices through the same kernel as serving | P6-2 | v2 application.replay re-scores a frozen decision; research trade replay on real prices still runs engine.replay. |
| `research-tools` | Research data tools: real-fill pull and quality, trade table build/prune, signal screen, log diagnostics | P6-2 | Needs a pinned v2 snapshot read path (or a declared frozen legacy-materialized read set) so research reads the official data. |
| `export-offline-file` | Offline single-file export of a pinned release (phone without network) | P6-5 | No v2 export packages a pinned release with live actions disabled (system design §9). |
| `export-remote-delivery` | Authenticated remote snapshot for phone access (Cloudflare Access), with access probe | P6-5 | publication.py: remote targets need their own conditional-write adapter and are not enabled; the delivery stage has no job kind. |
| `export-restore-drill` | Restore a deployment from backup, replay an original score offline, reconcile ledger/report | P6-5 | restore_backup exists; the drill (restore, offline replay, ledger/report reconcile) does not. |

#### Needs a user decision

- **UD-1** (`board-refresh-action`): Accept `ops plan nightly` + `ops submit` (CLI) in place of the desk's POST /api/refresh button, or require an authenticated operator POST route in P6-4 (spec: a CLI needs explicit user acceptance).
- **UD-2** (`board-adhoc-rescore`): The desk's GET /api/score ad-hoc re-score has no v2 path. Options: (a) a P6-4 supervised what-if job returning a job id, (b) a CLI over engine.v2.scoring.application.score_one with user acceptance, (c) defer to 8B as part of the strike explorer, accepting a capability gap at cutover.
- **UD-3** (`nightly-backup`): The legacy nightly --backup ran git push + private-mirror sync; the v2 backup is local-only by design. Confirm the code/private mirror stays an operator CLI (tools/private_mirror.py) and is not a nightly stage.
- **UD-4** (`research-tools`<br>`research-replay`): Research tools read the mutable legacy Tier-2 store. Should P6-2 move them to pinned v2 snapshot reads before cutover, or may they keep a frozen legacy-materialized read set as a dormant-historical path until 8A?

#### Excluded entrypoints (migration and verification tooling, not consumers)

- `cli:tools/baseline_export.py`<br>`cli:tools/capture_attach_probe.py`<br>`cli:tools/capture_tier0_corpus.py`<br>`cli:tools/replay_tier1.py`: Phase 0/1 baseline and corpus capture/verification tooling; evidence producers, not consumers.
- `cli:tools/mutation_pilot.py`<br>`cli:tools/mutation_report.py`<br>`cli:tools/mutation_results.py`: Mutation-testing CI tooling.
- `cli:tools/oc_check.py`: Agent-delegation check wrapper (gates + bounded tests for an agent worktree); development tooling, not a consumer.
- `cli:tools/phase5_calibration_keys.py`<br>`cli:tools/phase5_inventory.py`<br>`cli:tools/phase5_prepare_release.py`<br>`cli:tools/prepare_phase4_tier4_caches.py`<br>`cli:tools/phase6_inventory.py`: Phase 4/5/6 migration preparation and inventory tooling.
- `cli:tools/bounded_run.py`: Resource-bounded process runner used by every heavy job; infrastructure, not a capability.

#### Active adapter and read-set inventory

| Edge | Kind | Reads | Consumer | Owner (removal) | Ledger label |
|---|---|---|---|---|---|
| `engine.v2.ops.legacy_adapter::engine.score.score_calendar` | exact-symbol | declared legacy input manifest | supervised legacy_* nightly job kinds (engine/v2/ops/nightly.py) | P4 | phase-4 scoring extraction |
| `engine.v2.ops.legacy_adapter::engine.score.Scorer` | exact-symbol | declared legacy input manifest | supervised legacy_* nightly job kinds (engine/v2/ops/nightly.py) | P4 | phase-4 scoring extraction |
| `engine.v2.ops.legacy_adapter::engine.score.ScoreRequest` | exact-symbol | declared legacy input manifest | supervised legacy_* nightly job kinds (engine/v2/ops/nightly.py) | P4 | phase-4 scoring extraction |
| `engine.v2.ops.legacy_adapter::engine.score.FillModel` | exact-symbol | declared legacy input manifest | supervised legacy_* nightly job kinds (engine/v2/ops/nightly.py) | P4 | phase-4 scoring extraction |
| `engine.v2.ops.legacy_adapter::engine.score.UNSCORABLE` | exact-symbol | declared legacy input manifest | supervised legacy_* nightly job kinds (engine/v2/ops/nightly.py) | P4 | phase-4 scoring extraction |
| `engine.v2.ops.legacy_adapter::engine.score.unscorable_result` | exact-symbol | declared legacy input manifest | supervised legacy_* nightly job kinds (engine/v2/ops/nightly.py) | P4 | phase-4 scoring extraction |
| `engine.v2.ops.legacy_adapter::engine.evaluate.evaluate` | exact-symbol | declared spec, scores, trades and price evidence | supervised legacy_* nightly job kinds (engine/v2/ops/nightly.py) | P6-2 | phase-4 evaluation extraction |
| `engine.v2.ops.legacy_adapter::engine.calendar.trading_calendar` | exact-symbol | private calendar and coverage tables | supervised legacy_* nightly job kinds (engine/v2/ops/nightly.py) | P3B | phase-2 data access |
| `engine.v2.ops.legacy_adapter::engine.data.finality.resolve_final_session` | exact-symbol | private daily_market and option_chains | supervised legacy_* nightly job kinds (engine/v2/ops/nightly.py) | P3B | phase-2 data access |
| `engine.v2.ops.legacy_adapter::engine.features.FeatureContext` | exact-symbol | private feature and market inputs | supervised legacy_* nightly job kinds (engine/v2/ops/nightly.py) | P4 | phase-4 feature extraction |
| `engine.v2.ops.legacy_adapter::engine.ledger.build_prediction_rows` | exact-symbol | private score and copied ledger inputs | supervised legacy_* nightly job kinds (engine/v2/ops/nightly.py) | P6-3 | phase-1 ledger authority migration |
| `engine.v2.ops.legacy_adapter::engine.ledger.score_outcomes` | exact-symbol | private score and copied ledger inputs | supervised legacy_* nightly job kinds (engine/v2/ops/nightly.py) | P6-3 | phase-1 ledger authority migration |
| `engine.v2.ops.legacy_adapter::engine.jsonio.json_safe` | exact-symbol | score frame in memory | supervised legacy_* nightly job kinds (engine/v2/ops/nightly.py) | P4 | phase-4 scoring extraction |
| `engine.v2.ops.legacy_adapter::engine.dashboard.nightly.strike_ladder` | exact-symbol | declared private legacy inputs | supervised legacy_* nightly job kinds (engine/v2/ops/nightly.py) | P4 | phase-4 extraction |
| `engine.v2.ops.legacy_adapter::engine.dashboard.nightly.refresh_calendar_data` | exact-symbol | declared private legacy inputs | supervised legacy_* nightly job kinds (engine/v2/ops/nightly.py) | P3B | phase-4 extraction |
| `engine.v2.ops.legacy_adapter::engine.dashboard.nightly.validate_refresh` | exact-symbol | declared private legacy inputs | supervised legacy_* nightly job kinds (engine/v2/ops/nightly.py) | P3B | phase-4 extraction |
| `engine.v2.ops.legacy_adapter::engine.dashboard.model_evidence.build_model_evidence` | exact-symbol | private registry/features | supervised legacy_* nightly job kinds (engine/v2/ops/nightly.py) | P5 | phase-5 model extraction |
| `engine.v2.ops.legacy_adapter::engine.dashboard.render.render_bundle` | exact-symbol | private score, model evidence and ledger-generation artifacts | supervised legacy_* nightly job kinds (engine/v2/ops/nightly.py) | P6-4 | phase-6 serving extraction |
| `engine.v2.ops.legacy_adapter::engine.dashboard.render.build_meta` | exact-symbol | private score artifact | supervised legacy_* nightly job kinds (engine/v2/ops/nightly.py) | P6-4 | phase-6 serving extraction |
| `engine.v2.ops.legacy_adapter::engine.dashboard.render.build_health` | exact-symbol | private ledger-generation artifact | supervised legacy_* nightly job kinds (engine/v2/ops/nightly.py) | P6-4 | phase-6 serving extraction |
| `engine.v2.ops.legacy_adapter::engine.dashboard.render.freshness_summary` | exact-symbol | private staged legacy tree | supervised legacy_* nightly job kinds (engine/v2/ops/nightly.py) | P6-4 | phase-6 serving extraction |
| `engine.v2.ops.legacy_adapter::engine.dashboard.render.quota_state` | exact-symbol | private staged legacy tree | supervised legacy_* nightly job kinds (engine/v2/ops/nightly.py) | P6-4 | phase-6 serving extraction |
| `engine.v2.ops.legacy_adapter::engine.dashboard.render.size_model_mae_from_ledger` | exact-symbol | private ledger-generation artifact | supervised legacy_* nightly job kinds (engine/v2/ops/nightly.py) | P6-4 | phase-6 serving extraction |
| `engine.v2.ops.legacy_adapter::engine.dashboard.selfcheck.selfcheck` | exact-symbol | private serialized bundle | supervised legacy_* nightly job kinds (engine/v2/ops/nightly.py) | P6-4 | phase-6 serving extraction |
| `engine.v2.ops.legacy_adapter::engine.dashboard.selfcheck.DEFAULT_N` | exact-symbol | private serialized bundle | supervised legacy_* nightly job kinds (engine/v2/ops/nightly.py) | P6-4 | phase-6 serving extraction |
| `engine.v2.ops.legacy_adapter::engine.dashboard.selfcheck.scrub_mismatches` | exact-symbol | private serialized bundle | supervised legacy_* nightly job kinds (engine/v2/ops/nightly.py) | P6-4 | phase-6 serving extraction |
| `engine.v2.ops.legacy_adapter::command:experiments/EXP-182_d_1_gated_execution_parity_registered/run.py` | exact-symbol | private clone of EXP-182 wrapper, EXP-181 runner and its resolved source/data dependencies | supervised legacy_* nightly job kinds (engine/v2/ops/nightly.py) | P5 | phase-5 experiment orchestration extraction |
| `engine.v2.data.legacy_adapter::engine.data.schemas.SCHEMAS` | exact-symbol | in-process module constants only; no filesystem or network access | snapshot import and legacy materialization (engine/v2/data) | P3B | phase-3 ingestion |
| `engine.v2.data.legacy_adapter::engine.data.features.panel.PANEL_COLUMNS` | exact-symbol | in-process module constant only; no filesystem or network access | snapshot import and legacy materialization (engine/v2/data) | P3B | phase-3 ingestion |
| `engine.v2.data.legacy_adapter::engine.data.features.tier4.COLUMNS` | exact-symbol | in-process module constant only; no filesystem or network access | snapshot import and legacy materialization (engine/v2/data) | P3B | phase-3 ingestion |
| `engine.v2.data.legacy_adapter::engine.data.features.tier4.KEY_COLUMNS` | exact-symbol | in-process module constant only; no filesystem or network access | snapshot import and legacy materialization (engine/v2/data) | P3B | phase-3 ingestion |
| `engine.v2.data.legacy_adapter::engine.data.store._read_part` | exact-symbol | one materialized Parquet file at a time, from the private attempt-local dest_root only | snapshot import and legacy materialization (engine/v2/data) | P4 | phase-4 scoring extraction |
| `engine.v2.data.legacy_adapter::engine.data.schemas.coerce` | exact-symbol | in-memory frame already read from the private dest_root; no filesystem or network access of its own | snapshot import and legacy materialization (engine/v2/data) | P4 | phase-4 scoring extraction |
| `engine.v2.ops.legacy_adapter::engine.data.finality.covered_tickers` | exact-symbol | private daily_market and option_chains | supervised legacy_* nightly job kinds (engine/v2/ops/nightly.py) | P3B | phase-2 data access |
| `engine.v2.ops.legacy_adapter::engine.data.finality.session_finality` | exact-symbol | the run's own verified materialization root's daily_market/option_chains curated parquet, read directly (never through engine.data.store, which stays rooted at the barrier legacy tree for this same worker process) | supervised legacy_* nightly job kinds (engine/v2/ops/nightly.py) | P3B | phase-2 data access |
| `engine.v2.ops.legacy_adapter::engine.data.finality._market_wide_complete` | exact-symbol | none directly; held only as the comparison baseline for finality_compatibility's monkeypatch detection | supervised legacy_* nightly job kinds (engine/v2/ops/nightly.py) | P3B | phase-2 data access |
| `engine.v2.ops.legacy_adapter::engine.data.finality._coverage_frame` | exact-symbol | none directly; held only as the comparison baseline for finality_compatibility's monkeypatch detection | supervised legacy_* nightly job kinds (engine/v2/ops/nightly.py) | P3B | phase-2 data access |
| `engine.v2.data.legacy_adapter::engine.data.schemas.SOURCE_PRIORITY` | exact-symbol | none (module-level constant) | snapshot import and legacy materialization (engine/v2/data) | P4 | phase-4 scoring extraction |
| `engine.v2.data.legacy_adapter::engine.paths.ROOT` | exact-symbol | in-process module constant only; no filesystem or network access | snapshot import and legacy materialization (engine/v2/data) | P4 | phase-4 scoring extraction |
| `engine.v2.data.legacy_adapter::engine.paths.GSPC_DAILY` | exact-symbol | in-process module constant only; no filesystem or network access | snapshot import and legacy materialization (engine/v2/data) | P4 | phase-4 scoring extraction |
| `engine.v2.data.legacy_adapter::engine.paths.SNAPSHOT_FILE` | exact-symbol | in-process module constant only; no filesystem or network access | snapshot import and legacy materialization (engine/v2/data) | P4 | phase-4 scoring extraction |
| `engine.v2.data.legacy_adapter::engine.paths.FEATURES` | exact-symbol | in-process module constant only; no filesystem or network access | snapshot import and legacy materialization (engine/v2/data) | P4 | phase-4 scoring extraction |
| `engine.v2.data.legacy_adapter::engine.paths.DATA` | exact-symbol | in-process module constant only; no filesystem or network access | snapshot import and legacy materialization (engine/v2/data) | P4 | phase-4 scoring extraction |
| `engine.v2.data.legacy_adapter::engine.models.registry.REGISTRY_PATH` | exact-symbol | in-process module constant only; no filesystem or network access | snapshot import and legacy materialization (engine/v2/data) | P4 | phase-4 scoring extraction |
| `engine.v2.data.legacy_adapter::engine.models.registry.ARTIFACT_DIR` | exact-symbol | in-process module constant only; no filesystem or network access | snapshot import and legacy materialization (engine/v2/data) | P4 | phase-4 scoring extraction |
| `engine.v2.data.legacy_adapter::engine.data.features.tier4.SERVING_DIR` | exact-symbol | in-process module constant only; no filesystem or network access | snapshot import and legacy materialization (engine/v2/data) | P4 | phase-4 scoring extraction |
| `engine.v2.data.legacy_adapter::engine.data.features.tier4.serving_fold` | exact-symbol | none; pure date arithmetic | snapshot import and legacy materialization (engine/v2/data) | P4 | phase-4 scoring extraction |
| `engine.v2.data.legacy_adapter::engine.data.features.tier4.read_serving_header` | exact-symbol | one pinned joblib file under data/models/tier4, size-capped and hash-verified before it is opened | snapshot import and legacy materialization (engine/v2/data) | P4 | phase-4 scoring extraction |
| `engine.v2.ops.legacy_adapter::engine.dashboard.nightly._panel_staleness_flags` | exact-symbol | declared legacy input manifest | supervised legacy_* nightly job kinds (engine/v2/ops/nightly.py) | P4 | phase-4 scoring extraction |
| `engine.v2.ops.legacy_adapter::engine.dashboard.nightly._date_conflict_flag` | exact-symbol | declared legacy input manifest | supervised legacy_* nightly job kinds (engine/v2/ops/nightly.py) | P4 | phase-4 scoring extraction |
| `engine.v2.ops.legacy_adapter::engine.data.store.read_table` | exact-symbol | declared legacy input manifest | supervised legacy_* nightly job kinds (engine/v2/ops/nightly.py) | P4 | phase-4 scoring extraction |
| `engine.v2.ops.legacy_adapter::engine.dashboard.model_evidence.load_model_evidence` | exact-symbol | declared legacy input manifest | supervised legacy_* nightly job kinds (engine/v2/ops/nightly.py) | P4 | phase-4 scoring extraction |
| `engine.v2.ops.legacy_adapter::engine.data.fetch.iter_cached` | exact-symbol | declared legacy input manifest | supervised legacy_* nightly job kinds (engine/v2/ops/nightly.py) | P4 | phase-4 scoring extraction |
| `engine.v2.ops.legacy_adapter::engine.calendar.projected_trading_days` | exact-symbol | none -- pure date rule, no filesystem read | supervised legacy_* nightly job kinds (engine/v2/ops/nightly.py) | P4 | phase-4 scoring extraction |
| `engine.v2.ops.legacy_adapter::engine.data.pulls.price_refresh.load_events` | exact-symbol | Tier-2 earnings_events (curated) | supervised legacy_* nightly job kinds (engine/v2/ops/nightly.py) | P4 | phase-4 data extraction |
| `engine.v2.ops.legacy_adapter::engine.data.pulls.price_refresh.load_price_universe` | exact-symbol | engine.paths.RAW_YF (read-only) + Tier-1 fetch store | supervised legacy_* nightly job kinds (engine/v2/ops/nightly.py) | P4 | phase-4 data extraction |
| `engine.v2.ops.legacy_adapter::engine.data.pulls.price_refresh.load_fetch_history` | exact-symbol | Tier-1 fetch store (yfinance history metadata) | supervised legacy_* nightly job kinds (engine/v2/ops/nightly.py) | P4 | phase-4 data extraction |
| `engine.v2.ops.legacy_adapter::engine.data.pulls.price_refresh.plan_refresh` | exact-symbol | none (pure function; inputs already loaded) | supervised legacy_* nightly job kinds (engine/v2/ops/nightly.py) | P4 | phase-4 data extraction |
| `engine.v2.ops.legacy_adapter::engine.data.pulls.price_refresh.run_refresh` | exact-symbol | none beyond the injected plan | supervised legacy_* nightly job kinds (engine/v2/ops/nightly.py) | P4 | phase-4 data extraction |
| `engine.v2.ops.legacy_adapter::engine.data.fetch.Fetcher` | exact-symbol | none beyond what run_refresh/Fetcher.fetch requests | supervised legacy_* nightly job kinds (engine/v2/ops/nightly.py) | P4 | phase-4 data extraction |
| `engine.v2.ops.legacy_adapter::engine.ledger.SCHEMA_VERSION` | exact-symbol | a module-level constant only; no legacy I/O | supervised legacy_* nightly job kinds (engine/v2/ops/nightly.py) | P4 | phase-4 scoring extraction |
| `engine.v2.scoring.compatibility::engine.score.Scorer` | exact-symbol | explicit request fields and pinned scorer inputs | Phase 4 compatibility scoring seam | P5 | phase-5 scoring extraction |
| `engine.v2.scoring.compatibility::engine.score.ScoreRequest` | exact-symbol | explicit request fields | Phase 4 compatibility scoring seam | P5 | phase-5 scoring extraction |
| `engine.v2.scoring.compatibility::engine.fills.FillModel` | exact-symbol | explicit fill model alpha | Phase 4 compatibility scoring seam | P5 | phase-5 scoring extraction |
| `engine.v2.models.inventory::engine.models.registry` | exact-symbol | engine/models/registry.json and the champion joblib files it points at, one at a time | P5-1 model inventory report (tools/phase5_inventory.py) | P5 | phase-5 native training/serving extraction |
| `engine.v2.models.inventory::engine.paths` | exact-symbol | in-process path constants only; no filesystem access beyond what callers already declare | P5-1 model inventory report (tools/phase5_inventory.py) | P5 | phase-5 native training/serving extraction |
| `engine.v2.models.training.legacy_adapter::engine.models.no_fit.forbid_fitting` | exact-symbol | one thread-local flag | no-fit guard shared with the legacy scorer | P5 | phase-5 native training/serving extraction |
| `engine.v2.ledger.legacy_adapter::engine.ledger.scored_pairs` | exact-symbol | catalog decisions/outcomes rows (engine.v2.ledger.catalog_reader), no legacy file read | ops ledger status\|calibrate\|book (engine/v2/ledger/status.py, calibration.py, portfolio.py) | P6-3 | phase-6 native ledger/portfolio extraction |
| `engine.v2.ledger.legacy_adapter::engine.ledger.settlement_summary` | exact-symbol | catalog decisions/outcomes rows (engine.v2.ledger.catalog_reader), no legacy file read | ops ledger status\|calibrate\|book (engine/v2/ledger/status.py, calibration.py, portfolio.py) | P6-3 | phase-6 native ledger/portfolio extraction |
| `engine.v2.ledger.legacy_adapter::engine.ledger._strategy_calibration` | exact-symbol | catalog decisions/outcomes rows (engine.v2.ledger.catalog_reader), no legacy file read | ops ledger status\|calibrate\|book (engine/v2/ledger/status.py, calibration.py, portfolio.py) | P6-3 | phase-6 native ledger/portfolio extraction |
| `engine.v2.ledger.legacy_adapter::engine.portfolio.build_book` | exact-symbol | catalog decisions/outcomes rows (engine.v2.ledger.catalog_reader), no legacy file read | ops ledger status\|calibrate\|book (engine/v2/ledger/status.py, calibration.py, portfolio.py) | P6-3 | phase-6 native ledger/portfolio extraction |
| `engine.v2.ledger.legacy_adapter::engine.portfolio.summarize` | exact-symbol | catalog decisions/outcomes rows (engine.v2.ledger.catalog_reader), no legacy file read | ops ledger status\|calibrate\|book (engine/v2/ledger/status.py, calibration.py, portfolio.py) | P6-3 | phase-6 native ledger/portfolio extraction |
| `engine/v2/serving/bridge.py` | phase3-bridge | verified legacy score.json and rendered bundle rows (via legacy_bundle), Phase 2 event refs | serving projections -> v2 API and React board/detail | P6-4 | - |
| `engine/v2/serving/legacy_bundle.py` | file-format | bundle data/board.json\|.js, data/meta.json\|.js, data/tickers/{T}.json\|.js and the bundle manifest | bridge; tools/v2_dashboard_project.py | P6-4 | - |
| `engine/v2/serving/operations.py` | compatibility-view | published releases/<id>/ legacy bundle bytes and the health sidecar | operations shell (board/explorer/book/models/derivation/flags views) | P6-4 | - |
| `tools/v2_dashboard_project.py` | file-format | saved score.json, rendered bundle dir, Phase 2 catalog and store | projection publication (P3-1b/P3-4) | P6-4 | - |
| `tools/v2_dashboard_verified_input.py` | file-format | a delivered Phase 2 release (bundle and score artifacts) | projection publication input | P6-4 | - |
| `engine/v2/ops/render_inputs.py` | file-format | ledger-generation tarball, model evidence artifact, prior selfcheck report | legacy_render action | P6-4 | - |
| `engine/v2/ops/legacy_actions.py` | legacy-runtime | allowlisted legacy action names only | worker dispatch of legacy_* job kinds | P6-2 | - |
| `engine/v2/data/legacy_materialization.py` | file-format | pinned snapshot tables; writes the legacy tree layout the legacy scorer reads | materialize stage before legacy_* actions | P6-2 | - |
| `engine/v2/data/legacy_nightly_read_plan.py` | file-format | declares data/raw/fetch/**, daily_market and option_chains for barrier-only legacy stages | legacy_finality / legacy_score read sets | P6-2 | - |
| `engine/v2/data/legacy_mapping.py` | file-format | legacy table schemas (through data/legacy_adapter) and legacy_annotations.json | snapshot import table contracts | 8A | - |
| `engine/v2/models/adapters.py` | file-format | verified frozen legacy-format model artifacts (joblib members) | Phase 5 frozen inference (loader) | 8A | - |
| `engine/v2/ledger/export.py` | file-format | catalog decisions/outcomes; writes legacy ledger predictions/outcomes files | legacy book/render readers via ledger_export | P6-3 | - |
| `engine/v2/ops/ledger_history_import.py` | file-format | legacy ledger predictions/*.jsonl and outcomes files | ops ledger import-history (catalog bootstrap) | P6-3 | - |
| `engine/models/training/chooser.py` | legacy-reader | experiments/EXP-169_menu7prime_confirmation/run.py loaded by filesystem path | legacy chooser training/serving (DYN-SV) | P4 | - |
| `tools/phase5_training_job.py` | legacy-runtime | legacy dataset builders over the Tier-2/3/4 store | P5-3 supervised training job | P5 | - |

<!-- phase6-inventory:end -->
