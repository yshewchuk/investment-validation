# Rearchitecture Phase 3A — Operator Runbook

The operational companion to [Phase 3 — parity and launch](rearchitecture_phase3_parity_launch.md)
§10 ("During implementation add `guides/rearchitecture_phase3_runbook.md`
with exact, tested commands"). The command paths were first tested
synthetically on 2026-09-14. The current real source is Phase 2 release
`reld9a81e6ded7f45c13b5afe21`; its saved-score source population is 111.
Fresh source-bound D15/D19 receipts are pending rebuild. Do not use an earlier candidate, subset, or
receipt as proof for a new serving generation.

Scope is the saved-score preview only; see the
[delivery plan](rearchitecture_delivery_plan.md) and
[closeout queue](rearchitecture_phase3_parity_launch.md#11-resume-here-bounded-closeout-queue).
The per-step UNVERIFIED notes below describe the original command authoring
pass, not an assertion that no later real receipts exist. The
[status file](rearchitecture_phase3_status.md) records those later receipts.
P3A-C4 has exercised the real fenced A → B → A sequence. The final gate
package remains the completion authority for this runbook.

This draft does not claim the launch is complete. No assembled final acceptance
has been run. The bare Phase 3 gate is expected to be red without evidence: see
`checks/phase3_acceptance.json` for the L01-L14 matrix it checks and
`tests/test_checks_phase3_gate.py::test_real_repo_bare_gate_is_red_with_missing_evidence`
for the standing proof.

## Current P3A verification sequence (pending real rebuild)

The historical fenced drills and synthetic receipts document implementation
coverage only; they are not acceptance proof. A real rebuild under
`/tmp/phase3a-supervised-proof` must create fresh verified inputs, A/B
projection bindings, and the final private gate package before this status
can turn green. Do not substitute historical receipt ids or hand-written logs.

```bash
/usr/bin/python3 tools/v2_dashboard_verified_input.py --catalog OPS/catalog.sqlite \
  --store-root OPS --release-id RELEASE --score-artifact-id SCORE --bundle-artifact-id BUNDLE \
  --bundle-dir BUNDLE_DIR --snapshot-artifact-id SNAPSHOT --materialization-request-artifact-id REQUEST \
  --finality-artifact-id FINALITY --model-evidence-artifact-id EVIDENCE \
  --score-comparison-receipt-ref D15_REF --render-comparison-receipt-ref D19_REF \
  --score-comparison-receipt D15_PATH --render-comparison-receipt D19_PATH \
  --render-job-id RENDER_JOB --render-artifact-id BUNDLE --output preview_input.json
/usr/bin/python3 tools/v2_dashboard_project.py --preview-input preview_input.json --source-provenance source_provenance.json \
  --score-json SCORE_JSON --bundle-dir BUNDLE_DIR --bundle-format legacy --snapshot-id SNAPSHOT --catalog OPS/catalog.sqlite \
  --store-root OPS --serving-root SERVING_ROOT --requested-as-of SESSION --resolved-as-of SESSION
/usr/bin/python3 tools/v2_dashboard_publish.py --verify-sequence --root OPS --store-root SECRET_ROOT \
  --source-publication-job A_SOURCE --projection-binding A_BINDING --b-source-publication-job B_SOURCE \
  --b-projection-binding B_BINDING --operation-id proof --sequence-log publication-log.json
/usr/bin/python3 checks/rearchitecture_phase3_publish.py --publication-catalog OPS/catalog.sqlite \
  --publication-store-root OPS <other-required-checker-inputs>
```

## 0. Implemented commands (original verification: 2026-09-14)

- `python3 -m engine.v2.dashboard.preview` — compatibility preview launcher.
  **Exists, command verified to start/refuse correctly** (its own test suite,
  `tests/test_v2_dashboard_preview.py`); never run against a real accepted
  release by this task.
- `python3 -m engine.v2.serving.api` — the read API (P3-2). **Exists,
  verified against a synthetic `serving.sqlite`** (`tests/test_v2_serving_api.py`);
  not run against a real projected release by this task.
- `tools/v2_dashboard_project.py --bundle-format legacy` — the offline
  projection coordinator, including the REAL legacy render-bundle adapter
  (P3-4, merged). **Exists, verified against a small real-shaped bundle
  rendered through the actual `engine.dashboard.render.render_bundle`**
  (`tests/test_v2_serving_legacy_bundle.py`); not yet run against the full
  board of a real accepted Phase 2 release.
- `python3 -m engine.v2.ops` (`plan`, `submit`, `snapshot plan-import|submit|
  promote|rollback`, `get`, `logs`, `cancel`, `resume`, `explain`, `serve`,
  `init`, `doctor`, `health`, `reconcile`) — the general nightly/operations
  CLI. **Exists** (Phase 1/2 work); this guide does not touch `engine/v2/ops`.
- **Publishing a v2 serving release through the fenced ops workflow EXISTS
  (P3-1c, merged at `3037387`).** `engine.v2.ops.effects_graph.
  publication_effect` accepts an optional named `"projection_binding.json"`
  input binding alongside the existing `bundle.tar`/`finality.json`/
  `selfcheck.json`/`engineering_gate.json` ones; when bound, it is copied
  into the published release's own files, `engine.v2.serving.api`'s current-
  resolver reads it, and `engine.v2.serving.projections.
  verify_projection_binding` refuses a tampered or stale one. Proven end to
  end (real synthetic serving candidates, a real synthetic ops catalog, a
  real `engine.v2.serving.api` app over real HTTP) by
  `tests/test_v2_serving_publication_binding.py`. See step 3 for what an
  operator actually runs — there is still no single CLI flag for it, only
  the `engine.v2.ops` job-graph primitives the test exercises.

## 1. Select accepted Phase 2 inputs; inspect release/snapshot/as-of, populations, refusals, evidence

**Status: commands exist and are read-only; UNVERIFIED against a real
accepted Phase 2 release (P3-4).**

Inspect the Phase 2 catalog with a SQLite read-only connection (do not use
`open_catalog`, which can initialize/migrate a catalog):

```bash
/usr/bin/python3 -c "
import sqlite3
conn = sqlite3.connect(\"file:/absolute/path/to/catalog.sqlite?mode=ro\", uri=True)
print(conn.execute(\"SELECT scope, snapshot_id FROM data_snapshot_heads\").fetchall())
"
```

Inspect a specific job's populations/refusals through the ops CLI (never
writes; `get`/`explain`/`logs` are read commands):

```bash
/usr/bin/python3 -m engine.v2.ops get <job_id> --json
/usr/bin/python3 -m engine.v2.ops explain <job_id> --json
```

Build (never fake) the Phase 2 evidence manifest for the release under
consideration, reusing the Phase 2 builder (this guide's own prerequisite,
per §10 — "validate Phase 2 evidence against its producer commit"):

```bash
/usr/bin/python3 checks/rearchitecture_phase2_evidence_build.py \
    --root <phase2-root> --scope <scope> --artifact-root /private/phase2-artifacts \
    --score-receipt <score_receipt.json> --render-receipt <render_receipt.json> \
    --corpus-receipt <corpus_receipt.json> --rollback-receipt <rollback_receipt.json> \
    --fault-matrix <fault_matrix.json> --output /private/phase2-evidence.json
```

## 2. Prepare a candidate projection (no fetch, no rescore, no auto-publish)

**Status: command exists and is verified synthetically; UNVERIFIED on a real
release (P3-4).**

```bash
/usr/bin/python3 tools/v2_dashboard_project.py \
    --preview-input <preview_input.json> --score-json <score.json> \
    --bundle-dir <rendered-bundle-root> --bundle-format legacy \
    --snapshot-id <snapshot_id> --catalog <phase2-catalog.sqlite> \
    --store-root <phase2-store-root> --serving-root <serving-root> \
    --requested-as-of <YYYY-MM-DD> --resolved-as-of <YYYY-MM-DD>
```

Prints `{"release_id": ..., "findings": {...}}` (or a refusal `Problem`) as
one JSON document. This step never mixes fresh quotes with stale forecasts
and never fetches or rescoring — every input above is already on disk, per
guide §10 item 2 ("This command must not fetch, rescore, or publish
automatically").

## 3. Publish through the existing fenced workflow (shadow target)

**Status: EXISTS (P3-1c, merged at `3037387`), verified synthetically by
`tests/test_v2_serving_publication_binding.py`; UNVERIFIED against a real
accepted release by this task.** The fenced publish mechanism itself is real
and proven end to end over real HTTP. What is still missing is a convenience
CLI flag — today an operator (or the coordinator script that will replace
this paragraph once it exists) drives the job-graph primitives directly:

```bash
/usr/bin/python3 -c "
from engine.v2.serving import projections
from engine.v2.foundation import SystemClock
import sqlite3, json
serving_conn = sqlite3.connect('<serving-root>/serving.sqlite')
serving_conn.row_factory = sqlite3.Row
binding = projections.projection_binding(serving_conn, '<release_id>')
print(json.dumps(binding))
" > /private/projection_binding.json
```

Register those bytes as an ops artifact and bind them into the publication
job's `input_bindings` under the name `"projection_binding.json"` — the same
mechanism `bundle.tar`/`finality.json`/`selfcheck.json`/`engineering_gate.json`
already use (`engine.v2.ops.input_bindings.resolve_and_record`,
`engine.v2.ops.effects_graph.publication_effect`; see
`tests/test_v2_serving_publication_binding.py::
test_publication_effect_binds_the_operator_supplied_projection_and_publishes`
for the exact call sequence — a parent job producing a named
`"projection_binding"` output, then a `publication` job whose
`input_bindings["projection_binding.json"]` points at it):

```bash
/usr/bin/python3 -m engine.v2.ops plan nightly --as-of <YYYY-MM-DD> --mode shadow \
    --input-mode snapshot --snapshot-scope <scope>
/usr/bin/python3 -m engine.v2.ops submit --plan <plan.json> --idempotency-key <key>
```

Once published, the API's current-resolver reads `CURRENT` -> the release's
own `projection_binding.json` -> the committed projection
(`engine.v2.serving.projections.verify_projection_binding`), refusing a
tampered document or one naming an uncommitted/mismatched release. An
ordinary bundle-only publication (no projection candidate bound yet, e.g.
still on the P3-0 compatibility preview) behaves exactly as before —
binding the projection is additive, never required.

## 4. Start the compatibility preview and the read API

**Status: the read API has been exercised through a real fenced publication
pointer. Use the scope-root argument below; omitting it intentionally leaves
current-release resolution unavailable.**

```bash
set -a; source .env; set +a   # never print $V2_DASHBOARD_TOKEN
V2_DASHBOARD_TOKEN=$V2_DASHBOARD_TOKEN /usr/bin/python3 -m engine.v2.dashboard.preview \
    --host 127.0.0.1 --port 8765 --release-root <release-root> --health-path <health.json>
```

```bash
V2_DASHBOARD_TOKEN=$V2_DASHBOARD_TOKEN /usr/bin/python3 -m engine.v2.serving.api \
    --host 127.0.0.1 --port 8766 --serving-db <serving-root>/serving.sqlite \
    --store-root <serving-root>/objects --serving-root <serving-root> \
    --publication-root <ops-root>/releases/<scope>
```

Both refuse a non-loopback `--host` without `--allow-non-loopback`, and both
refuse to start with no `V2_DASHBOARD_TOKEN` set. Neither command's output
ever includes the token (`engine/v2/dashboard/preview.py`,
`engine/v2/serving/api.py`, `tests/test_v2_serving_api.py`).

## 5. Project the next accepted release; publish after validation

**Status: projection and publication have been exercised end to end against
the verified Phase 2 source. A fresh update uses a distinct projection
generation; a rollback republishes the retained target through the fenced
effect and never copies release directories or writes `CURRENT` directly.**

Repeat step 2 against the newly accepted Phase 2 release or the next
validated shadow generation, using `--requested-as-of`/`--resolved-as-of`
from the REAL as-of the generation was produced for — never falsified, per
guide §10 item 5. Publish it per step 3, generating that release's own
`projection_binding.json` (`projections.projection_binding`). Then run the
Phase 3 gate against the assembled evidence before treating the generation
as launched — `--accepted-release` takes the PreviewRelease document paired
with that SAME real binding document (`checks/rearchitecture_phase3_
evidence.py`'s release-binding check refuses a `PreviewRelease` whose paired
`binding_ref` does not name it):

The manifest command below is an abbreviated shape, not a complete acceptance
invocation: all required L-row receipts and browser/engineering/coverage/
performance/inventory/deferred-work refs must be supplied using the existing
builder options. P3A-C4 records the full tested invocation privately. P3A-C2
must also represent the inherited accepted D14/D15 disposition; the current
validator does not yet do so. Preserve strict findings and never substitute
an old generation receipt for a fresh failing one.

```bash
/usr/bin/python3 checks/rearchitecture_phase3_evidence_build.py \
    --artifact-root /private/phase3-artifacts \
    --phase2-evidence /private/phase2-evidence.json --phase2-artifact-root /private/phase2-artifacts \
    --preview-input <preview_input.json> \
    --accepted-release <release_1.json>=<release_1_binding.json> \
    --accepted-release <release_2.json>=<release_2_binding.json> \
    --population-manifest <population_manifest.json> \
    --comparison-receipt full_population_parity=<receipt.json> \
    --output /private/phase3-evidence.json
/usr/bin/python3 -u checks/rearchitecture_phase3_gate.py \
    --evidence-manifest /private/phase3-evidence.json --write-report \
    --report-path /private/phase3-report/REPORT.md
```

`--write-report` refuses a `--report-path` that resolves inside this repo
(`checks/rearchitecture_phase3_gate.py::write_report`,
`ReportPathError`); omit `--report-path` to use the default private location
under the system temp directory.

## 6. Return to the prior release; stop the preview; locate private reports

**Status: EXISTS for the pointer move (step 3's same fenced `publish_local`
republishes an earlier release id — `tests/test_v2_serving_publication_
binding.py`'s R1/R2 tests prove both generations stay retained and
independently servable); UNVERIFIED against a real release by this task.**

- Roll back: `publish_local` refuses an older `occurrence` from replacing a
  newer one (`newest = MAX(occurrence) WHERE published_at IS NOT NULL`), so
  a rollback is NOT literally republishing the prior release id at its old
  occurrence. It is the same restage-under-a-fresh-id path guide §5.5 names
  and `tests/test_v2_ops_same_session_replan.py`/`tests/test_v2_serving_
  publication_binding.py`'s R1/R2 tests prove: re-run step 3's publish with
  a FRESH occurrence whose staged content (files, and its own
  `projection_binding.json`) equals the prior release's — the prior
  release's own committed row and files are never deleted or edited, only
  `CURRENT` moves.
- Stop the preview/API process: `Ctrl-C`, or `kill <pid>` — both launchers
  run `serve_forever`/`uvicorn.run` in the foreground and shut down cleanly
  on interrupt (`engine/v2/dashboard/preview.py::_shutdown`).
- Private reports/logs: the Phase 3 gate's `REPORT.md` lands wherever
  `--report-path` said (never inside this repo — refused, see step 5); ops
  logs are under the `--root` given to `engine.v2.ops` (default
  `data/operations`, itself gitignored).

Do not delete original artifacts or decisions when performing any of the
above (guide §10).

### Retained-input serving publication (supervised)

After the verified projection tool has registered `projection_binding.json`,
submit a fresh, immutable publication job. The source publication must
already be succeeded; it supplies retained finality, bundle, selfcheck and
engineering bindings. This command only submits work:

```
/usr/bin/python3 tools/v2_dashboard_publish.py --root /private/ops \
  --source-publication-job job_SOURCE --projection-binding art_BINDING \
  --operation-id publish-a
/usr/bin/python3 -m engine.v2.ops.cli --root /private/ops serve --once
```

Repeat the identical `--operation-id` to retrieve the same job. To roll A
back after B, pass A retained projection binding with a new operation id
(for example `rollback-a-001`), then run the supervisor command again. Do
not update a source job, attempt, lease, fence, release or `CURRENT` by hand.

## Deferred / not this task

- Receipt producers now exist, including engineering history, browser, coverage
  and performance. Reuse the implementations and historical refs in the status
  file. Remaining work is fresh same-generation acceptance, the coverage
  ratchet, inherited-disposition handling and an assembled report, not another
  set of parallel producers.
- Incremental ingestion is Phase 3B. Native scoring/models, complete consumer
  parity and official cutover are Phases 4–7; this runbook does not activate them.
