# Authorized final read: lifecycle design (#490)

Design proposal, moved out of `engine/v2/research/ARCHITECTURE.md` (the "Authorized final read" section) on 2026-11-15 because it describes behaviour that is not built: no production authorization entrypoint, final-read caller or result retrieval exists. The implemented request preparation, population reader and evaluator stay in that section as contract. Where this guide and the section differ, the section governs. The text below is the former section text, unchanged.

**Authority/ownership.** A proposed `tools/v2_final_holdout.py` leaf composes ops'
private authorization/state owner with a private research final evaluator. It is
the sole supported evaluator caller; ops/research remain layer-7 peers with no imports between them. Foundation supplies membership and pure request validation, never grants or state.
Training's lower layer cannot import either owner. Selection/sweep/loaders gain no
grant argument, final purpose, environment bypass or returned holdout frame.
Import/call-site regression guards also cover unlayered tools/experiments, which
the layer checker alone does not cover. Only the dedicated leaf may compose these owners; registered training/selection/sweep entrypoints cannot reach that leaf.
The leaf checks live ops state/fence before its sole evaluator invocation and before publication; research checks the immutable bound request, never imports ops state.
No reusable grant escapes that invocation. This is procedural/audited, not a security boundary against arbitrary code, a forged terminal or catalog edits.

**Durability.** Ops owns the append-only authorization/spend ledger in its catalog;
one transaction records who, UTC when, explicit confirmation, request digest and
the permanent decision spend. An interactive terminal and fresh explicit user
confirmation of the displayed binding are required; flags/piped input/agents cannot assent.
`ops.final_holdout.authorize_final_holdout` is currently test-consumed only: it verifies the exact catalog-owned registration, displays canonical request bytes and requires a terminal-entered phrase containing the digest and recorded process OS identity. This is procedural confirmation, not authentication of the human at the terminal. Its persisted acknowledgement is not a read grant or completeness verdict; the later fenced reader must still verify winner/revocation and the preregistered completeness artifact before scans.
Unique `experiment_id` plus compare-and-swap state serialize authorization and reads.
States are `authorized_unread`, `read_started`, `result_staged`, `complete`, or terminal `revoked`/`indeterminate`; every transition appends evidence, never refunds. Authorization has no expiry; revocation or a changed winner ends eligibility with `HOLDOUT_ACCESS_DENIED`, retaining the spend. Interrupted reads stay spent with no continuation; no renewal resets spend.

| Proposed condition | Required outcome before any returned metric/report |
| --- | --- |
| Missing user record, noninteractive authorization, tuning/training/selection/sweep, new read after `read_started`, or revoked/changed winner | Non-retryable `HOLDOUT_ACCESS_DENIED`, before holdout outcome scans or metrics; no partial result. Recheck winner/revocation at the fenced start and publication. Revocation after start but before completion burns the read and withholds its result. |
| Duplicate authorization | Identical request returns the original acknowledgement only after the same interactive checks; never another grant/spend. Changed bytes under the same decision refuse `IDEMPOTENCY_CONFLICT`; changed registration refuses `EXPERIMENT_IDENTITY_CONFLICT`. |
| Authorization recorded, no read started | A first user-invoked read may atomically commit `read_started` with one owner/fence before any holdout outcome scan. This never continues an interrupted read; missing/corrupt state never means unspent. |
| Crash after `read_started`, before durable `result_staged` | Once ownership is proven gone, mark `indeterminate`; never reread, even if the crash might precede the first scan. No result is released and the spend remains. |
| Crash after durable result, before completion receipt | Only a catalog-bound `result_staged` reference to fully synced immutable result bytes permits artifact-only reconciliation. Verify hashes/pins and the current fence/winner, then commit the receipt reference; never evaluate again. Unbound files cannot prove completion. |
| Receipt/diagnostic/ledger publication failure | Preserve the original refusal and durable state; no success output. Retry storage reconciliation only. Missing/malformed evidence is `VALIDATION_FAILED`, conflicting identities `IDEMPOTENCY_CONFLICT`, transient storage `RESOURCE_UNAVAILABLE`; a visible file alone, including failed sync/rollback residue, is not committed evidence. |
| Missing/mismatched snapshot or failed bounded scan | `SNAPSHOT_UNRESOLVED` or the existing repository refusal; no fallback, cache, partial report or scan retry. Failure after `read_started` keeps the spend. |

Results remain private until completion binds the request, authorizing act, spend, both memberships, per-set counts/metrics and overlap count; never average the sets.
Random-set temporal-neighbour correlation and post-release-selection provenance remain disclosed (#373).
Only the authorizing user may view results through the dedicated tool, with fresh interactive confirmation. Results belong in a separate namespace excluded from agent-facing tools and covered by an agent read-deny rule; no encryption or OS containment. Other callers refuse `HOLDOUT_ACCESS_DENIED` before output.
Ops durably audits the viewing actor, UTC when, verified request/report references and checked revocation-history revision before output; failed checks/audit publication release nothing. Retrieval returns unchanged verified artifacts with that history alongside them; it never scans, recomputes, tunes, reauthorizes or spends again.
Agent reconciliation may verify private artifacts but cannot return their metrics/report. No loader, training, selection or sweep report interface or `--no-ledger` final read is available.
Completion is the release point: later revocation appends audit history and accompanies retrieval; it cannot erase the completed report or renew read access.
