# Cutover remaining work

Verified 2026-10-02 against `origin/main` (head `aade62a`) and GitHub issues.
Current state only. Companion to the [delivery plan](rearchitecture_delivery_plan.md)
and the [Phase 7 cutover plan](rearchitecture_phase7_cutover.md); where this file
conflicts with them or with a component `ARCHITECTURE.md`, those win.

## Where the side-by-side path stands

The native scoring chain is built and wired up to the raw-row input, which is
not: with the scheduled plan pinning a snapshot, `nightly.submit_native_score_batch_shadow_if_ready`
still raises `VALIDATION_FAILED` ("raw-row producer ... not built yet"), so no
`native_score_batch` has run on a real night and `native_parity` has nothing to
pair. The retained serving projection returns `no_report` when the parity report is absent.

## Done

- Release reader, `SourceBundle` assembler, batch dispatch: #47, #59, #48, #67, #66.
- `native_score_batch` shadow submission and snapshot-pinned nightly plan:
  #88, #126, #145, #150, #237 (planning bound to the verified snapshot; with
  #212 this closes the CAS/`calendar_version` guard, #200).
- `native_parity` job, pairing core, supervisor sidecar (runs every tick):
  #132, #185, #191, #227, #211, #242.
- Parity serving: saved-report projection (#189, #198);
  retained captured comparison added to that projection (#255), exported by
  `tools/captured_native_comparison.py` (#254).
- Raw-row producer pieces: board enumeration (#212), calendar rows (#239),
  history advance (#245), regime math (#248), run-up math (#250), daily-state
  input staging (#257), shared serving fold ownership and verified size-fold
  authoring (#249, #258), EOD availability preflight (#259, no affirmative
  admission), React/serving boundary doc (#261).
  The nightly submission path calls none of them yet.
- Related fixes: #238 (corrupt score-batch artifacts refused), #240 (refresh
  key parsing).

## Remaining

- [ ] **Raw-row producer, end to end.** Panel, Tier-4 forecast and quote row
  staging, per-row refusal handling, and replacing the raise in `nightly.py`
  with the real build and submit. No PR open. Shadow-run prerequisite:
  [#243](https://github.com/yshewchuk/investment-validation/issues/243)
  (intraday earnings-row admission). Forward forecast design:
  `engine/v2/scoring/ARCHITECTURE.md`.
- [ ] **Genuine EOD source availability and finality evidence**
  ([#260](https://github.com/yshewchuk/investment-validation/issues/260))
  is required before any non-shadow caller and before cutover. Deferred for
  shadow runs only (see the [#309 design](https://github.com/yshewchuk/investment-validation/pull/309));
  currently has no owner (no assignee or PR aimed at it).
- [ ] **First real `native_score_batch` and `native_parity` run** on a pinned
  night; nothing exists to classify tolerances against until then.
- [ ] **React side-by-side screen.** Owned by the React screen slice of #327;
  the operations parity HTML/JSON preview is retired.
- [x] **React transport design.** #327 selects authenticated
  `/api/v1/native_parity*` reads through the serving API.
- [ ] **Production release root.** The binding is the `MODEL_RELEASE_ROOT`
  environment variable and fails closed when unset. Production's value is
  unverified from the repository.
- [ ] **Scheduler install.** `ops/systemd/native-nightly-trigger.{service,timer}`
  (30-minute timer) are checked in; installation and enablement on the
  production host are unverified.
- [ ] **`forward_calendar_refresh` submitter.** Registered job kind, no nightly
  graph node or tick-loop submitter
  ([#206](https://github.com/yshewchuk/investment-validation/issues/206)).

Later Phase 7 work, not started: replacing `LegacyScoreBridge`
(`engine/v2/serving/bridge.py`), native decisions-predictions, the nightly
pool/residual refresh implementation
([#192](https://github.com/yshewchuk/investment-validation/issues/192);
related gaps #137, #149, #154), and the 10-session qualification run, during
which the legacy board keeps publishing.
