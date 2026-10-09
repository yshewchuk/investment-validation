# V2 experiment runbook

This is the operating contract for experiments on the v2 platform. The design
and implementation status are tracked in [the v2 experiment platform](https://github.com/yshewchuk/investment-validation/pull/372).
Legacy `phase2_experiment_framework.md` describes a historical workflow, not
the v2 procedure.

## Define and preregister

1. State the question, hypothesis, primary outcome, candidate variant, and
   what result would change the decision. Keep one primary comparison; label
   additional analyses exploratory.
2. Declare the entry rule, strategy and economic parameters, feature/model
   choices, exit rule, fill convention, seed, folds, and evaluation measures
   before observing experiment results. A variant identity binds its resolved
   specification and code to its inputs; changed economics are a new variant.
3. Run on the latest pinned native v2 snapshot. This is the user decision of
   **2026-10-08**: an experiment never consumes another experiment's outputs
   or an old data subset. If comparing with an incumbent, recompute it as a
   variant arm on the same snapshot, folds, and holdouts.
4. Record the preregistration, variant identity, pinned snapshot, planned folds,
   holdout context, and intended comparison in the experiment record before
   evaluating results. Keep the primary outcome distinct from exploratory
   sweeps.

## Smoke-test, then run

5. Give every runner `--no-ledger` and pass it on the first subset or smoke
   test, before running that test. This is the user decision recorded for
   #488 on **2026-10-04**. Smoke runs check execution only; they are not
   evaluated variants and must not create ledger rows.
6. After the smoke test, run the registered variant against its pinned
   snapshot and declared folds. Keep each arm's actual consumed inputs bound
   to its variant identity. Do not change the spec after seeing results; a
   changed specification is a newly preregistered variant.
7. Keep training and model/threshold selection inside the declared folds.
   Report out-of-fold results and all attempted variants. A comparison arm
   must use the same snapshot, folds, and holdout policy as its candidate.

## Holdouts and final reads

The user decision from **2026-10-04** (#373) defines two holdouts:

- A 3% random holdout assigned by a versioned hash of event identity.
- The latest six calendar months, rolling monthly.
- The union is excluded from training, selection folds, and sweeps. Report
  random and rolling results side by side; never average them.
- A winner's final read on either holdout is spent for that decision. Events
  released from the rolling set can be reused only when they are not members of
  the versioned random holdout. Random-holdout events remain excluded from
  future training and selection. Reused rolling-only events must be labelled
  `post-release selection`; they are not holdout evidence.

**Planned enforcement:** exclusion from pinned reads is tracked in [#514](https://github.com/yshewchuk/investment-validation/pull/514).
Do not describe this enforcement as available until that work is merged.

The user decision from **2026-10-08** (#490) is that only the user may
authorize a final holdout read. An agent or supervisor may prepare a request
bound to the frozen winner and decision, but may never authorize it. The
authorization step must be interactive, require explicit confirmation, and
be recorded in the ledger. **Planned:** the final-read boundary and recording
are tracked in [#490](https://github.com/yshewchuk/investment-validation/issues/490);
there is no available final-read authorization interface until implemented.

## Prices, reporting, and completion

Use daily marks from the existing `engine/fills.py` fill convention; do not
introduce a new mark source. This is the user decision from **2026-10-04**.
If marks cannot establish which exit came first, take the conservative outcome.
Label mark-based P&L separately from fill-validated results. The v2 daily
exit walker that applies this contract is **planned** in [#516](https://github.com/yshewchuk/investment-validation/pull/516).

8. Write `REPORT.md` for the completed experiment. Include the question and
   preregistered variant, snapshot and provenance, folds, holdout labels,
   results by holdout, attempted-variant count, limitations, and figures that
   support the claims. Keep mark-based and fill-validated P&L distinct.
9. Record the completed experiment in the ledger and retain its figures. Once
   the report, ledger row, and figures are final, sync them to the private
   mirror once with [`python3 tools/private_mirror.py --experiment EXP-123 --push`](../tools/private_mirror.py),
   replacing `EXP-123` with the completed experiment ID. The CLI inventories
   configured mirror roots first, then `--experiment` selects files for that
   experiment and the shared ledger. Pruning is limited to that experiment,
   and the option limits paths passed to the helper's `git add`. The helper's
   commit includes the entire mirror-clone index and its push updates that
   branch. Before using `--push`, check that the index contains only the
   intended sync changes. Do not sync intermediate iterations. Use `--dry-run`
   first if anything about the run was unusual. The report, ledger row, and
   figures are the durable record.
10. Once the experiment is complete and its `REPORT.md` has been written and
    mirrored using step 9, it need not remain runnable and its intermediate
    data need not be retained. This is the user decision of **2026-10-08**.
    Trade tables and score Parquets are disposable. To recreate data, rerun
    the experiment on the current snapshot; the original snapshot need not be
    retained for that purpose.

## Platform gaps

Prediction experiments and trustworthy prediction/variant sweeps are
**planned** in [#325](https://github.com/yshewchuk/investment-validation/issues/325).
Do not claim those harnesses or sweep behavior are available until their
implementation is merged. The policy above applies when those capabilities
are implemented.
