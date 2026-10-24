# Static feature-space exploration preview

This opt-in foundation expands declared feature blocks, not learned importance.
It does not change the closed H/HM/HMO compiler, its 33 candidates/603-fit example,
existing v1 admission, or the stage-2 prohibition on adaptive/greedy selection.
No feature search, registration, private-data read or winner is enabled here.

## Contract and algorithm

`experiments.magnitude_feature_subsets.plan_subsets(request)` accepts closed JSON
schema `magnitude_feature_subsets_request.v1` with exactly these fields:

| Field | Required declaration |
| --- | --- |
| `dictionary`, `column_list_id` | 1–512 ordered unique lower-case column identifiers, each at most 64 characters; ID equals `content_hash(dictionary)`. This verifies names/order only, not panel dictionary metadata, sources or admission. |
| `core` | Nonempty dictionary-ordered fixed columns, retained in every subset. |
| `optional_blocks` | 0–64 `{name, columns}` objects, unique names in lexical order; each nonempty column list is dictionary-ordered. Core and blocks must disjointly partition the dictionary. |
| `model_configuration_ids` | 1–256 sorted unique canonical SHA-256 IDs; opaque declared identities, not verified recipes or compatible estimators. |
| `horizons` | Exactly the integer list `[5, 3, 1]`. |
| `outer_years`, `inner_years` | 1–64 sorted unique four-digit outer years (0002–9998), exact inner-map keys; each 1–64 sorted unique inner years strictly precedes its outer year. |
| `max_subsets`, `max_candidates`, `max_fits` | Explicit positive integer caps, each at most 2,147,483,647. No inferred or automatically increased budget. |

Enumerate core, core plus each single block, full dictionary minus each single
block, and full dictionary. Deduplicate identical ordered columns, then sort by
subset ID. For K optional blocks the unique count is 1/2/4 for K=0/1/2 and
2K+2 otherwise; this is linear in blocks, not exhaustive 2^P feature search.
Standard Python sets/lists and native canonical hashing suffice; sklearn's
SequentialFeatureSelector/RFE require estimator fits and are intentionally not used.

Bounds, membership, exact partition and temporal declarations validate before
hashing or expansion; analytical counts must fit all caps before subsets are built.
With S subsets, M model configurations, I total inner folds and O outer folds,
C=S×M candidates and search fits=3×(C×I+O). Counts cover every planned inner
attempt plus one selected outer fit per horizon/year; failures consume attempts.
They exclude baseline/comparison fits and are not operational resource enforcement.
Four optional blocks produce 10 subsets; 11 model IDs, six inner folds and three
outer folds require **110 candidates and 1,989 search fits**. This is a tested
example only; a caller must explicitly preregister this larger budget.

The detached `magnitude_feature_subset_plan.v1` preview carries the exact request,
its `plan_id`, explicit horizons, subsets and counts. Each subset binds its schema,
`column_list_id` and ordered `columns` in `subset_id`. No missing/quote indicators
are added; downstream preprocessing may use only the selected columns. Renaming
blocks or changing declarations changes the plan identity, not necessarily the
intrinsic subset identity. Malformed/oversized requests and insufficient caps
refuse nonretryable `INVALID_EXPERIMENT_SPEC`, without partial output or I/O.

## Integration still required

1. Bind subsets to an admitted versioned full panel dictionary/source policy,
   model recipes, cohort/eligibility and immutable executable candidate identities.
   `column_list_id` is **not** the dispatcher's `feature_dictionary_hash`.
2. Preregister the exact grid, fold schedule, comparison and resource budgets.
   Preprocess/fit each candidate on qualified inner training only; record all
   attempts and failures. No full-data importance ranking or subset pruning.
3. Preserve pooled inner RMSE, exact candidate-ID tie breaks, selected-only outer
   fits and latest-outer inner winner. Neither outer labels nor random/temporal
   holdouts may choose features; no alternative selection path is introduced.

These are later dependency-gated slices, not behavior of this preview. The
caller-supplied dictionary makes no SPY/GSPC, availability or protected-read claim.
