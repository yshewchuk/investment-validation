# RECOVERY.md — restoring this program on a fresh machine

The laptop is not the system of record. This document is the full restore path,
and it is verified mechanically: `checks/phase0_checks.py --only recovery_drill`
clones the public repo into a temporary directory and asserts that it contains
no secrets and no data, that `engine` imports, that the unit suite passes, and
that the steps below are actually present in this file.

## What lives where, and why

| Location | Contents | Rationale |
|---|---|---|
| **Public GitHub repo** | `engine/`, `checks/`, `tests/`, `guides/`, `dashboard/` code, the master plan, `.env.example`, this file | Code survives the laptop. The strategy *logic* is disclosed by design; positions, findings, and data are not. |
| **Private mirror repo** | `ledger/`, `reports/`, research findings (HANDOFF, VERDICT docs), `config/thesis/`, and the pre-engine research code under `earnings_predictions/` and `bt/` | Irreplaceable and not regenerable. Also position-revealing, so never public. |
| **Nowhere (deliberately)** | `.env`, all raw/curated/feature data, `polygon_cache/` | Secrets belong in a password manager. Market data is re-pullable at quota cost; ~57k files and ~1.2 GB do not belong in git. |

The accepted residual gap: **raw market data is not backed up.** Losing the
machine costs an ORATS quota cycle to re-pull, not the research.

## Restore, step by step

### 1. Clone the code

```bash
git clone https://github.com/<user>/<public-repo>.git investing-plan
cd investing-plan
```

### 2. Install dependencies

Python 3.14. From the system packages: `numpy`, `pandas`, `scipy`,
`scikit-learn`, `matplotlib`, `requests`, `playwright`, `yfinance`, `joblib`,
`fastapi`, `uvicorn`.

Two more are needed and may require an override on a PEP-668 system:

```bash
pip install --break-system-packages pyarrow pytest
```

`pyarrow` gives Parquet storage and `pytest` runs the suite. If `pyarrow`
genuinely cannot be installed, the store falls back to `csv.gz` automatically
(`engine/data/store.py::HAVE_PARQUET`) — same contracts, slower.

### 3. Restore credentials

Copy `.env.example` to `.env` and fill in the values **from the password
manager**. Never from any repository — the values are in neither remote by
design.

```bash
cp .env.example .env   # then edit; see the file for the variable names
```

Verify without echoing anything:

```bash
python3 -c "from checks.repo_hygiene import load_secrets; from engine import paths; \
print(len(load_secrets(paths.ENV_FILE)), 'secret patterns loaded')"
```

### 4. Restore the irreplaceable artifacts

```bash
git clone https://github.com/<user>/<private-mirror-repo>.git /tmp/mirror
rsync -a /tmp/mirror/ ./
```

This restores `ledger/`, `reports/`, the research findings docs, `config/thesis/`,
and the pre-engine research code.

### 5. Re-pull market data

Data is re-acquired, not restored. It is quota-budgeted and resumable, so it can
be run across several monthly cycles.

```bash
# Always plan and cost it first — this spends nothing:
python3 -m engine.data.pulls.sep2026_plan --dry-run

# Then, deliberately:
python3 -m engine.data.pulls.sep2026_plan --confirm
```

The quota guard refuses to spend below a 3,000-call live-operations reserve
(override only with `ORATS_ALLOW_RESERVE=1`, and only on purpose). Re-running
after an interruption is free: anything already in the Tier-1 cache is a cache
hit.

### 6. Rebuild the tiers

```bash
python3 -m engine.data.rebuild
```

Rebuilds Tier 2 from Tier 1 and Tier 3 from Tier 2, with **no network access**,
then regenerates `data/MANIFEST.md` and the snapshot hash.

### 7. Verify the restore

```bash
python3 checks/phase0_checks.py
```

All checks must be green. The one that matters most is `migration`: it proves
the rebuilt panel still reproduces `events_with_orats_sum.csv`, the panel every
verdict in the program rests on. A red migration test means the restore changed
a number, and nothing downstream should be trusted until it is green.

## If a secret is ever exposed

**Rotate it immediately.** Deleting the commit is not sufficient and never has
been: a value pushed to a public remote must be assumed captured, regardless of
how quickly the history is rewritten. Rotate at the provider, update `.env`, and
only then worry about the history.

`checks/repo_hygiene.py` exists to make this scenario not happen: it reads
`.env` at runtime and greps every staged blob for the current secret *values*
(plus base64 and URL-encoded forms), so a key pasted into any file of any type
is caught before the commit. It runs as a pre-commit hook and should also be run
manually before any push:

```bash
python3 checks/repo_hygiene.py --all
```

There is no `--no-verify` habit here. If the hook is wrong, fix the hook.
