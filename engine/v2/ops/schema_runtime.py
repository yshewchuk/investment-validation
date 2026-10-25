"""Additive runtime schema; migrations already applied are never edited."""

STATEMENTS = (
    """CREATE TABLE artifacts (
        artifact_id TEXT PRIMARY KEY, ref_json TEXT NOT NULL,
        producer_attempt_id TEXT REFERENCES attempts(attempt_id), created_at TEXT NOT NULL
    ) STRICT""",
    """CREATE TABLE checkpoints (
        cache_key TEXT PRIMARY KEY, stage_id TEXT NOT NULL, shard_key TEXT NOT NULL,
        receipt_json TEXT NOT NULL, producer_attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id)
    ) STRICT""",
    """CREATE TABLE attempt_outputs (
        attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id), name TEXT NOT NULL,
        artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id), PRIMARY KEY(attempt_id, name)
    ) STRICT""",
    """CREATE TABLE process_members (
        attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id), pid INTEGER NOT NULL,
        start_ticks INTEGER NOT NULL, identity_json TEXT NOT NULL,
        PRIMARY KEY(attempt_id, pid, start_ticks)
    ) STRICT""",
    """CREATE TABLE provider_accounts (
        account TEXT PRIMARY KEY, generation TEXT NOT NULL, remaining INTEGER NOT NULL,
        live_reserve INTEGER NOT NULL, uncertain INTEGER NOT NULL DEFAULT 0,
        blocked_code TEXT, next_eligible_at TEXT
    ) STRICT""",
    """CREATE TABLE provider_reservations (
        account TEXT NOT NULL REFERENCES provider_accounts(account),
        attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id), fence INTEGER NOT NULL,
        reserved_calls INTEGER NOT NULL CHECK(reserved_calls >= 0),
        used_calls INTEGER NOT NULL DEFAULT 0, released_at TEXT, PRIMARY KEY(account, attempt_id)
    ) STRICT""",
    "CREATE UNIQUE INDEX provider_one_consumer ON provider_reservations(account) WHERE released_at IS NULL",
    """CREATE TABLE store_leases (
        domain TEXT NOT NULL, attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id),
        mode TEXT NOT NULL CHECK(mode IN ('read','write')), released_at TEXT,
        PRIMARY KEY(domain, attempt_id)
    ) STRICT""",
    "CREATE TABLE store_domains (domain TEXT PRIMARY KEY, dirty INTEGER NOT NULL) STRICT",
    """CREATE TABLE nightly_runs (
        run_id TEXT PRIMARY KEY, occurrence TEXT NOT NULL, scope TEXT NOT NULL,
        mode TEXT NOT NULL, plan_hash TEXT NOT NULL, plan_json TEXT NOT NULL, created_at TEXT NOT NULL
    ) STRICT""",
    """CREATE TABLE run_stages (
        run_id TEXT NOT NULL REFERENCES nightly_runs(run_id), stage_id TEXT NOT NULL,
        job_id TEXT NOT NULL REFERENCES jobs(job_id), PRIMARY KEY(run_id, stage_id)
    ) STRICT""",
    """CREATE TABLE watermarks (
        pipeline TEXT NOT NULL, scope TEXT NOT NULL, stage TEXT NOT NULL,
        occurrence TEXT NOT NULL, receipt_ref TEXT NOT NULL, completed_at TEXT NOT NULL,
        PRIMARY KEY(pipeline, scope, stage)
    ) STRICT""",
    """CREATE TABLE releases (
        release_id TEXT PRIMARY KEY, occurrence TEXT NOT NULL, manifest_json TEXT NOT NULL,
        manifest_hash TEXT NOT NULL, expected_current TEXT, eligible INTEGER NOT NULL,
        published_at TEXT, delivered_at TEXT
    ) STRICT""",
    """CREATE TABLE outbox (
        effect_id TEXT PRIMARY KEY, kind TEXT NOT NULL, logical_key TEXT NOT NULL,
        payload_hash TEXT NOT NULL, payload_json TEXT NOT NULL,
        state TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
        receipt_json TEXT, UNIQUE(kind, logical_key)
    ) STRICT""",
    "CREATE INDEX outbox_pending ON outbox(state, kind)",
    """CREATE TABLE health_observations (
        occurrence TEXT NOT NULL, kind TEXT NOT NULL, ok INTEGER,
        receipt_json TEXT NOT NULL, PRIMARY KEY(occurrence, kind)
    ) STRICT""",
    """CREATE TABLE experiment_runs (
        run_id TEXT PRIMARY KEY, spec_hash TEXT NOT NULL, input_hash TEXT NOT NULL,
        mode TEXT NOT NULL CHECK(mode IN ('smoke','primary')),
        evidence_json TEXT NOT NULL, backup_key TEXT, UNIQUE(spec_hash, input_hash, mode)
    ) STRICT""",
    """CREATE TABLE hypotheses (
        spec_hash TEXT PRIMARY KEY, input_hash TEXT NOT NULL, payload_hash TEXT NOT NULL,
        payload_json TEXT NOT NULL, run_id TEXT NOT NULL REFERENCES experiment_runs(run_id)
    ) STRICT""",
)
