"""Scalar normalization preserves wire identity and real admission semantics."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from engine.v2.foundation import from_document
from engine.v2.ops import provider_budget
from engine.v2.ops.catalog import dumps, transaction
from engine.v2.ops.errors import OpsError
from engine.v2.ops.profiles import DEFAULT_POLICY
from engine.v2.ops.provider_requirements import ProviderRequirement, provider_requirements
from engine.v2.ops.scheduler import claim_next
from engine.v2.ops.submission import (
    JobKind,
    KindRegistry,
    RetryPolicy,
    get_job,
    request_digest,
    submit,
    submit_graph,
)
from tests.ops_support import POLICY, catalog, request, sample


@dataclass(frozen=True)
class ProviderParameters:
    """Let persisted malformed values reach the scheduler's own guards."""

    provider_calls: Any = 1
    provider_calls_by_account: Any = None


REGISTRY = KindRegistry([JobKind(
    name="tiny_provider", worker="tiny", parameters=ProviderParameters,
    resource_classes=frozenset({"delivery"}), effects=("staged",),
    retry=RetryPolicy("bounded", 3, (1, 2)), checkpoint_contract="rows.v1.0",
    namespaces=frozenset({"shadow"}))])

# Captured from the unchanged scalar implementation, not rebuilt from the
# current serializer or dataclass defaults.
LEGACY_JOB_JSON = (
    '{"checkpoint_contract_ref":"rows.v1.0","deadline_at":null,'
    '"dependency_job_ids":[],"environment_ref":"env","implementation_ref":"code",'
    '"input_refs":[],"kind":"tiny_provider","output_namespace":"shadow",'
    '"parameters":{"provider_calls":3},"priority":0,"provider_budget_ref":"acct",'
    '"resource_class":"delivery","retry_policy_ref":"bounded",'
    '"schema_version":"job_spec.v1.0","spec_hash":null}'
)
LEGACY_REQUEST_JSON = (
    '{"idempotency_key":"legacy-wire","job":' + LEGACY_JOB_JSON
    + ',"namespace":"shadow","principal":"operator",'
    '"schema_version":"submit_request.v1.0"}'
)
LEGACY_DIGEST = "sha256:652ecc9e7b85bd321b4d2c57e88b73433cf0a35e979ca0b2a65af4902dfce21f"
CLAIM_TABLES = ("attempts", "resource_reservations", "cpu_assignments",
                "provider_reservations", "store_leases")


@pytest.fixture
def ops(tmp_path):
    """An isolated real catalog with a fixed clock and synthetic capacity."""
    conn, clock, supervisor = catalog(tmp_path)
    yield conn, clock, supervisor
    conn.close()


def _request(key="one", *, parameters=None, account="acct", **changes):
    """Build a synthetic provider request without normalizing its parameters."""
    return request(key, kind="tiny_provider", provider_budget_ref=account,
                   parameters={} if parameters is None else parameters, **changes)


def _claim(ops):
    """Run the real scheduler against the fixture's fixed clock and capacity."""
    conn, clock, supervisor = ops
    return claim_next(conn, policy=DEFAULT_POLICY, sample=sample(clock),
                      supervisor=supervisor, clock=clock)


def _assert_no_claim_rows(conn):
    """Verify refusal left no attempt, reservation, CPU assignment or lease."""
    for table in CLAIM_TABLES:
        assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0, table


def _assert_waiting(conn, job_id, reconsider):
    """Verify provider refusal stays queued without consuming an attempt."""
    receipt = get_job(conn, job_id)
    assert receipt.state == "queued"
    assert receipt.attempt_count == 0
    assert receipt.active_attempt_id is None
    assert receipt.fence == 0
    assert receipt.queue_reason.code == "PROVIDER_UNAVAILABLE"
    assert receipt.queue_reason.reconsider == reconsider


@pytest.mark.parametrize(("account", "parameters", "expected"), [
    (None, {}, ()),
    (None, {"provider_calls": "invalid"}, ()),
    ("", {}, ()),
    ("", {"provider_calls": "invalid"}, ()),
    ("acct", {}, (ProviderRequirement("acct", 1),)),
    ("acct", {"provider_calls": 3}, (ProviderRequirement("acct", 3),)),
    ("acct", {"provider_calls": "3"}, (ProviderRequirement("acct", "3"),)),
    ("acct", {"provider_calls": 1.5}, (ProviderRequirement("acct", 1.5),)),
    ("acct", {"provider_calls": None}, (ProviderRequirement("acct", None),)),
    ("acct", {"provider_calls": True}, (ProviderRequirement("acct", True),)),
])
def test_normalization_is_a_tuple_without_coercion_or_job_mutation(account, parameters, expected):
    """Missing calls default to one; explicit values keep their original type."""
    job = _request(account=account, parameters=parameters).job
    before = deepcopy(job)
    wire = dumps(job)
    requirements = provider_requirements(job)
    assert isinstance(requirements, tuple)
    assert requirements == expected
    if account and "provider_calls" in parameters:
        assert requirements[0].calls is parameters["provider_calls"]
    assert job == before
    assert job.parameters is parameters
    assert dumps(job) == wire


@pytest.mark.parametrize(("field", "value"), [("account", "other"), ("calls", 9)])
def test_normalized_requirement_fields_are_frozen(field, value):
    """Callers cannot reassign an account or estimate after normalization."""
    requirement, = provider_requirements(_request(parameters={"provider_calls": 3}).job)
    with pytest.raises(FrozenInstanceError):
        setattr(requirement, field, value)
    assert requirement == ProviderRequirement("acct", 3)


def test_legacy_wire_digest_and_repeated_submission_are_unchanged(ops):
    """A repeated scalar request returns the same row without rewriting it."""
    conn, clock, _ = ops
    req = _request("legacy-wire", parameters={"provider_calls": 3})
    assert dumps(req.job) == LEGACY_JOB_JSON
    assert dumps(req) == LEGACY_REQUEST_JSON
    assert request_digest(req) == LEGACY_DIGEST
    first = submit(conn, REGISTRY, POLICY, req, clock=clock)
    before = dict(conn.execute("SELECT * FROM jobs").fetchone())
    changes = conn.total_changes
    clock.advance(30)
    repeated = submit(conn, REGISTRY, POLICY, req, clock=clock)
    assert repeated == first
    assert conn.total_changes == changes
    assert dict(conn.execute("SELECT * FROM jobs").fetchone()) == before
    assert before["spec_json"] == LEGACY_JOB_JSON
    assert before["request_digest"] == LEGACY_DIGEST
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
    assert dumps(req) == LEGACY_REQUEST_JSON
    _assert_no_claim_rows(conn)


@pytest.mark.parametrize("entry_point", ["submit", "submit_graph"])
@pytest.mark.parametrize("account", [None, "acct"])
@pytest.mark.parametrize("mapping", [{}, None, {"acct": 3}], ids=["empty", "null", "valid"])
def test_reserved_map_is_rejected_before_any_submission_sql(ops, entry_point, account, mapping):
    """The shared guard rejects a key the kind schema would otherwise accept."""
    conn, clock, _ = ops
    parameters = {"provider_calls": 3, "provider_calls_by_account": mapping}
    assert from_document(ProviderParameters, parameters).provider_calls_by_account == mapping
    req = _request(parameters=parameters, account=account)
    statements = []
    before = conn.total_changes
    conn.set_trace_callback(statements.append)
    try:
        with pytest.raises(OpsError) as exc:
            if entry_point == "submit":
                submit(conn, REGISTRY, POLICY, req, clock=clock)
            else:
                submit_graph(conn, REGISTRY, POLICY, [_request("valid-first"), req], clock=clock)
    finally:
        conn.set_trace_callback(None)
    assert exc.value.code == "INVALID_REQUEST"
    assert exc.value.problem.details == {"field": "job.parameters.provider_calls_by_account"}
    assert statements == []
    assert conn.total_changes == before
    assert not conn.in_transaction
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
    _assert_no_claim_rows(conn)


@pytest.mark.parametrize(("account", "parameters", "calls"), [
    ("acct", {"provider_calls": 3}, 3),
    ("acct", {}, 1),
    ("acct", {"provider_calls": True}, 1),
    (None, {"provider_calls": 3}, None),
    (None, {"provider_calls": "invalid"}, None),
    ("", {}, None),
    ("", {"provider_calls": 3}, None),
    ("", {"provider_calls": "invalid"}, None),
])
def test_real_claim_reserves_only_the_original_scalar_requirement(ops, account, parameters, calls):
    """Claim holds the scalar lease; absent accounts require no provider lease."""
    conn, clock, _ = ops
    if account:
        provider_budget.configure_account(conn, account, "generation", remaining=10, live_reserve=2)
    req = _request(parameters=parameters, account=account)
    receipt = submit(conn, REGISTRY, POLICY, req, clock=clock)
    claimed = _claim(ops)
    assert claimed is not None
    assert claimed.job_id == receipt.job_id
    assert claimed.spec == req.job
    rows = conn.execute("SELECT * FROM provider_reservations").fetchall()
    if calls is None:
        assert rows == []
    else:
        assert len(rows) == 1
        assert rows[0]["account"] == account
        assert rows[0]["attempt_id"] == claimed.attempt_id
        assert rows[0]["fence"] == claimed.fence
        assert rows[0]["reserved_calls"] == calls
        assert rows[0]["used_calls"] == 0
        assert rows[0]["released_at"] is None
        assert conn.execute("SELECT remaining FROM provider_accounts").fetchone()[0] == 10
    assert get_job(conn, receipt.job_id).attempt_count == 1
    assert conn.execute("SELECT spec_json FROM jobs").fetchone()[0] == dumps(req.job)


@pytest.mark.parametrize("calls", [0, -1, "3", 1.5, None, False, [], {}])
@pytest.mark.parametrize("account_state", ["available", "missing", "blocked"])
def test_invalid_scalar_waits_without_attempt_and_keeps_account_guard_precedence(
        ops, calls, account_state):
    """Unusable accounts win over malformed counts, as before normalization."""
    conn, clock, _ = ops
    if account_state != "missing":
        provider_budget.configure_account(conn, "acct", "generation", remaining=10, live_reserve=0)
    if account_state == "blocked":
        provider_budget.record_response(conn, "acct", 401, clock=clock)
    req = _request(parameters={"provider_calls": calls})
    receipt = submit(conn, REGISTRY, POLICY, req, clock=clock)
    assert _claim(ops) is None
    reconsider = "specification_change" if account_state == "available" else "operator_action"
    _assert_waiting(conn, receipt.job_id, reconsider)
    _assert_no_claim_rows(conn)
    assert conn.execute("SELECT spec_json FROM jobs").fetchone()[0] == dumps(req.job)


def test_call_estimate_above_reservation_cap_rolls_back_the_whole_claim(ops):
    """The historical reservation cap still raises after admission, atomically."""
    conn, clock, _ = ops
    provider_budget.configure_account(conn, "acct", "generation", remaining=2_000_000, live_reserve=0)
    req = _request(parameters={"provider_calls": 1_000_001})
    submit(conn, REGISTRY, POLICY, req, clock=clock)
    job_before = dict(conn.execute("SELECT * FROM jobs").fetchone())
    account_before = dict(conn.execute("SELECT * FROM provider_accounts").fetchone())
    statements = []
    conn.set_trace_callback(statements.append)
    try:
        with pytest.raises(OpsError) as exc:
            _claim(ops)
    finally:
        conn.set_trace_callback(None)
    assert exc.value.code == "INVALID_REQUEST"
    assert "BEGIN IMMEDIATE" in statements
    assert statements[-1] == "ROLLBACK"
    assert not conn.in_transaction
    assert dict(conn.execute("SELECT * FROM jobs").fetchone()) == job_before
    assert dict(conn.execute("SELECT * FROM provider_accounts").fetchone()) == account_before
    _assert_no_claim_rows(conn)


@pytest.mark.parametrize("account", [None, "acct"])
@pytest.mark.parametrize("mapping", [{}, None, {"acct": 3}], ids=["empty", "null", "valid"])
def test_reserved_map_in_stored_job_waits_without_attempt_or_reservation(ops, account, mapping):
    """Legacy or hand-seeded unsupported shapes fail closed at claim time."""
    conn, clock, _ = ops
    provider_budget.configure_account(conn, "acct", "generation", remaining=10, live_reserve=0)
    req = _request(account=account)
    receipt = submit(conn, REGISTRY, POLICY, req, clock=clock)
    stored = replace(req.job, parameters={"provider_calls": 3, "provider_calls_by_account": mapping})
    with transaction(conn):
        conn.execute("UPDATE jobs SET spec_json = ? WHERE job_id = ?",
                     (dumps(stored), receipt.job_id))
    assert _claim(ops) is None
    _assert_waiting(conn, receipt.job_id, "specification_change")
    _assert_no_claim_rows(conn)
    assert conn.execute("SELECT spec_json FROM jobs").fetchone()[0] == dumps(stored)


@pytest.mark.parametrize("shape", ["invalid_scalar", "stored_map"])
def test_unusable_requirement_does_not_prevent_an_independent_job_from_claiming(ops, shape):
    """An unusable higher-priority requirement cannot starve a healthy account."""
    conn, clock, _ = ops
    for account in ("acct", "healthy"):
        provider_budget.configure_account(conn, account, "generation", remaining=10, live_reserve=0)
    invalid_request = _request("invalid", parameters={"provider_calls": "3"}, priority=10)
    invalid = submit(conn, REGISTRY, POLICY, invalid_request, clock=clock)
    if shape == "stored_map":
        stored = replace(invalid_request.job, parameters={"provider_calls_by_account": {"acct": 3}})
        with transaction(conn):
            conn.execute("UPDATE jobs SET spec_json = ? WHERE job_id = ?",
                         (dumps(stored), invalid.job_id))
    healthy_request = _request("healthy", account="healthy", parameters={"provider_calls": 2})
    healthy = submit(conn, REGISTRY, POLICY, healthy_request, clock=clock)
    claimed = _claim(ops)
    assert claimed is not None
    assert claimed.job_id == healthy.job_id
    _assert_waiting(conn, invalid.job_id, "specification_change")
    attempts = conn.execute("SELECT job_id FROM attempts").fetchall()
    assert [row[0] for row in attempts] == [healthy.job_id]
    reservations = conn.execute("SELECT account, reserved_calls FROM provider_reservations").fetchall()
    assert [tuple(row) for row in reservations] == [("healthy", 2)]
    account = conn.execute("SELECT remaining FROM provider_accounts WHERE account = 'acct'").fetchone()
    assert account[0] == 10


def test_provider_contract_table_ends_before_following_prose():
    """A blank boundary keeps the adjacent contract out of the Markdown table."""
    root = Path(__file__).resolve().parents[3]
    document = (root / "engine/v2/ops/ARCHITECTURE.md").read_text()
    preceding, _ = document.split("`forward_calendar_refresh` remains direct-submit only;", 1)
    assert preceding.rstrip().endswith("|")
    assert preceding.endswith("\n\n")
