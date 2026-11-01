"""Only explicit lifecycle-spec parsing normalizes shared INVALID_REQUEST errors."""
import json
from hashlib import sha256

import pytest

from engine.v2.foundation import ArtifactStore, to_document
from engine.v2.ops.errors import fail
from experiments import distribution_cli as cli
from tests.v2.research.test_distribution_cli_errors import operator as operator
from tests.v2.research.test_distribution_cli_errors import refused

LIFECYCLE = "invalid distribution lifecycle metadata"
REQUIRED = ("hypothesis", "primary_arm_id", "economic_params", "price_source")


def fingerprint(operator):
    root = operator.paths["store-root"]
    objects = tuple(sorted((str(path.relative_to(root)), sha256(path.read_bytes()).hexdigest())
        for path in root.rglob("*") if path.is_file()))
    return operator.paths["catalog"].read_bytes(), objects


@pytest.fixture
def spec_operator(operator, monkeypatch):
    # Closed DELETE-mode fixture: byte equality is a durable-main-file oracle,
    # not a claim about SQLite coordination sidecars on an arbitrary live WAL DB.
    ArtifactStore(operator.paths["store-root"]).publish_bytes(b"untouched sentinel", schema_ref="synthetic.v1")
    before, modes = fingerprint(operator), []
    connect = cli.sqlite3.connect
    def readonly(database, *args, **kwargs):
        assert database.endswith("?mode=ro") and kwargs["uri"] is True
        modes.append("ro")
        return connect(database, *args, **kwargs)
    def forbidden(*args, **kwargs):
        pytest.fail("rejected metadata reached writable fallback, publication or trial construction")
    monkeypatch.setattr(cli.sqlite3, "connect", readonly)
    monkeypatch.setattr(cli, "connect", forbidden)
    monkeypatch.setattr(cli, "DistributionTrials", forbidden)
    monkeypatch.setattr(ArtifactStore, "publish_bytes", forbidden)
    yield operator
    assert modes == ["ro"] and operator.calls == ["contained", "catalog", "released"]
    assert fingerprint(operator) == before


def edit(operator, which, change):
    path = operator.paths[which]
    document = json.loads(path.read_text(encoding="utf-8"))
    change(document)
    path.write_text(json.dumps(document), encoding="utf-8")


@pytest.mark.parametrize("which", ["spec", "source-spec"])
@pytest.mark.parametrize("field", REQUIRED)
@pytest.mark.parametrize("missing", [True, False], ids=["missing", "null"])
def test_missing_or_null_shared_required_fields_are_lifecycle_spec_refusals(spec_operator, capsys, which, field, missing):
    def change(document):
        if missing:
            del document[field]
        else:
            document[field] = None
    edit(spec_operator, which, change)
    expected = fail("INVALID_EXPERIMENT_SPEC", LIFECYCLE)
    assert refused(spec_operator, capsys, expected.code) == to_document(expected.problem)


@pytest.mark.parametrize("which", ["spec", "source-spec"])
@pytest.mark.parametrize("field,value,message,details", [
    ("arms", {}, "arms and folds must be JSON arrays", {"field": "arms", "type": "dict"}),
    ("folds", "SECRET malformed container", "arms and folds must be JSON arrays", {"field": "folds", "type": "str"}),
    ("input_files", None, LIFECYCLE, {}),
    ("unknown_field", "SECRET ignored value", "experiment specification declares fields this resolver does not know", {"fields": ["unknown_field"]}),
])
def test_real_parser_container_and_unknown_field_problem_contracts_survive(spec_operator, capsys, which, field, value, message, details):
    edit(spec_operator, which, lambda document: document.__setitem__(field, value))
    expected = fail("INVALID_EXPERIMENT_SPEC", message, details=details)
    assert refused(spec_operator, capsys, expected.code) == to_document(expected.problem)


@pytest.mark.parametrize("which", ["spec", "source-spec"])
def test_nonobject_spec_keeps_native_json_reader_problem(spec_operator, capsys, which):
    spec_operator.paths[which].write_text('["SECRET nonobject"]', encoding="utf-8")
    expected = fail("INVALID_EXPERIMENT_SPEC", "search requires valid JSON object metadata")
    assert refused(spec_operator, capsys, expected.code) == to_document(expected.problem)


@pytest.mark.parametrize("fault", ["missing_hash", "null_size", "unknown_field"])
def test_artifact_ref_document_refusals_keep_lifecycle_classification(spec_operator, capsys, fault):
    def change(document):
        if fault == "missing_hash":
            del document["content_hash"]
        elif fault == "null_size":
            document["byte_size"] = None
        else:
            document["unknown_field"] = "SECRET ref value"
    edit(spec_operator, "source-plan-ref", change)
    expected = fail("INVALID_EXPERIMENT_SPEC", LIFECYCLE)
    assert refused(spec_operator, capsys, expected.code) == to_document(expected.problem)


@pytest.mark.parametrize("point,code", [
    (point, code) for point in ("parser", "native", "trials")
    for code in ("INVALID_REQUEST", "HOLDOUT_ACCESS_DENIED", "EXPERIMENT_IDENTITY_CONFLICT", "RESOURCE_UNAVAILABLE")
    if point != "parser" or code != "INVALID_REQUEST"
])
def test_other_parser_and_all_downstream_ops_problems_are_not_globally_normalized(spec_operator, monkeypatch, capsys, point, code):
    original = fail(code, "synthetic boundary refusal", stage=point, details={"boundary": point},
        retry_after_seconds=7, dependency_refs=("synthetic-input",))
    calls = []
    def reject(*args, **kwargs):
        calls.append(point)
        raise original
    if point == "parser":
        monkeypatch.setattr(cli, "experiment_spec_from_document", reject)
    elif point == "native":
        monkeypatch.setattr(cli, "read_native_registration", reject)
    else:
        def registration(*args, **kwargs):
            calls.append("native-read")
            return object()
        monkeypatch.setattr(cli, "read_native_registration", registration)
        monkeypatch.setattr(cli, "DistributionTrials", reject)
    assert refused(spec_operator, capsys, code) == to_document(original.problem)
    assert calls == (["native-read", "native-read", "trials"] if point == "trials" else [point])
