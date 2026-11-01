"""No lifecycle action may silently accept unbound own/source file declarations."""
import json
import sqlite3
from contextlib import closing
from copy import deepcopy
from hashlib import sha256

import pytest

from engine.v2.foundation import ArtifactStore, to_document
from engine.v2.ops.errors import fail
from engine.v2.ops.experiments import experiment_spec_from_document
from experiments import distribution_cli as cli
from tests.v2.research.magnitude_trial_support import prediction_spec
from tests.v2.research.test_distribution_cli_errors import operator as operator
from tests.v2.research.test_distribution_cli_errors import refused

ACTIONS = ("prepare", "register", "preflight", "seal", "run", "collect", "report")
MESSAGE = "distribution lifecycle requires no file inputs"


def durable(operator):
    # Compare logical durable catalog contents, not raw files/sidecars: register
    # and seal legitimately open the catalog writable before metadata parsing.
    with closing(sqlite3.Connection(operator.paths["catalog"])) as conn:
        catalog = tuple(conn.iterdump())
    root = operator.paths["store-root"]
    objects = tuple(sorted((str(path.relative_to(root)), sha256(path.read_bytes()).hexdigest())
        for path in root.rglob("*") if path.is_file()))
    return catalog, objects


@pytest.fixture
def input_operator(operator, monkeypatch):
    ArtifactStore(operator.paths["store-root"]).publish_bytes(b"untouched sentinel", schema_ref="synthetic.v1")
    before, writes = durable(operator), []
    connect = cli.connect
    def writable(*args, **kwargs):
        assert operator.argv[0] in ("register", "seal"), "invalid input reached writable run fallback"
        writes.append(operator.argv[0])
        return connect(*args, **kwargs)
    def forbidden(*args, **kwargs):
        pytest.fail("invalid file declaration reached trials, registration mutation or CAS publication")
    monkeypatch.setattr(cli, "connect", writable)
    monkeypatch.setattr(cli, "DistributionTrials", forbidden)
    monkeypatch.setattr(cli, "reserve_native_registration", forbidden)
    monkeypatch.setattr(ArtifactStore, "publish_bytes", forbidden)
    yield operator
    assert operator.calls == ["contained", "catalog", "released"]
    assert len(operator.connections) == 1
    assert writes == ([operator.argv[0]] if operator.argv[0] in ("register", "seal") else [])
    assert durable(operator) == before


def declaration(operator, which, values):
    path = operator.paths[which]
    document = json.loads(path.read_text(encoding="utf-8"))
    if values is None:
        document.pop("input_files", None)
    else:
        document["input_files"] = values
    path.write_text(json.dumps(document), encoding="utf-8")


@pytest.mark.parametrize("action", ACTIONS)
@pytest.mark.parametrize("which", ["spec", "source-spec"])
def test_nonempty_own_or_source_declaration_refuses_every_action_before_native_effects(input_operator, capsys, action, which):
    input_operator.argv[0] = action
    declared = str(input_operator.paths[which].parent / "SECRET-unbound-file.csv")
    declaration(input_operator, which, [declared])
    expected = fail("INVALID_EXPERIMENT_SPEC", MESSAGE)
    # The inherited native-reader trap must not be reached. These assertions use
    # actual parsing/dispatch, not a fake implementation of the declaration rule.
    assert refused(input_operator, capsys, expected.code, private=(declared,)) == to_document(expected.problem)


@pytest.mark.parametrize("action", ACTIONS)
@pytest.mark.parametrize("values", [[], None], ids=["empty", "omitted"])
def test_empty_and_default_omitted_declarations_reach_native_admission(input_operator, monkeypatch, capsys, action, values):
    input_operator.argv[0] = action
    for which in ("spec", "source-spec"):
        declaration(input_operator, which, values)
    expected = fail("HOLDOUT_ACCESS_DENIED", "synthetic downstream admission boundary",
        stage="native", details={"boundary": "source registration"})
    reads = []
    def downstream(conn, store, spec, **kwargs):
        reads.append(spec)
        assert spec.input_files == ()
        raise expected
    monkeypatch.setattr(cli, "read_native_registration", downstream)
    assert refused(input_operator, capsys, expected.code) == to_document(expected.problem)
    assert len(reads) == 1


@pytest.mark.parametrize("identity", ["own", "source"])
def test_shared_spec_hash_does_not_bind_an_added_input_file(identity):
    document = to_document(prediction_spec(experiment_id="synthetic-" + identity))
    document.pop("input_files", None)
    empty = experiment_spec_from_document(document)
    changed = deepcopy(document)
    changed["input_files"] = ["SECRET-unbound-file.csv"]
    declared = experiment_spec_from_document(changed)
    assert empty.input_files == () and declared.input_files == ("SECRET-unbound-file.csv",)
    assert declared.spec_hash == empty.spec_hash  # A matching spec hash cannot substitute for this guard.
