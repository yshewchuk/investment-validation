"""P6-1 capability matrix: discovery, check findings and negative controls.

Synthetic trees only (plus one source-only run over the real checkout); no
data, ledger or model file is opened.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from checks import phase6_inventory as check
from tools import phase6_inventory as inv

LEGACY_APP = '''\
from fastapi import FastAPI
app = FastAPI()

@app.get("/api/meta")
def meta():
    return {}
'''

V2_API = '''\
def create_app(app, auth):
    @app.get("/api/v1/events", dependencies=auth)
    def events():
        return {}
    return app
'''

OPS_SERVER = '''\
_VIEWS = ("board",)

class Handler:
    def do_GET(self):
        path = self.path
        if path == "/health.json":
            return 1
        if path.lstrip("/") in _VIEWS:
            return 2
        if path.startswith("/release/"):
            return 3
'''

OPS_CLI = '''\
import argparse

def _add_ledger(commands):
    ledger = commands.add_parser("ledger")
    ledger_sub = ledger.add_subparsers(dest="x")
    ledger_sub.add_parser("import-history")

def parser():
    result = argparse.ArgumentParser()
    commands = result.add_subparsers(dest="command")
    for name in ("init", "health"):
        commands.add_parser(name)
    _add_ledger(commands)
    return result
'''

LEDGER = '''\
def _append(path, text):
    with open(path, "a") as handle:
        handle.write(text)

def write_predictions(rows):
    _append("p", "x")

def snapshot():
    write_predictions([])

def read_predictions():
    return []

def main():
    pass

if __name__ == "__main__":
    main()
'''

BASE_TOML = '''\
[[row]]
id = "{area}-row"
area = "{area}"
capability = "c"
old = {old}
new = {new}
producer = "p"
identity = "i"
tests = ["tests/test_x.py::test_ok"]
disposition = "{disposition}"
owner = "P6-4"
'''


def _write(root: Path, rel: str, text: str = "") -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _declarations(extra: str = "") -> str:
    parts = []
    for area in inv.REQUIRED_AREAS:
        if area == "board":
            old = '["route:legacy-app GET /api/meta", "cli:engine/ledger.py"]'
            new = ('["route:v2-api GET /api/v1/events", "route:v2-ops GET /health.json", '
                   '"route:v2-ops GET /board", "route:v2-ops GET /release/*"]')
        elif area == "decisions":
            old = ('["writer:engine/ledger.py::write_predictions", "writer:engine/ledger.py::snapshot", '
                   '"py:engine/ledger.py::read_predictions"]')
            new = ('["cli:ops init", "cli:ops health", "cli:ops ledger", "cli:ops ledger import-history", '
                   '"py:engine/v2/ledger/export.py::export_generation"]')
        else:
            old, new = "[]", '["py:engine/v2/ledger/export.py::export_generation"]'
        parts.append(BASE_TOML.format(area=area, old=old, new=new, disposition="native"))
    parts.append('[edge_owner_map]\n"phase-4 scoring extraction" = "P4"\n')
    parts.append('[[edge]]\nid = "edge:engine/v2/serving/bridge.py"\nmodule = "engine/v2/serving/bridge.py"\n'
                 'reads = "bundle"\nconsumer = "projections"\nowner = "P6-4"\n')
    return "\n".join(parts) + extra


def _ledger(label: str = "phase-4 scoring extraction") -> str:
    return json.dumps({"count": 1, "adapters": [{
        "module": "engine.v2.ops.legacy_adapter", "legacy_symbol": "engine.score.Scorer",
        "read_set": "manifest", "removal_phase": label}]})


def make_tree(root: Path, *, toml_extra: str = "", ledger_label: str = "phase-4 scoring extraction") -> Path:
    _write(root, inv.LEGACY_APP, LEGACY_APP)
    _write(root, inv.V2_API, V2_API)
    _write(root, inv.V2_OPS_SERVER, OPS_SERVER)
    _write(root, inv.LEGACY_SPA, "const X = 1;\n")
    _write(root, inv.LEGACY_HTML, "<html></html>\n")
    _write(root, inv.REACT_ROUTES, "export {};\n")
    _write(root, inv.LEGACY_NIGHTLY)
    _write(root, inv.V2_NIGHTLY)
    _write(root, inv.V2_ACTIONS, "def run():\n    pass\n")
    _write(root, inv.V2_EFFECTS)
    _write(root, "engine/v2/ops/cli.py", OPS_CLI)
    for rel in inv.WRITER_MODULES:
        _write(root, rel)
    _write(root, "engine/ledger.py", LEDGER)
    _write(root, "engine/v2/ledger/export.py", "def export_generation():\n    pass\n")
    _write(root, "engine/v2/serving/bridge.py", "def build_bridges():\n    pass\n")
    _write(root, "tools/__init__.py")
    _write(root, "tests/test_x.py", "def test_ok():\n    pass\n")
    _write(root, str(inv.ADAPTER_LEDGER), _ledger(ledger_label))
    _write(root, str(inv.DECLARATIONS), _declarations(toml_extra))
    _write(root, str(inv.GUIDE), f"# guide\n\n{inv.GUIDE_BEGIN}\n{inv.GUIDE_END}\n")
    return root


def codes(root: Path, *, check_guide: bool = False) -> list[str]:
    _, findings = check.run(root, check_guide=check_guide)
    return [f["code"] for f in findings]


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    return make_tree(tmp_path)


# ------------------------------------------------------------- discovery


def test_discovery_finds_every_surface(tree):
    found = inv.discover_entrypoints(tree)
    assert {"route:legacy-app GET /api/meta", "route:v2-api GET /api/v1/events",
            "route:v2-ops GET /health.json", "route:v2-ops GET /board", "route:v2-ops GET /release/*",
            "cli:ops init", "cli:ops health", "cli:ops ledger", "cli:ops ledger import-history",
            "cli:engine/ledger.py"} <= set(found)


def test_writer_discovery_follows_same_module_helpers(tree):
    found = inv.discover_entrypoints(tree)
    # snapshot writes only through write_predictions -> _append; main is a CLI, not a writer
    assert "writer:engine/ledger.py::snapshot" in found
    assert "writer:engine/ledger.py::write_predictions" in found
    assert "writer:engine/ledger.py::main" not in found
    assert "writer:engine/ledger.py::read_predictions" not in found


def test_synthetic_tree_is_clean(tree):
    assert codes(tree) == []


# -------------------------------------------------------- negative controls


def test_planted_legacy_route_without_row_is_caught(tree):
    app = tree / inv.LEGACY_APP
    app.write_text(app.read_text() + '\n@app.get("/api/planted")\ndef planted():\n    return {}\n')
    _, findings = check.run(tree, check_guide=False)
    assert [(f["code"], f["subject"]) for f in findings] == [
        ("UNOWNED_ENTRYPOINT", "route:legacy-app GET /api/planted")]


def test_planted_v2_route_and_cli_without_row_are_caught(tree):
    api = tree / inv.V2_API
    api.write_text(api.read_text().replace(
        "    return app", '    @app.post("/api/v1/jobs")\n    def jobs():\n        return {}\n    return app'))
    _write(tree, "tools/new_tool.py", "if __name__ == '__main__':\n    pass\n")
    _, findings = check.run(tree, check_guide=False)
    assert {f["subject"] for f in findings if f["code"] == "UNOWNED_ENTRYPOINT"} == {
        "route:v2-api POST /api/v1/jobs", "cli:tools/new_tool.py"}


def test_row_without_disposition_or_owner_is_caught(tmp_path):
    extra = '[[row]]\nid = "bare"\narea = "board"\nnew = ["cli:ops init"]\n'
    assert {"NO_DISPOSITION", "NO_OWNER"} <= set(codes(make_tree(tmp_path, toml_extra=extra)))


def test_invalid_disposition_and_owner_are_caught(tmp_path):
    extra = ('[[row]]\nid = "bad"\narea = "board"\nnew = ["cli:ops init"]\n'
             'disposition = "done"\nowner = "P9"\n')
    assert {"BAD_DISPOSITION", "BAD_OWNER"} <= set(codes(make_tree(tmp_path, toml_extra=extra)))


@pytest.mark.parametrize("ref", ["route:v2-api GET /api/v1/nowhere", "py:engine/v2/ledger/export.py::gone",
                                 "py:engine/v2/absent.py"])
def test_native_row_claiming_absent_entrypoint_is_caught(tmp_path, ref):
    extra = (f'[[row]]\nid = "ghost"\narea = "board"\nnew = ["{ref}"]\n'
             'disposition = "native"\nowner = "P6-4"\n')
    _, findings = check.run(make_tree(tmp_path, toml_extra=extra), check_guide=False)
    assert ("NEW_ENTRYPOINT_MISSING", "ghost", ref) in {(f["code"], f["subject"], f["detail"]) for f in findings}


def test_missing_marker_must_match_disposition(tmp_path):
    extra = ('[[row]]\nid = "m1"\narea = "board"\nnew = ["MISSING"]\ndisposition = "native"\nowner = "P6-4"\n'
             '[[row]]\nid = "m2"\narea = "board"\nnew = ["cli:ops init"]\ndisposition = "missing"\nowner = "P6-4"\n')
    _, findings = check.run(make_tree(tmp_path, toml_extra=extra), check_guide=False)
    assert {f["subject"] for f in findings if f["code"] == "MISSING_MISMATCH"} == {"m1", "m2"}


def test_interactive_action_replaced_by_cli_needs_acceptance(tmp_path):
    extra = ('[[row]]\nid = "button"\narea = "board"\nnew = ["cli:ops init"]\n'
             'disposition = "native"\nowner = "P6-4"\ninteractive = true\n')
    assert "INTERACTIVE_CLI_UNACCEPTED" in codes(make_tree(tmp_path, toml_extra=extra))


def test_accepted_cli_with_recorded_decision_passes(tmp_path):
    extra = ('[[row]]\nid = "button"\narea = "board"\nnew = ["cli:ops init"]\n'
             'disposition = "CLI-with-user-acceptance-needed"\nowner = "P6-4"\ninteractive = true\n'
             'decision = "UD-1"\n[[user_decision]]\nid = "UD-1"\nrows = ["button"]\nquestion = "q"\n')
    assert codes(make_tree(tmp_path, toml_extra=extra)) == []


def test_listed_test_and_stale_old_entrypoint_are_caught(tmp_path):
    extra = ('[[row]]\nid = "t"\narea = "board"\nold = ["route:legacy-app GET /api/gone"]\n'
             'new = ["cli:ops init"]\ntests = ["tests/test_x.py::test_absent", "tests/test_none.py"]\n'
             'disposition = "native"\nowner = "P6-4"\n')
    found = codes(make_tree(tmp_path, toml_extra=extra))
    assert found.count("TEST_MISSING") == 2 and "STALE_OLD_ENTRYPOINT" in found


def test_dormant_rows_must_be_inactive(tmp_path):
    extra = ('[[row]]\nid = "old"\narea = "research"\nnew = []\n'
             'disposition = "dormant-historical"\nowner = "8A"\n')
    assert "DORMANT_ACTIVE" in codes(make_tree(tmp_path, toml_extra=extra))


def test_required_area_absent_is_caught(tree):
    decl = tree / inv.DECLARATIONS
    text = decl.read_text()
    start = text.index('id = "export-row"') - len("[[row]]\n")
    end = text.index("[edge_owner_map]")
    decl.write_text(text[:start] + text[end:])
    assert codes(tree) == ["REQUIRED_AREA_ABSENT"]


def test_unowned_adapter_edge_is_caught(tmp_path):
    root = make_tree(tmp_path, ledger_label="phase-9 someday")
    _, findings = check.run(root, check_guide=False)
    assert [(f["code"], f["subject"]) for f in findings] == [
        ("ADAPTER_EDGE_UNOWNED", "edge:engine.v2.ops.legacy_adapter::engine.score.Scorer")]


def test_undeclared_adapter_module_is_caught(tree):
    _write(tree, "engine/v2/scoring/legacy_seam.py", "from engine.score import Scorer\n")
    _write(tree, "engine/v2/data/reader.py", "import engine.ledger\n")
    _, findings = check.run(tree, check_guide=False)
    assert {f["subject"] for f in findings if f["code"] == "ADAPTER_MODULE_UNDECLARED"} == {
        "engine/v2/scoring/legacy_seam.py", "engine/v2/data/reader.py"}


def test_guide_section_must_match_a_fresh_render(tree):
    assert "GUIDE_STALE" in codes(tree, check_guide=True)
    inv.write_guide(tree, inv.render_markdown(inv.build_document(tree)))
    assert codes(tree, check_guide=True) == []
    decl = tree / inv.DECLARATIONS
    decl.write_text(decl.read_text().replace('capability = "c"', 'capability = "changed"', 1))
    assert codes(tree, check_guide=True) == ["GUIDE_STALE"]


def test_render_lists_missing_rows_and_decisions(tmp_path):
    extra = textwrap.dedent('''\
        [[row]]
        id = "gap"
        area = "export"
        capability = "offline file"
        new = ["MISSING"]
        disposition = "missing"
        owner = "P6-5"
        notes = "no export"
        [[user_decision]]
        id = "UD-9"
        rows = ["gap"]
        question = "accept?"
        ''')
    markdown = inv.render_markdown(inv.build_document(make_tree(tmp_path, toml_extra=extra)))
    assert "| `gap` | offline file | P6-5 | no export |" in markdown
    assert "**UD-9**" in markdown


# ------------------------------------------------------------- real source


def test_real_checkout_inventory_is_complete_and_guide_current():
    document, findings = check.run()
    assert findings == []
    tally = inv.counts(document)
    assert tally["missing"] > 0 and sum(tally.values()) == len(document["rows"])
    assert all(edge["owner"] in inv.EDGE_OWNERS for edge in document["adapter_edges"])
