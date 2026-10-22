"""A really admitted native cohort round-trips through the closed manifest inventory."""
# ruff: noqa: F811
from engine.v2.foundation import ArtifactStore, content_hash
from experiments import magnitude_search_shards as shards
from experiments.magnitude_search_inputs import load_inputs
from tests.v2.research.magnitude_trial_support import trial_cases
from tests.v2.research.test_magnitude_search_inputs import _manifest
from tests.v2.research.test_magnitude_search_plan import source as base_source  # noqa: F401
from tests.v2.research.test_magnitude_search_trials import case  # noqa: F401


def _list_paths(value, prefix=""):
    found = set()
    for name, item in value.items():
        if isinstance(item, list):
            found.add(prefix + name)
        elif isinstance(item, dict):
            found |= _list_paths(item, prefix + name + ".")
    return found


def test_real_admitted_cohort_lists_are_exactly_the_inventory(case, tmp_path):
    started = case.recorder.start(*trial_cases(case)[0])
    inputs = started["inputs"]
    assert _list_paths(inputs) == {member["section"] for member in shards.LIST_MEMBERS}
    store = ArtifactStore(tmp_path / "store")
    manifest = _manifest(store, inputs)
    loaded = load_inputs(store, manifest, shards.manifest_id(manifest))
    assert content_hash(loaded) == content_hash(inputs)
