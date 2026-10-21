"""Test helper: publish small cohort inputs as shards and return the fit child's source dict."""
from engine.v2.foundation import ArtifactStore
from experiments.magnitude_search_inputs import publish_inputs
from experiments.magnitude_search_shards import LIST_MEMBERS


def fit_source(inputs, root):
    """Shard ``inputs`` into a store under ``root``; absent list members are published empty."""
    inputs = {**inputs}
    for member in LIST_MEMBERS:
        *parents, leaf = member["section"].split(".")
        target = inputs
        for name in parents:
            target[name] = target = {**target.get(name, {})} if isinstance(target.get(name, {}), dict) else {}
        target.setdefault(leaf, [])
    store = ArtifactStore(root)
    manifest, inputs_id = publish_inputs(store, inputs)
    return {"manifest": manifest, "inputs_id": inputs_id, "store_root": str(store.root)}
