"""Regression test for native sidecar resource/thread identity wiring."""

from engine.v2.foundation import content_hash
from engine.v2.ops import nightly
from engine.v2.ops.fingerprints import environment_identity
from engine.v2.ops.profiles import DEFAULT_POLICY, profile_named


def test_native_sidecar_thread_identity():
    cases = (
        ("native_score_batch", "projection"),
        ("native_parity", "validation"),
    )
    for kind, expected_resource_class in cases:
        identity = nightly._sidecar_runtime_identity(kind)
        resource_class = identity["resource_class"]
        environment_ref = identity["environment_ref"]
        assert resource_class == expected_resource_class

        profile = profile_named(DEFAULT_POLICY, resource_class)
        thread_count = profile.thread_count or profile.cpu_count

        assert environment_ref == content_hash(environment_identity(thread_count))
