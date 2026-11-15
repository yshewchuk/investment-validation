import pytest

from engine.v2.foundation import content_hash
from engine.v2.models.magnitude_mlp_recipe import mlp_recipe, validate_mlp_recipe


_NEW_PROFILES = [
    ("chrono15_relu128x64_l2_1", [128, 64]),
    ("chrono15_relu128x64x32_l2_1", [128, 64, 32]),
]


@pytest.mark.parametrize("profile,layers", _NEW_PROFILES)
def test_new_profiles_use_the_chrono15_budget(profile, layers):
    recipe = mlp_recipe(columns=["a", "b"], feature_dictionary_hash="sha256:" + "c" * 64,
                        profile=profile)
    params = recipe["estimator"]["params"]
    assert params["hidden_layer_sizes"] == layers
    assert params["alpha"] == 1
    assert params["max_iter"] == 400
    assert params["n_iter_no_change"] == 20
    assert recipe["nonconvergence"] == "epoch_budget_success_no_retry"
    assert recipe["requires_event_dates"] is True
    assert validate_mlp_recipe(recipe) == recipe


def test_existing_profiles_keep_their_content_hashes():
    h = "sha256:c6fe90a9d18e31e0f1c42ff7a63a5b41ba7ec18f8bcea69f172f32a0224f7ac8"
    expected = {
        "chrono15_relu32_l2_1":
            "sha256:6bd23f0aaafc675a1db0a02f6f439ed7ab0e9926c1bcc1efbf6cd476e751b107",
        "chrono15_relu64x32_l2_1":
            "sha256:cf172325e28ece5d58795bcac197218afeac16c8b23a20705d9b75fe2a245a27",
        "chrono10_relu64x32_l2_1":
            "sha256:483d13e930b43671ed92359cdd72e06cfe6c6076d84787e42e5c98a65cc2a50f",
    }
    for profile, digest in expected.items():
        assert content_hash(mlp_recipe(columns=["a", "b"], feature_dictionary_hash=h,
                                       profile=profile)) == digest
