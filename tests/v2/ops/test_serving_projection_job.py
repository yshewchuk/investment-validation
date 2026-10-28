from engine.v2.ops.plans import NIGHTLY_GRAPH


def test_serving_projection_graph():
    assert NIGHTLY_GRAPH["serving_projection"] == ("publication",)
