import pytest

from engine.v2.ops.native_registration import (
    allows_hgb_axes,
    allows_lead_selector,
    allows_neural,
    allows_size_family,
    allows_workers,
    qualified,
    request_version,
    uses_extended_neural_profiles,
    uses_v3_compiler,
)


ACTUAL_COLUMNS = {
    "workers": allows_workers,
    "qualified_request_envelope": qualified,
    "prepare_plan": qualified,
    "read_plan": qualified,
    "native_registration": qualified,
    "v2_cli_containment": qualified,
    "v3_compiler_dispatch": uses_v3_compiler,
    "lead_selector": allows_lead_selector,
    "expanded_hgb_axes": allows_hgb_axes,
    "fixed_size_family": allows_size_family,
    "extra_panel_columns": allows_size_family,
    "neural_profiles": allows_neural,
    "v7_neural_profiles": uses_extended_neural_profiles,
}

EXPECTED_TRUE_VERSIONS = {
    "workers": range(2, 8),
    "qualified_request_envelope": range(2, 8),
    "prepare_plan": range(2, 8),
    "read_plan": range(2, 8),
    "native_registration": range(2, 8),
    "v2_cli_containment": range(2, 8),
    "v3_compiler_dispatch": range(3, 8),
    "lead_selector": range(2, 8),
    "expanded_hgb_axes": range(4, 8),
    "fixed_size_family": range(5, 8),
    "extra_panel_columns": range(5, 8),
    "neural_profiles": range(6, 8),
    "v7_neural_profiles": range(7, 8),
}

SCHEMAS = [
    "regression_search_request.v1",
    "magnitude_search_request.v1",
    "magnitude_search_request.v02",
    "magnitude_search_request.v2",
    "magnitude_search_request.v3",
    "magnitude_search_request.v4",
    "magnitude_search_request.v5",
    "magnitude_search_request.v6",
    "magnitude_search_request.v7",
    "magnitude_search_request.v99",
    "garbage",
    None,
    123,
    [],
]

EXPECTED_VERSIONS = [None, 1, None, 2, 3, 4, 5, 6, 7, 99, None, None, None, None]


@pytest.mark.parametrize("schema,expected", list(zip(SCHEMAS, EXPECTED_VERSIONS)))
def test_magnitude_search_request_schema(schema, expected):
    version = request_version(schema)
    assert version == expected
    for column, actual in ACTUAL_COLUMNS.items():
        expected_cell = expected is not None and expected in EXPECTED_TRUE_VERSIONS[column]
        assert bool(actual(version)) == expected_cell, f"{schema}:{column}"
