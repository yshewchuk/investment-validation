from checks.rearchitecture_phase3_browser import _source_population_keys


def test_source_population_keys_requires_unique_nonempty_saved_keys():
    assert _source_population_keys({"expected_population": ["AAA|STR-THRU|2026-09-10"]}) == {
        "AAA|STR-THRU|2026-09-10"}
    for document in ({}, {"expected_population": []},
                     {"expected_population": ["AAA|STR-THRU|2026-09-10", "AAA|STR-THRU|2026-09-10"]}):
        try:
            _source_population_keys(document)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid source population was accepted")
