from checks.rearchitecture_phase4_gate import REQUIRED, check

INCOMPLETE = "population is incomplete"


def _passing_evidence(population):
    return {
        "schema_version": "phase4_acceptance.v1.0",
        "population": population,
        "status": "PASS",
        "native_parity": {"complete": True},
        "factory_parity": {"complete": True},
        "subjects": {subject: {"status": "PASS", "controls": {}} for subject in REQUIRED},
    }


def _findings(population):
    return check(_passing_evidence(population))["findings"]


def test_missing_population_counts_are_incomplete():
    assert INCOMPLETE in _findings({})
    assert INCOMPLETE in _findings({"note": "x"})


def test_zero_and_unequal_population_counts_are_incomplete():
    assert INCOMPLETE in _findings({"expected": 0, "supported": 0, "compared": 0})
    assert INCOMPLETE in _findings({"expected": 20, "supported": 20, "compared": 19})


def test_boolean_population_counts_are_not_counts():
    assert INCOMPLETE in _findings({"expected": True, "supported": True, "compared": True})


def test_complete_positive_population_is_accepted():
    result = check(_passing_evidence({"expected": 20, "supported": 20, "compared": 20}))
    assert INCOMPLETE not in result["findings"]
    assert result["ok"] is True
