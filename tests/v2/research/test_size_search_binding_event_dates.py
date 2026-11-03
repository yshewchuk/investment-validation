import pytest

from experiments import size_search_binding


@pytest.mark.parametrize("value", [1, "yes", None])
def test_fold_inputs_rejects_non_bool_date_requirement_before_admission(monkeypatch, value):
    verified = []

    def require(condition):
        if not condition:
            raise ValueError("invalid event date requirement")

    monkeypatch.setattr(size_search_binding, "_require", require)
    monkeypatch.setattr(size_search_binding, "verify_installation", lambda binding: verified.append(binding))

    with pytest.raises(ValueError, match="invalid event date requirement"):
        size_search_binding.fold_inputs(None, None, None, requires_event_dates=value)

    assert verified == []