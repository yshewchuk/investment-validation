from typing import Mapping


def score_one(request, fields: Mapping):
    return _record_values(request, fields)
