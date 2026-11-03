"""Closed prediction-only policy for synthetic operator boundary tests."""


def request():
    return {"schema_version": "distribution_prediction_request.v1", "max_events": 50,
        "limits": {"parent": {"memory_bytes": 4 << 30, "cpu_seconds": 60, "wall_seconds": 60},
            "job_wall_seconds": 60},
        "uncertainty": {"seed": 0, "resamples": 4, "block_days": 7,
            "block_origin": "2023-01-01", "confidence": .95}}
