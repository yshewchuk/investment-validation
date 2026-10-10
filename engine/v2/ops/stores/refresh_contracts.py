"""Dependency-free names shared by the refresh stores and their executor staging.

A leaf: it imports nothing from ``engine.v2.ops``. The names live here so
``incremental_data`` (which reads the forward-calendar document name) and
``refresh_staging`` (which writes the documents) need not import each other.
"""
from __future__ import annotations

#: One entry per refresh kind that needs a staged identity document.
REFRESH_INPUT_DOCUMENT_NAMES = {
    "incremental_refresh": "incremental_refresh_input.json",
    "computed_moves_refresh": "computed_moves_refresh_input.json",
    "forward_calendar_refresh": "forward_calendar_refresh_input.json",
}
