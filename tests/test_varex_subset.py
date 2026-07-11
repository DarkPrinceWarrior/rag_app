from __future__ import annotations

import runpy
from pathlib import Path

_SCRIPT = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "download_varex_complex_subset.py")
)
_schema_complexity = _SCRIPT["_schema_complexity"]
_selection_score = _SCRIPT["_selection_score"]


def test_schema_complexity_counts_nested_and_array_fields() -> None:
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string"},
                        "amount": {"type": "number"},
                    },
                },
            },
        },
    }

    assert _schema_complexity(schema) == (3, 3, 1)


def test_selection_score_prefers_more_complex_schema() -> None:
    simple = {
        "schema": '{"type":"object","properties":{"name":{"type":"string"}}}',
        "ground_truth": '{"name":"A"}',
    }
    nested = {
        "schema": (
            '{"type":"object","properties":{"person":{"type":"object",'
            '"properties":{"name":{"type":"string"},"date":{"type":"string"}}}}}'
        ),
        "ground_truth": '{"person":{"name":"A","date":"2026-01-01"}}',
    }

    assert _selection_score(nested) > _selection_score(simple)
