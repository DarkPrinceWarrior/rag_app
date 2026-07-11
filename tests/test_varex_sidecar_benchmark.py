from __future__ import annotations

import runpy
from pathlib import Path

import pytest

_SCRIPT = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "benchmark_varex_sidecars.py")
)
_parse_json_output = _SCRIPT["_parse_json_output"]
_extraction_prompt = _SCRIPT["_extraction_prompt"]


def test_parse_json_output_accepts_plain_and_fenced_objects() -> None:
    assert _parse_json_output('{"value": 3}') == {"value": 3}
    assert _parse_json_output('```json\n{"value": 3}\n```') == {"value": 3}


def test_parse_json_output_rejects_non_object() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        _parse_json_output('[{"value": 3}]')


def test_extraction_prompt_requires_instance_not_schema_echo() -> None:
    prompt = _extraction_prompt(
        {"type": "object", "properties": {"id": {"type": "string"}}}
    )

    assert "Return ONLY valid JSON" in prompt
    assert "instance of the JSON" in prompt
    assert '"id"' in prompt
