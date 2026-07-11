from __future__ import annotations

import runpy
from pathlib import Path

import pytest

_SCRIPT = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "benchmark_varex_sidecars.py")
)
_parse_json_output = _SCRIPT["_parse_json_output"]
_extraction_prompt = _SCRIPT["_extraction_prompt"]
_load_summary = _SCRIPT["_load_summary"]
_prediction_complete = _SCRIPT["_prediction_complete"]


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


def test_resume_requires_matching_protocol_metadata(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(
        '{"protocol_version": 1, "model": "old", "results": {}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="protocol_version"):
        _load_summary(
            path,
            expected={"protocol_version": 2, "model": "new"},
            resume=True,
        )


def test_prediction_complete_requires_valid_successful_object(tmp_path: Path) -> None:
    prediction = tmp_path / "doc.pred.json"
    prediction.write_text('{"field": "value"}', encoding="utf-8")

    assert _prediction_complete(prediction, {"status": "ok"})
    assert not _prediction_complete(prediction, {"status": "error"})
    prediction.write_text('{"_error": "failed"}', encoding="utf-8")
    assert not _prediction_complete(prediction, {"status": "ok"})
