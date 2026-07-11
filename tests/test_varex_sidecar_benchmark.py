from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

_SCRIPT = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "benchmark_varex_sidecars.py"))
_parse_json_output = _SCRIPT["_parse_json_output"]
_extraction_prompt = _SCRIPT["_extraction_prompt"]
_load_summary = _SCRIPT["_load_summary"]
_nuextract_contract = _SCRIPT["_nuextract_contract"]
_prediction_complete = _SCRIPT["_prediction_complete"]
_request_payload = _SCRIPT["_request_payload"]
_select_pages = _SCRIPT["_select_pages"]


def test_parse_json_output_accepts_plain_and_fenced_objects() -> None:
    assert _parse_json_output('{"value": 3}') == {"value": 3}
    assert _parse_json_output('```json\n{"value": 3}\n```') == {"value": 3}


def test_parse_json_output_rejects_non_object() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        _parse_json_output('[{"value": 3}]')


def test_extraction_prompt_requires_instance_not_schema_echo() -> None:
    prompt = _extraction_prompt({"type": "object", "properties": {"id": {"type": "string"}}})

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

    assert _prediction_complete(prediction, {"status": "ok", "finish_reason": "stop"})
    assert not _prediction_complete(prediction, {"status": "error"})
    prediction.write_text('{"_error": "failed"}', encoding="utf-8")
    assert not _prediction_complete(prediction, {"status": "ok", "finish_reason": "stop"})


def test_nuextract_contract_rejects_dropped_schema_and_keeps_descriptions() -> None:
    schema = {"type": "object", "properties": {"pressure": {"type": "number"}}}

    contract = _nuextract_contract(
        schema,
        converter=lambda value, **kwargs: ({"pressure": "number"}, []),
        description_getter=lambda value: "$.pressure: Design pressure",
    )

    assert json.loads(contract["template"]) == {"pressure": "number"}
    assert "$.pressure: Design pressure" in contract["instructions"]
    with pytest.raises(ValueError, match="dropped branches"):
        _nuextract_contract(
            schema,
            converter=lambda value, **kwargs: ({}, ["$.pressure"]),
            description_getter=lambda value: "$.pressure: Design pressure",
        )


def test_nuextract_request_uses_template_kwargs_without_text_prompt(monkeypatch) -> None:
    monkeypatch.setitem(
        _request_payload.__globals__,
        "_nuextract_contract",
        lambda schema: {"template": '{"pressure":"number"}', "instructions": "pressure"},
    )

    payload = _request_payload(
        model="nuextract3-pinned",
        image_b64="abc",
        schema={"type": "object"},
        max_tokens=8192,
        profile="nuextract3",
    )

    assert payload["messages"][0]["content"] == [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,abc", "detail": "high"},
        }
    ]
    assert payload["chat_template_kwargs"] == {
        "template": '{"pressure":"number"}',
        "instructions": "pressure",
        "enable_thinking": False,
    }


def test_limit_per_split_keeps_deterministic_balanced_smoke() -> None:
    pages = [
        {"doc_id": "n1", "split": "Nested"},
        {"doc_id": "n2", "split": "Nested"},
        {"doc_id": "t1", "split": "Table"},
        {"doc_id": "n3", "split": "Nested"},
        {"doc_id": "t2", "split": "Table"},
    ]

    assert [page["doc_id"] for page in _select_pages(pages, 1)] == ["n1", "t1"]
    assert _select_pages(pages, 0) is pages
