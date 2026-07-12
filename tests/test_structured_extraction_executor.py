from __future__ import annotations

import asyncio
import json

import pytest

from rag_app.pipeline.structured_extraction_executor import (
    InferenceRequest,
    InferenceResponse,
    StructuredExtractionExecutionError,
    TransientInferenceError,
    execute_nested_extraction,
    execute_table_extraction,
)
from rag_app.pipeline.structured_extraction_protocol import (
    TablePredictionError,
    normalize_request_schema,
    validate_nested_prediction,
)


def _tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "contract": {
                "type": "object",
                "properties": {
                    "number": {"type": "string"},
                    "amount": {"type": "number"},
                },
            }
        },
    }


def test_valid_nested_response() -> None:
    async def inference(request: InferenceRequest) -> InferenceResponse:
        assert request.temperature == 0
        assert request.response_format == "json_object"
        return InferenceResponse(
            json.dumps({"contract": {"amount": 10.5, "number": "A-1"}}),
            "stop",
            0.2,
        )

    result = asyncio.run(
        execute_nested_extraction(
            _schema(), image_url="data:image/png;base64,AA==", token_counter=_tokens, inference=inference
        )
    )

    assert result.value == {"contract": {"amount": 10.5, "number": "A-1"}}
    assert result.first_pass_success
    assert not result.recovered
    assert len(result.attempts) == 1


def test_length_response_recovers_by_deterministic_split() -> None:
    async def inference(request: InferenceRequest) -> InferenceResponse:
        properties = request.schema["properties"]["contract"]["properties"]
        if len(properties) > 1:
            return InferenceResponse("{", "length", 1.0)
        if "number" in properties:
            value = {"contract": {"number": "A-1"}}
        else:
            value = {"contract": {"amount": 10.5}}
        return InferenceResponse(json.dumps(value), "stop", 0.1)

    result = asyncio.run(
        execute_nested_extraction(
            _schema(), image_url="data:image/png;base64,AA==", token_counter=_tokens, inference=inference
        )
    )

    assert result.value == {"contract": {"amount": 10.5, "number": "A-1"}}
    assert not result.first_pass_success
    assert result.recovered
    assert [attempt.error_code for attempt in result.attempts] == ["finish_reason", None, None]


def test_transient_failure_retries_identical_chunk_once() -> None:
    calls: list[str] = []

    async def inference(request: InferenceRequest) -> InferenceResponse:
        calls.append(request.chunk_id)
        if len(calls) == 1:
            raise TransientInferenceError("503")
        return InferenceResponse(json.dumps({"contract": {"amount": 10.5, "number": "A-1"}}), "stop")

    result = asyncio.run(
        execute_nested_extraction(
            _schema(), image_url="data:image/png;base64,AA==", token_counter=_tokens, inference=inference
        )
    )

    assert calls == ["c000", "c000"]
    assert [attempt.error_code for attempt in result.attempts] == ["transient", None]
    assert not result.first_pass_success
    assert not result.recovered


def test_terminal_invalid_single_leaf_fails_closed() -> None:
    schema = {"type": "object", "properties": {"number": {"type": "string"}}}

    async def inference(request: InferenceRequest) -> InferenceResponse:
        del request
        return InferenceResponse("not-json", "stop")

    try:
        asyncio.run(
            execute_nested_extraction(
                schema,
                image_url="data:image/png;base64,AA==",
                token_counter=_tokens,
                inference=inference,
            )
        )
    except StructuredExtractionExecutionError as exc:
        assert "terminal invalid_json" in str(exc)
    else:
        raise AssertionError("invalid single-leaf response must fail")


def test_schema_echo_and_missing_fields_are_rejected() -> None:
    normalized = normalize_request_schema(_schema())
    bad_values = [normalized, {"contract": {"number": "A-1"}}]
    for value in bad_values:
        try:
            validate_nested_prediction(value, normalized)
        except TablePredictionError:
            pass
        else:
            if value is normalized:
                # Exact schema echo has the same keys but fails at nested value types.
                raise AssertionError("schema echo must not validate as an instance")
            raise AssertionError("missing field must not validate")


def test_attempt_budget_is_hard_limit() -> None:
    async def inference(request: InferenceRequest) -> InferenceResponse:
        del request
        raise TransientInferenceError("503")

    try:
        asyncio.run(
            execute_nested_extraction(
                _schema(),
                image_url="data:image/png;base64,AA==",
                token_counter=_tokens,
                inference=inference,
                max_attempts=1,
            )
        )
    except StructuredExtractionExecutionError as exc:
        assert "attempt budget" in str(exc)
    else:
        raise AssertionError("attempt budget must stop retries")


def test_duplicate_keys_and_non_finite_json_are_rejected() -> None:
    schema = {"type": "object", "properties": {"number": {"type": "number"}}}
    for payload in ('{"number":1,"number":2}', '{"number":NaN}'):

        async def inference(request: InferenceRequest, value: str = payload) -> InferenceResponse:
            del request
            return InferenceResponse(value, "stop")

        try:
            asyncio.run(
                execute_nested_extraction(
                    schema,
                    image_url="data:image/png;base64,AA==",
                    token_counter=_tokens,
                    inference=inference,
                )
            )
        except StructuredExtractionExecutionError as exc:
            assert "terminal invalid_json" in str(exc)
        else:
            raise AssertionError("non-strict JSON must fail")


def _table_schema(*, values: int = 4, scalar: bool = True) -> dict:
    properties: dict = {
        "rows": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    **{f"value_{index}": {"type": "string"} for index in range(values)},
                },
            },
        }
    }
    if scalar:
        properties["document_number"] = {"type": "string"}
    return {"type": "object", "properties": properties}


def _table_response(request: InferenceRequest, rows: list[dict]) -> InferenceResponse:
    array_name = next(iter(request.schema["properties"]))
    fields = request.schema["properties"][array_name]["items"]["properties"]
    projected = [{field: row[field] for field in fields} for row in rows]
    return InferenceResponse(json.dumps({array_name: projected}), "stop", 0.1)


def test_table_length_recovers_by_splitting_value_fields_and_repeating_anchors() -> None:
    rows = [
        {
            "id": "A",
            "name": "Alpha",
            **{f"value_{index}": f"a{index}" for index in range(4)},
        },
        {
            "id": "B",
            "name": "Beta",
            **{f"value_{index}": f"b{index}" for index in range(4)},
        },
    ]
    requests: list[InferenceRequest] = []

    async def inference(request: InferenceRequest) -> InferenceResponse:
        requests.append(request)
        if "document_number" in request.schema["properties"]:
            return InferenceResponse('{"document_number":"D-1"}', "stop", 0.05)
        if request.chunk_id == "t000-c000":
            return InferenceResponse("{", "length", 0.2)
        child_rows = list(reversed(rows)) if request.chunk_id.endswith("s01") else rows
        return _table_response(request, child_rows)

    result = asyncio.run(
        execute_table_extraction(
            _table_schema(),
            image_url="data:image/png;base64,AA==",
            token_counter=_tokens,
            inference=inference,
            max_columns=4,
        )
    )

    assert result.value == {"document_number": "D-1", "rows": rows}
    assert result.recovered
    child_requests = [request for request in requests if "/s" in request.chunk_id]
    assert [request.chunk_id for request in child_requests] == [
        "t000-c000/s00",
        "t000-c000/s01",
    ]
    for request in child_requests:
        fields = request.schema["properties"]["rows"]["items"]["properties"]
        assert {"id", "name"} <= set(fields)
        assert len(set(fields) - {"id", "name"}) == 2


def test_invalid_table_response_recovers_by_split() -> None:
    schema = _table_schema(values=2, scalar=False)
    source = {
        "id": "A",
        "name": "Alpha",
        "value_0": "zero",
        "value_1": "one",
    }

    async def inference(request: InferenceRequest) -> InferenceResponse:
        if "/s" not in request.chunk_id:
            return InferenceResponse('{"rows":[{"id":"A"}]}', "stop")
        return _table_response(request, [source])

    result = asyncio.run(
        execute_table_extraction(
            schema,
            image_url="data:image/png;base64,AA==",
            token_counter=_tokens,
            inference=inference,
            max_columns=2,
        )
    )

    assert result.value == {"rows": [source]}
    assert result.recovered
    assert [attempt.error_code for attempt in result.attempts] == [
        "schema_mismatch",
        None,
        None,
    ]


def test_unsplittable_table_chunk_fails_closed() -> None:
    async def inference(request: InferenceRequest) -> InferenceResponse:
        del request
        return InferenceResponse("not-json", "stop")

    with pytest.raises(StructuredExtractionExecutionError, match="unsplittable table chunk"):
        asyncio.run(
            execute_table_extraction(
                _table_schema(values=1, scalar=False),
                image_url="data:image/png;base64,AA==",
                token_counter=_tokens,
                inference=inference,
            )
        )


def test_table_split_depth_and_attempt_budgets_are_hard_limits() -> None:
    async def always_truncated(request: InferenceRequest) -> InferenceResponse:
        del request
        return InferenceResponse("{", "length")

    with pytest.raises(StructuredExtractionExecutionError, match="split depth exhausted"):
        asyncio.run(
            execute_table_extraction(
                _table_schema(values=4, scalar=False),
                image_url="data:image/png;base64,AA==",
                token_counter=_tokens,
                inference=always_truncated,
                max_columns=4,
                max_split_depth=1,
            )
        )

    with pytest.raises(StructuredExtractionExecutionError, match="attempt budget"):
        asyncio.run(
            execute_table_extraction(
                _table_schema(values=4, scalar=False),
                image_url="data:image/png;base64,AA==",
                token_counter=_tokens,
                inference=always_truncated,
                max_columns=4,
                max_attempts=1,
            )
        )


def test_table_transient_failure_retries_identical_chunk_once() -> None:
    calls: list[str] = []
    row = {"id": "A", "name": "Alpha", "value_0": "zero"}

    async def inference(request: InferenceRequest) -> InferenceResponse:
        calls.append(request.chunk_id)
        if len(calls) == 1:
            raise TransientInferenceError("503")
        return _table_response(request, [row])

    result = asyncio.run(
        execute_table_extraction(
            _table_schema(values=1, scalar=False),
            image_url="data:image/png;base64,AA==",
            token_counter=_tokens,
            inference=inference,
        )
    )

    assert result.value == {"rows": [row]}
    assert calls == ["t000-c000", "t000-c000"]
    assert [attempt.error_code for attempt in result.attempts] == ["transient", None]


def test_table_failure_never_returns_partial_multi_table_result() -> None:
    schema = {
        "type": "object",
        "properties": {
            "approvals": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "status": {"type": "string"},
                    },
                },
            },
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "value": {"type": "string"},
                    },
                },
            },
        },
    }
    completed: list[str] = []

    async def inference(request: InferenceRequest) -> InferenceResponse:
        array_name = next(iter(request.schema["properties"]))
        if array_name == "approvals":
            completed.append(array_name)
            return _table_response(request, [{"id": "A", "status": "approved"}])
        return InferenceResponse("not-json", "stop")

    with pytest.raises(StructuredExtractionExecutionError, match="unsplittable table chunk"):
        asyncio.run(
            execute_table_extraction(
                schema,
                image_url="data:image/png;base64,AA==",
                token_counter=_tokens,
                inference=inference,
            )
        )
    assert completed == ["approvals"]
