from __future__ import annotations

import asyncio
import json

from rag_app.pipeline.structured_extraction_executor import (
    InferenceRequest,
    InferenceResponse,
    StructuredExtractionExecutionError,
    TransientInferenceError,
    execute_nested_extraction,
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
        return InferenceResponse(
            json.dumps({"contract": {"amount": 10.5, "number": "A-1"}}), "stop"
        )

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
