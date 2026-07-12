"""Bounded executor Flat/Nested JSON Schema extraction без I/O и model binding."""

from __future__ import annotations

import json
import math
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from rag_app.pipeline.structured_extraction_protocol import (
    DEFAULT_MAX_LEAVES,
    DEFAULT_MAX_SCHEMA_TOKENS,
    PROMPT_VERSION,
    SchemaChunk,
    StructuredExtractionProtocolError,
    build_varex_prompt,
    canonical_json_bytes,
    chunk_nested_schema,
    is_schema_echo,
    normalize_request_schema,
    validate_nested_prediction,
)


class StructuredExtractionExecutionError(RuntimeError):
    """Model attempts exhausted without a fully validated result."""


class TransientInferenceError(RuntimeError):
    """Retryable transport/server failure (408/429/5xx)."""


@dataclass(frozen=True)
class InferenceRequest:
    chunk_id: str
    prompt: str
    schema: dict[str, Any]
    image_url: str
    max_tokens: int
    temperature: float = 0.0
    response_format: str = "json_object"


@dataclass(frozen=True)
class InferenceResponse:
    content: str
    finish_reason: str
    latency_s: float | None = None


class StructuredInference(Protocol):
    def __call__(self, request: InferenceRequest) -> Awaitable[InferenceResponse]: ...


@dataclass(frozen=True)
class ModelAttempt:
    chunk_id: str
    attempt: int
    split_depth: int
    status: str
    error_code: str | None
    finish_reason: str | None
    latency_s: float | None
    raw_response: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class StructuredExecutionResult:
    value: dict[str, Any]
    attempts: tuple[ModelAttempt, ...]
    first_pass_success: bool
    recovered: bool
    prompt_version: str = PROMPT_VERSION


class _ResponseError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_response(
    response: InferenceResponse,
    *,
    schema: Mapping[str, Any],
    max_response_bytes: int,
) -> dict[str, Any]:
    if response.finish_reason != "stop":
        raise _ResponseError(
            "finish_reason",
            f"model finish_reason must be stop, got {response.finish_reason!r}",
        )
    encoded = response.content.encode("utf-8")
    if len(encoded) > max_response_bytes:
        raise _ResponseError("response_too_large", "model response exceeds byte limit")
    try:
        value = json.loads(
            response.content,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise _ResponseError("invalid_json", "model response is not strict JSON") from exc
    if not isinstance(value, dict):
        raise _ResponseError("not_object", "model response must be a JSON object")
    if is_schema_echo(value, expected_schema=schema):
        raise _ResponseError("schema_echo", "model returned schema instead of values")
    try:
        validate_nested_prediction(value, schema, max_value_bytes=max_response_bytes)
    except StructuredExtractionProtocolError as exc:
        raise _ResponseError("schema_mismatch", "model response does not match schema") from exc
    return value


def _merge_values(left: Any, right: Any, *, path: str = "") -> Any:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        merged = {key: value for key, value in left.items()}
        for key, value in right.items():
            child_path = f"{path}/{key}"
            if key in merged:
                merged[key] = _merge_values(merged[key], value, path=child_path)
            else:
                merged[key] = value
        return merged
    if canonical_json_bytes(left) != canonical_json_bytes(right):
        raise StructuredExtractionExecutionError(f"conflicting chunk values at {path or '/'}")
    return left


async def execute_nested_extraction(
    schema: Mapping[str, Any],
    *,
    image_url: str,
    token_counter: Callable[[str], int],
    inference: StructuredInference,
    max_leaves: int = DEFAULT_MAX_LEAVES,
    max_schema_tokens: int = DEFAULT_MAX_SCHEMA_TOKENS,
    max_output_tokens: int = 4096,
    max_attempts: int = 24,
    max_split_depth: int = 3,
    max_response_bytes: int = 4 * 1024 * 1024,
) -> StructuredExecutionResult:
    """Execute bounded Flat/Nested extraction with one transport retry and split recovery."""

    if not image_url:
        raise ValueError("image_url must be non-empty")
    if max_output_tokens <= 0 or max_attempts <= 0 or max_response_bytes <= 0:
        raise ValueError("output, attempt, and response limits must be positive")
    if max_split_depth < 0:
        raise ValueError("max_split_depth must be non-negative")

    normalized = normalize_request_schema(schema)
    initial_chunks = chunk_nested_schema(
        normalized,
        token_counter=token_counter,
        max_leaves=max_leaves,
        max_schema_tokens=max_schema_tokens,
    )
    if not initial_chunks:
        raise StructuredExtractionExecutionError("schema has no extractable leaves")

    attempts: list[ModelAttempt] = []
    split_used = False

    async def call_model(chunk: SchemaChunk, *, logical_id: str, depth: int) -> dict[str, Any]:
        nonlocal split_used
        response_error: _ResponseError | None = None
        for transport_attempt in (1, 2):
            if len(attempts) >= max_attempts:
                raise StructuredExtractionExecutionError("model attempt budget exhausted")
            request = InferenceRequest(
                chunk_id=logical_id,
                prompt=build_varex_prompt(chunk.schema),
                schema=chunk.schema,
                image_url=image_url,
                max_tokens=max_output_tokens,
            )
            try:
                response = await inference(request)
            except TransientInferenceError:
                attempts.append(
                    ModelAttempt(
                        chunk_id=logical_id,
                        attempt=transport_attempt,
                        split_depth=depth,
                        status="error",
                        error_code="transient",
                        finish_reason=None,
                        latency_s=None,
                    )
                )
                if transport_attempt == 1:
                    continue
                raise StructuredExtractionExecutionError(
                    f"transient inference retry exhausted for {logical_id}"
                ) from None
            except Exception as exc:
                attempts.append(
                    ModelAttempt(
                        chunk_id=logical_id,
                        attempt=transport_attempt,
                        split_depth=depth,
                        status="error",
                        error_code=type(exc).__name__,
                        finish_reason=None,
                        latency_s=None,
                    )
                )
                raise StructuredExtractionExecutionError(
                    f"non-retryable inference failure for {logical_id}: {type(exc).__name__}"
                ) from exc
            if response.latency_s is not None and (
                not math.isfinite(response.latency_s) or response.latency_s < 0
            ):
                attempts.append(
                    ModelAttempt(
                        chunk_id=logical_id,
                        attempt=transport_attempt,
                        split_depth=depth,
                        status="error",
                        error_code="invalid_latency",
                        finish_reason=response.finish_reason,
                        latency_s=None,
                        raw_response=response.content,
                    )
                )
                raise StructuredExtractionExecutionError(
                    "model latency must be finite and non-negative"
                )
            try:
                parsed = _parse_response(
                    response,
                    schema=chunk.schema,
                    max_response_bytes=max_response_bytes,
                )
            except _ResponseError as exc:
                attempts.append(
                    ModelAttempt(
                        chunk_id=logical_id,
                        attempt=transport_attempt,
                        split_depth=depth,
                        status="error",
                        error_code=exc.code,
                        finish_reason=response.finish_reason,
                        latency_s=response.latency_s,
                        raw_response=response.content,
                    )
                )
                response_error = exc
                break
            attempts.append(
                ModelAttempt(
                    chunk_id=logical_id,
                    attempt=transport_attempt,
                    split_depth=depth,
                    status="ok",
                    error_code=None,
                    finish_reason=response.finish_reason,
                    latency_s=response.latency_s,
                    raw_response=response.content,
                )
            )
            return parsed

        if response_error is None:
            raise StructuredExtractionExecutionError(f"model failed without response for {logical_id}")
        if depth >= max_split_depth or chunk.leaf_count <= 1:
            raise StructuredExtractionExecutionError(
                f"terminal {response_error.code} for {logical_id}"
            ) from response_error
        child_max_leaves = max(1, (chunk.leaf_count + 1) // 2)
        children = chunk_nested_schema(
            chunk.schema,
            token_counter=token_counter,
            max_leaves=child_max_leaves,
            max_schema_tokens=max_schema_tokens,
        )
        if len(children) <= 1:
            raise StructuredExtractionExecutionError(
                f"cannot split invalid response for {logical_id}"
            ) from response_error
        split_used = True
        merged: dict[str, Any] = {}
        for index, child in enumerate(children):
            value = await call_model(
                child,
                logical_id=f"{logical_id}/s{index:02d}",
                depth=depth + 1,
            )
            merged = _merge_values(merged, value)
        validate_nested_prediction(merged, chunk.schema, max_value_bytes=max_response_bytes)
        return merged

    merged: dict[str, Any] = {}
    for chunk in initial_chunks:
        value = await call_model(chunk, logical_id=chunk.chunk_id, depth=0)
        merged = _merge_values(merged, value)
    validate_nested_prediction(merged, normalized, max_value_bytes=max_response_bytes)
    had_error = any(attempt.status == "error" for attempt in attempts)
    return StructuredExecutionResult(
        value=merged,
        attempts=tuple(attempts),
        first_pass_success=not had_error and not split_used,
        recovered=had_error and split_used,
    )
