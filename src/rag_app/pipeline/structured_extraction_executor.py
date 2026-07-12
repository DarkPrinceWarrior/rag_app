"""Bounded Flat/Nested/Table JSON Schema executor без I/O и model binding."""

from __future__ import annotations

import json
import math
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from rag_app.pipeline.structured_extraction_protocol import (
    DEFAULT_MAX_LEAVES,
    DEFAULT_MAX_SCHEMA_TOKENS,
    DEFAULT_TABLE_MAX_ANCHORS,
    DEFAULT_TABLE_MAX_CELL_BYTES,
    DEFAULT_TABLE_MAX_COLUMNS,
    DEFAULT_TABLE_MAX_ROWS,
    PROMPT_VERSION,
    SchemaChunk,
    StructuredExtractionProtocolError,
    TableArrayPlan,
    TableSchemaChunk,
    build_varex_prompt,
    canonical_json_bytes,
    chunk_nested_schema,
    is_schema_echo,
    merge_table_array_predictions,
    normalize_request_schema,
    split_table_schema,
    validate_nested_prediction,
)


class StructuredExtractionExecutionError(RuntimeError):
    """Model attempts exhausted without a fully validated result."""


class RetryableStructuredExtractionError(StructuredExtractionExecutionError):
    """All bounded attempts failed only because the model transport was unavailable."""


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


def _parse_json_response(
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
    return value


def _parse_response(
    response: InferenceResponse,
    *,
    schema: Mapping[str, Any],
    max_response_bytes: int,
) -> dict[str, Any]:
    value = _parse_json_response(
        response,
        schema=schema,
        max_response_bytes=max_response_bytes,
    )
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
                raise RetryableStructuredExtractionError(
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
                raise StructuredExtractionExecutionError("model latency must be finite and non-negative")
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


def _table_property_name(chunk: TableSchemaChunk) -> str:
    properties = chunk.schema.get("properties")
    if not isinstance(properties, Mapping) or len(properties) != 1:
        raise StructuredExtractionExecutionError(
            f"table chunk {chunk.chunk_id} must contain one root property"
        )
    return next(iter(properties))


def _single_chunk_table_plan(
    chunk: TableSchemaChunk,
    *,
    max_rows: int,
    max_cell_bytes: int,
) -> TableArrayPlan:
    return TableArrayPlan(
        array_path=chunk.array_path,
        fields=(*chunk.anchor_fields, *chunk.value_fields),
        anchor_fields=chunk.anchor_fields,
        max_rows=max_rows,
        max_cell_bytes=max_cell_bytes,
        chunks=(chunk,),
    )


def _parse_table_response(
    response: InferenceResponse,
    *,
    chunk: TableSchemaChunk,
    max_rows: int,
    max_cell_bytes: int,
    max_response_bytes: int,
) -> dict[str, Any]:
    value = _parse_json_response(
        response,
        schema=chunk.schema,
        max_response_bytes=max_response_bytes,
    )
    try:
        merge_table_array_predictions(
            _single_chunk_table_plan(
                chunk,
                max_rows=max_rows,
                max_cell_bytes=max_cell_bytes,
            ),
            {chunk.chunk_id: value},
        )
    except StructuredExtractionProtocolError as exc:
        raise _ResponseError("schema_mismatch", "model response does not match table schema") from exc
    return value


def _split_table_chunk(
    chunk: TableSchemaChunk,
    *,
    token_counter: Callable[[str], int],
    max_rows: int,
    max_cell_bytes: int,
    max_schema_tokens: int,
) -> tuple[TableSchemaChunk, ...]:
    if not chunk.anchor_fields or len(chunk.value_fields) <= 1:
        return ()
    child_max_columns = max(1, (len(chunk.value_fields) + 1) // 2)
    child_plan = split_table_schema(
        chunk.schema,
        token_counter=token_counter,
        max_columns=child_max_columns,
        max_anchors=len(chunk.anchor_fields),
        max_rows=max_rows,
        max_cell_bytes=max_cell_bytes,
        max_schema_tokens=max_schema_tokens,
    )
    if child_plan.scalar_chunks or len(child_plan.tables) != 1:
        return ()
    children = child_plan.tables[0].chunks
    if len(children) <= 1:
        return ()
    if any(
        child.array_path != chunk.array_path or child.anchor_fields != chunk.anchor_fields
        for child in children
    ):
        return ()
    child_fields = tuple(field for child in children for field in child.value_fields)
    if child_fields != chunk.value_fields:
        return ()
    return children


def _scalar_schema(
    normalized: Mapping[str, Any],
    table_plans: tuple[TableArrayPlan, ...],
) -> dict[str, Any] | None:
    properties = normalized.get("properties")
    if not isinstance(properties, Mapping):
        raise StructuredExtractionExecutionError("normalized schema properties are invalid")
    table_names = {_table_property_name(table.chunks[0]) for table in table_plans if table.chunks}
    scalar_properties = {key: value for key, value in properties.items() if key not in table_names}
    if not scalar_properties:
        return None
    result = {key: value for key, value in normalized.items() if key not in {"properties", "required"}}
    result["properties"] = scalar_properties
    result["required"] = sorted(scalar_properties)
    return result


def _validate_complete_table_result(
    value: Mapping[str, Any],
    *,
    normalized: Mapping[str, Any],
    scalar_schema: Mapping[str, Any] | None,
    table_plans: tuple[TableArrayPlan, ...],
    max_response_bytes: int,
) -> None:
    try:
        if len(canonical_json_bytes(value)) > max_response_bytes:
            raise StructuredExtractionExecutionError("merged extraction exceeds byte limit")
    except (TypeError, ValueError, RecursionError) as exc:
        raise StructuredExtractionExecutionError("merged extraction is not canonical JSON") from exc

    properties = normalized.get("properties")
    if not isinstance(properties, Mapping) or set(value) != set(properties):
        raise StructuredExtractionExecutionError("merged extraction root keys do not match schema")

    if scalar_schema is not None:
        scalar_properties = scalar_schema.get("properties")
        if not isinstance(scalar_properties, Mapping):
            raise StructuredExtractionExecutionError("scalar schema properties are invalid")
        scalar_value = {key: value[key] for key in scalar_properties}
        try:
            validate_nested_prediction(
                scalar_value,
                scalar_schema,
                max_value_bytes=max_response_bytes,
            )
        except StructuredExtractionProtocolError as exc:
            raise StructuredExtractionExecutionError(
                "merged scalar extraction does not match schema"
            ) from exc

    for table in table_plans:
        if not table.chunks:
            raise StructuredExtractionExecutionError("runtime table plan has no chunks")
        array_name = _table_property_name(table.chunks[0])
        rows = value[array_name]
        predictions: dict[str, Any] = {}
        try:
            for chunk in table.chunks:
                fields = (*chunk.anchor_fields, *chunk.value_fields)
                projected_rows = (
                    None if rows is None else [{field: row[field] for field in fields} for row in rows]
                )
                predictions[chunk.chunk_id] = {array_name: projected_rows}
            rebuilt = merge_table_array_predictions(table, predictions)
        except (KeyError, TypeError, StructuredExtractionProtocolError) as exc:
            raise StructuredExtractionExecutionError(
                f"merged table extraction does not match schema at {table.array_path}"
            ) from exc
        if canonical_json_bytes(rebuilt) != canonical_json_bytes({array_name: rows}):
            raise StructuredExtractionExecutionError(
                f"merged table extraction is unstable at {table.array_path}"
            )


async def execute_table_extraction(
    schema: Mapping[str, Any],
    *,
    image_url: str,
    token_counter: Callable[[str], int],
    inference: StructuredInference,
    max_columns: int = DEFAULT_TABLE_MAX_COLUMNS,
    max_anchors: int = DEFAULT_TABLE_MAX_ANCHORS,
    max_rows: int = DEFAULT_TABLE_MAX_ROWS,
    max_cell_bytes: int = DEFAULT_TABLE_MAX_CELL_BYTES,
    max_leaves: int = DEFAULT_MAX_LEAVES,
    max_schema_tokens: int = DEFAULT_MAX_SCHEMA_TOKENS,
    flat_nested_max_output_tokens: int = 4096,
    table_max_output_tokens: int = 8192,
    max_attempts: int = 48,
    max_split_depth: int = 3,
    max_response_bytes: int = 4 * 1024 * 1024,
) -> StructuredExecutionResult:
    """Execute scalar and root table extraction with bounded split recovery."""

    if not image_url:
        raise ValueError("image_url must be non-empty")
    if any(
        limit <= 0
        for limit in (
            flat_nested_max_output_tokens,
            table_max_output_tokens,
            max_attempts,
            max_response_bytes,
        )
    ):
        raise ValueError("output, attempt, and response limits must be positive")
    if max_split_depth < 0:
        raise ValueError("max_split_depth must be non-negative")

    normalized = normalize_request_schema(schema)
    plan = split_table_schema(
        normalized,
        token_counter=token_counter,
        max_columns=max_columns,
        max_anchors=max_anchors,
        max_rows=max_rows,
        max_cell_bytes=max_cell_bytes,
        max_leaves=max_leaves,
        max_schema_tokens=max_schema_tokens,
    )
    scalar_schema = _scalar_schema(normalized, plan.tables)
    attempts: list[ModelAttempt] = []
    merged: dict[str, Any] = {}
    split_used = False

    if scalar_schema is not None:
        scalar_result = await execute_nested_extraction(
            scalar_schema,
            image_url=image_url,
            token_counter=token_counter,
            inference=inference,
            max_leaves=max_leaves,
            max_schema_tokens=max_schema_tokens,
            max_output_tokens=flat_nested_max_output_tokens,
            max_attempts=max_attempts,
            max_split_depth=max_split_depth,
            max_response_bytes=max_response_bytes,
        )
        attempts.extend(scalar_result.attempts)
        merged = _merge_values(merged, scalar_result.value)
        split_used = scalar_result.recovered

    async def call_table_chunk(
        chunk: TableSchemaChunk,
        *,
        logical_id: str,
        depth: int,
        table: TableArrayPlan,
    ) -> tuple[tuple[TableSchemaChunk, ...], dict[str, Any]]:
        nonlocal split_used
        runtime_chunk = replace(chunk, chunk_id=logical_id)
        response_error: _ResponseError | None = None
        for transport_attempt in (1, 2):
            if len(attempts) >= max_attempts:
                raise StructuredExtractionExecutionError("model attempt budget exhausted")
            request = InferenceRequest(
                chunk_id=logical_id,
                prompt=build_varex_prompt(runtime_chunk.schema),
                schema=runtime_chunk.schema,
                image_url=image_url,
                max_tokens=table_max_output_tokens,
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
                raise RetryableStructuredExtractionError(
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
                raise StructuredExtractionExecutionError("model latency must be finite and non-negative")
            try:
                parsed = _parse_table_response(
                    response,
                    chunk=runtime_chunk,
                    max_rows=table.max_rows,
                    max_cell_bytes=table.max_cell_bytes,
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
            return (runtime_chunk,), {logical_id: parsed}

        if response_error is None:
            raise StructuredExtractionExecutionError(f"model failed without response for {logical_id}")
        if depth >= max_split_depth:
            raise StructuredExtractionExecutionError(
                f"terminal {response_error.code} for {logical_id}: split depth exhausted"
            ) from response_error
        try:
            children = _split_table_chunk(
                runtime_chunk,
                token_counter=token_counter,
                max_rows=table.max_rows,
                max_cell_bytes=table.max_cell_bytes,
                max_schema_tokens=max_schema_tokens,
            )
        except StructuredExtractionProtocolError as exc:
            raise StructuredExtractionExecutionError(
                f"cannot split invalid table chunk {logical_id}"
            ) from exc
        if not children:
            raise StructuredExtractionExecutionError(
                f"terminal {response_error.code} for unsplittable table chunk {logical_id}"
            ) from response_error
        split_used = True
        runtime_children: list[TableSchemaChunk] = []
        child_predictions: dict[str, Any] = {}
        for index, child in enumerate(children):
            child_chunks, predictions = await call_table_chunk(
                child,
                logical_id=f"{logical_id}/s{index:02d}",
                depth=depth + 1,
                table=table,
            )
            runtime_children.extend(child_chunks)
            overlap = set(child_predictions) & set(predictions)
            if overlap:
                raise StructuredExtractionExecutionError(f"duplicate runtime table chunks: {sorted(overlap)}")
            child_predictions.update(predictions)
        return tuple(runtime_children), child_predictions

    runtime_tables: list[TableArrayPlan] = []
    for table in plan.tables:
        runtime_chunks: list[TableSchemaChunk] = []
        predictions: dict[str, Any] = {}
        for chunk in table.chunks:
            chunks, chunk_predictions = await call_table_chunk(
                chunk,
                logical_id=chunk.chunk_id,
                depth=0,
                table=table,
            )
            runtime_chunks.extend(chunks)
            overlap = set(predictions) & set(chunk_predictions)
            if overlap:
                raise StructuredExtractionExecutionError(f"duplicate runtime table chunks: {sorted(overlap)}")
            predictions.update(chunk_predictions)
        runtime_table = replace(table, chunks=tuple(runtime_chunks))
        try:
            table_value = merge_table_array_predictions(runtime_table, predictions)
        except StructuredExtractionProtocolError as exc:
            raise StructuredExtractionExecutionError(f"table merge failed at {table.array_path}") from exc
        merged = _merge_values(merged, table_value)
        runtime_tables.append(runtime_table)

    _validate_complete_table_result(
        merged,
        normalized=normalized,
        scalar_schema=scalar_schema,
        table_plans=tuple(runtime_tables),
        max_response_bytes=max_response_bytes,
    )
    had_error = any(attempt.status == "error" for attempt in attempts)
    return StructuredExecutionResult(
        value=merged,
        attempts=tuple(attempts),
        first_pass_success=not had_error and not split_used,
        recovered=had_error and split_used,
    )
