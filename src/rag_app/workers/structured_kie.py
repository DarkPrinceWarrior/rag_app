"""Bounded KIE job for the isolated structured-sidecar worker."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import logging
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from arq import Retry
from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

from rag_app.config import settings
from rag_app.pipeline.structured_artifacts import (
    ArtifactType,
    KIEArtifact,
    KIEPayload,
    artifact_content_sha256,
    canonical_artifact_bytes,
)
from rag_app.pipeline.structured_extraction_executor import (
    RetryableStructuredExtractionError,
    StructuredExecutionResult,
    StructuredExtractionExecutionError,
    execute_nested_extraction,
    execute_table_extraction,
)
from rag_app.pipeline.structured_extraction_protocol import (
    PROMPT_VERSION,
    PROTOCOL_VERSION,
    StructuredExtractionProtocolError,
    canonical_json_sha256,
    normalize_request_schema,
)
from rag_app.workers.structured_lifecycle import (
    StructuredArtifactClaim,
    build_candidate_artifact_key,
    claim_structured_artifact,
    fail_structured_artifact,
    publish_structured_artifact,
)

logger = logging.getLogger(__name__)

_EXPECTED_PROTOCOL_VERSION = f"structured-v{PROTOCOL_VERSION}"
_MAX_SOURCE_BYTES = 20 * 1024 * 1024
_MAX_IMAGE_PIXELS = 50_000_000
_MAX_IMAGE_DIMENSION = 20_000
_MAX_MODEL_ATTEMPTS = 16
_MAX_SPLIT_DEPTH = 3
_MAX_RESPONSE_BYTES = 4 * 1024 * 1024
_RETRY_DELAY_SECONDS = 30
_ALLOWED_OPTIONS = {"max_tokens", "schema_mode", "temperature"}
_IMAGE_MIME = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}


class _JobFailure(RuntimeError):
    def __init__(self, code: str, *, retryable: bool) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable


def _conservative_token_count(value: str) -> int:
    """A byte upper bound avoids silently undercounting non-Latin schemas."""

    return len(value.encode("utf-8"))


def _validate_claim(claim: StructuredArtifactClaim) -> None:
    if claim.artifact_type != ArtifactType.kie.value or claim.backend != "granite":
        raise _JobFailure("unsupported_claim", retryable=False)
    if claim.model != settings.structured_model_name:
        raise _JobFailure("model_mismatch", retryable=False)
    if claim.model_revision != settings.structured_model_revision:
        raise _JobFailure("model_revision_mismatch", retryable=False)
    if claim.prompt_version != PROMPT_VERSION:
        raise _JobFailure("prompt_version_mismatch", retryable=False)
    if claim.protocol_version != _EXPECTED_PROTOCOL_VERSION:
        raise _JobFailure("protocol_version_mismatch", retryable=False)
    if claim.schema_version != 2:
        raise _JobFailure("schema_version_mismatch", retryable=False)
    if not 1 <= claim.attempt_count <= claim.max_attempts <= settings.structured_job_max_attempts:
        raise _JobFailure("attempt_bounds_invalid", retryable=False)
    if claim.schema_sha256 != canonical_json_sha256(claim.request_schema):
        raise _JobFailure("schema_hash_mismatch", retryable=False)
    if not claim.source_key.startswith(f"{claim.document_id}/"):
        raise _JobFailure("source_key_invalid", retryable=False)


def _validated_options(value: Mapping[str, Any]) -> tuple[str, int]:
    unknown = set(value) - _ALLOWED_OPTIONS
    if unknown:
        raise _JobFailure("request_options_invalid", retryable=False)
    temperature = value.get("temperature", 0)
    if isinstance(temperature, bool) or not isinstance(temperature, int | float) or temperature != 0:
        raise _JobFailure("request_options_invalid", retryable=False)
    max_tokens = value.get("max_tokens", settings.structured_model_max_tokens)
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or not 1 <= max_tokens <= settings.structured_model_max_tokens
    ):
        raise _JobFailure("request_options_invalid", retryable=False)
    schema_mode = value.get("schema_mode", "auto")
    if schema_mode not in {"auto", "nested", "table"}:
        raise _JobFailure("request_options_invalid", retryable=False)
    return str(schema_mode), max_tokens


def _root_has_table(schema: Mapping[str, Any]) -> bool:
    normalized = normalize_request_schema(schema)
    properties = normalized.get("properties")
    if not isinstance(properties, Mapping):
        raise StructuredExtractionProtocolError("normalized schema properties are invalid")
    for child in properties.values():
        if not isinstance(child, Mapping):
            raise StructuredExtractionProtocolError("property schema is invalid")
        raw_type = child.get("type")
        is_array = raw_type == "array" or isinstance(raw_type, list) and "array" in raw_type
        items = child.get("items")
        if not is_array or not isinstance(items, Mapping):
            continue
        item_type = items.get("type")
        if item_type == "object" or isinstance(item_type, list) and "object" in item_type:
            return True
    return False


def _validated_image_data_url(payload: bytes) -> str:
    if not payload or len(payload) > _MAX_SOURCE_BYTES:
        raise _JobFailure("source_size_invalid", retryable=False)
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image_format = image.format
            width, height = image.size
            if (
                image_format not in _IMAGE_MIME
                or width <= 0
                or height <= 0
                or width > _MAX_IMAGE_DIMENSION
                or height > _MAX_IMAGE_DIMENSION
                or width * height > _MAX_IMAGE_PIXELS
            ):
                raise _JobFailure("source_image_invalid", retryable=False)
            image.verify()
    except _JobFailure:
        raise
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise _JobFailure("source_image_invalid", retryable=False) from exc
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{_IMAGE_MIME[image_format]};base64,{encoded}"


async def _execute_claim(
    claim: StructuredArtifactClaim,
    *,
    image_url: str,
    inference: Any,
) -> StructuredExecutionResult:
    schema_mode, max_tokens = _validated_options(claim.request_options)
    use_table = schema_mode == "table" or (schema_mode == "auto" and _root_has_table(claim.request_schema))
    if use_table:
        return await execute_table_extraction(
            claim.request_schema,
            image_url=image_url,
            token_counter=_conservative_token_count,
            inference=inference,
            flat_nested_max_output_tokens=max_tokens,
            table_max_output_tokens=max_tokens,
            max_attempts=_MAX_MODEL_ATTEMPTS,
            max_split_depth=_MAX_SPLIT_DEPTH,
            max_response_bytes=_MAX_RESPONSE_BYTES,
        )
    return await execute_nested_extraction(
        claim.request_schema,
        image_url=image_url,
        token_counter=_conservative_token_count,
        inference=inference,
        max_output_tokens=max_tokens,
        max_attempts=_MAX_MODEL_ATTEMPTS,
        max_split_depth=_MAX_SPLIT_DEPTH,
        max_response_bytes=_MAX_RESPONSE_BYTES,
    )


def _artifact_payload(
    claim: StructuredArtifactClaim,
    result: StructuredExecutionResult,
) -> tuple[bytes, dict[str, Any]]:
    artifact = KIEArtifact(
        schema_version=claim.schema_version,
        artifact_id=claim.artifact_id,
        document_id=claim.document_id,
        parse_revision=claim.parse_revision,
        page_idx=claim.page_idx,
        source_sha256=claim.source_sha256,
        backend=claim.backend,
        model=claim.model,
        prompt_version=claim.prompt_version,
        generated_at=datetime.now(UTC),
        artifact_type=ArtifactType.kie,
        payload=KIEPayload(
            schema_sha256=claim.schema_sha256,
            result=result.value,
        ),
    )
    payload = canonical_artifact_bytes(artifact)
    latency_s = round(
        sum(attempt.latency_s or 0.0 for attempt in result.attempts),
        3,
    )
    summary = {
        "first_pass_success": result.first_pass_success,
        "job_attempt": claim.attempt_count,
        "latency_s": latency_s,
        "model_calls": len(result.attempts),
        "model_error_calls": sum(attempt.status == "error" for attempt in result.attempts),
        "model_revision": claim.model_revision,
        "protocol_version": claim.protocol_version,
        "recovered": result.recovered,
        "request_hash": claim.request_hash,
        "schema_sha256": claim.schema_sha256,
    }
    return payload, summary


async def _record_failure(
    ctx: dict[str, Any],
    claim: StructuredArtifactClaim,
    *,
    code: str,
    retryable: bool,
) -> dict[str, str]:
    async with ctx["sessionmaker"]() as session:
        status = await fail_structured_artifact(
            session,
            claim,
            error_code=code,
            retryable=retryable,
            retry_delay_seconds=_RETRY_DELAY_SECONDS,
        )
        await session.commit()
    if status == "queued":
        raise Retry(defer=_RETRY_DELAY_SECONDS)
    if status is None:
        return {"status": "stale", "error": code}
    return {"status": status, "error": code}


async def _remove_candidate(storage: Any, key: str | None) -> None:
    if key is None:
        return
    try:
        await storage.remove_object(settings.bucket_artifacts, key)
    except Exception:
        logger.warning("structured KIE candidate cleanup failed", exc_info=True)


async def run_structured_kie(ctx: dict[str, Any], artifact_id: str) -> dict[str, Any]:
    """Claim, execute and conditionally publish one exact-schema KIE artifact."""

    if not settings.structured_extraction_enabled:
        return {"status": "disabled"}
    try:
        parsed_artifact_id = uuid.UUID(str(artifact_id))
    except (TypeError, ValueError, AttributeError):
        return {"status": "invalid", "error": "invalid_artifact_id"}

    async with ctx["sessionmaker"]() as session:
        claim = await claim_structured_artifact(
            session,
            parsed_artifact_id,
            lease_seconds=settings.structured_job_lease_s,
        )
        await session.commit()
    if claim is None:
        return {"status": "skipped"}

    storage = ctx["storage"]
    candidate_key: str | None = None
    try:
        _validate_claim(claim)
        inference = ctx.get("structured_client")
        if inference is None:
            raise _JobFailure("worker_not_ready", retryable=True)
        try:
            source = await storage.get_bytes(settings.bucket_artifacts, claim.source_key)
        except Exception as exc:
            raise _JobFailure("source_read_failed", retryable=True) from exc
        if hashlib.sha256(source).hexdigest() != claim.source_sha256:
            raise _JobFailure("source_hash_mismatch", retryable=False)
        image_url = await asyncio.to_thread(_validated_image_data_url, source)
        result = await _execute_claim(claim, image_url=image_url, inference=inference)
        payload, summary = _artifact_payload(claim, result)
        candidate_key = build_candidate_artifact_key(claim)
        try:
            await storage.put_bytes(
                settings.bucket_artifacts,
                candidate_key,
                payload,
                "application/json",
            )
        except Exception as exc:
            raise _JobFailure("artifact_write_failed", retryable=True) from exc

        async with ctx["sessionmaker"]() as session:
            published = await publish_structured_artifact(
                session,
                claim,
                artifact_key=candidate_key,
                content_sha256=artifact_content_sha256(payload),
                size_bytes=len(payload),
                summary=summary,
            )
            await session.commit()
        if not published:
            await _remove_candidate(storage, candidate_key)
            return {"status": "stale"}
        return {
            "status": "ready",
            "artifact_id": str(claim.artifact_id),
            "artifact_key": candidate_key,
        }
    except asyncio.CancelledError:
        await _remove_candidate(storage, candidate_key)
        raise
    except _JobFailure as exc:
        await _remove_candidate(storage, candidate_key)
        return await _record_failure(
            ctx,
            claim,
            code=exc.code,
            retryable=exc.retryable,
        )
    except RetryableStructuredExtractionError:
        await _remove_candidate(storage, candidate_key)
        return await _record_failure(
            ctx,
            claim,
            code="model_transient",
            retryable=True,
        )
    except StructuredExtractionExecutionError:
        await _remove_candidate(storage, candidate_key)
        return await _record_failure(
            ctx,
            claim,
            code="model_output_invalid",
            retryable=False,
        )
    except StructuredExtractionProtocolError:
        await _remove_candidate(storage, candidate_key)
        return await _record_failure(
            ctx,
            claim,
            code="schema_invalid",
            retryable=False,
        )
    except ValidationError:
        await _remove_candidate(storage, candidate_key)
        return await _record_failure(
            ctx,
            claim,
            code="artifact_invalid",
            retryable=False,
        )
    except Exception:
        logger.exception("structured KIE job failed without persisting raw model output")
        await _remove_candidate(storage, candidate_key)
        return await _record_failure(
            ctx,
            claim,
            code="internal_error",
            retryable=True,
        )
