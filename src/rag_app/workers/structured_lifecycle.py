"""Durable database lifecycle for isolated structured-extraction jobs.

The helpers only persist and transition requests. They do not call a model,
enqueue ARQ jobs, or enable the sidecar in production.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import exists, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.db.models import Document, DocumentStructuredArtifact
from rag_app.pipeline.structured_extraction_protocol import canonical_json_sha256

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ERROR_CODE_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
_ARTIFACT_TYPES = {"kie", "chart", "diagram"}
_MAX_SUMMARY_BYTES = 64 * 1024


@dataclass(frozen=True)
class StructuredArtifactRequest:
    document_id: uuid.UUID
    parse_revision: int
    page_idx: int
    artifact_type: str
    backend: str
    model: str
    model_revision: str
    prompt_version: str
    protocol_version: str
    schema_version: int
    request_hash: str
    request_schema: dict[str, Any]
    schema_sha256: str
    request_options: dict[str, Any]
    source_key: str
    source_sha256: str
    max_attempts: int = 3


@dataclass(frozen=True)
class StructuredArtifactClaim:
    artifact_id: uuid.UUID
    document_id: uuid.UUID
    parse_revision: int
    page_idx: int
    artifact_type: str
    backend: str
    model: str
    model_revision: str
    prompt_version: str
    protocol_version: str
    schema_version: int
    request_hash: str
    request_schema: dict[str, Any]
    schema_sha256: str
    request_options: dict[str, Any]
    source_key: str
    source_sha256: str
    attempt_count: int
    max_attempts: int
    claim_token: uuid.UUID


@dataclass(frozen=True)
class StructuredDispatchCandidate:
    artifact_id: uuid.UUID
    attempt_number: int


@dataclass(frozen=True)
class StructuredSweepResult:
    superseded_ids: tuple[uuid.UUID, ...]
    requeued_ids: tuple[uuid.UUID, ...]
    exhausted_ids: tuple[uuid.UUID, ...]
    dispatch: tuple[StructuredDispatchCandidate, ...]


def _aware_utc(value: datetime | None) -> datetime:
    result = value or datetime.now(UTC)
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return result


def _current_revision_exists():
    return exists(
        select(1).where(
            Document.id == DocumentStructuredArtifact.document_id,
            Document.parse_revision == DocumentStructuredArtifact.parse_revision,
        )
    )


def _validate_request(request: StructuredArtifactRequest) -> None:
    if request.parse_revision < 0 or request.page_idx < 0:
        raise ValueError("parse_revision and page_idx must be non-negative")
    if request.artifact_type not in _ARTIFACT_TYPES:
        raise ValueError("unsupported artifact_type")
    if not all(
        (
            request.backend,
            request.model,
            request.model_revision,
            request.prompt_version,
            request.protocol_version,
            request.source_key,
        )
    ):
        raise ValueError("request identity fields must be non-empty")
    if request.schema_version < 1:
        raise ValueError("schema_version must be positive")
    if not 1 <= request.max_attempts <= 10:
        raise ValueError("max_attempts must be between 1 and 10")
    if not isinstance(request.request_schema, dict) or not isinstance(request.request_options, dict):
        raise ValueError("request schema and options must be JSON objects")
    if not _SHA256_RE.fullmatch(request.request_hash):
        raise ValueError("request_hash must be lowercase hexadecimal")
    if not _SHA256_RE.fullmatch(request.source_sha256):
        raise ValueError("source_sha256 must be lowercase hexadecimal")
    if request.schema_sha256 != canonical_json_sha256(request.request_schema):
        raise ValueError("schema_sha256 does not match request_schema")
    expected_prefix = f"{request.document_id}/"
    if not request.source_key.startswith(expected_prefix):
        raise ValueError("source_key must use the document prefix")
    try:
        json.dumps(request.request_options, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("request_options must contain canonical JSON values") from exc


async def get_or_create_structured_request(
    session: AsyncSession,
    request: StructuredArtifactRequest,
) -> uuid.UUID | None:
    """Persist one immutable request, serialized against document reparsing.

    ``None`` means that the document/revision/page is no longer current. The
    caller owns the transaction and must commit before enqueueing an ARQ job.
    """

    _validate_request(request)
    document = (
        await session.execute(
            select(Document.parse_revision, Document.page_count)
            .where(Document.id == request.document_id)
            .with_for_update()
        )
    ).one_or_none()
    if (
        document is None
        or document.parse_revision != request.parse_revision
        or document.page_count is None
        or request.page_idx >= document.page_count
    ):
        return None

    artifact_id = uuid.uuid4()
    created = (
        await session.execute(
            pg_insert(DocumentStructuredArtifact)
            .values(
                id=artifact_id,
                document_id=request.document_id,
                parse_revision=request.parse_revision,
                page_idx=request.page_idx,
                artifact_type=request.artifact_type,
                backend=request.backend,
                model=request.model,
                model_revision=request.model_revision,
                prompt_version=request.prompt_version,
                protocol_version=request.protocol_version,
                schema_version=request.schema_version,
                request_hash=request.request_hash,
                request_schema=request.request_schema,
                schema_sha256=request.schema_sha256,
                request_options=request.request_options,
                source_key=request.source_key,
                source_sha256=request.source_sha256,
                max_attempts=request.max_attempts,
                status="queued",
            )
            .on_conflict_do_nothing(constraint="uq_structured_artifact_request")
            .returning(DocumentStructuredArtifact.id)
        )
    ).scalar_one_or_none()
    if created is not None:
        return created

    existing = (
        await session.execute(
            select(
                DocumentStructuredArtifact.id,
                DocumentStructuredArtifact.schema_sha256,
                DocumentStructuredArtifact.model_revision,
                DocumentStructuredArtifact.protocol_version,
                DocumentStructuredArtifact.source_key,
                DocumentStructuredArtifact.source_sha256,
            ).where(
                DocumentStructuredArtifact.document_id == request.document_id,
                DocumentStructuredArtifact.parse_revision == request.parse_revision,
                DocumentStructuredArtifact.page_idx == request.page_idx,
                DocumentStructuredArtifact.artifact_type == request.artifact_type,
                DocumentStructuredArtifact.backend == request.backend,
                DocumentStructuredArtifact.request_hash == request.request_hash,
            )
        )
    ).one()
    expected = (
        request.schema_sha256,
        request.model_revision,
        request.protocol_version,
        request.source_key,
        request.source_sha256,
    )
    if tuple(existing[1:]) != expected:
        raise RuntimeError("idempotency row does not match immutable request inputs")
    return existing.id


def build_candidate_artifact_key(claim: StructuredArtifactClaim) -> str:
    """Return an attempt-unique key; stale workers cannot overwrite a winner."""

    if claim.artifact_type not in _ARTIFACT_TYPES:
        raise ValueError("unsupported artifact_type")
    return (
        f"{claim.document_id}/sidecars/r{claim.parse_revision}/p{claim.page_idx:06d}/"
        f"{claim.artifact_type}/{claim.artifact_id}/attempt-{claim.claim_token}.json"
    )


async def claim_structured_artifact(
    session: AsyncSession,
    artifact_id: uuid.UUID,
    *,
    lease_seconds: int,
    now: datetime | None = None,
    claim_token: uuid.UUID | None = None,
) -> StructuredArtifactClaim | None:
    """Atomically claim one due request only for the current parse revision."""

    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive")
    claimed_at = _aware_utc(now)
    token = claim_token or uuid.uuid4()
    row = (
        await session.execute(
            update(DocumentStructuredArtifact)
            .where(
                DocumentStructuredArtifact.id == artifact_id,
                DocumentStructuredArtifact.status == "queued",
                DocumentStructuredArtifact.attempt_count
                < DocumentStructuredArtifact.max_attempts,
                or_(
                    DocumentStructuredArtifact.next_attempt_at.is_(None),
                    DocumentStructuredArtifact.next_attempt_at <= claimed_at,
                ),
                _current_revision_exists(),
            )
            .values(
                status="running",
                attempt_count=DocumentStructuredArtifact.attempt_count + 1,
                claim_token=token,
                claimed_at=claimed_at,
                lease_expires_at=claimed_at + timedelta(seconds=lease_seconds),
                next_attempt_at=None,
                finished_at=None,
                error=None,
                updated_at=claimed_at,
            )
            .returning(
                DocumentStructuredArtifact.id.label("artifact_id"),
                DocumentStructuredArtifact.document_id,
                DocumentStructuredArtifact.parse_revision,
                DocumentStructuredArtifact.page_idx,
                DocumentStructuredArtifact.artifact_type,
                DocumentStructuredArtifact.backend,
                DocumentStructuredArtifact.model,
                DocumentStructuredArtifact.model_revision,
                DocumentStructuredArtifact.prompt_version,
                DocumentStructuredArtifact.protocol_version,
                DocumentStructuredArtifact.schema_version,
                DocumentStructuredArtifact.request_hash,
                DocumentStructuredArtifact.request_schema,
                DocumentStructuredArtifact.schema_sha256,
                DocumentStructuredArtifact.request_options,
                DocumentStructuredArtifact.source_key,
                DocumentStructuredArtifact.source_sha256,
                DocumentStructuredArtifact.attempt_count,
                DocumentStructuredArtifact.max_attempts,
                DocumentStructuredArtifact.claim_token,
            )
        )
    ).mappings().one_or_none()
    if row is None:
        return None
    return StructuredArtifactClaim(
        artifact_id=row["artifact_id"],
        document_id=row["document_id"],
        parse_revision=row["parse_revision"],
        page_idx=row["page_idx"],
        artifact_type=row["artifact_type"],
        backend=row["backend"],
        model=row["model"],
        model_revision=row["model_revision"],
        prompt_version=row["prompt_version"],
        protocol_version=row["protocol_version"],
        schema_version=row["schema_version"],
        request_hash=row["request_hash"],
        request_schema=dict(row["request_schema"]),
        schema_sha256=row["schema_sha256"],
        request_options=dict(row["request_options"]),
        source_key=row["source_key"],
        source_sha256=row["source_sha256"],
        attempt_count=row["attempt_count"],
        max_attempts=row["max_attempts"],
        claim_token=row["claim_token"],
    )


def _validate_publish(
    claim: StructuredArtifactClaim,
    *,
    artifact_key: str,
    content_sha256: str,
    size_bytes: int,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    if artifact_key != build_candidate_artifact_key(claim):
        raise ValueError("artifact_key is not scoped to this claim")
    if not _SHA256_RE.fullmatch(content_sha256):
        raise ValueError("content_sha256 must be lowercase hexadecimal")
    if size_bytes <= 0:
        raise ValueError("size_bytes must be positive")
    value = dict(summary)
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("summary must contain canonical JSON values") from exc
    if len(encoded) > _MAX_SUMMARY_BYTES:
        raise ValueError("summary is too large")
    return value


async def publish_structured_artifact(
    session: AsyncSession,
    claim: StructuredArtifactClaim,
    *,
    artifact_key: str,
    content_sha256: str,
    size_bytes: int,
    summary: Mapping[str, Any],
    now: datetime | None = None,
) -> bool:
    """Publish metadata iff this attempt still owns the current revision."""

    finished_at = _aware_utc(now)
    safe_summary = _validate_publish(
        claim,
        artifact_key=artifact_key,
        content_sha256=content_sha256,
        size_bytes=size_bytes,
        summary=summary,
    )
    published = (
        await session.execute(
            update(DocumentStructuredArtifact)
            .where(
                DocumentStructuredArtifact.id == claim.artifact_id,
                DocumentStructuredArtifact.status == "running",
                DocumentStructuredArtifact.claim_token == claim.claim_token,
                _current_revision_exists(),
            )
            .values(
                status="ready",
                artifact_key=artifact_key,
                content_sha256=content_sha256,
                size_bytes=size_bytes,
                summary=safe_summary,
                error=None,
                claim_token=None,
                lease_expires_at=None,
                next_attempt_at=None,
                finished_at=finished_at,
                updated_at=finished_at,
            )
            .returning(DocumentStructuredArtifact.id)
        )
    ).scalar_one_or_none()
    return published is not None


async def fail_structured_artifact(
    session: AsyncSession,
    claim: StructuredArtifactClaim,
    *,
    error_code: str,
    retryable: bool,
    retry_delay_seconds: int = 30,
    now: datetime | None = None,
) -> str | None:
    """Release a claim to a bounded retry or terminal error without raw output."""

    if not _ERROR_CODE_RE.fullmatch(error_code):
        raise ValueError("error_code must be a bounded machine-readable code")
    if retry_delay_seconds < 0:
        raise ValueError("retry_delay_seconds must be non-negative")
    changed_at = _aware_utc(now)
    requeue = retryable and claim.attempt_count < claim.max_attempts
    status = "queued" if requeue else "error"
    finished_at = None if requeue else changed_at
    next_attempt_at = (
        changed_at + timedelta(seconds=retry_delay_seconds) if requeue else None
    )
    changed = (
        await session.execute(
            update(DocumentStructuredArtifact)
            .where(
                DocumentStructuredArtifact.id == claim.artifact_id,
                DocumentStructuredArtifact.status == "running",
                DocumentStructuredArtifact.claim_token == claim.claim_token,
                _current_revision_exists(),
            )
            .values(
                status=status,
                error=error_code,
                claim_token=None,
                lease_expires_at=None,
                next_attempt_at=next_attempt_at,
                finished_at=finished_at,
                updated_at=changed_at,
            )
            .returning(DocumentStructuredArtifact.status)
        )
    ).scalar_one_or_none()
    return changed


async def supersede_stale_structured_artifacts(
    session: AsyncSession,
    *,
    document_id: uuid.UUID | None = None,
    now: datetime | None = None,
) -> tuple[uuid.UUID, ...]:
    """Make every non-current revision terminal; safe inside reparse transaction."""

    changed_at = _aware_utc(now)
    statement = update(DocumentStructuredArtifact).where(
        DocumentStructuredArtifact.status != "superseded",
        ~_current_revision_exists(),
    )
    if document_id is not None:
        statement = statement.where(DocumentStructuredArtifact.document_id == document_id)
    ids = (
        await session.execute(
            statement.values(
                status="superseded",
                claim_token=None,
                lease_expires_at=None,
                next_attempt_at=None,
                finished_at=func.coalesce(DocumentStructuredArtifact.finished_at, changed_at),
                updated_at=changed_at,
            ).returning(DocumentStructuredArtifact.id)
        )
    ).scalars().all()
    return tuple(ids)


async def sweep_structured_artifacts(
    session: AsyncSession,
    *,
    now: datetime | None = None,
    retry_delay_seconds: int = 30,
    dispatch_limit: int = 100,
) -> StructuredSweepResult:
    """Recover stale leases and list due rows for idempotent ARQ dispatch."""

    if retry_delay_seconds < 0:
        raise ValueError("retry_delay_seconds must be non-negative")
    if dispatch_limit <= 0:
        raise ValueError("dispatch_limit must be positive")
    swept_at = _aware_utc(now)
    current_revision = _current_revision_exists()
    superseded = await supersede_stale_structured_artifacts(session, now=swept_at)

    exhausted = tuple(
        (
            await session.execute(
                update(DocumentStructuredArtifact)
                .where(
                    DocumentStructuredArtifact.status == "running",
                    DocumentStructuredArtifact.lease_expires_at <= swept_at,
                    DocumentStructuredArtifact.attempt_count
                    >= DocumentStructuredArtifact.max_attempts,
                    current_revision,
                )
                .values(
                    status="error",
                    error="lease_expired_attempts_exhausted",
                    claim_token=None,
                    lease_expires_at=None,
                    next_attempt_at=None,
                    finished_at=swept_at,
                    updated_at=swept_at,
                )
                .returning(DocumentStructuredArtifact.id)
            )
        ).scalars().all()
    )
    requeued = tuple(
        (
            await session.execute(
                update(DocumentStructuredArtifact)
                .where(
                    DocumentStructuredArtifact.status == "running",
                    DocumentStructuredArtifact.lease_expires_at <= swept_at,
                    DocumentStructuredArtifact.attempt_count
                    < DocumentStructuredArtifact.max_attempts,
                    current_revision,
                )
                .values(
                    status="queued",
                    error="lease_expired",
                    claim_token=None,
                    lease_expires_at=None,
                    next_attempt_at=swept_at + timedelta(seconds=retry_delay_seconds),
                    finished_at=None,
                    updated_at=swept_at,
                )
                .returning(DocumentStructuredArtifact.id)
            )
        ).scalars().all()
    )
    due = (
        await session.execute(
            select(
                DocumentStructuredArtifact.id,
                DocumentStructuredArtifact.attempt_count,
            )
            .where(
                DocumentStructuredArtifact.status == "queued",
                DocumentStructuredArtifact.attempt_count
                < DocumentStructuredArtifact.max_attempts,
                or_(
                    DocumentStructuredArtifact.next_attempt_at.is_(None),
                    DocumentStructuredArtifact.next_attempt_at <= swept_at,
                ),
                current_revision,
            )
            .order_by(
                DocumentStructuredArtifact.next_attempt_at.asc().nullsfirst(),
                DocumentStructuredArtifact.created_at,
                DocumentStructuredArtifact.id,
            )
            .limit(dispatch_limit)
            .with_for_update(skip_locked=True)
        )
    ).all()
    return StructuredSweepResult(
        superseded_ids=superseded,
        requeued_ids=requeued,
        exhausted_ids=exhausted,
        dispatch=tuple(
            StructuredDispatchCandidate(row.id, row.attempt_count + 1) for row in due
        ),
    )
