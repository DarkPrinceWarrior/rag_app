from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy.dialects import postgresql

from rag_app.pipeline.structured_extraction_protocol import canonical_json_sha256
from rag_app.workers.structured_lifecycle import (
    StructuredArtifactClaim,
    StructuredArtifactRequest,
    build_candidate_artifact_key,
    claim_structured_artifact,
    fail_structured_artifact,
    get_or_create_structured_request,
    publish_structured_artifact,
    supersede_stale_structured_artifacts,
    sweep_structured_artifacts,
)

_DOC_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
_ARTIFACT_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")
_CLAIM_TOKEN = uuid.UUID("33333333-3333-3333-3333-333333333333")
_NOW = datetime(2026, 7, 12, 9, 0, tzinfo=UTC)


class _Scalars:
    def __init__(self, values: list[Any]) -> None:
        self.values = values

    def all(self) -> list[Any]:
        return self.values


class _Mappings:
    def __init__(self, value: Any) -> None:
        self.value = value

    def one_or_none(self) -> Any:
        return self.value


class _Result:
    def __init__(
        self,
        *,
        one_or_none: Any = None,
        scalar: Any = None,
        scalars: list[Any] | None = None,
        mappings: Any = None,
        all_rows: list[Any] | None = None,
        one: Any = None,
    ) -> None:
        self._one_or_none = one_or_none
        self._scalar = scalar
        self._scalars = scalars or []
        self._mappings = mappings
        self._all = all_rows or []
        self._one = one

    def one_or_none(self) -> Any:
        return self._one_or_none

    def scalar_one_or_none(self) -> Any:
        return self._scalar

    def scalars(self) -> _Scalars:
        return _Scalars(self._scalars)

    def mappings(self) -> _Mappings:
        return _Mappings(self._mappings)

    def all(self) -> list[Any]:
        return self._all

    def one(self) -> Any:
        return self._one


class _Session:
    def __init__(self, results: list[_Result]) -> None:
        self.results = results
        self.statements: list[Any] = []

    async def execute(self, statement: Any) -> _Result:
        self.statements.append(statement)
        return self.results.pop(0)


def _sql(statement: Any) -> str:
    return str(statement.compile(dialect=postgresql.dialect()))


def _request() -> StructuredArtifactRequest:
    schema = {
        "type": "object",
        "properties": {"tag": {"type": ["string", "null"]}},
        "required": ["tag"],
        "additionalProperties": False,
    }
    return StructuredArtifactRequest(
        document_id=_DOC_ID,
        parse_revision=4,
        page_idx=7,
        artifact_type="kie",
        backend="granite",
        model="ibm-granite/granite-vision-4.1-4b",
        model_revision="a" * 40,
        prompt_version="varex-v1",
        protocol_version="structured-v1",
        schema_version=1,
        request_hash="b" * 64,
        request_schema=schema,
        schema_sha256=canonical_json_sha256(schema),
        request_options={"temperature": 0, "max_tokens": 4096},
        source_key=f"{_DOC_ID}/sidecars/r4/p000007/source.png",
        source_sha256="c" * 64,
    )


def _claim() -> StructuredArtifactClaim:
    request = _request()
    return StructuredArtifactClaim(
        artifact_id=_ARTIFACT_ID,
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
        attempt_count=1,
        max_attempts=3,
        claim_token=_CLAIM_TOKEN,
    )


def test_request_creation_locks_revision_and_uses_unique_conflict_key() -> None:
    session = _Session(
        [
            _Result(one_or_none=SimpleNamespace(parse_revision=4, page_count=10)),
            _Result(scalar=_ARTIFACT_ID),
        ]
    )

    created = asyncio.run(
        get_or_create_structured_request(session, _request())  # type: ignore[arg-type]
    )

    assert created == _ARTIFACT_ID
    assert "FOR UPDATE" in _sql(session.statements[0])
    insert_sql = _sql(session.statements[1])
    assert "ON CONFLICT ON CONSTRAINT uq_structured_artifact_request DO NOTHING" in insert_sql
    assert "request_schema" in insert_sql
    assert "model_revision" in insert_sql


def test_request_creation_rejects_stale_revision_and_mismatched_source() -> None:
    stale = _Session([_Result(one_or_none=SimpleNamespace(parse_revision=5, page_count=10))])
    assert (
        asyncio.run(
            get_or_create_structured_request(stale, _request())  # type: ignore[arg-type]
        )
        is None
    )

    invalid = _request()
    invalid = StructuredArtifactRequest(
        **{**invalid.__dict__, "source_key": "another-document/source.png"}
    )
    with pytest.raises(ValueError, match="document prefix"):
        asyncio.run(
            get_or_create_structured_request(_Session([]), invalid)  # type: ignore[arg-type]
        )


def test_claim_is_atomic_due_and_revision_guarded() -> None:
    request = _request()
    mapping = {
        "artifact_id": _ARTIFACT_ID,
        "document_id": request.document_id,
        "parse_revision": request.parse_revision,
        "page_idx": request.page_idx,
        "artifact_type": request.artifact_type,
        "backend": request.backend,
        "model": request.model,
        "model_revision": request.model_revision,
        "prompt_version": request.prompt_version,
        "protocol_version": request.protocol_version,
        "schema_version": request.schema_version,
        "request_hash": request.request_hash,
        "request_schema": request.request_schema,
        "schema_sha256": request.schema_sha256,
        "request_options": request.request_options,
        "source_key": request.source_key,
        "source_sha256": request.source_sha256,
        "attempt_count": 1,
        "max_attempts": 3,
        "claim_token": _CLAIM_TOKEN,
    }
    session = _Session([_Result(mappings=mapping)])

    claim = asyncio.run(
        claim_structured_artifact(
            session,  # type: ignore[arg-type]
            _ARTIFACT_ID,
            lease_seconds=240,
            now=_NOW,
            claim_token=_CLAIM_TOKEN,
        )
    )

    assert claim == _claim()
    statement = _sql(session.statements[0])
    assert "document_structured_artifacts.status" in statement
    assert "document_structured_artifacts.next_attempt_at" in statement
    assert (
        "document_structured_artifacts.attempt_count"
        " < document_structured_artifacts.max_attempts"
    ) in statement
    assert "EXISTS (SELECT 1" in statement
    assert "documents.parse_revision = document_structured_artifacts.parse_revision" in statement


def test_candidate_key_and_publish_are_bound_to_claim_token() -> None:
    claim = _claim()
    key = build_candidate_artifact_key(claim)
    assert key.endswith(f"/{_ARTIFACT_ID}/attempt-{_CLAIM_TOKEN}.json")
    session = _Session([_Result(scalar=_ARTIFACT_ID)])

    assert asyncio.run(
        publish_structured_artifact(
            session,  # type: ignore[arg-type]
            claim,
            artifact_key=key,
            content_sha256="d" * 64,
            size_bytes=128,
            summary={"field_count": 3},
            now=_NOW,
        )
    )
    statement = _sql(session.statements[0])
    assert "document_structured_artifacts.claim_token" in statement
    assert "documents.parse_revision = document_structured_artifacts.parse_revision" in statement
    assert "artifact_key" in statement

    with pytest.raises(ValueError, match="not scoped"):
        asyncio.run(
            publish_structured_artifact(
                _Session([]),  # type: ignore[arg-type]
                claim,
                artifact_key=f"{_ARTIFACT_ID}.json",
                content_sha256="d" * 64,
                size_bytes=128,
                summary={},
            )
        )


def test_fail_requeues_only_current_claim_without_raw_output() -> None:
    session = _Session([_Result(scalar="queued")])
    status = asyncio.run(
        fail_structured_artifact(
            session,  # type: ignore[arg-type]
            _claim(),
            error_code="inference.timeout",
            retryable=True,
            retry_delay_seconds=30,
            now=_NOW,
        )
    )
    assert status == "queued"
    statement = _sql(session.statements[0])
    assert "document_structured_artifacts.claim_token" in statement
    assert "documents.parse_revision = document_structured_artifacts.parse_revision" in statement
    with pytest.raises(ValueError, match="machine-readable"):
        asyncio.run(
            fail_structured_artifact(
                _Session([]),  # type: ignore[arg-type]
                _claim(),
                error_code="raw response: secret value",
                retryable=False,
            )
        )


def test_supersede_uses_document_revision_not_age() -> None:
    session = _Session([_Result(scalars=[_ARTIFACT_ID])])
    changed = asyncio.run(
        supersede_stale_structured_artifacts(
            session,  # type: ignore[arg-type]
            document_id=_DOC_ID,
            now=_NOW,
        )
    )
    assert changed == (_ARTIFACT_ID,)
    statement = _sql(session.statements[0])
    assert "NOT (EXISTS (SELECT 1" in statement
    assert "document_structured_artifacts.document_id" in statement
    assert "claim_token" in statement


def test_sweep_separates_superseded_retry_exhausted_and_due_rows() -> None:
    superseded_id = uuid.UUID("44444444-4444-4444-4444-444444444444")
    exhausted_id = uuid.UUID("55555555-5555-5555-5555-555555555555")
    requeued_id = uuid.UUID("66666666-6666-6666-6666-666666666666")
    due_id = uuid.UUID("77777777-7777-7777-7777-777777777777")
    session = _Session(
        [
            _Result(scalars=[superseded_id]),
            _Result(scalars=[exhausted_id]),
            _Result(scalars=[requeued_id]),
            _Result(all_rows=[SimpleNamespace(id=due_id, attempt_count=1)]),
        ]
    )

    result = asyncio.run(
        sweep_structured_artifacts(
            session,  # type: ignore[arg-type]
            now=_NOW,
            retry_delay_seconds=30,
            dispatch_limit=20,
        )
    )

    assert result.superseded_ids == (superseded_id,)
    assert result.exhausted_ids == (exhausted_id,)
    assert result.requeued_ids == (requeued_id,)
    assert result.dispatch[0].artifact_id == due_id
    assert result.dispatch[0].attempt_number == 2
    exhausted_sql = _sql(session.statements[1])
    retry_sql = _sql(session.statements[2])
    due_sql = _sql(session.statements[3])
    assert "lease_expires_at" in exhausted_sql and "attempt_count >=" in exhausted_sql
    assert "lease_expires_at" in retry_sql and "attempt_count <" in retry_sql
    assert "FOR UPDATE SKIP LOCKED" in due_sql
    assert "documents.parse_revision = document_structured_artifacts.parse_revision" in due_sql


def test_lifecycle_rejects_naive_time_and_invalid_publish_metadata() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        asyncio.run(
            claim_structured_artifact(
                _Session([]),  # type: ignore[arg-type]
                _ARTIFACT_ID,
                lease_seconds=240,
                now=datetime(2026, 7, 12, 9, 0),
            )
        )
    with pytest.raises(ValueError, match="canonical JSON"):
        asyncio.run(
            publish_structured_artifact(
                _Session([]),  # type: ignore[arg-type]
                _claim(),
                artifact_key=build_candidate_artifact_key(_claim()),
                content_sha256="d" * 64,
                size_bytes=128,
                summary={"latency": float("nan")},
            )
        )
