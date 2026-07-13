from __future__ import annotations

import asyncio
import hashlib
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession, create_async_engine

from rag_app.db.models import (
    AuditLog,
    ChatMessage,
    ChatSession,
    Chunk,
    Document,
    DocumentStatus,
    DocumentStructuredArtifact,
    DocumentTranslation,
    Folder,
    MemoryCandidate,
    MemoryEvent,
    MemoryItem,
    MemoryItemSource,
    PageEmbedding,
    Segment,
    SegmentKind,
    SegmentVersion,
)
from rag_app.db.rls import PROTECTED_TABLES

_ADMIN_URL_ENV = "RAG_RLS_TEST_ADMIN_URL"
_ROLE = "rag_api"
_CANARY_KEYS = (
    "folder_a",
    "folder_b",
    "folder_null",
    "document_a",
    "document_b",
    "document_null",
    "segment_a",
    "segment_b",
    "segment_null",
    "chunk_a",
    "chunk_b",
    "chunk_null",
    "translation_a",
    "translation_b",
    "translation_null",
    "version_a",
    "version_b",
    "version_null",
    "session_a",
    "session_b",
    "session_null",
    "message_a",
    "message_b",
    "message_null",
    "page_a",
    "page_b",
    "page_null",
    "artifact_a",
    "artifact_b",
    "artifact_null",
    "audit_a",
    "audit_b",
    "event_a",
    "event_b",
    "item_a",
    "item_b",
    "candidate_a",
    "candidate_b",
    "memory_audit_a",
    "memory_audit_b",
    "memory_audit_null",
)


@dataclass(frozen=True)
class _VisibleTable:
    table: str
    key_column: str
    keys: tuple[str, ...]
    admin_all: bool = True


_VISIBLE_TABLES = (
    _VisibleTable("documents", "id", ("document_a", "document_b", "document_null")),
    _VisibleTable("folders", "id", ("folder_a", "folder_b", "folder_null")),
    _VisibleTable("segments", "id", ("segment_a", "segment_b", "segment_null")),
    _VisibleTable("chunks", "id", ("chunk_a", "chunk_b", "chunk_null")),
    _VisibleTable(
        "document_translations",
        "id",
        ("translation_a", "translation_b", "translation_null"),
    ),
    _VisibleTable(
        "segment_versions", "id", ("version_a", "version_b", "version_null")
    ),
    _VisibleTable("chat_sessions", "id", ("session_a", "session_b", "session_null")),
    _VisibleTable("chat_messages", "id", ("message_a", "message_b", "message_null")),
    _VisibleTable("page_embeddings", "id", ("page_a", "page_b", "page_null")),
    _VisibleTable(
        "document_structured_artifacts",
        "id",
        ("artifact_a", "artifact_b", "artifact_null"),
    ),
    _VisibleTable("audit_log", "id", ("audit_a", "audit_b")),
    _VisibleTable("memory_events", "id", ("event_a", "event_b"), admin_all=False),
    _VisibleTable("memory_items", "id", ("item_a", "item_b"), admin_all=False),
    _VisibleTable(
        "memory_candidates", "id", ("candidate_a", "candidate_b"), admin_all=False
    ),
    _VisibleTable(
        "memory_item_sources", "item_id", ("item_a", "item_b"), admin_all=False
    ),
    _VisibleTable(
        "memory_audit_log",
        "id",
        ("memory_audit_a", "memory_audit_b", "memory_audit_null"),
    ),
)


def _async_admin_url(raw_url: str) -> str:
    url = make_url(raw_url)
    database = url.database or ""
    if "rls_test" not in database.lower():
        pytest.fail(
            f"{_ADMIN_URL_ENV} must target a database whose name contains 'rls_test'; "
            f"got {database!r}"
        )
    if not url.drivername.startswith("postgresql"):
        pytest.fail(f"{_ADMIN_URL_ENV} must use PostgreSQL")
    if url.drivername == "postgresql":
        url = url.set(drivername="postgresql+asyncpg")
    if url.drivername != "postgresql+asyncpg":
        pytest.fail(f"{_ADMIN_URL_ENV} must use the asyncpg driver or a plain PostgreSQL URL")
    return url.render_as_string(hide_password=False)


def _ids() -> dict[str, Any]:
    ids: dict[str, Any] = {key: uuid.uuid4() for key in _CANARY_KEYS}
    marker = uuid.uuid4().int % 1_000_000_000
    ids.update(
        {
            "memory_audit_a": -(marker + 1),
            "memory_audit_b": -(marker + 2),
            "memory_audit_null": -(marker + 3),
        }
    )
    return ids


def _digest(label: str) -> str:
    return hashlib.sha256(f"rls-postgres-integration:{label}".encode()).hexdigest()


def _document(
    *,
    document_id: uuid.UUID,
    label: str,
    owner_sub: str | None,
    folder_id: uuid.UUID | None,
) -> Document:
    return Document(
        id=document_id,
        filename=f"rls-{label}.pdf",
        content_type="application/pdf",
        size_bytes=1,
        status=DocumentStatus.uploaded,
        s3_key_original=f"rls-test/{document_id}/original.pdf",
        segment_count=0,
        translated_count=0,
        owner_sub=owner_sub,
        folder_id=folder_id,
    )


def _artifact(
    *,
    artifact_id: uuid.UUID,
    document_id: uuid.UUID,
    label: str,
    page_idx: int = 0,
) -> DocumentStructuredArtifact:
    return DocumentStructuredArtifact(
        id=artifact_id,
        document_id=document_id,
        parse_revision=0,
        page_idx=page_idx,
        artifact_type="kie",
        backend="rls-test",
        model="rls-test-model",
        prompt_version="rls-test-v1",
        schema_version=2,
        request_hash=_digest(f"request:{label}"),
        request_schema={"type": "object"},
        schema_sha256=_digest(f"schema:{label}"),
        model_revision="rls-test-revision",
        protocol_version="rls-test-v1",
        request_options={},
        source_key=f"rls-test/{document_id}/page-{page_idx}.png",
        source_sha256=_digest(f"source:{label}"),
        status="queued",
        attempt_count=0,
        max_attempts=3,
        summary={},
    )


async def _seed(conn: AsyncConnection, ids: dict[str, Any], tenant_id: uuid.UUID) -> None:
    async with AsyncSession(
        bind=conn,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    ) as session:
        session.add_all(
            [
                Folder(id=ids["folder_a"], name="rls-a", owner_sub="owner-a"),
                Folder(id=ids["folder_b"], name="rls-b", owner_sub="owner-b"),
                Folder(id=ids["folder_null"], name="rls-null", owner_sub=None),
            ]
        )
        await session.flush()

        session.add_all(
            [
                _document(
                    document_id=ids["document_a"],
                    label="a",
                    owner_sub="owner-a",
                    folder_id=ids["folder_a"],
                ),
                _document(
                    document_id=ids["document_b"],
                    label="b",
                    owner_sub="owner-b",
                    folder_id=ids["folder_b"],
                ),
                _document(
                    document_id=ids["document_null"],
                    label="null",
                    owner_sub=None,
                    folder_id=ids["folder_null"],
                ),
            ]
        )
        await session.flush()

        session.add_all(
            [
                Segment(
                    id=ids["segment_a"],
                    document_id=ids["document_a"],
                    idx=0,
                    kind=SegmentKind.paragraph,
                    source_text="a",
                    meta={},
                ),
                Segment(
                    id=ids["segment_b"],
                    document_id=ids["document_b"],
                    idx=0,
                    kind=SegmentKind.paragraph,
                    source_text="b",
                    meta={},
                ),
                Segment(
                    id=ids["segment_null"],
                    document_id=ids["document_null"],
                    idx=0,
                    kind=SegmentKind.paragraph,
                    source_text="null",
                    meta={},
                ),
                Chunk(
                    id=ids["chunk_a"],
                    document_id=ids["document_a"],
                    idx=0,
                    text_en="a",
                    text_ru="a",
                    meta={},
                ),
                Chunk(
                    id=ids["chunk_b"],
                    document_id=ids["document_b"],
                    idx=0,
                    text_en="b",
                    text_ru="b",
                    meta={},
                ),
                Chunk(
                    id=ids["chunk_null"],
                    document_id=ids["document_null"],
                    idx=0,
                    text_en="null",
                    text_ru="null",
                    meta={},
                ),
                DocumentTranslation(
                    id=ids["translation_a"],
                    document_id=ids["document_a"],
                    target_lang="en",
                    data={},
                ),
                DocumentTranslation(
                    id=ids["translation_b"],
                    document_id=ids["document_b"],
                    target_lang="en",
                    data={},
                ),
                DocumentTranslation(
                    id=ids["translation_null"],
                    document_id=ids["document_null"],
                    target_lang="en",
                    data={},
                ),
                SegmentVersion(
                    id=ids["version_a"],
                    segment_id=ids["segment_a"],
                    document_id=ids["document_a"],
                    old_text="a-old",
                    new_text="a-new",
                ),
                SegmentVersion(
                    id=ids["version_b"],
                    segment_id=ids["segment_b"],
                    document_id=ids["document_b"],
                    old_text="b-old",
                    new_text="b-new",
                ),
                SegmentVersion(
                    id=ids["version_null"],
                    segment_id=ids["segment_null"],
                    document_id=ids["document_null"],
                    old_text="null-old",
                    new_text="null-new",
                ),
                PageEmbedding(
                    id=ids["page_a"], document_id=ids["document_a"], page_idx=0, meta={}
                ),
                PageEmbedding(
                    id=ids["page_b"], document_id=ids["document_b"], page_idx=0, meta={}
                ),
                PageEmbedding(
                    id=ids["page_null"],
                    document_id=ids["document_null"],
                    page_idx=0,
                    meta={},
                ),
                _artifact(
                    artifact_id=ids["artifact_a"],
                    document_id=ids["document_a"],
                    label="a",
                ),
                _artifact(
                    artifact_id=ids["artifact_b"],
                    document_id=ids["document_b"],
                    label="b",
                ),
                _artifact(
                    artifact_id=ids["artifact_null"],
                    document_id=ids["document_null"],
                    label="null",
                ),
            ]
        )
        await session.flush()

        session.add_all(
            [
                ChatSession(
                    id=ids["session_a"],
                    title="a",
                    owner_sub="owner-a",
                    document_id=ids["document_a"],
                    folder_id=ids["folder_a"],
                ),
                ChatSession(
                    id=ids["session_b"],
                    title="b",
                    owner_sub="owner-b",
                    document_id=ids["document_b"],
                    folder_id=ids["folder_b"],
                ),
                ChatSession(
                    id=ids["session_null"],
                    title="null",
                    owner_sub=None,
                    document_id=ids["document_null"],
                    folder_id=ids["folder_null"],
                ),
                AuditLog(id=ids["audit_a"], user_sub="owner-a", action="rls_test"),
                AuditLog(id=ids["audit_b"], user_sub="owner-b", action="rls_test"),
                MemoryEvent(
                    id=ids["event_a"],
                    tenant_id=tenant_id,
                    user_id="owner-a",
                    event_type="rls_test",
                    payload={},
                ),
                MemoryEvent(
                    id=ids["event_b"],
                    tenant_id=tenant_id,
                    user_id="owner-b",
                    event_type="rls_test",
                    payload={},
                ),
                MemoryItem(
                    id=ids["item_a"],
                    tenant_id=tenant_id,
                    user_id="owner-a",
                    scope="user",
                    kind="fact",
                    content="a",
                    source_event_ids=[ids["event_a"]],
                ),
                MemoryItem(
                    id=ids["item_b"],
                    tenant_id=tenant_id,
                    user_id="owner-b",
                    scope="user",
                    kind="fact",
                    content="b",
                    source_event_ids=[ids["event_b"]],
                ),
                MemoryCandidate(
                    id=ids["candidate_a"],
                    tenant_id=tenant_id,
                    user_id="owner-a",
                    action="create",
                    proposed={"content": "a"},
                    confidence=0.9,
                ),
                MemoryCandidate(
                    id=ids["candidate_b"],
                    tenant_id=tenant_id,
                    user_id="owner-b",
                    action="create",
                    proposed={"content": "b"},
                    confidence=0.9,
                ),
            ]
        )
        await session.flush()

        session.add_all(
            [
                ChatMessage(
                    id=ids["message_a"], session_id=ids["session_a"], role="user", content="a"
                ),
                ChatMessage(
                    id=ids["message_b"], session_id=ids["session_b"], role="user", content="b"
                ),
                ChatMessage(
                    id=ids["message_null"],
                    session_id=ids["session_null"],
                    role="user",
                    content="null",
                ),
                MemoryItemSource(item_id=ids["item_a"], event_id=ids["event_a"]),
                MemoryItemSource(item_id=ids["item_b"], event_id=ids["event_b"]),
            ]
        )
        await session.commit()

    await conn.execute(
        text(
            "INSERT INTO memory_audit_log "
            "(id, tenant_id, user_id, action, actor) OVERRIDING SYSTEM VALUE VALUES "
            "(:id_a, :tenant, 'owner-a', 'rls_test', 'test'), "
            "(:id_b, :tenant, 'owner-b', 'rls_test', 'test'), "
            "(:id_null, :tenant, NULL, 'rls_test', 'system')"
        ),
        {
            "id_a": ids["memory_audit_a"],
            "id_b": ids["memory_audit_b"],
            "id_null": ids["memory_audit_null"],
            "tenant": tenant_id,
        },
    )


async def _set_api_principal(
    conn: AsyncConnection,
    *,
    user_sub: str,
    is_admin: bool,
    tenant_id: uuid.UUID | None,
) -> None:
    await conn.exec_driver_sql("RESET ROLE")
    await conn.exec_driver_sql(f"SET LOCAL ROLE {_ROLE}")
    values = {
        "user": user_sub,
        "admin": "on" if is_admin else "off",
        "tenant": str(tenant_id) if tenant_id else "",
    }
    await conn.execute(
        text(
            "SELECT set_config('app.user_id', :user, true), "
            "set_config('app.is_admin', :admin, true), "
            "set_config('app.tenant_id', :tenant, true), "
            "set_config('app.project_id', '', true), "
            "set_config('app.document_id', '', true)"
        ),
        values,
    )
    assert await conn.scalar(text("SELECT current_user")) == _ROLE


async def _visible_ids(
    conn: AsyncConnection,
    spec: _VisibleTable,
    ids: dict[str, Any],
) -> set[Any]:
    params = {f"id_{index}": ids[key] for index, key in enumerate(spec.keys)}
    placeholders = ", ".join(f":id_{index}" for index in range(len(spec.keys)))
    rows = await conn.scalars(
        text(
            f"SELECT {spec.key_column} FROM {spec.table} "
            f"WHERE {spec.key_column} IN ({placeholders})"
        ),
        params,
    )
    return set(rows)


async def _assert_visibility(
    conn: AsyncConnection,
    ids: dict[str, Any],
    *,
    principal: str,
    is_admin: bool,
    tenant_id: uuid.UUID | None,
    expected_suffix: str | None,
) -> None:
    await _set_api_principal(
        conn,
        user_sub=principal,
        is_admin=is_admin,
        tenant_id=tenant_id,
    )
    for spec in _VISIBLE_TABLES:
        if is_admin and spec.admin_all:
            expected = {ids[key] for key in spec.keys}
        elif is_admin:
            expected = set()
        elif expected_suffix is None:
            expected = set()
        else:
            expected = {ids[key] for key in spec.keys if key.endswith(expected_suffix)}
        assert await _visible_ids(conn, spec, ids) == expected, spec.table


def _assert_rls_error(error: DBAPIError, table: str) -> None:
    message = str(error).lower()
    assert "row-level security" in message or "row level security" in message, (
        f"{table}: expected an RLS rejection, got {error!r}"
    )


async def _expect_sql_rls_denied(
    conn: AsyncConnection,
    *,
    table: str,
    statement: str,
    params: dict[str, Any],
) -> None:
    savepoint = await conn.begin_nested()
    try:
        with pytest.raises(DBAPIError) as captured:
            await conn.execute(text(statement), params)
        _assert_rls_error(captured.value, table)
    finally:
        await savepoint.rollback()


async def _expect_sql_permission_denied(
    conn: AsyncConnection,
    *,
    statement: str,
    params: dict[str, Any],
) -> None:
    savepoint = await conn.begin_nested()
    try:
        with pytest.raises(DBAPIError) as captured:
            await conn.execute(text(statement), params)
        assert "permission denied" in str(captured.value).lower()
    finally:
        await savepoint.rollback()


async def _expect_insert_rls_denied(
    conn: AsyncConnection,
    *,
    table: str,
    factory: Callable[[], object],
) -> None:
    async with AsyncSession(
        bind=conn,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    ) as session:
        session.add(factory())
        with pytest.raises(DBAPIError) as captured:
            await session.flush()
        _assert_rls_error(captured.value, table)
        await session.rollback()


async def _assert_foreign_update_delete_hidden(
    conn: AsyncConnection,
    ids: dict[str, Any],
) -> None:
    for spec in _VISIBLE_TABLES:
        if spec.table == "audit_log":
            continue
        foreign_key = next(key for key in spec.keys if key.endswith("_b"))
        params = {"foreign_id": ids[foreign_key]}
        updated = await conn.execute(
            text(
                f"UPDATE {spec.table} SET {spec.key_column} = {spec.key_column} "
                f"WHERE {spec.key_column} = :foreign_id"
            ),
            params,
        )
        assert updated.rowcount == 0, spec.table
        deleted = await conn.execute(
            text(f"DELETE FROM {spec.table} WHERE {spec.key_column} = :foreign_id"),
            params,
        )
        assert deleted.rowcount == 0, spec.table


async def _assert_forbidden_inserts(
    conn: AsyncConnection,
    ids: dict[str, Any],
) -> None:
    await _expect_insert_rls_denied(
        conn,
        table="folders",
        factory=lambda: Folder(id=uuid.uuid4(), name=f"rls-b-{uuid.uuid4()}", owner_sub="owner-b"),
    )
    await _expect_insert_rls_denied(
        conn,
        table="folders",
        factory=lambda: Folder(id=uuid.uuid4(), name=f"rls-null-{uuid.uuid4()}", owner_sub=None),
    )
    await _expect_insert_rls_denied(
        conn,
        table="documents",
        factory=lambda: _document(
            document_id=uuid.uuid4(), label="foreign", owner_sub="owner-b", folder_id=None
        ),
    )
    await _expect_insert_rls_denied(
        conn,
        table="documents",
        factory=lambda: _document(
            document_id=uuid.uuid4(), label="null", owner_sub=None, folder_id=None
        ),
    )
    await _expect_insert_rls_denied(
        conn,
        table="segments",
        factory=lambda: Segment(
            id=uuid.uuid4(),
            document_id=ids["document_b"],
            idx=99,
            kind=SegmentKind.paragraph,
            source_text="foreign",
            meta={},
        ),
    )
    await _expect_insert_rls_denied(
        conn,
        table="chunks",
        factory=lambda: Chunk(
            id=uuid.uuid4(),
            document_id=ids["document_b"],
            idx=99,
            text_en="foreign",
            text_ru="foreign",
            meta={},
        ),
    )
    await _expect_insert_rls_denied(
        conn,
        table="document_translations",
        factory=lambda: DocumentTranslation(
            id=uuid.uuid4(), document_id=ids["document_b"], target_lang="zh", data={}
        ),
    )
    await _expect_insert_rls_denied(
        conn,
        table="segment_versions",
        factory=lambda: SegmentVersion(
            id=uuid.uuid4(),
            segment_id=ids["segment_b"],
            document_id=ids["document_b"],
            old_text="foreign",
            new_text="foreign",
        ),
    )
    await _expect_insert_rls_denied(
        conn,
        table="chat_sessions",
        factory=lambda: ChatSession(id=uuid.uuid4(), title="foreign", owner_sub="owner-b"),
    )
    await _expect_insert_rls_denied(
        conn,
        table="chat_sessions",
        factory=lambda: ChatSession(id=uuid.uuid4(), title="null", owner_sub=None),
    )
    await _expect_insert_rls_denied(
        conn,
        table="chat_messages",
        factory=lambda: ChatMessage(
            id=uuid.uuid4(), session_id=ids["session_b"], role="user", content="foreign"
        ),
    )
    await _expect_insert_rls_denied(
        conn,
        table="page_embeddings",
        factory=lambda: PageEmbedding(
            id=uuid.uuid4(), document_id=ids["document_b"], page_idx=99, meta={}
        ),
    )
    await _expect_insert_rls_denied(
        conn,
        table="document_structured_artifacts",
        factory=lambda: _artifact(
            artifact_id=uuid.uuid4(),
            document_id=ids["document_b"],
            label=f"foreign-{uuid.uuid4()}",
            page_idx=99,
        ),
    )
    await _expect_insert_rls_denied(
        conn,
        table="audit_log",
        factory=lambda: AuditLog(id=uuid.uuid4(), user_sub="owner-b", action="rls_test"),
    )
    await _expect_insert_rls_denied(
        conn,
        table="memory_item_sources",
        factory=lambda: MemoryItemSource(
            item_id=ids["item_b"],
            event_id=ids["event_a"],
        ),
    )
    await _expect_insert_rls_denied(
        conn,
        table="memory_events",
        factory=lambda: MemoryEvent(
            id=uuid.uuid4(),
            tenant_id=ids["tenant"],
            user_id="owner-b",
            event_type="rls_test",
            payload={},
        ),
    )
    await _expect_insert_rls_denied(
        conn,
        table="memory_items",
        factory=lambda: MemoryItem(
            id=uuid.uuid4(),
            tenant_id=ids["tenant"],
            user_id="owner-b",
            scope="user",
            kind="fact",
            content="foreign",
            source_event_ids=[],
        ),
    )
    await _expect_insert_rls_denied(
        conn,
        table="memory_candidates",
        factory=lambda: MemoryCandidate(
            id=uuid.uuid4(),
            tenant_id=ids["tenant"],
            user_id="owner-b",
            action="create",
            proposed={"content": "foreign"},
            confidence=0.9,
        ),
    )
    await _expect_sql_rls_denied(
        conn,
        table="memory_audit_log",
        statement=(
            "INSERT INTO memory_audit_log "
            "(id, tenant_id, user_id, action, actor) OVERRIDING SYSTEM VALUE "
            "VALUES (:id, :tenant, 'owner-b', 'rls_test', 'test')"
        ),
        params={"id": -(uuid.uuid4().int % 1_000_000_000 + 10), "tenant": ids["tenant"]},
    )


async def _assert_forbidden_owner_and_fk_swaps(
    conn: AsyncConnection,
    ids: dict[str, Any],
) -> None:
    swaps = (
        (
            "documents",
            "UPDATE documents SET owner_sub = :owner_b WHERE id = :row_a",
            {"owner_b": "owner-b", "row_a": ids["document_a"]},
        ),
        (
            "folders",
            "UPDATE folders SET owner_sub = :owner_b WHERE id = :row_a",
            {"owner_b": "owner-b", "row_a": ids["folder_a"]},
        ),
        (
            "documents",
            "UPDATE documents SET folder_id = :parent_b WHERE id = :row_a",
            {"parent_b": ids["folder_b"], "row_a": ids["document_a"]},
        ),
        (
            "segments",
            "UPDATE segments SET document_id = :parent_b WHERE id = :row_a",
            {"parent_b": ids["document_b"], "row_a": ids["segment_a"]},
        ),
        (
            "chunks",
            "UPDATE chunks SET document_id = :parent_b WHERE id = :row_a",
            {"parent_b": ids["document_b"], "row_a": ids["chunk_a"]},
        ),
        (
            "document_translations",
            "UPDATE document_translations SET document_id = :parent_b WHERE id = :row_a",
            {"parent_b": ids["document_b"], "row_a": ids["translation_a"]},
        ),
        (
            "segment_versions",
            "UPDATE segment_versions SET document_id = :parent_b WHERE id = :row_a",
            {"parent_b": ids["document_b"], "row_a": ids["version_a"]},
        ),
        (
            "segment_versions",
            "UPDATE segment_versions SET segment_id = :parent_b WHERE id = :row_a",
            {"parent_b": ids["segment_b"], "row_a": ids["version_a"]},
        ),
        (
            "chat_sessions",
            "UPDATE chat_sessions SET document_id = :parent_b WHERE id = :row_a",
            {"parent_b": ids["document_b"], "row_a": ids["session_a"]},
        ),
        (
            "chat_sessions",
            "UPDATE chat_sessions SET folder_id = :parent_b WHERE id = :row_a",
            {"parent_b": ids["folder_b"], "row_a": ids["session_a"]},
        ),
        (
            "chat_messages",
            "UPDATE chat_messages SET session_id = :parent_b WHERE id = :row_a",
            {"parent_b": ids["session_b"], "row_a": ids["message_a"]},
        ),
        (
            "page_embeddings",
            "UPDATE page_embeddings SET document_id = :parent_b WHERE id = :row_a",
            {"parent_b": ids["document_b"], "row_a": ids["page_a"]},
        ),
        (
            "document_structured_artifacts",
            "UPDATE document_structured_artifacts SET document_id = :parent_b WHERE id = :row_a",
            {"parent_b": ids["document_b"], "row_a": ids["artifact_a"]},
        ),
        (
            "memory_item_sources",
            "UPDATE memory_item_sources SET event_id = :parent_b "
            "WHERE item_id = :row_a AND event_id = :event_a",
            {
                "parent_b": ids["event_b"],
                "row_a": ids["item_a"],
                "event_a": ids["event_a"],
            },
        ),
        (
            "memory_events",
            "UPDATE memory_events SET user_id = :owner_b WHERE id = :row_a",
            {"owner_b": "owner-b", "row_a": ids["event_a"]},
        ),
        (
            "memory_items",
            "UPDATE memory_items SET user_id = :owner_b WHERE id = :row_a",
            {"owner_b": "owner-b", "row_a": ids["item_a"]},
        ),
        (
            "memory_candidates",
            "UPDATE memory_candidates SET user_id = :owner_b WHERE id = :row_a",
            {"owner_b": "owner-b", "row_a": ids["candidate_a"]},
        ),
        (
            "memory_audit_log",
            "UPDATE memory_audit_log SET user_id = :owner_b WHERE id = :row_a",
            {"owner_b": "owner-b", "row_a": ids["memory_audit_a"]},
        ),
    )
    for table, statement, params in swaps:
        await _expect_sql_rls_denied(
            conn,
            table=table,
            statement=statement,
            params=params,
        )


async def _run_postgres_rls_test(admin_url: str) -> None:
    engine = create_async_engine(
        admin_url,
        pool_size=1,
        max_overflow=0,
        pool_pre_ping=True,
    )
    assert engine.pool.size() == 1
    try:
        async with engine.connect() as conn:
            transaction = await conn.begin()
            try:
                database = await conn.scalar(text("SELECT current_database()"))
                assert database is not None and "rls_test" in database.lower()
                can_bypass = await conn.scalar(
                    text(
                        "SELECT rolsuper OR rolbypassrls FROM pg_roles "
                        "WHERE rolname = current_user"
                    )
                )
                assert can_bypass is True, "admin URL must use a superuser or BYPASSRLS role"
                api_role = (
                    await conn.execute(
                        text(
                            "SELECT rolsuper, rolbypassrls FROM pg_roles "
                            "WHERE rolname = :role"
                        ),
                        {"role": _ROLE},
                    )
                ).one_or_none()
                assert api_role == (False, False), "rag_api must exist and must not bypass RLS"
                worker_role = (
                    await conn.execute(
                        text(
                            "SELECT rolsuper, rolbypassrls FROM pg_roles "
                            "WHERE rolname = 'rag_worker'"
                        )
                    )
                ).one_or_none()
                assert worker_role == (False, True)
                for role in ("rag_api", "rag_worker"):
                    for table in PROTECTED_TABLES:
                        for privilege in ("SELECT", "INSERT", "UPDATE", "DELETE"):
                            granted = await conn.scalar(
                                text("SELECT has_table_privilege(:role, :table, :privilege)"),
                                {"role": role, "table": table, "privilege": privilege},
                            )
                            expected = not (
                                role == "rag_api"
                                and table == "audit_log"
                                and privilege in {"UPDATE", "DELETE"}
                            )
                            assert bool(granted) is expected, (
                                f"unexpected {role} {privilege} privilege on {table}"
                            )
                    assert await conn.scalar(
                        text(
                            "SELECT has_sequence_privilege"
                            "(:role, 'memory_audit_log_id_seq', 'USAGE')"
                        ),
                        {"role": role},
                    )

                nullable_owner_columns = await conn.scalar(
                    text(
                        "SELECT count(*) FROM information_schema.columns "
                        "WHERE table_schema = current_schema() "
                        "AND table_name IN ('documents', 'folders', 'chat_sessions') "
                        "AND column_name = 'owner_sub' AND is_nullable = 'YES'"
                    )
                )
                assert nullable_owner_columns == 0
                # Exercise the policy independently from the NOT NULL guard.
                # The outer rollback restores these transactional DDL changes.
                for table in ("documents", "folders", "chat_sessions"):
                    await conn.exec_driver_sql(
                        f"ALTER TABLE {table} ALTER COLUMN owner_sub DROP NOT NULL"
                    )

                ids = _ids()
                tenant_id = uuid.uuid4()
                ids["tenant"] = tenant_id
                await _seed(conn, ids, tenant_id)

                await _assert_visibility(
                    conn,
                    ids,
                    principal="owner-a",
                    is_admin=False,
                    tenant_id=tenant_id,
                    expected_suffix="_a",
                )
                await _assert_visibility(
                    conn,
                    ids,
                    principal="owner-b",
                    is_admin=False,
                    tenant_id=tenant_id,
                    expected_suffix="_b",
                )
                await _assert_visibility(
                    conn,
                    ids,
                    principal="rls-admin",
                    is_admin=True,
                    tenant_id=tenant_id,
                    expected_suffix=None,
                )
                await _assert_visibility(
                    conn,
                    ids,
                    principal="",
                    is_admin=False,
                    tenant_id=None,
                    expected_suffix=None,
                )

                await _set_api_principal(
                    conn,
                    user_sub="owner-a",
                    is_admin=False,
                    tenant_id=tenant_id,
                )
                await _assert_foreign_update_delete_hidden(conn, ids)
                await _assert_forbidden_inserts(conn, ids)
                await _assert_forbidden_owner_and_fk_swaps(conn, ids)
                await _expect_sql_permission_denied(
                    conn,
                    statement="UPDATE audit_log SET action = action WHERE id = :row_a",
                    params={"row_a": ids["audit_a"]},
                )
                await _expect_sql_permission_denied(
                    conn,
                    statement="DELETE FROM audit_log WHERE id = :row_a",
                    params={"row_a": ids["audit_a"]},
                )
            finally:
                await transaction.rollback()
    finally:
        await engine.dispose()


def test_fail_closed_rls_with_real_postgres() -> None:
    raw_url = os.getenv(_ADMIN_URL_ENV)
    if not raw_url:
        pytest.skip(f"set {_ADMIN_URL_ENV} to run the isolated PostgreSQL RLS test")
    asyncio.run(_run_postgres_rls_test(_async_admin_url(raw_url)))
