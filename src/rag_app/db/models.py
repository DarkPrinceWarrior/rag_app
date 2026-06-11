"""Схема БД (этап 1): documents + segments.

Дальше по roadmap добавятся: users, versions, glossary, chat_sessions,
chunks(+vector), audit_log. Миграции (alembic) вводим на этапе 2,
пока — create_all при старте.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, DateTime, Enum, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class DocumentStatus(enum.StrEnum):
    uploaded = "uploaded"
    parsing = "parsing"
    parsed = "parsed"
    translating = "translating"
    translated = "translated"
    exporting = "exporting"
    done = "done"
    error = "error"


class SegmentKind(enum.StrEnum):
    heading = "heading"
    paragraph = "paragraph"
    table = "table"
    equation = "equation"
    image = "image"


# Виды сегментов, которые отправляются на перевод.
TRANSLATABLE_KINDS = (SegmentKind.heading, SegmentKind.paragraph, SegmentKind.table, SegmentKind.image)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename: Mapped[str] = mapped_column(String(512))
    content_type: Mapped[str] = mapped_column(String(128), default="application/octet-stream")
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus, name="document_status"), default=DocumentStatus.uploaded
    )
    error: Mapped[str | None] = mapped_column(Text, default=None)

    s3_key_original: Mapped[str] = mapped_column(String(1024))
    s3_key_content_list: Mapped[str | None] = mapped_column(String(1024), default=None)
    s3_key_export_docx: Mapped[str | None] = mapped_column(String(1024), default=None)

    page_count: Mapped[int | None] = mapped_column(Integer, default=None)
    segment_count: Mapped[int] = mapped_column(Integer, default=0)
    translated_count: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    segments: Mapped[list[Segment]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )


class Segment(Base):
    __tablename__ = "segments"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    idx: Mapped[int] = mapped_column(Integer)  # порядок в документе
    page_idx: Mapped[int | None] = mapped_column(Integer, default=None)
    kind: Mapped[SegmentKind] = mapped_column(Enum(SegmentKind, name="segment_kind"))
    heading_level: Mapped[int | None] = mapped_column(Integer, default=None)
    source_text: Mapped[str] = mapped_column(Text, default="")
    translated_text: Mapped[str | None] = mapped_column(Text, default=None)
    # bbox, table_rows / table_rows_ru, подписи и т.п.
    meta: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)

    document: Mapped[Document] = relationship(back_populates="segments")
