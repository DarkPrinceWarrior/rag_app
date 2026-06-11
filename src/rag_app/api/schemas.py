from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from rag_app.db.models import Document, DocumentStatus


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    kind: str
    size_bytes: int
    status: DocumentStatus
    error: str | None
    page_count: int | None
    segment_count: int
    translated_count: int
    review_count: int = 0  # сегменты с needs_review (числовая валидация)
    exports: list[str] = []  # доступные виды скачивания (kind для /download/{kind})
    folder_id: uuid.UUID | None = None
    chunk_count: int = 0  # RAG-индекс
    created_at: datetime

    @classmethod
    def from_doc(cls, doc: Document, review_count: int = 0) -> DocumentOut:
        out = cls.model_validate(doc)
        out.review_count = review_count
        out.exports = [
            kind
            for kind, attr in (
                ("docx", "s3_key_export_docx"),
                ("pdf", "s3_key_export_pdf"),
                ("pdf_dual", "s3_key_export_pdf_dual"),
                ("source", "s3_key_export_source"),
            )
            if getattr(doc, attr)
        ]
        return out
