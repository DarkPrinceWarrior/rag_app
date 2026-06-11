from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from rag_app.db.models import Document, DocumentStatus


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    size_bytes: int
    status: DocumentStatus
    error: str | None
    page_count: int | None
    segment_count: int
    translated_count: int
    has_docx: bool = False
    created_at: datetime

    @classmethod
    def from_doc(cls, doc: Document) -> DocumentOut:
        out = cls.model_validate(doc)
        out.has_docx = doc.s3_key_export_docx is not None
        return out
