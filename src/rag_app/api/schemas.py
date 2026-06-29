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
    has_view: bool = False  # PDF-рендер OOXML готов целиком (оригинал И перевод)
    # раздельно: оригинал рендерится рано (после парсинга), перевод — на экспорте.
    # Вьювер показывает «как в Microsoft» по оригиналу сразу, не дожидаясь перевода.
    has_view_orig: bool = False
    has_view_ru: bool = False
    # движок парсинга pdf_text: null → дефолт (mineru). mineru | dots_mocr | paddle_vl
    parser_backend: str | None = None
    # язык-источник, определённый автоматически (ru|en|zh; "auto" — ещё не определён).
    # Цель перевода всегда русский. Для бейджа направления в библиотеке.
    source_lang: str | None = None
    # метаданные (ТЗ §4.7.2/§4.7.3): тип источника + объект строительства
    source_type: str = "file"
    project_object: str | None = None
    # PNG первой страницы для карточки библиотеки: рендерится из PDF/view_orig.
    preview_url: str | None = None
    created_at: datetime

    @classmethod
    def from_doc(cls, doc: Document, review_count: int = 0) -> DocumentOut:
        out = cls.model_validate(doc)
        out.review_count = review_count
        out.has_view_orig = bool(doc.s3_key_view_orig)
        out.has_view_ru = bool(doc.s3_key_view_ru)
        out.has_view = out.has_view_orig and out.has_view_ru
        if out.has_view_orig or (doc.content_type == "application/pdf") or doc.filename.lower().endswith(".pdf"):
            out.preview_url = f"/api/documents/{doc.id}/preview.png"
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
