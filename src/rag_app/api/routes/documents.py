from __future__ import annotations

import io
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select

from rag_app.api.schemas import DocumentOut
from rag_app.config import settings
from rag_app.db.models import Document, DocumentStatus, Segment

router = APIRouter(prefix="/api/documents", tags=["documents"])

# Этап 2: PDF (текст и сканы) + OOXML. TXT/HTML — позже (roadmap § 3.1).
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".pptx"}
_SIGNATURES = {".pdf": b"%PDF", ".docx": b"PK", ".xlsx": b"PK", ".pptx": b"PK"}


@router.post("", response_model=DocumentOut, status_code=201)
async def upload_document(request: Request, file: UploadFile) -> DocumentOut:
    filename = file.filename or "document.pdf"
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(415, f"поддерживаются {allowed}, получено: {ext or 'без расширения'}")

    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"файл больше {settings.max_upload_mb} МБ")
    if not data.startswith(_SIGNATURES[ext]):
        raise HTTPException(415, f"содержимое не похоже на {ext}")

    doc_id = uuid.uuid4()
    s3_key = f"{doc_id}/{filename}"
    await request.app.state.storage.put_bytes(
        settings.bucket_originals, s3_key, data, content_type="application/pdf"
    )

    async with request.app.state.sessionmaker() as session:
        doc = Document(
            id=doc_id,
            filename=filename,
            content_type=file.content_type or "application/pdf",
            size_bytes=len(data),
            s3_key_original=s3_key,
        )
        session.add(doc)
        await session.commit()
        await session.refresh(doc)

    await request.app.state.arq.enqueue_job(
        "parse_document", str(doc_id), _job_id=f"parse:{doc_id}:{uuid.uuid4().hex[:8]}"
    )
    return DocumentOut.from_doc(doc)


@router.get("", response_model=list[DocumentOut])
async def list_documents(request: Request) -> list[DocumentOut]:
    async with request.app.state.sessionmaker() as session:
        docs = (
            (await session.execute(select(Document).order_by(Document.created_at.desc()).limit(200)))
            .scalars()
            .all()
        )
        reviews = dict(
            (
                await session.execute(
                    select(Segment.document_id, func.count())
                    .where(Segment.needs_review)
                    .group_by(Segment.document_id)
                )
            ).all()
        )
    return [DocumentOut.from_doc(d, reviews.get(d.id, 0)) for d in docs]


@router.get("/{doc_id}", response_model=DocumentOut)
async def get_document(request: Request, doc_id: uuid.UUID) -> DocumentOut:
    doc = await _get_or_404(request, doc_id)
    async with request.app.state.sessionmaker() as session:
        review_count = (
            await session.execute(
                select(func.count())
                .select_from(Segment)
                .where(Segment.document_id == doc_id, Segment.needs_review)
            )
        ).scalar_one()
    return DocumentOut.from_doc(doc, review_count)


@router.post("/{doc_id}/retry", response_model=DocumentOut)
async def retry_document(request: Request, doc_id: uuid.UUID) -> DocumentOut:
    """Перезапуск пайплайна с парсинга (после ошибки)."""
    doc = await _get_or_404(request, doc_id)
    if doc.status not in (DocumentStatus.error, DocumentStatus.done):
        raise HTTPException(409, f"документ в работе (статус {doc.status.value})")
    await request.app.state.arq.enqueue_job(
        "parse_document", str(doc_id), _job_id=f"parse:{doc_id}:{uuid.uuid4().hex[:8]}"
    )
    return DocumentOut.from_doc(doc)


_EXPORT_KINDS = {
    "docx": ("s3_key_export_docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    "pdf": ("s3_key_export_pdf", "application/pdf"),
    "pdf_dual": ("s3_key_export_pdf_dual", "application/pdf"),
    "source": ("s3_key_export_source", "application/octet-stream"),
}


@router.get("/{doc_id}/download/{kind}")
async def download(request: Request, doc_id: uuid.UUID, kind: str) -> StreamingResponse:
    doc = await _get_or_404(request, doc_id)
    if kind == "original":
        bucket, key = settings.bucket_originals, doc.s3_key_original
        media = doc.content_type or "application/octet-stream"
        out_name = doc.filename
    elif kind in _EXPORT_KINDS:
        attr, media = _EXPORT_KINDS[kind]
        key = getattr(doc, attr)
        if not key:
            raise HTTPException(404, f"экспорт «{kind}» ещё не готов")
        bucket = settings.bucket_exports
        out_name = Path(key).name
    else:
        raise HTTPException(404, f"kind должен быть один из: original, {', '.join(_EXPORT_KINDS)}")

    data = await request.app.state.storage.get_bytes(bucket, key)
    quoted = out_name.encode("ascii", "ignore").decode() or "document"
    return StreamingResponse(
        io.BytesIO(data),
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{quoted}"'},
    )


async def _get_or_404(request: Request, doc_id: uuid.UUID) -> Document:
    async with request.app.state.sessionmaker() as session:
        doc = await session.get(Document, doc_id)
    if doc is None:
        raise HTTPException(404, "документ не найден")
    return doc
