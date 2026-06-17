from __future__ import annotations

import io
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy import text as sql

from rag_app.api.audit import audit
from rag_app.api.auth import User, require_user
from rag_app.api.schemas import DocumentOut
from rag_app.config import settings
from rag_app.db.models import Document, DocumentStatus, Segment
from rag_app.rag.memory.rls import apply_scope_guc

router = APIRouter(prefix="/api/documents", tags=["documents"], dependencies=[require_user])


def _owner_filter(stmt, user: User):
    """RBAC: admin видит всё; user — свои + документы dev-периода (owner NULL)."""
    if user.is_admin:
        return stmt
    return stmt.where((Document.owner_sub == user.sub) | (Document.owner_sub.is_(None)))

# ТЗ §4.2: PDF (текст/скан), OOXML, изображения документов (JPG/PNG → OCR-ветка),
# plain-текст (TXT). Изображения оборачиваются в 1-страничный PDF на загрузке.
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".pptx", ".jpg", ".jpeg", ".png", ".txt"}
_SIGNATURES = {
    ".pdf": b"%PDF",
    ".docx": b"PK",
    ".xlsx": b"PK",
    ".pptx": b"PK",
    ".jpg": b"\xff\xd8\xff",
    ".jpeg": b"\xff\xd8\xff",
    ".png": b"\x89PNG",
}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


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
    sig = _SIGNATURES.get(ext)
    if sig and not data.startswith(sig):
        raise HTTPException(415, f"содержимое не похоже на {ext}")

    content_type = file.content_type or "application/octet-stream"
    # ТЗ §4.2: изображение документа/чертежа → 1-страничный PDF, дальше как скан
    if ext in _IMAGE_EXTS:
        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PDF", resolution=150.0)
        except Exception as exc:
            raise HTTPException(415, f"не удалось прочитать изображение: {exc}") from None
        data = buf.getvalue()
        filename = Path(filename).stem + ".pdf"
        content_type = "application/pdf"

    doc_id = uuid.uuid4()
    s3_key = f"{doc_id}/{filename}"
    await request.app.state.storage.put_bytes(
        settings.bucket_originals, s3_key, data, content_type=content_type
    )

    async with request.app.state.sessionmaker() as session:
        doc = Document(
            id=doc_id,
            owner_sub=request.state.user.sub if settings.auth_enabled else None,
            filename=filename,
            content_type=content_type,
            size_bytes=len(data),
            s3_key_original=s3_key,
        )
        session.add(doc)
        await session.commit()
        await session.refresh(doc)

    await request.app.state.arq.enqueue_job(
        "parse_document", str(doc_id), _job_id=f"parse:{doc_id}:{uuid.uuid4().hex[:8]}"
    )
    await audit(request, "upload", "document", str(doc_id), {"filename": filename, "bytes": len(data)})
    return DocumentOut.from_doc(doc)


@router.get("", response_model=list[DocumentOut])
async def list_documents(request: Request) -> list[DocumentOut]:
    async with request.app.state.sessionmaker() as session:
        stmt = _owner_filter(select(Document), request.state.user)
        docs = (
            (await session.execute(stmt.order_by(Document.created_at.desc()).limit(200)))
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
    await audit(request, "retry", "document", str(doc_id))
    return DocumentOut.from_doc(doc)


@router.post("/{doc_id}/describe")
async def describe_document(request: Request, doc_id: uuid.UUID) -> dict:
    """Запуск VL-описания рисунков документа (скан/чертёж/P&ID) по требованию.
    Раскрывает смысл изображений текстом и переиндексирует (см. describe_images)."""
    doc = await _get_or_404(request, doc_id)
    if not settings.vl_enabled:
        raise HTTPException(409, "VL-описание выключено (vl_enabled=false)")
    await request.app.state.arq.enqueue_job(
        "describe_images", str(doc_id), _job_id=f"vl:{doc_id}:{uuid.uuid4().hex[:8]}"
    )
    await audit(request, "describe", "document", str(doc_id))
    return {"status": "queued", "kind": doc.kind}


class ReparseOcrIn(BaseModel):
    # en | east_slavic (рус/укр/бел) | cyrillic | latin | ch | … (см. mineru -l)
    lang: str = "east_slavic"


@router.post("/{doc_id}/reparse-ocr")
async def reparse_ocr(request: Request, doc_id: uuid.UUID, body: ReparseOcrIn) -> dict:
    """Переразбор через форс-OCR — восстановление PDF с битым ToUnicode-cmap
    текстового слоя (MinerU `-m ocr -l <lang>`); выбор сохраняется на документе."""
    async with request.app.state.sessionmaker() as session:
        doc = await session.get(Document, doc_id)
        if doc is None:
            raise HTTPException(404, "документ не найден")
        if doc.status not in (DocumentStatus.error, DocumentStatus.done):
            raise HTTPException(409, f"документ в работе (статус {doc.status.value})")
        doc.parse_force_ocr = True
        doc.ocr_lang = body.lang
        doc.status = DocumentStatus.uploaded
        await session.commit()
    await request.app.state.arq.enqueue_job(
        "parse_document", str(doc_id), _job_id=f"parse:{doc_id}:{uuid.uuid4().hex[:8]}"
    )
    await audit(request, "reparse_ocr", "document", str(doc_id), {"lang": body.lang})
    return {"status": "queued", "ocr_lang": body.lang}


_EXPORT_KINDS = {
    "docx": ("s3_key_export_docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    "pdf": ("s3_key_export_pdf", "application/pdf"),
    "pdf_dual": ("s3_key_export_pdf_dual", "application/pdf"),
    "source": ("s3_key_export_source", "application/octet-stream"),
}


@router.get("/{doc_id}/download/{kind}")
async def download(request: Request, doc_id: uuid.UUID, kind: str) -> Response:
    doc = await _get_or_404(request, doc_id)
    if kind == "original":
        bucket, key = settings.bucket_originals, doc.s3_key_original
        media = doc.content_type or "application/octet-stream"
        out_name = doc.filename
    elif kind in ("view_orig", "view_ru"):
        # PDF-рендер OOXML для просмотра «как в Microsoft» (LibreOffice)
        key = doc.s3_key_view_orig if kind == "view_orig" else doc.s3_key_view_ru
        if not key:
            raise HTTPException(404, "PDF-просмотр не готов")
        bucket, media, out_name = settings.bucket_exports, "application/pdf", Path(key).name
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
    await audit(request, "download", "document", str(doc_id), {"kind": kind})
    quoted = out_name.encode("ascii", "ignore").decode() or "document"
    total = len(data)
    base = {
        "Content-Disposition": f'attachment; filename="{quoted}"',
        "Accept-Ranges": "bytes",  # pdf.js тянет страницы лениво порейндж-запросами
    }
    # HTTP Range (RFC 7233): большие PDF открываются сразу — pdf.js запрашивает
    # сначала структуру (хвост файла), затем страницы по мере листания, а не весь
    # файл целиком. Без Range отдаём всё, но с Accept-Ranges.
    rng = request.headers.get("range", "")
    if rng.startswith("bytes="):
        try:
            s, _, e = rng[6:].partition("-")
            start = int(s) if s else 0
            end = int(e) if e else total - 1
            end = min(end, total - 1)
            if start > end or start >= total:
                raise ValueError
        except ValueError:
            return Response(status_code=416, headers={"Content-Range": f"bytes */{total}", **base})
        chunk = data[start : end + 1]
        return Response(
            content=chunk,
            status_code=206,
            media_type=media,
            headers={**base, "Content-Range": f"bytes {start}-{end}/{total}", "Content-Length": str(len(chunk))},
        )
    return Response(content=data, media_type=media, headers={**base, "Content-Length": str(total)})


@router.delete("/{doc_id}", status_code=204)
async def delete_document(request: Request, doc_id: uuid.UUID) -> None:
    """Удаление документа из библиотеки: объекты в S3 (оригинал, артефакт парсинга,
    экспорты) + запись в БД (каскадом — сегменты, чанки, эмбеддинги страниц,
    чат-сессии этого документа) + чистка памяти, привязанной к документу."""
    doc = await _get_or_404(request, doc_id)
    storage = request.app.state.storage

    # 1) S3 — best-effort по всем известным ключам документа
    await storage.remove_object(settings.bucket_originals, doc.s3_key_original)
    if doc.s3_key_content_list:
        await storage.remove_object(settings.bucket_artifacts, doc.s3_key_content_list)
    for attr, _media in _EXPORT_KINDS.values():
        key = getattr(doc, attr)
        if key:
            await storage.remove_object(settings.bucket_exports, key)

    # 2) БД — удаление документа (FK ondelete=CASCADE уносит segments/chunks/
    #    page_embeddings/chat_sessions+messages этого документа)
    async with request.app.state.sessionmaker() as session:
        obj = await session.get(Document, doc_id)
        if obj is not None:
            await session.delete(obj)
            await session.commit()

    # 3) Память документа (слой памяти расцеплён с FK) — мягкое удаление; под
    #    RLS FORCE нужен GUC скоупа владельца. Никогда не блокирует удаление.
    user: User = request.state.user
    try:
        memory = request.app.state.memory
        async with request.app.state.sessionmaker() as session:
            await apply_scope_guc(session, memory.scope_for(user.sub, document_id=doc_id))
            await session.execute(
                sql(
                    "UPDATE memory_items SET status='deleted', deleted_at=now()"
                    " WHERE user_id=:u AND document_id=:d AND status<>'deleted'"
                ),
                {"u": user.sub, "d": str(doc_id)},
            )
            await session.commit()
    except Exception:  # noqa: BLE001 — чистка памяти не должна валить удаление
        pass

    await audit(request, "delete", "document", str(doc_id), {"filename": doc.filename})


async def _get_or_404(request: Request, doc_id: uuid.UUID) -> Document:
    async with request.app.state.sessionmaker() as session:
        doc = await session.get(Document, doc_id)
    if doc is None:
        raise HTTPException(404, "документ не найден")
    user: User = request.state.user
    if not user.is_admin and doc.owner_sub is not None and doc.owner_sub != user.sub:
        raise HTTPException(404, "документ не найден")  # не раскрываем существование
    return doc
