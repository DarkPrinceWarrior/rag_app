"""Сегменты: side-by-side просмотр и правки переводов (roadmap § 6 PATCH /segments)."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select, update

from rag_app.api.audit import audit
from rag_app.api.auth import User, require_user
from rag_app.db.models import Document, DocumentStatus, Segment, SegmentVersion

router = APIRouter(prefix="/api", tags=["segments"], dependencies=[require_user])


class SegmentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    idx: int
    page_idx: int | None
    kind: str
    heading_level: int | None
    source_text: str
    translated_text: str | None
    needs_review: bool
    validation: dict | None
    # геометрия для pdf.js-вьювера (§ 7): bbox в пунктах + размер страницы
    bbox: list[float] | None = None
    page_size: list[float] | None = None
    # ячейки таблицы со спанами — для рендера объединённых ячеек в UI:
    # table_cells — оригинал, table_cells_ru — перевод по позиции ячейки
    table_cells: list[list[dict]] | None = None
    table_cells_ru: list[list[dict]] | None = None
    caption: str | None = None
    caption_ru: str | None = None
    # URL картинки сегмента-рисунка (извлечена из оригинала) — для MD-просмотра
    image_url: str | None = None
    # адрес узла OOXML (DOCX: {"p"} абзац / {"t","r","c","p"} ячейка) —
    # для реконструкции таблиц в MD-просмотре DOCX
    location: dict | None = None
    # положение сегмента в ЛЕВОМ (оригинал) и ПРАВОМ (перевод) рендер-PDF —
    # {page, bbox(top-left,pt), pagesize} — для кросс-навигации по клику (pdf_text/docx)
    loc_left: dict | None = None
    loc_right: dict | None = None

    @classmethod
    def from_segment(cls, s: Segment) -> SegmentOut:
        out = cls.model_validate(s)
        meta = s.meta or {}
        out.bbox = meta.get("bbox_pt")
        out.page_size = meta.get("page_size_pt")
        out.table_cells = meta.get("table_cells")
        out.table_cells_ru = meta.get("table_cells_ru")
        out.caption = meta.get("caption")
        out.caption_ru = meta.get("caption_ru")
        out.location = meta.get("location")
        out.loc_left = meta.get("loc_left")
        out.loc_right = meta.get("loc_right")
        img = meta.get("img_s3")
        if img:
            out.image_url = f"/api/documents/{s.document_id}/image/{img.rsplit('/', 1)[-1]}"
        return out


class SegmentPatch(BaseModel):
    translated_text: str


@router.get("/documents/{doc_id}/segments", response_model=list[SegmentOut])
async def list_segments(
    request: Request,
    doc_id: uuid.UUID,
    limit: int = Query(4000, ge=1, le=100_000),
) -> list[SegmentOut]:
    # Бэкстоп против патологических документов (xlsx-дата-дампы на сотни тысяч
    # ячеек вешали вьювер): отдаём первые `limit` сегментов по idx. Дефолт
    # большой, но конечный — обычные документы (тысячи сегментов) не обрезаются.
    async with request.app.state.sessionmaker() as session:
        if await session.get(Document, doc_id) is None:
            raise HTTPException(404, "документ не найден")
        segments = (
            (
                await session.execute(
                    select(Segment)
                    .where(Segment.document_id == doc_id)
                    .order_by(Segment.idx)
                    .limit(limit)
                )
            )
            .scalars()
            .all()
        )
    return [SegmentOut.from_segment(s) for s in segments]


async def _segment_or_404(session, segment_id: uuid.UUID, user: User) -> Segment:
    seg = await session.get(Segment, segment_id)
    if seg is None:
        raise HTTPException(404, "сегмент не найден")
    doc = await session.get(Document, seg.document_id)  # RBAC §4.7.1
    if doc is not None and not user.is_admin and doc.owner_sub is not None and doc.owner_sub != user.sub:
        raise HTTPException(404, "сегмент не найден")
    return seg


@router.patch("/segments/{segment_id}", response_model=SegmentOut)
async def patch_segment(request: Request, segment_id: uuid.UUID, body: SegmentPatch) -> SegmentOut:
    user: User = request.state.user
    async with request.app.state.sessionmaker() as session:
        seg = await _segment_or_404(session, segment_id, user)
        old = seg.translated_text
        changed = old != body.translated_text
        if changed:  # история правок (ТЗ §4.7.2): версионируем только реальное изменение
            session.add(
                SegmentVersion(
                    segment_id=seg.id, document_id=seg.document_id,
                    old_text=old, new_text=body.translated_text,
                    editor_sub=user.sub, editor_name=user.username,
                )
            )
        seg.translated_text = body.translated_text
        seg.needs_review = False  # ручная правка снимает флаг валидации
        await session.commit()
        await session.refresh(seg)
    await audit(
        request, "segment_edit", "segment", str(segment_id),
        {"document_id": str(seg.document_id), "changed": changed},
    )
    return SegmentOut.model_validate(seg)


@router.get("/segments/{segment_id}/versions")
async def segment_versions(request: Request, segment_id: uuid.UUID) -> list[dict]:
    """История правок перевода сегмента (ТЗ §4.7.2): было→стало, кто, когда."""
    user: User = request.state.user
    async with request.app.state.sessionmaker() as session:
        await _segment_or_404(session, segment_id, user)
        rows = (
            (
                await session.execute(
                    select(SegmentVersion)
                    .where(SegmentVersion.segment_id == segment_id)
                    .order_by(SegmentVersion.created_at.desc())
                )
            )
            .scalars()
            .all()
        )
    return [
        {
            "id": str(v.id),
            "old_text": v.old_text,
            "new_text": v.new_text,
            "editor": v.editor_name or v.editor_sub or "—",
            "created_at": v.created_at.isoformat(),
        }
        for v in rows
    ]


@router.post("/documents/{doc_id}/reexport")
async def reexport_document(request: Request, doc_id: uuid.UUID) -> dict:
    """Пересборка экспортов после ручных правок сегментов."""
    async with request.app.state.sessionmaker() as session:
        doc = await session.get(Document, doc_id)
        if doc is None:
            raise HTTPException(404, "документ не найден")
        if doc.status not in (DocumentStatus.done, DocumentStatus.error):
            raise HTTPException(409, f"документ в работе (статус {doc.status.value})")
        await session.execute(
            update(Document).where(Document.id == doc_id).values(status=DocumentStatus.translated)
        )
        await session.commit()
    await request.app.state.arq.enqueue_job(
        "export_document", str(doc_id), _job_id=f"export:{doc_id}:{uuid.uuid4().hex[:8]}"
    )
    return {"status": "queued"}
