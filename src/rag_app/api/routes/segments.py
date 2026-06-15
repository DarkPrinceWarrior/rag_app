"""Сегменты: side-by-side просмотр и правки переводов (roadmap § 6 PATCH /segments)."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select, update

from rag_app.api.audit import audit
from rag_app.api.auth import require_user
from rag_app.db.models import Document, DocumentStatus, Segment

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
        return out


class SegmentPatch(BaseModel):
    translated_text: str


@router.get("/documents/{doc_id}/segments", response_model=list[SegmentOut])
async def list_segments(request: Request, doc_id: uuid.UUID) -> list[SegmentOut]:
    async with request.app.state.sessionmaker() as session:
        if await session.get(Document, doc_id) is None:
            raise HTTPException(404, "документ не найден")
        segments = (
            (
                await session.execute(
                    select(Segment).where(Segment.document_id == doc_id).order_by(Segment.idx)
                )
            )
            .scalars()
            .all()
        )
    return [SegmentOut.from_segment(s) for s in segments]


@router.patch("/segments/{segment_id}", response_model=SegmentOut)
async def patch_segment(request: Request, segment_id: uuid.UUID, body: SegmentPatch) -> SegmentOut:
    async with request.app.state.sessionmaker() as session:
        seg = await session.get(Segment, segment_id)
        if seg is None:
            raise HTTPException(404, "сегмент не найден")
        seg.translated_text = body.translated_text
        seg.needs_review = False  # ручная правка снимает флаг валидации
        await session.commit()
        await session.refresh(seg)
    await audit(request, "segment_edit", "segment", str(segment_id), {"document_id": str(seg.document_id)})
    return SegmentOut.model_validate(seg)


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
