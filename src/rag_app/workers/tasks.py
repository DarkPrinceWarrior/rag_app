"""ARQ-задачи: parse_document → translate_document → export_document.

Цепочка статусов: uploaded → parsing → parsed → translating → translated
→ exporting → done; любая ошибка → status=error + текст в documents.error.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select, update

from rag_app.config import settings
from rag_app.db.models import (
    TRANSLATABLE_KINDS,
    Document,
    DocumentStatus,
    Segment,
    SegmentKind,
)
from rag_app.llm.client import SegmentContext, Translator, needs_translation
from rag_app.pipeline.export_docx import build_docx
from rag_app.pipeline.parse import NoTextLayerError, load_content_list, pdf_info, run_mineru
from rag_app.pipeline.segments import content_list_to_segments
from rag_app.storage.s3 import Storage

logger = logging.getLogger(__name__)


async def _set_status(ctx: dict, doc_id: uuid.UUID, status: DocumentStatus, error: str | None = None) -> None:
    async with ctx["sessionmaker"]() as session:
        await session.execute(
            update(Document).where(Document.id == doc_id).values(status=status, error=error)
        )
        await session.commit()


async def _get_doc(ctx: dict, doc_id: uuid.UUID) -> Document:
    async with ctx["sessionmaker"]() as session:
        doc = await session.get(Document, doc_id)
        if doc is None:
            raise RuntimeError(f"документ {doc_id} не найден")
        return doc


# ---------------------------------------------------------------- parse

async def parse_document(ctx: dict, doc_id_str: str) -> str:
    doc_id = uuid.UUID(doc_id_str)
    storage: Storage = ctx["storage"]
    doc = await _get_doc(ctx, doc_id)
    await _set_status(ctx, doc_id, DocumentStatus.parsing)
    logger.info("parse %s (%s)", doc_id, doc.filename)

    try:
        with tempfile.TemporaryDirectory(prefix="rag_parse_") as tmp:
            tmp_path = Path(tmp)
            local_pdf = tmp_path / (Path(doc.filename).stem + ".pdf")
            await storage.download_to(settings.bucket_originals, doc.s3_key_original, local_pdf)

            n_pages, has_text = await asyncio.to_thread(pdf_info, local_pdf)
            if not has_text:
                raise NoTextLayerError(
                    "PDF без текстового слоя (скан). OCR-ветка появится на этапе 2."
                )

            out_dir = tmp_path / "mineru_out"
            content_list_path = await run_mineru(local_pdf, out_dir)
            items = load_content_list(content_list_path)
            drafts = content_list_to_segments(items)
            if not drafts:
                raise RuntimeError("парсер не извлёк ни одного блока")

            artifact_key = f"{doc_id}/content_list.json"
            await storage.put_bytes(
                settings.bucket_artifacts,
                artifact_key,
                content_list_path.read_bytes(),
                content_type="application/json",
            )

        async with ctx["sessionmaker"]() as session:
            await session.execute(delete(Segment).where(Segment.document_id == doc_id))
            session.add_all(
                Segment(
                    document_id=doc_id,
                    idx=d.idx,
                    page_idx=d.page_idx,
                    kind=d.kind,
                    heading_level=d.heading_level,
                    source_text=d.source_text,
                    meta=d.meta,
                )
                for d in drafts
            )
            await session.execute(
                update(Document)
                .where(Document.id == doc_id)
                .values(
                    status=DocumentStatus.parsed,
                    error=None,
                    page_count=n_pages,
                    segment_count=len(drafts),
                    translated_count=0,
                    s3_key_content_list=artifact_key,
                )
            )
            await session.commit()

        await ctx["redis"].enqueue_job(
            "translate_document", doc_id_str, _job_id=f"translate:{doc_id}:{uuid.uuid4().hex[:8]}"
        )
        return f"parsed: {len(drafts)} segments, {n_pages} pages"

    except Exception as exc:
        logger.exception("parse %s failed", doc_id)
        await _set_status(ctx, doc_id, DocumentStatus.error, f"парсинг: {exc}")
        raise


# ---------------------------------------------------------------- translate

async def _translate_segment(translator: Translator, seg: Segment, context: SegmentContext) -> dict[str, Any]:
    """Возвращает values для UPDATE сегмента."""
    if seg.kind == SegmentKind.table:
        rows: list[list[str]] = seg.meta.get("table_rows") or []
        caption: str = seg.meta.get("caption") or ""
        rows_ru = [
            [await translator.translate(cell, context) if needs_translation(cell) else cell for cell in row]
            for row in rows
        ]
        caption_ru = await translator.translate(caption, context) if needs_translation(caption) else caption
        meta = dict(seg.meta)
        meta["table_rows_ru"] = rows_ru
        meta["caption_ru"] = caption_ru
        preview = "\n".join(" | ".join(r) for r in rows_ru)
        return {"translated_text": (caption_ru + "\n" + preview).strip(), "meta": meta}
    translated = await translator.translate(seg.source_text, context)
    return {"translated_text": translated}


async def translate_document(ctx: dict, doc_id_str: str) -> str:
    doc_id = uuid.UUID(doc_id_str)
    translator: Translator = ctx["translator"]
    await _set_status(ctx, doc_id, DocumentStatus.translating)

    async with ctx["sessionmaker"]() as session:
        segments = list(
            (
                await session.execute(
                    select(Segment).where(Segment.document_id == doc_id).order_by(Segment.idx)
                )
            )
            .scalars()
            .all()
        )

    # Контекст (roadmap § 3.4): заголовок раздела + предыдущий абзац.
    contexts: dict[uuid.UUID, SegmentContext] = {}
    cur_heading: str | None = None
    prev_text: str | None = None
    for seg in segments:
        contexts[seg.id] = SegmentContext(heading=cur_heading, prev_text=prev_text)
        if seg.kind == SegmentKind.heading:
            cur_heading = seg.source_text
            prev_text = None
        elif seg.kind == SegmentKind.paragraph:
            prev_text = seg.source_text

    todo = [s for s in segments if s.kind in TRANSLATABLE_KINDS and s.translated_text is None]
    done_count = len([s for s in segments if s.kind in TRANSLATABLE_KINDS]) - len(todo)
    logger.info("translate %s: %d сегментов (готово ранее: %d)", doc_id, len(todo), done_count)

    sem = asyncio.Semaphore(settings.translate_concurrency)
    failures: list[str] = []

    async def work(seg: Segment) -> tuple[uuid.UUID, dict[str, Any]] | None:
        async with sem:
            try:
                return seg.id, await _translate_segment(translator, seg, contexts[seg.id])
            except Exception as exc:
                failures.append(f"сегмент {seg.idx}: {exc}")
                logger.error("translate %s seg %d: %s", doc_id, seg.idx, exc)
                return None

    try:
        pending = [asyncio.ensure_future(work(s)) for s in todo]
        buffer: list[tuple[uuid.UUID, dict[str, Any]]] = []

        async def flush() -> None:
            nonlocal done_count, buffer
            if not buffer:
                return
            async with ctx["sessionmaker"]() as session:
                for seg_id, values in buffer:
                    await session.execute(update(Segment).where(Segment.id == seg_id).values(**values))
                done_count += len(buffer)
                await session.execute(
                    update(Document).where(Document.id == doc_id).values(translated_count=done_count)
                )
                await session.commit()
            buffer = []

        for fut in asyncio.as_completed(pending):
            result = await fut
            if result is not None:
                buffer.append(result)
            if len(buffer) >= 20:
                await flush()
        await flush()

        if failures:
            raise RuntimeError(
                f"не переведено сегментов: {len(failures)}; первые ошибки: " + "; ".join(failures[:3])
            )

        await _set_status(ctx, doc_id, DocumentStatus.translated)
        await ctx["redis"].enqueue_job(
            "export_document", doc_id_str, _job_id=f"export:{doc_id}:{uuid.uuid4().hex[:8]}"
        )
        return f"translated: {len(todo)} segments"

    except Exception as exc:
        logger.exception("translate %s failed", doc_id)
        await _set_status(ctx, doc_id, DocumentStatus.error, f"перевод: {exc}")
        raise


# ---------------------------------------------------------------- export

async def export_document(ctx: dict, doc_id_str: str) -> str:
    doc_id = uuid.UUID(doc_id_str)
    storage: Storage = ctx["storage"]
    doc = await _get_doc(ctx, doc_id)
    await _set_status(ctx, doc_id, DocumentStatus.exporting)

    try:
        async with ctx["sessionmaker"]() as session:
            segments = list(
                (
                    await session.execute(
                        select(Segment).where(Segment.document_id == doc_id).order_by(Segment.idx)
                    )
                )
                .scalars()
                .all()
            )

        data = await asyncio.to_thread(build_docx, doc.filename, segments)
        export_key = f"{doc_id}/{Path(doc.filename).stem}.ru.docx"
        await storage.put_bytes(
            settings.bucket_exports,
            export_key,
            data,
            content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

        async with ctx["sessionmaker"]() as session:
            await session.execute(
                update(Document)
                .where(Document.id == doc_id)
                .values(status=DocumentStatus.done, error=None, s3_key_export_docx=export_key)
            )
            await session.commit()
        return f"exported: {export_key} ({len(data)} bytes)"

    except Exception as exc:
        logger.exception("export %s failed", doc_id)
        await _set_status(ctx, doc_id, DocumentStatus.error, f"экспорт: {exc}")
        raise
