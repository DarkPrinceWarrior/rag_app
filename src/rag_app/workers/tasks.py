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
    DocumentKind,
    DocumentStatus,
    GlossaryTerm,
    Segment,
    SegmentKind,
)
from rag_app.llm.client import SegmentContext, Translator, needs_translation, pick_glossary_terms
from rag_app.pipeline import ooxml
from rag_app.pipeline.babeldoc import BabelDocUnavailableError, run_babeldoc
from rag_app.pipeline.export_docx import build_docx
from rag_app.pipeline.parse import load_block_geometry, load_content_list, pdf_info, run_mineru
from rag_app.pipeline.scan_pdf import build_scan_overlay
from rag_app.pipeline.segments import content_list_to_segments
from rag_app.pipeline.validate import ValidationResult, validate_numbers
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
        ext = Path(doc.filename).suffix.lower().lstrip(".")
        artifact_key: str | None = None
        with tempfile.TemporaryDirectory(prefix="rag_parse_") as tmp:
            tmp_path = Path(tmp)
            local_file = tmp_path / Path(doc.filename).name
            await storage.download_to(settings.bucket_originals, doc.s3_key_original, local_file)

            if ext == "pdf":
                # roadmap § 3.1: детект текстового слоя за миллисекунды;
                # сканы идут той же командой — mineru -m auto решает постранично
                n_pages, has_text = await asyncio.to_thread(pdf_info, local_file)
                kind = DocumentKind.pdf_text if has_text else DocumentKind.pdf_scan
                out_dir = tmp_path / "mineru_out"
                content_list_path = await run_mineru(local_file, out_dir)
                items = load_content_list(content_list_path)
                drafts = content_list_to_segments(items)
                # геометрия в пунктах из middle.json — для оверлея сканов и
                # подсветки цитат (этап 3); content_list-bbox в другом масштабе
                geo = load_block_geometry(content_list_path)
                for d in drafts:
                    if d.kind in (SegmentKind.table, SegmentKind.image, SegmentKind.equation):
                        bbox_pt = geo.pop_typed(d.page_idx, d.kind.value)
                    else:
                        bbox_pt = geo.match_text(d.page_idx, d.source_text)
                    size = geo.page_sizes.get(d.page_idx) if d.page_idx is not None else None
                    if bbox_pt and size:
                        d.meta["bbox_pt"] = bbox_pt
                        d.meta["page_size_pt"] = list(size)
                artifact_key = f"{doc_id}/content_list.json"
                await storage.put_bytes(
                    settings.bucket_artifacts,
                    artifact_key,
                    content_list_path.read_bytes(),
                    content_type="application/json",
                )
            elif ext in ("docx", "xlsx", "pptx"):
                kind = DocumentKind(ext)
                drafts = await asyncio.to_thread(ooxml.extract, ext, local_file)
                n_pages = (max(d.page_idx for d in drafts) + 1) if ext == "pptx" and drafts else None
            else:
                raise RuntimeError(f"неподдерживаемый формат: .{ext}")

        if not drafts:
            raise RuntimeError("парсер не извлёк ни одного блока")

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
                    kind=kind.value,
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
        return f"parsed [{kind.value}]: {len(drafts)} segments, {n_pages} pages"

    except Exception as exc:
        logger.exception("parse %s failed", doc_id)
        await _set_status(ctx, doc_id, DocumentStatus.error, f"парсинг: {exc}")
        raise


# ---------------------------------------------------------------- translate

async def _translate_validated(
    translator: Translator, text: str, context: SegmentContext
) -> tuple[str, ValidationResult]:
    """Перевод + числовая валидация; один ре-перевод с фидбеком (roadmap § 3.4 п.3)."""
    if not needs_translation(text):
        return text, ValidationResult(ok=True)
    translated = await translator.translate(text, context)
    result = validate_numbers(text, translated)
    if result.ok:
        return translated, result
    feedback = (
        f"числа из оригинала искажены или потеряны: {', '.join(result.missing)}. "
        "Перенеси ВСЕ числа, единицы и обозначения без изменений."
    )
    translated2 = await translator.translate(text, context, feedback=feedback)
    result2 = validate_numbers(text, translated2)
    return (translated2, result2) if result2.ok else (translated2, result2)


async def _translate_segment(translator: Translator, seg: Segment, context: SegmentContext) -> dict[str, Any]:
    """Возвращает values для UPDATE сегмента."""
    if seg.kind == SegmentKind.table:
        rows: list[list[str]] = seg.meta.get("table_rows") or []
        caption: str = seg.meta.get("caption") or ""
        rows_ru: list[list[str]] = []
        failures: list[dict[str, Any]] = []
        for r_i, row in enumerate(rows):
            row_ru: list[str] = []
            for c_i, cell in enumerate(row):
                cell_ru, vr = await _translate_validated(translator, cell, context)
                row_ru.append(cell_ru)
                if not vr.ok:
                    failures.append({"row": r_i, "col": c_i, **vr.as_dict()})
            rows_ru.append(row_ru)
        caption_ru, cap_vr = await _translate_validated(translator, caption, context)
        if not cap_vr.ok:
            failures.append({"caption": True, **cap_vr.as_dict()})
        meta = dict(seg.meta)
        meta["table_rows_ru"] = rows_ru
        meta["caption_ru"] = caption_ru
        preview = "\n".join(" | ".join(r) for r in rows_ru)
        return {
            "translated_text": (caption_ru + "\n" + preview).strip(),
            "meta": meta,
            "needs_review": bool(failures),
            "validation": {"cells": failures} if failures else None,
        }

    translated, vr = await _translate_validated(translator, seg.source_text, context)
    return {
        "translated_text": translated,
        "needs_review": not vr.ok,
        "validation": None if vr.ok else vr.as_dict(),
    }


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

    # Глоссарий (roadmap § 3.4 п.1): длинные термины первыми — приоритет точных фраз.
    async with ctx["sessionmaker"]() as session:
        rows = (await session.execute(select(GlossaryTerm.en_term, GlossaryTerm.ru_term))).all()
    all_terms = sorted(((en, ru) for en, ru in rows), key=lambda t: -len(t[0]))

    # Контекст (roadmap § 3.4): заголовок раздела + предыдущий абзац + термины.
    contexts: dict[uuid.UUID, SegmentContext] = {}
    cur_heading: str | None = None
    prev_text: str | None = None
    for seg in segments:
        terms = pick_glossary_terms(seg.source_text, all_terms)
        if seg.kind == SegmentKind.heading:
            # заголовки — без текстового контекста: модель может «утащить» его в ответ
            contexts[seg.id] = SegmentContext(glossary=terms)
            cur_heading = seg.source_text
            prev_text = None
            continue
        contexts[seg.id] = SegmentContext(heading=cur_heading, prev_text=prev_text, glossary=terms)
        if seg.kind == SegmentKind.paragraph:
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

_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_OOXML_MIME = {
    "docx": _DOCX_MIME,
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


async def _export_pdf_layout(ctx: dict, doc: Document, local_pdf: Path, tmp: Path) -> dict[str, Any]:
    """BabelDOC: PDF с сохранённой вёрсткой (mono + dual). Недоступен — не фейл."""
    try:
        mono, dual = await run_babeldoc(
            local_pdf, tmp / "babeldoc_out", ocr_workaround=settings.babeldoc_auto_ocr_workaround
        )
    except BabelDocUnavailableError as exc:
        logger.warning("babeldoc недоступен: %s", exc)
        return {}
    storage: Storage = ctx["storage"]
    values: dict[str, Any] = {}
    stem = Path(doc.filename).stem
    if mono is not None:
        key = f"{doc.id}/{stem}.ru.pdf"
        await storage.put_bytes(settings.bucket_exports, key, mono.read_bytes(), "application/pdf")
        values["s3_key_export_pdf"] = key
    if dual is not None:
        key = f"{doc.id}/{stem}.en-ru.pdf"
        await storage.put_bytes(settings.bucket_exports, key, dual.read_bytes(), "application/pdf")
        values["s3_key_export_pdf_dual"] = key
    return values


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

        values: dict[str, Any] = {"status": DocumentStatus.done, "error": None}
        stem = Path(doc.filename).stem

        if doc.kind in (DocumentKind.pdf_text, DocumentKind.pdf_scan):
            # 1) редактируемый DOCX из сегментов
            data = await asyncio.to_thread(build_docx, doc.filename, segments)
            docx_key = f"{doc_id}/{stem}.ru.docx"
            await storage.put_bytes(settings.bucket_exports, docx_key, data, _DOCX_MIME)
            values["s3_key_export_docx"] = docx_key
            # 2) PDF с исходной вёрсткой
            with tempfile.TemporaryDirectory(prefix="rag_export_") as tmp:
                tmp_path = Path(tmp)
                local_pdf = tmp_path / Path(doc.filename).name
                await storage.download_to(settings.bucket_originals, doc.s3_key_original, local_pdf)
                if doc.kind == DocumentKind.pdf_scan:
                    # BabelDOC сканы не переводит (нет текстового слоя) —
                    # собственный оверлей по bbox (roadmap § 9, запасной путь)
                    mono_data, dual_data = await asyncio.to_thread(
                        build_scan_overlay, local_pdf, segments
                    )
                    mono_key = f"{doc_id}/{stem}.ru.pdf"
                    dual_key = f"{doc_id}/{stem}.en-ru.pdf"
                    await storage.put_bytes(
                        settings.bucket_exports, mono_key, mono_data, "application/pdf"
                    )
                    await storage.put_bytes(
                        settings.bucket_exports, dual_key, dual_data, "application/pdf"
                    )
                    values["s3_key_export_pdf"] = mono_key
                    values["s3_key_export_pdf_dual"] = dual_key
                else:
                    values.update(await _export_pdf_layout(ctx, doc, local_pdf, tmp_path))
        else:
            # OOXML: переводы обратно в копию оригинала, формат и вёрстка не меняются
            ext = doc.kind if isinstance(doc.kind, str) else doc.kind.value
            translations = {
                ooxml.location_key(s.meta["location"]): s.translated_text
                for s in segments
                if s.translated_text is not None and s.meta.get("location")
            }
            with tempfile.TemporaryDirectory(prefix="rag_export_") as tmp:
                tmp_path = Path(tmp)
                src = tmp_path / Path(doc.filename).name
                dst = tmp_path / f"{stem}.ru.{ext}"
                await storage.download_to(settings.bucket_originals, doc.s3_key_original, src)
                applied = await asyncio.to_thread(ooxml.inject, ext, src, dst, translations)
                logger.info("export %s: %d сегментов записано в %s", doc_id, applied, dst.name)
                source_key = f"{doc_id}/{dst.name}"
                await storage.put_bytes(
                    settings.bucket_exports, source_key, dst.read_bytes(), _OOXML_MIME[ext]
                )
                values["s3_key_export_source"] = source_key

        async with ctx["sessionmaker"]() as session:
            await session.execute(update(Document).where(Document.id == doc_id).values(**values))
            await session.commit()
        return f"exported: {', '.join(k for k in values if k.startswith('s3_'))}"

    except Exception as exc:
        logger.exception("export %s failed", doc_id)
        await _set_status(ctx, doc_id, DocumentStatus.error, f"экспорт: {exc}")
        raise
