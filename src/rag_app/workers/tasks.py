"""ARQ-задачи: parse_document → translate_document → export_document.

Цепочка статусов: uploaded → parsing → parsed → translating → translated
→ exporting → done; любая ошибка → status=error + текст в documents.error.
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import delete, func, select, update

from rag_app.config import settings
from rag_app.db.models import (
    TRANSLATABLE_KINDS,
    Chunk,
    Document,
    DocumentKind,
    DocumentStatus,
    GlossaryTerm,
    PageEmbedding,
    Segment,
    SegmentKind,
)
from rag_app.llm.client import SegmentContext, Translator, needs_translation, pick_glossary_terms
from rag_app.llm.embeddings import Embedder
from rag_app.llm.vision import VisionClient
from rag_app.llm.visual import VisualEmbedder
from rag_app.observability import log_translate_trace
from rag_app.pipeline import ooxml
from rag_app.pipeline.babeldoc import BabelDocUnavailableError, run_babeldoc, write_glossary_csv
from rag_app.pipeline.dots import dots_to_segments, run_dots
from rag_app.pipeline.export_docx import build_docx
from rag_app.pipeline.office_render import render_to_pdf
from rag_app.pipeline.paddle_vl import paddle_to_segments, run_paddle
from rag_app.pipeline.parse import (
    PDFIUM_LOCK,
    backfill_text_layer,
    load_block_geometry,
    load_content_list,
    pdf_info,
    run_mineru,
)
from rag_app.pipeline.scan_pdf import build_scan_overlay
from rag_app.pipeline.segments import SegmentDraft, content_list_to_segments
from rag_app.pipeline.validate import ValidationResult, validate_numbers
from rag_app.rag.chunking import segments_to_chunks
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


async def _upload_segment_images(
    storage: Storage, doc_id: uuid.UUID, base_dir: Path, drafts: list[SegmentDraft]
) -> None:
    """Файлы картинок (meta.img_path относительно base_dir) → MinIO для вставки
    в MD-просмотр. Ключ детерминированный: {doc_id}/img/{имя}; в meta кладём
    img_s3. Общий для PDF (MinerU) и DOCX (OOXML)."""
    for d in drafts:
        rel = d.meta.get("img_path")
        if not rel:
            continue
        img_file = base_dir / rel
        if not img_file.is_file():
            continue
        img_key = f"{doc_id}/img/{img_file.name}"
        await storage.put_bytes(
            settings.bucket_artifacts,
            img_key,
            img_file.read_bytes(),
            content_type=mimetypes.guess_type(img_file.name)[0] or "image/jpeg",
        )
        d.meta["img_s3"] = img_key


# ---------------------------------------------------------------- parse

def _parser_backend(doc: Document) -> str:
    """Выбранный движок парсинга pdf_text: на документе или дефолт из settings."""
    return doc.parser_backend or settings.pdf_parser_backend


async def _vlm_segments(backend: str, pdf_path: Path, out_dir: Path) -> list[SegmentDraft]:
    """Парс PDF альтернативным VLM-движком (dots.mocr / PaddleOCR-VL) → SegmentDraft."""
    if backend == "dots_mocr":
        page_dir = await run_dots(pdf_path, out_dir)
        return await asyncio.to_thread(dots_to_segments, page_dir, pdf_path)
    if backend == "paddle_vl":
        await run_paddle(pdf_path, out_dir)
        return await asyncio.to_thread(paddle_to_segments, out_dir)
    raise RuntimeError(f"неизвестный backend парсера: {backend}")


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
                backend = _parser_backend(doc)
                out_dir = tmp_path / "parser_out"
                if not doc.parse_force_ocr and has_text and backend in ("dots_mocr", "paddle_vl"):
                    # альтернативный VLM-движок (сравнение парсеров): свой формат
                    # вывода → SegmentDraft напрямую, без mineru content_list/geo
                    kind = DocumentKind.pdf_text
                    drafts = await _vlm_segments(backend, local_file, out_dir)
                    # PaddleOCR-VL вырезает рисунки в файлы (dots — нет) → грузим в
                    # img_s3, чтобы они появились в текст-просмотре (как у MinerU)
                    if backend == "paddle_vl":
                        await _upload_segment_images(storage, doc_id, out_dir, drafts)
                else:
                    if doc.parse_force_ocr:
                        # битый ToUnicode-cmap текстового слоя → OCR с картинки
                        # VLM-бэкендом (MinerU 3.3, multilingual — кириллица/таблицы/
                        # надстрочные); экспорт через оверлей (как скан), а не babeldoc
                        kind = DocumentKind.pdf_scan
                        content_list_path = await run_mineru(
                            local_file,
                            out_dir,
                            backend=settings.mineru_force_ocr_backend,
                            method="ocr",
                            lang=doc.ocr_lang,
                        )
                    else:
                        kind = DocumentKind.pdf_text if has_text else DocumentKind.pdf_scan
                        content_list_path = await run_mineru(local_file, out_dir)
                    items = load_content_list(content_list_path)
                    drafts = content_list_to_segments(items)
                    # pdf_text: VLM местами роняет/прореживает целые страницы —
                    # достраиваем их абзацами из текстового слоя (истина для PDF
                    # с текстом), дедуп против VLM. Сканам слой не поможет (no-op).
                    if kind == DocumentKind.pdf_text:
                        drafts, filled = await asyncio.to_thread(
                            backfill_text_layer, local_file, drafts
                        )
                        if filled:
                            logger.info("parse %s: достроены страницы из слоя: %s", doc_id, filled)
                    # геометрия в пунктах из middle.json — для оверлея сканов и
                    # подсветки цитат (этап 3); content_list-bbox в другом масштабе
                    geo = load_block_geometry(content_list_path)
                    for d in drafts:
                        if d.kind in (SegmentKind.table, SegmentKind.image, SegmentKind.equation):
                            bbox_pt = geo.pop_typed(d.page_idx, d.kind.value)
                        else:
                            bbox_pt = geo.match_text(d.page_idx, d.source_text)
                            # списки/оглавления content_list схлопывает в один абзац —
                            # восстанавливаем переносы и отступы из строк middle.json
                            reflowed = geo.reflow(d.page_idx, d.source_text)
                            if reflowed:
                                d.source_text = reflowed
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
                    # картинки/рисунки/графики из оригинала → MinIO (для вставки
                    # в MD-просмотр). MinerU извлекает их в out_dir рядом с
                    # content_list; раньше они выбрасывались вместе с tmp.
                    await _upload_segment_images(storage, doc_id, content_list_path.parent, drafts)
            elif ext in ("docx", "xlsx", "pptx"):
                kind = DocumentKind(ext)
                backend = _parser_backend(doc)
                if (
                    ext == "docx"
                    and not doc.parse_force_ocr
                    and backend in ("dots_mocr", "paddle_vl")
                    and settings.office_render_enabled
                ):
                    # DOCX через VLM-движок (сравнение парсеров): рендерим в PDF
                    # (LibreOffice) и парсим его — «текст» покажет VLM-сегменты,
                    # «как в Microsoft» рендерится отдельным job'ом ниже. Round-trip
                    # перевода обратно в .docx для этого пути недоступен (нет
                    # location-ключей) — это режим сравнения, не продакшен-экспорт.
                    rendered = await render_to_pdf(
                        local_file, tmp_path, settings.office_render_timeout_s
                    )
                    rpdf = tmp_path / "rendered.pdf"
                    rpdf.write_bytes(rendered)
                    out_dir = tmp_path / "parser_out"
                    drafts = await _vlm_segments(backend, rpdf, out_dir)
                    if backend == "paddle_vl":
                        await _upload_segment_images(storage, doc_id, out_dir, drafts)
                    n_pages, _ = await asyncio.to_thread(pdf_info, rpdf)
                else:
                    # DOCX: встроенные картинки извлекаем в tmp и грузим в MinIO
                    # (для MD-просмотра); xlsx/pptx картинок-сегментов не дают
                    img_dir = tmp_path / "ooxml_img"
                    img_dir.mkdir(exist_ok=True)
                    drafts = await asyncio.to_thread(ooxml.extract, ext, local_file, img_dir)
                    if ext == "docx":
                        await _upload_segment_images(storage, doc_id, img_dir, drafts)
                    n_pages = (max(d.page_idx for d in drafts) + 1) if ext == "pptx" and drafts else None
            elif ext == "txt":
                # plain-текст (ТЗ §4.2): абзацы (разделённые пустой строкой) → сегменты
                kind = DocumentKind.text
                text = local_file.read_text(encoding="utf-8", errors="replace")
                paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
                drafts = [
                    SegmentDraft(idx=i, kind=SegmentKind.paragraph, source_text=p)
                    for i, p in enumerate(paras)
                ]
                n_pages = None
            else:
                raise RuntimeError(f"неподдерживаемый формат: .{ext}")

        if not drafts:
            # скан без распознаваемого текста (OCR дал мусор и его отфильтровали,
            # либо страница — чистый рисунок): не валим документ — кладём
            # плейсхолдер-картинку, describe_images (VL) обогатит его описанием.
            if kind == DocumentKind.pdf_scan:
                drafts = [SegmentDraft(0, SegmentKind.image, "", 0)]
                logger.info("parse %s: скан без текста — плейсхолдер-картинка (ждём VL)", doc_id)
            else:
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
        # OOXML: ранний рендер оригинала в PDF параллельно переводу — чтобы
        # «как в Microsoft» (дефолт DOCX) открывался сразу, не ждя экспорта.
        if kind in (DocumentKind.docx, DocumentKind.xlsx, DocumentKind.pptx):
            await ctx["redis"].enqueue_job(
                "render_original_view", doc_id_str, _job_id=f"vieworig:{doc_id}:{uuid.uuid4().hex[:8]}"
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
        grid: list[list[str]] = seg.meta.get("table_rows") or []
        cells: list[list[dict[str, Any]]] | None = seg.meta.get("table_cells")
        caption: str = seg.meta.get("caption") or ""
        failures: list[dict[str, Any]] = []
        cache: dict[str, str] = {}  # перевод каждой уникальной ячейки один раз

        async def tr(text: str, loc: dict[str, Any] | None = None) -> str:
            if text not in cache:
                ru, vr = await _translate_validated(translator, text, context)
                cache[text] = ru
                if not vr.ok and loc is not None:
                    failures.append({**loc, **vr.as_dict()})
            return cache[text]

        meta = dict(seg.meta)
        meta["caption_ru"] = await tr(caption, {"caption": True}) if caption else ""

        # ровная сетка → table_rows_ru (нужна DOCX-экспорту, export_docx.py)
        rows_ru: list[list[str]] = []
        for r_i, row in enumerate(grid):
            row_ru = [await tr(cell, {"row": r_i, "col": c_i}) for c_i, cell in enumerate(row)]
            rows_ru.append(row_ru)
        meta["table_rows_ru"] = rows_ru

        # сырые ячейки со спанами → table_cells_ru (для merged-рендера во вьювере,
        # перевод по позиции ячейки — подзаголовки не «уезжают»). Кэш переиспользует
        # уже переведённые тексты из сетки выше.
        if cells:
            cells_ru: list[list[dict[str, Any]]] = []
            for row in cells:
                row_ru_cells: list[dict[str, Any]] = []
                for c in row:
                    row_ru_cells.append(
                        {"text": await tr(c["text"]), "colspan": c["colspan"], "rowspan": c["rowspan"]}
                    )
                cells_ru.append(row_ru_cells)
            meta["table_cells_ru"] = cells_ru
            preview = "\n".join(" | ".join(c["text"] for c in row) for row in cells_ru)
        else:
            preview = "\n".join(" | ".join(r) for r in rows_ru)

        return {
            "translated_text": (meta["caption_ru"] + "\n" + preview).strip(),
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
    t_task = time.monotonic()
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

        doc = await _get_doc(ctx, doc_id)
        log_translate_trace(
            doc_id_str, doc.filename, doc.kind, len(todo), time.monotonic() - t_task, settings.llm_model
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


# ---------------------------------------------------------------- index (RAG, этап 3)

async def index_document(ctx: dict, doc_id_str: str) -> str:
    """Чанкинг по структуре → эмбеддинги EN/RU → chunks (roadmap § 5).

    Не влияет на статус перевода: ошибки идут в documents.index_error.
    """
    doc_id = uuid.UUID(doc_id_str)
    embedder: Embedder = ctx["embedder"]
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
        drafts = segments_to_chunks(segments)
        if not drafts:
            raise RuntimeError("нет чанков (документ пуст?)")

        emb_en = await embedder.embed([d.text_en for d in drafts])
        emb_ru = await embedder.embed([d.text_ru for d in drafts])

        async with ctx["sessionmaker"]() as session:
            await session.execute(delete(Chunk).where(Chunk.document_id == doc_id))
            session.add_all(
                Chunk(
                    document_id=doc_id,
                    idx=d.idx,
                    kind=d.kind,
                    heading_path=d.heading_path,
                    page_start=d.page_start,
                    page_end=d.page_end,
                    text_en=d.text_en,
                    text_ru=d.text_ru,
                    emb_en=e_en,
                    emb_ru=e_ru,
                    meta=d.meta,
                )
                for d, e_en, e_ru in zip(drafts, emb_en, emb_ru, strict=True)
            )
            await session.execute(
                update(Document)
                .where(Document.id == doc_id)
                .values(chunk_count=len(drafts), indexed_at=func.now(), index_error=None)
            )
            await session.commit()
        logger.info("index %s: %d чанков", doc_id, len(drafts))
        # pdf_scan (скан/чертёж/P&ID): дообогащаем VL-описанием смысла изображения
        # и переиндексируем. Маркер meta.vl_describe на сегментах не даёт зациклиться.
        if settings.vl_enabled and not any((s.meta or {}).get("vl_describe") for s in segments):
            doc = await _get_doc(ctx, doc_id)
            if doc.kind == DocumentKind.pdf_scan.value:
                await ctx["redis"].enqueue_job(
                    "describe_images", doc_id_str, _job_id=f"vl:{doc_id}:{uuid.uuid4().hex[:8]}"
                )
        # визуальный индекс (Этап 2): эмбеддинги страниц-рисунков для визуального
        # retrieval — авто для pdf_scan (все страницы) и любых доков с вырезанными
        # рисунками (img_s3). index_pages_visual сам отфильтрует страницы.
        if settings.visual_enabled:
            vdoc = await _get_doc(ctx, doc_id)
            vkind = vdoc.kind if isinstance(vdoc.kind, str) else vdoc.kind.value
            if vkind == DocumentKind.pdf_scan.value or any(
                (s.meta or {}).get("img_s3") for s in segments
            ):
                await ctx["redis"].enqueue_job(
                    "index_pages_visual", doc_id_str, _job_id=f"vis:{doc_id}:{uuid.uuid4().hex[:8]}"
                )
        return f"indexed: {len(drafts)} chunks"
    except Exception as exc:
        logger.exception("index %s failed", doc_id)
        async with ctx["sessionmaker"]() as session:
            await session.execute(
                update(Document).where(Document.id == doc_id).values(index_error=str(exc)[:1000])
            )
            await session.commit()
        raise


# ---------------------------------------------------------------- VL: описание рисунков

def _render_pdf_pages(pdf_bytes: bytes, max_pages: int, scale: float) -> list[tuple[int, bytes]]:
    """Страницы PDF → PNG (pypdfium2). pdfium не потокобезопасен — под общим локом."""
    import io

    import pypdfium2 as pdfium

    out: list[tuple[int, bytes]] = []
    with PDFIUM_LOCK:
        pdf = pdfium.PdfDocument(pdf_bytes)
        try:
            for i in range(min(len(pdf), max_pages)):
                pil = pdf[i].render(scale=scale).to_pil().convert("RGB")
                buf = io.BytesIO()
                pil.save(buf, format="PNG")
                out.append((i, buf.getvalue()))
        finally:
            pdf.close()
    return out


_MATCH_NOISE = re.compile(r"[^0-9a-zа-яё]+", re.IGNORECASE)


def _norm_match(s: str) -> str:
    """Нормализация для сопоставления текста: только буквы/цифры, нижний регистр —
    устойчиво к пунктуации, разным тире (–—-), small-caps, лишним пробелам."""
    return " ".join(_MATCH_NOISE.sub(" ", s or "").lower().split())


def _assign_pdf_pages(pdf_bytes: bytes, segments: list[Segment]) -> tuple[dict[uuid.UUID, int], int]:
    """Сегмент → его ФИЗИЧЕСКАЯ страница в отрендеренном PDF (по тексту оригинала).

    Для DOCX (поток без страниц): сопоставляем каждый абзац странице LibreOffice-
    рендера — тогда правый «текст»-просмотр листается синхронно с оригиналом, как
    у PDF. Возвращает ({segment_id: page_idx}, число_страниц). Сегменты идут по
    порядку, страницы монотонны → ищем ТОЛЬКО от курсора вперёд; несовпавший
    сегмент остаётся на текущей странице (без прыжков назад в оглавление/колонтитул).
    """
    import pypdfium2 as pdfium

    with PDFIUM_LOCK:
        pdf = pdfium.PdfDocument(pdf_bytes)
        try:
            n = len(pdf)
            pages: list[str] = []
            for i in range(n):
                page = pdf[i]
                tp = page.get_textpage()
                pages.append(_norm_match(tp.get_text_range()))
                tp.close()
                page.close()
        finally:
            pdf.close()

    # ЯКОРЯ + ИНТЕРПОЛЯЦИЯ + СНАП. Жадный курсор сбивается на повторяющемся тексте
    # (колонтитулы, оглавление). (1) Хиты каждого сегмента — страницы, где есть его
    # ключ (считаем один раз). (2) Якоря — сегменты с УНИКАЛЬНЫМ длинным совпадением
    # (ровно одна страница), монотонно: надёжный скелет. (3) Для каждого сегмента —
    # линейная оценка страницы между якорями, затем СНАП к ближайшей к оценке
    # странице из его хитов (используем все ~99% совпадений → заполняем пробелы).
    seg_hits: list[list[int]] = []
    seg_long: list[bool] = []
    for s in segments:
        full = _norm_match(s.source_text)
        key = full[:40]
        seg_long.append(len(full) >= 15)
        seg_hits.append([p for p in range(n) if key in pages[p]] if key else [])

    anchors: list[tuple[int, int]] = []
    last_page = -1
    for idx, hits in enumerate(seg_hits):
        if seg_long[idx] and len(hits) == 1 and hits[0] >= last_page:
            anchors.append((idx, hits[0]))
            last_page = hits[0]

    out: dict[uuid.UUID, int] = {}
    if not anchors:
        return {s.id: 0 for s in segments}, n
    ai = 0
    for idx, s in enumerate(segments):
        while ai + 1 < len(anchors) and anchors[ai + 1][0] <= idx:
            ai += 1
        if idx <= anchors[0][0]:
            est = float(anchors[0][1])
        elif idx >= anchors[-1][0]:
            est = float(anchors[-1][1])
        else:
            i0, p0 = anchors[ai]
            i1, p1 = anchors[ai + 1]
            est = float(p0) if i1 == i0 else p0 + (idx - i0) * (p1 - p0) / (i1 - i0)
        hits = seg_hits[idx]
        page = min(hits, key=lambda p: abs(p - est)) if hits else round(est)
        out[s.id] = max(0, min(int(page), n - 1))
    return out, n


async def describe_images(ctx: dict, doc_id_str: str) -> str:
    """VL-обогащение СКАНОВ (pdf_scan): вся страница — рисунок (P&ID/чертёж/фото) →
    Qwen3.5-VL раскрывает смысл по-русски → сегменты-описания (kind=image) →
    переиндексация. Нужно для сканов без текстового слоя (искомость в чате).

    pdf_text/docx/pptx рисунки НЕ предописываются: их вырезанные кропы (img_s3)
    подаются в Qwen3.5-vision ON-DEMAND в чате (rag/chat.py), а в текст-просмотре —
    картинка + родная подпись."""
    if not settings.vl_enabled:
        return "vl disabled"
    doc_id = uuid.UUID(doc_id_str)
    storage: Storage = ctx["storage"]
    doc = await _get_doc(ctx, doc_id)
    kind = doc.kind if isinstance(doc.kind, str) else doc.kind.value
    if kind != DocumentKind.pdf_scan.value:
        return f"skip: kind={kind} (рисунки описываются on-demand в чате)"
    try:
        with tempfile.TemporaryDirectory(prefix="rag_vl_") as tmp:
            local = Path(tmp) / "src.pdf"
            await storage.download_to(settings.bucket_originals, doc.s3_key_original, local)
            pages = await asyncio.to_thread(
                _render_pdf_pages, local.read_bytes(), settings.vl_max_pages, settings.vl_render_scale
            )
        vision = VisionClient()
        described: list[tuple[int, str]] = []
        for pidx, png in pages:
            try:
                desc = await vision.describe(png)
            except Exception as exc:  # noqa: BLE001 — страница пропускается, не валим документ
                logger.warning("vl describe %s p%d: %s", doc_id, pidx, exc)
                continue
            if desc:
                described.append((pidx, desc))
        if not described:
            return "vl: нет описаний"

        async with ctx["sessionmaker"]() as session:
            # идемпотентность: убрать прежние VL-описания этого документа
            await session.execute(
                delete(Segment).where(
                    Segment.document_id == doc_id,
                    Segment.meta.op("->>")("vl_describe") == "true",
                )
            )
            base_idx = (
                await session.execute(
                    select(func.coalesce(func.max(Segment.idx), 0)).where(
                        Segment.document_id == doc_id
                    )
                )
            ).scalar_one()
            for i, (pidx, desc) in enumerate(described, 1):
                session.add(
                    Segment(
                        document_id=doc_id,
                        idx=base_idx + i,
                        page_idx=pidx,
                        kind=SegmentKind.image,
                        source_text=desc,
                        translated_text=desc,  # VL уже выдаёт русский — переводить нечего
                        meta={"vl_describe": True},
                    )
                )
            cnt = (
                await session.execute(
                    select(func.count()).select_from(Segment).where(Segment.document_id == doc_id)
                )
            ).scalar_one()
            trc = (
                await session.execute(
                    select(func.count())
                    .select_from(Segment)
                    .where(Segment.document_id == doc_id, Segment.translated_text.isnot(None))
                )
            ).scalar_one()
            await session.execute(
                update(Document)
                .where(Document.id == doc_id)
                .values(segment_count=cnt, translated_count=trc)
            )
            await session.commit()

        await ctx["redis"].enqueue_job(
            "index_document", doc_id_str, _job_id=f"index:{doc_id}:{uuid.uuid4().hex[:8]}"
        )
        return f"vl: {len(described)} описаний на {len(pages)} стр."
    except Exception as exc:  # noqa: BLE001 — VL необязателен, не валим документ
        logger.exception("describe_images %s failed", doc_id)
        return f"vl error: {exc}"


# ------------------------------------------------- визуальный индекс (§ 12.1 шаг 4)

async def _figure_pages(ctx: dict, doc_id: uuid.UUID) -> set[int]:
    """Страницы документа, на которых есть вырезанный рисунок (img_s3) — их и
    эмбеддим визуально для pdf_text/docx/pptx (чисто текстовые покрывает текст-поиск)."""
    async with ctx["sessionmaker"]() as session:
        rows = (
            await session.execute(
                select(Segment.page_idx)
                .where(
                    Segment.document_id == doc_id,
                    Segment.kind == SegmentKind.image,
                    Segment.meta.op("->>")("img_s3").isnot(None),
                    Segment.page_idx.isnot(None),
                )
                .distinct()
            )
        ).scalars().all()
    return {p for p in rows if p is not None}


async def index_pages_visual(ctx: dict, doc_id_str: str) -> str:
    """Эмбеддинги страниц-картинок (Qwen3-VL-Embedding-8B) для визуального retrieval:
    печати/штампы/чертежи/схемы, где текстовый контур слаб. pdf_scan — все страницы
    (вся страница = визуал); pdf_text/docx/pptx — только страницы с вырезанными
    рисунками (img_s3), остальное берёт текстовый поиск."""
    doc_id = uuid.UUID(doc_id_str)
    if not settings.visual_enabled:
        return "visual disabled"
    doc = await _get_doc(ctx, doc_id)
    kind = doc.kind if isinstance(doc.kind, str) else doc.kind.value
    storage: Storage = ctx["storage"]
    visual: VisualEmbedder = ctx["visual"]

    # источник рендера + набор страниц (None = все)
    if kind == DocumentKind.pdf_scan.value:
        src_bucket, src_key, page_filter = settings.bucket_originals, doc.s3_key_original, None
    elif kind == DocumentKind.pdf_text.value:
        src_bucket, src_key = settings.bucket_originals, doc.s3_key_original
        page_filter = await _figure_pages(ctx, doc_id)
    elif kind in ("docx", "pptx") and doc.s3_key_view_orig:
        src_bucket, src_key = settings.bucket_exports, doc.s3_key_view_orig
        page_filter = await _figure_pages(ctx, doc_id)
    else:
        return f"skip: kind={kind}"
    if page_filter is not None and not page_filter:
        return "visual: нет страниц с рисунками"

    def render_pages(pdf_path: Path) -> list[tuple[int, bytes]]:
        import io as _io

        import pypdfium2 as pdfium

        with PDFIUM_LOCK:
            pdf = pdfium.PdfDocument(str(pdf_path))
            try:
                out: list[tuple[int, bytes]] = []
                for i in range(len(pdf)):
                    if page_filter is not None and i not in page_filter:
                        continue
                    img = pdf[i].render(scale=settings.visual_render_scale).to_pil().convert("RGB")
                    buf = _io.BytesIO()
                    img.save(buf, format="JPEG", quality=85)
                    out.append((i, buf.getvalue()))
                return out
            finally:
                pdf.close()

    try:
        with tempfile.TemporaryDirectory(prefix="rag_visual_") as tmp:
            local_pdf = Path(tmp) / "doc.pdf"
            await storage.download_to(src_bucket, src_key, local_pdf)
            pages = await asyncio.to_thread(render_pages, local_pdf)
        if not pages:
            return "visual: нет страниц"

        embs: list[tuple[int, list[float]]] = []
        for pidx, jpeg in pages:  # последовательно: vision-башня прожорлива
            embs.append((pidx, await visual.embed_page(jpeg)))

        async with ctx["sessionmaker"]() as session:
            await session.execute(delete(PageEmbedding).where(PageEmbedding.document_id == doc_id))
            session.add_all(
                PageEmbedding(document_id=doc_id, page_idx=p, emb=e) for p, e in embs
            )
            await session.execute(
                update(Document)
                .where(Document.id == doc_id, Document.index_error.like("visual:%"))
                .values(index_error=None)
            )
            await session.commit()
        logger.info("visual index %s: %d страниц (%s)", doc_id, len(embs), kind)
        return f"visual indexed: {len(embs)} pages"
    except Exception as exc:
        logger.exception("visual index %s failed", doc_id)
        async with ctx["sessionmaker"]() as session:
            await session.execute(
                update(Document)
                .where(Document.id == doc_id)
                .values(index_error=f"visual: {str(exc)[:500]}")
            )
            await session.commit()
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
    # утверждённую терминологию отдаём и в PDF-контур (раньше только в DOCX)
    async with ctx["sessionmaker"]() as session:
        terms = (
            await session.execute(select(GlossaryTerm.en_term, GlossaryTerm.ru_term))
        ).all()
    glossary_file = write_glossary_csv([(t.en_term, t.ru_term) for t in terms], tmp / "glossary.csv")
    try:
        mono, dual = await run_babeldoc(
            local_pdf,
            tmp / "babeldoc_out",
            ocr_workaround=settings.babeldoc_auto_ocr_workaround,
            glossary_file=glossary_file,
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


async def render_original_view(ctx: dict, doc_id_str: str) -> str:
    """Ранний рендер ОРИГИНАЛА OOXML в PDF (LibreOffice) — чтобы «как в Microsoft»
    открывался сразу после парсинга, не дожидаясь перевода/экспорта. Ключ
    детерминированный (`{doc_id}/view_orig.pdf`); export потом перезапишет тем же.
    Необязателен (рендер может упасть) — документ не блокируется."""
    if not settings.office_render_enabled:
        return "office render disabled"
    doc_id = uuid.UUID(doc_id_str)
    storage: Storage = ctx["storage"]
    doc = await _get_doc(ctx, doc_id)
    kind = doc.kind if isinstance(doc.kind, str) else doc.kind.value
    if kind not in ("docx", "xlsx", "pptx"):
        return f"skip ({kind})"
    try:
        with tempfile.TemporaryDirectory(prefix="rag_vieworig_") as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / Path(doc.filename).name
            await storage.download_to(settings.bucket_originals, doc.s3_key_original, src)
            pdf = await render_to_pdf(src, tmp_path, settings.office_render_timeout_s)
            key = f"{doc_id}/view_orig.pdf"
            await storage.put_bytes(settings.bucket_exports, key, pdf, "application/pdf")
        async with ctx["sessionmaker"]() as session:
            await session.execute(
                update(Document).where(Document.id == doc_id).values(s3_key_view_orig=key)
            )
            await session.commit()
        return f"view_orig ready: {key}"
    except Exception as exc:  # noqa: BLE001 — рендер необязателен
        logger.warning("render_original_view %s: %s", doc_id, exc)
        return f"failed: {exc}"


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
                elif settings.translated_pdf_from_docx:
                    # «вёрстка» перевода = чистый reflow-PDF из НАШЕГО DOCX
                    # (build_docx → LibreOffice). Таблицы/абзацы переносятся, поэтому
                    # overflow / «скачущий шрифт» / утечки тегов и английских связок
                    # невозможны by construction (в отличие от пиксель-подгонки
                    # BabelDOC). Плюс быстрее (LibreOffice ~секунды против минут
                    # LLM-вызовов BabelDOC) и без нагрузки на GPU.
                    try:
                        docx_path = tmp_path / f"{stem}.ru.docx"
                        docx_path.write_bytes(data)
                        pdf_bytes = await render_to_pdf(
                            docx_path, tmp_path, settings.office_render_timeout_s
                        )
                        pdf_key = f"{doc_id}/{stem}.ru.pdf"
                        await storage.put_bytes(
                            settings.bucket_exports, pdf_key, pdf_bytes, "application/pdf"
                        )
                        values["s3_key_export_pdf"] = pdf_key
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "export %s: reflow-PDF из DOCX не собрался (%s) — оставляем DOCX",
                            doc_id,
                            exc,
                        )
                else:
                    # BabelDOC (пиксель-вёрстка) — за флагом (translated_pdf_from_docx=False).
                    # Сам убивает подпроцесс по таймауту (babeldoc_timeout_s) — на тяжёлых
                    # PDF очень медленный. При сбое оставляем DOCX, документ не блокируется.
                    try:
                        values.update(await _export_pdf_layout(ctx, doc, local_pdf, tmp_path))
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "export %s: BabelDOC PDF не собрался (%s) — оставляем DOCX", doc_id, exc
                        )
        elif doc.kind == DocumentKind.text:
            # plain TXT (ТЗ §4.2): только редактируемый DOCX из сегментов
            data = await asyncio.to_thread(build_docx, doc.filename, segments)
            docx_key = f"{doc_id}/{stem}.ru.docx"
            await storage.put_bytes(settings.bucket_exports, docx_key, data, _DOCX_MIME)
            values["s3_key_export_docx"] = docx_key
        else:
            # OOXML: переводы обратно в копию оригинала, формат и вёрстка не меняются
            ext = doc.kind if isinstance(doc.kind, str) else doc.kind.value
            if ext == "xlsx":
                # XLSX дедуплицирован на extract (1 сегмент = 1 уникальный текст);
                # inject_xlsx применяет перевод ПО ИСХОДНОМУ ТЕКСТУ ячейки → ключ
                # словаря = source_text, перевод раскладывается на все дубликаты.
                translations = {
                    s.source_text: s.translated_text
                    for s in segments
                    if s.translated_text is not None
                }
            else:
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
                # просмотр «как в Microsoft»: оригинал и перевод → PDF (LibreOffice),
                # показываются в pdf.js-вьювере вместо плоского текста
                if settings.office_render_enabled:
                    try:
                        orig_pdf = await render_to_pdf(src, tmp_path, settings.office_render_timeout_s)
                        ru_pdf = await render_to_pdf(dst, tmp_path, settings.office_render_timeout_s)
                        ok_key, rk_key = f"{doc_id}/view_orig.pdf", f"{doc_id}/view_ru.pdf"
                        await storage.put_bytes(settings.bucket_exports, ok_key, orig_pdf, "application/pdf")
                        await storage.put_bytes(settings.bucket_exports, rk_key, ru_pdf, "application/pdf")
                        values["s3_key_view_orig"] = ok_key
                        values["s3_key_view_ru"] = rk_key
                        # DOCX: привязываем сегменты к физическим страницам оригинала
                        # (LibreOffice-PDF) → правый «текст»-просмотр листается
                        # синхронно с оригиналом, как у PDF (page_idx был null).
                        if ext == "docx":
                            page_map, n_pages = await asyncio.to_thread(
                                _assign_pdf_pages, orig_pdf, segments
                            )
                            if page_map:
                                async with ctx["sessionmaker"]() as session:
                                    await session.execute(
                                        update(Segment),
                                        [{"id": sid, "page_idx": p} for sid, p in page_map.items()],
                                    )
                                    await session.commit()
                                values["page_count"] = n_pages
                    except Exception as exc:  # noqa: BLE001 — рендер необязателен
                        logger.warning("export %s: LibreOffice-рендер не удался (%s)", doc_id, exc)

        async with ctx["sessionmaker"]() as session:
            await session.execute(update(Document).where(Document.id == doc_id).values(**values))
            await session.commit()

        await ctx["redis"].enqueue_job(
            "index_document", doc_id_str, _job_id=f"index:{doc_id}:{uuid.uuid4().hex[:8]}"
        )
        await ctx["redis"].enqueue_job(
            "index_pages_visual", doc_id_str, _job_id=f"vindex:{doc_id}:{uuid.uuid4().hex[:8]}"
        )
        return f"exported: {', '.join(k for k in values if k.startswith('s3_'))}"

    except Exception as exc:
        logger.exception("export %s failed", doc_id)
        await _set_status(ctx, doc_id, DocumentStatus.error, f"экспорт: {exc}")
        raise
