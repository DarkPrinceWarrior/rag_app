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
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from sqlalchemy import and_, delete, func, or_, select, update

from rag_app.config import settings
from rag_app.db.models import (
    TRANSLATABLE_KINDS,
    Chunk,
    Document,
    DocumentKind,
    DocumentStatus,
    DocumentTranslation,
    GlossaryTerm,
    PageEmbedding,
    Segment,
    SegmentKind,
    document_segment_filter,
    is_document_segment,
)
from rag_app.llm.client import (
    SegmentContext,
    Translator,
    detect_lang,
    needs_translation,
    pick_glossary_terms,
)
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
from rag_app.pipeline.page_fallback import (
    extract_selected_pdf_pages,
    page_fallback_allowed,
    page_fallback_error_metadata,
    page_fallback_metadata,
    remap_selected_page_drafts,
    select_page_fallbacks,
)
from rag_app.pipeline.page_routing import RouteRole, merge_page_replacements
from rag_app.pipeline.page_routing_shadow import (
    PageRoutingPlan,
    build_page_routing_plan,
    page_router_allowed,
    page_routing_metadata,
)
from rag_app.pipeline.parse import (
    PDFIUM_LOCK,
    backfill_text_layer,
    load_block_geometry,
    load_content_list,
    pdf_info,
    read_pdf_text_by_page,
    run_mineru,
)
from rag_app.pipeline.parse_quality import evaluate_parse, quality_metadata
from rag_app.pipeline.scan_pdf import build_scan_overlay
from rag_app.pipeline.segments import SegmentDraft, content_list_to_segments
from rag_app.pipeline.technical_entities import (
    audit_unconfirmed_entities,
    protect_entities,
    restore_entities,
)
from rag_app.pipeline.translation_memory import (
    TranslationMemoryMatch,
    TranslationMemoryScope,
    TranslationMemoryService,
)
from rag_app.pipeline.validate import ValidationResult, validate_numbers, validate_standards
from rag_app.rag.chunking import segments_to_chunks
from rag_app.storage.s3 import Storage

logger = logging.getLogger(__name__)


async def _set_status(
    ctx: dict,
    doc_id: uuid.UUID,
    status: DocumentStatus,
    error: str | None = None,
    *,
    parse_revision: int | None = None,
) -> None:
    async with ctx["sessionmaker"]() as session:
        stmt = update(Document).where(Document.id == doc_id)
        if parse_revision is not None:
            stmt = stmt.where(Document.parse_revision == parse_revision)
        await session.execute(stmt.values(status=status, error=error))
        await session.commit()


async def _get_doc(ctx: dict, doc_id: uuid.UUID) -> Document:
    async with ctx["sessionmaker"]() as session:
        doc = await session.get(Document, doc_id)
        if doc is None:
            raise RuntimeError(f"документ {doc_id} не найден")
        return doc


async def _claim_parse(ctx: dict, doc_id: uuid.UUID, parse_revision: int | None) -> int | None:
    """Атомарно занять или перезахватить ту же ревизию после прерывания."""

    async with ctx["sessionmaker"]() as session:
        stmt = update(Document).where(Document.id == doc_id)
        if parse_revision is None:
            # Совместимость со старыми задачами без ревизии: только первый claim.
            stmt = stmt.where(Document.status == DocumentStatus.uploaded)
        else:
            stmt = stmt.where(
                Document.parse_revision == parse_revision,
                or_(
                    Document.status.in_((DocumentStatus.uploaded, DocumentStatus.parsing)),
                    and_(
                        Document.status == DocumentStatus.error,
                        Document.error.like("парсинг:%"),
                    ),
                ),
            )
        claimed = (
            await session.execute(
                stmt.values(status=DocumentStatus.parsing, error=None).returning(Document.parse_revision)
            )
        ).scalar_one_or_none()
        await session.commit()
        return claimed


async def _claim_document_stage(
    ctx: dict,
    doc_id: uuid.UUID,
    parse_revision: int | None,
    *,
    ready: DocumentStatus,
    running: DocumentStatus,
    error_prefix: str,
) -> bool:
    """Revision-safe claim для downstream-стадии; legacy job остаётся совместимым."""

    if parse_revision is None:
        await _set_status(ctx, doc_id, running)
        return True

    async with ctx["sessionmaker"]() as session:
        claimed = (
            await session.execute(
                update(Document)
                .where(
                    Document.id == doc_id,
                    Document.parse_revision == parse_revision,
                    or_(
                        Document.status.in_((ready, running)),
                        and_(
                            Document.status == DocumentStatus.error,
                            Document.error.like(f"{error_prefix}%"),
                        ),
                    ),
                )
                .values(status=running, error=None)
                .returning(Document.id)
            )
        ).scalar_one_or_none()
        await session.commit()
    return claimed is not None


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


async def _vlm_segments(
    backend: str,
    pdf_path: Path,
    out_dir: Path,
    *,
    timeout_s: int | None = None,
) -> list[SegmentDraft]:
    """Парс PDF альтернативным VLM-движком (dots.mocr / PaddleOCR-VL) → SegmentDraft."""
    if backend == "dots_mocr":
        page_dir = await run_dots(pdf_path, out_dir)
        return await asyncio.to_thread(dots_to_segments, page_dir, pdf_path)
    if backend == "paddle_vl":
        await run_paddle(pdf_path, out_dir, timeout_s=timeout_s)
        page_sizes = await asyncio.to_thread(_pdf_page_sizes_from_path, pdf_path)
        return await asyncio.to_thread(paddle_to_segments, out_dir, page_sizes)
    raise RuntimeError(f"неизвестный backend парсера: {backend}")


def _pdf_page_sizes(pdf_bytes: bytes) -> dict[int, tuple[float, float]]:
    """Физические размеры страниц PDF в пунктах, под общей блокировкой pdfium."""

    import pypdfium2 as pdfium  # type: ignore[import-untyped]

    result: dict[int, tuple[float, float]] = {}
    with PDFIUM_LOCK:
        pdf = pdfium.PdfDocument(pdf_bytes)
        try:
            for page_idx in range(len(pdf)):
                page = pdf[page_idx]
                try:
                    width, height = page.get_size()
                finally:
                    page.close()
                result[page_idx] = (float(width), float(height))
        finally:
            pdf.close()
    return result


def _pdf_page_sizes_from_path(pdf_path: Path) -> dict[int, tuple[float, float]]:
    """Прочитать PDF и измерить страницы вне async event loop."""

    return _pdf_page_sizes(pdf_path.read_bytes())


async def _apply_paddle_page_fallback(
    storage: Storage,
    doc: Document,
    primary_final: list[SegmentDraft],
    primary_raw: list[SegmentDraft],
    routing: PageRoutingPlan,
    *,
    n_pages: int,
    parser_revision: int,
    native_text_by_page: Mapping[int, str] | None = None,
) -> tuple[list[SegmentDraft], dict[str, Any], frozenset[int]]:
    """Запустить Paddle отдельно и атомарно принять только выигравшие страницы."""

    with tempfile.TemporaryDirectory(prefix="rag_page_fallback_") as tmp:
        tmp_path = Path(tmp)
        local_file = tmp_path / "source.pdf"
        await storage.download_to(settings.bucket_originals, doc.s3_key_original, local_file)
        selected_pages = tuple(
            sorted(
                decision.page_idx
                for decision in routing.selected
                if decision.role == RouteRole.parser_fallback
            )
        )
        selected_file = tmp_path / "selected.pdf"
        await asyncio.to_thread(
            extract_selected_pdf_pages,
            local_file,
            selected_file,
            selected_pages,
        )
        out_dir = tmp_path / "paddle_out"
        reduced_drafts = await _vlm_segments(
            "paddle_vl",
            selected_file,
            out_dir,
            timeout_s=settings.parser_sidecar_timeout_s,
        )
        fallback_drafts = remap_selected_page_drafts(reduced_drafts, selected_pages)
        page_sizes = await asyncio.to_thread(_pdf_page_sizes_from_path, local_file)
        if len(page_sizes) != n_pages:
            raise RuntimeError(
                f"fallback PDF page count mismatch: expected={n_pages} actual={len(page_sizes)}"
            )
        plan = select_page_fallbacks(
            primary_raw,
            fallback_drafts,
            routing,
            primary_final=primary_final,
            native_text_by_page=native_text_by_page,
            n_pages=n_pages,
            page_sizes=page_sizes,
            parser_revision=parser_revision,
            min_score=settings.parser_page_router_min_score,
            min_margin=settings.parser_page_router_min_margin,
        )
        accepted_drafts = [draft for candidate in plan.candidates for draft in candidate.drafts]
        if accepted_drafts:
            await _upload_segment_images(storage, doc.id, out_dir, accepted_drafts)
        merged = merge_page_replacements(primary_final, plan.candidates, n_pages=n_pages)
    accepted_pages = frozenset(candidate.page_idx for candidate in plan.candidates)
    return merged, page_fallback_metadata(plan, backend="paddle_vl"), accepted_pages


async def parse_document(ctx: dict, doc_id_str: str, parse_revision: int | None = None) -> str:
    doc_id = uuid.UUID(doc_id_str)
    storage: Storage = ctx["storage"]
    claimed_revision = await _claim_parse(ctx, doc_id, parse_revision)
    if claimed_revision is None:
        async with ctx["sessionmaker"]() as session:
            current = await session.get(Document, doc_id)
        current_status = "missing" if current is None else current.status.value
        logger.info(
            "parse %s revision=%s skipped (status=%s)",
            doc_id,
            parse_revision,
            current_status,
        )
        return f"skipped parse revision={parse_revision}: status={current_status}"
    doc = await _get_doc(ctx, doc_id)
    logger.info("parse %s revision=%s (%s)", doc_id, claimed_revision, doc.filename)

    quality_payload: dict[str, Any] | None = None
    quality_backend: str | None = None
    raw_parser_drafts: list[SegmentDraft] | None = None
    backfilled_pages: list[int] = []
    native_text_by_page: dict[int, str] | None = None
    n_pages: int | None
    router_allowed = page_router_allowed(
        settings.parser_page_router_mode,
        owner_sub=doc.owner_sub,
        allowed_owner_subs=settings.parser_page_router_owner_subs,
    )
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
                quality_backend = backend
                out_dir = tmp_path / "parser_out"
                if not doc.parse_force_ocr and backend in ("dots_mocr", "paddle_vl"):
                    # Альтернативный VLM-движок должен получать и PDF с текстовым
                    # слоем, и image-only сканы. Раньше `has_text` молча отправлял
                    # любой скан обратно в MinerU, хотя в документе/аудите оставался
                    # явно выбранный Paddle/dots backend.
                    # Вывод → SegmentDraft напрямую, без mineru content_list/geo.
                    kind = DocumentKind.pdf_text if has_text else DocumentKind.pdf_scan
                    drafts = await _vlm_segments(backend, local_file, out_dir)
                    raw_parser_drafts = list(drafts)
                    if settings.parser_quality_shadow_enabled or router_allowed:
                        native_text_by_page = await asyncio.to_thread(read_pdf_text_by_page, local_file)
                    # PaddleOCR-VL вырезает рисунки в файлы (dots — нет) → грузим в
                    # img_s3, чтобы они появились в текст-просмотре (как у MinerU)
                    if backend == "paddle_vl":
                        await _upload_segment_images(storage, doc_id, out_dir, drafts)
                else:
                    quality_backend = "mineru"
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
                    if kind == DocumentKind.pdf_scan and not drafts and not doc.parse_force_ocr:
                        # `auto`/VLM может вернуть валидный, но пустой content_list
                        # для image-only PDF. Один раз повторяем тем же проверенным
                        # контуром, который использует ручное «Переразобрать (OCR)».
                        # Это не даёт обычной загрузке скана завершиться пустым
                        # документом, хотя принудительный OCR способен его прочитать.
                        logger.warning(
                            "parse %s: MinerU вернул пустой скан — повтор с force OCR",
                            doc_id,
                        )
                        out_dir = tmp_path / "parser_ocr_fallback"
                        content_list_path = await run_mineru(
                            local_file,
                            out_dir,
                            backend=settings.mineru_force_ocr_backend,
                            method="ocr",
                            lang=doc.ocr_lang,
                        )
                        items = load_content_list(content_list_path)
                        drafts = content_list_to_segments(items)
                        quality_backend = "mineru_ocr_fallback"
                    raw_parser_drafts = list(drafts)
                    # pdf_text: VLM местами роняет/прореживает целые страницы —
                    # достраиваем их абзацами из текстового слоя (истина для PDF
                    # с текстом), дедуп против VLM. Сканам слой не поможет (no-op).
                    if kind == DocumentKind.pdf_text:
                        native_text_by_page = await asyncio.to_thread(read_pdf_text_by_page, local_file)
                        drafts, backfilled_pages = await asyncio.to_thread(
                            backfill_text_layer,
                            local_file,
                            drafts,
                            native_text_by_page=native_text_by_page,
                        )
                        if backfilled_pages:
                            logger.info(
                                "parse %s: достроены страницы из слоя: %s",
                                doc_id,
                                backfilled_pages,
                            )
                    if not drafts:
                        raise RuntimeError(
                            "OCR-парсер не извлёк ни одного текстового или структурного блока"
                            if kind == DocumentKind.pdf_scan
                            else "парсер не извлёк ни одного блока"
                        )
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
                # Office всегда разбираем нативно по OOXML location. VLM-разбор
                # DOCX через промежуточный PDF не имеет адресов {t,r,c,p}: в
                # production он показывал текст, но экспорт применял 0 переводов
                # и всё равно завершался `done`. VLM остаётся только для PDF.
                img_dir = tmp_path / "ooxml_img"
                img_dir.mkdir(exist_ok=True)
                drafts = await asyncio.to_thread(ooxml.extract, ext, local_file, img_dir)
                if ext == "docx":
                    await _upload_segment_images(storage, doc_id, img_dir, drafts)
                page_indices = [d.page_idx for d in drafts if d.page_idx is not None]
                n_pages = max(page_indices) + 1 if ext == "pptx" and page_indices else None
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

        if ext == "pdf" and (settings.parser_quality_shadow_enabled or router_allowed):
            assert n_pages is not None
            assert raw_parser_drafts is not None
            assert quality_backend is not None
            quality = evaluate_parse(
                drafts,
                n_pages=n_pages,
                native_text_by_page=native_text_by_page,
            )
            raw_quality = evaluate_parse(
                raw_parser_drafts,
                n_pages=n_pages,
                native_text_by_page=native_text_by_page,
            )
            quality_payload = quality_metadata(
                quality,
                backend=quality_backend,
                raw_report=raw_quality,
                backfilled_pages=backfilled_pages,
            )
            if router_allowed:
                routing_plan = build_page_routing_plan(
                    drafts,
                    raw_parser_drafts,
                    n_pages=n_pages,
                    final_quality=quality,
                    native_text_by_page=native_text_by_page,
                    backfilled_pages=backfilled_pages,
                    explicit_backend=doc.parser_backend is not None,
                    min_raw_score=settings.parser_page_router_min_score,
                    max_pages=settings.parser_page_router_max_pages,
                )
                routing_summary = page_routing_metadata(
                    routing_plan,
                    mode=settings.parser_page_router_mode,
                )
                fallback_summary: dict[str, Any] | None = None
                fallback_attempts = sum(
                    decision.role == RouteRole.parser_fallback for decision in routing_plan.selected
                )
                if page_fallback_allowed(
                    enabled=settings.parser_page_fallback_enabled,
                    router_mode=settings.parser_page_router_mode,
                    router_allowed=router_allowed,
                    primary_backend=quality_backend,
                    attempted_page_count=fallback_attempts,
                ):
                    try:
                        drafts, fallback_summary, fallback_pages = await _apply_paddle_page_fallback(
                            storage,
                            doc,
                            drafts,
                            raw_parser_drafts,
                            routing_plan,
                            n_pages=n_pages,
                            parser_revision=claimed_revision,
                            native_text_by_page=native_text_by_page,
                        )
                        if fallback_summary["accepted_page_count"]:
                            quality = evaluate_parse(
                                drafts,
                                n_pages=n_pages,
                                native_text_by_page=native_text_by_page,
                            )
                            quality_payload = quality_metadata(
                                quality,
                                backend=quality_backend,
                                raw_report=raw_quality,
                                backfilled_pages=[
                                    page for page in backfilled_pages if page not in fallback_pages
                                ],
                            )
                    except Exception as exc:  # noqa: BLE001 - fallback не валит primary
                        logger.warning(
                            "page_fallback doc=%s backend=paddle_vl failed type=%s",
                            doc_id,
                            type(exc).__name__,
                        )
                        fallback_summary = page_fallback_error_metadata(
                            backend="paddle_vl",
                            attempted_page_count=fallback_attempts,
                            error_type=type(exc).__name__,
                        )
                quality_payload["page_routing"] = routing_summary
                if fallback_summary is not None:
                    quality_payload["page_fallback"] = fallback_summary
                logger.info(
                    "page_routing_shadow doc=%s mode=%s eligible=%s selected=%s types=%s roles=%s",
                    doc_id,
                    settings.parser_page_router_mode,
                    routing_summary["eligible_page_count"],
                    routing_summary["selected_page_count"],
                    routing_summary["type_counts"],
                    routing_summary["role_counts"],
                )
            logger.info(
                "parse_quality_shadow doc=%s backend=%s score=%.4f acceptable=%s "
                "raw_score=%.4f backfilled_pages=%s page_coverage=%.4f "
                "duplicate_ratio=%.4f integrity_ratio=%.4f reasons=%s",
                doc_id,
                quality_backend,
                quality.score,
                quality.acceptable,
                raw_quality.score,
                len(set(backfilled_pages)),
                quality.page_coverage,
                quality.duplicate_ratio,
                quality.integrity_ratio,
                ",".join(quality.reasons) or "none",
            )

        if not drafts:
            # Пустой placeholder раньше переводился в пустую строку, был невидим
            # во viewer и всё равно переводил документ в `done`. Сохранять такой
            # разбор нельзя: текущие хорошие сегменты должны остаться нетронутыми,
            # а пользователь должен увидеть явную ошибку OCR.
            raise RuntimeError(
                "OCR-парсер не извлёк ни одного текстового или структурного блока"
                if kind == DocumentKind.pdf_scan
                else "парсер не извлёк ни одного блока"
            )

        async with ctx["sessionmaker"]() as session:
            await session.execute(delete(Segment).where(Segment.document_id == doc_id))
            # Новая ревизия сегментов и старый RAG-индекс не должны сосуществовать:
            # удаляем chunks атомарно с заменой сегментов, а не ждём отдельный job.
            await session.execute(delete(Chunk).where(Chunk.document_id == doc_id))
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
            document_values: dict[str, Any] = {
                "status": DocumentStatus.parsed,
                "error": None,
                "kind": kind.value,
                "page_count": n_pages,
                "segment_count": len(drafts),
                "translated_count": 0,
                "chunk_count": 0,
                "indexed_at": None,
                "index_error": None,
                "s3_key_content_list": artifact_key,
                "parse_quality": quality_payload,
            }
            if ext in ("docx", "xlsx", "pptx"):
                document_values["parser_backend"] = "native_ooxml"
            updated = await session.execute(
                update(Document)
                .where(
                    Document.id == doc_id,
                    Document.parse_revision == claimed_revision,
                    Document.status == DocumentStatus.parsing,
                )
                .values(**document_values)
            )
            if updated.rowcount != 1:
                raise RuntimeError(f"устаревшая ревизия парсинга: {claimed_revision}")
            await session.commit()

        next_job = await ctx["redis"].enqueue_job(
            "translate_document",
            doc_id_str,
            claimed_revision,
            _job_id=f"translate:{doc_id}:{claimed_revision}",
        )
        if next_job is None:
            raise RuntimeError("очередь отклонила задачу перевода")
        # OOXML: ранний рендер оригинала в PDF параллельно переводу — чтобы
        # «как в Microsoft» (дефолт DOCX) открывался сразу, не ждя экспорта.
        if kind in (DocumentKind.docx, DocumentKind.xlsx, DocumentKind.pptx):
            try:
                await ctx["redis"].enqueue_job(
                    "render_original_view",
                    doc_id_str,
                    _job_id=f"vieworig:{doc_id}:{claimed_revision}",
                )
            except Exception as exc:  # ранний preview не блокирует основной pipeline
                logger.warning("render original %s enqueue failed: %s", doc_id, exc)
        return f"parsed [{kind.value}]: {len(drafts)} segments, {n_pages} pages"

    except asyncio.CancelledError:
        await _set_status(
            ctx,
            doc_id,
            DocumentStatus.error,
            "парсинг: задача отменена или воркер остановлен",
            parse_revision=claimed_revision,
        )
        raise
    except Exception as exc:
        logger.exception("parse %s failed", doc_id)
        await _set_status(
            ctx,
            doc_id,
            DocumentStatus.error,
            f"парсинг: {exc}",
            parse_revision=claimed_revision,
        )
        raise


# ---------------------------------------------------------------- translate


async def _translate_validated(
    translator: Translator, text: str, context: SegmentContext
) -> tuple[str, ValidationResult]:
    """Перевод + детерминированная защита сущностей и один повтор с фидбеком."""
    # Документ уже на целевом языке (русский → русский) — переводить нечего,
    # сегмент остаётся как есть. Иначе — гейт по целевому скрипту: переводим
    # любой не-русский текст, включая английские вставки в китайском документе.
    if context.source_lang == context.target_lang or not needs_translation(
        text, context.source_lang, context.target_lang
    ):
        return text, ValidationResult(ok=True)
    tm_mode = settings.translation_memory_mode
    exact = context.translation_memory_exact
    nearest_count = len(context.translation_memory_examples)
    exact_rejected = False
    if tm_mode == "enforce" and exact is not None:
        translated = exact[1]
        result = validate_numbers(text, translated)
        result.standards = validate_standards(text, translated)
        result.ok = result.ok and not result.standards
        result.translation_memory = {
            "schema_version": 1,
            "mode": tm_mode,
            "origin": "exact",
            "entry_id": exact[0],
            "exact_candidate": True,
            "exact_rejected": not result.ok,
            "nearest_candidates": nearest_count,
        }
        if result.ok:
            logger.info("translation_memory mode=enforce origin=exact entry_id=%s", exact[0])
            return translated, result
        exact_rejected = True
    mode = settings.translation_entity_guard_mode
    protected = protect_entities(text) if mode != "off" else None

    async def run(feedback: str | None = None) -> tuple[str, ValidationResult]:
        model_input = protected.text if mode == "enforce" and protected is not None else text
        raw = await translator.translate(model_input, context, feedback=feedback)
        restoration = restore_entities(raw, protected) if mode == "enforce" and protected else None
        translated = restoration.text if restoration else raw
        result = validate_numbers(text, translated)
        result.standards = validate_standards(text, translated)  # §4.3.5
        result.ok = result.ok and not result.standards
        if tm_mode != "off":
            result.translation_memory = {
                "schema_version": 1,
                "mode": tm_mode,
                "origin": "model",
                "entry_id": None,
                "exact_candidate": exact is not None,
                "exact_rejected": exact_rejected,
                "nearest_candidates": nearest_count,
            }
            logger.info(
                "translation_memory mode=%s origin=model exact_candidate=%s "
                "exact_rejected=%s nearest_candidates=%d",
                tm_mode,
                exact is not None,
                exact_rejected,
                nearest_count,
            )

        if protected is not None:
            unconfirmed = audit_unconfirmed_entities(text, translated)
            unconfirmed_total = sum(len(values) for values in unconfirmed.values())
            protected_total = len(protected.entities)
            placeholder_errors = 0
            if restoration is not None:
                placeholder_errors = (
                    len(restoration.missing_tokens)
                    + len(restoration.duplicated_tokens)
                    + len(restoration.unknown_tokens)
                )
                result.ok = result.ok and restoration.ok
            result.entity_guard = {
                "schema_version": 1,
                "mode": mode,
                "protected": protected.counts,
                "protected_total": protected_total,
                "unconfirmed": {kind: len(values) for kind, values in unconfirmed.items()},
                "unconfirmed_total": unconfirmed_total,
                "unconfirmed_rate": unconfirmed_total / protected_total if protected_total else 0.0,
                "placeholder_errors": placeholder_errors,
            }
            logger.info(
                "translation_entity_guard mode=%s protected=%d unconfirmed=%d "
                "unconfirmed_rate=%.6f placeholder_errors=%d",
                mode,
                protected_total,
                unconfirmed_total,
                result.entity_guard["unconfirmed_rate"],
                placeholder_errors,
            )
        return translated, result

    translated, result = await run()
    if result.ok:
        return translated, result
    issues = []
    if result.missing:
        issues.append(f"числа искажены/потеряны: {', '.join(result.missing)}")
    if result.standards:
        issues.append(f"обозначения стандартов искажены/потеряны: {', '.join(result.standards)}")
    if result.entity_guard and result.entity_guard["placeholder_errors"]:
        issues.append("защитные плейсхолдеры технических сущностей потеряны или дублированы")
    feedback = (
        f"{'; '.join(issues)}. Перенеси ВСЕ числа, единицы измерения и обозначения "
        "стандартов (ГОСТ/ISO/API/ASTM и т.п.) без изменений."
    )
    return await run(feedback)


def _context_with_memory(
    context: SegmentContext,
    text: str,
    memory_matches: Mapping[str, TranslationMemoryMatch] | None,
) -> SegmentContext:
    match = memory_matches.get(text) if memory_matches else None
    if match is None:
        return context
    exact = (str(match.exact.entry_id), match.exact.translation) if match.exact is not None else None
    examples = [(item.source_text, item.translation, item.score) for item in match.nearest]
    return replace(
        context,
        translation_memory_exact=exact,
        translation_memory_examples=examples,
    )


def _translation_unit_texts(seg: Segment) -> list[str]:
    if seg.kind != SegmentKind.table:
        return [seg.source_text]
    meta = seg.meta or {}
    values: list[str] = []
    caption = meta.get("caption") or ""
    if caption:
        values.append(caption)
    values.extend(cell for row in (meta.get("table_rows") or []) for cell in row if cell)
    values.extend(
        cell.get("text", "") for row in (meta.get("table_cells") or []) for cell in row if cell.get("text")
    )
    return list(dict.fromkeys(values))


async def _translate_segment(
    translator: Translator,
    seg: Segment,
    context: SegmentContext,
    memory_matches: Mapping[str, TranslationMemoryMatch] | None = None,
) -> dict[str, Any]:
    """Возвращает values для UPDATE сегмента."""
    if seg.kind == SegmentKind.table:
        grid: list[list[str]] = seg.meta.get("table_rows") or []
        cells: list[list[dict[str, Any]]] | None = seg.meta.get("table_cells")
        caption: str = seg.meta.get("caption") or ""
        failures: list[dict[str, Any]] = []
        cache: dict[str, str] = {}  # перевод каждой уникальной ячейки один раз
        guard_aggregate: dict[str, Any] = {
            "schema_version": 1,
            "mode": settings.translation_entity_guard_mode,
            "observations": 0,
            "protected": {},
            "protected_total": 0,
            "unconfirmed": {},
            "unconfirmed_total": 0,
            "unconfirmed_rate": 0.0,
            "placeholder_errors": 0,
        }
        memory_aggregate: dict[str, Any] = {
            "schema_version": 1,
            "mode": settings.translation_memory_mode,
            "observations": 0,
            "origins": {},
            "exact_candidates": 0,
            "exact_rejected": 0,
            "nearest_candidates": 0,
        }

        async def tr(text: str, loc: dict[str, Any] | None = None) -> str:
            if text not in cache:
                local_context = _context_with_memory(context, text, memory_matches)
                ru, vr = await _translate_validated(translator, text, local_context)
                cache[text] = ru
                if vr.entity_guard is not None:
                    guard_aggregate["observations"] += 1
                    for field in ("protected", "unconfirmed"):
                        for kind, count in vr.entity_guard[field].items():
                            current = guard_aggregate[field].get(kind, 0)
                            guard_aggregate[field][kind] = current + count
                    for field in ("protected_total", "unconfirmed_total", "placeholder_errors"):
                        guard_aggregate[field] += vr.entity_guard[field]
                if vr.translation_memory is not None:
                    memory_aggregate["observations"] += 1
                    origin = vr.translation_memory["origin"]
                    memory_aggregate["origins"][origin] = memory_aggregate["origins"].get(origin, 0) + 1
                    memory_aggregate["exact_candidates"] += int(vr.translation_memory["exact_candidate"])
                    memory_aggregate["exact_rejected"] += int(vr.translation_memory["exact_rejected"])
                    memory_aggregate["nearest_candidates"] += vr.translation_memory["nearest_candidates"]
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
            for cell_row in cells:
                row_ru_cells: list[dict[str, Any]] = []
                for c in cell_row:
                    row_ru_cells.append(
                        {"text": await tr(c["text"]), "colspan": c["colspan"], "rowspan": c["rowspan"]}
                    )
                cells_ru.append(row_ru_cells)
            meta["table_cells_ru"] = cells_ru
            preview = "\n".join(" | ".join(c["text"] for c in row) for row in cells_ru)
        else:
            preview = "\n".join(" | ".join(r) for r in rows_ru)

        validation: dict[str, Any] = {}
        if failures:
            validation["cells"] = failures
        if guard_aggregate["observations"]:
            protected_total = guard_aggregate["protected_total"]
            guard_aggregate["unconfirmed_rate"] = (
                guard_aggregate["unconfirmed_total"] / protected_total if protected_total else 0.0
            )
            validation["entity_guard"] = guard_aggregate
        if memory_aggregate["observations"]:
            validation["translation_memory"] = memory_aggregate
        return {
            "translated_text": (meta["caption_ru"] + "\n" + preview).strip(),
            "meta": meta,
            "needs_review": bool(failures),
            "validation": validation or None,
        }

    local_context = _context_with_memory(context, seg.source_text, memory_matches)
    translated, vr = await _translate_validated(translator, seg.source_text, local_context)
    return {
        "translated_text": translated,
        "needs_review": not vr.ok,
        "validation": (
            vr.as_dict()
            if vr.entity_guard is not None or vr.translation_memory is not None or not vr.ok
            else None
        ),
    }


async def translate_document(ctx: dict, doc_id_str: str, parse_revision: int | None = None) -> str:
    doc_id = uuid.UUID(doc_id_str)
    translator: Translator = ctx["translator"]
    t_task = time.monotonic()
    claimed = await _claim_document_stage(
        ctx,
        doc_id,
        parse_revision,
        ready=DocumentStatus.parsed,
        running=DocumentStatus.translating,
        error_prefix="перевод:",
    )
    if not claimed:
        return f"skipped translate revision={parse_revision}"

    doc0 = await _get_doc(ctx, doc_id)

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
        segments = [segment for segment in segments if is_document_segment(segment)]

    # Язык-источник определяем АВТОМАТИЧЕСКИ по тексту документа (домен: ru/en/zh,
    # ТЗ §4.3). Перевод ВСЕГДА на русский; русский документ не переводится
    # (identity-замыкание в _translate_validated). Если язык уже зафиксирован
    # (реразбор) — берём сохранённый, заново не определяем.
    src_lang = (getattr(doc0, "source_lang", None) or "").strip()
    if src_lang not in ("en", "ru", "zh"):
        sample = " ".join((s.source_text or "") for s in segments[:120])[:20000]
        src_lang = detect_lang(sample)
    tgt_lang = "ru"
    if doc0.source_lang != src_lang or doc0.target_lang != tgt_lang:
        async with ctx["sessionmaker"]() as session:
            await session.execute(
                update(Document)
                .where(Document.id == doc_id)
                .values(source_lang=src_lang, target_lang=tgt_lang)
            )
            await session.commit()

    # Глоссарий (roadmap § 3.4 п.1) — только для EN→RU (термины хранятся как
    # en_term/ru_term); для других направлений глоссарий не применяется.
    all_terms: list[tuple[str, str]] = []
    if (src_lang, tgt_lang) == ("en", "ru"):
        async with ctx["sessionmaker"]() as session:
            rows = (await session.execute(select(GlossaryTerm.en_term, GlossaryTerm.ru_term))).all()
        all_terms = sorted(((en, ru) for en, ru in rows), key=lambda t: -len(t[0]))

    # Контекст (roadmap § 3.4): заголовок раздела + предыдущий абзац + термины + направление.
    contexts: dict[uuid.UUID, SegmentContext] = {}
    cur_heading: str | None = None
    prev_text: str | None = None
    for seg in segments:
        terms = pick_glossary_terms(seg.source_text, all_terms) if all_terms else []
        if seg.kind == SegmentKind.heading:
            # заголовки — без текстового контекста: модель может «утащить» его в ответ
            contexts[seg.id] = SegmentContext(glossary=terms, source_lang=src_lang, target_lang=tgt_lang)
            cur_heading = seg.source_text
            prev_text = None
            continue
        contexts[seg.id] = SegmentContext(
            heading=cur_heading,
            prev_text=prev_text,
            glossary=terms,
            source_lang=src_lang,
            target_lang=tgt_lang,
        )
        if seg.kind == SegmentKind.paragraph:
            prev_text = seg.source_text

    todo = [s for s in segments if s.kind in TRANSLATABLE_KINDS and s.translated_text is None]
    done_count = len([s for s in segments if s.kind in TRANSLATABLE_KINDS]) - len(todo)
    memory_matches: dict[str, TranslationMemoryMatch] = {}
    if settings.translation_memory_mode != "off" and src_lang != tgt_lang:
        memory_texts = list(
            dict.fromkeys(text for seg in todo for text in _translation_unit_texts(seg) if text.strip())
        )
        memory_matches = await TranslationMemoryService(ctx["sessionmaker"], ctx["embedder"]).lookup_batch(
            memory_texts,
            source_lang=src_lang,
            target_lang=tgt_lang,
            scope=TranslationMemoryScope(
                owner_sub=doc0.owner_sub,
                folder_id=doc0.folder_id,
                project=doc0.project_object,
            ),
        )
    logger.info("translate %s: %d сегментов (готово ранее: %d)", doc_id, len(todo), done_count)

    sem = asyncio.Semaphore(settings.translate_concurrency)
    failures: list[str] = []

    async def work(seg: Segment) -> tuple[uuid.UUID, dict[str, Any]] | None:
        async with sem:
            try:
                return seg.id, await _translate_segment(translator, seg, contexts[seg.id], memory_matches)
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
        await _set_status(
            ctx,
            doc_id,
            DocumentStatus.translated,
            parse_revision=parse_revision,
        )
        next_job = await ctx["redis"].enqueue_job(
            "export_document",
            doc_id_str,
            parse_revision,
            _job_id=f"export:{doc_id}:{parse_revision}",
        )
        if next_job is None:
            raise RuntimeError("очередь отклонила задачу экспорта")
        return f"translated: {len(todo)} segments"

    except asyncio.CancelledError:
        await _set_status(
            ctx,
            doc_id,
            DocumentStatus.error,
            "перевод: задача отменена или воркер остановлен",
            parse_revision=parse_revision,
        )
        raise
    except Exception as exc:
        logger.exception("translate %s failed", doc_id)
        await _set_status(
            ctx,
            doc_id,
            DocumentStatus.error,
            f"перевод: {exc}",
            parse_revision=parse_revision,
        )
        raise


# ------------------------------------------- translate-to (доп. язык, ТЗ §4.3)


async def _set_translation_status(
    ctx: dict, doc_id: uuid.UUID, target_lang: str, status: str, **fields: Any
) -> None:
    async with ctx["sessionmaker"]() as session:
        row = (
            await session.execute(
                select(DocumentTranslation).where(
                    DocumentTranslation.document_id == doc_id,
                    DocumentTranslation.target_lang == target_lang,
                )
            )
        ).scalar_one_or_none()
        if row is None:
            row = DocumentTranslation(document_id=doc_id, target_lang=target_lang, status=status, **fields)
            session.add(row)
        else:
            row.status = status
            for k, v in fields.items():
                setattr(row, k, v)
        await session.commit()


async def _export_translation(
    ctx: dict, doc: Document, segments: list[Segment], target_lang: str
) -> dict[str, str]:
    """Экспортный артефакт перевода: DOCX для pdf/text, инъекция в копию оригинала
    для OOXML. Сегменты уже гидратированы (translated_text/meta под этот язык)."""
    storage: Storage = ctx["storage"]
    segments = [segment for segment in segments if is_document_segment(segment)]
    stem = Path(doc.filename).stem
    kind = doc.kind.value if hasattr(doc.kind, "value") else doc.kind
    out: dict[str, str] = {}
    if kind in (DocumentKind.pdf_text.value, DocumentKind.pdf_scan.value, DocumentKind.text.value):
        images: dict[str, bytes] = {}
        for s in segments:
            key = (s.meta or {}).get("img_s3") if s.kind == SegmentKind.image else None
            if key and key not in images:
                try:
                    images[key] = await storage.get_bytes(settings.bucket_artifacts, key)
                except Exception:  # noqa: BLE001
                    pass
        data = await asyncio.to_thread(build_docx, doc.filename, segments, images)
        key = f"{doc.id}/{target_lang}/{stem}.{target_lang}.docx"
        await storage.put_bytes(settings.bucket_exports, key, data, _DOCX_MIME)
        out["s3_key_docx"] = key
    else:  # OOXML — перевод обратно в копию оригинала, формат сохраняется
        ext = kind
        if ext == "xlsx":
            translations = {
                s.source_text: s.translated_text for s in segments if s.translated_text is not None
            }
        else:
            translations = {
                ooxml.location_key(s.meta["location"]): s.translated_text
                for s in segments
                if s.translated_text is not None and s.meta.get("location")
            }
        with tempfile.TemporaryDirectory(prefix="rag_tr_") as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / Path(doc.filename).name
            dst = tmp_path / f"{stem}.{target_lang}.{ext}"
            await storage.download_to(settings.bucket_originals, doc.s3_key_original, src)
            applied = await asyncio.to_thread(ooxml.inject, ext, src, dst, translations)
            if translations and applied == 0:
                raise RuntimeError(
                    "OOXML-экспорт не применил ни одного перевода: "
                    "структурные адреса сегментов не совпадают с оригиналом"
                )
            key = f"{doc.id}/{target_lang}/{dst.name}"
            await storage.put_bytes(settings.bucket_exports, key, dst.read_bytes(), _OOXML_MIME[ext])
            out["s3_key_source"] = key
    return out


async def translate_to_language(ctx: dict, doc_id_str: str, target_lang: str) -> str:
    """Перевод документа на ДОПОЛНИТЕЛЬНЫЙ язык (ТЗ §4.3: RU→EN/RU→ZH и пр.). Не
    трогает основной перевод (Segment.translated_text); результат — в строке
    DocumentTranslation(document, target_lang) + экспортный артефакт."""
    doc_id = uuid.UUID(doc_id_str)
    translator: Translator = ctx["translator"]
    t_task = time.monotonic()
    doc = await _get_doc(ctx, doc_id)
    src_lang = (getattr(doc, "source_lang", None) or "").strip() or "ru"

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
        segments = [segment for segment in segments if is_document_segment(segment)]

    todo = [s for s in segments if s.kind in TRANSLATABLE_KINDS]
    await _set_translation_status(
        ctx,
        doc_id,
        target_lang,
        "translating",
        segment_count=len(todo),
        translated_count=0,
        error=None,
    )

    contexts: dict[uuid.UUID, SegmentContext] = {}
    cur_heading: str | None = None
    prev_text: str | None = None
    for seg in segments:
        if seg.kind == SegmentKind.heading:
            contexts[seg.id] = SegmentContext(source_lang=src_lang, target_lang=target_lang)
            cur_heading = seg.source_text
            prev_text = None
            continue
        contexts[seg.id] = SegmentContext(
            heading=cur_heading, prev_text=prev_text, source_lang=src_lang, target_lang=target_lang
        )
        if seg.kind == SegmentKind.paragraph:
            prev_text = seg.source_text

    memory_matches: dict[str, TranslationMemoryMatch] = {}
    if settings.translation_memory_mode != "off" and src_lang != target_lang:
        memory_texts = list(
            dict.fromkeys(text for seg in todo for text in _translation_unit_texts(seg) if text.strip())
        )
        memory_matches = await TranslationMemoryService(ctx["sessionmaker"], ctx["embedder"]).lookup_batch(
            memory_texts,
            source_lang=src_lang,
            target_lang=target_lang,
            scope=TranslationMemoryScope(
                owner_sub=doc.owner_sub,
                folder_id=doc.folder_id,
                project=doc.project_object,
            ),
        )

    sem = asyncio.Semaphore(settings.translate_concurrency)
    failures: list[str] = []

    async def work(seg: Segment) -> tuple[uuid.UUID, dict[str, Any]] | None:
        async with sem:
            try:
                return seg.id, await _translate_segment(translator, seg, contexts[seg.id], memory_matches)
            except Exception as exc:
                failures.append(f"сегмент {seg.idx}: {exc}")
                logger.error("translate %s->%s seg %d: %s", doc_id, target_lang, seg.idx, exc)
                return None

    data: dict[str, Any] = {}
    review = 0
    try:
        results = await asyncio.gather(*[work(s) for s in todo])
        for r in results:
            if r is None:
                continue
            seg_id, vals = r
            entry: dict[str, Any] = {"text": vals.get("translated_text")}
            if vals.get("meta") is not None:
                entry["meta"] = vals["meta"]
            if vals.get("validation") is not None:
                entry["validation"] = vals["validation"]
            if vals.get("needs_review"):
                review += 1
            data[str(seg_id)] = entry
        if failures:
            raise RuntimeError(
                f"не переведено сегментов: {len(failures)}; первые: " + "; ".join(failures[:3])
            )

        # гидратация в памяти (без коммита) → сборка экспорта существующими билдерами
        by_id = {str(s.id): s for s in segments}
        for sid, entry in data.items():
            s = by_id.get(sid)
            if s is None:
                continue
            s.translated_text = entry.get("text")
            if entry.get("meta"):
                s.meta = {**(s.meta or {}), **entry["meta"]}
        await _set_translation_status(ctx, doc_id, target_lang, "exporting", data=data)
        export_keys = await _export_translation(ctx, doc, segments, target_lang)
        await _set_translation_status(
            ctx,
            doc_id,
            target_lang,
            "done",
            translated_count=len(data),
            needs_review_count=review,
            data=data,
            **export_keys,
        )
        logger.info(
            "translate %s->%s: %d сегм. за %.1fс",
            doc_id,
            target_lang,
            len(data),
            time.monotonic() - t_task,
        )
        return f"translated {doc_id}->{target_lang}: {len(data)} segments"
    except asyncio.CancelledError:
        await _set_translation_status(
            ctx,
            doc_id,
            target_lang,
            "error",
            error="задача отменена или воркер остановлен",
            data=data,
        )
        raise
    except Exception as exc:
        logger.exception("translate %s->%s failed", doc_id, target_lang)
        await _set_translation_status(ctx, doc_id, target_lang, "error", error=str(exc)[:500], data=data)
        raise


# ---------------------------------------------------------------- index (RAG, этап 3)


async def index_document(
    ctx: dict,
    doc_id_str: str,
    parse_revision: int | None = None,
) -> str:
    """Чанкинг по структуре → эмбеддинги EN/RU → chunks (roadmap § 5).

    Не влияет на статус перевода: ошибки идут в documents.index_error.
    """
    doc_id = uuid.UUID(doc_id_str)
    if parse_revision is None:
        # Legacy job до revision-aware контура нельзя безопасно сопоставить с
        # конкретным снимком Segment: reparse повышает revision раньше замены
        # строк. Такой job пропускаем; все текущие enqueue передают ревизию явно.
        logger.warning("index %s skipped: parse_revision отсутствует", doc_id)
        return "skipped index: missing parse_revision"
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
            if parse_revision is not None:
                current_revision = (
                    await session.execute(
                        select(Document.parse_revision).where(Document.id == doc_id).with_for_update()
                    )
                ).scalar_one_or_none()
                if current_revision != parse_revision:
                    logger.info(
                        "index %s revision=%s skipped (current=%s)",
                        doc_id,
                        parse_revision,
                        current_revision,
                    )
                    return f"skipped index revision={parse_revision}: current={current_revision}"
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
        try:
            # pdf_scan (скан/чертёж/P&ID): дообогащаем VL-описанием смысла изображения
            # и переиндексируем. Маркер meta.vl_describe на сегментах не даёт зациклиться.
            if settings.vl_enabled and not any((s.meta or {}).get("vl_describe") for s in segments):
                doc = await _get_doc(ctx, doc_id)
                if doc.kind == DocumentKind.pdf_scan.value:
                    await ctx["redis"].enqueue_job(
                        "describe_images",
                        doc_id_str,
                        parse_revision,
                        _job_id=f"vl:{doc_id}:{parse_revision}",
                    )
            # Визуальный индекс — отдельное необязательное обогащение. Ошибка чтения
            # документа/Redis после COMMIT не должна откатывать уже корректные chunks.
            if settings.visual_enabled:
                vdoc = await _get_doc(ctx, doc_id)
                vkind = vdoc.kind if isinstance(vdoc.kind, str) else vdoc.kind.value
                if vkind == DocumentKind.pdf_scan.value or any(
                    (s.meta or {}).get("img_s3") for s in segments
                ):
                    await ctx["redis"].enqueue_job(
                        "index_pages_visual",
                        doc_id_str,
                        parse_revision,
                        _job_id=f"vis:{doc_id}:{uuid.uuid4().hex[:8]}",
                    )
        except Exception as exc:  # noqa: BLE001 — основной текстовый индекс уже записан
            logger.warning(
                "index %s: необязательное VL/visual-обогащение не поставлено в очередь: %s",
                doc_id,
                exc,
            )
        return f"indexed: {len(drafts)} chunks"
    except Exception as exc:
        logger.exception("index %s failed", doc_id)
        async with ctx["sessionmaker"]() as session:
            if parse_revision is not None:
                current_revision = (
                    await session.execute(
                        select(Document.parse_revision).where(Document.id == doc_id).with_for_update()
                    )
                ).scalar_one_or_none()
                if current_revision != parse_revision:
                    logger.info(
                        "index cleanup %s revision=%s skipped (current=%s)",
                        doc_id,
                        parse_revision,
                        current_revision,
                    )
                    return f"skipped failed index revision={parse_revision}: current={current_revision}"
            # После переразбора прежние chunks относятся к другой ревизии
            # сегментов. Оставлять их при сбое нового индексирования опаснее, чем
            # временно исключить документ из RAG: поиск иначе цитирует уже
            # отсутствующий текст.
            await session.execute(delete(Chunk).where(Chunk.document_id == doc_id))
            await session.execute(
                update(Document)
                .where(Document.id == doc_id)
                .values(
                    chunk_count=0,
                    indexed_at=None,
                    index_error=str(exc)[:1000],
                )
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


def _locate_in_pdf(pdf_bytes: bytes, items: list[tuple[uuid.UUID, str]]) -> dict[uuid.UUID, dict[str, Any]]:
    """Сегмент (по его тексту) → положение в PDF: {page, bbox (top-left, pt), pagesize}.
    Для кросс-навигации панелей: клик по абзацу слева подсвечивает его справа и наоборот.
    Кандидат-страница — по нормализованному совпадению, bbox — поиском pypdfium2 по
    «сырому» тексту сегмента целиком (rect на каждую строку совпадения → накрывает
    весь абзац, не только первую строку); при неточном совпадении сужаем сниппет
    (весь текст → 60→40→25→15→10→6→4→3→2 слова) до первого удачного поиска.
    Несовпавшие сегменты пропускаются."""
    import pypdfium2 as pdfium

    out: dict[uuid.UUID, dict[str, Any]] = {}
    with PDFIUM_LOCK:
        pdf = pdfium.PdfDocument(pdf_bytes)
        try:
            n = len(pdf)
            pages: list[tuple[Any, Any, tuple[float, float]]] = []
            norm_texts: list[str] = []
            for i in range(n):
                page = pdf[i]
                tp = page.get_textpage()
                pages.append((page, tp, page.get_size()))
                norm_texts.append(_norm_match(tp.get_text_bounded()))
            for sid, text in items:
                raw = (text or "").strip()
                nm = _norm_match(raw)
                if len(nm) < 6:
                    continue
                snip = nm[:40]
                cand = next((i for i in range(n) if snip in norm_texts[i]), None)
                if cand is None:
                    continue
                _, tp, (w, h) = pages[cand]
                words = raw.split()
                bbox: list[float] | None = None
                # Сначала пробуем ВЕСЬ текст сегмента — тогда bbox накрывает все его
                # строки (rs — рект на каждую строку совпадения), а не только начало.
                # Узкие сниппеты (6→2 слова) — страховка на случай, если абзац целиком
                # не совпал дословно с текстовым слоем PDF (перенос/лигатура/разметка).
                ladder = sorted({len(words), 60, 40, 25, 15, 10, 6, 4, 3, 2}, reverse=True)
                for k in ladder:
                    if k <= 0 or k > len(words):
                        continue
                    needle = " ".join(words[:k])
                    if len(needle) < 4:
                        continue
                    m = tp.search(needle, match_case=False, match_whole_word=False).get_next()
                    if m:
                        start, count = m
                        rs = [tp.get_rect(r) for r in range(tp.count_rects(start, count))]
                        if rs:
                            left = min(r[0] for r in rs)
                            bot = min(r[1] for r in rs)
                            right = max(r[2] for r in rs)
                            top = max(r[3] for r in rs)
                            bbox = [round(left, 1), round(h - top, 1), round(right, 1), round(h - bot, 1)]
                        break
                if bbox:
                    out[sid] = {"page": cand, "bbox": bbox, "pagesize": [round(w, 1), round(h, 1)]}
        finally:
            pdf.close()
    return out


async def _store_cross_locs(
    ctx: dict, doc_id: uuid.UUID, segments: list[Segment], left_pdf: bytes, right_pdf: bytes
) -> None:
    """Посчитать и сохранить в meta положение каждого сегмента в ЛЕВОМ (оригинал) и
    ПРАВОМ (перевод) PDF — для кросс-навигации по клику между панелями вьювера."""
    items_l = [(s.id, s.source_text) for s in segments if s.source_text]
    items_r = [(s.id, s.translated_text) for s in segments if s.translated_text]
    left = await asyncio.to_thread(_locate_in_pdf, left_pdf, items_l)
    right = await asyncio.to_thread(_locate_in_pdf, right_pdf, items_r)
    if not left and not right:
        return
    async with ctx["sessionmaker"]() as session:
        rows = (await session.execute(select(Segment).where(Segment.document_id == doc_id))).scalars().all()
        for s in rows:
            meta = dict(s.meta or {})
            meta["loc_left"] = left.get(s.id)
            meta["loc_right"] = right.get(s.id)
            s.meta = meta
        await session.commit()
    logger.info("export %s: cross-loc left=%d right=%d", doc_id, len(left), len(right))


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


async def describe_images(
    ctx: dict,
    doc_id_str: str,
    parse_revision: int | None = None,
) -> str:
    """VL-обогащение СКАНОВ (pdf_scan): вся страница — рисунок (P&ID/чертёж/фото) →
    Qwen3.5-VL раскрывает смысл по-русски → сегменты-описания (kind=image) →
    переиндексация. Нужно для сканов без текстового слоя (искомость в чате).

    pdf_text/docx/pptx рисунки НЕ предописываются: их вырезанные кропы (img_s3)
    подаются в Qwen3.5-vision ON-DEMAND в чате (rag/chat.py), а в текст-просмотре —
    картинка + родная подпись."""
    if not settings.vl_enabled:
        return "vl disabled"
    doc_id = uuid.UUID(doc_id_str)
    if parse_revision is None:
        logger.warning("describe_images %s skipped: parse_revision отсутствует", doc_id)
        return "skipped VL: missing parse_revision"
    storage: Storage = ctx["storage"]
    doc = await _get_doc(ctx, doc_id)
    if doc.parse_revision != parse_revision:
        return f"skipped VL revision={parse_revision}: current={doc.parse_revision}"
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
            current_revision = (
                await session.execute(
                    select(Document.parse_revision).where(Document.id == doc_id).with_for_update()
                )
            ).scalar_one_or_none()
            if current_revision != parse_revision:
                logger.info(
                    "describe_images mutation %s revision=%s skipped (current=%s)",
                    doc_id,
                    parse_revision,
                    current_revision,
                )
                return f"skipped VL revision={parse_revision}: current={current_revision}"
            # идемпотентность: убрать прежние VL-описания этого документа
            await session.execute(
                delete(Segment).where(
                    Segment.document_id == doc_id,
                    Segment.meta.op("->>")("vl_describe") == "true",
                )
            )
            base_idx = (
                await session.execute(
                    select(func.coalesce(func.max(Segment.idx), 0)).where(Segment.document_id == doc_id)
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
                        # Внутренний русскоязычный контекст для RAG — не перевод
                        # документа и не пользовательский графический фрагмент.
                        translated_text=None,
                        meta={"vl_describe": True},
                    )
                )
            cnt = (
                await session.execute(
                    select(func.count())
                    .select_from(Segment)
                    .where(Segment.document_id == doc_id)
                    .where(document_segment_filter())
                )
            ).scalar_one()
            trc = (
                await session.execute(
                    select(func.count())
                    .select_from(Segment)
                    .where(
                        Segment.document_id == doc_id,
                        document_segment_filter(),
                        Segment.translated_text.isnot(None),
                    )
                )
            ).scalar_one()
            await session.execute(
                update(Document)
                .where(
                    Document.id == doc_id,
                    Document.parse_revision == parse_revision,
                )
                .values(segment_count=cnt, translated_count=trc)
            )
            await session.commit()

        await ctx["redis"].enqueue_job(
            "index_document",
            doc_id_str,
            parse_revision,
            _job_id=f"index:{doc_id}:{parse_revision}:vl",
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
            (
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
            )
            .scalars()
            .all()
        )
    return {p for p in rows if p is not None}


async def index_pages_visual(
    ctx: dict,
    doc_id_str: str,
    parse_revision: int | None = None,
) -> str:
    """Эмбеддинги страниц-картинок (Qwen3-VL-Embedding-8B) для визуального retrieval:
    печати/штампы/чертежи/схемы, где текстовый контур слаб. pdf_scan — все страницы
    (вся страница = визуал); pdf_text/docx/pptx — только страницы с вырезанными
    рисунками (img_s3), остальное берёт текстовый поиск."""
    doc_id = uuid.UUID(doc_id_str)
    if parse_revision is None:
        logger.warning("visual index %s skipped: parse_revision отсутствует", doc_id)
        return "skipped visual index: missing parse_revision"
    if not settings.visual_enabled:
        return "visual disabled"
    doc = await _get_doc(ctx, doc_id)
    if doc.parse_revision != parse_revision:
        return f"skipped visual revision={parse_revision}: current={doc.parse_revision}"
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
            current_revision = (
                await session.execute(
                    select(Document.parse_revision).where(Document.id == doc_id).with_for_update()
                )
            ).scalar_one_or_none()
            if current_revision != parse_revision:
                logger.info(
                    "visual index mutation %s revision=%s skipped (current=%s)",
                    doc_id,
                    parse_revision,
                    current_revision,
                )
                return f"skipped visual revision={parse_revision}: current={current_revision}"
            await session.execute(delete(PageEmbedding).where(PageEmbedding.document_id == doc_id))
            session.add_all(PageEmbedding(document_id=doc_id, page_idx=p, emb=e) for p, e in embs)
            await session.execute(
                update(Document)
                .where(
                    Document.id == doc_id,
                    Document.parse_revision == parse_revision,
                    Document.index_error.like("visual:%"),
                )
                .values(index_error=None)
            )
            await session.commit()
        logger.info("visual index %s: %d страниц (%s)", doc_id, len(embs), kind)
        return f"visual indexed: {len(embs)} pages"
    except Exception as exc:
        logger.exception("visual index %s failed", doc_id)
        async with ctx["sessionmaker"]() as session:
            current_revision = (
                await session.execute(
                    select(Document.parse_revision).where(Document.id == doc_id).with_for_update()
                )
            ).scalar_one_or_none()
            if current_revision != parse_revision:
                logger.info(
                    "visual index cleanup %s revision=%s skipped (current=%s)",
                    doc_id,
                    parse_revision,
                    current_revision,
                )
                return f"skipped failed visual revision={parse_revision}: current={current_revision}"
            await session.execute(
                update(Document)
                .where(
                    Document.id == doc_id,
                    Document.parse_revision == parse_revision,
                )
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
        terms = (await session.execute(select(GlossaryTerm.en_term, GlossaryTerm.ru_term))).all()
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
            await session.execute(update(Document).where(Document.id == doc_id).values(s3_key_view_orig=key))
            await session.commit()
        return f"view_orig ready: {key}"
    except Exception as exc:  # noqa: BLE001 — рендер необязателен
        logger.warning("render_original_view %s: %s", doc_id, exc)
        return f"failed: {exc}"


async def export_document(ctx: dict, doc_id_str: str, parse_revision: int | None = None) -> str:
    doc_id = uuid.UUID(doc_id_str)
    storage: Storage = ctx["storage"]
    doc = await _get_doc(ctx, doc_id)
    claimed = await _claim_document_stage(
        ctx,
        doc_id,
        parse_revision,
        ready=DocumentStatus.translated,
        running=DocumentStatus.exporting,
        error_prefix="экспорт:",
    )
    if not claimed:
        return f"skipped export revision={parse_revision}"

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
            segments = [segment for segment in segments if is_document_segment(segment)]

        values: dict[str, Any] = {"status": DocumentStatus.done, "error": None}
        stem = Path(doc.filename).stem

        if doc.kind in (DocumentKind.pdf_text, DocumentKind.pdf_scan):
            # 1) редактируемый DOCX из сегментов (+ встроенные рисунки)
            images: dict[str, bytes] = {}
            for s in segments:
                key = (s.meta or {}).get("img_s3") if s.kind == SegmentKind.image else None
                if key and key not in images:
                    try:
                        images[key] = await storage.get_bytes(settings.bucket_artifacts, key)
                    except Exception:  # noqa: BLE001 — рисунок необязателен, заглушка в DOCX
                        pass
            data = await asyncio.to_thread(build_docx, doc.filename, segments, images)
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
                    mono_data, dual_data = await asyncio.to_thread(build_scan_overlay, local_pdf, segments)
                    mono_key = f"{doc_id}/{stem}.ru.pdf"
                    dual_key = f"{doc_id}/{stem}.en-ru.pdf"
                    await storage.put_bytes(settings.bucket_exports, mono_key, mono_data, "application/pdf")
                    await storage.put_bytes(settings.bucket_exports, dual_key, dual_data, "application/pdf")
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
                        pdf_bytes = await render_to_pdf(docx_path, tmp_path, settings.office_render_timeout_s)
                        pdf_key = f"{doc_id}/{stem}.ru.pdf"
                        await storage.put_bytes(
                            settings.bucket_exports, pdf_key, pdf_bytes, "application/pdf"
                        )
                        values["s3_key_export_pdf"] = pdf_key
                        # кросс-навигация: положение сегмента в оригинале (слева) и
                        # reflow-PDF перевода (справа) — клик подсвечивает на др. стороне
                        await _store_cross_locs(ctx, doc_id, segments, local_pdf.read_bytes(), pdf_bytes)
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
                    s.source_text: s.translated_text for s in segments if s.translated_text is not None
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
                if translations and applied == 0:
                    raise RuntimeError(
                        "OOXML-экспорт не применил ни одного перевода: "
                        "структурные адреса сегментов не совпадают с оригиналом"
                    )
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
                            page_map, n_pages = await asyncio.to_thread(_assign_pdf_pages, orig_pdf, segments)
                            if page_map:
                                async with ctx["sessionmaker"]() as session:
                                    await session.execute(
                                        update(Segment),
                                        [{"id": sid, "page_idx": p} for sid, p in page_map.items()],
                                    )
                                    await session.commit()
                                values["page_count"] = n_pages
                            # кросс-навигация docx: положение сегмента в оригинале
                            # (view_orig) и переводе (view_ru) — клик подсвечивает напротив
                            await _store_cross_locs(ctx, doc_id, segments, orig_pdf, ru_pdf)
                    except Exception as exc:  # noqa: BLE001 — рендер необязателен
                        logger.warning("export %s: LibreOffice-рендер не удался (%s)", doc_id, exc)

        async with ctx["sessionmaker"]() as session:
            stmt = update(Document).where(Document.id == doc_id)
            if parse_revision is not None:
                stmt = stmt.where(Document.parse_revision == parse_revision)
            updated = await session.execute(stmt.values(**values))
            if updated.rowcount != 1:
                raise RuntimeError(f"устаревшая ревизия экспорта: {parse_revision}")
            await session.commit()

        await ctx["redis"].enqueue_job(
            "index_document",
            doc_id_str,
            parse_revision,
            _job_id=f"index:{doc_id}:{parse_revision}",
        )
        await ctx["redis"].enqueue_job(
            "index_pages_visual",
            doc_id_str,
            parse_revision,
            _job_id=f"vindex:{doc_id}:{parse_revision}",
        )
        return f"exported: {', '.join(k for k in values if k.startswith('s3_'))}"

    except asyncio.CancelledError:
        await _set_status(
            ctx,
            doc_id,
            DocumentStatus.error,
            "экспорт: задача отменена или воркер остановлен",
            parse_revision=parse_revision,
        )
        raise
    except Exception as exc:
        logger.exception("export %s failed", doc_id)
        await _set_status(
            ctx,
            doc_id,
            DocumentStatus.error,
            f"экспорт: {exc}",
            parse_revision=parse_revision,
        )
        raise
