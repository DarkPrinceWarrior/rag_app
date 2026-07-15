"""Чанкинг по структуре документа (roadmap § 5 п.1), не по символам.

Чанк = раздел (заголовок + его абзацы до следующего заголовка), таблицы —
отдельными чанками. Метаданные: путь заголовков, страницы, bbox и id
сегментов (для подсветки цитат). Длинные разделы режутся по абзацам,
короткие соседние куски одного раздела не плодятся отдельно.
"""

from __future__ import annotations

import re
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from rag_app.config import settings
from rag_app.db.models import Segment, SegmentKind

# подпись рисунка/таблицы (Figure 9.1 / Рис. 9.1 / Table 2 / Схема 3 …) — для
# индексации вырезанного рисунка по его подписи (поиск + on-demand vision в чате)
_CAPTION_RE = re.compile(
    r"^\s*(figure|fig\.?|рис\.?|рисунок|table|таблиц[аы]|схема|диаграмма)\s*\.?\s*\d", re.I
)


def _is_caption(text: str | None) -> bool:
    return bool(_CAPTION_RE.match(text or ""))


@dataclass
class ChunkDraft:
    idx: int
    kind: str  # section | table
    heading_path: str
    text_en: str
    text_ru: str
    page_start: int | None
    page_end: int | None
    meta: dict[str, Any] = field(default_factory=dict)


def _heading_path(stack: list[str]) -> str:
    return " → ".join(stack)


def _annotate_hierarchy(drafts: list[ChunkDraft], segments: list[Segment]) -> None:
    """Attach stable, citation-safe parent/ordering metadata to leaf chunks."""

    if not drafts or not segments:
        return
    section_stack: list[uuid.UUID] = []
    section_by_segment: dict[str, tuple[uuid.UUID, ...]] = {}
    segments_by_id = {str(segment.id): segment for segment in segments}
    for segment in segments:
        if segment.kind == SegmentKind.heading:
            level = max(segment.heading_level or 1, 1)
            del section_stack[level - 1 :]
            section_stack.append(segment.id)
        section_by_segment[str(segment.id)] = tuple(section_stack)

    section_chunks: dict[str, list[ChunkDraft]] = defaultdict(list)
    for draft in drafts:
        segment_ids = [str(value) for value in draft.meta.get("segment_ids", [])]
        member_segments = [
            segments_by_id[segment_id]
            for segment_id in segment_ids
            if segment_id in segments_by_id
        ]
        if not member_segments:
            continue
        first = min(member_segments, key=lambda segment: (segment.idx, segment.id.int))
        section_path = section_by_segment.get(str(first.id), ())
        root_id = uuid.uuid5(uuid.NAMESPACE_URL, f"rag-section-root:{first.document_id}")
        section_id = section_path[-1] if section_path else root_id
        draft.meta.update(
            {
                "section_id": str(section_id),
                "section_path": [str(value) for value in section_path],
                "parent_id": str(section_id),
                "source_ordinal": min(segment.idx for segment in member_segments),
            }
        )
        table_groups = {
            str((segment.meta or {}).get("table_merge_group"))
            for segment in member_segments
            if (segment.meta or {}).get("table_merge_group")
        }
        continuation_groups = {
            str((segment.meta or {}).get("continuation_group"))
            for segment in member_segments
            if (segment.meta or {}).get("continuation_group")
        }
        logical_table_ids = table_groups or continuation_groups
        if draft.kind == "table" and len(logical_table_ids) == 1:
            draft.meta["logical_table_id"] = next(iter(logical_table_ids))
        continuation_indexes = {
            value
            for segment in member_segments
            if isinstance((value := (segment.meta or {}).get("continuation_index")), int)
        }
        if draft.kind == "table" and len(continuation_indexes) == 1:
            draft.meta["continuation_index"] = next(iter(continuation_indexes))
        section_chunks[str(section_id)].append(draft)

    for chunks in section_chunks.values():
        chunks.sort(key=lambda draft: (int(draft.meta["source_ordinal"]), draft.idx))
        for ordinal, draft in enumerate(chunks):
            draft.meta["ordinal_in_section"] = ordinal


def _flush(
    drafts: list[ChunkDraft],
    stack: list[str],
    buf: list[Segment],
    kind: str = "section",
) -> None:
    if not buf:
        return
    en_parts = [s.source_text for s in buf if s.source_text]
    ru_parts = [s.translated_text or s.source_text for s in buf]
    pages = [s.page_idx for s in buf if s.page_idx is not None]
    drafts.append(
        ChunkDraft(
            idx=len(drafts),
            kind=kind,
            heading_path=_heading_path(stack),
            text_en="\n".join(en_parts).strip(),
            text_ru="\n".join(ru_parts).strip(),
            page_start=min(pages) if pages else None,
            page_end=max(pages) if pages else None,
            meta={
                "segment_ids": [str(s.id) for s in buf],
                "bboxes": [
                    {"page": s.page_idx, "bbox": s.meta.get("bbox_pt")}
                    for s in buf
                    if s.meta.get("bbox_pt") is not None
                ],
            },
        )
    )
    buf.clear()


def segments_to_chunks(segments: list[Segment]) -> list[ChunkDraft]:
    drafts: list[ChunkDraft] = []
    stack: list[str] = []  # путь заголовков
    buf: list[Segment] = []
    buf_chars = 0

    for i, seg in enumerate(segments):
        if seg.kind == SegmentKind.heading:
            _flush(drafts, stack, buf)
            buf_chars = 0
            level = max(seg.heading_level or 1, 1)
            del stack[level - 1 :]
            stack.append(seg.source_text.strip())
            # заголовок входит в текст следующего чанка
            buf.append(seg)
            buf_chars = len(seg.source_text)
        elif seg.kind == SegmentKind.table:
            table_buf = [seg]
            _flush(drafts, stack + ["таблица"], table_buf, kind="table")
        elif seg.kind == SegmentKind.image:
            # Вырезанный рисунок (img_s3) → ОТДЕЛЬНЫЙ image-чанк: текст = подпись (своя
            # или из соседнего абзаца «Figure N»), meta.img_s3 = кроп для on-demand
            # vision в чате. Если ни картинки, ни подписи — пропускаем.
            img_key = (seg.meta or {}).get("img_s3")
            cap_en = (seg.source_text or "").strip()
            cap_ru = (seg.translated_text or seg.source_text or "").strip()
            if not cap_en and i + 1 < len(segments):
                nxt = segments[i + 1]
                if nxt.kind == SegmentKind.paragraph and _is_caption(nxt.source_text):
                    cap_en = (nxt.source_text or "").strip()
                    cap_ru = (nxt.translated_text or nxt.source_text or "").strip()
            if img_key or cap_en:
                _flush(drafts, stack, buf)
                buf_chars = 0
                page = seg.page_idx
                meta: dict[str, Any] = {"segment_ids": [str(seg.id)]}
                if img_key:
                    meta["img_s3"] = img_key
                drafts.append(
                    ChunkDraft(
                        idx=len(drafts),
                        kind="image",
                        heading_path=_heading_path(stack),
                        text_en=cap_en or f"Figure (page {(page or 0) + 1})",
                        text_ru=cap_ru or f"Рисунок (стр. {(page or 0) + 1})",
                        page_start=page,
                        page_end=page,
                        meta=meta,
                    )
                )
        elif seg.kind in (SegmentKind.paragraph, SegmentKind.equation):
            text_len = len(seg.source_text)
            if buf_chars + text_len > settings.chunk_max_chars and buf_chars > settings.chunk_min_chars:
                _flush(drafts, stack, buf)
                buf_chars = 0
            buf.append(seg)
            buf_chars += text_len

    _flush(drafts, stack, buf)
    # таблицы и рисунки не фильтруем по длине: короткое описание — всё равно ценный чанк
    kept = [
        d
        for d in drafts
        if d.kind in ("table", "image") or len(d.text_en) + len(d.text_ru) >= settings.chunk_min_chars // 2
    ]
    for i, d in enumerate(kept):
        d.idx = i
    _annotate_hierarchy(kept, segments)
    return kept
