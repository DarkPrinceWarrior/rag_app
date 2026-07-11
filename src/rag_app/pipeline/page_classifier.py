"""Deterministic page-type signals for shadow parser routing."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rag_app.db.models import SegmentKind
from rag_app.pipeline.page_routing import PageType
from rag_app.pipeline.parse_quality import ParseQualityReport
from rag_app.pipeline.segments import SegmentDraft

_SPACE_RE = re.compile(r"\s+")
_FIELD_LINE_RE = re.compile(r"(?m)^\s*[^:\n|]{1,48}:\s*(?:\S.*)?$")
_FORM_MARK_RE = re.compile(r"(?:_{3,}|\.{5,}|\[\s*[xX]?\s*\]|[\u2610\u2611\u2612\u25a1])")
_FORM_HINT_RE = re.compile(
    r"\b(?:application\s+form|questionnaire|invoice|purchase\s+order|bill\s+to|"
    r"ship\s+to|form|\u0444\u043e\u0440\u043c\u0430|\u0430\u043d\u043a\u0435\u0442\u0430|\u0437\u0430\u044f\u0432\u043b\u0435\u043d\u0438\u0435|\u043e\u043f\u0440\u043e\u0441\u043d\u044b\u0439\s+\u043b\u0438\u0441\u0442)\b",
    re.IGNORECASE,
)
_CHART_HINT_RE = re.compile(
    r"\b(?:chart|graph|plot|histogram|scatter|bar\s+chart|pie\s+chart|"
    r"\u0433\u0440\u0430\u0444\u0438\u043a|\u0433\u0438\u0441\u0442\u043e\u0433\u0440\u0430\u043c\u043c\u0430|\u0434\u0438\u0430\u0433\u0440\u0430\u043c\u043c\u0430)\b",
    re.IGNORECASE,
)
_DIAGRAM_HINT_RE = re.compile(
    r"\b(?:flowchart|block\s+diagram|schematic|wiring\s+diagram|process\s+flow|"
    r"p\s*&\s*id|p&id|\u0431\u043b\u043e\u043a-\u0441\u0445\u0435\u043c\u0430|\u0441\u0445\u0435\u043c\u0430|\u0447\u0435\u0440\u0442\u0435\u0436)\b",
    re.IGNORECASE,
)
_META_TEXT_KEYS = ("caption", "category", "label", "title", "type")


@dataclass(frozen=True)
class PageTypeClassification:
    """A conservative classification and the aggregate evidence behind it."""

    page_idx: int
    page_type: PageType
    confidence: float
    reasons: tuple[str, ...]


def _normalized_text(value: str) -> str:
    return _SPACE_RE.sub(" ", value).strip()


def _meta_text(meta: Mapping[str, Any]) -> str:
    return " ".join(str(meta[key]) for key in _META_TEXT_KEYS if meta.get(key))


def _classify_page(
    page_idx: int,
    drafts: Sequence[SegmentDraft],
    *,
    native_text: str,
    quality: ParseQualityReport | None,
) -> PageTypeClassification:
    counts = Counter(draft.kind for draft in drafts)
    parsed_text = "\n".join(draft.source_text for draft in drafts if draft.source_text)
    metadata_text = " ".join(_meta_text(draft.meta) for draft in drafts)
    evidence_text = "\n".join((parsed_text, metadata_text, native_text))
    parsed_chars = len(_normalized_text(parsed_text))
    native_chars = len(_normalized_text(native_text))
    image_count = counts[SegmentKind.image]
    table_count = counts[SegmentKind.table]
    equation_count = counts[SegmentKind.equation]
    text_count = counts[SegmentKind.heading] + counts[SegmentKind.paragraph]

    field_count = len(_FIELD_LINE_RE.findall(evidence_text))
    form_mark_count = len(_FORM_MARK_RE.findall(evidence_text))
    has_form_hint = bool(_FORM_HINT_RE.search(evidence_text))
    if (has_form_hint and field_count + form_mark_count >= 1) or field_count >= 4 or form_mark_count >= 3:
        reasons = ["form_layout"]
        if has_form_hint:
            reasons.append("form_keyword")
        if field_count:
            reasons.append("label_value_fields")
        if form_mark_count:
            reasons.append("fillable_marks")
        return PageTypeClassification(page_idx, PageType.form, 0.92, tuple(reasons))

    has_chart_hint = bool(_CHART_HINT_RE.search(evidence_text))
    has_diagram_hint = bool(_DIAGRAM_HINT_RE.search(evidence_text))
    if image_count and has_chart_hint and has_diagram_hint:
        return PageTypeClassification(
            page_idx,
            PageType.mixed,
            0.78,
            ("image_segments", "chart_and_diagram_cues"),
        )
    if image_count and has_chart_hint:
        return PageTypeClassification(
            page_idx,
            PageType.chart,
            0.90,
            ("image_segments", "chart_caption_or_metadata"),
        )
    if image_count and has_diagram_hint:
        return PageTypeClassification(
            page_idx,
            PageType.diagram,
            0.90,
            ("image_segments", "diagram_caption_or_metadata"),
        )

    structured_kinds = sum(bool(counts[kind]) for kind in (
        SegmentKind.table,
        SegmentKind.image,
        SegmentKind.equation,
    ))
    if structured_kinds >= 2 and (text_count or parsed_chars >= 80):
        return PageTypeClassification(
            page_idx,
            PageType.mixed,
            0.78,
            ("multiple_structured_kinds", "text_content"),
        )
    if table_count:
        return PageTypeClassification(
            page_idx,
            PageType.table,
            0.90,
            ("table_segments",),
        )
    if equation_count and text_count:
        return PageTypeClassification(
            page_idx,
            PageType.mixed,
            0.72,
            ("equation_segments", "text_content"),
        )
    if image_count and (text_count or parsed_chars >= 80):
        return PageTypeClassification(
            page_idx,
            PageType.mixed,
            0.70,
            ("image_segments", "text_content", "no_visual_subtype_cue"),
        )

    page_quality = quality.pages[page_idx] if quality is not None else None
    if native_chars < 20 and parsed_chars >= 40 and not table_count and not image_count:
        return PageTypeClassification(
            page_idx,
            PageType.scan,
            0.80,
            ("ocr_text_without_native_text",),
        )
    if parsed_chars >= 40:
        return PageTypeClassification(
            page_idx,
            PageType.text,
            0.85 if native_chars else 0.75,
            ("text_segments", "native_text_present") if native_chars else ("text_segments",),
        )
    if native_chars >= 40:
        return PageTypeClassification(
            page_idx,
            PageType.text,
            0.65,
            ("native_text_only",),
        )

    unknown_reasons: list[str] = []
    if page_quality is not None and page_quality.segment_count == 0:
        unknown_reasons.append("no_parser_segments")
    if page_quality is not None and page_quality.text_chars < 40:
        unknown_reasons.append("sparse_page_quality")
    if image_count:
        unknown_reasons.extend(("image_segments", "no_visual_subtype_cue"))
    if equation_count:
        unknown_reasons.append("equation_only")
    if not unknown_reasons:
        unknown_reasons.append("insufficient_evidence")
    return PageTypeClassification(page_idx, PageType.unknown, 0.0, tuple(unknown_reasons))


def classify_page_types(
    drafts: Sequence[SegmentDraft],
    *,
    n_pages: int,
    quality: ParseQualityReport | None = None,
    native_text_by_page: Mapping[int, str] | None = None,
) -> tuple[PageTypeClassification, ...]:
    """Classify every physical page without running or mutating a parser."""

    if n_pages < 0:
        raise ValueError("n_pages must be non-negative")
    if quality is not None and len(quality.pages) != n_pages:
        raise ValueError("quality page count must match n_pages")

    by_page: dict[int, list[SegmentDraft]] = {page_idx: [] for page_idx in range(n_pages)}
    for draft in drafts:
        if draft.page_idx is not None and 0 <= draft.page_idx < n_pages:
            by_page[draft.page_idx].append(draft)

    native = native_text_by_page or {}
    return tuple(
        _classify_page(
            page_idx,
            by_page[page_idx],
            native_text=native.get(page_idx, ""),
            quality=quality,
        )
        for page_idx in range(n_pages)
    )
