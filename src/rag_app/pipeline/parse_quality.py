from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rag_app.db.models import SegmentKind
from rag_app.pipeline.segments import SegmentDraft

_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class PageParseQuality:
    page_idx: int
    segment_count: int
    text_chars: int
    has_structured_content: bool


@dataclass(frozen=True)
class ParseQualityReport:
    """Deterministic quality signals, not a parser-provided confidence score."""

    score: float
    acceptable: bool
    page_coverage: float
    native_text_coverage: float | None
    duplicate_ratio: float
    integrity_ratio: float
    pages: tuple[PageParseQuality, ...]
    reasons: tuple[str, ...]


def quality_metadata(
    report: ParseQualityReport,
    *,
    backend: str,
    raw_report: ParseQualityReport | None = None,
    backfilled_pages: Sequence[int] = (),
) -> dict[str, Any]:
    """Serialize privacy-safe shadow signals for storage on a document."""

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "backend": backend,
        "score": report.score,
        "acceptable": report.acceptable,
        "page_count": len(report.pages),
        "content_page_count": sum(
            page.text_chars > 0 or page.has_structured_content for page in report.pages
        ),
        "structured_page_count": sum(page.has_structured_content for page in report.pages),
        "page_coverage": report.page_coverage,
        "native_text_coverage": report.native_text_coverage,
        "duplicate_ratio": report.duplicate_ratio,
        "integrity_ratio": report.integrity_ratio,
        "reasons": list(report.reasons),
    }
    if raw_report is None:
        return metadata

    backfilled_page_count = len(set(backfilled_pages))
    metadata.update(
        {
            "schema_version": 2,
            "raw_parser": quality_metadata(raw_report, backend=backend),
            "backfilled_page_count": backfilled_page_count,
            "backfilled_page_ratio": round(
                backfilled_page_count / max(1, len(report.pages)),
                4,
            ),
        }
    )
    return metadata


def _normalized_text(value: str) -> str:
    return _SPACE_RE.sub(" ", value).strip().casefold()


def _valid_bbox(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return False
    try:
        x0, y0, x1, y1 = (float(item) for item in value)
    except (TypeError, ValueError):
        return False
    return x1 > x0 and y1 > y0


def _valid_table_grid(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    if not all(isinstance(row, list) for row in value):
        return False
    widths = {len(row) for row in value}
    return len(widths) == 1 and 0 not in widths


def _integrity_ratio(drafts: Sequence[SegmentDraft]) -> tuple[float, int, int]:
    checked = 0
    invalid = 0
    for draft in drafts:
        bbox = draft.meta.get("bbox")
        if bbox is not None:
            checked += 1
            invalid += not _valid_bbox(bbox)

        if draft.kind == SegmentKind.table and "table_rows" in draft.meta:
            checked += 1
            invalid += not _valid_table_grid(draft.meta["table_rows"])

    ratio = 1.0 if checked == 0 else 1.0 - invalid / checked
    return ratio, invalid, checked


def _duplicate_ratio(drafts: Sequence[SegmentDraft]) -> float:
    texts = [_normalized_text(draft.source_text) for draft in drafts]
    texts = [text for text in texts if len(text) >= 20]
    if not texts:
        return 0.0
    counts = Counter(texts)
    duplicate_chars = sum((count - 1) * len(text) for text, count in counts.items() if count > 1)
    total_chars = sum(len(text) for text in texts)
    return duplicate_chars / total_chars


def evaluate_parse(
    drafts: Sequence[SegmentDraft],
    *,
    n_pages: int,
    native_text_by_page: Mapping[int, str] | None = None,
    min_score: float = 0.70,
) -> ParseQualityReport:
    """Evaluate parser output without invoking a model or mutating stored artifacts."""

    if n_pages < 0:
        raise ValueError("n_pages must be non-negative")
    if not 0.0 <= min_score <= 1.0:
        raise ValueError("min_score must be between 0 and 1")

    reasons: list[str] = []
    valid_pages = {
        draft.page_idx for draft in drafts if draft.page_idx is not None and 0 <= draft.page_idx < n_pages
    }
    invalid_page_count = sum(
        draft.page_idx is not None and not 0 <= draft.page_idx < n_pages for draft in drafts
    )

    page_coverage = 1.0 if n_pages == 0 else len(valid_pages) / n_pages
    if not drafts:
        reasons.append("empty_output")
    if page_coverage < 1.0:
        reasons.append("missing_pages")
    if invalid_page_count:
        reasons.append("invalid_page_indices")

    pages: list[PageParseQuality] = []
    for page_idx in range(n_pages):
        page_drafts = [draft for draft in drafts if draft.page_idx == page_idx]
        pages.append(
            PageParseQuality(
                page_idx=page_idx,
                segment_count=len(page_drafts),
                text_chars=sum(len(_normalized_text(draft.source_text)) for draft in page_drafts),
                has_structured_content=any(
                    draft.kind in {SegmentKind.image, SegmentKind.table, SegmentKind.equation}
                    for draft in page_drafts
                ),
            )
        )

    total_text_chars = sum(page.text_chars for page in pages)
    content_pages = sum(page.text_chars > 0 or page.has_structured_content for page in pages)
    content_coverage = 1.0 if n_pages == 0 else content_pages / n_pages
    density_denominator = max(1, sum(page.text_chars > 0 for page in pages)) * 40
    text_density = min(1.0, total_text_chars / density_denominator)
    if total_text_chars == 0 and any(page.has_structured_content for page in pages):
        text_density = 0.5
    if drafts and content_coverage < 1.0:
        reasons.append("pages_without_content")
    if drafts and text_density < 0.5:
        reasons.append("sparse_text")

    duplicate_ratio = _duplicate_ratio(drafts)
    if duplicate_ratio > 0.25:
        reasons.append("duplicate_content")

    integrity_ratio, invalid_structures, _ = _integrity_ratio(drafts)
    if invalid_structures:
        reasons.append("invalid_structures")
    if invalid_page_count:
        integrity_ratio *= max(0.0, 1.0 - invalid_page_count / max(1, len(drafts)))

    native_text_coverage: float | None = None
    if native_text_by_page:
        expected_chars = sum(len(_normalized_text(text)) for text in native_text_by_page.values())
        if expected_chars:
            native_text_coverage = min(1.0, total_text_chars / expected_chars)
            if native_text_coverage < 0.5:
                reasons.append("low_native_text_coverage")

    density_signal = text_density
    if native_text_coverage is not None:
        density_signal = (text_density + native_text_coverage) / 2

    score = (
        0.35 * page_coverage
        + 0.10 * content_coverage
        + 0.25 * density_signal
        + 0.15 * (1.0 - duplicate_ratio)
        + 0.15 * integrity_ratio
    )
    if not drafts:
        score = 0.0
    score = round(max(0.0, min(1.0, score)), 4)

    return ParseQualityReport(
        score=score,
        acceptable=score >= min_score,
        page_coverage=round(page_coverage, 4),
        native_text_coverage=None if native_text_coverage is None else round(native_text_coverage, 4),
        duplicate_ratio=round(duplicate_ratio, 4),
        integrity_ratio=round(integrity_ratio, 4),
        pages=tuple(pages),
        reasons=tuple(reasons),
    )


def should_select_fallback(
    primary: ParseQualityReport | None,
    fallback: ParseQualityReport | None,
    *,
    min_score: float = 0.70,
    min_margin: float = 0.05,
) -> bool:
    """Choose fallback only for a failed/weak primary and a materially better result."""

    if not 0.0 <= min_score <= 1.0:
        raise ValueError("min_score must be between 0 and 1")
    if not 0.0 <= min_margin <= 1.0:
        raise ValueError("min_margin must be between 0 and 1")
    if fallback is None or fallback.score < min_score:
        return False
    if primary is None:
        return True
    return primary.score < min_score and fallback.score >= primary.score + min_margin
