"""Чистая тенёвая логика постраничной маршрутизации и безопасного merge."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from rag_app.pipeline.segments import SegmentDraft


class PageType(StrEnum):
    text = "text"
    table = "table"
    form = "form"
    chart = "chart"
    diagram = "diagram"
    scan = "scan"
    mixed = "mixed"
    unknown = "unknown"


class RouteRole(StrEnum):
    primary = "primary"
    parser_fallback = "parser_fallback"
    structured_extraction = "structured_extraction"
    chart_extraction = "chart_extraction"
    diagram_extraction = "diagram_extraction"


@dataclass(frozen=True)
class PageRoutingSignals:
    page_idx: int
    page_type: PageType
    raw_score: float
    final_score: float
    backfilled: bool = False
    explicit_backend: bool = False


@dataclass(frozen=True)
class PageRouteDecision:
    page_idx: int
    role: RouteRole
    reason: str


@dataclass(frozen=True)
class PageCandidate:
    """Результат одного backend для целой физической страницы."""

    page_idx: int
    backend: str
    parser_revision: int
    drafts: tuple[SegmentDraft, ...]
    require_geometry: bool = True


def propose_page_route(
    signals: PageRoutingSignals,
    *,
    min_raw_score: float = 0.70,
) -> PageRouteDecision:
    """Предложить роль sidecar/fallback; функция не запускает модель и не пишет данные."""

    if not 0.0 <= min_raw_score <= 1.0:
        raise ValueError("min_raw_score must be between 0 and 1")
    if signals.explicit_backend:
        return PageRouteDecision(signals.page_idx, RouteRole.primary, "explicit_backend")
    if signals.page_type == PageType.form:
        return PageRouteDecision(
            signals.page_idx,
            RouteRole.structured_extraction,
            "form_requires_schema_extraction",
        )
    if signals.page_type == PageType.chart:
        return PageRouteDecision(
            signals.page_idx,
            RouteRole.chart_extraction,
            "chart_requires_data_extraction",
        )
    if signals.page_type == PageType.diagram:
        return PageRouteDecision(
            signals.page_idx,
            RouteRole.diagram_extraction,
            "diagram_requires_structured_representation",
        )
    if signals.page_type == PageType.table:
        return PageRouteDecision(
            signals.page_idx,
            RouteRole.parser_fallback,
            "table_requires_structure_check",
        )
    if signals.raw_score < min_raw_score:
        return PageRouteDecision(signals.page_idx, RouteRole.parser_fallback, "low_raw_score")
    if signals.backfilled:
        return PageRouteDecision(signals.page_idx, RouteRole.parser_fallback, "backfilled_primary")
    return PageRouteDecision(signals.page_idx, RouteRole.primary, "primary_acceptable")


def _valid_geometry(meta: Mapping[str, Any]) -> bool:
    bbox = meta.get("bbox_pt")
    page_size = meta.get("page_size_pt")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    if not isinstance(page_size, (list, tuple)) or len(page_size) != 2:
        return False
    try:
        x0, y0, x1, y1 = (float(value) for value in bbox)
        width, height = (float(value) for value in page_size)
    except (TypeError, ValueError):
        return False
    return width > 0 and height > 0 and 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height


def _copy_draft(
    draft: SegmentDraft,
    *,
    idx: int,
    page_idx: int | None,
    meta_updates: Mapping[str, Any] | None = None,
) -> SegmentDraft:
    meta = copy.deepcopy(draft.meta)
    if meta_updates:
        meta.update(meta_updates)
    return SegmentDraft(
        idx=idx,
        kind=draft.kind,
        source_text=draft.source_text,
        page_idx=page_idx,
        heading_level=draft.heading_level,
        meta=meta,
    )


def merge_page_replacements(
    primary: Sequence[SegmentDraft],
    candidates: Sequence[PageCandidate],
    *,
    n_pages: int,
) -> list[SegmentDraft]:
    """Атомарно заменить только целые страницы и перестроить глобальный reading order."""

    if n_pages < 0:
        raise ValueError("n_pages must be non-negative")

    by_page: dict[int, PageCandidate] = {}
    for candidate in candidates:
        if not 0 <= candidate.page_idx < n_pages:
            raise ValueError(f"candidate page_idx out of range: {candidate.page_idx}")
        if candidate.page_idx in by_page:
            raise ValueError(f"duplicate candidate page_idx: {candidate.page_idx}")
        if not candidate.drafts:
            raise ValueError(f"candidate page {candidate.page_idx} is empty")
        for draft in candidate.drafts:
            if draft.page_idx not in (0, candidate.page_idx):
                raise ValueError(
                    f"candidate page {candidate.page_idx} contains page_idx={draft.page_idx}"
                )
            if candidate.require_geometry and not _valid_geometry(draft.meta):
                raise ValueError(
                    f"candidate page {candidate.page_idx} lacks canonical bbox_pt geometry"
                )
        by_page[candidate.page_idx] = candidate

    primary_by_page: dict[int, list[SegmentDraft]] = {}
    tail: list[SegmentDraft] = []
    for draft in primary:
        if draft.page_idx is None:
            tail.append(draft)
        elif not 0 <= draft.page_idx < n_pages:
            raise ValueError(f"primary page_idx out of range: {draft.page_idx}")
        else:
            primary_by_page.setdefault(draft.page_idx, []).append(draft)

    merged: list[SegmentDraft] = []
    for page_idx in range(n_pages):
        selected = by_page.get(page_idx)
        if selected is None:
            merged.extend(
                _copy_draft(draft, idx=0, page_idx=page_idx)
                for draft in primary_by_page.get(page_idx, [])
            )
            continue
        merged.extend(
            _copy_draft(
                draft,
                idx=0,
                page_idx=page_idx,
                meta_updates={
                    "parser_backend": selected.backend,
                    "parser_revision": selected.parser_revision,
                },
            )
            for draft in selected.drafts
        )

    merged.extend(_copy_draft(draft, idx=0, page_idx=None) for draft in tail)
    for idx, draft in enumerate(merged):
        draft.idx = idx
    return merged
