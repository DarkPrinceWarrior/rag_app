"""Консервативный выбор постраничного parser fallback без вызова моделей."""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rag_app.db.models import SegmentKind
from rag_app.pipeline.page_routing import PageCandidate, RouteRole
from rag_app.pipeline.page_routing_shadow import PageRoutingPlan
from rag_app.pipeline.parse_quality import evaluate_parse, should_select_fallback
from rag_app.pipeline.segments import SegmentDraft


@dataclass(frozen=True)
class PageFallbackDecision:
    page_idx: int
    accepted: bool
    reason: str


@dataclass(frozen=True)
class PageFallbackPlan:
    candidates: tuple[PageCandidate, ...]
    decisions: tuple[PageFallbackDecision, ...]


def page_fallback_allowed(
    *,
    enabled: bool,
    router_mode: str,
    router_allowed: bool,
    primary_backend: str,
    attempted_page_count: int,
) -> bool:
    """Разрешить model call только для canary-владельца и MinerU primary."""

    if attempted_page_count < 0:
        raise ValueError("attempted_page_count must be non-negative")
    return (
        enabled
        and router_mode == "canary"
        and router_allowed
        and primary_backend == "mineru"
        and attempted_page_count > 0
    )


def _page_drafts(drafts: Sequence[SegmentDraft], page_idx: int) -> list[SegmentDraft]:
    return [
        SegmentDraft(
            idx=index,
            kind=draft.kind,
            source_text=draft.source_text,
            page_idx=0,
            heading_level=draft.heading_level,
            meta=copy.deepcopy(draft.meta),
        )
        for index, draft in enumerate(drafts)
        if draft.page_idx == page_idx
    ]


def _has_table(drafts: Sequence[SegmentDraft]) -> bool:
    return any(draft.kind == SegmentKind.table for draft in drafts)


def _with_page_geometry(
    drafts: Sequence[SegmentDraft],
    *,
    page_idx: int,
    page_size: tuple[float, float],
) -> tuple[SegmentDraft, ...]:
    width, height = page_size
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid page size for page {page_idx}")
    result: list[SegmentDraft] = []
    for index, draft in enumerate(drafts):
        meta = copy.deepcopy(draft.meta)
        meta.update(
            {
                "bbox_pt": [0.0, 0.0, float(width), float(height)],
                "page_size_pt": [float(width), float(height)],
                "geometry_precision": "full_page",
            }
        )
        result.append(
            SegmentDraft(
                idx=index,
                kind=draft.kind,
                source_text=draft.source_text,
                page_idx=page_idx,
                heading_level=draft.heading_level,
                meta=meta,
            )
        )
    return tuple(result)


def select_page_fallbacks(
    primary_raw: Sequence[SegmentDraft],
    fallback: Sequence[SegmentDraft],
    routing: PageRoutingPlan,
    *,
    n_pages: int,
    page_sizes: Mapping[int, tuple[float, float]],
    parser_revision: int,
    min_score: float = 0.70,
    min_margin: float = 0.05,
    backend: str = "paddle_vl",
) -> PageFallbackPlan:
    """Принять только явно выбранные router страницы с доказанным улучшением."""

    if n_pages < 0:
        raise ValueError("n_pages must be non-negative")
    if parser_revision < 0:
        raise ValueError("parser_revision must be non-negative")
    if not backend:
        raise ValueError("backend must be non-empty")
    for draft in fallback:
        if draft.page_idx is None or not 0 <= draft.page_idx < n_pages:
            raise ValueError(f"fallback page_idx out of range: {draft.page_idx}")

    candidates: list[PageCandidate] = []
    decisions: list[PageFallbackDecision] = []
    for route in routing.selected:
        if route.role != RouteRole.parser_fallback:
            continue
        primary_page = _page_drafts(primary_raw, route.page_idx)
        fallback_page = _page_drafts(fallback, route.page_idx)
        if not fallback_page:
            decisions.append(PageFallbackDecision(route.page_idx, False, "candidate_missing"))
            continue
        primary_report = evaluate_parse(primary_page, n_pages=1)
        fallback_report = evaluate_parse(fallback_page, n_pages=1)
        if route.reason == "table_requires_structure_check":
            accepted = (
                not _has_table(primary_page)
                and _has_table(fallback_page)
                and fallback_report.acceptable
                and fallback_report.score >= min_score
            )
            reason = "table_structure_recovered" if accepted else "table_not_improved"
        else:
            accepted = should_select_fallback(
                primary_report,
                fallback_report,
                min_score=min_score,
                min_margin=min_margin,
            )
            reason = "quality_improved" if accepted else "quality_not_improved"
        if not accepted:
            decisions.append(PageFallbackDecision(route.page_idx, False, reason))
            continue
        page_size = page_sizes.get(route.page_idx)
        if page_size is None:
            decisions.append(PageFallbackDecision(route.page_idx, False, "page_size_missing"))
            continue
        candidates.append(
            PageCandidate(
                page_idx=route.page_idx,
                backend=backend,
                parser_revision=parser_revision,
                drafts=_with_page_geometry(
                    fallback_page,
                    page_idx=route.page_idx,
                    page_size=page_size,
                ),
            )
        )
        decisions.append(PageFallbackDecision(route.page_idx, True, reason))
    return PageFallbackPlan(tuple(candidates), tuple(decisions))


def page_fallback_metadata(
    plan: PageFallbackPlan,
    *,
    backend: str,
    status: str = "ok",
) -> dict[str, Any]:
    """Обезличенная сводка без текста и номеров страниц."""

    reasons = Counter(decision.reason for decision in plan.decisions)
    return {
        "schema_version": 1,
        "status": status,
        "backend": backend,
        "attempted_page_count": len(plan.decisions),
        "accepted_page_count": len(plan.candidates),
        "reason_counts": dict(sorted(reasons.items())),
    }


def page_fallback_error_metadata(
    *,
    backend: str,
    attempted_page_count: int,
    error_type: str,
) -> dict[str, Any]:
    """Безопасная сводка ошибки без сообщения, путей и содержимого документа."""

    if attempted_page_count < 0:
        raise ValueError("attempted_page_count must be non-negative")
    if not backend or not error_type:
        raise ValueError("backend and error_type must be non-empty")
    return {
        "schema_version": 1,
        "status": "error",
        "backend": backend,
        "attempted_page_count": attempted_page_count,
        "accepted_page_count": 0,
        "reason_counts": {error_type: 1},
    }
