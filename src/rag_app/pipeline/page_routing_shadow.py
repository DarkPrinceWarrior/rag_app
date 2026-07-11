"""Планирование теневой постраничной маршрутизации без вызова моделей."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rag_app.pipeline.page_classifier import PageTypeClassification, classify_page_types
from rag_app.pipeline.page_routing import (
    PageRouteDecision,
    PageRoutingSignals,
    RouteRole,
    propose_page_route,
)
from rag_app.pipeline.parse_quality import ParseQualityReport, evaluate_parse
from rag_app.pipeline.segments import SegmentDraft

_ROLE_PRIORITY = {
    RouteRole.structured_extraction: 0,
    RouteRole.chart_extraction: 1,
    RouteRole.diagram_extraction: 2,
    RouteRole.parser_fallback: 3,
    RouteRole.primary: 4,
}


@dataclass(frozen=True)
class PageRoutingPlan:
    classifications: tuple[PageTypeClassification, ...]
    decisions: tuple[PageRouteDecision, ...]
    selected: tuple[PageRouteDecision, ...]


def page_router_allowed(
    mode: str,
    *,
    owner_sub: str | None,
    allowed_owner_subs: Sequence[str],
) -> bool:
    if mode == "off":
        return False
    if mode == "shadow":
        return True
    if mode == "canary":
        return owner_sub is not None and owner_sub in allowed_owner_subs
    raise ValueError(f"unknown page router mode: {mode}")


def _drafts_for_page(
    drafts: Sequence[SegmentDraft],
    page_idx: int,
) -> list[SegmentDraft]:
    return [
        SegmentDraft(
            idx=index,
            kind=draft.kind,
            source_text=draft.source_text,
            page_idx=0,
            heading_level=draft.heading_level,
            meta=draft.meta,
        )
        for index, draft in enumerate(drafts)
        if draft.page_idx == page_idx
    ]


def _page_score(
    drafts: Sequence[SegmentDraft],
    *,
    page_idx: int,
    native_text: str | None,
) -> float:
    native = None if native_text is None else {0: native_text}
    return evaluate_parse(
        _drafts_for_page(drafts, page_idx),
        n_pages=1,
        native_text_by_page=native,
    ).score


def build_page_routing_plan(
    final_drafts: Sequence[SegmentDraft],
    raw_drafts: Sequence[SegmentDraft],
    *,
    n_pages: int,
    final_quality: ParseQualityReport,
    native_text_by_page: Mapping[int, str] | None = None,
    backfilled_pages: Sequence[int] = (),
    explicit_backend: bool = False,
    min_raw_score: float = 0.70,
    max_pages: int = 12,
) -> PageRoutingPlan:
    """Классифицировать страницы и выбрать ограниченный набор shadow-кандидатов."""

    if n_pages < 0:
        raise ValueError("n_pages must be non-negative")
    if len(final_quality.pages) != n_pages:
        raise ValueError("final quality page count must match n_pages")
    if max_pages < 0:
        raise ValueError("max_pages must be non-negative")

    native = native_text_by_page or {}
    classifications = classify_page_types(
        final_drafts,
        n_pages=n_pages,
        quality=final_quality,
        native_text_by_page=native,
    )
    backfilled = set(backfilled_pages)
    decisions: list[PageRouteDecision] = []
    for classification in classifications:
        page_idx = classification.page_idx
        native_text = native.get(page_idx)
        decisions.append(
            propose_page_route(
                PageRoutingSignals(
                    page_idx=page_idx,
                    page_type=classification.page_type,
                    raw_score=_page_score(
                        raw_drafts,
                        page_idx=page_idx,
                        native_text=native_text,
                    ),
                    final_score=_page_score(
                        final_drafts,
                        page_idx=page_idx,
                        native_text=native_text,
                    ),
                    backfilled=page_idx in backfilled,
                    explicit_backend=explicit_backend,
                ),
                min_raw_score=min_raw_score,
            )
        )

    eligible = [decision for decision in decisions if decision.role != RouteRole.primary]
    selected = sorted(
        eligible,
        key=lambda decision: (_ROLE_PRIORITY[decision.role], decision.page_idx),
    )[:max_pages]
    return PageRoutingPlan(classifications, tuple(decisions), tuple(selected))


def page_routing_metadata(plan: PageRoutingPlan, *, mode: str) -> dict[str, Any]:
    """Сериализовать только агрегаты, без текста документа и индексов страниц."""

    type_counts = Counter(item.page_type.value for item in plan.classifications)
    role_counts = Counter(item.role.value for item in plan.decisions)
    reason_counts = Counter(item.reason for item in plan.decisions)
    eligible_count = sum(item.role != RouteRole.primary for item in plan.decisions)
    return {
        "schema_version": 1,
        "mode": mode,
        "page_count": len(plan.decisions),
        "type_counts": dict(sorted(type_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "low_confidence_type_count": sum(
            item.confidence < 0.70 for item in plan.classifications
        ),
        "eligible_page_count": eligible_count,
        "selected_page_count": len(plan.selected),
        "truncated_page_count": max(0, eligible_count - len(plan.selected)),
    }
