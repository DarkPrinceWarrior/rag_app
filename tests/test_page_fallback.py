from __future__ import annotations

from rag_app.db.models import SegmentKind
from rag_app.pipeline.page_fallback import (
    page_fallback_allowed,
    page_fallback_error_metadata,
    page_fallback_metadata,
    select_page_fallbacks,
)
from rag_app.pipeline.page_routing import PageRouteDecision, RouteRole
from rag_app.pipeline.page_routing_shadow import PageRoutingPlan
from rag_app.pipeline.segments import SegmentDraft


def _draft(page: int, text: str, kind: SegmentKind = SegmentKind.paragraph) -> SegmentDraft:
    meta = {"table_cells": [[{"text": text}]]} if kind == SegmentKind.table else {}
    return SegmentDraft(0, kind, text, page, meta=meta)


def _routing(*decisions: PageRouteDecision) -> PageRoutingPlan:
    return PageRoutingPlan((), decisions, decisions)


def test_accepts_missing_table_structure_with_full_page_geometry() -> None:
    routing = _routing(
        PageRouteDecision(1, RouteRole.parser_fallback, "table_requires_structure_check")
    )
    plan = select_page_fallbacks(
        [_draft(1, "plain text")],
        [_draft(1, "A", SegmentKind.table)],
        routing,
        n_pages=2,
        page_sizes={1: (600.0, 800.0)},
        parser_revision=3,
    )

    assert len(plan.candidates) == 1
    candidate = plan.candidates[0]
    assert candidate.page_idx == 1
    assert candidate.parser_revision == 3
    assert candidate.drafts[0].meta["bbox_pt"] == [0.0, 0.0, 600.0, 800.0]
    assert candidate.drafts[0].meta["geometry_precision"] == "full_page"


def test_rejects_table_when_primary_already_has_structure() -> None:
    routing = _routing(
        PageRouteDecision(0, RouteRole.parser_fallback, "table_requires_structure_check")
    )
    plan = select_page_fallbacks(
        [_draft(0, "A", SegmentKind.table)],
        [_draft(0, "B", SegmentKind.table)],
        routing,
        n_pages=1,
        page_sizes={0: (600.0, 800.0)},
        parser_revision=1,
    )

    assert plan.candidates == ()
    assert plan.decisions[0].reason == "table_not_improved"


def test_low_quality_requires_minimum_score_and_margin() -> None:
    routing = _routing(PageRouteDecision(0, RouteRole.parser_fallback, "low_raw_score"))
    primary = [_draft(0, "")]
    fallback = [_draft(0, "useful extracted text")]

    accepted = select_page_fallbacks(
        primary,
        fallback,
        routing,
        n_pages=1,
        page_sizes={0: (600.0, 800.0)},
        parser_revision=1,
    )
    rejected = select_page_fallbacks(
        fallback,
        fallback,
        routing,
        n_pages=1,
        page_sizes={0: (600.0, 800.0)},
        parser_revision=1,
    )

    assert len(accepted.candidates) == 1
    assert rejected.candidates == ()
    assert rejected.decisions[0].reason == "quality_not_improved"


def test_ignores_structured_sidecar_roles_and_reports_only_aggregates() -> None:
    routing = _routing(
        PageRouteDecision(0, RouteRole.structured_extraction, "form_requires_schema_extraction"),
        PageRouteDecision(1, RouteRole.parser_fallback, "low_raw_score"),
    )
    plan = select_page_fallbacks(
        [_draft(1, "")],
        [],
        routing,
        n_pages=2,
        page_sizes={1: (600.0, 800.0)},
        parser_revision=1,
    )

    assert len(plan.decisions) == 1
    assert page_fallback_metadata(plan, backend="paddle_vl") == {
        "schema_version": 1,
        "status": "ok",
        "backend": "paddle_vl",
        "attempted_page_count": 1,
        "accepted_page_count": 0,
        "reason_counts": {"candidate_missing": 1},
    }


def test_rejects_out_of_range_fallback_page() -> None:
    routing = _routing(PageRouteDecision(0, RouteRole.parser_fallback, "low_raw_score"))

    try:
        select_page_fallbacks(
            [],
            [_draft(2, "bad")],
            routing,
            n_pages=1,
            page_sizes={0: (600.0, 800.0)},
            parser_revision=1,
        )
    except ValueError as exc:
        assert "page_idx out of range" in str(exc)
    else:
        raise AssertionError("out-of-range fallback page must fail closed")


def test_error_metadata_does_not_include_exception_message() -> None:
    assert page_fallback_error_metadata(
        backend="paddle_vl",
        attempted_page_count=2,
        error_type="RuntimeError",
    ) == {
        "schema_version": 1,
        "status": "error",
        "backend": "paddle_vl",
        "attempted_page_count": 2,
        "accepted_page_count": 0,
        "reason_counts": {"RuntimeError": 1},
    }


def test_model_call_requires_canary_allowlist_flag_and_mineru() -> None:
    common = {
        "enabled": True,
        "router_mode": "canary",
        "router_allowed": True,
        "primary_backend": "mineru",
        "attempted_page_count": 1,
    }
    assert page_fallback_allowed(**common)
    for key, value in (
        ("enabled", False),
        ("router_mode", "shadow"),
        ("router_allowed", False),
        ("primary_backend", "paddle_vl"),
        ("attempted_page_count", 0),
    ):
        candidate = dict(common)
        candidate[key] = value
        assert not page_fallback_allowed(**candidate)
