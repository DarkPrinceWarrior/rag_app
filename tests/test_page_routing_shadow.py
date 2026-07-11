from __future__ import annotations

import pytest

from rag_app.db.models import SegmentKind
from rag_app.pipeline.page_routing import RouteRole
from rag_app.pipeline.page_routing_shadow import (
    build_page_routing_plan,
    page_router_allowed,
    page_routing_metadata,
)
from rag_app.pipeline.parse_quality import evaluate_parse
from rag_app.pipeline.segments import SegmentDraft


def _draft(
    idx: int,
    page_idx: int,
    text: str,
    *,
    kind: SegmentKind = SegmentKind.paragraph,
    meta: dict[str, object] | None = None,
) -> SegmentDraft:
    return SegmentDraft(idx, kind, text, page_idx, meta=meta or {})


def test_backfilled_empty_raw_page_routes_to_fallback() -> None:
    final = [_draft(0, 0, "A" * 120)]
    quality = evaluate_parse(final, n_pages=1, native_text_by_page={0: "A" * 120})

    plan = build_page_routing_plan(
        final,
        [],
        n_pages=1,
        final_quality=quality,
        native_text_by_page={0: "A" * 120},
        backfilled_pages=[0],
    )

    assert plan.decisions[0].role == RouteRole.parser_fallback
    assert plan.decisions[0].reason == "low_raw_score"
    assert plan.selected == plan.decisions


def test_explicit_backend_disables_all_automatic_routes() -> None:
    final = [_draft(0, 0, "Invoice\nName: A\nDate: B\nCode: C\nTotal: D")]
    quality = evaluate_parse(final, n_pages=1)

    plan = build_page_routing_plan(
        final,
        final,
        n_pages=1,
        final_quality=quality,
        explicit_backend=True,
    )

    assert plan.decisions[0].role == RouteRole.primary
    assert plan.selected == ()


def test_specialized_roles_precede_parser_fallback_under_page_cap() -> None:
    final = [
        _draft(0, 0, "table", kind=SegmentKind.table, meta={"table_rows": [["A"]]}),
        _draft(1, 1, "Application form\nName: A\nDate: B\nCode: C\nTotal: D"),
    ]
    quality = evaluate_parse(final, n_pages=2)

    plan = build_page_routing_plan(
        final,
        final,
        n_pages=2,
        final_quality=quality,
        max_pages=1,
    )

    assert [decision.role for decision in plan.decisions] == [
        RouteRole.parser_fallback,
        RouteRole.structured_extraction,
    ]
    assert plan.selected[0].role == RouteRole.structured_extraction


def test_metadata_contains_only_aggregates() -> None:
    final = [_draft(0, 0, "A" * 80)]
    native = {0: "A" * 80}
    quality = evaluate_parse(final, n_pages=1, native_text_by_page=native)
    plan = build_page_routing_plan(
        final,
        [],
        n_pages=1,
        final_quality=quality,
        native_text_by_page=native,
    )

    metadata = page_routing_metadata(plan, mode="shadow")

    assert metadata["eligible_page_count"] == 1
    assert metadata["selected_page_count"] == 1
    assert metadata["type_counts"] == {"text": 1}
    serialized = repr(metadata)
    assert "page_idx" not in serialized
    assert "A" * 10 not in serialized


def test_plan_rejects_invalid_page_limit() -> None:
    quality = evaluate_parse([], n_pages=0)

    with pytest.raises(ValueError, match="max_pages"):
        build_page_routing_plan(
            [],
            [],
            n_pages=0,
            final_quality=quality,
            max_pages=-1,
        )


def test_router_mode_and_canary_allowlist() -> None:
    assert not page_router_allowed("off", owner_sub="a", allowed_owner_subs=["a"])
    assert page_router_allowed("shadow", owner_sub=None, allowed_owner_subs=[])
    assert page_router_allowed("canary", owner_sub="a", allowed_owner_subs=["a"])
    assert not page_router_allowed("canary", owner_sub="b", allowed_owner_subs=["a"])
    assert not page_router_allowed("canary", owner_sub=None, allowed_owner_subs=[])

    with pytest.raises(ValueError, match="unknown"):
        page_router_allowed("enabled", owner_sub="a", allowed_owner_subs=["a"])
