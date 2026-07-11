from __future__ import annotations

import pytest

from rag_app.db.models import SegmentKind
from rag_app.pipeline.page_routing import (
    PageCandidate,
    PageRouteDecision,
    PageRoutingSignals,
    PageType,
    RouteRole,
    merge_page_replacements,
    propose_page_route,
)
from rag_app.pipeline.segments import SegmentDraft


def _draft(
    idx: int,
    page_idx: int | None,
    text: str,
    *,
    kind: SegmentKind = SegmentKind.paragraph,
    meta: dict[str, object] | None = None,
) -> SegmentDraft:
    return SegmentDraft(idx, kind, text, page_idx, meta=meta or {})


def _geometry() -> dict[str, object]:
    return {"bbox_pt": [10, 20, 190, 280], "page_size_pt": [200, 300]}


@pytest.mark.parametrize(
    ("page_type", "role"),
    [
        (PageType.form, RouteRole.structured_extraction),
        (PageType.chart, RouteRole.chart_extraction),
        (PageType.diagram, RouteRole.diagram_extraction),
        (PageType.table, RouteRole.parser_fallback),
    ],
)
def test_page_type_selects_specialized_role(page_type: PageType, role: RouteRole) -> None:
    decision = propose_page_route(PageRoutingSignals(2, page_type, 1.0, 1.0))

    assert decision.role == role


def test_explicit_backend_is_never_routed() -> None:
    signals = PageRoutingSignals(
        page_idx=1,
        page_type=PageType.form,
        raw_score=0.0,
        final_score=1.0,
        backfilled=True,
        explicit_backend=True,
    )

    assert propose_page_route(signals) == PageRouteDecision(
        page_idx=1,
        role=RouteRole.primary,
        reason="explicit_backend",
    )


def test_low_raw_score_and_backfill_trigger_parser_evaluation() -> None:
    low = PageRoutingSignals(0, PageType.text, 0.4, 1.0)
    backfilled = PageRoutingSignals(1, PageType.text, 0.9, 1.0, backfilled=True)

    assert propose_page_route(low).reason == "low_raw_score"
    assert propose_page_route(backfilled).reason == "backfilled_primary"


def test_merge_replaces_whole_page_and_remaps_local_page_zero() -> None:
    primary = [
        _draft(7, 0, "primary page zero"),
        _draft(8, 1, "old table", kind=SegmentKind.table, meta={"table_rows": [["old"]]}),
        _draft(9, 2, "primary page two"),
        _draft(10, None, "document tail"),
    ]
    table_meta = {
        **_geometry(),
        "table_rows": [["A", "B"]],
        "table_cells": [{"row": 0, "col": 0, "rowspan": 1, "colspan": 2}],
    }
    candidate = PageCandidate(
        page_idx=1,
        backend="candidate",
        parser_revision=4,
        drafts=(_draft(0, 0, "new table", kind=SegmentKind.table, meta=table_meta),),
    )

    merged = merge_page_replacements(primary, [candidate], n_pages=3)

    assert [draft.idx for draft in merged] == [0, 1, 2, 3]
    assert [draft.source_text for draft in merged] == [
        "primary page zero",
        "new table",
        "primary page two",
        "document tail",
    ]
    assert [draft.page_idx for draft in merged] == [0, 1, 2, None]
    assert merged[1].meta["parser_backend"] == "candidate"
    assert merged[1].meta["parser_revision"] == 4
    assert merged[1].meta["table_rows"] == [["A", "B"]]
    assert primary[1].source_text == "old table"


@pytest.mark.parametrize(
    "candidate",
    [
        PageCandidate(3, "candidate", 1, (_draft(0, 0, "bad", meta=_geometry()),)),
        PageCandidate(1, "candidate", 1, ()),
        PageCandidate(1, "candidate", 1, (_draft(0, None, "bad", meta=_geometry()),)),
        PageCandidate(1, "candidate", 1, (_draft(0, 0, "no bbox"),)),
        PageCandidate(
            1,
            "candidate",
            1,
            (_draft(0, 0, "outside", meta={"bbox_pt": [0, 0, 220, 300], "page_size_pt": [200, 300]}),),
        ),
    ],
)
def test_merge_rejects_invalid_candidate(candidate: PageCandidate) -> None:
    with pytest.raises(ValueError):
        merge_page_replacements([_draft(0, 0, "primary")], [candidate], n_pages=3)


def test_merge_rejects_duplicate_page_candidates() -> None:
    first = PageCandidate(0, "a", 1, (_draft(0, 0, "a", meta=_geometry()),))
    second = PageCandidate(0, "b", 1, (_draft(0, 0, "b", meta=_geometry()),))

    with pytest.raises(ValueError, match="duplicate candidate"):
        merge_page_replacements([], [first, second], n_pages=1)
