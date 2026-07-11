from __future__ import annotations

from rag_app.db.models import SegmentKind
from rag_app.pipeline.parse_quality import evaluate_parse, quality_metadata, should_select_fallback
from rag_app.pipeline.segments import SegmentDraft


def _draft(
    idx: int,
    page_idx: int,
    text: str,
    *,
    kind: SegmentKind = SegmentKind.paragraph,
    meta: dict[str, object] | None = None,
) -> SegmentDraft:
    return SegmentDraft(idx=idx, kind=kind, source_text=text, page_idx=page_idx, meta=meta or {})


def test_healthy_parse_is_acceptable_and_deterministic() -> None:
    drafts = [
        _draft(0, 0, "This specification defines the required pressure vessel materials."),
        _draft(1, 1, "The maximum allowable working pressure is 16.5 MPa."),
    ]

    first = evaluate_parse(drafts, n_pages=2)
    second = evaluate_parse(drafts, n_pages=2)

    assert first == second
    assert first.acceptable
    assert first.score == 1.0
    assert first.page_coverage == 1.0
    assert first.reasons == ()


def test_missing_page_and_low_native_text_coverage_are_reported() -> None:
    report = evaluate_parse(
        [_draft(0, 0, "Short result")],
        n_pages=2,
        native_text_by_page={0: "A" * 200, 1: "B" * 200},
    )

    assert not report.acceptable
    assert report.page_coverage == 0.5
    assert report.native_text_coverage is not None
    assert report.native_text_coverage < 0.5
    assert "missing_pages" in report.reasons
    assert "low_native_text_coverage" in report.reasons


def test_duplicate_content_reduces_score() -> None:
    repeated = "Repeated OCR block that should not fill every page."
    unique = evaluate_parse(
        [_draft(0, 0, repeated), _draft(1, 1, "A different complete paragraph for page two.")],
        n_pages=2,
    )
    duplicate = evaluate_parse(
        [_draft(0, 0, repeated), _draft(1, 1, repeated), _draft(2, 1, repeated)],
        n_pages=2,
    )

    assert duplicate.duplicate_ratio > 0.25
    assert duplicate.score < unique.score
    assert "duplicate_content" in duplicate.reasons


def test_invalid_table_and_bbox_are_reported() -> None:
    report = evaluate_parse(
        [
            _draft(
                0,
                0,
                "Broken table",
                kind=SegmentKind.table,
                meta={"table_rows": [["A", "B"], ["C"]], "bbox": [10, 10, 5, 20]},
            )
        ],
        n_pages=1,
    )

    assert report.integrity_ratio == 0.0
    assert "invalid_structures" in report.reasons


def test_structured_image_page_counts_as_content_without_fake_missing_page() -> None:
    report = evaluate_parse(
        [_draft(0, 0, "", kind=SegmentKind.image, meta={"bbox": [0, 0, 100, 100]})],
        n_pages=1,
    )

    assert report.page_coverage == 1.0
    assert report.acceptable
    assert "missing_pages" not in report.reasons
    assert "pages_without_content" not in report.reasons


def test_empty_output_is_never_acceptable() -> None:
    report = evaluate_parse([], n_pages=3)

    assert report.score == 0.0
    assert not report.acceptable
    assert "empty_output" in report.reasons


def test_fallback_requires_weak_primary_quality_and_margin() -> None:
    weak = evaluate_parse([_draft(0, 0, "short")], n_pages=2)
    healthy = evaluate_parse(
        [
            _draft(0, 0, "Complete content on the first document page."),
            _draft(1, 1, "Complete content on the second document page."),
        ],
        n_pages=2,
    )

    assert should_select_fallback(None, healthy)
    assert should_select_fallback(weak, healthy)
    assert not should_select_fallback(healthy, weak)
    assert not should_select_fallback(healthy, healthy)
    assert not should_select_fallback(weak, None)


def test_quality_metadata_contains_only_aggregate_signals() -> None:
    report = evaluate_parse(
        [
            _draft(0, 0, "Confidential source text"),
            _draft(1, 1, "", kind=SegmentKind.table, meta={"table_rows": [["A"]]}),
        ],
        n_pages=2,
    )

    metadata = quality_metadata(report, backend="mineru")

    assert metadata == {
        "schema_version": 1,
        "backend": "mineru",
        "score": report.score,
        "acceptable": report.acceptable,
        "page_count": 2,
        "content_page_count": 2,
        "structured_page_count": 1,
        "page_coverage": report.page_coverage,
        "native_text_coverage": None,
        "duplicate_ratio": report.duplicate_ratio,
        "integrity_ratio": report.integrity_ratio,
        "reasons": list(report.reasons),
    }
    assert "Confidential source text" not in repr(metadata)


def test_quality_metadata_separates_raw_parser_from_backfilled_result() -> None:
    raw_report = evaluate_parse([], n_pages=2)
    final_report = evaluate_parse(
        [
            _draft(0, 0, "Recovered first page"),
            _draft(1, 1, "Recovered second page"),
        ],
        n_pages=2,
    )

    metadata = quality_metadata(
        final_report,
        backend="mineru",
        raw_report=raw_report,
        backfilled_pages=[0, 1, 1],
    )

    assert metadata["schema_version"] == 2
    assert metadata["score"] == final_report.score
    assert metadata["raw_parser"]["score"] == 0.0
    assert metadata["raw_parser"]["reasons"] == ["empty_output", "missing_pages"]
    assert metadata["backfilled_page_count"] == 2
    assert metadata["backfilled_page_ratio"] == 1.0
