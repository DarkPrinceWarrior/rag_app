from __future__ import annotations

import pytest

from rag_app.db.models import SegmentKind
from rag_app.pipeline.page_classifier import classify_page_types
from rag_app.pipeline.page_routing import PageType
from rag_app.pipeline.parse_quality import evaluate_parse
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


def test_classifies_text_and_table_pages_deterministically() -> None:
    drafts = [
        _draft(0, 0, "This specification defines the vessel design pressure."),
        _draft(
            1,
            1,
            "Item | Material\nShell | ASTM A516",
            kind=SegmentKind.table,
            meta={"table_rows": [["Item", "Material"], ["Shell", "ASTM A516"]]},
        ),
    ]
    quality = evaluate_parse(drafts, n_pages=2)
    native = {0: "This specification defines the vessel design pressure.", 1: ""}

    first = classify_page_types(drafts, n_pages=2, quality=quality, native_text_by_page=native)
    second = classify_page_types(drafts, n_pages=2, quality=quality, native_text_by_page=native)

    assert first == second
    assert [item.page_type for item in first] == [PageType.text, PageType.table]
    assert first[1].reasons == ("table_segments",)


def test_form_fields_override_table_layout() -> None:
    drafts = [
        _draft(
            0,
            0,
            "Application form\nName: ______\nCompany: ______\nDate: ______\nApproved: [ ]",
            kind=SegmentKind.table,
        )
    ]

    result = classify_page_types(drafts, n_pages=1)

    assert result[0].page_type == PageType.form
    assert "label_value_fields" in result[0].reasons
    assert "fillable_marks" in result[0].reasons


def test_image_caption_distinguishes_chart_and_diagram() -> None:
    drafts = [
        _draft(
            0,
            0,
            "Figure 7. Pressure histogram by test cycle",
            kind=SegmentKind.image,
        ),
        _draft(
            1,
            1,
            "",
            kind=SegmentKind.image,
            meta={"caption": "Process flow schematic for the separator unit"},
        ),
    ]

    result = classify_page_types(drafts, n_pages=2)

    assert result[0].page_type == PageType.chart
    assert result[1].page_type == PageType.diagram


def test_native_text_separates_digital_text_from_ocr_scan() -> None:
    drafts = [
        _draft(0, 0, "OCR output from a scanned page with enough text to classify the source."),
    ]
    native = {
        0: "",
        1: "A native PDF text layer remains useful even when the parser missed this page.",
    }
    quality = evaluate_parse(drafts, n_pages=2, native_text_by_page=native)

    result = classify_page_types(drafts, n_pages=2, quality=quality, native_text_by_page=native)

    assert result[0].page_type == PageType.scan
    assert result[1].page_type == PageType.text
    assert result[1].reasons == ("native_text_only",)


def test_multiple_structured_kinds_are_mixed() -> None:
    drafts = [
        _draft(0, 0, "Operating envelope", kind=SegmentKind.paragraph),
        _draft(1, 0, "Pressure | Limit\nDesign | 16.5 MPa", kind=SegmentKind.table),
        _draft(2, 0, "Figure without a semantic caption", kind=SegmentKind.image),
    ]

    result = classify_page_types(drafts, n_pages=1)

    assert result[0].page_type == PageType.mixed
    assert result[0].reasons == ("multiple_structured_kinds", "text_content")


def test_image_without_subtype_cue_is_unknown() -> None:
    drafts = [_draft(0, 0, "", kind=SegmentKind.image)]
    quality = evaluate_parse(drafts, n_pages=1)

    result = classify_page_types(drafts, n_pages=1, quality=quality)

    assert result[0].page_type == PageType.unknown
    assert result[0].confidence == 0.0
    assert result[0].reasons == ("sparse_page_quality", "image_segments", "no_visual_subtype_cue")


def test_validates_page_count_and_ignores_out_of_range_drafts() -> None:
    quality = evaluate_parse([], n_pages=1)

    with pytest.raises(ValueError, match="non-negative"):
        classify_page_types([], n_pages=-1)
    with pytest.raises(ValueError, match="quality page count"):
        classify_page_types([], n_pages=2, quality=quality)

    result = classify_page_types([_draft(0, 9, "out of range")], n_pages=1)
    assert result[0].page_type == PageType.unknown
    assert result[0].reasons == ("insufficient_evidence",)
