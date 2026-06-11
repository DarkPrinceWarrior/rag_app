from __future__ import annotations

from rag_app.db.models import SegmentKind
from rag_app.pipeline.segments import content_list_to_segments, parse_table_html


def test_table_html_parsing() -> None:
    html = (
        "<table><tr><th>Item</th><th>Value</th></tr>"
        "<tr><td>Pressure</td><td>16.5 MPa</td></tr></table>"
    )
    assert parse_table_html(html) == [["Item", "Value"], ["Pressure", "16.5 MPa"]]


def test_content_list_mapping() -> None:
    items = [
        {"type": "text", "text": "Scope", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": "This section defines scope.", "page_idx": 0},
        {"type": "text", "text": "   ", "page_idx": 0},  # пустое — выбрасывается
        {
            "type": "table",
            "table_body": "<table><tr><td>A</td><td>B</td></tr></table>",
            "table_caption": ["Table 1. Parameters"],
            "page_idx": 1,
        },
        {"type": "equation", "text": "\\[E=mc^2\\]", "page_idx": 1},
        {"type": "image", "image_caption": ["Figure 1. Flow"], "img_path": "x.jpg", "page_idx": 2},
    ]
    segs = content_list_to_segments(items)
    kinds = [s.kind for s in segs]
    assert kinds == [
        SegmentKind.heading,
        SegmentKind.paragraph,
        SegmentKind.table,
        SegmentKind.equation,
        SegmentKind.image,
    ]
    assert segs[0].heading_level == 1
    assert segs[2].meta["table_rows"] == [["A", "B"]]
    assert segs[2].meta["caption"] == "Table 1. Parameters"
    assert [s.idx for s in segs] == [0, 1, 2, 3, 4]


def test_needs_translation() -> None:
    from rag_app.llm.client import needs_translation

    assert needs_translation("Pressure vessel")
    assert not needs_translation("16.5")
    assert not needs_translation("")
    assert not needs_translation("Давление")
