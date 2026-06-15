from __future__ import annotations

from rag_app.db.models import SegmentKind
from rag_app.pipeline.segments import content_list_to_segments, parse_table, parse_table_html


def test_table_html_parsing() -> None:
    html = (
        "<table><tr><th>Item</th><th>Value</th></tr>"
        "<tr><td>Pressure</td><td>16.5 MPa</td></tr></table>"
    )
    assert parse_table_html(html) == [["Item", "Value"], ["Pressure", "16.5 MPa"]]


def test_table_colspan_rowspan_grid() -> None:
    # шапка с объединёнными ячейками (как у MinerU): rowspan на «Pump»,
    # colspan на «500 rpm» → ровная сетка, колонки не съезжают.
    html = (
        "<table>"
        "<tr><td rowspan=2>Pump</td><td colspan=2>500 rpm</td></tr>"
        "<tr><td>Flow</td><td>Head</td></tr>"
        "<tr><td>A</td><td>1</td><td>2</td></tr>"
        "</table>"
    )
    grid = parse_table_html(html)
    assert grid == [["Pump", "500 rpm", ""], ["", "Flow", "Head"], ["A", "1", "2"]]
    assert len({len(r) for r in grid}) == 1  # все строки одной ширины
    # сырые ячейки со спанами — для merged-рендера в UI
    cells, _ = parse_table(html)
    assert cells[0] == [
        {"text": "Pump", "colspan": 1, "rowspan": 2},
        {"text": "500 rpm", "colspan": 2, "rowspan": 1},
    ]
    assert cells[1] == [
        {"text": "Flow", "colspan": 1, "rowspan": 1},
        {"text": "Head", "colspan": 1, "rowspan": 1},
    ]


def test_table_segment_carries_cells() -> None:
    items = [
        {
            "type": "table",
            "table_body": "<table><tr><td colspan=2>H</td></tr><tr><td>a</td><td>b</td></tr></table>",
            "page_idx": 0,
        }
    ]
    seg = content_list_to_segments(items)[0]
    assert seg.kind == SegmentKind.table
    assert seg.meta["table_cells"][0][0] == {"text": "H", "colspan": 2, "rowspan": 1}
    # превью построчно — одна строка на строку таблицы
    assert seg.source_text == "H\na | b"


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


def test_unique_header_promoted_to_leading_title() -> None:
    # воспроизводит реальный баг: MinerU кладёт титульную шапку как type=header
    # в КОНЕЦ content_list; она должна стать первым сегментом-заголовком.
    items = [
        {"type": "text", "text": "1. Scope", "text_level": 2, "page_idx": 0},
        {"type": "text", "text": "Body paragraph.", "page_idx": 0},
        {"type": "header", "text": "TECHNICAL SPECIFICATION", "page_idx": 0},
    ]
    segs = content_list_to_segments(items)
    assert segs[0].kind == SegmentKind.heading
    assert segs[0].source_text == "TECHNICAL SPECIFICATION"
    assert segs[0].heading_level == 1
    assert [s.source_text for s in segs] == ["TECHNICAL SPECIFICATION", "1. Scope", "Body paragraph."]
    assert [s.idx for s in segs] == [0, 1, 2]


def test_running_header_footer_dropped() -> None:
    # бегущий колонтитул (повтор на ≥2 страницах) и любой footer — выбрасываются
    items = [
        {"type": "header", "text": "Acme Corp — Confidential", "page_idx": 0},
        {"type": "text", "text": "Page one body.", "page_idx": 0},
        {"type": "header", "text": "Acme Corp — Confidential", "page_idx": 1},
        {"type": "text", "text": "Page two body.", "page_idx": 1},
        {"type": "footer", "text": "Page 2", "page_idx": 1},
    ]
    segs = content_list_to_segments(items)
    assert [s.source_text for s in segs] == ["Page one body.", "Page two body."]


def test_needs_translation() -> None:
    from rag_app.llm.client import needs_translation

    assert needs_translation("Pressure vessel")
    assert not needs_translation("16.5")
    assert not needs_translation("")
    assert not needs_translation("Давление")
