from __future__ import annotations

import uuid

from rag_app.db.models import Segment, SegmentKind
from rag_app.rag.chunking import segments_to_chunks


def _seg(idx: int, kind: SegmentKind, text: str, level: int | None = None, page: int = 0) -> Segment:
    return Segment(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        idx=idx,
        kind=kind,
        heading_level=level,
        source_text=text,
        translated_text=f"RU:{text}",
        page_idx=page,
        meta={},
    )


def test_chunking_by_structure() -> None:
    segs = [
        _seg(0, SegmentKind.heading, "1. Scope", level=1, page=0),
        _seg(1, SegmentKind.paragraph, "Scope paragraph one with enough text to keep. " * 5, page=0),
        _seg(2, SegmentKind.heading, "1.1 Details", level=2, page=1),
        _seg(3, SegmentKind.paragraph, "Details paragraph with sufficient length. " * 5, page=1),
        _seg(4, SegmentKind.table, "Item | Value\nPressure | 16.5", page=1),
        _seg(5, SegmentKind.heading, "2. Materials", level=1, page=2),
        _seg(6, SegmentKind.paragraph, "Materials paragraph body long enough for a chunk. " * 5, page=2),
    ]
    segs[4].meta = {"table_rows": [["Item", "Value"]], "bbox_pt": [1, 2, 3, 4]}
    chunks = segments_to_chunks(segs)

    paths = [c.heading_path for c in chunks]
    assert "1. Scope" in paths
    assert "1. Scope → 1.1 Details" in paths
    assert "2. Materials" in paths
    table = next(c for c in chunks if c.kind == "table")
    assert "таблица" in table.heading_path
    assert table.meta["bboxes"][0]["page"] == 1

    sec1 = next(c for c in chunks if c.heading_path == "1. Scope")
    assert sec1.page_start == 0
    assert "RU:" in sec1.text_ru and "RU:" not in sec1.text_en
    # глубина пути сбрасывается по уровню: «2. Materials» не вложен в «1. Scope»
    sec2 = next(c for c in chunks if c.heading_path == "2. Materials")
    assert "Scope" not in sec2.heading_path


def test_long_section_splits() -> None:
    segs = [_seg(0, SegmentKind.heading, "Big", level=1)]
    segs += [
        _seg(i + 1, SegmentKind.paragraph, f"Paragraph {i} " + "word " * 300, page=i)
        for i in range(6)
    ]
    chunks = segments_to_chunks(segs)
    assert len(chunks) > 1
    assert all(c.heading_path == "Big" for c in chunks)
    assert len({c.meta["section_id"] for c in chunks}) == 1
    assert [c.meta["ordinal_in_section"] for c in chunks] == list(range(len(chunks)))
    assert [c.meta["source_ordinal"] for c in chunks] == sorted(
        c.meta["source_ordinal"] for c in chunks
    )


def test_hierarchy_metadata_keeps_table_in_its_section_source_order() -> None:
    document_id = uuid.uuid4()
    heading = _seg(0, SegmentKind.heading, "Section", level=1)
    paragraph = _seg(1, SegmentKind.paragraph, "Body text " * 30)
    table = _seg(2, SegmentKind.table, "A | B\n1 | 2")
    trailing = _seg(3, SegmentKind.paragraph, "Trailing text " * 30)
    for segment in (heading, paragraph, table, trailing):
        segment.document_id = document_id
    table.meta = {"table_merge_group": "logical-table-1", "continuation_index": 0}

    chunks = segments_to_chunks([heading, paragraph, table, trailing])

    assert {chunk.meta["section_id"] for chunk in chunks} == {str(heading.id)}
    assert sorted(chunk.meta["ordinal_in_section"] for chunk in chunks) == list(
        range(len(chunks))
    )
    table_chunk = next(chunk for chunk in chunks if chunk.kind == "table")
    assert table_chunk.meta["logical_table_id"] == "logical-table-1"
    assert table_chunk.meta["continuation_index"] == 0


def test_hierarchy_metadata_assigns_stable_root_to_headingless_document() -> None:
    document_id = uuid.uuid4()
    segments = [
        _seg(index, SegmentKind.paragraph, f"Root paragraph {index} " + "word " * 300)
        for index in range(3)
    ]
    for segment in segments:
        segment.document_id = document_id

    first = segments_to_chunks(segments)
    second = segments_to_chunks(segments)

    assert first
    assert {chunk.meta["section_id"] for chunk in first} == {
        str(uuid.uuid5(uuid.NAMESPACE_URL, f"rag-section-root:{document_id}"))
    }
    assert [chunk.meta["section_id"] for chunk in first] == [
        chunk.meta["section_id"] for chunk in second
    ]
