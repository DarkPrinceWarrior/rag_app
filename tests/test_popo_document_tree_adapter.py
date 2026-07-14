from __future__ import annotations

import copy
import uuid
from typing import Any

import pytest

from rag_app.db.models import Segment, SegmentKind
from rag_app.pipeline.document_tree import (
    DocumentTreeError,
    popo_annotations_to_document_tree,
)
from rag_app.rag.tree_chunking import document_tree_to_chunks

DOC_ID = uuid.UUID("22222222-2222-4222-8222-222222222222")


@pytest.fixture
def popo_27_blocks() -> tuple[list[dict[str, Any]], dict[str, Segment]]:
    """Регрессия smoke: все 27 annotated blocks, включая nested visual."""

    annotations: list[dict[str, Any]] = []
    mapping: dict[str, Segment] = {}
    for block_id in range(1, 28):
        source_id = f"source-{block_id}"
        block_type = "text"
        kind = SegmentKind.paragraph
        level = 99  # для non-heading должен быть полностью проигнорирован
        heading_level = None
        if block_id == 1:
            block_type, kind, level, heading_level = "title", SegmentKind.heading, -1, 1
        elif block_id == 10:
            block_type, kind, level, heading_level = "title", SegmentKind.heading, -1, 2
        elif block_id in {21, 22}:
            block_type, kind = "table", SegmentKind.table
        elif block_id in {23, 24}:
            block_type, kind = "image", SegmentKind.image
        elif block_id == 25:
            block_type = "image_caption"

        annotation = {
            "source_id": source_id,
            "id": block_id,
            "type": block_type,
            "level": level,
            "contd": -1,
            "image": -1,
            "table_merge": -1,
            # Недоверенные upstream-поля намеренно не используются адаптером.
            "content": f"upstream-{block_id}",
            "bbox": [0.0, 0.0, 1.0, 1.0],
        }
        annotations.append(annotation)
        mapping[source_id] = Segment(
            id=uuid.uuid5(DOC_ID, source_id),
            document_id=DOC_ID,
            idx=block_id - 1,
            page_idx=(block_id - 1) // 9,
            kind=kind,
            heading_level=heading_level,
            source_text=f"canonical-{block_id}-" + "x" * 30,
            translated_text=f"перевод-{block_id}-" + "я" * 30,
            meta={
                "bbox_pt": [10.0, float(block_id), 500.0, float(block_id + 10)],
                "page_size_pt": [600.0, 800.0],
            },
        )

    annotations[1]["contd"] = 3  # text 2 -> text 3
    annotations[20]["table_merge"] = 22
    annotations[21]["table_merge"] = 21
    annotations[23]["image"] = 23  # visual 24 nested under visual 23
    annotations[24]["image"] = 24  # caption 25 belongs to visual 24
    return annotations, mapping


def _build(
    annotations: list[dict[str, Any]],
    mapping: dict[str, Segment],
):
    return popo_annotations_to_document_tree(
        annotations,
        source_segments=mapping,
        document_id=DOC_ID,
        parse_revision=8,
        source_sha256="b" * 64,
        model_revision="mineru-popo-smoke-revision",
    )


def test_popo_27_block_regression_preserves_every_anchor_and_nested_visual(
    popo_27_blocks: tuple[list[dict[str, Any]], dict[str, Segment]],
) -> None:
    annotations, mapping = popo_27_blocks

    artifact = _build(annotations, mapping)

    anchors = [anchor for node in artifact.nodes for anchor in node.anchors]
    assert len(artifact.nodes) == 28  # root + 27 source blocks
    assert len(anchors) == 27
    assert {anchor.segment_id for anchor in anchors} == {
        segment.id for segment in mapping.values()
    }
    assert artifact.metrics == {"source_blocks": 27, "anchored_segments": 27}
    assert artifact.annotation_sha256 is not None

    by_source = {
        node.attrs.get("popo_source_id"): node
        for node in artifact.nodes
        if node.attrs.get("popo_source_id")
    }
    assert by_source["source-10"].parent_id == by_source["source-1"].stable_id
    assert by_source["source-24"].parent_id == by_source["source-23"].stable_id
    assert by_source["source-25"].parent_id == by_source["source-24"].stable_id
    assert by_source["source-2"].attrs["continuation_group"] == by_source["source-3"].attrs[
        "continuation_group"
    ]
    assert by_source["source-21"].attrs["table_merge_group"] == by_source[
        "source-22"
    ].attrs["table_merge_group"]
    assert {edge.kind for edge in artifact.edges} == {
        "continues",
        "table_continues",
        "attached_to",
        "caption_of",
    }

    chunks = document_tree_to_chunks(list(mapping.values()), artifact)
    chunk_segment_ids = {
        segment_id for chunk in chunks for segment_id in chunk.meta["segment_ids"]
    }
    assert chunk_segment_ids == {str(segment.id) for segment in mapping.values()}

    table_chunks = [chunk for chunk in chunks if chunk.kind == "table"]
    assert len(table_chunks) == 1
    assert table_chunks[0].meta["segment_ids"] == [
        str(mapping["source-21"].id),
        str(mapping["source-22"].id),
    ]
    assert table_chunks[0].meta["table_merge_group"] == by_source[
        "source-21"
    ].attrs["table_merge_group"]
    assert len(table_chunks[0].meta["tree_node_ids"]) == 2
    assert len(table_chunks[0].meta["bboxes"]) == 2
    assert "canonical-21" in table_chunks[0].text_en
    assert "canonical-22" in table_chunks[0].text_en

    continued_chunk = next(
        chunk
        for chunk in chunks
        if str(mapping["source-2"].id) in chunk.meta["segment_ids"]
    )
    assert continued_chunk.meta["continuation_group"] == by_source[
        "source-2"
    ].attrs["continuation_group"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda blocks: blocks[4].update(type="unsupported"), "unsupported Popo block"),
        (lambda blocks: blocks[1].update(contd=99), "unknown block"),
        (lambda blocks: blocks[1].update(source_id=blocks[0]["source_id"]), "duplicate source_id"),
        (
            lambda blocks: (blocks[1].update(contd=3), blocks[2].update(contd=2)),
            "cycle",
        ),
        (lambda blocks: blocks[22].update(image=24), "cycle"),
        (lambda blocks: blocks[20].update(table_merge=27), "between tables"),
        (lambda blocks: blocks[20].update(table_merge=21), "self-reference"),
        (lambda blocks: blocks[20].update(table_merge=99), "unknown block"),
    ],
)
def test_popo_adapter_fails_closed_on_unsupported_unknown_duplicate_cycle_or_conflict(
    popo_27_blocks: tuple[list[dict[str, Any]], dict[str, Segment]],
    mutate,
    message: str,
) -> None:
    annotations, mapping = popo_27_blocks
    broken = copy.deepcopy(annotations)
    mutate(broken)

    with pytest.raises(DocumentTreeError, match=message):
        _build(broken, mapping)


def test_popo_adapter_requires_exact_one_to_one_source_coverage(
    popo_27_blocks: tuple[list[dict[str, Any]], dict[str, Segment]],
) -> None:
    annotations, mapping = popo_27_blocks
    missing = dict(mapping)
    missing.pop("source-27")
    with pytest.raises(DocumentTreeError, match="coverage"):
        _build(annotations, missing)

    duplicate = dict(mapping)
    duplicate["source-27"] = duplicate["source-26"]
    with pytest.raises(DocumentTreeError, match="one-to-one"):
        _build(annotations, duplicate)


def test_tree_chunker_does_not_merge_non_adjacent_table_group(
    popo_27_blocks: tuple[list[dict[str, Any]], dict[str, Segment]],
) -> None:
    annotations, mapping = popo_27_blocks
    annotations[20]["table_merge"] = 27
    annotations[21].update(type="text", table_merge=-1)
    annotations[26].update(type="table", table_merge=21)
    mapping["source-22"].kind = SegmentKind.paragraph
    mapping["source-27"].kind = SegmentKind.table

    artifact = _build(annotations, mapping)
    table_chunks = [
        chunk
        for chunk in document_tree_to_chunks(list(mapping.values()), artifact)
        if chunk.kind == "table"
    ]

    assert len(table_chunks) == 2
    assert table_chunks[0].meta["table_merge_group"] == table_chunks[1].meta[
        "table_merge_group"
    ]


def test_table_merge_chain_forms_one_component_and_one_rag_chunk(
    popo_27_blocks: tuple[list[dict[str, Any]], dict[str, Segment]],
) -> None:
    annotations, mapping = popo_27_blocks
    annotations[19].update(type="table", table_merge=21)
    annotations[20]["table_merge"] = 22
    annotations[21]["table_merge"] = 21
    mapping["source-20"].kind = SegmentKind.table

    artifact = _build(annotations, mapping)
    by_source = {
        node.attrs.get("popo_source_id"): node
        for node in artifact.nodes
        if node.attrs.get("popo_source_id")
    }
    groups = {
        by_source[f"source-{block_id}"].attrs["table_merge_group"]
        for block_id in (20, 21, 22)
    }
    table_edges = [edge for edge in artifact.edges if edge.kind == "table_continues"]
    table_chunks = [
        chunk
        for chunk in document_tree_to_chunks(list(mapping.values()), artifact)
        if chunk.kind == "table"
    ]

    assert len(groups) == 1
    assert len(table_edges) == 2
    assert len(table_chunks) == 1
    assert table_chunks[0].meta["segment_ids"] == [
        str(mapping[f"source-{block_id}"].id) for block_id in (20, 21, 22)
    ]


def test_table_merge_rejects_non_adjacent_page_evidence(
    popo_27_blocks: tuple[list[dict[str, Any]], dict[str, Segment]],
) -> None:
    annotations, mapping = popo_27_blocks
    mapping["source-22"].page_idx = 6

    with pytest.raises(DocumentTreeError, match="monotonic adjacent pages"):
        _build(annotations, mapping)


def test_popo_adapter_rejects_bbox_outside_page(
    popo_27_blocks: tuple[list[dict[str, Any]], dict[str, Segment]],
) -> None:
    annotations, mapping = popo_27_blocks
    mapping["source-5"].meta = {
        **mapping["source-5"].meta,
        "bbox_pt": [-1.0, 10.0, 500.0, 20.0],
    }

    with pytest.raises(ValueError, match="invalid bounds"):
        _build(annotations, mapping)
