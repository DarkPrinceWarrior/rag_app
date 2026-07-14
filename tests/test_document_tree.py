from __future__ import annotations

import json
import uuid

import pytest
from pydantic import ValidationError

from rag_app.config import Settings
from rag_app.db.models import Segment, SegmentKind
from rag_app.pipeline.document_tree import (
    DocumentTreeArtifact,
    canonical_document_tree_bytes,
    document_tree_sha256,
    segments_to_document_tree,
)
from rag_app.rag.chunking import segments_to_chunks
from rag_app.rag.tree_chunking import (
    document_tree_to_chunks,
    select_document_chunks,
)

DOC_ID = uuid.UUID("11111111-1111-4111-8111-111111111111")
SOURCE_SHA = "a" * 64


def _segment(
    idx: int,
    kind: SegmentKind,
    text: str,
    *,
    page: int = 0,
    level: int | None = None,
    bbox: list[float] | None = None,
) -> Segment:
    meta = {}
    if bbox is not None:
        meta = {"bbox_pt": bbox, "page_size_pt": [600.0, 800.0]}
    return Segment(
        id=uuid.uuid5(DOC_ID, f"segment:{idx}"),
        document_id=DOC_ID,
        idx=idx,
        page_idx=page,
        kind=kind,
        heading_level=level,
        source_text=text,
        translated_text=f"RU {text}",
        meta=meta,
    )


def _segments() -> list[Segment]:
    return [
        _segment(0, SegmentKind.heading, "1 Scope", level=1, bbox=[20, 30, 220, 55]),
        _segment(1, SegmentKind.paragraph, "A" * 160, bbox=[20, 70, 560, 150]),
        _segment(2, SegmentKind.paragraph, "B" * 160, page=1, bbox=[20, 40, 560, 120]),
        _segment(3, SegmentKind.table, "P | T\n1 | 2", page=1, bbox=[20, 150, 560, 300]),
    ]


def _artifact(segments: list[Segment] | None = None) -> DocumentTreeArtifact:
    return segments_to_document_tree(
        segments or _segments(),
        document_id=DOC_ID,
        parse_revision=7,
        source_sha256=SOURCE_SHA,
    )


def test_document_tree_default_mode_is_off() -> None:
    assert Settings(_env_file=None).document_tree_mode == "off"


def test_linear_artifact_is_deterministic_versioned_and_canonical() -> None:
    first = _artifact()
    second = _artifact()

    assert first == second
    assert first.schema_version == 1
    assert first.artifact_type == "document_tree"
    assert first.nodes[0].kind == "document"
    assert document_tree_sha256(first) == document_tree_sha256(second)
    assert json.loads(canonical_document_tree_bytes(first))["schema_version"] == 1

    heading = first.nodes[1]
    assert first.nodes[2].parent_id == heading.stable_id
    assert first.nodes[3].parent_id == heading.stable_id


def test_leaf_stable_ids_survive_new_segment_uuids() -> None:
    original = _segments()
    reparsed = _segments()
    for position, segment in enumerate(reparsed):
        segment.id = uuid.uuid5(DOC_ID, f"reparsed:{position}")

    first = _artifact(original)
    second = _artifact(reparsed)

    assert [node.stable_id for node in first.nodes] == [node.stable_id for node in second.nodes]
    assert first.input_sha256 != second.input_sha256
    assert first.tree_id != second.tree_id


def test_artifact_rejects_orphans_and_noncontiguous_ordinals() -> None:
    artifact = _artifact()
    data = artifact.model_dump(mode="json")
    data["nodes"][1]["parent_id"] = str(uuid.uuid4())
    with pytest.raises(ValidationError, match="orphan"):
        DocumentTreeArtifact.model_validate(data)

    data = artifact.model_dump(mode="json")
    data["nodes"][2]["ordinal"] = 9
    with pytest.raises(ValidationError, match="contiguous"):
        DocumentTreeArtifact.model_validate(data)


def test_tree_chunks_preserve_segment_ids_pages_and_geometry() -> None:
    segments = _segments()
    artifact = _artifact(segments)

    chunks = document_tree_to_chunks(segments, artifact)

    assert chunks
    all_ids = [segment_id for chunk in chunks for segment_id in chunk.meta["segment_ids"]]
    assert set(all_ids) == {str(segment.id) for segment in segments}
    assert all(chunk.meta["tree_id"] == str(artifact.tree_id) for chunk in chunks)
    assert all(chunk.meta["tree_node_ids"] for chunk in chunks)
    geometry = [item for chunk in chunks for item in chunk.meta["bboxes"]]
    assert {item["page"] for item in geometry} == {0, 1}
    assert all(item["page_size"] == [600.0, 800.0] for item in geometry)


def test_off_and_shadow_never_change_selected_chunks() -> None:
    segments = _segments()
    artifact = _artifact(segments)
    baseline = segments_to_chunks(segments)

    off = select_document_chunks(segments, mode="off", artifact=artifact)
    shadow = select_document_chunks(segments, mode="shadow", artifact=artifact)

    assert off.selected == baseline
    assert off.shadow is None
    assert not off.used_tree
    assert shadow.selected == baseline
    assert shadow.shadow is not None
    assert not shadow.used_tree


def test_active_uses_tree_order_and_missing_coverage_falls_back() -> None:
    segments = _segments()
    artifact = _artifact(segments)
    data = artifact.model_dump(mode="json")
    # Два абзаца — соседи одного heading. Меняем только их tree-order.
    data["nodes"][2]["ordinal"], data["nodes"][3]["ordinal"] = (
        data["nodes"][3]["ordinal"],
        data["nodes"][2]["ordinal"],
    )
    reordered = DocumentTreeArtifact.model_validate(data)

    active = select_document_chunks(segments, mode="active", artifact=reordered)
    assert active.used_tree
    section = next(chunk for chunk in active.selected if chunk.kind == "section")
    assert "B" * 40 in section.text_en
    assert section.text_en.index("B" * 40) < section.text_en.index("A" * 40)

    missing_data = artifact.model_dump(mode="json")
    missing_data["nodes"] = missing_data["nodes"][:-1]
    missing = DocumentTreeArtifact.model_validate(missing_data)
    fallback = select_document_chunks(segments, mode="active", artifact=missing)
    assert not fallback.used_tree
    assert fallback.selected == fallback.baseline
    assert fallback.fallback_reason == "DocumentTreeChunkError"

    segments[1].source_text = "changed source text"
    stale = select_document_chunks(segments, mode="active", artifact=artifact)
    assert not stale.used_tree
    assert stale.fallback_reason == "DocumentTreeChunkError"
