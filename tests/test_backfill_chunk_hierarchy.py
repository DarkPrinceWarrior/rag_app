from __future__ import annotations

import uuid

import pytest

from rag_app.db.models import Chunk
from rag_app.rag.chunking import ChunkDraft
from rag_app.rag.hierarchy_backfill import HierarchyBackfillError, build_hierarchy_updates


def _draft(*, idx: int = 0, segment_id: str = "segment-1") -> ChunkDraft:
    return ChunkDraft(
        idx=idx,
        kind="section",
        heading_path="Section",
        text_en="Source",
        text_ru="Перевод",
        page_start=0,
        page_end=0,
        meta={
            "segment_ids": [segment_id],
            "section_id": "section-1",
            "section_path": ["section-1"],
            "parent_id": "section-1",
            "source_ordinal": 1,
            "ordinal_in_section": 0,
        },
    )


def _chunk(*, idx: int = 0, segment_id: str = "segment-1") -> Chunk:
    return Chunk(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        idx=idx,
        kind="section",
        heading_path="Section",
        text_en="Source",
        text_ru="Перевод",
        page_start=0,
        page_end=0,
        meta={"segment_ids": [segment_id], "custom": "preserved"},
    )


def test_build_hierarchy_updates_preserves_non_hierarchy_metadata() -> None:
    chunk = _chunk()

    update = build_hierarchy_updates([_draft()], [chunk])[0]

    assert update.chunk_id == chunk.id
    assert update.changed is True
    assert update.meta["custom"] == "preserved"
    assert update.meta["section_id"] == "section-1"
    assert update.meta["ordinal_in_section"] == 0


def test_build_hierarchy_updates_is_idempotent() -> None:
    chunk = _chunk()
    first = build_hierarchy_updates([_draft()], [chunk])[0]
    chunk.meta = first.meta

    second = build_hierarchy_updates([_draft()], [chunk])[0]

    assert second.changed is False


def test_build_hierarchy_updates_fails_closed_on_binding_mismatch() -> None:
    with pytest.raises(HierarchyBackfillError, match="binding mismatch"):
        build_hierarchy_updates([_draft(segment_id="segment-a")], [_chunk(segment_id="segment-b")])
