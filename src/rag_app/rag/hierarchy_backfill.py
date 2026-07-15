"""Fail-closed binding for in-place chunk hierarchy metadata backfills."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from rag_app.db.models import Chunk
from rag_app.rag.chunking import ChunkDraft

_HIERARCHY_KEYS = frozenset(
    {
        "section_id",
        "section_path",
        "parent_id",
        "source_ordinal",
        "ordinal_in_section",
        "logical_table_id",
        "continuation_index",
    }
)


class HierarchyBackfillError(RuntimeError):
    """Existing chunks do not match a deterministic rebuild from their segments."""


@dataclass(frozen=True, slots=True)
class ChunkHierarchyUpdate:
    chunk_id: uuid.UUID
    meta: dict[str, Any]
    changed: bool


def build_hierarchy_updates(
    drafts: Sequence[ChunkDraft], chunks: Sequence[Chunk]
) -> tuple[ChunkHierarchyUpdate, ...]:
    """Bind rebuilt drafts 1:1 to stored chunks without relying on text matching."""

    if len(drafts) != len(chunks):
        raise HierarchyBackfillError("draft/chunk count mismatch")
    updates: list[ChunkHierarchyUpdate] = []
    for draft, chunk in zip(drafts, chunks, strict=True):
        draft_segments = tuple(str(value) for value in draft.meta.get("segment_ids", []))
        chunk_segments = tuple(str(value) for value in (chunk.meta or {}).get("segment_ids", []))
        if (
            draft.idx != chunk.idx
            or draft.kind != chunk.kind
            or draft.heading_path != chunk.heading_path
            or draft_segments != chunk_segments
        ):
            raise HierarchyBackfillError(f"chunk binding mismatch at idx={chunk.idx}")
        current_meta = dict(chunk.meta or {})
        updated_meta = {
            key: value for key, value in current_meta.items() if key not in _HIERARCHY_KEYS
        }
        updated_meta.update(
            {key: draft.meta[key] for key in _HIERARCHY_KEYS if key in draft.meta}
        )
        updates.append(
            ChunkHierarchyUpdate(
                chunk_id=chunk.id,
                meta=updated_meta,
                changed=updated_meta != current_meta,
            )
        )
    return tuple(updates)
