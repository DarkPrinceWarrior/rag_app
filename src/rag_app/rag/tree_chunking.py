"""Pure tree-aware chunk selection with a fail-safe flat fallback."""

from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

from rag_app.db.models import Segment
from rag_app.pipeline.document_tree import (
    DocumentTreeArtifact,
    DocumentTreeError,
    document_tree_input_sha256,
)
from rag_app.rag.chunking import ChunkDraft, segments_to_chunks

DocumentTreeMode = Literal["off", "shadow", "active"]


class DocumentTreeChunkError(DocumentTreeError):
    """Дерево нельзя применить к данной ревизии набора сегментов."""


@dataclass(frozen=True, slots=True)
class _TreeSegment:
    id: uuid.UUID
    document_id: uuid.UUID
    idx: int
    page_idx: int | None
    kind: object
    heading_level: int | None
    source_text: str
    translated_text: str | None
    meta: dict


@dataclass(frozen=True, slots=True)
class TreeChunkSelection:
    """Результат выбора: shadow никогда не меняет selected."""

    selected: list[ChunkDraft]
    baseline: list[ChunkDraft]
    shadow: list[ChunkDraft] | None
    used_tree: bool
    fallback_reason: str | None = None


def _ordered_nodes(artifact: DocumentTreeArtifact):
    by_parent = defaultdict(list)
    roots = []
    for node in artifact.nodes:
        if node.parent_id is None:
            roots.append(node)
        else:
            by_parent[node.parent_id].append(node)
    for children in by_parent.values():
        children.sort(key=lambda node: node.ordinal)

    def walk(node):
        yield node
        for child in by_parent[node.stable_id]:
            yield from walk(child)

    yield from walk(roots[0])


def _group_value(attrs: dict, key: str) -> str | None:
    value = attrs.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise DocumentTreeChunkError(f"tree {key} must be a non-empty string")
    return value


def _join_chunk_text(left: str, right: str) -> str:
    return "\n".join(part for part in (left, right) if part).strip()


def _merge_page_start(left: int | None, right: int | None) -> int | None:
    values = [value for value in (left, right) if value is not None]
    return min(values) if values else None


def _merge_page_end(left: int | None, right: int | None) -> int | None:
    values = [value for value in (left, right) if value is not None]
    return max(values) if values else None


def _merge_adjacent_table_groups(drafts: list[ChunkDraft]) -> list[ChunkDraft]:
    """Merge only consecutive table chunks carrying the same trusted Popo group."""

    merged: list[ChunkDraft] = []
    aggregate_keys = {"segment_ids", "bboxes", "tree_node_ids"}
    for draft in drafts:
        previous = merged[-1] if merged else None
        group = draft.meta.get("table_merge_group")
        can_merge = (
            previous is not None
            and draft.kind == "table"
            and previous.kind == "table"
            and isinstance(group, str)
            and group
            and previous.meta.get("table_merge_group") == group
            and previous.heading_path == draft.heading_path
        )
        if not can_merge:
            merged.append(draft)
            continue

        assert previous is not None
        combined_meta = dict(previous.meta)
        for key, value in draft.meta.items():
            if key in aggregate_keys:
                continue
            if key in combined_meta and combined_meta[key] != value:
                raise DocumentTreeChunkError(
                    f"table merge group has conflicting metadata: {key}"
                )
            combined_meta[key] = value
        for key in ("segment_ids", "tree_node_ids"):
            values = [*previous.meta.get(key, []), *draft.meta.get(key, [])]
            combined_meta[key] = list(dict.fromkeys(values))
        combined_meta["bboxes"] = [
            *previous.meta.get("bboxes", []),
            *draft.meta.get("bboxes", []),
        ]
        merged[-1] = ChunkDraft(
            idx=previous.idx,
            kind="table",
            heading_path=previous.heading_path,
            text_en=_join_chunk_text(previous.text_en, draft.text_en),
            text_ru=_join_chunk_text(previous.text_ru, draft.text_ru),
            page_start=_merge_page_start(previous.page_start, draft.page_start),
            page_end=_merge_page_end(previous.page_end, draft.page_end),
            meta=combined_meta,
        )

    for index, draft in enumerate(merged):
        draft.idx = index
    return merged


def document_tree_to_chunks(
    segments: Sequence[Segment],
    artifact: DocumentTreeArtifact,
) -> list[ChunkDraft]:
    """Построить прежний ChunkDraft-контракт в порядке проверенного дерева.

    Текст и перевод всегда берутся из текущих Segment, а не из sidecar. Это
    сохраняет ручные правки. Sidecar определяет только порядок/иерархию и
    содержит неизменяемые anchors для цитат.
    """

    by_id = {segment.id: segment for segment in segments}
    if len(by_id) != len(segments):
        raise DocumentTreeChunkError("segment ids must be unique")
    if any(segment.document_id != artifact.document_id for segment in segments):
        raise DocumentTreeChunkError("tree and segments belong to different documents")
    if document_tree_input_sha256(segments) != artifact.input_sha256:
        raise DocumentTreeChunkError("tree input hash does not match current segments")

    ordered: list[_TreeSegment] = []
    segment_node_ids: dict[str, str] = {}
    segment_node_attrs: dict[str, dict] = {}
    seen: set[uuid.UUID] = set()
    next_idx = 0
    for node in _ordered_nodes(artifact):
        for anchor in node.anchors:
            segment = by_id.get(anchor.segment_id)
            if segment is None:
                raise DocumentTreeChunkError("tree references an unknown segment")
            if segment.id in seen:
                raise DocumentTreeChunkError("tree references a segment more than once")
            seen.add(segment.id)
            segment_node_ids[str(segment.id)] = str(node.stable_id)
            segment_node_attrs[str(segment.id)] = node.attrs
            ordered.append(
                _TreeSegment(
                    id=segment.id,
                    document_id=segment.document_id,
                    idx=next_idx,
                    page_idx=segment.page_idx,
                    kind=segment.kind,
                    heading_level=(
                        node.heading_level
                        if str(getattr(segment.kind, "value", segment.kind)) == "heading"
                        else segment.heading_level
                    ),
                    source_text=segment.source_text,
                    translated_text=segment.translated_text,
                    meta=dict(segment.meta or {}),
                )
            )
            next_idx += 1

    if seen != set(by_id):
        raise DocumentTreeChunkError("tree does not cover every segment")

    drafts = segments_to_chunks(cast(list[Segment], ordered))
    segment_lookup = {str(segment.id): segment for segment in segments}
    tree_position = {str(segment.id): position for position, segment in enumerate(ordered)}
    # Старый чанкер добавляет отдельную таблицу немедленно, а накопленный перед
    # ней section сбрасывает позднее. Для tree-кандидата возвращаем чанки именно
    # в порядке первого исходного узла; baseline/off при этом не меняется.
    drafts.sort(
        key=lambda draft: min(
            tree_position[str(segment_id)] for segment_id in draft.meta.get("segment_ids", [])
        )
    )
    for index, draft in enumerate(drafts):
        draft.idx = index
    for draft in drafts:
        segment_ids = [str(value) for value in draft.meta.get("segment_ids", [])]
        node_ids = list(
            dict.fromkeys(segment_node_ids[segment_id] for segment_id in segment_ids)
        )
        continuation_groups = list(
            dict.fromkeys(
                group
                for segment_id in segment_ids
                if (
                    group := _group_value(
                        segment_node_attrs[segment_id], "continuation_group"
                    )
                )
                is not None
            )
        )
        table_groups = list(
            dict.fromkeys(
                group
                for segment_id in segment_ids
                if (
                    group := _group_value(
                        segment_node_attrs[segment_id], "table_merge_group"
                    )
                )
                is not None
            )
        )
        if table_groups and (draft.kind != "table" or len(table_groups) != 1):
            raise DocumentTreeChunkError(
                "table merge group must resolve to exactly one table chunk group"
            )
        # Сохраняем текущую форму bboxes и только добавляем page_size для
        # автономной навигации, если старый segment_id позже исчезнет при reparse.
        bboxes = []
        for segment_id in segment_ids:
            segment = segment_lookup[segment_id]
            meta = segment.meta or {}
            bbox = meta.get("bbox_pt")
            if bbox is None:
                continue
            item = {"page": segment.page_idx, "bbox": bbox}
            if meta.get("page_size_pt") is not None:
                item["page_size"] = meta["page_size_pt"]
            bboxes.append(item)
        draft.meta = {
            **draft.meta,
            "segment_ids": segment_ids,
            "bboxes": bboxes,
            "tree_id": str(artifact.tree_id),
            "tree_node_ids": node_ids,
        }
        if continuation_groups:
            draft.meta["continuation_groups"] = continuation_groups
            if len(continuation_groups) == 1:
                draft.meta["continuation_group"] = continuation_groups[0]
        if table_groups:
            draft.meta["table_merge_group"] = table_groups[0]
    return _merge_adjacent_table_groups(drafts)


def select_document_chunks(
    segments: Sequence[Segment],
    *,
    mode: DocumentTreeMode,
    artifact: DocumentTreeArtifact | None,
) -> TreeChunkSelection:
    """Выбрать чанки без изменения default behavior.

    off: дерево даже не читается; shadow: кандидат возвращается отдельно;
    active: валидный кандидат выбирается, иначе используется прежний baseline.
    """

    if mode not in {"off", "shadow", "active"}:
        raise ValueError(f"unsupported document tree mode: {mode}")
    baseline = segments_to_chunks(list(segments))
    if mode == "off":
        return TreeChunkSelection(
            selected=baseline,
            baseline=baseline,
            shadow=None,
            used_tree=False,
        )
    if artifact is None:
        return TreeChunkSelection(
            selected=baseline,
            baseline=baseline,
            shadow=None,
            used_tree=False,
            fallback_reason="missing_artifact",
        )
    try:
        candidate = document_tree_to_chunks(segments, artifact)
    except DocumentTreeError as exc:
        return TreeChunkSelection(
            selected=baseline,
            baseline=baseline,
            shadow=None,
            used_tree=False,
            fallback_reason=type(exc).__name__,
        )
    if mode == "shadow":
        return TreeChunkSelection(
            selected=baseline,
            baseline=baseline,
            shadow=candidate,
            used_tree=False,
        )
    return TreeChunkSelection(
        selected=candidate,
        baseline=baseline,
        shadow=candidate,
        used_tree=True,
    )
