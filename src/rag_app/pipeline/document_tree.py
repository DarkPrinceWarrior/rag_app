"""Строгий versioned sidecar дерева документа без привязки к runtime-модели.

Артефакт хранит только проверенные ссылки на канонические сегменты. Модель
постобработки (например, MinerU-Popo) не назначает устойчивые идентификаторы:
их вычисляет сервер из координат/OOXML-location исходных блоков.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rag_app.db.models import Segment, SegmentKind

DOCUMENT_TREE_SCHEMA_VERSION = 1
_MAX_ARTIFACT_BYTES = 32 * 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TREE_NAMESPACE = uuid.UUID("2096be12-2633-54ef-9a68-943f212768ef")

DocumentTreeNodeKind = Literal[
    "document",
    "section",
    "heading",
    "paragraph",
    "table",
    "image",
    "equation",
    "list",
    "list_item",
    "caption",
]
DocumentTreeEdgeKind = Literal[
    "continues",
    "table_continues",
    "caption_of",
    "attached_to",
    "next",
]

_POPO_BLOCK_TYPES = {
    "title",
    "text",
    "list_item",
    "equation",
    "image",
    "chart",
    "seal",
    "table",
    "image_caption",
    "table_caption",
    "image_footnote",
    "table_footnote",
    "page_title",
    "page_number",
    "page_footnote",
    "header",
    "aside_text",
    "footer",
}
_POPO_CAPTION_TYPES = {
    "image_caption",
    "table_caption",
    "image_footnote",
    "table_footnote",
}
_POPO_VISUAL_TYPES = {"image", "chart", "seal", "table"}
_POPO_TEXT_TYPES = {"text", "list_item"}


class DocumentTreeError(ValueError):
    """Артефакт не может быть безопасно связан с каноническими сегментами."""


class _StrictTreeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DocumentTreeAnchor(_StrictTreeModel):
    """Неизменяемая ссылка узла на геометрию канонического сегмента."""

    source_ref: str = Field(min_length=1, max_length=512)
    segment_id: uuid.UUID
    page_idx: int | None = Field(default=None, ge=0)
    bbox_pt: tuple[float, float, float, float] | None = None
    page_size_pt: tuple[float, float] | None = None
    location: dict[str, int | str] | None = None

    @model_validator(mode="after")
    def validate_geometry(self) -> DocumentTreeAnchor:
        if (self.bbox_pt is None) != (self.page_size_pt is None):
            raise ValueError("bbox_pt and page_size_pt must be provided together")
        if self.bbox_pt is not None and self.page_idx is None:
            raise ValueError("bbox geometry requires page_idx")
        if self.bbox_pt is not None:
            x0, y0, x1, y1 = self.bbox_pt
            width, height = self.page_size_pt or (0.0, 0.0)
            if not all(math.isfinite(value) for value in (*self.bbox_pt, width, height)):
                raise ValueError("tree geometry must contain finite numbers")
            if (
                width <= 0
                or height <= 0
                or x0 < 0
                or y0 < 0
                or x1 < x0
                or y1 < y0
                or x1 > width
                or y1 > height
            ):
                raise ValueError("tree geometry has invalid bounds")
        return self


class DocumentTreeNode(_StrictTreeModel):
    stable_id: uuid.UUID
    parent_id: uuid.UUID | None = None
    ordinal: int = Field(ge=0)
    kind: DocumentTreeNodeKind
    heading_level: int | None = Field(default=None, ge=1, le=9)
    title: str | None = Field(default=None, max_length=4096)
    page_start: int | None = Field(default=None, ge=0)
    page_end: int | None = Field(default=None, ge=0)
    anchors: tuple[DocumentTreeAnchor, ...] = Field(default=(), max_length=10_000)
    attrs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_page_range(self) -> DocumentTreeNode:
        if (self.page_start is None) != (self.page_end is None):
            raise ValueError("page_start and page_end must be provided together")
        if self.page_start is not None and self.page_end is not None:
            if self.page_end < self.page_start:
                raise ValueError("page_end must not precede page_start")
            anchor_pages = [anchor.page_idx for anchor in self.anchors if anchor.page_idx is not None]
            if anchor_pages and (
                min(anchor_pages) < self.page_start or max(anchor_pages) > self.page_end
            ):
                raise ValueError("node page range must contain all anchor pages")
        return self


class DocumentTreeEdge(_StrictTreeModel):
    source_id: uuid.UUID
    target_id: uuid.UUID
    kind: DocumentTreeEdgeKind
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class DocumentTreeArtifact(_StrictTreeModel):
    """DocumentTreeArtifact v1, пригодный для shadow A/B и tree-aware чанкинга."""

    artifact_type: Literal["document_tree"] = "document_tree"
    schema_version: Literal[1] = 1
    tree_id: uuid.UUID
    document_id: uuid.UUID
    parse_revision: int = Field(ge=0)
    source_sha256: str
    input_sha256: str
    annotation_sha256: str | None = None
    builder: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    model_revision: str | None = Field(default=None, max_length=128)
    nodes: tuple[DocumentTreeNode, ...] = Field(min_length=1, max_length=100_000)
    edges: tuple[DocumentTreeEdge, ...] = Field(default=(), max_length=200_000)
    metrics: dict[str, int | float | str | bool | None] = Field(default_factory=dict)

    @field_validator("source_sha256", "input_sha256", "annotation_sha256")
    @classmethod
    def validate_sha256(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("tree hashes must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def validate_tree(self) -> DocumentTreeArtifact:
        by_id = {node.stable_id: node for node in self.nodes}
        if len(by_id) != len(self.nodes):
            raise ValueError("tree node stable_id values must be unique")

        roots = [node for node in self.nodes if node.parent_id is None]
        if len(roots) != 1 or roots[0].kind != "document":
            raise ValueError("tree must contain exactly one document root")
        if any(node.kind == "document" for node in self.nodes if node.parent_id is not None):
            raise ValueError("document node may only be the root")

        child_ordinals: dict[uuid.UUID, set[int]] = defaultdict(set)
        for node in self.nodes:
            if node.parent_id is None:
                continue
            if node.parent_id not in by_id:
                raise ValueError("tree contains an orphan node")
            if node.ordinal in child_ordinals[node.parent_id]:
                raise ValueError("sibling ordinals must be unique")
            child_ordinals[node.parent_id].add(node.ordinal)
        for ordinals in child_ordinals.values():
            if ordinals != set(range(len(ordinals))):
                raise ValueError("sibling ordinals must be contiguous from zero")

        root_id = roots[0].stable_id
        for node in self.nodes:
            visited: set[uuid.UUID] = set()
            cursor = node
            while cursor.parent_id is not None:
                if cursor.stable_id in visited:
                    raise ValueError("tree contains a parent cycle")
                visited.add(cursor.stable_id)
                cursor = by_id[cursor.parent_id]
            if cursor.stable_id != root_id:
                raise ValueError("all nodes must descend from the document root")

        for edge in self.edges:
            if edge.source_id not in by_id or edge.target_id not in by_id:
                raise ValueError("tree edge references an unknown node")
            if edge.source_id == edge.target_id:
                raise ValueError("tree edge cannot reference the same node twice")
        return self


def canonical_document_tree_bytes(
    artifact: DocumentTreeArtifact,
    *,
    max_bytes: int = _MAX_ARTIFACT_BYTES,
) -> bytes:
    """Каноническое JSON-представление для content hash и private sidecar."""

    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    payload = json.dumps(
        artifact.model_dump(mode="json", exclude_none=True),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(payload) > max_bytes:
        raise ValueError(f"document tree artifact exceeds {max_bytes} bytes")
    return payload


def document_tree_sha256(artifact: DocumentTreeArtifact) -> str:
    return hashlib.sha256(canonical_document_tree_bytes(artifact)).hexdigest()


def _kind_value(segment: Segment) -> str:
    return segment.kind.value if isinstance(segment.kind, SegmentKind) else str(segment.kind)


def _clean_geometry(meta: dict[str, Any]) -> tuple[
    tuple[float, float, float, float] | None,
    tuple[float, float] | None,
]:
    bbox = meta.get("bbox_pt")
    page_size = meta.get("page_size_pt")
    if not (
        isinstance(bbox, (list, tuple))
        and len(bbox) == 4
        and isinstance(page_size, (list, tuple))
        and len(page_size) == 2
    ):
        return None, None
    try:
        return tuple(float(value) for value in bbox), tuple(float(value) for value in page_size)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None, None


def _source_ref(segment: Segment) -> str:
    meta = segment.meta or {}
    location = meta.get("location")
    if isinstance(location, dict) and location:
        encoded = json.dumps(location, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return f"ooxml:{encoded}"
    bbox, _ = _clean_geometry(meta)
    kind = _kind_value(segment)
    if segment.page_idx is not None and bbox is not None:
        quantized = ",".join(str(round(value)) for value in bbox)
        return f"pdf:{segment.page_idx}:{kind}:{quantized}"
    digest = hashlib.sha256((segment.source_text or "").encode("utf-8")).hexdigest()[:20]
    page = "none" if segment.page_idx is None else str(segment.page_idx)
    return f"linear:{page}:{segment.idx}:{kind}:{digest}"


def document_tree_input_sha256(segments: Sequence[Segment]) -> str:
    """Привязка sidecar к точному набору исходных сегментов и геометрии."""

    manifest = []
    for segment in sorted(segments, key=lambda item: item.idx):
        bbox, page_size = _clean_geometry(segment.meta or {})
        manifest.append(
            {
                "id": str(segment.id),
                "idx": segment.idx,
                "kind": _kind_value(segment),
                "page_idx": segment.page_idx,
                "heading_level": segment.heading_level,
                "text_sha256": hashlib.sha256(
                    (segment.source_text or "").encode("utf-8")
                ).hexdigest(),
                "bbox_pt": bbox,
                "page_size_pt": page_size,
                "location": (segment.meta or {}).get("location"),
            }
        )
    payload = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def segments_to_document_tree(
    segments: Sequence[Segment],
    *,
    document_id: uuid.UUID,
    parse_revision: int,
    source_sha256: str,
    builder: str = "linear-shadow",
    model_revision: str | None = None,
) -> DocumentTreeArtifact:
    """Построить детерминированный baseline tree из текущих Segment.

    Это чистый адаптер: он не меняет сегменты и не обращается к БД/MinIO.
    Heading-узлы образуют иерархию, остальные блоки принадлежат ближайшему heading.
    """

    if parse_revision < 0:
        raise DocumentTreeError("parse_revision must be non-negative")
    if not _SHA256_RE.fullmatch(source_sha256):
        raise DocumentTreeError("source_sha256 must be lowercase SHA-256")
    ordered = sorted(segments, key=lambda item: item.idx)
    if any(segment.document_id != document_id for segment in ordered):
        raise DocumentTreeError("all tree segments must belong to one document")

    input_sha256 = document_tree_input_sha256(ordered)
    document_namespace = uuid.uuid5(_TREE_NAMESPACE, str(document_id))
    tree_id = uuid.uuid5(
        document_namespace,
        f"tree:v1:{parse_revision}:{source_sha256}:{builder}:{model_revision or ''}:{input_sha256}",
    )
    root_id = uuid.uuid5(document_namespace, "root")
    nodes: list[DocumentTreeNode] = [
        DocumentTreeNode(stable_id=root_id, ordinal=0, kind="document")
    ]
    headings: dict[int, uuid.UUID] = {}
    next_ordinal: dict[uuid.UUID, int] = defaultdict(int)
    duplicate_refs: dict[str, int] = defaultdict(int)

    for segment in ordered:
        kind_value = _kind_value(segment)
        try:
            node_kind: DocumentTreeNodeKind = kind_value  # type: ignore[assignment]
            if node_kind not in {
                "heading", "paragraph", "table", "image", "equation",
            }:
                raise ValueError
        except ValueError as exc:
            raise DocumentTreeError(f"unsupported segment kind: {kind_value}") from exc

        node_level: int | None = None
        if node_kind == "heading":
            node_level = max(segment.heading_level or 1, 1)
            parent_levels = [candidate for candidate in headings if candidate < node_level]
            parent_id = headings[max(parent_levels)] if parent_levels else root_id
            headings = {
                candidate: value for candidate, value in headings.items() if candidate < node_level
            }
        else:
            parent_id = headings[max(headings)] if headings else root_id

        ref = _source_ref(segment)
        suffix = duplicate_refs[ref]
        duplicate_refs[ref] += 1
        unique_ref = ref if suffix == 0 else f"{ref}#{suffix}"
        stable_id = uuid.uuid5(document_namespace, unique_ref)
        bbox, page_size = _clean_geometry(segment.meta or {})
        raw_location = (segment.meta or {}).get("location")
        location = (
            {str(key): value for key, value in raw_location.items() if isinstance(value, (int, str))}
            if isinstance(raw_location, dict)
            else None
        )
        anchor = DocumentTreeAnchor(
            source_ref=unique_ref,
            segment_id=segment.id,
            page_idx=segment.page_idx,
            bbox_pt=bbox,
            page_size_pt=page_size,
            location=location or None,
        )
        ordinal = next_ordinal[parent_id]
        next_ordinal[parent_id] += 1
        nodes.append(
            DocumentTreeNode(
                stable_id=stable_id,
                parent_id=parent_id,
                ordinal=ordinal,
                kind=node_kind,
                heading_level=node_level,
                title=segment.source_text if node_kind == "heading" else None,
                page_start=segment.page_idx,
                page_end=segment.page_idx,
                anchors=(anchor,),
            )
        )
        if node_kind == "heading":
            assert node_level is not None
            headings[node_level] = stable_id

    return DocumentTreeArtifact(
        tree_id=tree_id,
        document_id=document_id,
        parse_revision=parse_revision,
        source_sha256=source_sha256,
        input_sha256=input_sha256,
        builder=builder,
        model_revision=model_revision,
        nodes=tuple(nodes),
    )


@dataclass(frozen=True, slots=True)
class _PopoBlock:
    source_id: str
    block_id: int
    block_type: str
    level: int
    contd: int
    image: int
    table_merge: int


@dataclass(slots=True)
class _PopoNodeState:
    source_id: str
    block_id: int
    block_type: str
    stable_id: uuid.UUID
    parent_id: uuid.UUID
    kind: DocumentTreeNodeKind
    heading_level: int | None
    title: str | None
    anchor: DocumentTreeAnchor
    attrs: dict[str, Any] = field(default_factory=dict)


def _strict_popo_int(value: Any, *, field_name: str, minimum: int = -1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DocumentTreeError(f"Popo {field_name} must be an integer >= {minimum}")
    return value


def _parse_popo_blocks(annotated_blocks: Sequence[Mapping[str, Any]]) -> list[_PopoBlock]:
    parsed: list[_PopoBlock] = []
    for raw in annotated_blocks:
        if not isinstance(raw, Mapping):
            raise DocumentTreeError("every Popo annotation must be an object")
        source_id = raw.get("source_id")
        if not isinstance(source_id, str) or not source_id or len(source_id) > 512:
            raise DocumentTreeError("Popo source_id must be a non-empty string")
        block_type = raw.get("type")
        if not isinstance(block_type, str) or block_type not in _POPO_BLOCK_TYPES:
            raise DocumentTreeError(f"unsupported Popo block type: {block_type!r}")
        block_id = _strict_popo_int(raw.get("id"), field_name="id", minimum=1)
        level = _strict_popo_int(raw.get("level", -1), field_name="level")
        contd = _strict_popo_int(raw.get("contd", -1), field_name="contd")
        image = _strict_popo_int(raw.get("image", -1), field_name="image")
        table_merge = _strict_popo_int(
            raw.get("table_merge", -1), field_name="table_merge"
        )
        if block_type != "table" and table_merge != -1:
            raise DocumentTreeError("table_merge is only valid for table blocks")
        parsed.append(
            _PopoBlock(
                source_id=source_id,
                block_id=block_id,
                block_type=block_type,
                level=level,
                contd=contd,
                image=image,
                table_merge=table_merge,
            )
        )

    source_ids = [block.source_id for block in parsed]
    if len(source_ids) != len(set(source_ids)):
        raise DocumentTreeError("Popo annotations contain duplicate source_id values")
    block_ids = [block.block_id for block in parsed]
    if len(block_ids) != len(set(block_ids)):
        raise DocumentTreeError("Popo annotations contain duplicate block ids")
    if set(block_ids) != set(range(1, len(parsed) + 1)):
        raise DocumentTreeError("Popo block ids must be contiguous from one")
    return sorted(parsed, key=lambda block: block.block_id)


def _popo_node_kind(block: _PopoBlock, segment: Segment) -> DocumentTreeNodeKind:
    segment_kind = _kind_value(segment)
    if block.block_type == "title":
        if segment_kind != "heading":
            raise DocumentTreeError("Popo title must map to a heading segment")
        return "heading"
    if block.block_type in _POPO_CAPTION_TYPES:
        if segment_kind not in {"paragraph", "image", "table"}:
            raise DocumentTreeError("Popo caption maps to an incompatible segment")
        return "caption"
    if block.block_type == "list_item":
        if segment_kind != "paragraph":
            raise DocumentTreeError("Popo list_item must map to a paragraph segment")
        return "list_item"
    if block.block_type in {"image", "chart", "seal"}:
        if segment_kind != "image":
            raise DocumentTreeError("Popo visual block must map to an image segment")
        return "image"
    if block.block_type == "table":
        if segment_kind != "table":
            raise DocumentTreeError("Popo table must map to a table segment")
        return "table"
    if block.block_type == "equation":
        if segment_kind != "equation":
            raise DocumentTreeError("Popo equation must map to an equation segment")
        return "equation"
    if segment_kind not in {"paragraph", "heading"}:
        raise DocumentTreeError("Popo text block maps to an incompatible segment")
    return "paragraph" if segment_kind == "paragraph" else "heading"


def _assert_acyclic_links(links: Mapping[int, int], *, label: str) -> None:
    for start in links:
        visited: set[int] = set()
        cursor = start
        while cursor in links:
            if cursor in visited:
                raise DocumentTreeError(f"Popo {label} relations contain a cycle")
            visited.add(cursor)
            cursor = links[cursor]


def _component_groups(
    pairs: Sequence[tuple[int, int]],
    *,
    namespace: uuid.UUID,
    label: str,
) -> dict[int, str]:
    neighbours: dict[int, set[int]] = defaultdict(set)
    for left, right in pairs:
        neighbours[left].add(right)
        neighbours[right].add(left)
    groups: dict[int, str] = {}
    for start in neighbours:
        if start in groups:
            continue
        stack = [start]
        component: set[int] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(neighbours[current] - component)
        stable = uuid.uuid5(namespace, f"{label}:" + ",".join(map(str, sorted(component))))
        for block_id in component:
            groups[block_id] = str(stable)
    return groups


def popo_annotations_to_document_tree(
    annotated_blocks: Sequence[Mapping[str, Any]],
    *,
    source_segments: Mapping[str, Segment],
    document_id: uuid.UUID,
    parse_revision: int,
    source_sha256: str,
    model_revision: str,
    builder: str = "mineru-popo",
) -> DocumentTreeArtifact:
    """Безопасно преобразовать Popo annotated blocks, не вызывая get_json_tree.

    `source_segments` является единственным источником текста и геометрии.
    Annotated blocks могут менять только heading hierarchy и валидные связи;
    неизвестная/дублированная/потерянная ссылка отклоняет весь артефакт.
    """

    if parse_revision < 0:
        raise DocumentTreeError("parse_revision must be non-negative")
    if not _SHA256_RE.fullmatch(source_sha256):
        raise DocumentTreeError("source_sha256 must be lowercase SHA-256")
    if not model_revision or len(model_revision) > 128:
        raise DocumentTreeError("model_revision must be a non-empty bounded string")
    if any(not isinstance(source_id, str) or not source_id for source_id in source_segments):
        raise DocumentTreeError("source segment keys must be non-empty strings")

    blocks = _parse_popo_blocks(annotated_blocks)
    annotated_source_ids = {block.source_id for block in blocks}
    mapped_source_ids = set(source_segments)
    if annotated_source_ids != mapped_source_ids:
        raise DocumentTreeError("Popo source_id coverage must exactly match source segments")

    mapped_segments = list(source_segments.values())
    if len({segment.id for segment in mapped_segments}) != len(mapped_segments):
        raise DocumentTreeError("source_id mapping must be one-to-one with segments")
    if any(segment.document_id != document_id for segment in mapped_segments):
        raise DocumentTreeError("all Popo source segments must belong to one document")

    input_sha256 = document_tree_input_sha256(mapped_segments)
    annotations_manifest = [
        {
            "source_id": block.source_id,
            "id": block.block_id,
            "type": block.block_type,
            "level": block.level,
            "contd": block.contd,
            "image": block.image,
            "table_merge": block.table_merge,
        }
        for block in blocks
    ]
    annotation_sha256 = hashlib.sha256(
        json.dumps(
            annotations_manifest,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    document_namespace = uuid.uuid5(_TREE_NAMESPACE, str(document_id))
    tree_id = uuid.uuid5(
        document_namespace,
        f"tree:v1:{parse_revision}:{source_sha256}:{builder}:{model_revision}:"
        f"{input_sha256}:{annotation_sha256}",
    )
    root_id = uuid.uuid5(document_namespace, "root")

    duplicate_refs: dict[str, int] = defaultdict(int)
    headings: dict[int, uuid.UUID] = {}
    states: list[_PopoNodeState] = []
    by_block_id: dict[int, _PopoNodeState] = {}
    for block in blocks:
        segment = source_segments[block.source_id]
        node_kind = _popo_node_kind(block, segment)
        if node_kind == "heading":
            # level=-1 означает отсутствие решения модели, а не потерю title.
            heading_level = block.level if block.level > 0 else max(segment.heading_level or 1, 1)
            parent_levels = [candidate for candidate in headings if candidate < heading_level]
            parent_id = headings[max(parent_levels)] if parent_levels else root_id
            headings = {
                candidate: value
                for candidate, value in headings.items()
                if candidate < heading_level
            }
        else:
            heading_level = None
            parent_id = headings[max(headings)] if headings else root_id

        ref = _source_ref(segment)
        suffix = duplicate_refs[ref]
        duplicate_refs[ref] += 1
        unique_ref = ref if suffix == 0 else f"{ref}#{suffix}"
        bbox, page_size = _clean_geometry(segment.meta or {})
        raw_location = (segment.meta or {}).get("location")
        location = (
            {str(key): value for key, value in raw_location.items() if isinstance(value, (int, str))}
            if isinstance(raw_location, dict)
            else None
        )
        state = _PopoNodeState(
            source_id=block.source_id,
            block_id=block.block_id,
            block_type=block.block_type,
            stable_id=uuid.uuid5(document_namespace, unique_ref),
            parent_id=parent_id,
            kind=node_kind,
            heading_level=heading_level,
            title=segment.source_text if node_kind == "heading" else None,
            anchor=DocumentTreeAnchor(
                source_ref=unique_ref,
                segment_id=segment.id,
                page_idx=segment.page_idx,
                bbox_pt=bbox,
                page_size_pt=page_size,
                location=location or None,
            ),
            attrs={"popo_source_id": block.source_id, "popo_block_id": block.block_id},
        )
        states.append(state)
        by_block_id[block.block_id] = state
        if node_kind == "heading":
            assert heading_level is not None
            headings[heading_level] = state.stable_id

    edges: list[DocumentTreeEdge] = []
    contd_links: dict[int, int] = {}
    incoming_contd: set[int] = set()
    contd_pairs: list[tuple[int, int]] = []
    for block in blocks:
        if block.contd < 0:
            continue
        target = by_block_id.get(block.contd)
        source = by_block_id[block.block_id]
        if target is None:
            raise DocumentTreeError("Popo contd references an unknown block")
        if source.block_id == target.block_id:
            raise DocumentTreeError("Popo contd cannot self-reference")
        if block.block_type not in _POPO_TEXT_TYPES or target.block_type not in _POPO_TEXT_TYPES:
            raise DocumentTreeError("Popo contd is only valid between text/list blocks")
        if target.block_id in incoming_contd:
            raise DocumentTreeError("Popo contd has conflicting incoming relations")
        if source.parent_id != target.parent_id:
            raise DocumentTreeError("Popo contd cannot cross heading parents")
        incoming_contd.add(target.block_id)
        contd_links[source.block_id] = target.block_id
        contd_pairs.append((source.block_id, target.block_id))
        edges.append(
            DocumentTreeEdge(
                source_id=source.stable_id,
                target_id=target.stable_id,
                kind="continues",
            )
        )
    _assert_acyclic_links(contd_links, label="contd")

    table_pairs: list[tuple[int, int]] = []
    handled_tables: set[frozenset[int]] = set()
    by_annotation_id = {block.block_id: block for block in blocks}
    for block in blocks:
        if block.table_merge < 0:
            continue
        target_block = by_annotation_id.get(block.table_merge)
        if target_block is None:
            raise DocumentTreeError("Popo table_merge references an unknown block")
        if target_block.block_type != "table":
            raise DocumentTreeError("Popo table_merge is only valid between tables")
        pair_key = frozenset((block.block_id, target_block.block_id))
        if len(pair_key) != 2:
            raise DocumentTreeError("Popo table_merge cannot self-reference")
        if pair_key in handled_tables:
            continue
        handled_tables.add(pair_key)
        left, right = sorted(pair_key)
        left_node, right_node = by_block_id[left], by_block_id[right]
        if left_node.parent_id != right_node.parent_id:
            raise DocumentTreeError("Popo table_merge cannot cross heading parents")
        left_page = left_node.anchor.page_idx
        right_page = right_node.anchor.page_idx
        if (
            left_page is not None
            and right_page is not None
            and not 0 <= right_page - left_page <= 1
        ):
            raise DocumentTreeError(
                "Popo table_merge requires monotonic adjacent pages"
            )
        table_pairs.append((left, right))
        edges.append(
            DocumentTreeEdge(
                source_id=left_node.stable_id,
                target_id=right_node.stable_id,
                kind="table_continues",
            )
        )

    contd_groups = _component_groups(
        contd_pairs, namespace=document_namespace, label="contd-group"
    )
    table_groups = _component_groups(
        table_pairs, namespace=document_namespace, label="table-group"
    )
    for state in states:
        if state.block_id in contd_groups:
            state.attrs["continuation_group"] = contd_groups[state.block_id]
        if state.block_id in table_groups:
            state.attrs["table_merge_group"] = table_groups[state.block_id]

    image_links: dict[int, int] = {}
    for block in blocks:
        if block.image < 0:
            continue
        source = by_block_id[block.block_id]
        target = by_block_id.get(block.image)
        if target is None:
            raise DocumentTreeError("Popo image relation references an unknown block")
        if source.block_id == target.block_id:
            raise DocumentTreeError("Popo image relation cannot self-reference")
        if block.block_type in _POPO_CAPTION_TYPES and target.block_type in _POPO_VISUAL_TYPES:
            source.parent_id = target.stable_id
            edge_source, edge_target, edge_kind = source, target, "caption_of"
        elif block.block_type in _POPO_VISUAL_TYPES and (
            target.block_type in _POPO_VISUAL_TYPES or target.block_type == "title"
        ):
            source.parent_id = target.stable_id
            edge_source, edge_target, edge_kind = source, target, "attached_to"
        else:
            raise DocumentTreeError("unsupported Popo image relation types")
        image_links[source.block_id] = target.block_id
        edges.append(
            DocumentTreeEdge(
                source_id=edge_source.stable_id,
                target_id=edge_target.stable_id,
                kind=edge_kind,  # type: ignore[arg-type]
            )
        )
    _assert_acyclic_links(image_links, label="image")

    children: dict[uuid.UUID, list[_PopoNodeState]] = defaultdict(list)
    for state in states:
        children[state.parent_id].append(state)
    for siblings in children.values():
        siblings.sort(key=lambda state: state.block_id)
    ordinal_by_id = {
        state.stable_id: ordinal
        for siblings in children.values()
        for ordinal, state in enumerate(siblings)
    }
    nodes: list[DocumentTreeNode] = [
        DocumentTreeNode(stable_id=root_id, ordinal=0, kind="document")
    ]
    for state in states:
        page_idx = state.anchor.page_idx
        nodes.append(
            DocumentTreeNode(
                stable_id=state.stable_id,
                parent_id=state.parent_id,
                ordinal=ordinal_by_id[state.stable_id],
                kind=state.kind,
                heading_level=state.heading_level,
                title=state.title,
                page_start=page_idx,
                page_end=page_idx,
                anchors=(state.anchor,),
                attrs=state.attrs,
            )
        )

    return DocumentTreeArtifact(
        tree_id=tree_id,
        document_id=document_id,
        parse_revision=parse_revision,
        source_sha256=source_sha256,
        input_sha256=input_sha256,
        annotation_sha256=annotation_sha256,
        builder=builder,
        model_revision=model_revision,
        nodes=tuple(nodes),
        edges=tuple(edges),
        metrics={"source_blocks": len(blocks), "anchored_segments": len(mapped_segments)},
    )
