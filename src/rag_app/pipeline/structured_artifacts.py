"""Строгий контракт advisory-артефактов KIE, графиков и диаграмм."""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UNSAFE_TEXT_RE = re.compile(r"(?:^\s*(?:data:|file://)|<\s*(?:script|iframe|object|embed)\b)", re.I)
_MAX_ARTIFACT_BYTES = 5 * 1024 * 1024


class ArtifactType(StrEnum):
    kie = "kie"
    chart = "chart"
    diagram = "diagram"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    @model_validator(mode="after")
    def reject_embedded_resources(self) -> Self:
        for value in self.__dict__.values():
            if isinstance(value, str) and _UNSAFE_TEXT_RE.search(value):
                raise ValueError("embedded resources and active HTML are not allowed")
        return self


class EvidenceBBox(_StrictModel):
    bbox_pt: tuple[float, float, float, float]
    page_size_pt: tuple[float, float]

    @model_validator(mode="after")
    def validate_geometry(self) -> Self:
        x0, y0, x1, y1 = self.bbox_pt
        width, height = self.page_size_pt
        values = (*self.bbox_pt, *self.page_size_pt)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("geometry values must be finite")
        if width <= 0 or height <= 0 or not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("bbox_pt must be inside page_size_pt")
        return self


ScalarValue = str | int | float | bool | None


class KIEField(_StrictModel):
    name: str = Field(min_length=1, max_length=128)
    value: ScalarValue = Field(default=None)
    normalized_value: str | None = Field(default=None, max_length=2048)
    unit: str | None = Field(default=None, max_length=64)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: EvidenceBBox | None = None

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: ScalarValue) -> ScalarValue:
        if isinstance(value, str) and len(value) > 4096:
            raise ValueError("field value is too long")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("field value must be finite")
        return value


class KIEPayload(_StrictModel):
    fields: list[KIEField] = Field(default_factory=list, max_length=1000)
    schema_sha256: str | None = None
    result: dict[str, Any] | None = None

    @field_validator("schema_sha256")
    @classmethod
    def validate_schema_hash(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256_RE.fullmatch(value):
            raise ValueError("schema_sha256 must be lowercase hexadecimal")
        return value

    @field_validator("result")
    @classmethod
    def validate_exact_result(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        stack: list[tuple[Any, int]] = [(value, 0)]
        visited = 0
        while stack:
            node, depth = stack.pop()
            visited += 1
            if visited > 20_000 or depth > 64:
                raise ValueError("KIE result exceeds structural limits")
            if isinstance(node, dict):
                for key, child in node.items():
                    if not isinstance(key, str) or not key or len(key) > 256:
                        raise ValueError("KIE result keys must be bounded strings")
                    if _UNSAFE_TEXT_RE.search(key):
                        raise ValueError("embedded resources and active HTML are not allowed")
                    stack.append((child, depth + 1))
            elif isinstance(node, list):
                stack.extend((child, depth + 1) for child in node)
            elif isinstance(node, str):
                if len(node) > 16_384:
                    raise ValueError("KIE result string is too long")
                if _UNSAFE_TEXT_RE.search(node):
                    raise ValueError("embedded resources and active HTML are not allowed")
            elif isinstance(node, float):
                if not math.isfinite(node):
                    raise ValueError("KIE result numbers must be finite")
            elif node is not None and not isinstance(node, int | bool):
                raise ValueError("KIE result contains a non-JSON value")
        return value

    @model_validator(mode="after")
    def validate_representation(self) -> Self:
        exact = self.result is not None
        if exact and (self.schema_sha256 is None or self.fields):
            raise ValueError("exact KIE result requires schema hash and no flat fields")
        if not exact and self.schema_sha256 is not None:
            raise ValueError("flat KIE fields cannot carry an exact schema hash")
        return self


class ChartAxis(_StrictModel):
    name: str = Field(min_length=1, max_length=128)
    unit: str | None = Field(default=None, max_length=64)


class ChartPoint(_StrictModel):
    x: str | float
    y: str | float

    @field_validator("x", "y")
    @classmethod
    def validate_coordinate(cls, value: str | float) -> str | float:
        if isinstance(value, str) and len(value) > 256:
            raise ValueError("chart coordinate is too long")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("chart coordinate must be finite")
        return value


class ChartSeries(_StrictModel):
    name: str = Field(min_length=1, max_length=256)
    points: list[ChartPoint] = Field(max_length=5000)


class ChartPayload(_StrictModel):
    x_axis: ChartAxis
    y_axis: ChartAxis
    series: list[ChartSeries] = Field(max_length=64)
    evidence: EvidenceBBox

    @model_validator(mode="after")
    def validate_total_points(self) -> Self:
        if sum(len(series.points) for series in self.series) > 10_000:
            raise ValueError("chart contains too many points")
        return self


class DiagramNode(_StrictModel):
    id: str = Field(min_length=1, max_length=128)
    label: str = Field(default="", max_length=2048)
    kind: str | None = Field(default=None, max_length=128)
    evidence: EvidenceBBox | None = None


class DiagramEdge(_StrictModel):
    source: str = Field(min_length=1, max_length=128)
    target: str = Field(min_length=1, max_length=128)
    label: str | None = Field(default=None, max_length=512)


class DiagramPayload(_StrictModel):
    nodes: list[DiagramNode] = Field(max_length=2000)
    edges: list[DiagramEdge] = Field(max_length=4000)
    evidence: EvidenceBBox

    @model_validator(mode="after")
    def validate_graph(self) -> Self:
        node_ids = [node.id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("diagram node ids must be unique")
        known = set(node_ids)
        if any(edge.source not in known or edge.target not in known for edge in self.edges):
            raise ValueError("diagram edge references an unknown node")
        return self


class _ArtifactEnvelope(_StrictModel):
    schema_version: int = Field(default=1, ge=1, le=2)
    artifact_id: uuid.UUID
    document_id: uuid.UUID
    parse_revision: int = Field(ge=0)
    page_idx: int = Field(ge=0)
    source_sha256: str
    backend: str = Field(min_length=1, max_length=64)
    model: str = Field(min_length=1, max_length=128)
    prompt_version: str = Field(min_length=1, max_length=32)
    generated_at: datetime

    @field_validator("source_sha256")
    @classmethod
    def validate_source_hash(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("source_sha256 must be lowercase hexadecimal")
        return value

    @field_validator("generated_at")
    @classmethod
    def validate_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("generated_at must be timezone-aware")
        return value


class KIEArtifact(_ArtifactEnvelope):
    artifact_type: Literal[ArtifactType.kie]
    payload: KIEPayload

    @model_validator(mode="after")
    def validate_kie_version(self) -> Self:
        expected = 2 if self.payload.result is not None else 1
        if self.schema_version != expected:
            raise ValueError(f"KIE payload requires schema_version={expected}")
        return self


class ChartArtifact(_ArtifactEnvelope):
    schema_version: Literal[1] = 1
    artifact_type: Literal[ArtifactType.chart]
    payload: ChartPayload


class DiagramArtifact(_ArtifactEnvelope):
    schema_version: Literal[1] = 1
    artifact_type: Literal[ArtifactType.diagram]
    payload: DiagramPayload


StructuredArtifact = Annotated[
    KIEArtifact | ChartArtifact | DiagramArtifact,
    Field(discriminator="artifact_type"),
]
STRUCTURED_ARTIFACT_ADAPTER: TypeAdapter[StructuredArtifact] = TypeAdapter(StructuredArtifact)


def validate_structured_artifact(value: Any) -> StructuredArtifact:
    return STRUCTURED_ARTIFACT_ADAPTER.validate_python(value)


def canonical_artifact_bytes(
    artifact: StructuredArtifact,
    *,
    max_bytes: int = _MAX_ARTIFACT_BYTES,
) -> bytes:
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
        raise ValueError(f"structured artifact exceeds {max_bytes} bytes")
    return payload


def artifact_content_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def build_artifact_key(
    *,
    document_id: uuid.UUID,
    parse_revision: int,
    page_idx: int,
    artifact_type: ArtifactType,
    artifact_id: uuid.UUID,
) -> str:
    if parse_revision < 0:
        raise ValueError("parse_revision must be non-negative")
    if page_idx < 0:
        raise ValueError("page_idx must be non-negative")
    return (
        f"{document_id}/sidecars/r{parse_revision}/p{page_idx:06d}/"
        f"{artifact_type.value}/{artifact_id}.json"
    )
