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
    fields: list[KIEField] = Field(max_length=1000)


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
    schema_version: Literal[1] = 1
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


class ChartArtifact(_ArtifactEnvelope):
    artifact_type: Literal[ArtifactType.chart]
    payload: ChartPayload


class DiagramArtifact(_ArtifactEnvelope):
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
