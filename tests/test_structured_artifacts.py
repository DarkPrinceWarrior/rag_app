from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from rag_app.pipeline.structured_artifacts import (
    ArtifactType,
    KIEArtifact,
    artifact_content_sha256,
    build_artifact_key,
    canonical_artifact_bytes,
    validate_structured_artifact,
)

_DOC_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
_ARTIFACT_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")


def _base(artifact_type: str, payload: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": artifact_type,
        "artifact_id": _ARTIFACT_ID,
        "document_id": _DOC_ID,
        "parse_revision": 3,
        "page_idx": 7,
        "source_sha256": "a" * 64,
        "backend": "granite",
        "model": "granite-vision-4.1-4b",
        "prompt_version": "kie-v1",
        "generated_at": datetime(2026, 7, 11, tzinfo=UTC),
        "payload": payload,
    }


def _bbox() -> dict[str, object]:
    return {"bbox_pt": [10, 20, 190, 280], "page_size_pt": [200, 300]}


def test_validate_kie_artifact_and_canonical_hash() -> None:
    artifact = validate_structured_artifact(
        _base(
            "kie",
            {
                "fields": [
                    {
                        "name": "pressure",
                        "value": 16.5,
                        "unit": "MPa",
                        "confidence": 0.9,
                        "evidence": _bbox(),
                    }
                ]
            },
        )
    )

    assert isinstance(artifact, KIEArtifact)
    first = canonical_artifact_bytes(artifact)
    second = canonical_artifact_bytes(artifact)
    assert first == second
    assert len(artifact_content_sha256(first)) == 64


def test_validate_chart_and_diagram_artifacts() -> None:
    chart = validate_structured_artifact(
        _base(
            "chart",
            {
                "x_axis": {"name": "time", "unit": "s"},
                "y_axis": {"name": "pressure", "unit": "MPa"},
                "series": [{"name": "P1", "points": [{"x": 0.0, "y": 1.0}]}],
                "evidence": _bbox(),
            },
        )
    )
    diagram = validate_structured_artifact(
        _base(
            "diagram",
            {
                "nodes": [{"id": "pump", "label": "P-101"}, {"id": "tank"}],
                "edges": [{"source": "pump", "target": "tank"}],
                "evidence": _bbox(),
            },
        )
    )

    assert chart.artifact_type == ArtifactType.chart
    assert diagram.artifact_type == ArtifactType.diagram


def test_rejects_invalid_geometry_unknown_edge_and_extra_fields() -> None:
    invalid_bbox = _base(
        "kie",
        {"fields": [{"name": "x", "value": "y", "evidence": {
            "bbox_pt": [0, 0, 220, 300], "page_size_pt": [200, 300]
        }}]},
    )
    with pytest.raises(ValidationError, match="bbox_pt"):
        validate_structured_artifact(invalid_bbox)

    unknown_edge = _base(
        "diagram",
        {
            "nodes": [{"id": "a"}],
            "edges": [{"source": "a", "target": "missing"}],
            "evidence": _bbox(),
        },
    )
    with pytest.raises(ValidationError, match="unknown node"):
        validate_structured_artifact(unknown_edge)

    extra = _base("kie", {"fields": []})
    extra["artifact_key"] = "must-not-be-accepted"
    with pytest.raises(ValidationError, match="Extra inputs"):
        validate_structured_artifact(extra)


def test_rejects_naive_time_bad_hash_and_oversized_payload() -> None:
    naive = _base("kie", {"fields": []})
    naive["generated_at"] = datetime(2026, 7, 11)
    with pytest.raises(ValidationError, match="timezone-aware"):
        validate_structured_artifact(naive)

    bad_hash = _base("kie", {"fields": []})
    bad_hash["source_sha256"] = "../bad"
    with pytest.raises(ValidationError, match="source_sha256"):
        validate_structured_artifact(bad_hash)

    artifact = validate_structured_artifact(_base("kie", {"fields": []}))
    with pytest.raises(ValueError, match="exceeds"):
        canonical_artifact_bytes(artifact, max_bytes=10)


@pytest.mark.parametrize("value", ["data:text/html;base64,AAAA", "<script>alert(1)</script>"])
def test_rejects_embedded_resources_and_active_html(value: str) -> None:
    payload = _base("kie", {"fields": [{"name": "field", "value": value}]})

    with pytest.raises(ValidationError, match="embedded resources"):
        validate_structured_artifact(payload)


def test_artifact_key_is_revision_and_page_scoped() -> None:
    key = build_artifact_key(
        document_id=_DOC_ID,
        parse_revision=3,
        page_idx=7,
        artifact_type=ArtifactType.kie,
        artifact_id=_ARTIFACT_ID,
    )

    assert key == (
        "11111111-1111-1111-1111-111111111111/sidecars/r3/p000007/"
        "kie/22222222-2222-2222-2222-222222222222.json"
    )

    with pytest.raises(ValueError, match="page_idx"):
        build_artifact_key(
            document_id=_DOC_ID,
            parse_revision=3,
            page_idx=-1,
            artifact_type=ArtifactType.kie,
            artifact_id=_ARTIFACT_ID,
        )
