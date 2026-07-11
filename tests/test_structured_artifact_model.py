from __future__ import annotations

from rag_app.db.models import DocumentStructuredArtifact


def test_structured_artifact_table_contract() -> None:
    table = DocumentStructuredArtifact.__table__

    assert table.name == "document_structured_artifacts"
    assert table.c.document_id.foreign_keys
    assert next(iter(table.c.document_id.foreign_keys)).ondelete == "CASCADE"
    assert table.c.artifact_key.unique
    assert {"parse_revision", "page_idx", "artifact_type", "request_hash"} <= set(
        table.c.keys()
    )
    assert "ix_structured_artifact_lookup" in {index.name for index in table.indexes}


def test_structured_artifact_has_idempotency_constraint() -> None:
    names = {constraint.name for constraint in DocumentStructuredArtifact.__table__.constraints}

    assert "uq_structured_artifact_request" in names
    assert "ck_structured_artifact_status" in names
    assert "ck_structured_artifact_type" in names
    assert "ck_structured_ready_payload" in names
