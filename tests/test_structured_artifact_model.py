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
    assert {
        "request_schema",
        "schema_sha256",
        "model_revision",
        "protocol_version",
        "request_options",
        "source_key",
        "attempt_count",
        "max_attempts",
        "claim_token",
        "claimed_at",
        "lease_expires_at",
        "next_attempt_at",
        "finished_at",
    } <= set(table.c.keys())
    assert "ix_structured_artifact_lookup" in {index.name for index in table.indexes}
    assert "ix_structured_artifact_sweep" in {index.name for index in table.indexes}


def test_structured_artifact_has_idempotency_constraint() -> None:
    names = {constraint.name for constraint in DocumentStructuredArtifact.__table__.constraints}

    assert "uq_structured_artifact_request" in names
    assert "ck_structured_artifact_status" in names
    assert "ck_structured_artifact_type" in names
    assert "ck_structured_ready_payload" in names
    assert {
        "ck_structured_schema_hash",
        "ck_structured_attempt_bounds",
        "ck_structured_running_lease",
        "ck_structured_nonrunning_claim",
        "ck_structured_next_attempt_status",
        "ck_structured_finished_status",
    } <= names
