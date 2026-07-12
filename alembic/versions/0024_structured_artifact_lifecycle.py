"""Durable claim/publish lifecycle for structured sidecars.

Revision ID: 0024
Revises: 0023
Create Date: 2026-07-12
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0024"
down_revision = "0023"
branch_labels = None
depends_on = None

_LEGACY_SCHEMA_SHA256 = "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"


def upgrade() -> None:
    # Add nullable columns first so an already-populated 0023 table can be
    # upgraded without exposing incomplete rows to the worker.
    op.add_column(
        "document_structured_artifacts",
        sa.Column("request_schema", postgresql.JSONB(), nullable=True),
    )
    op.add_column(
        "document_structured_artifacts",
        sa.Column("schema_sha256", sa.String(64), nullable=True),
    )
    op.add_column(
        "document_structured_artifacts",
        sa.Column("model_revision", sa.String(128), nullable=True),
    )
    op.add_column(
        "document_structured_artifacts",
        sa.Column("protocol_version", sa.String(32), nullable=True),
    )
    op.add_column(
        "document_structured_artifacts",
        sa.Column("request_options", postgresql.JSONB(), nullable=True),
    )
    op.add_column(
        "document_structured_artifacts",
        sa.Column("source_key", sa.String(1024), nullable=True),
    )
    op.add_column(
        "document_structured_artifacts",
        sa.Column("attempt_count", sa.Integer(), nullable=True, server_default="0"),
    )
    op.add_column(
        "document_structured_artifacts",
        sa.Column("max_attempts", sa.Integer(), nullable=True, server_default="3"),
    )
    op.add_column(
        "document_structured_artifacts",
        sa.Column("claim_token", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "document_structured_artifacts",
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "document_structured_artifacts",
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "document_structured_artifacts",
        sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "document_structured_artifacts",
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )

    # 0023 never ran a model job. Fail closed if an installation nevertheless
    # contains rows: retain them for audit, but make every legacy row terminal.
    op.execute(
        sa.text(
            "UPDATE document_structured_artifacts SET "
            "request_schema = '{}'::jsonb, "
            "schema_sha256 = :schema_sha256, "
            "model_revision = 'legacy-unknown', "
            "protocol_version = 'legacy-unknown', "
            "request_options = '{}'::jsonb, "
            "source_key = 'legacy/unavailable', "
            "attempt_count = 0, max_attempts = 3, "
            "status = 'superseded', claim_token = NULL, "
            "claimed_at = NULL, lease_expires_at = NULL, next_attempt_at = NULL, "
            "finished_at = now(), updated_at = now()"
        ).bindparams(schema_sha256=_LEGACY_SCHEMA_SHA256)
    )

    for column in (
        "request_schema",
        "schema_sha256",
        "model_revision",
        "protocol_version",
        "request_options",
        "source_key",
        "attempt_count",
        "max_attempts",
    ):
        op.alter_column(
            "document_structured_artifacts",
            column,
            existing_nullable=True,
            nullable=False,
        )

    op.create_check_constraint(
        "ck_structured_schema_hash",
        "document_structured_artifacts",
        "schema_sha256 ~ '^[0-9a-f]{64}$'",
    )
    op.create_check_constraint(
        "ck_structured_request_schema_object",
        "document_structured_artifacts",
        "jsonb_typeof(request_schema) = 'object'",
    )
    op.create_check_constraint(
        "ck_structured_request_options_object",
        "document_structured_artifacts",
        "jsonb_typeof(request_options) = 'object'",
    )
    op.create_check_constraint(
        "ck_structured_request_identity",
        "document_structured_artifacts",
        "length(model_revision) > 0 AND length(protocol_version) > 0 AND length(source_key) > 0",
    )
    op.create_check_constraint(
        "ck_structured_attempt_bounds",
        "document_structured_artifacts",
        "attempt_count >= 0 AND max_attempts BETWEEN 1 AND 10 AND attempt_count <= max_attempts",
    )
    op.create_check_constraint(
        "ck_structured_running_lease",
        "document_structured_artifacts",
        "status <> 'running' OR (claim_token IS NOT NULL AND claimed_at IS NOT NULL "
        "AND lease_expires_at IS NOT NULL AND lease_expires_at > claimed_at)",
    )
    op.create_check_constraint(
        "ck_structured_nonrunning_claim",
        "document_structured_artifacts",
        "status = 'running' OR (claim_token IS NULL AND lease_expires_at IS NULL)",
    )
    op.create_check_constraint(
        "ck_structured_next_attempt_status",
        "document_structured_artifacts",
        "next_attempt_at IS NULL OR status = 'queued'",
    )
    op.create_check_constraint(
        "ck_structured_finished_status",
        "document_structured_artifacts",
        "((status IN ('ready', 'error', 'superseded')) = (finished_at IS NOT NULL))",
    )
    op.create_index(
        "ix_structured_artifact_sweep",
        "document_structured_artifacts",
        ["status", "next_attempt_at", "lease_expires_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_structured_artifact_sweep", table_name="document_structured_artifacts")
    for constraint in (
        "ck_structured_finished_status",
        "ck_structured_next_attempt_status",
        "ck_structured_nonrunning_claim",
        "ck_structured_running_lease",
        "ck_structured_attempt_bounds",
        "ck_structured_request_identity",
        "ck_structured_request_options_object",
        "ck_structured_request_schema_object",
        "ck_structured_schema_hash",
    ):
        op.drop_constraint(
            constraint,
            "document_structured_artifacts",
            type_="check",
        )
    for column in (
        "finished_at",
        "next_attempt_at",
        "lease_expires_at",
        "claimed_at",
        "claim_token",
        "max_attempts",
        "attempt_count",
        "source_key",
        "request_options",
        "protocol_version",
        "model_revision",
        "schema_sha256",
        "request_schema",
    ):
        op.drop_column("document_structured_artifacts", column)
