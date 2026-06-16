"""Очередь кандидатов памяти (docs/MEMORY_rev4_mem0_articles.md §3.4, §15.3):
автоэкстрактор пишет сюда, в боевую память (memory_items) попадает только после
accept/auto_accept через consolidation.

Revision ID: 0011
Revises: 0010
Create Date: 2026-06-16
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "memory_candidates",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", sa.String(64), nullable=False),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("thread_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("action", sa.Text(), nullable=False),
        sa.Column(
            "target_item_id", postgresql.UUID(as_uuid=True),
            sa.ForeignKey("memory_items.id"), nullable=True,
        ),
        sa.Column("proposed", postgresql.JSONB(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("rationale", sa.Text(), nullable=True),
        sa.Column("fingerprint", sa.Text(), nullable=True),
        sa.Column("memory_provider", sa.Text(), nullable=False, server_default="internal"),
        sa.Column("external_memory_id", sa.Text(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False, server_default="pending"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("decided_by", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "action IN ('create','update','delete','supersede')", name="chk_cand_action"
        ),
        sa.CheckConstraint(
            "status IN ('pending','accepted','rejected','auto_accepted')", name="chk_cand_status"
        ),
    )
    op.create_index("idx_cand_pending", "memory_candidates", ["tenant_id", "user_id", "status"])


def downgrade() -> None:
    op.drop_table("memory_candidates")
