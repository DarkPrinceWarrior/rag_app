"""Fail-closed RLS for every user-scoped API table.

Revision ID: 0025
Revises: 0024
Create Date: 2026-07-13
"""

from __future__ import annotations

from alembic import op

revision = "0025"
down_revision = "0024"
branch_labels = None
depends_on = None

_DIRECT_OWNER = {
    "documents": "owner_sub",
    "folders": "owner_sub",
    "chat_sessions": "owner_sub",
    "audit_log": "user_sub",
}
_VIA_DOCUMENT = (
    "chunks",
    "segments",
    "document_translations",
    "segment_versions",
    "page_embeddings",
    "document_structured_artifacts",
)
_VIA_CHAT_SESSION = ("chat_messages",)
_NEWLY_PROTECTED = (
    "page_embeddings",
    "chat_sessions",
    "chat_messages",
    "audit_log",
    "memory_item_sources",
)
_ALL_TABLES = (
    *_DIRECT_OWNER,
    *_VIA_DOCUMENT,
    *_VIA_CHAT_SESSION,
    "memory_item_sources",
)

_ADMIN = "current_setting('app.is_admin', true) = 'on'"
_USER = "current_setting('app.user_id', true)"
_TENANT_UUID = "nullif(current_setting('app.tenant_id', true), '')::uuid"


def _drop_policy(table: str) -> None:
    op.execute(f"DROP POLICY IF EXISTS {table}_owner ON {table}")


def _create_policy(table: str, predicate: str) -> None:
    op.execute(
        f"CREATE POLICY {table}_owner ON {table} "
        f"USING ({predicate}) WITH CHECK ({predicate})"
    )


def _document_predicate(table: str, *, allow_null_owner: bool = False) -> str:
    owner = (
        f"(d.owner_sub IS NULL OR d.owner_sub = {_USER})"
        if allow_null_owner
        else f"d.owner_sub = {_USER}"
    )
    return (
        f"{_ADMIN} OR EXISTS (SELECT 1 FROM documents d "
        f"WHERE d.id = {table}.document_id AND {owner})"
    )


def _grant_if_role_exists(role: str) -> None:
    tables = ", ".join(_ALL_TABLES)
    op.execute(
        f"""
        DO $grant$
        BEGIN
          IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{role}') THEN
            EXECUTE 'GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE {tables} TO {role}';
          END IF;
        END
        $grant$;
        """
    )


def upgrade() -> None:
    # The lifecycle table may have been created by an application role on an
    # installation where migrations were not run as the canonical owner.
    op.execute(
        """
        DO $owner$
        BEGIN
          IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rag') THEN
            EXECUTE 'ALTER TABLE document_structured_artifacts OWNER TO rag';
          END IF;
        END
        $owner$;
        """
    )

    op.execute(
        """
        DO $preflight$
        DECLARE
          mismatches bigint;
        BEGIN
          SELECT
            (SELECT count(*) FROM documents d JOIN folders f ON f.id = d.folder_id
             WHERE d.owner_sub IS DISTINCT FROM f.owner_sub)
            + (SELECT count(*) FROM chat_sessions cs JOIN documents d ON d.id = cs.document_id
               WHERE cs.owner_sub IS DISTINCT FROM d.owner_sub)
            + (SELECT count(*) FROM chat_sessions cs JOIN folders f ON f.id = cs.folder_id
               WHERE cs.owner_sub IS DISTINCT FROM f.owner_sub)
          INTO mismatches;
          IF mismatches > 0 THEN
            RAISE EXCEPTION
              'RLS preflight failed: % cross-owner document/folder/chat links', mismatches;
          END IF;
        END
        $preflight$;
        """
    )

    # Legacy NULL owners were inventoried before deployment.  Refuse the
    # migration instead of silently turning unowned rows into shared data.
    for table in ("documents", "folders", "chat_sessions"):
        op.execute(f"ALTER TABLE {table} ALTER COLUMN owner_sub SET NOT NULL")

    for table in _ALL_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        _drop_policy(table)

    _create_policy(
        "documents",
        (
            f"{_ADMIN} OR (owner_sub = {_USER} "
            "AND (folder_id IS NULL OR EXISTS (SELECT 1 FROM folders f "
            f"WHERE f.id = documents.folder_id AND f.owner_sub = {_USER})))"
        ),
    )
    _create_policy("folders", f"{_ADMIN} OR owner_sub = {_USER}")
    _create_policy(
        "chat_sessions",
        (
            f"{_ADMIN} OR (owner_sub = {_USER} "
            "AND (document_id IS NULL OR EXISTS (SELECT 1 FROM documents d "
            f"WHERE d.id = chat_sessions.document_id AND d.owner_sub = {_USER})) "
            "AND (folder_id IS NULL OR EXISTS (SELECT 1 FROM folders f "
            f"WHERE f.id = chat_sessions.folder_id AND f.owner_sub = {_USER})))"
        ),
    )
    _create_policy("audit_log", f"{_ADMIN} OR user_sub = {_USER}")

    for table in _VIA_DOCUMENT:
        if table == "segment_versions":
            _create_policy(
                table,
                (
                    f"{_ADMIN} OR EXISTS (SELECT 1 FROM documents d "
                    "JOIN segments s ON s.id = segment_versions.segment_id "
                    "WHERE d.id = segment_versions.document_id "
                    "AND s.document_id = d.id "
                    f"AND d.owner_sub = {_USER})"
                ),
            )
        else:
            _create_policy(table, _document_predicate(table))

    for table in _VIA_CHAT_SESSION:
        _create_policy(
            table,
            (
                f"{_ADMIN} OR EXISTS (SELECT 1 FROM chat_sessions cs "
                f"WHERE cs.id = {table}.session_id AND cs.owner_sub = {_USER})"
            ),
        )

    _create_policy(
        "memory_item_sources",
        (
            "EXISTS (SELECT 1 FROM memory_items mi "
            "JOIN memory_events me ON me.id = memory_item_sources.event_id "
            "WHERE mi.id = memory_item_sources.item_id "
            f"AND mi.tenant_id = {_TENANT_UUID} AND me.tenant_id = {_TENANT_UUID} "
            f"AND mi.user_id = {_USER} AND me.user_id = {_USER})"
        ),
    )

    # System audit rows have user_id=NULL.  They are visible only to an
    # authenticated admin in the explicitly selected tenant, never to users.
    op.execute("DROP POLICY IF EXISTS p_memory_audit_log_scope ON memory_audit_log")
    op.execute(
        "CREATE POLICY p_memory_audit_log_scope ON memory_audit_log "
        f"USING (tenant_id = {_TENANT_UUID} AND ({_ADMIN} OR user_id = {_USER})) "
        f"WITH CHECK (tenant_id = {_TENANT_UUID} AND ({_ADMIN} OR user_id = {_USER}))"
    )

    _grant_if_role_exists("rag_api")
    _grant_if_role_exists("rag_worker")
    op.execute(
        """
        DO $audit_grant$
        BEGIN
          IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rag_api') THEN
            REVOKE UPDATE, DELETE ON TABLE audit_log FROM rag_api;
          END IF;
        END
        $audit_grant$;
        """
    )


def downgrade() -> None:
    for table in _ALL_TABLES:
        op.execute(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY")
        _drop_policy(table)

    # Restore the permissive policies from 0020 for compatibility with legacy
    # rows whose owner_sub is NULL.
    for table, owner_column in (("documents", "owner_sub"), ("folders", "owner_sub")):
        predicate = f"{_ADMIN} OR {owner_column} IS NULL OR {owner_column} = {_USER}"
        _create_policy(table, predicate)

    for table in ("chunks", "segments", "document_translations", "segment_versions"):
        _create_policy(table, _document_predicate(table, allow_null_owner=True))

    # Restore the policy introduced by 0023. Ownership remains normalized to
    # rag because the previous owner cannot be reconstructed reliably.
    _create_policy(
        "document_structured_artifacts",
        _document_predicate("document_structured_artifacts", allow_null_owner=True),
    )

    op.execute("DROP POLICY IF EXISTS p_memory_audit_log_scope ON memory_audit_log")
    op.execute(
        "CREATE POLICY p_memory_audit_log_scope ON memory_audit_log "
        f"USING (tenant_id = {_TENANT_UUID} "
        f"AND (user_id IS NULL OR user_id = {_USER})) "
        f"WITH CHECK (tenant_id = {_TENANT_UUID} "
        f"AND (user_id IS NULL OR user_id = {_USER}))"
    )

    for table in _NEWLY_PROTECTED:
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")

    for table in ("documents", "folders", "chat_sessions"):
        op.execute(f"ALTER TABLE {table} ALTER COLUMN owner_sub DROP NOT NULL")
