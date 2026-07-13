from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


class _Operations:
    def __init__(self) -> None:
        self.sql: list[str] = []

    def execute(self, statement: str) -> None:
        self.sql.append(" ".join(str(statement).split()))


def _load_migration() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / "alembic"
        / "versions"
        / "0025_rls_fail_closed.py"
    )
    spec = importlib.util.spec_from_file_location("migration_0025_rls_fail_closed", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_upgrade_installs_strict_forced_policies() -> None:
    migration = _load_migration()
    operations = _Operations()
    migration.op = operations

    migration.upgrade()

    assert migration.revision == "0025"
    assert migration.down_revision == "0024"

    upgrade_sql = "\n".join(operations.sql)
    for table in migration._ALL_TABLES:
        assert f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY" in upgrade_sql
        assert f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY" in upgrade_sql
        assert f"CREATE POLICY {table}_owner ON {table}" in upgrade_sql

    policy_sql = [statement for statement in operations.sql if statement.startswith("CREATE POLICY")]
    assert len(policy_sql) == len(migration._ALL_TABLES) + 1
    assert all(" USING (" in statement and ") WITH CHECK (" in statement for statement in policy_sql)
    assert all("owner_sub IS NULL" not in statement for statement in policy_sql)
    assert all("user_sub IS NULL" not in statement for statement in policy_sql)
    assert "ALTER TABLE documents ALTER COLUMN owner_sub SET NOT NULL" in upgrade_sql
    assert "ALTER TABLE folders ALTER COLUMN owner_sub SET NOT NULL" in upgrade_sql
    assert "ALTER TABLE chat_sessions ALTER COLUMN owner_sub SET NOT NULL" in upgrade_sql
    assert "CREATE POLICY p_memory_audit_log_scope" in upgrade_sql
    assert "user_id IS NULL" not in upgrade_sql
    assert "RLS preflight failed" in upgrade_sql
    assert "IS DISTINCT FROM" in upgrade_sql
    assert "sv.document_id IS DISTINCT FROM s.document_id" in upgrade_sql
    for table in (
        "memory_events",
        "memory_items",
        "memory_candidates",
        "memory_audit_log",
    ):
        assert table in migration._GRANT_TABLES
    assert "GRANT USAGE, SELECT ON SEQUENCE memory_audit_log_id_seq" in upgrade_sql


def test_upgrade_covers_new_user_scoped_tables() -> None:
    migration = _load_migration()
    operations = _Operations()
    migration.op = operations

    migration.upgrade()
    policies = {
        statement.split()[2]: statement
        for statement in operations.sql
        if statement.startswith("CREATE POLICY")
    }
    policy_sql = "\n".join(policies.values())

    assert "CREATE POLICY page_embeddings_owner" in policy_sql
    assert "page_embeddings.document_id" in policy_sql
    assert "owner_sub = current_setting('app.user_id', true)" in policies["chat_sessions_owner"]
    assert "CREATE POLICY chat_messages_owner" in policy_sql
    assert "chat_sessions cs" in policy_sql
    assert "cs.owner_sub = current_setting('app.user_id', true)" in policy_sql
    assert "user_sub = current_setting('app.user_id', true)" in policies["audit_log_owner"]
    assert "CREATE POLICY memory_item_sources_owner" in policy_sql
    assert "mi.user_id = current_setting('app.user_id', true)" in policy_sql
    assert "me.user_id = current_setting('app.user_id', true)" in policy_sql
    assert "JOIN segments s ON s.id = segment_versions.segment_id" in policies[
        "segment_versions_owner"
    ]
    assert "s.document_id = d.id" in policies["segment_versions_owner"]


def test_document_policies_do_not_reintroduce_legacy_null_access() -> None:
    migration = _load_migration()

    for table in migration._VIA_DOCUMENT:
        predicate = migration._document_predicate(table)
        assert "owner_sub IS NULL" not in predicate
        assert f"d.id = {table}.document_id" in predicate
        assert "d.owner_sub = current_setting('app.user_id', true)" in predicate
