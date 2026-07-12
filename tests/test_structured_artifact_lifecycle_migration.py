from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any


class _Operations:
    def __init__(self) -> None:
        self.columns: list[str] = []
        self.altered: list[str] = []
        self.constraints: dict[str, str] = {}
        self.indexes: dict[str, tuple[str, ...]] = {}
        self.sql: list[str] = []

    def add_column(self, _table: str, column: Any) -> None:
        self.columns.append(column.name)

    def alter_column(self, _table: str, column: str, **_kwargs: Any) -> None:
        self.altered.append(column)

    def create_check_constraint(self, name: str, _table: str, condition: str) -> None:
        self.constraints[name] = condition

    def create_index(self, name: str, _table: str, columns: list[str]) -> None:
        self.indexes[name] = tuple(columns)

    def execute(self, statement: Any) -> None:
        self.sql.append(str(statement))


def _load_migration() -> ModuleType:
    path = (
        Path(__file__).parents[1]
        / "alembic"
        / "versions"
        / "0024_structured_artifact_lifecycle.py"
    )
    spec = importlib.util.spec_from_file_location("migration_0024", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_migration_adds_durable_request_and_lease_contract() -> None:
    migration = _load_migration()
    operations = _Operations()
    migration.op = operations

    migration.upgrade()

    assert migration.revision == "0024"
    assert migration.down_revision == "0023"
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
    } == set(operations.columns)
    assert {
        "request_schema",
        "schema_sha256",
        "model_revision",
        "protocol_version",
        "request_options",
        "source_key",
        "attempt_count",
        "max_attempts",
    } == set(operations.altered)
    assert operations.indexes["ix_structured_artifact_sweep"] == (
        "status",
        "next_attempt_at",
        "lease_expires_at",
    )
    assert "claim_token IS NOT NULL" in operations.constraints["ck_structured_running_lease"]
    assert "attempt_count <= max_attempts" in operations.constraints["ck_structured_attempt_bounds"]
    assert "finished_at IS NOT NULL" in operations.constraints["ck_structured_finished_status"]


def test_migration_fails_closed_for_preexisting_0023_rows() -> None:
    migration = _load_migration()
    operations = _Operations()
    migration.op = operations

    migration.upgrade()

    backfill = " ".join(operations.sql)
    assert "UPDATE document_structured_artifacts" in backfill
    assert "status = 'superseded'" in backfill
    assert "claim_token = NULL" in backfill
    assert "lease_expires_at = NULL" in backfill
    assert "finished_at = now()" in backfill
