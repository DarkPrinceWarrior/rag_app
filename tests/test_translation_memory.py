from __future__ import annotations

import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.dialects import postgresql

from rag_app.config import Settings
from rag_app.db.models import TranslationMemory
from rag_app.pipeline.translation_memory import (
    TranslationMemoryScope,
    _scope_predicates,
    normalize_source,
    source_hash,
)


def test_normalization_is_unicode_and_whitespace_stable_but_case_sensitive() -> None:
    assert normalize_source("  Pressure\u00a0１６．５  MPa ") == "Pressure 16.5 MPa"
    assert source_hash("Pressure  16.5 MPa") == source_hash("Pressure\u00a016.5 MPa")
    assert source_hash("Pressure") != source_hash("pressure")


def test_scope_predicates_pin_owner_folder_project_and_domain() -> None:
    folder_id = uuid.UUID("10000000-0000-0000-0000-000000000001")
    scope = TranslationMemoryScope(
        owner_sub="owner-a",
        folder_id=folder_id,
        project="EPC-1",
        domain="oil-gas",
    )
    stmt = select(TranslationMemory).where(*_scope_predicates(scope))
    sql = str(
        stmt.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )

    assert "translation_memory.owner_sub = 'owner-a'" in sql
    assert str(folder_id) in sql
    assert "translation_memory.project = 'EPC-1'" in sql
    assert "translation_memory.domain = 'oil-gas'" in sql
    assert "translation_memory.status = 'approved'" in sql
    assert "translation_memory.revoked_at IS NULL" in sql


def test_translation_memory_rollout_defaults_off() -> None:
    configured = Settings(_env_file=None)

    assert configured.translation_memory_mode == "off"
    assert configured.translation_memory_nearest_top_k == 2


def test_translation_memory_migration_is_rls_and_revocation_safe() -> None:
    migration = (
        Path(__file__).parents[1] / "alembic/versions/0027_translation_memory.py"
    ).read_text("utf-8")

    assert "FORCE ROW LEVEL SECURITY" in migration
    assert "translation_memory_owner" in migration
    assert "uq_translation_memory_active_exact" in migration
    assert "status = 'approved' AND revoked_at IS NULL" in migration
    assert 'sa.ForeignKey("folders.id", ondelete="CASCADE")' in migration
    assert "rag_api" in migration and "rag_worker" in migration
