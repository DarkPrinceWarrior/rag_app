from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import select
from sqlalchemy.dialects import postgresql

from rag_app.config import Settings
from rag_app.db.models import TranslationMemory
from rag_app.pipeline.translation_memory import (
    TranslationMemoryScope,
    TranslationMemoryService,
    _scope_predicates,
    normalize_source,
    source_hash,
)


class _Rows:
    def __init__(self, rows: list[object]) -> None:
        self._rows = rows

    def scalars(self) -> _Rows:
        return self

    def all(self) -> list[object]:
        return self._rows


class _Session:
    def __init__(self, responses: list[list[object]]) -> None:
        self._responses = responses

    async def __aenter__(self) -> _Session:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def execute(self, _statement: object) -> _Rows:
        return _Rows(self._responses.pop(0))


class _SessionMaker:
    def __init__(self, sessions: list[_Session]) -> None:
        self._sessions = sessions

    def __call__(self) -> _Session:
        return self._sessions.pop(0)


class _Embedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str], batch: int | None = None) -> list[list[float]]:
        self.calls.append(texts)
        return [[0.0] * 1024 for _ in texts]


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


def test_exact_lookup_is_scoped_and_bypasses_embedding() -> None:
    text = "Pressure 16.5 MPa"
    row = SimpleNamespace(
        id=uuid.UUID("10000000-0000-0000-0000-000000000001"),
        source_hash=source_hash(text),
        source_normalized=normalize_source(text),
        source_text=text,
        approved_translation="Давление 16.5 MPa",
    )
    embedder = _Embedder()
    service = TranslationMemoryService(
        _SessionMaker([_Session([[row]])]),  # type: ignore[arg-type]
        embedder,
    )

    result = asyncio.run(
        service.lookup_batch(
            [text, text, "   "],
            source_lang="en",
            target_lang="ru",
            scope=TranslationMemoryScope(
                owner_sub="owner-a",
                folder_id=None,
                project=None,
            ),
        )
    )

    assert list(result) == [text]
    assert result[text].exact is not None
    assert result[text].exact.translation == "Давление 16.5 MPa"
    assert result[text].nearest == ()
    assert embedder.calls == []


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
