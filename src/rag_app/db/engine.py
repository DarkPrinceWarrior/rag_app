from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from rag_app.config import settings
from rag_app.db import rls as _rls  # noqa: F401 — регистрирует after_begin RLS-GUC хук


def create_engine() -> AsyncEngine:
    return create_async_engine(settings.database_url, pool_pre_ping=True)


def create_sessionmaker(engine: AsyncEngine) -> async_sessionmaker:
    return async_sessionmaker(engine, expire_on_commit=False)


# Схемой управляет alembic: `uv run alembic upgrade head` (см. alembic/).
