from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from rag_app.config import settings
from rag_app.db.models import Base


def create_engine() -> AsyncEngine:
    return create_async_engine(settings.database_url, pool_pre_ping=True)


def create_sessionmaker(engine: AsyncEngine) -> async_sessionmaker:
    return async_sessionmaker(engine, expire_on_commit=False)


async def init_db(engine: AsyncEngine) -> None:
    """Создание схемы (этап 1; alembic — на этапе 2)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
