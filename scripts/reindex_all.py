"""Переиндексация RAG-чанков для всех завершённых документов.

Запуск: uv run python scripts/reindex_all.py
"""

from __future__ import annotations

import asyncio
import uuid

from arq import create_pool
from arq.connections import RedisSettings
from sqlalchemy import select

from rag_app.config import settings
from rag_app.db.engine import create_engine, create_sessionmaker
from rag_app.db.models import Document, DocumentStatus


async def main() -> None:
    engine = create_engine()
    sessionmaker = create_sessionmaker(engine)
    pool = await create_pool(
        RedisSettings(host=settings.redis_host, port=settings.redis_port, database=settings.redis_db)
    )
    async with sessionmaker() as session:
        docs = (
            (await session.execute(select(Document).where(Document.status == DocumentStatus.done)))
            .scalars()
            .all()
        )
    for doc in docs:
        await pool.enqueue_job(
            "index_document", str(doc.id), _job_id=f"index:{doc.id}:{uuid.uuid4().hex[:8]}"
        )
        print(f"queued: {doc.filename} ({doc.id})")
    await engine.dispose()
    print(f"всего: {len(docs)}")


if __name__ == "__main__":
    asyncio.run(main())
