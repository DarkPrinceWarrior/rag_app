"""Визуальная переиндексация всех сканов (§ 12.1 шаг 4).

Запуск: uv run python scripts/reindex_visual.py
"""

from __future__ import annotations

import asyncio
import uuid

from arq import create_pool
from arq.connections import RedisSettings
from sqlalchemy import select

from rag_app.config import settings
from rag_app.db.engine import create_engine, create_sessionmaker
from rag_app.db.models import Document, DocumentKind, DocumentStatus


async def main() -> None:
    engine = create_engine()
    sessionmaker = create_sessionmaker(engine)
    pool = await create_pool(
        RedisSettings(host=settings.redis_host, port=settings.redis_port, database=settings.redis_db)
    )
    async with sessionmaker() as session:
        docs = (
            (
                await session.execute(
                    select(Document).where(
                        Document.status == DocumentStatus.done,
                        Document.kind == DocumentKind.pdf_scan.value,
                    )
                )
            )
            .scalars()
            .all()
        )
    for doc in docs:
        await pool.enqueue_job(
            "index_pages_visual", str(doc.id), _job_id=f"vindex:{doc.id}:{uuid.uuid4().hex[:8]}"
        )
        print(f"queued: {doc.filename} ({doc.id})")
    await engine.dispose()
    print(f"сканов: {len(docs)}")


if __name__ == "__main__":
    asyncio.run(main())
