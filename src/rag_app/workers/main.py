"""ARQ-воркер. Запуск: uv run arq rag_app.workers.main.WorkerSettings

Этап 1 — один пул на все задачи; разделение на parse/translate/index-пулы
(roadmap § 6) — после появления OCR-ветки.
"""

from __future__ import annotations

import logging

from arq import cron
from arq.connections import RedisSettings
from openai import AsyncOpenAI

from rag_app.config import settings
from rag_app.db.engine import create_engine, create_sessionmaker
from rag_app.llm.client import Translator
from rag_app.llm.embeddings import Embedder, Reranker
from rag_app.llm.visual import VisualEmbedder
from rag_app.rag.memory import MemoryService
from rag_app.storage.s3 import Storage
from rag_app.workers.memory_tasks import consolidate_memory, extract_memory
from rag_app.workers.tasks import (
    describe_images,
    export_document,
    index_document,
    index_pages_visual,
    parse_document,
    render_original_view,
    translate_document,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


async def startup(ctx: dict) -> None:
    ctx["engine"] = create_engine()
    ctx["sessionmaker"] = create_sessionmaker(ctx["engine"])
    ctx["storage"] = Storage()
    await ctx["storage"].ensure_buckets()
    ctx["translator"] = Translator()
    ctx["visual"] = VisualEmbedder()
    ctx["embedder"] = Embedder()
    # слой памяти (Этап 2): экстракция/consolidation на тех же embedder/reranker + LLM
    ctx["reranker"] = Reranker()
    ctx["memory"] = MemoryService(ctx["embedder"], ctx["reranker"])
    ctx["llm"] = AsyncOpenAI(
        base_url=settings.llm_base_url, api_key=settings.llm_api_key, timeout=120.0
    )


async def shutdown(ctx: dict) -> None:
    await ctx["engine"].dispose()


class WorkerSettings:
    functions = [
        parse_document,
        translate_document,
        export_document,
        render_original_view,
        index_document,
        index_pages_visual,
        describe_images,
        extract_memory,
        consolidate_memory,
    ]
    # consolidation памяти раз в полчаса (auto-accept + позже purge)
    cron_jobs = [cron(consolidate_memory, minute={0, 30}, run_at_startup=False)]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings(
        host=settings.redis_host, port=settings.redis_port, database=settings.redis_db
    )
    job_timeout = settings.job_timeout_s
    max_jobs = 4  # parse — GPU-bound; параллелизм перевода живёт внутри задачи
    keep_result = 3600
