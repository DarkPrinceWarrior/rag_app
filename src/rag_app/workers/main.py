"""ARQ-воркер. Запуск: uv run arq rag_app.workers.main.WorkerSettings

Этап 1 — один пул на все задачи; разделение на parse/translate/index-пулы
(roadmap § 6) — после появления OCR-ветки.
"""

from __future__ import annotations

import logging

from arq.connections import RedisSettings

from rag_app.config import settings
from rag_app.db.engine import create_engine, create_sessionmaker, init_db
from rag_app.llm.client import Translator
from rag_app.storage.s3 import Storage
from rag_app.workers.tasks import export_document, parse_document, translate_document

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


async def startup(ctx: dict) -> None:
    ctx["engine"] = create_engine()
    ctx["sessionmaker"] = create_sessionmaker(ctx["engine"])
    await init_db(ctx["engine"])
    ctx["storage"] = Storage()
    await ctx["storage"].ensure_buckets()
    ctx["translator"] = Translator()


async def shutdown(ctx: dict) -> None:
    await ctx["engine"].dispose()


class WorkerSettings:
    functions = [parse_document, translate_document, export_document]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings(
        host=settings.redis_host, port=settings.redis_port, database=settings.redis_db
    )
    job_timeout = settings.job_timeout_s
    max_jobs = 4  # parse — GPU-bound; параллелизм перевода живёт внутри задачи
    keep_result = 3600
