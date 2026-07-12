"""Изолированный ARQ worker для bounded structured KIE.

Запуск: uv run arq rag_app.workers.structured_sidecar.WorkerSettings

Feature flag по умолчанию выключен; очередь не пересекается с основным worker.
"""

from __future__ import annotations

import logging

from arq.connections import RedisSettings

from rag_app.config import settings
from rag_app.db.engine import create_engine, create_sessionmaker
from rag_app.llm.structured import GraniteStructuredClient
from rag_app.storage.s3 import Storage
from rag_app.workers.structured_kie import run_structured_kie

logger = logging.getLogger(__name__)


async def startup(ctx: dict) -> None:
    engine = create_engine()
    storage = Storage()
    ctx["engine"] = engine
    ctx["sessionmaker"] = create_sessionmaker(engine)
    ctx["storage"] = storage
    await storage.ensure_buckets()
    if settings.structured_extraction_enabled:
        ctx["structured_client"] = GraniteStructuredClient(
            base_url=settings.structured_model_base_url,
            api_key=settings.structured_model_api_key,
            model=settings.structured_model_name,
            timeout_s=settings.structured_model_timeout_s,
        )
    logger.info("structured sidecar worker ready (queue=%s)", settings.structured_sidecar_queue_name)


async def shutdown(ctx: dict) -> None:
    client = ctx.get("structured_client")
    if client is not None:
        await client.close()
    engine = ctx.get("engine")
    if engine is not None:
        await engine.dispose()


async def health_probe(ctx: dict) -> dict[str, str]:
    """Минимальная зарегистрированная задача для валидного ARQ worker."""
    return {"status": "ok", "queue": settings.structured_sidecar_queue_name}


class WorkerSettings:
    functions = [health_probe, run_structured_kie]
    cron_jobs: list = []
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings(
        host=settings.redis_host,
        port=settings.redis_port,
        database=settings.redis_db,
    )
    queue_name = settings.structured_sidecar_queue_name
    health_check_interval = settings.structured_sidecar_health_check_interval_s
    health_check_key = f"{queue_name}:health-check"
    job_timeout = settings.parser_sidecar_timeout_s
    max_jobs = 1
    max_tries = settings.structured_job_max_attempts
    keep_result = 3600
