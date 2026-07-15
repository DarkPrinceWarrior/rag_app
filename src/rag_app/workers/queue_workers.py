"""ARQ WorkerSettings для раздельных очередей; legacy worker остаётся drain."""

from __future__ import annotations

from arq.connections import RedisSettings

from rag_app.config import settings
from rag_app.workers.main import shutdown, startup
from rag_app.workers.memory_tasks import consolidate_memory, extract_memory
from rag_app.workers.queueing import instrument_job
from rag_app.workers.tasks import (
    describe_images,
    export_document,
    index_document,
    index_pages_visual,
    parse_document,
    render_original_view,
    translate_document,
    translate_to_language,
)


def _redis_settings() -> RedisSettings:
    return RedisSettings(
        host=settings.redis_host,
        port=settings.redis_port,
        database=settings.redis_db,
    )


class ParseWorkerSettings:
    cron_jobs: list = []
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = _redis_settings()
    job_timeout = settings.job_timeout_s
    keep_result = 3600
    max_tries = settings.queue_max_tries
    queue_name = settings.queue_parse_name
    functions = [instrument_job(parse_document, "parse")]
    max_jobs = 1


class TranslateWorkerSettings:
    cron_jobs: list = []
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = _redis_settings()
    job_timeout = settings.job_timeout_s
    keep_result = 3600
    max_tries = settings.queue_max_tries
    queue_name = settings.queue_translate_name
    functions = [
        instrument_job(translate_document, "translate"),
        instrument_job(translate_to_language, "translate"),
    ]
    max_jobs = 2


class ExportIndexWorkerSettings:
    cron_jobs: list = []
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = _redis_settings()
    job_timeout = settings.job_timeout_s
    keep_result = 3600
    max_tries = settings.queue_max_tries
    queue_name = settings.queue_export_index_name
    functions = [
        instrument_job(export_document, "export_index"),
        instrument_job(render_original_view, "export_index"),
        instrument_job(index_document, "export_index"),
        instrument_job(index_pages_visual, "export_index"),
        instrument_job(describe_images, "export_index"),
    ]
    max_jobs = 2


class MemoryWorkerSettings:
    cron_jobs: list = []
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = _redis_settings()
    job_timeout = settings.job_timeout_s
    keep_result = 3600
    max_tries = settings.queue_max_tries
    queue_name = settings.queue_memory_name
    functions = [
        instrument_job(extract_memory, "memory"),
        instrument_job(consolidate_memory, "memory"),
    ]
    max_jobs = 2
