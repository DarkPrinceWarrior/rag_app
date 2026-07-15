"""Совместимая маршрутизация ARQ, bounded retry, DLQ и Redis-метрики."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
import uuid
from collections.abc import Callable
from functools import wraps
from typing import Any, Literal

import httpx
from arq import Retry
from arq.connections import ArqRedis
from arq.worker import Function, func
from sqlalchemy import select

from rag_app.config import settings
from rag_app.db.models import Segment

logger = logging.getLogger(__name__)

QueueStage = Literal["parse", "translate", "export_index", "memory"]

LEGACY_QUEUE = "arq:queue"
ENQUEUED_HASH = "rag:metrics:arq:enqueued"
COMPLETED_HASH = "rag:metrics:arq:completed"
RETRY_HASH = "rag:metrics:arq:retry"
ERROR_HASH = "rag:metrics:arq:error"
DURATION_BUCKET_HASH = "rag:metrics:arq:duration_bucket"
DURATION_SUM_HASH = "rag:metrics:arq:duration_sum"
DURATION_COUNT_HASH = "rag:metrics:arq:duration_count"
NUMBER_GUARD_HASH = "rag:metrics:translation:number_guard"
DURATION_BUCKETS = (1.0, 5.0, 15.0, 60.0, 300.0, 900.0, 1800.0, 3600.0, math.inf)

JOB_STAGES: dict[str, QueueStage] = {
    "parse_document": "parse",
    "translate_document": "translate",
    "translate_to_language": "translate",
    "export_document": "export_index",
    "render_original_view": "export_index",
    "index_document": "export_index",
    "index_pages_visual": "export_index",
    "describe_images": "export_index",
    "extract_memory": "memory",
    "consolidate_memory": "memory",
}


def queue_name(stage: QueueStage) -> str:
    return {
        "parse": settings.queue_parse_name,
        "translate": settings.queue_translate_name,
        "export_index": settings.queue_export_index_name,
        "memory": settings.queue_memory_name,
    }[stage]


def all_queue_names() -> tuple[str, ...]:
    return (
        LEGACY_QUEUE,
        settings.queue_parse_name,
        settings.queue_translate_name,
        settings.queue_export_index_name,
        settings.queue_memory_name,
    )


def route_queue(function: str, mode: Literal["legacy", "split"]) -> str:
    stage = JOB_STAGES.get(function)
    if mode == "split" and stage is not None:
        return queue_name(stage)
    return LEGACY_QUEUE


def _field(*parts: object) -> str:
    return "|".join(str(part) for part in parts)


async def _increment(redis: Any, key: str, field: str, amount: int = 1) -> None:
    try:
        await redis.hincrby(key, field, amount)
    except Exception:
        logger.warning("queue metric increment failed", exc_info=True)


async def record_duration(redis: Any, stage: QueueStage, queue: str, seconds: float) -> None:
    seconds = max(float(seconds), 0.0)
    label = _field(stage, queue)
    try:
        await redis.hincrbyfloat(DURATION_SUM_HASH, label, seconds)
    except Exception:
        logger.warning("queue duration sum failed", exc_info=True)
    await _increment(redis, DURATION_COUNT_HASH, label)
    for upper in DURATION_BUCKETS:
        if seconds <= upper:
            le = "+Inf" if math.isinf(upper) else f"{upper:g}"
            await _increment(redis, DURATION_BUCKET_HASH, _field(stage, queue, le))


async def record_dlq(
    redis: Any,
    *,
    queue: str,
    function: str,
    job_id: str,
    attempt: int,
    error: BaseException,
) -> None:
    payload = json.dumps(
        {
            "job_id": job_id,
            "function": function,
            "attempt": attempt,
            "error_type": type(error).__name__,
            "failed_at_unix": int(time.time()),
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    key = f"rag:dlq:{queue}"
    try:
        await redis.lpush(key, payload)
        await redis.ltrim(key, 0, settings.queue_dlq_max_entries - 1)
    except Exception:
        logger.warning("queue DLQ write failed", exc_info=True)


async def record_document_number_guard(ctx: dict[str, Any], doc_id: object) -> None:
    """Сохранить абсолютные агрегаты документа без document_id label в Prometheus."""

    try:
        document_id = uuid.UUID(str(doc_id))
        async with ctx["sessionmaker"]() as session:
            rows = list(
                (
                    await session.execute(
                        select(Segment.validation).where(
                            Segment.document_id == document_id,
                            Segment.validation.is_not(None),
                        )
                    )
                )
                .scalars()
                .all()
            )
        protected = 0
        unconfirmed = 0
        for validation in rows:
            if not isinstance(validation, dict):
                continue
            guard = validation.get("entity_guard")
            if not isinstance(guard, dict):
                continue
            protected += int((guard.get("protected") or {}).get("number", 0))
            unconfirmed += int((guard.get("unconfirmed") or {}).get("number", 0))
        await ctx["redis"].hset(NUMBER_GUARD_HASH, str(document_id), f"{protected}|{unconfirmed}")
    except Exception:
        logger.warning("number guard metric snapshot failed", exc_info=True)


class JobRouter:
    """Прозрачная обёртка: legacy по умолчанию, split только для новых enqueue."""

    def __init__(self, redis: ArqRedis, *, mode: Literal["legacy", "split"] | None = None) -> None:
        self.redis = redis
        self.mode = mode or settings.queue_rollout_mode

    async def enqueue_job(self, function: str, *args: Any, **kwargs: Any) -> Any:
        queue = kwargs.get("_queue_name") or route_queue(function, self.mode)
        kwargs["_queue_name"] = queue
        job = await self.redis.enqueue_job(function, *args, **kwargs)
        if job is not None:
            await _increment(self.redis, ENQUEUED_HASH, _field(queue, function))
        return job

    async def aclose(self, *args: Any, **kwargs: Any) -> Any:
        return await self.redis.aclose(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.redis, name)


def _retryable(error: BaseException) -> bool:
    return isinstance(error, (TimeoutError, ConnectionError, OSError, httpx.TransportError))


def instrument_job(function: Callable[..., Any], stage: QueueStage) -> Function:
    """Обернуть новую split-функцию, не меняя её публичное ARQ-имя."""

    @wraps(function)
    async def wrapped(ctx: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        started = time.monotonic()
        redis = ctx["redis"]
        queue = queue_name(stage)
        attempt = int(ctx.get("job_try", 1))
        job_id = str(ctx.get("job_id", "unknown"))
        succeeded = False
        try:
            result = await function(ctx, *args, **kwargs)
            succeeded = True
            if function.__name__ == "translate_document" and args:
                await record_document_number_guard(ctx, args[0])
            return result
        except Retry as error:
            if attempt < settings.queue_max_tries:
                await _increment(redis, RETRY_HASH, _field(queue, function.__name__))
            else:
                await _increment(redis, ERROR_HASH, _field(queue, function.__name__))
                await record_dlq(
                    redis,
                    queue=queue,
                    function=function.__name__,
                    job_id=job_id,
                    attempt=attempt,
                    error=error,
                )
            raise
        except asyncio.CancelledError as error:
            if attempt < settings.queue_max_tries:
                await asyncio.shield(
                    _increment(redis, RETRY_HASH, _field(queue, function.__name__))
                )
            else:
                await asyncio.shield(
                    _increment(redis, ERROR_HASH, _field(queue, function.__name__))
                )
                await asyncio.shield(
                    record_dlq(
                        redis,
                        queue=queue,
                        function=function.__name__,
                        job_id=job_id,
                        attempt=attempt,
                        error=error,
                    )
                )
            raise
        except Exception as error:
            if _retryable(error) and attempt < settings.queue_max_tries:
                await _increment(redis, RETRY_HASH, _field(queue, function.__name__))
                delay = min(
                    settings.queue_retry_base_s * (2 ** (attempt - 1)),
                    settings.queue_retry_cap_s,
                )
                raise Retry(defer=delay) from error
            await _increment(redis, ERROR_HASH, _field(queue, function.__name__))
            await record_dlq(
                redis,
                queue=queue,
                function=function.__name__,
                job_id=job_id,
                attempt=attempt,
                error=error,
            )
            raise
        finally:
            await asyncio.shield(record_duration(redis, stage, queue, time.monotonic() - started))
            if succeeded:
                await asyncio.shield(
                    _increment(redis, COMPLETED_HASH, _field(queue, function.__name__))
                )

    return func(wrapped, name=function.__name__, max_tries=settings.queue_max_tries)
