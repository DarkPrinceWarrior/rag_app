"""Loopback Prometheus exporter: ARQ/Redis counters and translation quality."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import FastAPI, Response
from prometheus_client import CollectorRegistry, Gauge, generate_latest

from rag_app.config import settings
from rag_app.workers.queueing import (
    COMPLETED_HASH,
    DURATION_BUCKET_HASH,
    DURATION_COUNT_HASH,
    DURATION_SUM_HASH,
    ENQUEUED_HASH,
    ERROR_HASH,
    NUMBER_GUARD_HASH,
    RETRY_HASH,
    all_queue_names,
)


def _decode(value: bytes | str) -> str:
    return value.decode() if isinstance(value, bytes) else value


def _split_field(value: bytes | str, parts: int) -> tuple[str, ...] | None:
    labels = tuple(_decode(value).split("|"))
    return labels if len(labels) == parts else None


def _number_guard_counts(rows: dict[bytes | str, bytes | str]) -> tuple[int, int]:
    protected = 0
    unconfirmed = 0
    for raw in rows.values():
        values = _decode(raw).split("|")
        if len(values) != 2:
            continue
        try:
            protected += int(values[0])
            unconfirmed += int(values[1])
        except ValueError:
            continue
    return protected, unconfirmed


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.redis = await create_pool(
        RedisSettings(host=settings.redis_host, port=settings.redis_port, database=settings.redis_db)
    )
    try:
        yield
    finally:
        await app.state.redis.aclose()


app = FastAPI(
    title="DocRAGenslate metrics exporter",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)


async def _queue_metrics(registry: CollectorRegistry, redis: Any) -> None:
    depth = Gauge("rag_arq_queue_depth", "ARQ queue depth", ["queue"], registry=registry)
    age = Gauge(
        "rag_arq_queue_oldest_age_seconds",
        "Age of the oldest ready ARQ job",
        ["queue"],
        registry=registry,
    )
    dlq = Gauge("rag_arq_dlq_depth", "Bounded dead-letter queue depth", ["queue"], registry=registry)
    now = time.time()
    for queue in all_queue_names():
        depth.labels(queue).set(await redis.zcard(queue))
        oldest = await redis.zrange(queue, 0, 0, withscores=True)
        score = float(oldest[0][1]) / 1000 if oldest else now
        age.labels(queue).set(max(now - score, 0.0))
        dlq.labels(queue).set(await redis.llen(f"rag:dlq:{queue}"))

    for key, name, description in (
        (ENQUEUED_HASH, "rag_arq_jobs_enqueued_total", "Jobs accepted by the router"),
        (COMPLETED_HASH, "rag_arq_jobs_completed_total", "Split jobs completed"),
        (RETRY_HASH, "rag_arq_job_retries_total", "Split job retries"),
        (ERROR_HASH, "rag_arq_job_errors_total", "Terminal split job errors"),
    ):
        metric = Gauge(name, description, ["queue", "function"], registry=registry)
        for field, raw in (await redis.hgetall(key)).items():
            labels = _split_field(field, 2)
            if labels is not None:
                metric.labels(*labels).set(float(raw))

    duration_bucket = Gauge(
        "rag_document_stage_duration_seconds_bucket",
        "Cumulative split-job stage duration buckets",
        ["stage", "queue", "le"],
        registry=registry,
    )
    for field, raw in (await redis.hgetall(DURATION_BUCKET_HASH)).items():
        labels = _split_field(field, 3)
        if labels is not None:
            duration_bucket.labels(*labels).set(float(raw))
    for key, suffix, description in (
        (DURATION_SUM_HASH, "sum", "Cumulative split-job stage duration"),
        (DURATION_COUNT_HASH, "count", "Observed split-job stage executions"),
    ):
        metric = Gauge(
            f"rag_document_stage_duration_seconds_{suffix}",
            description,
            ["stage", "queue"],
            registry=registry,
        )
        for field, raw in (await redis.hgetall(key)).items():
            labels = _split_field(field, 2)
            if labels is not None:
                metric.labels(*labels).set(float(raw))


async def _quality_metrics(registry: CollectorRegistry, redis: Any) -> None:
    protected, unconfirmed = _number_guard_counts(await redis.hgetall(NUMBER_GUARD_HASH))
    total = Gauge(
        "rag_translation_number_entities_total",
        "Number entities observed by the deterministic translation guard",
        ["status"],
        registry=registry,
    )
    total.labels("protected").set(protected)
    total.labels("unconfirmed").set(unconfirmed)
    ratio = Gauge(
        "rag_translation_number_unconfirmed_ratio",
        "Unconfirmed number entities divided by protected number entities",
        registry=registry,
    )
    ratio.set(unconfirmed / protected if protected else 0.0)


@app.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    registry = CollectorRegistry()
    await _queue_metrics(registry, app.state.redis)
    await _quality_metrics(registry, app.state.redis)
    return Response(generate_latest(registry), media_type="text/plain; version=0.0.4; charset=utf-8")


@app.get("/healthz", include_in_schema=False)
async def healthz() -> dict[str, str]:
    await app.state.redis.ping()
    return {"status": "ok"}
