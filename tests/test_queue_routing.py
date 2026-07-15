"""Совместимость legacy drain и маршрутизация только новых ARQ job."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest
from arq import Retry

from rag_app.config import Settings, settings
from rag_app.workers.main import WorkerSettings
from rag_app.workers.queue_workers import (
    ExportIndexWorkerSettings,
    MemoryWorkerSettings,
    ParseWorkerSettings,
    TranslateWorkerSettings,
)
from rag_app.workers.queueing import (
    COMPLETED_HASH,
    ERROR_HASH,
    RETRY_HASH,
    JobRouter,
    instrument_job,
    route_queue,
)
from rag_app.workers.tasks import parse_document

ROOT = Path(__file__).resolve().parents[1]


class FakeRedis:
    def __init__(self) -> None:
        self.jobs: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.hashes: dict[str, dict[str, float]] = defaultdict(dict)
        self.lists: dict[str, list[str]] = defaultdict(list)

    async def enqueue_job(self, function: str, *args: Any, **kwargs: Any) -> object:
        self.jobs.append((function, args, kwargs))
        return object()

    async def hincrby(self, key: str, field: str, amount: int) -> None:
        self.hashes[key][field] = self.hashes[key].get(field, 0) + amount

    async def hincrbyfloat(self, key: str, field: str, amount: float) -> None:
        self.hashes[key][field] = self.hashes[key].get(field, 0) + amount

    async def lpush(self, key: str, payload: str) -> None:
        self.lists[key].insert(0, payload)

    async def ltrim(self, key: str, start: int, end: int) -> None:
        self.lists[key] = self.lists[key][start : end + 1]


@pytest.mark.asyncio
async def test_legacy_mode_preserves_queue_and_job_id() -> None:
    redis = FakeRedis()
    router = JobRouter(redis, mode="legacy")  # type: ignore[arg-type]
    await router.enqueue_job("parse_document", "doc", 7, _job_id="parse:doc:7")
    _, _, kwargs = redis.jobs[0]
    assert kwargs["_queue_name"] == "arq:queue"
    assert kwargs["_job_id"] == "parse:doc:7"


@pytest.mark.asyncio
async def test_split_routes_only_mapped_new_jobs_and_respects_explicit_queue() -> None:
    redis = FakeRedis()
    router = JobRouter(redis, mode="split")  # type: ignore[arg-type]
    await router.enqueue_job("translate_document", "doc", 7, _job_id="translate:doc:7")
    await router.enqueue_job("unknown_control_job")
    await router.enqueue_job("parse_document", _queue_name="arq:manual")
    assert redis.jobs[0][2]["_queue_name"] == settings.queue_translate_name
    assert redis.jobs[1][2]["_queue_name"] == "arq:queue"
    assert redis.jobs[2][2]["_queue_name"] == "arq:manual"


def test_legacy_worker_remains_drain_consumer() -> None:
    assert parse_document in WorkerSettings.functions
    assert getattr(WorkerSettings, "queue_name", "arq:queue") == "arq:queue"
    assert route_queue("parse_document", "legacy") == "arq:queue"


def test_split_worker_profiles_are_disjoint() -> None:
    profiles = (
        ParseWorkerSettings,
        TranslateWorkerSettings,
        ExportIndexWorkerSettings,
        MemoryWorkerSettings,
    )
    queues = {profile.queue_name for profile in profiles}
    names = [{function.name for function in profile.functions} for profile in profiles]
    assert len(queues) == 4
    assert set().union(*names) == set(route_queue_name for route_queue_name in (
        "parse_document",
        "translate_document",
        "translate_to_language",
        "export_document",
        "render_original_view",
        "index_document",
        "index_pages_visual",
        "describe_images",
        "extract_memory",
        "consolidate_memory",
    ))
    assert sum(len(group) for group in names) == len(set().union(*names))
    assert all(profile.cron_jobs == [] for profile in profiles)


@pytest.mark.asyncio
async def test_transport_error_retries_with_bound() -> None:
    async def transient(_ctx: dict[str, Any]) -> None:
        raise OSError("temporary")

    job = instrument_job(transient, "parse")
    redis = FakeRedis()
    with pytest.raises(Retry):
        await job.coroutine({"redis": redis, "job_try": 1, "job_id": "job-1"})
    assert redis.hashes[RETRY_HASH][f"{settings.queue_parse_name}|transient"] == 1
    assert not redis.lists


@pytest.mark.asyncio
async def test_business_error_goes_directly_to_bounded_dlq() -> None:
    async def invalid(_ctx: dict[str, Any]) -> None:
        raise ValueError("document content is deliberately absent from DLQ")

    job = instrument_job(invalid, "parse")
    redis = FakeRedis()
    with pytest.raises(ValueError):
        await job.coroutine({"redis": redis, "job_try": 1, "job_id": "job-2"})
    assert redis.hashes[ERROR_HASH][f"{settings.queue_parse_name}|invalid"] == 1
    payload = redis.lists[f"rag:dlq:{settings.queue_parse_name}"][0]
    assert "ValueError" in payload
    assert "document content" not in payload


@pytest.mark.asyncio
async def test_success_records_completion_without_changing_result() -> None:
    async def valid(_ctx: dict[str, Any]) -> str:
        return "ok"

    job = instrument_job(valid, "parse")
    redis = FakeRedis()
    result = await job.coroutine({"redis": redis, "job_try": 1, "job_id": "job-3"})
    assert result == "ok"
    assert redis.hashes[COMPLETED_HASH][f"{settings.queue_parse_name}|valid"] == 1


@pytest.mark.asyncio
async def test_final_cancellation_records_terminal_error_and_dlq() -> None:
    async def cancelled(_ctx: dict[str, Any]) -> None:
        raise asyncio.CancelledError

    job = instrument_job(cancelled, "parse")
    redis = FakeRedis()
    with pytest.raises(asyncio.CancelledError):
        await job.coroutine(
            {
                "redis": redis,
                "job_try": settings.queue_max_tries,
                "job_id": "job-cancelled",
            }
        )
    assert redis.hashes[ERROR_HASH][f"{settings.queue_parse_name}|cancelled"] == 1
    assert redis.lists[f"rag:dlq:{settings.queue_parse_name}"]


def test_core_pipeline_job_ids_are_revision_stable() -> None:
    source = (ROOT / "src/rag_app/workers/tasks.py").read_text()
    assert '_job_id=f"translate:{doc_id}:{claimed_revision}"' in source
    assert '_job_id=f"export:{doc_id}:{parse_revision}"' in source
    assert '_job_id=f"index:{doc_id}:{parse_revision}"' in source
    assert '_job_id=f"vindex:{doc_id}:{parse_revision}"' in source
    assert '_job_id=f"vieworig:{doc_id}:{claimed_revision}"' in source


def test_queue_config_rejects_aliases_and_inverted_retry_window() -> None:
    with pytest.raises(ValueError, match="unique non-legacy"):
        Settings(queue_parse_name="arq:queue")
    with pytest.raises(ValueError, match="retry base"):
        Settings(queue_retry_base_s=301, queue_retry_cap_s=300)
