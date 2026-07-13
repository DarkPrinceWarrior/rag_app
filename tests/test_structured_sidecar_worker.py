from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from arq.constants import default_queue_name
from arq.worker import create_worker
from pydantic import ValidationError

from rag_app.config import Settings, settings
from rag_app.workers import structured_sidecar


class _Engine:
    def __init__(self) -> None:
        self.disposed = False

    async def dispose(self) -> None:
        self.disposed = True


class _Storage:
    def __init__(self) -> None:
        self.buckets_ready = False

    async def ensure_buckets(self) -> None:
        self.buckets_ready = True


class _Client:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def test_worker_uses_isolated_queue_and_registers_bounded_kie_job() -> None:
    worker = structured_sidecar.WorkerSettings

    assert worker.queue_name == settings.structured_sidecar_queue_name
    assert worker.queue_name != default_queue_name
    assert worker.functions == [structured_sidecar.health_probe, structured_sidecar.run_structured_kie]
    assert worker.cron_jobs == []
    assert worker.max_jobs == 1
    assert worker.max_tries == settings.structured_job_max_attempts
    assert worker.job_timeout == settings.parser_sidecar_timeout_s
    assert worker.health_check_interval == settings.structured_sidecar_health_check_interval_s
    assert worker.health_check_key == f"{worker.queue_name}:health-check"

    async def build_worker():
        return create_worker(worker, burst=True)

    instance = asyncio.run(build_worker())
    assert [function.name for function in instance.functions.values()] == [
        "health_probe",
        "run_structured_kie",
    ]


@pytest.mark.parametrize("queue_name", ["", "arq:queue", "sidecar", "arq:structured-sidecar:BAD"])
def test_settings_reject_queue_collision_or_invalid_namespace(queue_name: str) -> None:
    with pytest.raises(ValidationError, match="isolated"):
        Settings(_env_file=None, structured_sidecar_queue_name=queue_name)


def test_settings_accepts_isolated_queue_suffix() -> None:
    configured = Settings(
        _env_file=None,
        structured_sidecar_queue_name=" arq:structured-sidecar:canary_1 ",
    )
    assert configured.structured_sidecar_queue_name == "arq:structured-sidecar:canary_1"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("parser_sidecar_timeout_s", 0),
        ("parser_sidecar_timeout_s", -1),
        ("structured_sidecar_health_check_interval_s", 0),
        ("structured_sidecar_health_check_interval_s", -1),
    ],
)
def test_settings_rejects_non_positive_sidecar_timing(field: str, value: int) -> None:
    with pytest.raises(ValidationError, match="timing values must be positive"):
        Settings(_env_file=None, **{field: value})


def test_startup_and_shutdown_use_shared_infrastructure(monkeypatch) -> None:
    engine = _Engine()
    storage = _Storage()
    sessionmaker = object()
    monkeypatch.setattr(structured_sidecar, "create_engine", lambda: engine)
    role_check = AsyncMock()
    monkeypatch.setattr(structured_sidecar, "assert_worker_rls_role", role_check)
    monkeypatch.setattr(structured_sidecar, "create_sessionmaker", lambda value: sessionmaker)
    monkeypatch.setattr(structured_sidecar, "Storage", lambda: storage)
    ctx: dict = {}

    asyncio.run(structured_sidecar.startup(ctx))
    role_check.assert_awaited_once_with(engine)

    assert ctx == {
        "engine": engine,
        "sessionmaker": sessionmaker,
        "storage": storage,
    }
    assert storage.buckets_ready is True

    asyncio.run(structured_sidecar.shutdown(ctx))
    assert engine.disposed is True


def test_enabled_startup_owns_and_closes_pinned_model_client(monkeypatch) -> None:
    engine = _Engine()
    storage = _Storage()
    client = _Client()
    monkeypatch.setattr(settings, "structured_extraction_enabled", True)
    monkeypatch.setattr(structured_sidecar, "create_engine", lambda: engine)
    role_check = AsyncMock()
    monkeypatch.setattr(structured_sidecar, "assert_worker_rls_role", role_check)
    monkeypatch.setattr(structured_sidecar, "create_sessionmaker", lambda value: object())
    monkeypatch.setattr(structured_sidecar, "Storage", lambda: storage)
    monkeypatch.setattr(structured_sidecar, "GraniteStructuredClient", lambda **kwargs: client)
    ctx: dict = {}

    asyncio.run(structured_sidecar.startup(ctx))
    role_check.assert_awaited_once_with(engine)

    assert ctx["structured_client"] is client
    asyncio.run(structured_sidecar.shutdown(ctx))
    assert client.closed is True
    assert engine.disposed is True


def test_shutdown_tolerates_failed_startup() -> None:
    asyncio.run(structured_sidecar.shutdown({}))
