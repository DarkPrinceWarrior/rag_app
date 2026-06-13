"""Наблюдаемость (roadmap § 10): трассы Langfuse + ошибки в Sentry.

Langfuse включается LANGFUSE_PUBLIC_KEY/SECRET_KEY/HOST, Sentry — SENTRY_DSN
(всё в .env). Без ключей всё no-op; ошибки трейсинга/инициализации никогда
не роняют бизнес-операцию. Модуль импортируется и API, и воркером — Sentry
инициализируется в обоих процессах на импорте.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# LANGFUSE_* лежат в .env проекта; pydantic-settings его читает только для
# Settings и НЕ экспортирует в os.environ — SDK же смотрит именно в окружение.
load_dotenv()

_enabled = bool(os.getenv("LANGFUSE_PUBLIC_KEY"))
_client = None

if _enabled:
    try:
        from langfuse import Langfuse

        _client = Langfuse()
        logger.info("langfuse: трейсинг включён (%s)", os.getenv("LANGFUSE_HOST"))
    except Exception:
        logger.exception("langfuse: не инициализировался — трейсинг выключен")
        _enabled = False


# Sentry: захват необработанных исключений в API и воркере. Интеграции FastAPI
# и asyncio подхватываются sentry-sdk автоматически при наличии пакетов.
if os.getenv("SENTRY_DSN"):
    try:
        import sentry_sdk

        sentry_sdk.init(
            dsn=os.environ["SENTRY_DSN"],
            environment=os.getenv("SENTRY_ENVIRONMENT", "prod"),
            traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.0")),
            send_default_pii=False,  # on-prem: не отправляем тела/заголовки с данными
        )
        logger.info("sentry: включён (env=%s)", os.getenv("SENTRY_ENVIRONMENT", "prod"))
    except Exception:
        logger.exception("sentry: не инициализировался")


def log_chat_trace(
    question: str,
    chunks: list[Any],
    answer: str,
    model: str,
    ms: int,
    session_id: str,
    user_sub: str | None,
) -> None:
    if not _enabled:
        return
    try:
        from langfuse import propagate_attributes

        with propagate_attributes(
            user_id=user_sub or "anonymous", session_id=session_id, trace_name="rag-chat"
        ):
            with _client.start_as_current_observation(name="rag-chat") as span:
                span.update(input={"question": question})
                with span.start_as_current_observation(name="retrieve") as retr:
                    retr.update(
                        output=[
                            {
                                "file": c.filename,
                                "heading": c.heading_path[:120],
                                "score": round(c.score, 4),
                            }
                            for c in chunks
                        ]
                    )
                with span.start_as_current_generation(name="answer", model=model) as gen:
                    gen.update(output=answer[:4000], metadata={"latency_ms": ms})
                span.update(output={"answer": answer[:2000], "citations": len(chunks)})
    except Exception:
        logger.exception("langfuse: chat-трейс не записался")


def log_agent_trace(
    question: str,
    mode: str,
    steps: list[dict[str, Any]],
    stop_reason: str,
    tokens: int,
    n_chunks: int,
    ms: int,
    session_id: str,
    user_sub: str | None,
) -> None:
    """Трасса agentic-цикла (§ 5 п.7): шаги tool-вызовов + стоп-условие."""
    if not _enabled:
        return
    try:
        from langfuse import propagate_attributes

        with propagate_attributes(
            user_id=user_sub or "anonymous", session_id=session_id, trace_name="rag-agent"
        ):
            with _client.start_as_current_observation(name="rag-agent") as span:
                span.update(
                    input={"question": question, "mode": mode},
                    output={
                        "steps": [
                            {"tool": s.get("tool"), "arg": str(s.get("arg", ""))[:120]} for s in steps
                        ],
                        "stop_reason": stop_reason,
                        "iters": len(steps),
                        "chunks": n_chunks,
                    },
                    metadata={"tokens": tokens, "latency_ms": ms},
                )
    except Exception:
        logger.exception("langfuse: agent-трейс не записался")


def log_translate_trace(
    doc_id: str, filename: str, kind: str, segments: int, seconds: float, model: str
) -> None:
    if not _enabled:
        return
    try:
        from langfuse import propagate_attributes

        with propagate_attributes(trace_name="translate-document", metadata={"kind": kind}):
            with _client.start_as_current_observation(name="translate-document") as span:
                span.update(
                    input={"document_id": doc_id, "filename": filename},
                    output={"segments": segments, "seconds": round(seconds, 1)},
                    metadata={"model": model},
                )
    except Exception:
        logger.exception("langfuse: translate-трейс не записался")
