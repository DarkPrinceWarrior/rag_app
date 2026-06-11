"""Трейсинг в Langfuse (roadmap § 10): каждый RAG-запрос и перевод документа.

Включается наличием LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY/LANGFUSE_HOST
в окружении (.env). Без ключей — все вызовы no-op; ошибки трейсинга
никогда не роняют бизнес-операцию.
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
