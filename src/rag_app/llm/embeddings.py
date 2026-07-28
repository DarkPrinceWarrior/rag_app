"""Клиенты эмбеддингов (vLLM --runner pooling) и reranker'а (vLLM /v1/rerank).

§ 12.1 шаг 1: Qwen3-Embedding-0.6B + Qwen3-Reranker-4B. Серия instruction-aware:
- эмбеддинг ЗАПРОСА — с инструкцией («Instruct: …\nQuery: …»), документов — без;
- reranker получает запрос в формате «<Instruct>: …\n<Query>: …».
Замена движка (TEI, SGLang) или отказ от инструкций — сменой base_url/конфига.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from typing import Any

import httpx
from openai import AsyncOpenAI

from rag_app.config import settings

logger = logging.getLogger(__name__)

_EMBED_BODY_MAX_CHARS = 8000


def _mrl(vec: list[float], dim: int) -> list[float]:
    """MRL-усечение вектора до dim + L2-нормировка. Текстовая Qwen3-Embedding
    Matryoshka-обучена → усечение валидно (для нативного dim — no-op). Норму держим
    для консистентности (косинус `<=>` к ней инвариантен)."""
    v = vec[:dim]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _embedding_body(text: str) -> str:
    """Нормализовать и ограничить только пользовательскую часть входа."""

    return text.strip()[:_EMBED_BODY_MAX_CHARS] or "."


def _nemotron_input(text: str, kind: str) -> str:
    """Добавить канонический Nemotron-префикс ровно один раз."""

    marker = f"{kind}:"
    raw = text.strip()
    if raw.casefold().startswith(marker):
        raw = raw[len(marker) :].lstrip()
    body = _embedding_body(raw)
    return f"{marker} {body}"


def _document_input(text: str) -> str:
    if settings.embed_input_profile == "nemotron3":
        return _nemotron_input(text, "passage")
    return _embedding_body(text)


def _query_input(query: str) -> str:
    if settings.embed_input_profile == "nemotron3":
        return _nemotron_input(query, "query")
    if settings.embed_query_instruction:
        prefix = f"Instruct: {settings.embed_query_instruction}\nQuery: "
        raw = query.strip()
        if raw.startswith(prefix):
            raw = raw[len(prefix) :]
        body = _embedding_body(raw)
        return f"{prefix}{body}"
    return _embedding_body(query)


class Embedder:
    def __init__(self) -> None:
        self.client = AsyncOpenAI(
            base_url=settings.embed_base_url, api_key="local", timeout=120.0
        )

    async def embed(self, texts: list[str], batch: int | None = None) -> list[list[float]]:
        """Эмбеддинги документов: raw для Qwen3, ``passage:`` для Nemotron-3.

        Пустое тело заменяется точкой, пользовательская часть ограничивается до
        добавления служебного префикса выбранного профиля.
        """
        out: list[list[float]] = []
        batch = batch or settings.embed_batch_size
        for i in range(0, len(texts), batch):
            chunk = [_document_input(text) for text in texts[i : i + batch]]
            resp = await self.client.embeddings.create(model=settings.embed_model, input=chunk)
            data = sorted(resp.data, key=lambda item: item.index)
            out.extend(_mrl(item.embedding, settings.embed_dim) for item in data)
        return out

    async def embed_query(self, query: str) -> list[float]:
        """Эмбеддинг запроса в формате выбранного профиля.

        Qwen3 получает настраиваемую инструкцию (пустая отключает её), Nemotron-3
        — обязательный ``query:``. Тело ограничивается до добавления префикса.
        """
        text = _query_input(query)
        resp = await self.client.embeddings.create(model=settings.embed_model, input=[text])
        return _mrl(resp.data[0].embedding, settings.embed_dim)


# Официальный шаблон Qwen3-Reranker: vLLM с is_original_qwen3_reranker НЕ
# оборачивает вход сам — без шаблона скоры слипаются (0.34/0.27 на контрольной
# паре), с ним разделение 0.97/0.00.
_QWEN3_RR_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on "
    'the Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
_QWEN3_RR_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def build_rerank_payload(query: str, texts: list[str]) -> dict[str, Any]:
    """Собрать точный HTTP payload, включая официальный Qwen3-шаблон.

    Вынесено отдельно, чтобы квалификация могла доказать, что шаблон находится
    в фактически отправляемых строках, а не полагаться на серверный флаг,
    который отдельные версии vLLM могли молча игнорировать.
    """

    q = query[:2000]
    docs = [text[:4000] for text in texts]
    if settings.rerank_model.startswith("qwen3-reranker"):
        q = f"{_QWEN3_RR_PREFIX}<Instruct>: {settings.rerank_instruction}\n<Query>: {q}\n"
        docs = [f"<Document>: {document}{_QWEN3_RR_SUFFIX}" for document in docs]
    return {"model": settings.rerank_model, "query": q, "documents": docs}


def reranker_template_sha256() -> str:
    """Хеш версии клиентского шаблона и инструкции для runtime attestation."""

    payload = {
        "model": settings.rerank_model,
        "instruction": settings.rerank_instruction,
        "qwen3_prefix": _QWEN3_RR_PREFIX,
        "qwen3_suffix": _QWEN3_RR_SUFFIX,
        "protocol": "manual-qwen3-reranker-template-v1",
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class Reranker:
    async def rerank(self, query: str, texts: list[str]) -> list[float]:
        """Релевантности (в исходном порядке texts) через /v1/rerank (Cohere-совместимый)."""
        if not texts:
            return []
        payload = build_rerank_payload(query, texts)
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{settings.rerank_base_url}/v1/rerank",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        scores = [0.0] * len(texts)
        for item in data["results"]:
            scores[item["index"]] = float(item["relevance_score"])
        return scores
