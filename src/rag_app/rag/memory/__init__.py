"""Слой межсессионной памяти (docs/MEMORY_rev4_mem0_articles.md §15).

native-first: память живёт в нашем Postgres+pgvector, ретрив — Qwen3-Embedding/
Reranker (`InternalAdapter`), генерация/экстракция — Qwen3.5. Mem0 — swappable
провайдер за `MemoryAdapter` (Этап 4), на старте не ставится.

Память ≠ RAG-коллекция: отдельные таблицы, lifecycle, gate и UI; в промпте
блок памяти подаётся ОТДЕЛЬНО от фрагментов документов и только как contextual
hints (§1, §6.2). Документ побеждает память по фактам/числам/срокам (§2.2.2).
"""

from __future__ import annotations

from rag_app.rag.memory.adapter import InternalAdapter, MemoryAdapter, MemoryHit, MemoryScope
from rag_app.rag.memory.gate import GateDecision, MemoryGate
from rag_app.rag.memory.service import MemoryService, fingerprint

__all__ = [
    "InternalAdapter",
    "MemoryAdapter",
    "MemoryHit",
    "MemoryScope",
    "GateDecision",
    "MemoryGate",
    "MemoryService",
    "fingerprint",
]
