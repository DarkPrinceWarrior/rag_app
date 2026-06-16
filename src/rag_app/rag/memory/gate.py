"""Memory gate (§5): единственная точка, решающая, попадёт ли item в промпт.

Между rerank и сборкой промпта. Чистая логика (без БД) — логирование gate_block
делает вызывающий сервис. allow только если прошли ВСЕ проверки:
scope → validity → trust → sensitivity → relevance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from rag_app.config import settings

if TYPE_CHECKING:
    from rag_app.rag.memory.adapter import MemoryHit, MemoryScope


@dataclass
class GateDecision:
    item_id: str
    decision: str  # allow | block
    reasons: list[str] = field(default_factory=list)
    blocked_by: str | None = None
    scores: dict[str, float] = field(default_factory=dict)


class MemoryGate:
    def __init__(
        self,
        min_confidence: float | None = None,
        min_rerank: float | None = None,
        *,
        allow_secret: bool = False,
    ) -> None:
        self.min_confidence = (
            settings.memory_gate_min_confidence if min_confidence is None else min_confidence
        )
        self.min_rerank = settings.memory_gate_min_rerank if min_rerank is None else min_rerank
        self.allow_secret = allow_secret

    def evaluate(self, hit: MemoryHit, scope: MemoryScope) -> GateDecision:
        d = GateDecision(
            item_id=str(hit.id),
            decision="allow",
            scores={
                "rerank": round(hit.rerank, 4),
                "confidence": round(hit.confidence, 4),
                "importance": round(hit.importance, 4),
            },
        )

        # 1. scope — defense-in-depth поверх SQL-фильтра: item не шире доступного
        if not _scope_allowed(hit, scope):
            return _block(d, "scope")
        d.reasons.append("scope_ok")

        # 2. validity — активен и не истёк (SQL уже отфильтровал, повторяем)
        if hit.status != "active":
            return _block(d, "validity")
        if hit.valid_to is not None and hit.valid_to <= _now():
            return _block(d, "validity")
        d.reasons.append("validity_ok")

        # 3. source trust — доверие источника
        if hit.confidence < self.min_confidence:
            return _block(d, "trust")
        d.reasons.append("trust_ok")

        # 4. sensitivity — secret не в обычный чат; sensitive — только в своём scope
        #    (записи уже user-scoped, потому sensitive допускается)
        if hit.sensitivity == "secret" and not self.allow_secret:
            return _block(d, "sensitivity")
        d.reasons.append("sensitivity_ok")

        # 5. relevance — финальный rerank-порог
        if hit.rerank < self.min_rerank:
            return _block(d, "relevance")
        d.reasons.append("relevance_ok")

        return d


def _now() -> datetime:
    return datetime.now(UTC)


def _scope_allowed(hit: MemoryHit, scope: MemoryScope) -> bool:
    if hit.scope in ("user", "org"):
        return True
    if hit.scope == "project":
        return scope.project_id is not None
    if hit.scope == "document":
        return scope.document_id is not None
    if hit.scope == "thread":
        return scope.thread_id is not None
    return False


def _block(d: GateDecision, by: str) -> GateDecision:
    d.decision = "block"
    d.blocked_by = by
    return d
