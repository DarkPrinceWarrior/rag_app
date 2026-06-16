"""Adversarial leakage-suite (§10): детерминированно доказывает scope-изоляцию
памяти — запрос одного пользователя/проекта/документа НИКОГДА не возвращает
items другого. Метрика: leakage_rate = 0.

Вставляет контролируемый набор items для двух пользователей (user/project/
document scope), затем гоняет ретрив из разных контекстов и проверяет, что
возвращается ТОЛЬКО разрешённое. Проверяются оба слоя: сырой scope-фильтр SQL
(`adapter.search`, до gate) и боевой путь (`retrieve_block`, после gate).

Работает и под RLS ENABLE, и под FORCE (выставляет GUC через apply_scope_guc).

Запуск на сервере: uv run python scripts/_leakage_suite.py
"""

from __future__ import annotations

import asyncio
import uuid

from rag_app.db.engine import create_engine, create_sessionmaker
from rag_app.llm.embeddings import Embedder, Reranker
from rag_app.rag.memory import MemoryService
from rag_app.rag.memory.rls import apply_scope_guc

U_A, U_B = "leak-userA", "leak-userB"
# фиксированный запрос, релевантный всем вставленным items (релевантность не
# важна для детекции утечки — важно, что ЧУЖОЕ не приходит ни при каких условиях)
QUERY = "ALPHA BETA secret fact договор раздел project document data"


async def main() -> None:
    engine = create_engine()
    sm = create_sessionmaker(engine)
    memory = MemoryService(Embedder(), Reranker())
    p_a, p_b = uuid.uuid4(), uuid.uuid4()
    d_a, d_b = uuid.uuid4(), uuid.uuid4()

    async def cleanup() -> None:
        # каждый пользователь — в своей транзакции: GUC (SET LOCAL) транзакционный,
        # под FORCE отложенный audit-INSERT проверяется против GUC момента commit
        for u in (U_A, U_B):
            async with sm() as s:
                await memory.purge_user(s, u)
                await s.commit()

    await cleanup()

    # контролируемый набор: имя → item.id. Каждый пользователь — в своей
    # транзакции (под FORCE GUC транзакционный, аудит-INSERT проверяется на commit).
    async with sm() as s:
        i1 = await memory.add_manual(
            s, memory.scope_for(U_A), scope_kind="user", kind="fact",
            content="ALPHA пользовательский секрет A")
        i2 = await memory.add_manual(
            s, memory.scope_for(U_A, project_id=p_a), scope_kind="project", kind="fact",
            content="ALPHA проектный договор A в проекте PA")
        i3 = await memory.add_manual(
            s, memory.scope_for(U_A, document_id=d_a), scope_kind="document", kind="fact",
            content="ALPHA документный раздел A в документе DA")
        await s.commit()
    async with sm() as s:
        i4 = await memory.add_manual(
            s, memory.scope_for(U_B), scope_kind="user", kind="fact",
            content="BETA пользовательский секрет B")
        i5 = await memory.add_manual(
            s, memory.scope_for(U_B, project_id=p_a), scope_kind="project", kind="fact",
            content="BETA проектный факт B в проекте PA")
        await s.commit()
    names = {i1.id: "I1", i2.id: "I2", i3.id: "I3", i4.id: "I4", i5.id: "I5"}

    # (контекст запроса, что разрешено видеть)
    contexts = [
        ("C1  uA / без контекста", memory.scope_for(U_A), {"I1"}),
        ("C2  uA / проект PA", memory.scope_for(U_A, project_id=p_a), {"I1", "I2"}),
        ("C3  uA / проект PB", memory.scope_for(U_A, project_id=p_b), {"I1"}),
        ("C4  uA / документ DA", memory.scope_for(U_A, document_id=d_a), {"I1", "I3"}),
        ("C5  uA / документ DB", memory.scope_for(U_A, document_id=d_b), {"I1"}),
        ("C6  uB / без контекста", memory.scope_for(U_B), {"I4"}),
        ("C7  uB / проект PA", memory.scope_for(U_B, project_id=p_a), {"I4", "I5"}),
    ]

    leaks = 0
    print(f"{'контекст':28} {'SQL-фильтр':22} {'после gate':22} вердикт")
    for name, scope, allowed in contexts:
        async with sm() as s:
            await apply_scope_guc(s, scope)
            raw = await memory.adapter.search(s, QUERY, scope, 50)
        raw_names = {names[h.id] for h in raw if h.id in names}
        raw_leak = raw_names - allowed

        async with sm() as s:
            _, gated = await memory.retrieve_block(s, QUERY, scope)
        gated_names = {names[h.id] for h in gated if h.id in names}
        gated_leak = gated_names - allowed

        leaks += len(raw_leak) + len(gated_leak)
        verdict = "УТЕЧКА!" if (raw_leak or gated_leak) else "ok"
        print(f"{name:28} {str(sorted(raw_names)):22} {str(sorted(gated_names)):22} {verdict}"
              + (f"  leak={sorted(raw_leak | gated_leak)}" if (raw_leak or gated_leak) else ""))

    print(f"\n=== leakage_rate: {leaks} утечек ===")
    print("ВЕРДИКТ:", "ИЗОЛЯЦИЯ ДЕРЖИТ (0 утечек)" if leaks == 0 else "ЕСТЬ УТЕЧКА — СМОТРЕТЬ ВЫШЕ")

    await cleanup()
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
