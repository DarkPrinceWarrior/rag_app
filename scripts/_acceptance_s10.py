"""Сценарная приёмка §10 (memory4) + измеримые пороги.

5 сценариев спеки (§10): doc-only, doc+memory, project-glossary, conflict,
agentic-multi-step — проверяем маршрутизацию (QueryRouter) и scoped-ретрив.
Плюс пороги: retrieve+gate p95 ≤ 200 мс (с реальным Qwen3-Reranker) и cap'ы
выдачи после gate (≤5 user/project/document, ≤3 thread).

Числовые/договорные факты берутся из документного RAG с цитатой, а не из памяти —
это гарантируется конструкцией промпта: блок памяти отделён и помечен как
contextual-hints (§6.2), факты документа обязаны идти с [n]. Здесь проверяем
маршрут + ретрив + структуру блока (не гоняем полный LLM-ответ на каждый кейс).

Запуск: set -a && . ./.env.api.local && uv run python scripts/_acceptance_s10.py
"""

from __future__ import annotations

import asyncio
import statistics
import time
import uuid

from openai import AsyncOpenAI

from rag_app.config import settings
from rag_app.db.engine import create_engine, create_sessionmaker
from rag_app.llm.embeddings import Embedder, Reranker
from rag_app.rag.agent import classify
from rag_app.rag.memory import MemoryService
from rag_app.rag.memory.service import INJECTION_PREFIX

_USER = "s10-accept"


async def main() -> None:
    eng = create_engine()
    sm = create_sessionmaker(eng)
    mem = MemoryService(Embedder(), Reranker())
    client = AsyncOpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key, timeout=120.0)
    doc_a, proj_p, proj_other = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()

    async def purge() -> None:
        async with sm() as s:
            await mem.purge_user(s, _USER)
            await s.commit()

    await purge()

    # --- наполнение памяти под сценарии ---
    async with sm() as s:
        # doc+memory: вчерашняя сводка по документу A
        await mem.write_summary(
            s, mem.scope_for(_USER, document_id=doc_a, thread_id=uuid.uuid4()),
            "Пользователь спрашивал про сроки поставки оборудования по этому документу.")
        # project-glossary: термин Schedule в проекте P
        await mem.add_manual(
            s, mem.scope_for(_USER, project_id=proj_p), scope_kind="project", kind="glossary",
            content="Термин Schedule переводится как «График производства работ».")
        # conflict: устаревший факт в памяти (документ скажет иначе)
        await mem.add_manual(
            s, mem.scope_for(_USER), scope_kind="user", kind="fact",
            content="Срок поставки по старому обсуждению — 30 дней (устаревшее).")
        await s.commit()

    results: list[tuple[str, bool, str]] = []

    # S1 doc-only: память не нужна
    r1 = await classify(client, "Найди раздел про ответственность сторон", [])
    results.append(("S1 doc-only", not r1.needs_memory, f"route={r1.route} needs_memory={r1.needs_memory}"))

    # S2 doc+memory: нужна память + сводка документа достаётся
    r2 = await classify(client, "А что там с поставкой оборудования, что обсуждали?", [])
    scope_a = mem.scope_for(_USER, document_id=doc_a, thread_id=uuid.uuid4())
    async with sm() as s:
        _, hits2 = await mem.retrieve_block(s, "а что там с поставкой оборудования?", scope_a)
    got_doc_sum = any(h.scope == "document" and "поставк" in h.content.lower() for h in hits2)
    results.append(("S2 doc+memory", r2.needs_memory and got_doc_sum,
                    f"needs_memory={r2.needs_memory}, сводка документа={got_doc_sum}"))

    # S3 project-glossary: термин достаётся в своём проекте, изолирован в чужом
    async with sm() as s:
        _, hp = await mem.retrieve_block(s, "переведи пункт документа со словом Schedule",
                                         mem.scope_for(_USER, project_id=proj_p))
    in_p = any(h.kind == "glossary" and "schedule" in h.content.lower() for h in hp)
    async with sm() as s:
        _, ho = await mem.retrieve_block(s, "переведи пункт документа со словом Schedule",
                                         mem.scope_for(_USER, project_id=proj_other))
    in_other = any(h.kind == "glossary" and "schedule" in h.content.lower() for h in ho)
    results.append(("S3 project-glossary", in_p and not in_other,
                    f"в своём проекте={in_p}, в чужом={in_other}"))

    # S4 conflict: память подаётся как hints (§6.2), не как authority → документ побеждает
    async with sm() as s:
        block4, _ = await mem.retrieve_block(s, "какой срок поставки оборудования?", mem.scope_for(_USER))
    has_prefix = bool(block4) and INJECTION_PREFIX.split("\n")[0] in block4
    results.append(("S4 conflict (doc wins)", has_prefix,
                    f"память помечена как contextual-hints={has_prefix} (числа — из RAG-цитаты)"))

    # S5 agentic-multi-step
    r5 = await classify(
        client, "Вытащи все спецификации материалов и сравни с тем, что мы обсуждали вчера", [])
    results.append(("S5 agentic-multi-step", r5.route == "agentic_multi_step",
                    f"route={r5.route} mode={r5.mode}"))

    # --- порог latency: retrieve+gate p95 ---
    lat: list[float] = []
    for _ in range(20):
        async with sm() as s:
            t0 = time.monotonic()
            await mem.retrieve_block(s, "что я просил и какой срок поставки", scope_a)
            lat.append((time.monotonic() - t0) * 1000)
    p95 = statistics.quantiles(lat, n=20)[-1]
    p50 = statistics.median(lat)

    # --- cap'ы выдачи после gate ---
    async with sm() as s:
        _, caphits = await mem.retrieve_block(s, "срок поставки schedule сводка", scope_a)
    by_scope: dict[str, int] = {}
    for h in caphits:
        by_scope[h.scope] = by_scope.get(h.scope, 0) + 1
    caps_ok = all(by_scope.get(sc, 0) <= lim for sc, lim in
                  (("user", 5), ("project", 5), ("document", 5), ("thread", 3)))

    print("=== Сценарии §10 ===")
    for name, ok, detail in results:
        print(f"  {'✓' if ok else '✗'} {name:24} {detail}")
    print("\n=== Пороги §10 ===")
    print(f"  retrieve+gate: p50={p50:.0f} мс, p95={p95:.0f} мс  (порог ≤200) → {'✓' if p95 <= 200 else '✗'}")
    print(f"  cap'ы выдачи (≤5/5/5/3): {by_scope} → {'✓' if caps_ok else '✗'}")
    all_ok = all(ok for _, ok, _ in results) and p95 <= 200 and caps_ok
    print("\nИТОГ:", "ПРИЁМКА §10 ПРОЙДЕНА" if all_ok else "ЕСТЬ ОТКЛОНЕНИЯ — см. выше")

    await purge()
    await eng.dispose()


if __name__ == "__main__":
    asyncio.run(main())
