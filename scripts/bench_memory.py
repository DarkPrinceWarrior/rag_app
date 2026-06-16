"""Бенчмарк слоя памяти vs Mem0 (docs/MEMORY_rev4_mem0_articles.md §15.5, §8.9).

Прогоняет вопросы из мульти-сессионного датасета (LoCoMo / LongMemEval) через
наш стек и сравнивает baselines по трём осям: точность, токенов/запрос, p95.

Baselines:
  - none      — без памяти (нижняя граница: модель не знает прошлых сессий);
  - internal  — наш InternalAdapter + MemoryGate (целевой контур);
  - mem0      — провайдер Mem0 (только если RAG_MEMORY_PROVIDER=mem0 и Mem0 поднят).
  (summary-baseline и Mem0 включаются по мере готовности; §15.5.)

Числа Mem0-блогов как acceptance НЕ принимаем — только собственный прогон
на нашем Qwen3.5 + нефтегаз-русском (§14.9).

Формат датасета (--data JSON, список разговоров):
  [{"id": "...",
    "turns": [["user","..."],["assistant","..."], ...],
    "qa": [{"q":"...","gold":"..."}, ...]}]

Запуск (на сервере, БД и vLLM подняты):
  uv run python scripts/bench_memory.py --selftest
  uv run python scripts/bench_memory.py --data longmemeval.json --limit 50 --judge
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
import uuid
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from rag_app.config import settings
from rag_app.db.engine import create_engine, create_sessionmaker
from rag_app.db.models import MemoryCandidate
from rag_app.llm.embeddings import Embedder, Reranker
from rag_app.rag.memory import MemoryService
from rag_app.rag.memory.consolidate import consolidate_pending
from rag_app.rag.memory.events import record_event
from rag_app.rag.memory.extract import extract_candidates, is_injection
from rag_app.rag.memory.service import fingerprint

_THINK_OFF = {"chat_template_kwargs": {"enable_thinking": False}}

# Мини-набор в духе LongMemEval (факт упомянут в ранней сессии → вопрос позже).
_SELFTEST: list[dict[str, Any]] = [
    {
        "id": "s1",
        "turns": [
            ["user", "Меня зовут Руслан, я архитектор. Отчёты присылай в формате XLSX."],
            ["assistant", "Принято, отчёты буду присылать в XLSX."],
            ["user", "Срок поставки насосов по договору 3086 — 90 дней."],
            ["assistant", "Зафиксировал: срок поставки насосов — 90 дней."],
        ],
        "qa": [
            {"q": "В каком формате я просил присылать отчёты?", "gold": "XLSX"},
            {"q": "Какой срок поставки насосов по договору 3086?", "gold": "90 дней"},
        ],
    },
    {
        "id": "s2",
        "turns": [
            ["user", "Рабочая среда на объекте — сероводородная (sour service)."],
            ["assistant", "Учту: оборудование под сероводородную среду."],
            ["user", "Расчётное давление трубопровода — 16,5 МПа."],
            ["assistant", "Принято, расчётное давление 16,5 МПа."],
        ],
        "qa": [
            {"q": "Какая рабочая среда на объекте?", "gold": "сероводородная"},
            {"q": "Какое расчётное давление трубопровода?", "gold": "16,5"},
        ],
    },
]

_ANSWER_SYSTEM = (
    "Ты ассистент по технической документации. Отвечай кратко и точно по-русски."
    " Используй приведённую память о пользователе и проекте, если она относится к вопросу."
)


async def _answer(client: AsyncOpenAI, question: str, memory_block: str | None) -> tuple[str, int]:
    system = _ANSWER_SYSTEM
    if memory_block:
        system += f"\n\n=== Память ===\n{memory_block}"
    resp = await client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": question}],
        temperature=0.0,
        max_tokens=200,
        extra_body=_THINK_OFF,
    )
    text = (resp.choices[0].message.content or "").strip()
    tokens = resp.usage.total_tokens if resp.usage else 0
    return text, tokens


async def _judge(client: AsyncOpenAI, question: str, gold: str, answer: str) -> bool:
    """LLM-судья: ответ содержит правильный факт? (строгий yes/no)."""
    resp = await client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Вопрос: {question}\nЭталонный факт: {gold}\nОтвет модели: {answer}\n\n"
                    "Содержит ли ответ эталонный факт (по смыслу)? Ответь одним словом: ДА или НЕТ."
                ),
            }
        ],
        temperature=0.0,
        max_tokens=5,
        extra_body=_THINK_OFF,
    )
    return "да" in (resp.choices[0].message.content or "").lower()


def _match(gold: str, answer: str) -> bool:
    """Дешёвая метрика без LLM: эталон встречается в ответе (норм. регистр/пробелы)."""
    g = " ".join(gold.lower().split())
    return g in " ".join(answer.lower().split())


async def _ingest(memory: MemoryService, client: AsyncOpenAI, sm, scope, turns: list) -> None:
    """Разговор → memory_events → extract → consolidate (порог 0 — принять всё)."""
    async with sm() as session:
        event_ids = []
        for role, content in turns:
            ev = await record_event(
                session, scope, "message_user" if role == "user" else "message_assistant",
                role=role, payload={"content": content},
            )
            await session.flush()
            event_ids.append(str(ev.id))
        await session.commit()

    transcript = "\n".join(f"{r}: {c}" for r, c in turns)
    candidates = await extract_candidates(client, transcript)
    async with sm() as session:
        for c in candidates:
            content = (c.get("content") or "").strip()
            if not content or is_injection(content):
                continue
            session.add(
                MemoryCandidate(
                    tenant_id=scope.tenant_id, user_id=scope.user_id,
                    project_id=scope.project_id, document_id=scope.document_id,
                    thread_id=scope.thread_id, action=c.get("action", "create"),
                    proposed={**c, "source_event_ids": event_ids},
                    confidence=float(c.get("confidence", 0.5)),
                    fingerprint=fingerprint(c.get("kind", "fact"), c.get("scope", "user"), None, content),
                )
            )
        await session.commit()
        # для бенча принимаем всех извлечённых кандидатов
        await consolidate_pending(session, memory, tenant_id=scope.tenant_id, auto_threshold=0.0)
        await session.commit()


async def run(args: argparse.Namespace) -> None:
    data = _SELFTEST if args.selftest else json.loads(Path(args.data).read_text("utf-8"))
    if args.limit:
        data = data[: args.limit]

    engine = create_engine()
    sm = create_sessionmaker(engine)
    client = AsyncOpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key, timeout=300.0)
    memory = MemoryService(Embedder(), Reranker())
    baselines = args.baselines.split(",")

    bench_user = f"bench-{uuid.uuid4().hex[:8]}"
    results: dict[str, dict[str, list]] = {b: {"hits": [], "tokens": [], "ms": []} for b in baselines}

    try:
        for conv in data:
            thread = uuid.uuid4()
            scope = memory.scope_for(bench_user, thread_id=thread)
            await _ingest(memory, client, sm, scope, conv["turns"])

            for qa in conv["qa"]:
                q, gold = qa["q"], qa["gold"]
                for b in baselines:
                    block = None
                    if b == "internal":
                        async with sm() as session:
                            block, _ = await memory.retrieve_block(session, q, scope)
                    t0 = time.monotonic()
                    ans, tok = await _answer(client, q, block)
                    ms = (time.monotonic() - t0) * 1000
                    hit = await _judge(client, q, gold, ans) if args.judge else _match(gold, ans)
                    results[b]["hits"].append(1 if hit else 0)
                    results[b]["tokens"].append(tok)
                    results[b]["ms"].append(ms)
                    print(f"[{b:8}] {'✓' if hit else '✗'} {q[:48]:48}  {tok:4}т {ms:6.0f}мс")
    finally:
        # бенч-память не оставляем в БД
        async with sm() as session:
            await memory.purge_user(session, bench_user)
            await session.commit()
        await engine.dispose()

    print("\n=== Итог (точность / ср.токенов / p95 мс) ===")
    for b in baselines:
        r = results[b]
        n = len(r["hits"]) or 1
        acc = sum(r["hits"]) / n
        p95 = statistics.quantiles(r["ms"], n=20)[-1] if len(r["ms"]) >= 2 else (r["ms"][0] if r["ms"] else 0)
        print(f"{b:10} acc={acc:.3f}  tokens={statistics.mean(r['tokens']):.0f}  p95={p95:.0f}мс  (n={n})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Бенчмарк памяти vs Mem0 (§15.5)")
    ap.add_argument("--data", help="JSON датасета (LoCoMo/LongMemEval); иначе --selftest")
    ap.add_argument("--selftest", action="store_true", help="встроенный мини-набор без внешних данных")
    ap.add_argument("--limit", type=int, default=0, help="ограничить число разговоров")
    ap.add_argument("--baselines", default="none,internal", help="none,internal[,mem0]")
    ap.add_argument("--judge", action="store_true", help="судить точность LLM-судьёй (иначе substring)")
    args = ap.parse_args()
    if not args.selftest and not args.data:
        ap.error("нужен --data <json> или --selftest")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
