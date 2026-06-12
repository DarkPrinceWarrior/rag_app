"""Retrieval-бенч эмбеддеров на реальной библиотеке (§ 12.1 шаг 3).

Фаза 1 (один раз): из случайных чанков прод-LLM генерирует по вопросу,
ответ на который содержится именно в этом чанке (chunk_id = эталон).
Кэш в --qa-file — ОДИН набор вопросов для всех сравниваемых эмбеддеров.

Фаза 2: эмбеддим все чанки (EN и RU) и вопросы (с инструкцией) заданным
эмбеддером, косинус в памяти (max по двум языкам — как LEAST в проде),
метрики recall@1 / recall@5 / MRR@10 + время эмбеддинга корпуса.

--truncate-dim N — MRL-усечение (отрезать и L2-нормировать): сравнение
качества при dim, влезающем в HNSW-лимит pgvector (2000).

Запуск (на сервере):
  uv run python scripts/eval_retrieval.py --make-qa 60          # фаза 1
  uv run python scripts/eval_retrieval.py --url http://127.0.0.1:8002/v1 --model qwen3-embedding-0.6b
  uv run python scripts/eval_retrieval.py --url http://127.0.0.1:8006/v1 --model qwen3-embedding-4b
  uv run python scripts/eval_retrieval.py --url ... --model qwen3-embedding-4b --truncate-dim 1024
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import time
from pathlib import Path

from openai import AsyncOpenAI
from sqlalchemy import select

from rag_app.config import settings
from rag_app.db.engine import create_engine, create_sessionmaker
from rag_app.db.models import Chunk

QA_PROMPT = """\
Вот фрагмент технической документации:

---
{text}
---

Сформулируй ОДИН конкретный вопрос на русском языке, ответ на который содержится
именно в этом фрагменте (про числа, требования или условия из него).
Выведи только сам вопрос, без пояснений."""


async def load_chunks() -> list[Chunk]:
    engine = create_engine()
    sessionmaker = create_sessionmaker(engine)
    async with sessionmaker() as session:
        chunks = list((await session.execute(select(Chunk).order_by(Chunk.idx))).scalars().all())
    await engine.dispose()
    return chunks


async def make_qa(n: int, qa_file: Path) -> None:
    chunks = await load_chunks()
    rng = random.Random(3086)  # фиксированный seed — воспроизводимый набор
    sample = rng.sample(chunks, min(n, len(chunks)))
    client = AsyncOpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key, timeout=120.0)
    sem = asyncio.Semaphore(8)

    async def gen(chunk: Chunk) -> dict:
        async with sem:
            prompt = QA_PROMPT.format(text=(chunk.text_ru or chunk.text_en)[:2500])
            resp = await client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=120,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return {"chunk_id": str(chunk.id), "question": (resp.choices[0].message.content or "").strip()}

    qa = await asyncio.gather(*(gen(c) for c in sample))
    qa = [x for x in qa if len(x["question"]) > 10]
    qa_file.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in qa), encoding="utf-8")
    print(f"вопросов: {len(qa)} → {qa_file}")


def _norm(vec: list[float], dim: int | None) -> list[float]:
    if dim:
        vec = vec[:dim]
    s = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / s for x in vec]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


async def evaluate(url: str, model: str, qa_file: Path, truncate_dim: int | None) -> None:
    chunks = await load_chunks()
    qa = [json.loads(line) for line in qa_file.read_text(encoding="utf-8").splitlines()]
    client = AsyncOpenAI(base_url=url, api_key="local", timeout=300.0)

    async def embed(texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 32):
            batch = [t.strip()[:8000] or "." for t in texts[i : i + 32]]
            resp = await client.embeddings.create(model=model, input=batch)
            out.extend(_norm(d.embedding, truncate_dim) for d in resp.data)
        return out

    t0 = time.monotonic()
    emb_en = await embed([c.text_en for c in chunks])
    emb_ru = await embed([c.text_ru for c in chunks])
    corpus_sec = time.monotonic() - t0

    instr = settings.embed_query_instruction
    q_texts = [f"Instruct: {instr}\nQuery: {x['question']}" for x in qa]
    q_emb = await embed(q_texts)

    ranks: list[int | None] = []
    for x, qv in zip(qa, q_emb, strict=True):
        sims = [max(_dot(qv, e), _dot(qv, r)) for e, r in zip(emb_en, emb_ru, strict=True)]
        order = sorted(range(len(chunks)), key=lambda i: -sims[i])
        rank = next((pos + 1 for pos, i in enumerate(order) if str(chunks[i].id) == x["chunk_id"]), None)
        ranks.append(rank)

    n = len(ranks)
    r1 = sum(1 for r in ranks if r == 1) / n
    r5 = sum(1 for r in ranks if r and r <= 5) / n
    mrr = sum(1 / r for r in ranks if r and r <= 10) / n
    dim = len(q_emb[0])
    print(
        f"{model}{f' (trunc {truncate_dim})' if truncate_dim else ''} | dim={dim} | "
        f"вопросов={n}, чанков={len(chunks)}\n"
        f"  recall@1={r1:.3f}  recall@5={r5:.3f}  MRR@10={mrr:.3f}  "
        f"индексация корпуса: {corpus_sec:.1f} c"
    )


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--make-qa", type=int, default=0)
    p.add_argument("--qa-file", default="/tmp/retrieval_qa.jsonl")
    p.add_argument("--url")
    p.add_argument("--model")
    p.add_argument("--truncate-dim", type=int, default=None)
    args = p.parse_args()

    qa_file = Path(args.qa_file)
    if args.make_qa:
        await make_qa(args.make_qa, qa_file)
        return
    if not (args.url and args.model):
        raise SystemExit("нужно --url и --model (или --make-qa N)")
    await evaluate(args.url, args.model, qa_file, args.truncate_dim)


if __name__ == "__main__":
    asyncio.run(main())
