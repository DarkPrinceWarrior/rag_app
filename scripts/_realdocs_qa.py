"""Одноразовый генератор retrieval-QA по конкретным документам (реальная
библиотека). Вопросы строятся из чанков заданных doc_id, эталон = chunk_id.
Корпус при оценке (eval_retrieval.py) — вся библиотека.

Запуск: PYTHONPATH=src .venv/bin/python scripts/_realdocs_qa.py <N> <id> [id ...]
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
import uuid

from openai import AsyncOpenAI
from sqlalchemy import select

from rag_app.config import settings
from rag_app.db.engine import create_engine, create_sessionmaker
from rag_app.db.models import Chunk

PROMPT = """\
Вот фрагмент технической документации:

---
{text}
---

Сформулируй ОДИН конкретный вопрос на русском языке, ответ на который содержится
именно в этом фрагменте (про числа, требования или условия из него).
Выведи только сам вопрос, без пояснений."""


async def main() -> None:
    n = int(sys.argv[1])
    ids = [uuid.UUID(x) for x in sys.argv[2:]]
    engine = create_engine()
    sm = create_sessionmaker(engine)
    async with sm() as s:
        chunks = list(
            (await s.execute(select(Chunk).where(Chunk.document_id.in_(ids)))).scalars().all()
        )
    await engine.dispose()
    rng = random.Random(3086)
    sample = rng.sample(chunks, min(n, len(chunks)))
    client = AsyncOpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key, timeout=120.0)
    sem = asyncio.Semaphore(8)

    async def gen(c: Chunk) -> dict:
        async with sem:
            text = (c.text_ru or c.text_en)[:2500]
            r = await client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": PROMPT.format(text=text)}],
                temperature=0.5,
                max_tokens=120,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return {"chunk_id": str(c.id), "question": (r.choices[0].message.content or "").strip()}

    qa = [x for x in await asyncio.gather(*(gen(c) for c in sample)) if len(x["question"]) > 10]
    with open("/tmp/realdocs_qa.jsonl", "w", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(x, ensure_ascii=False) for x in qa))
    print(f"chunks_in_scope={len(chunks)} qa={len(qa)} → /tmp/realdocs_qa.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
