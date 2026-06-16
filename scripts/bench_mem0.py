"""Mem0-ветка бенчмарка памяти (§15.5) — изолированный venv `/root/services/mem0-bench`.

Гоняет Mem0 OSS (self-hosted) на НАШЕМ стеке: LLM Qwen3.5 :8006 + эмбеддер
Qwen3-Embedding-0.6B :8002 + Chroma (встроенный вектор-стор). Тот же selftest,
тот же answer-промпт и тот же LLM-судья, что у `bench_memory.py` — отличается
ТОЛЬКО слой памяти (Mem0 add/search vs наш InternalAdapter). Числа сравнимы
напрямую с прогоном `bench_memory.py --selftest --judge`.

Запуск (на сервере): /root/services/mem0-bench/.venv/bin/python bench_mem0.py
"""

from __future__ import annotations

import os
import statistics
import time

os.environ.setdefault("OPENAI_API_KEY", "local")

import openai  # noqa: E402
from mem0 import Memory  # noqa: E402
from openai import OpenAI  # noqa: E402

# mem0 зовёт LLM без enable_thinking=false → Qwen3.5 уходит в <think>, выжигает
# бюджет и не отдаёт JSON-факты. Форсируем thinking-off на ВСЕ chat-вызовы
# процесса (наш answer/judge это и так делают).
_orig_create = openai.resources.chat.completions.Completions.create


def _create_no_think(self, *args, **kwargs):
    eb = dict(kwargs.get("extra_body") or {})
    eb.setdefault("chat_template_kwargs", {"enable_thinking": False})
    kwargs["extra_body"] = eb
    if os.environ.get("BENCH_DEBUG"):
        msgs = kwargs.get("messages", [])
        chars = sum(len(str(m.get("content", ""))) for m in msgs)
        print(f"[debug] LLM call: {len(msgs)} msgs, {chars} chars (~{chars // 4} tok), "
              f"max_tokens={kwargs.get('max_tokens')}")
        if chars > 5000:
            for m in msgs:
                c = str(m.get("content", ""))
                print(f"[debug]   role={m.get('role')} len={len(c)} head={c[:160]!r}")
                print(f"[debug]        tail={c[-160:]!r}")
    return _orig_create(self, *args, **kwargs)


openai.resources.chat.completions.Completions.create = _create_no_think

# Дефолтный ADDITIVE_EXTRACTION_PROMPT Mem0 2.0.6 — ~8.4к токенов, ОДИН не влезает
# в наш Qwen3.5 (max-len 8192) → add() падает, 0 памяти. Подменяем на короткий,
# сохраняя контракт вывода mem0: {"memory": [{"text": "<факт>"}, ...]}.
_SHORT_ADDITIVE = (
    "You are a memory extractor. From the conversation in the user message, extract "
    "durable user facts, preferences, agreements and constraints. Return ONLY a JSON "
    'object: {"memory": [{"text": "<concise self-contained fact>"}, ...]}. '
    'If nothing is worth storing, return {"memory": []}. Keep each fact short and in '
    "the same language as the input. No commentary, no markdown."
)
import mem0.configs.prompts as _m0p  # noqa: E402
import mem0.memory.main as _m0main  # noqa: E402

_m0p.ADDITIVE_EXTRACTION_PROMPT = _SHORT_ADDITIVE
_m0main.ADDITIVE_EXTRACTION_PROMPT = _SHORT_ADDITIVE

LLM_URL = "http://127.0.0.1:8006/v1"
LLM_MODEL = "qwen3.5-35b-a3b"
EMB_URL = "http://127.0.0.1:8002/v1"
EMB_MODEL = "qwen3-embedding-0.6b"

# тот же набор, что в scripts/bench_memory.py::_SELFTEST
_SELFTEST = [
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

client = OpenAI(base_url=LLM_URL, api_key="local")

# Дефолтный extraction-промпт Mem0 ~7.7к токенов — не влезает в наш Qwen3.5
# (max-len 8192). Короткий кастом (формат {"facts":[...]} Mem0 ожидает).
_MEM0_EXTRACT = """Ты извлекаешь устойчивые факты и предпочтения пользователя из диалога.
Верни строго JSON {"facts": ["короткое утверждение по-русски", ...]} — только конкретные
устойчивые факты, предпочтения и договорённости; если их нет, верни {"facts": []}.

Примеры:
Вход: Привет.
Выход: {"facts": []}
Вход: Отчёты присылай в формате XLSX, срок поставки 90 дней.
Выход: {"facts": ["Предпочитает отчёты в формате XLSX", "Срок поставки — 90 дней"]}

Верни только факты из следующего диалога в этом JSON-формате."""


def answer(question: str, memory_block: str | None) -> tuple[str, int]:
    system = _ANSWER_SYSTEM
    if memory_block:
        system += f"\n\n=== Память ===\n{memory_block}"
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": question}],
        temperature=0.0,
        max_tokens=200,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return (resp.choices[0].message.content or "").strip(), (resp.usage.total_tokens if resp.usage else 0)


def judge(question: str, gold: str, ans: str) -> bool:
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{
            "role": "user",
            "content": (
                f"Вопрос: {question}\nЭталонный факт: {gold}\nОтвет модели: {ans}\n\n"
                "Содержит ли ответ эталонный факт (по смыслу)? Ответь одним словом: ДА или НЕТ."
            ),
        }],
        temperature=0.0,
        max_tokens=5,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return "да" in (resp.choices[0].message.content or "").lower()


def main() -> None:
    config = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": LLM_MODEL,
                "openai_base_url": LLM_URL,
                "api_key": "local",
                "temperature": 0.0,
                "max_tokens": 1024,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": EMB_MODEL,
                "openai_base_url": EMB_URL,
                "api_key": "local",
                # без embedding_dims: наш Qwen3-Embedding не MRL, нативно отдаёт 1024
                # (mem0 иначе шлёт dimensions= и vLLM отвечает 400)
            },
        },
        "vector_store": {
            "provider": "chroma",
            "config": {"collection_name": "membench", "path": "/tmp/mem0_chroma_bench"},
        },
        "custom_fact_extraction_prompt": _MEM0_EXTRACT,
    }
    m = Memory.from_config(config)

    hits, tokens, ans_ms, search_ms = [], [], [], []
    for conv in _SELFTEST:
        uid = f"mem0bench-{conv['id']}"
        try:
            m.delete_all(user_id=uid)
        except Exception:
            pass
        msgs = [{"role": r, "content": c} for r, c in conv["turns"]]
        m.add(msgs, user_id=uid)
        for qa in conv["qa"]:
            q, gold = qa["q"], qa["gold"]
            t0 = time.monotonic()
            res = m.search(q, filters={"user_id": uid}, limit=5)  # mem0 2.x: filters=, не user_id=
            search_ms.append((time.monotonic() - t0) * 1000)
            rows = res["results"] if isinstance(res, dict) else res
            mems = [r.get("memory", "") for r in rows]
            block = "\n".join(f"- {x}" for x in mems) or None
            t1 = time.monotonic()
            ans, tok = answer(q, block)
            ans_ms.append((time.monotonic() - t1) * 1000)
            hit = judge(q, gold, ans)
            hits.append(1 if hit else 0)
            tokens.append(tok)
            mark = "✓" if hit else "✗"
            print(f"[mem0] {mark} {q[:42]:42} mem={len(mems)} {tok:4}т "
                  f"ans={ans_ms[-1]:5.0f}мс search={search_ms[-1]:5.0f}мс")
        try:
            m.delete_all(user_id=uid)
        except Exception:
            pass

    n = len(hits) or 1
    def p95(xs):
        return statistics.quantiles(xs, n=20)[-1] if len(xs) >= 2 else (xs[0] if xs else 0)
    print("\n=== Mem0 (тот же Qwen3.5/эмбеддер/судья) ===")
    print(f"mem0  acc={sum(hits)/n:.3f}  tokens={statistics.mean(tokens):.0f}  "
          f"answer_p95={p95(ans_ms):.0f}мс  search_p95={p95(search_ms):.0f}мс  (n={n})")


if __name__ == "__main__":
    main()
