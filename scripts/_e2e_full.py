"""Полноценный e2e по свежей библиотеке (HTTP к :8123, auth off).

Проверяет весь продукт ТЗ end-to-end на реальных EN-документах разных типов:
пайплайн (парс→перевод→экспорт→индекс), чат с цитатами (single + agentic),
tool list_documents, гибридный поиск, экстракцию таблиц, кросс-сессионную память.
Запуск на сервере: .venv/bin/python /tmp/e2e_full.py
"""

from __future__ import annotations

import asyncio
import json
import time

import httpx

BASE = "http://127.0.0.1:8123"
R: list[tuple[str, bool, str]] = []


def ok(name: str, cond: bool, detail: str = "") -> None:
    R.append((name, bool(cond), detail))
    print(f"  {'✓' if cond else '✗'} {name}" + (f" — {detail}" if detail else ""))


async def chat(c: httpx.AsyncClient, message: str, *, document_id=None, session_id=None,
               memory=True) -> dict:
    """POST /api/chat, разбор SSE. Возвращает {answer, citations, route, mode, session_id}."""
    body = {"message": message, "document_id": document_id, "session_id": session_id}
    url = "/api/chat" if memory else "/api/chat?memory=off"
    out = {"answer": "", "citations": [], "route": "", "mode": "", "session_id": session_id}
    async with c.stream("POST", url, json=body, timeout=180) as resp:
        async for line in resp.aiter_lines():
            line = line.strip()
            if not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
            t = ev.get("type")
            if t == "session":
                out["session_id"] = ev["session_id"]
            elif t == "mode":
                out["mode"], out["route"] = ev.get("mode", ""), ev.get("route", "")
            elif t == "delta":
                out["answer"] += ev.get("text", "")
            elif t == "done":
                out["citations"] = ev.get("citations", [])
            elif t == "error":
                out["answer"] += f"[ERROR: {ev.get('detail')}]"
    return out


async def main() -> None:
    async with httpx.AsyncClient(base_url=BASE) as c:
        # === A. Пайплайн: все документы обработаны ===
        print("\n=== A. Пайплайн (парс → перевод → экспорт → индекс) ===")
        docs = (await c.get("/api/documents", timeout=20)).json()
        by_name = {d["filename"]: d for d in docs}
        ok("документов в библиотеке = 7", len(docs) == 7, f"{len(docs)}")
        kinds = {d["kind"] for d in docs}
        ok("покрыты типы парсинга", {"pdf_text", "pdf_scan", "docx", "xlsx", "pptx", "text"} <= kinds,
           ",".join(sorted(kinds)))
        for d in sorted(docs, key=lambda x: x["filename"]):
            done = d["status"] == "done"
            tr = d["segment_count"] and d["translated_count"] >= d["segment_count"] * 0.98
            idx = d["chunk_count"] > 0
            exp = len(d["exports"]) > 0
            ok(f"{d['filename'][:28]:28} done/перевод/индекс/экспорт",
               done and tr and idx and exp,
               f"st={d['status']} {d['translated_count']}/{d['segment_count']} чанк={d['chunk_count']} exp={d['exports']}")

        done_docs = [d for d in docs if d["status"] == "done"]

        # === B. Чат single-hop по конкретному документу (с цитатами) ===
        print("\n=== B. Чат по документу (single-hop, цитаты) ===")
        for fn in ("oron_gas_pipeline_pressure_test.txt", "norsok_r002_lifting.pdf"):
            d = by_name.get(fn)
            if not d or d["status"] != "done":
                ok(f"чат по {fn[:24]}", False, "документ не готов")
                continue
            r = await chat(c, "Кратко: о чём этот документ и какие ключевые требования?",
                           document_id=d["id"])
            good = len(r["answer"]) > 80 and "не нашлось" not in r["answer"].lower()
            ok(f"ответ по {fn[:22]}", good, f"route={r['route']} цитат={len(r['citations'])} симв={len(r['answer'])}")

        # === C. Agentic: сведи в таблицу (markdown-таблица в ответе) ===
        print("\n=== C. Agentic-чат: сведи в таблицу ===")
        r = await chat(c, "Сравни требования к испытательному давлению в документах и сведи в таблицу.")
        has_table = "|" in r["answer"] and r["answer"].count("|") >= 4
        ok("agentic route", r["route"] in ("agentic_multi_step", "doc_plus_memory", "doc_only"), r["route"])
        ok("ответ оформлен таблицей (markdown)", has_table, f"'|' x{r['answer'].count('|')}")

        # === D. Tool list_documents: каталог библиотеки ===
        print("\n=== D. Tool list_documents ===")
        r = await chat(c, "Какие документы загружены в библиотеку? Перечисли названия.")
        names_hit = sum(1 for fn in by_name if fn.split(".")[0][:10].lower() in r["answer"].lower())
        ok("ответ из каталога (≥4 названий)", names_hit >= 4, f"совпало имён={names_hit}")

        # === E. Гибридный поиск по библиотеке ===
        print("\n=== E. Поиск по библиотеке (гибрид + reranker) ===")
        hits = (await c.get("/api/search", params={"q": "испытательное давление трубопровода"}, timeout=30)).json()
        ok("поиск вернул результаты", isinstance(hits, list) and len(hits) > 0, f"hits={len(hits) if isinstance(hits,list) else 'err'}")
        if isinstance(hits, list) and hits:
            ok("у результата есть документ+сниппет", bool(hits[0].get("filename") and hits[0].get("snippet")),
               hits[0].get("filename", ""))

        # === F. Экстракция таблицы (XLSX) ===
        print("\n=== F. Экстракция таблицы ===")
        xlsx = by_name.get("mechanical_equipment_schedule.xlsx")
        if xlsx and xlsx["status"] == "done":
            try:
                tbl = (await c.post("/api/extract/table",
                                    json={"query": "оборудование и его характеристики",
                                          "document_id": xlsx["id"]}, timeout=120)).json()
                ncols = len(tbl.get("columns", []))
                nrows = len(tbl.get("rows", []))
                ok("таблица извлечена", ncols >= 2 and nrows >= 1, f"колонок={ncols} строк={nrows}")
            except Exception as exc:
                ok("таблица извлечена", False, str(exc)[:80])
        else:
            ok("таблица извлечена", False, "xlsx не готов")

        # === G. Кросс-сессионная память ===
        print("\n=== G. Кросс-сессионная память (две сессии) ===")
        s1 = await chat(c, "Запомни моё предпочтение: технические отчёты всегда присылай в формате XLSX.")
        ok("сессия 1: подтверждение запоминания",
           "xlsx" in s1["answer"].lower() or "запом" in s1["answer"].lower(), s1["answer"][:60])
        # ждём асинхронную экстракцию+консолидацию памяти
        appeared = False
        for _ in range(20):
            mem = (await c.get("/api/memory", params={"scope": "user"}, timeout=15)).json()
            if any("xlsx" in (m.get("content", "").lower()) for m in (mem if isinstance(mem, list) else [])):
                appeared = True
                break
            await asyncio.sleep(3)
        ok("факт попал в память (user-scope)", appeared, "XLSX-предпочтение")
        s2 = await chat(c, "В каком формате я просил присылать технические отчёты?")
        ok("сессия 2 (новый чат): вспомнил из памяти", "xlsx" in s2["answer"].lower(),
           f"route={s2['route']} ответ={s2['answer'][:70]}")

        # === Итог ===
        passed = sum(1 for _, p, _ in R if p)
        print(f"\n=== ИТОГ e2e: {passed}/{len(R)} пройдено ===")
        for name, p, det in R:
            if not p:
                print(f"  ✗ {name} — {det}")
        print("РЕЗУЛЬТАТ:", "ВСЁ ЗЕЛЁНОЕ" if passed == len(R) else f"ОТКЛОНЕНИЯ: {len(R)-passed}")


if __name__ == "__main__":
    t0 = time.monotonic()
    asyncio.run(main())
    print(f"(e2e за {time.monotonic()-t0:.0f}s)")
