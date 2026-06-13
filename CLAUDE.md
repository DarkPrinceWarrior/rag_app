# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> `AGENTS.md` is kept byte-identical to this file (same instructions, same rules).
> Edit one, mirror the change to the other.

## Tooling

Code navigation uses three MCP servers, each with one job — do not duplicate them:

- **fff** — locate files and literal text (strings, comments, log messages).
  Use fff instead of shell `find`/`grep`/`rg`. One bare identifier per query;
  after two greps, read the code.
- **codegraph** — structural questions over a tree-sitter symbol graph.
  `codegraph_context "<task>"` is the primary tool (entry points + related
  symbols + code in one call). Also `codegraph_search` (symbol by name —
  prefer over `fff grep`), `codegraph_callers`/`codegraph_callees`,
  `codegraph_impact` (blast radius before a refactor), `codegraph_node`,
  `codegraph_explore`. Trust its results — full AST parse; do not re-verify
  with grep. `codegraph status` в начале сессии, `codegraph sync` после
  bulk-изменений.
- **serena** — LSP-precise symbol navigation and the only tool that *edits*
  at symbol level (`find_symbol`, `get_symbols_overview`,
  `find_referencing_symbols`, `replace_symbol_body`, `insert_*`,
  `rename_symbol`, `safe_delete_symbol`). Prefer over reading whole files.
  Project language: **python** (см. `.serena/project.yml`).

Cycle: locate (fff / `codegraph_search`) → understand (`codegraph_context`) →
assess risk (`codegraph_impact`) → read and edit (serena) → verify.

Other MCP / plugins: **context7** for version-sensitive library docs (prefer
over web search); **tavily** for general web search; **playwright** for browser
smoke-checks of any web UI.

## Memory (Honcho)

Use Honcho as the memory layer for this repository. Before answering questions
about project preferences, working rules, prior decisions, or remembered
context, consult Honcho in addition to this file and local repository docs.

Current Honcho MCP tools: `get_peer_card`, `set_peer_card`, `list_conclusions`,
`create_conclusions`, `chat`, `schedule_dream`.

Separate confirmed facts from inference. Treat files and command outputs as
confirmed; treat Honcho memory and architectural guesses as inference unless
verified locally.

## Project Purpose

**Корпоративный инструмент перевода и анализа технической документации (EN→RU)**
по ТЗ № 3086-СлЗап(333). Домен: нефтегаз / строительство / договоры.
Жёсткое ограничение: всё on-premise, только open-weight модели, ни один байт
документов не покидает периметр. Полное ТЗ и архитектура — `docs/roadmap.md`.

Продукт из трёх частей:
1. **Веб-приложение** — загрузка документов, перевод с сохранением структуры,
   side-by-side просмотр, правки, RAG-чат с документом, библиотека, экспорт.
2. **Браузерное расширение (WXT, MV3)** — перевод выделения/страницы.
3. **Бэкенд-платформа** — пайплайн: парсинг (MinerU/PaddleOCR-VL) → перевод
   (vLLM: Qwen3-32B-AWQ, Hunyuan-MT-7B) → реконструкция (BabelDOC / OOXML /
   python-docx) → индексация (pgvector, BGE-M3) → RAG-чат.

Ключевые компоненты и точки входа:
- `src/rag_app/api/main.py` — FastAPI-приложение (REST + примитивный UI).
- `src/rag_app/workers/` — ARQ-воркеры (parse / translate / export).
- `src/rag_app/pipeline/` — парсинг, сегментация, перевод, DOCX-экспорт.
- `src/rag_app/db/models.py` — схема Postgres (documents, segments, …).
- `docker-compose.yml` — инфраструктура: Postgres 17 + pgvector, Redis, MinIO.
- `deploy/` — systemd-юниты vLLM и скрипты раскладки моделей по GPU (см.
  `docs/roadmap.md` § 4.3: GPU0–1 Qwen3-32B, GPU2 Qwen3-VL, GPU3 Hunyuan-MT,
  GPU4 TEI/OCR, GPU5 резерв).

Статус: **все 5 этапов MVP завершены** (пайплайн перевода PDF/OOXML/сканов,
adaptive agentic-RAG-чат с цитатами — классификатор single/multi-hop + tool-цикл
`rag/agent.py`, § 5 п.7 — библиотека, браузерное расширение `extension/` на WXT,
SSO/RBAC/аудит/Langfuse, нагрузочное 20 документов). Оставшиеся доделки и план
обновления моделей — `docs/roadmap.md` § 12.1 и журнал (последняя строка 📋).
LLM-сервисы на a100: Qwen3-32B-AWQ GPU0 `:8001`, Hunyuan-MT-7B GPU3 `:8004`,
Qwen3-Embedding-0.6B GPU4 `:8002`, Qwen3-Reranker-4B GPU4 `:8003`,
Qwen3-VL-Embedding-8B GPU5 `:8007` — визуальный поиск по сканам
(systemd-юниты из `deploy/`; история замен моделей — roadmap § 12.1).
Инфраструктура: compose в корне (Postgres `:5433`, Redis, MinIO `:9000`,
Keycloak `:8180`) + `deploy/langfuse/` (`:8200`) + `deploy/monitoring/`
(Prometheus `:9090`, Grafana `:3001`, скрейпит публичный `/metrics` API);
API `:8100` (tmux `rag_api`), воркер — tmux `rag_worker`.
`RAG_AUTH_ENABLED=true` (включён 2026-06-13 — OIDC в расширении готов,
`chrome.identity` PKCE; `/healthz`, `/metrics`, `/api/config` остаются
публичными). Prod-доводка Keycloak (TLS/AD) и раздачи расширения (MDM/GPO) —
scaffold в `deploy/keycloak/PROD.md` и `deploy/extension-policy/`. Sentry —
опционально через `SENTRY_DSN` (no-op без него).

## Setup

Окружение управляется через **`uv`**, Python **3.13**:

```bash
uv venv --python 3.13 .venv        # создать окружение
uv add <package>                   # добавить зависимость (пишет в pyproject + uv.lock)
uv sync                            # установить из uv.lock
uv run python <script>.py          # запуск в окружении проекта
```

Не использовать «голый» `pip` — только `uv`.

## Server workflow (a100)

**Рабочая модель:** сервер — основная рабочая копия и место вычислений.
Локальная копия — синхронизированное зеркало для MCP-навигации и правок.

- **SSH:** `ssh a100` (LAN `192.168.101.12`, из офиса/VPN) или
  `ssh a100-remote` (из любой сети, через jump host `jump-37`). Один физический
  хост `zeta` (Proxmox), окружение — контейнер **LXC 135**.
- **Проектная директория на сервере:** `/root/projects/rag_app/`; `uv` —
  `/root/.local/bin/uv`; окружение `uv venv --python 3.13 .venv`.
- **GPU:** 6× A100-SXM4-40GB. **NVLink физически отсутствует** → межкарточно
  только PCIe, P2P выключен (`NCCL_P2P_DISABLE=1` — это норма для бокса).
  GPU0 занят сервисом `whisperx` → использовать `CUDA_VISIBLE_DEVICES=1..5`.
  Для нескольких карт: предпочтительно **независимые задачи 1-на-GPU**, либо DDP
  **внутри одного NUMA-острова** (`{1,2}` ↔ node0 / `{3,4,5}` ↔ node1).
- **Long-running** запускать в `tmux new -d -s <name>`.
- **Перенос файлов:** `rsync` на сервере нет — только `scp -p`.

## Conventions

- `from __future__ import annotations` в начале модулей; type hints; `X | Y` (3.10+).
- Зависимости — только через `uv` (не `pip install` напрямую).
- **Сначала на сервере**, затем синхронизация в локальный репозиторий.
- Экспертные отчёты/документы — на русском, без англоязычного жаргона; формулы — в LaTeX.
- Рабочий журнал проекта — `docs/roadmap.md` (схема Plan → Act → Verify → Report).
