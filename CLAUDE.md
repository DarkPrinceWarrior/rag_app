# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> `AGENTS.md` is kept byte-identical to this file (same instructions, same rules).
> Edit one, mirror the change to the other.
>
> ⚠️ **Описание проекта пока не заполнено** — см. раздел «Project Purpose (TODO)».
> Остальные разделы (инструменты, память, окружение, сервер, конвенции) — рабочие.

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
  with grep.
  > 🔧 Индекс ещё **не построен** (кода нет). После появления первого кода:
  > `codegraph init .`, затем `codegraph status` в начале сессии и
  > `codegraph sync` после bulk-изменений.
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

## Project Purpose (TODO)

> 🚧 **Заполнит Руслан.** Указать: назначение проекта, доменную область,
> источники и формат данных, ключевые компоненты, точки входа.

`rag_app` — RAG-приложение (Retrieval-Augmented Generation). Подробное описание
архитектуры и пайплайна будет добавлено сюда.

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
