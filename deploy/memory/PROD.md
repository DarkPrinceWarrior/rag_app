# Слой памяти — прод-доводка

Реализация: `docs/MEMORY_rev4_mem0_articles.md` §15. Код: `src/rag_app/rag/memory/`,
API `src/rag_app/api/routes/memory.py`, воркер `src/rag_app/workers/memory_tasks.py`,
миграции `0009`–`0012`.

## RLS: фазированный rollout (§3.7, §15.4)

Миграция `0012` включает на `memory_*` **ENABLE ROW LEVEL SECURITY** + политики
на GUC (`app.tenant_id` / `app.user_id` / `app.project_id` / `app.document_id`).
GUC выставляются через `set_config(..., is_local := true)` в транзакции запроса
(`src/rag_app/rag/memory/rls.py::apply_scope_guc`), вызовы вшиты во все
memory-пути (чат, CRUD, воркер).

Под **ENABLE** владелец таблиц (роль приложения `rag`) политики **обходит** —
поэтому деплой кода + миграции безопасен: даже при пропущенном GUC чат и память
работают. Это даёт защиту только от не-владельческих ролей.

### RLS FORCE — АКТИВЕН (2026-06-16)

Миграция `0013` включила FORCE на `memory_*` (`relforcerowsecurity=true`). Чтобы
FORCE действительно enforce'ил (суперпользователь Postgres обходит RLS даже под
FORCE), заведены **не-суперпользовательские роли** и сервисы репойнчены на них:

- **`rag_api`** (`NOSUPERUSER NOBYPASSRLS`) — роль API; RLS к ней применяется,
  GUC выставляется per-request (`apply_scope_guc` вшит во все memory-пути);
- **`rag_worker`** (`NOSUPERUSER BYPASSRLS`) — роль воркера; обходит RLS для
  кросс-юзерных job'ов (consolidation cron, retention purge).

Пароли — в gitignored `/root/projects/rag_app/.env.{api,worker}.local` (chmod 600).

**Проверено:** под `rag_api` запрос без GUC → 0 строк, с верным GUC → свои, с
чужим GUC → 0 (`scripts/_leakage_suite.py` = 0 утечек по 7 контекстам); чат
кросс-сессионно работает (rag_api + rag_worker); тест-остатков нет.

#### ⚠️ Канонический запуск сервисов (после reboot/рестарта — обязательно!)

Если поднять сервисы СТАРОЙ командой (под `rag`-суперюзером) — **FORCE молча
станет инертным**. Запускать ТОЛЬКО так (роль берётся из env-файла):

```bash
tmux new -d -s rag_api "cd /root/projects/rag_app && set -a && . ./.env.api.local && \
  /root/.local/bin/uv run uvicorn rag_app.api.main:app --port 8100 2>&1 | tee -a /tmp/rag_api.log"
tmux new -d -s rag_worker "cd /root/projects/rag_app && set -a && . ./.env.worker.local && \
  /root/.local/bin/uv run arq rag_app.workers.main.WorkerSettings 2>&1 | tee -a /tmp/rag_worker.log"
```

Проверка ролей: `SELECT usename FROM pg_stat_activity WHERE datname='rag_app'` →
должны быть `rag_api` (API) и `rag_worker` (воркер), не `rag`.

#### Ограничение под FORCE

`SET LOCAL` GUC — транзакционный: **нельзя писать строки нескольких пользователей
в одной транзакции** под `rag_api` (отложенный audit-INSERT проверяется на commit
против последнего GUC). Прод-пути это не нарушают (purge — один юзер на запрос,
воркер — BYPASSRLS). Откат FORCE — `alembic downgrade -1`.

## Retention / 152-ФЗ

- `memory_events.retention_until` — срок хранения эпизода; cron `consolidate_memory`
  (раз в 30 мин) удаляет истёкшие события (каскад чистит `memory_item_sources`) и
  помечает `deleted` авто-items без источников.
- `POST /api/memory/purge` (`{user_id}`; admin — любой, иначе только свой) — полное
  удаление памяти пользователя (право на забвение).
- `GET /api/memory/export?user_id=` — выгрузка памяти пользователя (право на доступ).

## Провайдер

`RAG_MEMORY_PROVIDER=internal` (нативный, по умолчанию). Mem0 — за `MemoryAdapter`,
включается на Этапе 4 после бенч-сравнения (`scripts/bench_memory.py`).
