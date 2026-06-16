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

### ⚠️ КРИТИЧНО: суперпользователь обходит RLS даже под FORCE

Миграция `0013` включает FORCE на `memory_*` (`relforcerowsecurity=true`), и
app-level scope-фильтр + gate дают `leakage_rate=0` (доказано
`scripts/_leakage_suite.py`). **НО** роль `rag` (дефолтный `POSTGRES_USER`
docker-образа) — **суперпользователь Postgres, а суперпользователи обходят RLS
безусловно**, даже FORCE. Проверено: `SELECT … FROM memory_items` без GUC под
`rag` возвращает строки. Значит **FORCE сам по себе пока инертен** — изоляцию
реально держит app-уровень.

Чтобы RLS-второй-контур заработал, нужны НЕ-суперпользовательские роли
(требует решения по безопасности прод-БД):

1. **Не-суперюзер роль для API** (RLS применяется только к не-суперюзерам):

   ```sql
   CREATE ROLE rag_api LOGIN PASSWORD '...' NOSUPERUSER NOBYPASSRLS;
   GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO rag_api;
   GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO rag_api;
   ```

   API (`tmux rag_api`) → `RAG_DATABASE_URL` под `rag_api`. GUC выставляется
   per-request (`apply_scope_guc`, уже вшит во все memory-пути).

2. **BYPASSRLS роль для воркера** (consolidation/purge ходят кросс-юзерно):

   ```sql
   CREATE ROLE rag_worker LOGIN PASSWORD '...' NOSUPERUSER BYPASSRLS;
   GRANT ALL ON ALL TABLES IN SCHEMA public TO rag_worker;
   GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO rag_worker;
   ```

   Воркеру (`tmux rag_worker`) → `RAG_DATABASE_URL` под `rag_worker`.

3. Прогнать `scripts/_leakage_suite.py` ПОД `rag_api`-ролью + явный тест
   «без GUC → 0 строк» (под не-суперюзером он теперь действительно режет).
4. Smoke под нагрузкой (чат + `/api/memory/*`) — выдача не пустеет (GUC доходит).
   Откат — `alembic downgrade -1` (NO FORCE).

> ⚠️ Без шага 1 (не-суперюзер API) FORCE — только заявленное состояние схемы,
> фактическая изоляция = app-фильтр + gate (доказан leakage-suite). Системные
> джобы под RLS одного тенанта не запускать — только `BYPASSRLS` (§3.7).

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
