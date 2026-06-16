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

### Финальный flip в FORCE (после верификации)

1. Завести роль воркера с обходом RLS (consolidation/purge ходят кросс-юзерно):

   ```sql
   CREATE ROLE rag_worker LOGIN PASSWORD '...' BYPASSRLS;
   GRANT ALL ON ALL TABLES IN SCHEMA public TO rag_worker;
   GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO rag_worker;
   ```

   Воркеру (`tmux rag_worker`) задать `RAG_DATABASE_URL` под `rag_worker`.
   API (`tmux rag_api`) остаётся под `rag` (GUC выставляется per-request).

2. Прогнать smoke под нагрузкой (чат + `/api/memory/*`) — убедиться, что выдача
   не пустеет (значит GUC доходит до всех запросов).

3. Включить FORCE:

   ```sql
   ALTER TABLE memory_events     FORCE ROW LEVEL SECURITY;
   ALTER TABLE memory_items      FORCE ROW LEVEL SECURITY;
   ALTER TABLE memory_candidates FORCE ROW LEVEL SECURITY;
   ALTER TABLE memory_audit_log  FORCE ROW LEVEL SECURITY;
   ```

4. Adversarial leakage-suite (§10): запрос пользователя B не возвращает items A —
   `leakage_rate = 0`. Откат — `NO FORCE`.

> ⚠️ Не запускать системные джобы (consolidation/purge) под RLS одного тенанта —
> только `BYPASSRLS`-роль (§3.7). При single-org tenant константа `RAG_TENANT_ID`.

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
