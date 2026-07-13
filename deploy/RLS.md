# RLS: fail-closed изоляция данных (ТЗ §4.7.1)

Фильтры владельца в приложении остаются первым контуром. PostgreSQL RLS является
независимым вторым контуром: пропущенный фильтр в маршруте не должен расширять
доступ. Боевой API работает только под отдельной ролью без административных
привилегий, а отсутствие принципала или ошибка установки контекста приводят к
отказу в доступе, а не к сервисному режиму.

## Роли и инварианты

| Роль | `SUPERUSER` | `BYPASSRLS` | Владелец защищённых таблиц | Назначение |
|---|---:|---:|---:|---|
| `rag` | да | да | да | только Alembic и административные DDL |
| `rag_api` | нет | нет | **нет** | FastAPI; RLS применяется ко всем защищённым таблицам |
| `rag_worker` | нет | да | нет | ARQ и системные задания между пользователями |

Правила эксплуатации:

- API нельзя запускать под `rag`, `rag_worker` или ролью-владельцем таблиц.
- Воркер использует `BYPASSRLS`, но получает только необходимые DML-права. Его
  DSN нельзя передавать API или пользовательским процессам.
- `audit_log` остаётся доступным API только на чтение и добавление: разрешены
  `SELECT` и `INSERT`, права
  `UPDATE` и `DELETE` отозваны. Системное обслуживание выполняет воркер.
- Alembic запускается отдельно под `rag`. После миграции все новые таблицы должны
  принадлежать `rag`, а права API и воркера выдаются явно в той же миграции.
- Между тремя ролями не должно быть членства, позволяющего API выполнить
  `SET ROLE` в привилегированную роль.
- Файлы `.env.api.local` и `.env.worker.local` имеют режим `0600`; значения DSN
  не выводятся в журналы и проверочные команды.

## Защищённый набор

Проверка запуска и проверочные запросы используют один полный набор из 16 таблиц:

1. Документы и производные данные: `documents`, `folders`, `chunks`, `segments`,
   `document_translations`, `segment_versions`,
   `document_structured_artifacts`, `page_embeddings`.
2. Чат: `chat_sessions`, `chat_messages`.
3. Аудит действий: `audit_log`.
4. Память: `memory_events`, `memory_items`, `memory_candidates`,
   `memory_audit_log`, `memory_item_sources`.

`glossary` намеренно не входит в этот набор: это общий утверждённый словарь для
всех аутентифицированных пользователей. Чтение и изменение словаря доступны
только после обязательной аутентификации и проверки роли приложения.

Политики прямых таблиц сравнивают `owner_sub` или `user_id` со строгим
пользовательским GUC. Производные таблицы проверяются через родительский документ,
чат или запись памяти. В `documents`, `folders` и `chat_sessions` владелец
обязателен на уровне `NOT NULL`; `NULL` не означает «общая строка». В
`memory_audit_log` tenant должен совпадать строго, а строку видит либо её точный
пользователь, либо аутентифицированный администратор в том же tenant. Поэтому
`user_id IS NULL` там доступен только такому администратору, но не анонимному
контексту. Системный воркер обходит политики своей ролью. Для каждой политики
обязательны и `USING`, и `WITH CHECK`.

До включения строгих политик проверяются `owner_sub IS NULL` и соответствие
владельцев по связям document-folder, chat-document и chat-folder. Все счётчики
нарушений должны быть равны нулю; владельца нельзя автоматически переназначать
для прохождения миграции.

На всех 16 таблицах используются `ENABLE ROW LEVEL SECURITY` и
`FORCE ROW LEVEL SECURITY`. API всё равно не должен владеть ими: это отдельный
инвариант минимальных привилегий, а не замена `FORCE`. Фактические значения
`relrowsecurity` и `relforcerowsecurity` всегда проверяются после миграции.

## Контекст запроса

Принципал по умолчанию недоверенный: `user_id` отсутствует, `is_admin=false`,
сервисный режим выключен. Аутентификация должна установить принципал до открытия
первой транзакции маршрута. Хук начала транзакции задаёт локальные GUC
`app.user_id` и `app.is_admin`; ошибка `set_config` прерывает транзакцию. Ошибку
нельзя проглатывать или заменять `app.is_admin=on`.

Пользователь без токена, с неверным `azp` или без роли приложения не доходит до
операций БД. Администратор получает `app.is_admin=on` только после полной проверки
OIDC-токена. После запроса значение `ContextVar` сбрасывается в `finally`, поэтому
контекст не переносится в соседний запрос или фоновую задачу.

Воркеру не нужен сервисный GUC для обхода пользовательских политик: обход задаёт
сама роль `rag_worker` с `BYPASSRLS`. Это отделяет право системной обработки от
данных входящего HTTP-запроса.

## Проверка запуска (startup gate)

API и воркер не должны переходить в состояние готовности, пока не проверена роль БД.
Проверка API обязана подтвердить:

- текущая роль не `SUPERUSER`, не `BYPASSRLS` и не является участником роли с
  этими привилегиями;
- все 16 таблиц существуют, API не владеет ни одной из них;
- `ENABLE` и `FORCE` включены, `row_security_active(table)=true`, политика
  существует для каждой таблицы.

Проверка воркера подтверждает `NOSUPERUSER` и `BYPASSRLS`. Владелец, точный состав
политик, строгий `NULL`, DML-права и ревизия Alembic проверяются миграционными
тестами и командами ниже. Любое расхождение startup gate является ошибкой
запуска, а не предупреждением. `/healthz` не считается успешной проверкой RLS
сам по себе: он используется только после успешного startup gate.

## Проверка без раскрытия секретов

Команды выполняются на A100 из `/root/projects/rag_app`. Они подключаются через
локальный Unix socket контейнера PostgreSQL и не читают и не печатают пароли.

### 1. Роли, членство и фактические подключения

```bash
docker exec -i rag-app-postgres-1 psql -X -U rag -d rag_app <<'SQL'
\pset pager off
SELECT rolname, rolsuper, rolcanlogin, rolcreaterole, rolcreatedb, rolbypassrls
FROM pg_roles
WHERE rolname IN ('rag', 'rag_api', 'rag_worker')
ORDER BY rolname;

SELECT pg_get_userbyid(member) AS member,
       pg_get_userbyid(roleid) AS granted_role,
       admin_option, inherit_option, set_option
FROM pg_auth_members
WHERE pg_get_userbyid(member) IN ('rag', 'rag_api', 'rag_worker')
   OR pg_get_userbyid(roleid) IN ('rag', 'rag_api', 'rag_worker')
ORDER BY member, granted_role;

SELECT usename, application_name, state, count(*)
FROM pg_stat_activity
WHERE datname = 'rag_app'
GROUP BY usename, application_name, state
ORDER BY usename, application_name, state;
SQL
```

Ожидается: API-соединения только от `rag_api`, соединения воркера от
`rag_worker`; членства между служебными ролями нет. `rag` может появляться только
на время административной проверки или миграции.

### 2. Полнота RLS, владельцы и FORCE

```bash
docker exec -i rag-app-postgres-1 psql -X -U rag -d rag_app <<'SQL'
\pset pager off
WITH expected(table_name) AS (
  VALUES
    ('documents'), ('folders'), ('chunks'), ('segments'),
    ('document_translations'), ('segment_versions'),
    ('document_structured_artifacts'), ('page_embeddings'),
    ('chat_sessions'), ('chat_messages'), ('audit_log'),
    ('memory_events'), ('memory_items'), ('memory_candidates'),
    ('memory_audit_log'), ('memory_item_sources')
)
SELECT e.table_name,
       COALESCE(pg_get_userbyid(c.relowner), '<missing>') AS owner,
       COALESCE(c.relrowsecurity, false) AS rls_enabled,
       COALESCE(c.relforcerowsecurity, false) AS rls_forced,
       count(p.polname) AS policies
FROM expected e
LEFT JOIN pg_class c
  ON c.oid = to_regclass(format('public.%I', e.table_name))
LEFT JOIN pg_policy p ON p.polrelid = c.oid
GROUP BY e.table_name, c.relowner, c.relrowsecurity, c.relforcerowsecurity
ORDER BY e.table_name;

WITH expected(table_name) AS (
  VALUES
    ('documents'), ('folders'), ('chunks'), ('segments'),
    ('document_translations'), ('segment_versions'),
    ('document_structured_artifacts'), ('page_embeddings'),
    ('chat_sessions'), ('chat_messages'), ('audit_log'),
    ('memory_events'), ('memory_items'), ('memory_candidates'),
    ('memory_audit_log'), ('memory_item_sources')
)
SELECT e.table_name AS violation
FROM expected e
LEFT JOIN pg_class c ON c.oid = to_regclass(format('public.%I', e.table_name))
WHERE c.oid IS NULL
   OR pg_get_userbyid(c.relowner) <> 'rag'
   OR NOT c.relrowsecurity
   OR NOT c.relforcerowsecurity
   OR NOT EXISTS (SELECT 1 FROM pg_policy p WHERE p.polrelid = c.oid)
ORDER BY e.table_name;
SQL
```

Первый запрос должен показать 16 строк, owner=`rag`, RLS=true и policy для каждой
таблицы. Второй запрос должен вернуть ноль строк. `rls_forced=true` обязателен для
всего защищённого набора.

### 3. Состав политик без вывода пользовательских данных

```bash
docker exec -i rag-app-postgres-1 psql -X -U rag -d rag_app <<'SQL'
\pset pager off
SELECT c.relname AS table_name,
       p.polname AS policy_name,
       CASE p.polcmd
         WHEN '*' THEN 'ALL' WHEN 'r' THEN 'SELECT' WHEN 'a' THEN 'INSERT'
         WHEN 'w' THEN 'UPDATE' WHEN 'd' THEN 'DELETE'
       END AS command,
       p.polpermissive AS permissive,
       p.polqual IS NOT NULL AS has_using,
       p.polwithcheck IS NOT NULL AS has_with_check
FROM pg_policy p
JOIN pg_class c ON c.oid = p.polrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public'
ORDER BY c.relname, p.polname;
SQL
```

Для защищённых таблиц не должно быть политик без `USING` или `WITH CHECK`.
Содержимое предикатов проверяется миграционными тестами; боевая команда
намеренно выводит только структуру policy, а не данные строк.

### 4. Запрет по умолчанию и реальное применение RLS

```bash
docker exec -i rag-app-postgres-1 psql -X -v ON_ERROR_STOP=1 \
  -U rag -d rag_app <<'SQL'
\pset pager off
BEGIN READ ONLY;
SET LOCAL ROLE rag_api;
SELECT current_user AS role,
       row_security_active('documents') AS documents,
       row_security_active('document_structured_artifacts') AS structured,
       row_security_active('chat_sessions') AS chat,
       row_security_active('memory_items') AS memory;
ROLLBACK;

BEGIN READ ONLY;
SET LOCAL ROLE rag_api;
SELECT set_config('app.user_id', '', true),
       set_config('app.is_admin', 'off', true),
       set_config('app.tenant_id', '', true),
       set_config('app.project_id', '', true),
       set_config('app.document_id', '', true);
SELECT
    (SELECT count(*) FROM documents)
  + (SELECT count(*) FROM folders)
  + (SELECT count(*) FROM chunks)
  + (SELECT count(*) FROM segments)
  + (SELECT count(*) FROM document_translations)
  + (SELECT count(*) FROM segment_versions)
  + (SELECT count(*) FROM document_structured_artifacts)
  + (SELECT count(*) FROM page_embeddings)
  + (SELECT count(*) FROM chat_sessions)
  + (SELECT count(*) FROM chat_messages)
  + (SELECT count(*) FROM audit_log)
  + (SELECT count(*) FROM memory_events)
  + (SELECT count(*) FROM memory_items)
  + (SELECT count(*) FROM memory_candidates)
  + (SELECT count(*) FROM memory_audit_log)
  + (SELECT count(*) FROM memory_item_sources)
  AS visible_without_principal;
ROLLBACK;

BEGIN READ ONLY;
SET LOCAL ROLE rag_worker;
SELECT current_user AS role,
       row_security_active('documents') AS documents,
       row_security_active('document_structured_artifacts') AS structured,
       row_security_active('chat_sessions') AS chat,
       row_security_active('memory_items') AS memory;
ROLLBACK;
SQL
```

Для `rag_api` все четыре значения должны быть `true`; для `rag_worker` —
`false`, поскольку его обход RLS является явным системным правом.
`visible_without_principal` должен быть равен нулю. Запрос выводит только
агрегат и не раскрывает пользовательские строки.

### 5. Alembic и сервисы

```bash
docker exec rag-app-postgres-1 \
  psql -X -U rag -d rag_app -Atc 'SELECT version_num FROM alembic_version'
curl -fsS http://127.0.0.1:8100/healthz
docker exec rag-app-redis-1 redis-cli LLEN arq:queue
docker exec rag-app-redis-1 redis-cli LLEN arq:structured-sidecar
```

После миграции выполняются leakage-тесты для владельца A, владельца B,
администратора, запроса без токена и токена с неверным `azp`. Приёмка: ноль
чужих документов, производных данных, чатов, записей аудита и памяти; ошибка
контекста RLS не расширяет доступ.
