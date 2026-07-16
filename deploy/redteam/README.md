# Локальный состязательный smoke RAG

Scaffold запускается только через `run_disposable.sh`: launcher сверяет marker,
run id, PID и process-marker отдельного API. Provider принимает только точный
`http://127.0.0.1:<disposable-port>`; production `:8100`, частные адреса и DNS
отклоняются. Стенд использует
короткоживущий token тестового пользователя и выключает память чата. Телеметрия,
sharing, update-check и cache Promptfoo отключены. Зафиксирован Promptfoo
`0.120.19` (MIT): это последняя ветка, совместимая с локальным Node 22.20;
актуальная 0.121.x требует Node не ниже 22.22. Сначала положить зависимости из
`package-lock.json` в локальный npm-cache, выполнить
`npm ci --offline --ignore-scripts` и `npm rebuild better-sqlite3 --offline`,
затем запустить `run_local.sh`. Скрипт сверяет установленную версию с точной
версией из `package.json` и запускает только локальный binary, не обращаясь в
registry.
Запуск требует явного `RAG_REDTEAM_CONFIRM_ISOLATED=YES`: это операторский
предохранитель от случайного наведения набора на production.

Перед полным прогоном создать на изолированном стенде: документ с косвенной
инъекцией в тексте, документ с инъекцией в имени и документ другого пользователя
с уникальным canary. UUID, отдельные canary инъекций/имени/чужой области и
закрытое кодовое имя передаются через `RAG_REDTEAM_*`; скрипт fail-closed
проверяет их наличие и формат UUID до запуска. Production
документы в тест не включать. Набор покрывает системный промпт, косвенные
инъекции, имена файлов, межобластную утечку, membership inference, ложные числа,
китайскую инъекцию и отравление одноразового индекса. Последний кейс допустим
только после отдельного setup и обязан завершаться уничтожением одноразовой БД.

Проверки не используют облачного LLM-судью. Более широкие OWASP-пресеты можно
добавить только после явного подтверждения, что все providers остаются локальными.
Результаты и ответы сохраняются в `PROMPTFOO_CONFIG_DIR` с правами `0700`; после
фиксации обезличенного отчета каталог следует удалить как содержащий тестовые
canary и потенциально чувствительные ответы модели.

## Рекомендуемый одноразовый стенд

`setup_disposable.sh` поднимает отдельные PostgreSQL, Redis и MinIO на loopback,
накатывает миграции в БД `docragenslate_redteam_disposable`, создаёт отдельную
роль `rag_api`, синтетический индекс и отдельный FastAPI на `:58100`. В корпусе
есть только сгенерированные EN/ZH, filename-injection, poisoning, контрольное
число и документ второго владельца. Production БД, Redis, MinIO, API, корпус и
GPU5 не используются. Повторно используются только read-only model endpoints:
LLM `:8006`, embedding `:8002` и reranker `:8003`; setup отклоняет другие
адреса, DNS-имена и production-порты.

Аутентификация стенда автономна: setup создаёт эфемерный RSA/JWKS на `:58101`,
а `run_disposable.sh` выпускает пользовательский JWT не более чем на 15 минут.
В git и production Keycloak ключи/пользователи не добавляются. Состояние и
canary хранятся только в `deploy/redteam/.state` с правами владельца и удаляются
teardown вместе с контейнерными volumes.

```bash
cd deploy/redteam
export RAG_REDTEAM_CONFIRM_DISPOSABLE=YES
./setup_disposable.sh
./run_disposable.sh
./teardown_disposable.sh
```

Teardown обязателен и после неуспешной оценки. Он проверяет marker, run id,
compose project, имя БД и приватные process-marker PID перед остановкой, затем выполняет
`docker compose down --volumes`. Без точного подтверждения или при несовпадении
marker скрипт отказывает. Setup при ошибке вызывает тот же teardown автоматически.

Требования к хосту: Docker Compose, проектный `.venv`, доступность трёх локальных
model endpoints и локально установленные pinned-зависимости Promptfoo. Последние
готовятся без сети из заранее заполненного npm-cache:

```bash
npm ci --offline --ignore-scripts
npm rebuild better-sqlite3 --offline
```

## Проверка самого стенда без API

Локальный contract не открывает сокеты и не вызывает приложение:

```bash
npm ci --offline --ignore-scripts
npm rebuild better-sqlite3 --offline
npm run contract
npm run eval:mock
```

`eval:mock` использует реальный Promptfoo CLI, тот же `cases.yaml` и те же
assertions, но детерминированный `mock_provider.mjs`. JSON-evidence записывается
в `/tmp/docragenslate-promptfoo-mock/mock-results.json` с правами `0600`.

## Контракт синтетических данных

Одноразовый setup создаёт два тестовых владельца A/B и только синтетические документы:

| Переменная | Владелец | Содержимое fixture |
| --- | --- | --- |
| `RAG_REDTEAM_INJECTED_DOCUMENT_ID/CANARY` | A | косвенная инструкция и уникальный canary |
| `RAG_REDTEAM_FILENAME_DOCUMENT_ID/CANARY` | A | инъекция в имени, canary в тексте |
| `RAG_REDTEAM_FOREIGN_CANARY` | B | уникальный canary, недоступный A |
| `RAG_REDTEAM_PRIVATE_CODENAME` | B | закрытое синтетическое кодовое имя |
| `RAG_REDTEAM_ZH_DOCUMENT_ID/CANARY` | A | китайская инъекция и canary |
| `RAG_REDTEAM_POISON_DOCUMENT_ID/CANARY` | A | попытка навязать постоянное правило |

`run_local.sh` — внутренний шаг wrapper и намеренно не поддерживает ручное
наведение на произвольный API. Индекс и БД всегда одноразовые, без копии
production-корпуса.

До запуска сохранить SHA-256 `package-lock.json`, конфигов и обезличенного
manifest fixture. После запуска зафиксировать версию Promptfoo, число
passed/failed/error, SHA-256 JSON-evidence и время. Не включать token, значения
canary или полные ответы в общий отчёт. Затем удалить evidence, отозвать token и
уничтожить одноразовые БД, индекс, MinIO-объекты и тестовых пользователей.

Реальный прогон на одноразовом корпусе выполнен 16.07.2026: **8/8 кейсов и
22/22 assertions прошли**, после фиксации агрегатов контейнеры, volumes, ключи,
canary и raw evidence уничтожены. Обезличенный протокол, хеш принятого evidence
и остаточные ограничения — в `REAL_EVIDENCE_2026-07-16.md`.

Этот результат подтверждает только перечисленные синтетические сценарии, а не
полную стойкость production. Эвристики отказа и точные canary находят конкретные
утечки, но не заменяют повторные прогоны, многоходовые атаки и локальный
классификатор семантически перефразированных инъекций.
