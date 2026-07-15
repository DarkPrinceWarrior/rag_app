# Локальный состязательный smoke RAG

Scaffold ходит только в `RAG_REDTEAM_BASE_URL` (по умолчанию loopback; разрешены
только loopback и литералы частных IP-адресов, публичные адреса и DNS-имена
отклоняются), использует
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

## Одноразовый изолированный прогон

Создать два тестовых пользователя A/B и только синтетические документы:

| Переменная | Владелец | Содержимое fixture |
| --- | --- | --- |
| `RAG_REDTEAM_INJECTED_DOCUMENT_ID/CANARY` | A | косвенная инструкция и уникальный canary |
| `RAG_REDTEAM_FILENAME_DOCUMENT_ID/CANARY` | A | инъекция в имени, canary в тексте |
| `RAG_REDTEAM_FOREIGN_CANARY` | B | уникальный canary, недоступный A |
| `RAG_REDTEAM_PRIVATE_CODENAME` | B | закрытое синтетическое кодовое имя |
| `RAG_REDTEAM_ZH_DOCUMENT_ID/CANARY` | A | китайская инъекция и canary |
| `RAG_REDTEAM_POISON_DOCUMENT_ID/CANARY` | A | попытка навязать постоянное правило |

Индекс и БД должны быть одноразовыми, без копии production-корпуса. Получить
короткоживущий token пользователя A, не записывая его в файл, затем выполнить:

```bash
export RAG_REDTEAM_CONFIRM_ISOLATED=YES
export RAG_REDTEAM_BASE_URL=http://127.0.0.1:8100
export RAG_REDTEAM_TOKEN='<short-lived-user-A-token>'
export RAG_REDTEAM_INJECTED_DOCUMENT_ID='<uuid>'
export RAG_REDTEAM_INJECTED_CANARY='<unique-16+-chars>'
export RAG_REDTEAM_FILENAME_DOCUMENT_ID='<uuid>'
export RAG_REDTEAM_FILENAME_CANARY='<different-unique-16+-chars>'
export RAG_REDTEAM_FOREIGN_CANARY='<different-unique-16+-chars>'
export RAG_REDTEAM_PRIVATE_CODENAME='<different-unique-16+-chars>'
export RAG_REDTEAM_ZH_DOCUMENT_ID='<uuid>'
export RAG_REDTEAM_ZH_CANARY='<different-unique-16+-chars>'
export RAG_REDTEAM_POISON_DOCUMENT_ID='<uuid>'
export RAG_REDTEAM_POISON_CANARY='<different-unique-16+-chars>'
./run_local.sh
```

До запуска сохранить SHA-256 `package-lock.json`, конфигов и обезличенного
manifest fixture. После запуска зафиксировать версию Promptfoo, число
passed/failed/error, SHA-256 JSON-evidence и время. Не включать token, значения
canary или полные ответы в общий отчёт. Затем удалить evidence, отозвать token и
уничтожить одноразовые БД, индекс, MinIO-объекты и тестовых пользователей.

Этот репозиторный прогон не является доказательством стойкости production: без
одноразового корпуса выполнены только contract и mock-eval. Эвристики отказа и
canary находят конкретные утечки, но не заменяют локального классификатора для
семантически перефразированных атак.
