# Приватный RAG gold set

## Назначение

Эталон содержит 200–500 вопросов по реальным сложным документам, но хранится
только вне репозитория либо в игнорируемом каталоге `.private/`. В git входят
только контракт, JSON Schema, валидатор и синтетические unit-тесты. Валидатор
не печатает вопросы, ответы, цитаты и неизвестные имена полей даже при ошибке.

Source of truth — модель `GoldRecord` в `src/rag_app/eval/gold_set.py`.
Машиночитаемая схема одного JSONL-объекта находится в
`docs/schemas/rag_gold_record.schema.json`. Межполевые и наборные ограничения
JSON Schema не выражает; их всегда проверяет Python-валидатор.

## Режимы допуска

- `candidate`: 200–500 валидных уникальных кейсов, не менее 20% настоящих
  no-answer (запрашиваемого факта нет), присутствует каждый обязательный класс.
  Допустимы статусы `candidate` и `reviewed`.
- `release`: все условия candidate, каждый обязательный класс представлен не
  менее чем 5 кейсами, каждый кейс имеет статус `reviewed`, а метаданные ревью
  привязаны к неизменяемому SHA-256 содержимого кейса.
- Leakage-пробы требуют отказа и намеренно не засчитываются в 20% no-answer.

Проверяются классы `single`, `multi`, `cross_document`; `text`, `table`,
`formula`, `figure`, `scan`; языки `ru`, `en`, `zh`; признаки `numbers`,
`units`, `standards`, `prompt_injection`, `leakage`.

## Стабильные ссылки

`DocumentSnapshot` не содержит имени файла или заголовка документа:

- `source_sha256` — SHA-256 исходных байтов объекта, прочитанных генератором
  непосредственно из MinIO; UUID документа, object key и имя файла не подходят;
- `parsed_content_sha256` — результат `parsed_chunks_sha256` для полной
  последовательности parse chunks документа: сортировка по `idx`, поля `idx`,
  `kind`, `heading_path`, `page_start`, `page_end`, канонический `text`; DB UUID,
  имя файла и переведённый текст в хэш не входят;
- `document_ref` — строго `doc-sha256:<source_sha256>`;
- `page_count` фиксирует проверяемую версию документа.

Текущая production-схема `documents` эти два хэша не хранит. Поэтому генератор
обязан до выпуска candidate прочитать исходный MinIO-объект, вычислить
`bytes_sha256(raw_bytes)`, загрузить все исходные parse chunks и вычислить
`parsed_chunks_sha256(...)`. Подстановка UUID/имени или хэша одного выбранного
чанка запрещена валидатором процесса и делает набор невоспроизводимым.

`EvidenceRef` также не содержит цитату. `content_sha256` считается по точному
каноническому фрагменту: NFC + LF + обрезка краёв для текста/таблицы/LaTeX,
либо по исходным байтам crop для рисунка/скана. Канонический идентификатор:

```text
ev-sha256:<source_sha256>:p<page>:<content_type>:<content_sha256>
```

Страница нумеруется с 1. Координаты `bbox`, если есть, нормализованы в `[0, 1]`
и задаются как `[x1, y1, x2, y2]`. Для `multi` нужны минимум две evidence-ссылки;
для `cross_document` — ссылки минимум на два разных хэша документов.

## Поля кейса

Обязательны все поля, включая явные `null` и пустые массивы. Неизвестные поля,
дубликаты JSON-ключей, неканонические хэши и неявные преобразования типов
запрещены. Для `answerable=false` `reference_answer`, его хэш и `evidence`
должны быть пустыми. Для `answerable=true` ответ, его хэш и evidence обязательны.

`question_sha256` и `reference_answer_sha256` считаются функцией `text_sha256`.
`case_id` — непрозрачный стабильный идентификатор вида `ragq-...`; в нём нельзя
кодировать название заказчика, документа или пользователя. Статус `reviewed`
требует `review.reviewer_id`, timezone-aware `reviewed_at` и `review.case_sha256`.
Последний вычисляется `gold_record_case_sha256(candidate_record)` до перевода
кейса в статус reviewed.

`scope_id` обязателен и имеет вид `scope-sha256:<64 hex>`. Генератор вычисляет
его как `make_scope_id(owner_sub)` и использует один scope при выборке всех
документов конкретного кейса, особенно `cross_document` и no-answer. Сырой
`owner_sub`, tenant/user UUID и пустое значение в JSONL не допускаются.

## Приватная генерация и продолжение

Длительный прогон использует обязательный идентификатор ревизии локальной
модели и приватный checkpoint по каждому слоту. Каталог checkpoint создаётся
внутри `--output-dir` с режимом `0700`; manifest, принятые слоты и курсоры
детерминированных отклонений имеют режим `0600`. Принятый слот предварительно
проходит строгий `GoldRecord`/sidecar bind. API-сбой не продвигает курсор.

```bash
uv run python scripts/generate_private_rag_eval.py \
  --output-dir /secure/rag_gold \
  --per-stratum 60 \
  --seed 2026071313 \
  --max-attempts 16 \
  --concurrency 2 \
  --model-revision <immutable-model-revision>
```

После fail-closed остановки тот же run продолжается явно; seed, corpus,
снимки документов, план, модель и версия контракта должны полностью совпасть.
Лимит попыток можно только увеличить:

```bash
uv run python scripts/generate_private_rag_eval.py \
  --output-dir /secure/rag_gold \
  --per-stratum 60 \
  --seed 2026071313 \
  --max-attempts 32 \
  --concurrency 2 \
  --model-revision <immutable-model-revision> \
  --resume
```

Повреждение, неизвестный файл, неверные права или несовпадение identity
останавливают resume до первого обращения к модели. Финальные JSONL и manifest
публикуются только после глобальной проверки уникальности, покрытия и привязки
всех кейсов. После успешной записи финальных артефактов checkpoint удаляется.

Если автоматический review оставил release без минимального покрытия отдельного
класса, исходные кейсы не редактируются и порог не снижается. Отдельный
checkpointed supplement генерирует новые случаи только для дефицитных классов,
заранее регистрирует все base-вопросы в глобальном реестре уникальности и
публикует новый объединённый candidate/sidecar. Manifest связывает результат с
SHA-256 исходных gold, sidecar и manifest, снимком production-корпуса, ревизией
модели и checkpoint lineage.

```bash
uv run python scripts/generate_private_rag_supplement.py \
  --base-gold /secure/rag_gold/base.jsonl \
  --base-sidecar /secure/rag_gold/base.generator.jsonl \
  --base-manifest /secure/rag_gold/base.manifest.json \
  --output-dir /secure/rag_gold/supplement \
  --standards-count 8 \
  --prompt-injection-count 10 \
  --leakage-count 10 \
  --model qwen3.5-35b-a3b \
  --model-revision <immutable-model-revision> \
  --llm-base-url http://127.0.0.1:8006/v1
```

Supplement использует `single_hop` для `standards`/`prompt_injection` и
`no_answer` для `leakage`. Входы обязаны иметь права `0600`, model endpoint и
PostgreSQL — быть loopback/read-only, а объединённый набор повторно проходит
полный candidate preflight и sidecar bind 1:1. При остановке продолжается тот же
checkpoint с `--resume`; после генерации весь объединённый candidate заново
проходит двухтраекторный review, старые вердикты не переносятся.

## Запуск

```bash
uv run python scripts/validate_rag_gold_set.py \
  .private/rag_gold/gold.jsonl --mode candidate

uv run python scripts/validate_rag_gold_set.py \
  /secure/rag_gold/gold.jsonl --mode release
```

Успешный вывод содержит только агрегированные счётчики. Автоматический gold-review
читает приватные UUID и точные цитаты из закрытого sidecar, но не добавляет их в
gold JSONL, отчёт или git. Входной sidecar может покрывать весь candidate-набор;
вместе с release создаётся отдельный filtered sidecar, чьи `case_id` совпадают
с release 1:1. Процедура выпуска описана в `docs/automated_rag_review.md`.
