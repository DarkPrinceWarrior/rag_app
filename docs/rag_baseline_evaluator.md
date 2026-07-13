# Локальный baseline-evaluator RAG

## Контракты

Эталон загружается только через `GoldRecord`/`load_gold_set`. Закрытый sidecar
имеет режим файла строго `0600` и загружается через
`PrivateSidecarRecord`/`load_private_sidecar` из
`src/rag_app/eval/private_sidecar.py`. Одна строка sidecar соответствует ровно
одному `case_id` и содержит:

- `gold_case_sha256` и `scope_id` для fail-closed привязки;
- `source_documents`: production UUID, стабильный `document_ref`, исходный язык;
- `exact_evidence`: `evidence_id`, production `chunk_id`, документ, страница,
  хэши chunk/цитаты и точная закрытая цитата;
- `retrieval_probe`: снимок top-8 для отрицательного кейса;
- `quantities.expected/supported`, классификацию и метаданные генерации.

`owner_sub` не сохраняется. Evaluator получает его read-only запросом по UUID
source documents, требует ровно одного владельца и проверяет
`make_scope_id(owner_sub) == GoldRecord.scope_id`. Значения sidecar, вопросы,
ответы, цитаты, object keys и `owner_sub` никогда не входят в сообщения об
ошибках и итоговый отчёт.

## Проверка production-снимка

До первого кейса каждого scope evaluator:

1. Разрешает owner только через loopback Postgres с
   `default_transaction_read_only=on`.
2. Загружает все `done` документы этого owner, хэширует реальные байты originals
   из loopback MinIO и полную исходную последовательность `text_en` chunks.
3. Требует точного совпадения множества `DocumentSnapshot`, включая page count и
   `parsed_content_sha256`.
4. Проверяет production chunk каждого exact evidence/probe: UUID, документ,
   позицию, heading/kind, хэш текста и вхождение точной цитаты.

Любое расхождение прекращает весь прогон. Retriever всегда получает найденный
`owner_sub`; admin-режим `owner_sub=None` запрещён. Все LLM/embedding/reranker,
MinIO и PostgreSQL endpoints обязаны быть literal loopback либо `localhost`.
Для baseline RRF-fallback отключён: ошибка production reranker прекращает весь
прогон, поэтому один отчёт не может незаметно смешать две схемы ранжирования.

## Метрики

Для `k = 1, 5, 10` считаются Recall@k, MRR@k и graded nDCG@k из существующего
`rag_metrics.py`. Дополнительно считаются citation precision/recall/F1,
сохранность пар число+единица, доля неподтверждённых чисел, answer/no-answer
accuracy и latency Retriever/Chat (mean, p50, p95). Answer/no-answer accuracy —
это детерминированная проверка ответа против мультиязычных маркеров отказа, а не
семантический judge качества полного ответа.

## Тип runner и agentic-RAG

Отчёт явно маркирует этот контур как `runner=retrieval_direct_answer`. Baseline
намеренно вызывает production `Retriever` напрямую для одинаковой ранжированной
выборки во всех hop-классах, затем production `ChatEngine` без истории и памяти.
Он ничего не пишет в chat/audit/memory таблицы и не выдаёт себя за agentic-RAG.
Показатель answerability фиксирует только корректность ответа/воздержания, а не
полную семантическую эквивалентность reference answer. Поэтому результаты этого
контура являются component baseline для retrieval, citations, чисел/единиц и
direct generation, но не итоговой оценкой всего adaptive agentic-RAG.

Сам `AgentLoop` и инструменты из `rag/agent.py` выполняют SELECT и технически
могут работать через read-only sessionmaker. Но production-ручка `/api/chat`
вокруг них записывает сообщения, сводки и memory events; кроме того,
`list_documents` регистрирует синтетический chunk, для которого текущие
evidence-метрики неприменимы. Поэтому реальный tool-loop следует измерять
отдельным read-only evaluator с собственными tool-trace/stop-reason метриками,
а не имитировать внутри этого baseline.

## Воспроизводимость

Поле `provenance` сохраняет только неприватные идентификаторы:

- режим и время начала оценки, Git SHA и признак dirty worktree; release-прогон
  разрешён только из чистой основной Git-копии;
- SHA-256 точных Gold JSONL и парного filtered sidecar;
- SHA-256 канонического набора `scope_id + DocumentSnapshot`, число scopes и
  снимков документов; production-содержимое уже проверено против этого набора;
- названия LLM, embedding, reranker и включённых visual-моделей, хеш ответа
  `/version`, хеш runtime process command line без раскрытия строки, полный
  SHA-256 manifest локальных config/index и файлов весов, их число и размер;
- `top_k`, retrieval limits/threshold, embedding dimension, бюджеты контекста,
  SHA-256 doc-only промпта, `temperature`, `top_p`, output tokens и стратегию seed;
- SHA-256 полного блока конфигурации.

В provenance не входят URL, ключи, имена файлов, `owner_sub`, вопросы, ответы,
цитаты или содержимое документов. Gold и sidecar повторно хэшируются после
прогона; production-снимок и model provenance также собираются повторно.
Изменение любого артефакта, корпуса, конфигурации, runtime или весов во время
оценки отклоняет весь отчёт. Seed детерминированно выводится из `case_id` в
зафиксированном namespace; production-вызовы пользователей сохраняют обычные
sampling-настройки без принудительного seed.

## Запуск

```bash
uv run python scripts/evaluate_rag_gold_set.py \
  .private/rag_gold/gold-release.jsonl \
  .private/rag_gold/gold-release.sidecar.jsonl \
  --top-k 10 \
  --report .private/rag_gold/baseline-report.json
```

`release` является режимом по умолчанию и требует, чтобы все записи прошли
review-gate. Второй позиционный аргумент в этом режиме — именно парный filtered
sidecar из `--release-sidecar-output`, а не полный sidecar candidate-набора:
множества `case_id` должны совпадать 1:1. `candidate` допускается только как явно
указанный режим для предварительных испытаний. Поскольку отчёт всегда содержит
метрики до `@10`, CLI принимает `top-k` только в диапазоне 10–64. Файл отчёта
записывается атомарно с правами `0600`.

В stdout выводятся только агрегаты без per-case данных. Локальные unit-тесты не
подключаются к Postgres, MinIO или моделям. Полный A100-прогон выполняется только
после выпуска совместимого sidecar генератором.
