# Автоматическая проверка RAG

Ручное рецензирование не используется. Десять пилотных учётных записей нужны
только для автоматических HTTP/RLS smoke-тестов.

## Два независимых контура

1. **Gold review** проверяет качество самого `GoldRecord`: вопрос, эталонный
   ответ, точные доказательства либо корректность no-answer. Только этот контур
   может выпустить новый release JSONL.
2. **Production RAG evaluation** сравнивает фактический `CandidateOutput`
   приложения с уже выпущенным gold set. Его метрики не меняют и не блокируют
   существование gold set.

CLI ниже запускает первый контур. Целевые данные берутся только из
`GoldRecord`; приватные UUID, точные цитаты, retrieval probe и аннотации
чисел/единиц берутся из общего `PrivateSidecarRecord` (`private-rag-generator-v1`).
Sidecar проверяется `bind_gold_sidecar` по `case_id`, `scope_id`,
`gold_record_case_sha256`, документам и evidence locator.

## Проверки

До вызова модели код fail-closed проверяет:

- точное вхождение `exact_quote` в текст чанка из read-only PostgreSQL;
- хеши цитаты и полного текста, UUID чанка и retrieval probe;
- единственного владельца всех документов в памяти и соответствие
  `make_scope_id(owner_sub)` значению GoldRecord;
- покрытие доказательств, no-answer для RU/EN/ZH, числа и единицы;
- отсутствие расхождений GoldRecord и закрытого sidecar.

Затем одна локальная Qwen выполняет два повторных прогона с разными строгими
заданиями и seed: `evidence_first` и `claim_by_claim`. Это одна модель, поэтому
статистическая независимость оценок не утверждается. При расхождении запускается
`adjudicator`; ошибка или невалидный ответ означает `reject`.

Judge получает доверенные `challenge_tags` и `content_types`, но они не служат
доказательством правильности и не разрешают автоматический `accept`. Вопрос и
контексты остаются недоверенными. Для `prompt_injection` и `leakage` наличие
атакующего текста является частью теста: пример допустим только когда ответ
игнорирует инструкцию, не раскрывает закрытые данные и независимо проходит все
проверки ответа, evidence, answerability и scope. Выполнение атаки означает
`reject`.

Принятые записи копируются без изменения содержимого, получают
`status=reviewed` и `ReviewMetadata` с `auto-qwen-consensus-v1` либо
`auto-qwen-adjudicated-v1`. После этого набор целиком повторно проходит
`validate_gold_set(mode="release")`. Если осталось меньше 200 записей или
потеряно минимальное покрытие классов, release-файл не создаётся.

## Запуск

```bash
uv run python scripts/run_automated_rag_review.py \
  --gold-set .private/rag-eval/gold-candidate.jsonl \
  --private-sidecar .private/rag-eval/private-sidecar.jsonl \
  --report-output .private/rag-eval/automated-review-report.json \
  --release-output .private/rag-eval/gold-release.jsonl \
  --release-sidecar-output .private/rag-eval/gold-release.sidecar.jsonl
```

Входной private sidecar может содержать все candidate-кейсы. При успешном
выпуске `--release-sidecar-output` получает только записи с `case_id` из
release; перед записью они повторно привязываются к release 1:1 через
`bind_gold_sidecar`.

Judge URL обязан указывать на loopback. Отчёт, release и парный filtered
sidecar каждый пишутся атомарно с правами `0600`; приватные тексты в отчёт не
копируются. Код возврата `2` означает, что ни release, ни release-sidecar не
созданы.

Оба входа обязаны быть обычными файлами (не symlink) с точными правами `0600`.
Все три выходных пути должны отсутствовать до старта и не совпадать. При
успешном gate файлы заранее полностью формируются во временных `0600`-файлах,
затем публикуются группой: filtered sidecar, release как маркер готовности пары,
отчёт. Ошибка на любом шаге удаляет всю частично опубликованную группу. Поэтому
повторный запуск всегда получает новый пустой каталог, а не переиспользует пути
предыдущего запуска.

## Production baseline

После выпуска парного release запускается read-only baseline с `top_k >= 10`:

```bash
uv run python scripts/evaluate_rag_gold_set.py \
  .private/rag-eval/gold-release.jsonl \
  .private/rag-eval/gold-release.sidecar.jsonl \
  --mode release \
  --top-k 10 \
  --report .private/rag-eval/baseline.json
```

PostgreSQL-соединения используют `default_transaction_read_only=on`; MinIO и
локальные model endpoints только читаются. До первого кейса и после последнего
повторно проверяются полные снимки owner scope: исходные и разобранные документы,
тексты, UUID и локаторы чанков, текстовые/визуальные embeddings и закрытые
evidence/retrieval-probe. Любое расхождение останавливает выпуск отчёта.

Provenance содержит SHA-256 входных артефактов, ожидаемого и фактического
корпуса, конфигурации RAG, стабильных `/v1/models` metadata и локального manifest
`config.json`/tokenizer/index для каждой работающей модели. Локальные пути,
owner, вопросы, ответы, контекст и цитаты в baseline-отчёт не попадают.
