# Автоматический выпускной шлюз RAG-моделей

## Назначение и границы

`scripts/compare_rag_baselines.py` сравнивает два закрытых
`rag-baseline-report-v1` и возвращает решение о выпуске одной новой модели.
Шлюз работает только для зафиксированного component-контура
`Retriever + direct ChatEngine`; он не выдает component-baseline за проверку
полного `AgentLoop`.

Политика v1 допускает замену только роли `llm` при неизменных коде,
prompt/config, корпусе, embedder и reranker. Прогонщик v1 не умеет честно
квалифицировать отдельную замену reranker, поэтому такая замена отклоняется, а
не считается поддержанной формально. Для reranker нужен отдельный парный
протокол с теневым индексом и измерением retrieval-нагрузки.

Дополнительные границы:

- новый embedding блокируется до появления теневого индекса: сравнение нового
  embedder по векторам production старой модели методологически неверно;
- visual-контур и полный agentic-RAG проверяются отдельными протоколами.

Обычный GitHub Actions не получает Gold, sidecar, ответы или A100-доступ. Он
выполняет только синтетические unit-тесты шлюза. Реальное решение формируется
локально на A100; итоговый JSON не содержит вопросов, ответов, case ID или текста
найденных фрагментов.

## Закрепленная база

`deploy/rag-eval/release-policy-v1.json` фиксирует:

- baseline report SHA-256
  `d9e2c72e1e325637a61d93aad8ddc9fcab30889443d5492ddc165d8cdedce23b`;
- Git SHA `52747e11ea39267d0b8094ef4b9ea1fa4a3c85bf`;
- Gold/sidecar, corpus/runtime snapshot и точную конфигурацию;
- разрешенную роль, лицензию `Apache-2.0`, пороги качества, нагрузки и отката.

Baseline и candidate обязаны быть сформированы одной чистой Git-ревизией,
равной `reference_git_sha` из policy. Новая версия кода, prompt,
retrieval-конфигурации, зависимостей или корпуса требует нового парного baseline
и новой версии policy. Молчаливое сравнение несопоставимых систем запрещено.
Подпись дополнительно закрепляет прямые исполняемые зависимости оценщика,
`pyproject.toml` и `uv.lock`; окружение A100 перед прогоном синхронизируется
командой `uv sync --frozen --group dev --extra parse`, чтобы не удалить
зафиксированный MinerU-контур из общего проектного venv.

`approved_model_licenses` намеренно пуст до появления конкретного candidate.
Даже модель с заявленной Apache-2.0 лицензией не пройдет только по строке SPDX:
перед A/B в отдельной версии policy закрепляются точный weight manifest, SHA
локального текста LICENSE и его первичный URL.

## Входы

Шлюзу обязательны восемь закрытых файлов:

1. Закрепленный baseline report.
2. HMAC-аттестация baseline report.
3. Candidate report, снятый той же Git-ревизией evaluator на том же Gold и
   корпусе.
4. HMAC-аттестация candidate report.
5. Закрытый Gold release и его закрытый sidecar.
6. `rag-model-qualification-raw-v1`, сформированный автоматическим
   A/B-прогонщиком.
7. HMAC-аттестация qualification.

Policy не передается параметром: comparator читает только
`deploy/rag-eval/release-policy-v1.json` и сверяет его с вкомпилированным
SHA-256. Отдельно передаются root-owned HMAC-ключ и путь нового решения.

Все закрытые входы должны быть обычными файлами, не symlink, с режимом `0600`
в каталоге без group/world-доступа. Внутри репозитория они допустимы только под
`.private/`. Хеши повторно проверяются непосредственно перед публикацией решения.

Qualification связан с SHA обоих отчетов, Gold/sidecar, Git прогонщика и
манифестом весов candidate.
Он обязан содержать фактические числовые результаты, а не ручной флаг `passed`:

- локальную лицензию и SHA текста LICENSE для точного weight manifest;
- минимум 30 длинных случаев RU/EN/ZH на 85--95% окна без overflow, OOM и
  truncation;
- не менее 200 запросов от 10 параллельных клиентов после прогрева: 0 ошибок,
  рестартов и OOM, p95 не хуже парного control более чем на 10%, throughput не
  ниже 90%;
- неизменяемого локального judge, prompt SHA и автоматическую semantic/safety/
  standards-проверку; prompt-injection и leakage должны пройти 100%;
- репетицию отката к reference не дольше 600 секунд, полный упорядоченный trace,
  повторное измерение Git/config/corpus/манифестов весов после команды отката,
  пробы всех активных модельных ролей и минимум 10/10 smoke.

Отсутствие любого блока означает отказ. Qualification нельзя заполнять вручную:
его формирует конкретный модельный A/B-harness, который записывает измерения и
provenance до возврата GPU в production.

HMAC защищает целостность артефактов и связывает их с конкретной root-owned
средой A100. Это не независимая подпись внешнего утверждающего лица: root,
имеющий доступ к ключу, остается границей доверия. Поэтому ключ хранится вне
репозитория с режимом `0600`, а выпуск дополнительно требует чистого Git,
закрепленной policy и сохранения decision SHA в журнале.

## Качество и статистика

Шлюз сначала пересчитывает все агрегаты из per-case метрик и отклоняет NaN,
Infinity, строки вместо чисел, изменившуюся eligibility и неполный provenance.
Затем выполняет 20 000 детерминированных парных bootstrap-выборок внутри страт
`language × hop/no-answer × answerable`.

Для всех guardrail-метрик используется односторонняя нижняя граница с поправкой
Bonferroni на семейство метрик. Целевая метрика обязана одновременно:

1. улучшиться не менее практического порога;
2. иметь нижнюю границу отдельного 95% парного интервала строго выше нуля.

Политика v1:

| Метрика | Максимальное ухудшение |
|---|---:|
| answerability accuracy | 2 п.п.; абсолютный floor 0,85 |
| Recall@1 | 2 п.п. |
| Recall@5, Recall@10 | 1 п.п. |
| MRR@10, nDCG@10 | 1 п.п. |
| citation precision/recall | 2 п.п. |
| quantity+unit accuracy/recall | 2 п.п. |
| unsupported-number rate | +1 п.п.; абсолютный ceiling 0,40 |
| последовательный p95 | меньшее из +10% и +250 мс |

Цель для `llm` -- `quantity_unit_accuracy`, практический минимум `+0,02`.
Дополнительно проверяются Recall@10 и
nDCG@10 по языкам, hop-type и типам контента, answerability по языкам и строгий
leakage-abstention без цитат и числовых утверждений.

Такая декомпозиция соответствует выводу RAGChecker о необходимости раздельно
диагностировать retrieval и generation, а ARES -- измерять relevance,
faithfulness и answer relevance. Bootstrap confidence bounds применяются и в
VERA; для сравнения NLP-систем важна именно разность с доверительным интервалом,
а не одно число без неопределенности. Источники:
<https://arxiv.org/abs/2408.08067>,
<https://arxiv.org/abs/2311.09476>,
<https://arxiv.org/abs/2409.03759>,
<https://aclanthology.org/2022.lrec-1.640/>.

## Запуск

Эталон создается из чистого immutable checkout ревизии оценщика. На A100
используется локальная роль `rag` из `.env`, потому что роль `rag_api` обязана
применять пользовательский RLS-контекст и без него ничего не видит. Сам evaluator
включает `default_transaction_read_only=on` и не выполняет записи:

```bash
set -a
. ./.env
set +a
uv run --no-sync python scripts/evaluate_rag_gold_set.py \
  /root/parser_trials/rag_eval_v1/release.jsonl \
  /root/parser_trials/rag_eval_v1/release.sidecar.jsonl \
  --mode release \
  --report /root/parser_trials/rag_eval_v1/reference.json \
  --attestation /root/parser_trials/rag_eval_v1/reference.attestation.json \
  --attestation-key /root/.config/docragenslate/rag-release-hmac.key
```

После создания baseline и candidate, raw qualification и их подписей запускается
сам шлюз:

```bash
uv run python scripts/compare_rag_baselines.py \
  /root/parser_trials/rag_eval_v1/reference.json \
  /root/parser_trials/rag_eval_v1/candidate.json \
  /root/parser_trials/rag_eval_v1/release.jsonl \
  /root/parser_trials/rag_eval_v1/release.sidecar.jsonl \
  /root/parser_trials/rag_eval_v1/qualification.json \
  --baseline-attestation /root/parser_trials/rag_eval_v1/reference.attestation.json \
  --candidate-attestation /root/parser_trials/rag_eval_v1/candidate.attestation.json \
  --qualification-attestation /root/parser_trials/rag_eval_v1/qualification.attestation.json \
  --attestation-key /root/.config/docragenslate/rag-release-hmac.key \
  --output /root/parser_trials/rag_eval_v1/release-decision.json
```

Коды завершения:

- `0` -- кандидат принят;
- `2` -- сопоставимый кандидат отклонен по quality/qualification gate;
- `3` -- входы недействительны, изменены или несопоставимы;
- `4` -- операционная ошибка.

Output создается только как новый файл, атомарно, с режимом `0600`. Повторный
запуск не перезаписывает прежнее решение. Перед production-переключением в
`docs/roadmap.md` записываются decision SHA, candidate weight SHA, точная policy,
метрики, решение и проверенный откат.

## Ограничения v1

- Direct baseline не измеряет route/tool trace, число шагов и stop reason агента.
- Semantic/safety передаются из отдельного автоматического qualification-run;
  они не выводятся из retrieval/citation метрик.
- Embedding-кандидат запрещен до shadow DB/index.
- Пороговый шлюз не заменяет canary и наблюдение production после переключения.
