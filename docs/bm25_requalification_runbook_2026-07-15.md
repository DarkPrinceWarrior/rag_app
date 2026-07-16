# Повторная квалификация pg_textsearch BM25

Дата протокола: **15 июля 2026 года**. Это повтор пункта 9 по § 5.4
`improvement_plan_2026-07-15.md`. До результата `pg_textsearch` был кандидатом;
после NO-GO production остается на `postgres_fts` и `RAG_SPARSE_BACKEND` не
меняется.

## Итог прогона 16 июля 2026 года

Повтор выполнен из чистого snapshot `8b3e580` с policy SHA-256
`05c1cd8f597e68e5b7d3679295922a273270dc95a249a47096bed4c0a20595d5`.
Frozen Gold/sidecar имеют ожидаемые SHA-256 `c5ac4752...1578` и
`1635d3a5...c717`; 236 кейсов разделены на 21 tuning и 215 locked.
Direct Qwen3-Reranker-4B работал в `float32`, по одной паре, с тремя повторами;
runtime probe: relevant `0,999671`, irrelevant `0,0000238`, template SHA-256
`484cbad9...5118`. Во всех 430 locked-наблюдениях ранги детерминированы,
fallback не использовался. 619 case-артефактов прошли строгий schema-readback,
имеют парные HMAC-конверты и совпадающие artifact SHA-256. Ключ попытки после
офисной паузы отсутствовал, поэтому подписи конвертов независимо не
перепроверялись и эти case-файлы не объявляются подписанным итоговым отчетом;
отдельный финальный report не сформирован, потому что следующий load-gate
завершился fail-closed.

Точное кластерное locked-решение — **NO-GO**:

| Метрика | Baseline | Кандидат | Изменение | 95%-й интервал | Итог |
| --- | ---: | ---: | ---: | ---: | --- |
| Lexical Recall@5 | 0,00000 | 0,48311 | +0,48311 | [0,22727; 0,63954] | PASS, цель +0,03 |
| Final Recall@5 | 0,91554 | 0,92568 | +0,01014 | [0,00000; 0,02857] | PASS |
| Final nDCG@10 | 0,91175 | 0,91959 | +0,00784 | [0,00127; 0,02247] | **FAIL**, CI low < +0,02 |
| MRR@10 | 0,94566 | 0,94454 | −0,00113 | [−0,00368; 0,00000] | PASS по non-inferiority |
| No-answer abstention | 0,16418 | 0,14925 | −0,01493 | [−0,04478; 0,00000] | **FAIL**, CI low < −0,01 |

Нагрузочная стадия записала все 400 наблюдений (200 на backend), но 126
запросов завершились `RuntimeError` около клиентского тайм-аута 60 секунд при
concurrency 10. Поэтому общий release gate закономерно не создал подписанный
report. Перезапуск только с большим HTTP timeout не нужен: два locked
quality-gate уже независимо дают NO-GO. Временный runtime `:18003` остановлен,
GPU5 освобождена до 28 ГиБ, production `mineru-vllm.service` и `:30010`
проверены. `RAG_SPARSE_BACKEND` остается `postgres_fts`; production-данные и
индексы не менялись. Изолированный контейнер кандидата `rag-p9-bm25` остановлен
и оставлен в состоянии `exited` для возможного аудита; production PostgreSQL и
параллельный red-team стенд не затрагивались.

Обезличенные evidence-файлы на A100 имеют режим `0600`: runtime SHA-256
`d6ebca75...a98f`, reranker-attempt `aa09ce78...7e91`, raw load
`5cfe0e2f...a34`, fail-closed gate log `429dc540...6cc`.

## Что исправлено в методологии

Предыдущий прогон зависел от состава пакета реранкера и потому ошибочно требовал
стабильности сырых оценок. Новый протокол:

1. замораживает Gold/sidecar, снимок корпуса, policy, модельные ревизии, образы,
   расширение и определения индексов по SHA-256;
2. подаёт пары `(query, document)` строго последовательно, по одной, в порядке
   SHA-256 запроса и текста; временный vLLM обязан иметь `--max-num-seqs 1`;
3. использует `float32`; `float16` допустим только как отдельный fallback-прогон;
4. проверяет ранги, а не побитовое совпадение сырых scores: нижняя граница
   rank agreement $0{,}90$, Jaccard итогового множества $0{,}80$, доля смены
   итогового множества не выше $0{,}01$;
5. решение принимает по парным Recall/MRR/NDCG и полноте RRF-пула (`hybrid`),
   с доверительными интервалами и допусками policy. Хеши сырых scores служат
   только аттестацией входа, но не воротами GO/NO-GO;
6. вручную вставленный официальный Qwen3-Reranker template хешируется, а живой
   probe обязан отделить релевантный текст от нерелевантного минимум на $0{,}20$.

Контур умеет подписывать итоговый report и его attestation существующим HMAC,
но в этом NO-GO-прогоне итоговый report намеренно не создан: fail-closed
load-gate не пройден. Qualification требует чистый Git, свежий пустой work-dir,
236 кейсов, не менее 200 закрытых кейсов и неизменный runtime corpus fingerprint.

## Окно A100 и матрица команд

Не запускать временный реранкер одновременно с production-моделью на той же GPU.
Сначала согласовать окно, выбрать свободную карту по `nvidia-smi`, остановить
только согласованный временный/кандидатный процесс и создать отдельную tmux-сессию.
Ни один из этих шагов не меняет production unit.

| Профиль | `--dtype` | Условие | Work-dir |
| --- | --- | --- | --- |
| основной | `float32` | запускать первым | `.private/bm25-fp32` |
| fallback | `float16` | только если FP32 не помещается; отдельный полный прогон | `.private/bm25-fp16` |

Пример временного сервера (номер GPU и порт заменить после координации):

```bash
tmux new-session -d -s bm25_reranker_fp32 \
  "CUDA_VISIBLE_DEVICES=<FREE_GPU> /root/services/vllm-qwen32b/.venv/bin/vllm serve \
  /root/models/Qwen3-Reranker-4B \
  --served-model-name qwen3-reranker-4b \
  --runner pooling \
  --hf-overrides '{\"architectures\":[\"Qwen3ForSequenceClassification\"],\"classifier_from_token\":[\"no\",\"yes\"],\"is_original_qwen3_reranker\":true}' \
  --host 127.0.0.1 --port 18003 \
  --max-num-seqs 1 --dtype float32 --max-model-len 8192"
```

Серверный `--chat-template` намеренно не задаётся: приложение вставляет шаблон
в query/document само. Одновременное применение обоих шаблонов запрещено и
отклоняется сборщиком evidence.

Зафиксировать точный argv процесса без ручного переписывания и снять probe:

```bash
mkdir -p .private/bm25-fp32
pgrep -f 'vllm serve /root/models/Qwen3-Reranker-4B.*18003' > .private/bm25-fp32/pid
cp /proc/$(tr -d '\n' < .private/bm25-fp32/pid)/cmdline \
  .private/bm25-fp32/reranker.argv
chmod 600 .private/bm25-fp32/reranker.argv

RERANK_BASE_URL=http://127.0.0.1:18003 \
uv run python scripts/capture_reranker_runtime.py \
  --process-argv .private/bm25-fp32/reranker.argv \
  --output .private/bm25-fp32/reranker-runtime.json
```

Если probe не проходит, профиль — NO-GO; понижать разрыв нельзя. Для FP16
повторить с новым процессом, новым runtime evidence и новым пустым work-dir.

## Полный A/B

Подставить ранее подготовленные private Gold/sidecar, revision evidence, ключ и
эксплуатационные evidence. PostgreSQL-кандидат должен быть отдельным loopback
контейнером, собранным существующим `deploy/postgres-bm25/Dockerfile`; production
PostgreSQL не используется как место эксперимента.

```bash
RERANK_BASE_URL=http://127.0.0.1:18003 \
uv run python scripts/evaluate_retrieval_bm25.py \
  --mode qualification \
  --gold .private/rag-gold.json \
  --sidecar .private/rag-sidecar.json \
  --policy deploy/rag-eval/retrieval-policy-v2.json \
  --work-dir .private/bm25-fp32/run \
  --database-url postgresql+asyncpg://<API_RLS>@127.0.0.1:<CANDIDATE_PORT>/<DB> \
  --provenance-database-url postgresql+asyncpg://<PROVENANCE_RO>@127.0.0.1:<CANDIDATE_PORT>/<DB> \
  --database-container rag-postgres-bm25 \
  --extension-binary-path .private/pg_textsearch.so \
  --embedding-revision-evidence .private/embedding-revision.json \
  --reranker-revision-evidence .private/reranker-revision.json \
  --reranker-runtime-evidence .private/bm25-fp32/reranker-runtime.json \
  --hmac-key .private/retrieval-hmac.key \
  --evidence .private/rls-evidence.json \
  --evidence .private/update-evidence.json \
  --evidence .private/delete-evidence.json \
  --evidence .private/restart-evidence.json \
  --operational-evidence .private/operational-evidence.json \
  --load-evidence .private/load-evidence.json \
  --repeats 3
```

Протокол выбирает параметры на tuning-части, затем единожды открывает locked
часть. Сравнивать `baseline` и `candidate` только внутри одного `run_id`.
Обязательные срезы: RU/EN/ZH, single/multi/cross-document и типы содержимого.
`release_accepted=true` допустим только при одновременном GO локального locked
decision и общего retrieval gate. Даже GO означает «кандидат квалифицирован»;
production-переключение требует отдельного окна и решения владельца.

## Китайская лексическая ветка

`jieba` имеет лицензию MIT, но сейчас добавлять зависимость и флаг преждевременно:
в таблице `chunks` и candidate SQL нет отдельного jieba-сегментированного поля,
а индексы `pg_textsearch` построены только по RU/EN expressions. Разбиение лишь
запроса оставит документный корпус одним токеном и создаст ложный shadow-результат.

Безопасный следующий эксперимент должен отдельно добавить:

- материализованное и хешируемое поле китайских токенов для каждого chunk;
- его заполнение одной закреплённой версией `jieba` через `uv`, с MIT в реестре;
- отдельный candidate index и manifest SHA-256;
- режимы `off`/`shadow`, где shadow считает результаты, но не влияет на выдачу;
- собственный парный срез ZH на том же frozen corpus.

До выполнения этих условий Han-запросы намеренно остаются на `postgres_fts`, как
и проверяет текущий evaluator. Это не GO китайского BM25 и не production switch.
