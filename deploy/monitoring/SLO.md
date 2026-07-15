# SLO DocRAGenslate

Первые 30 дней значения считаются стартовыми эксплуатационными целями и
пересматриваются только по production-данным. Плановые окна и заранее
согласованные квалификационные прогоны исключаются; аварийные рестарты входят.

| Сигнал | SLI | Стартовый SLO / guardrail | Окно |
| --- | --- | --- | --- |
| Доступность API | доля HTTP без 5xx при `up=1` | ≥99,5% | 30 дней |
| Латентность обычного API | p95 HTTP без потокового чата/загрузки | ≤2 с | 7 дней |
| Начало ответа чата | p95 vLLM TTFT Qwen3.5 | ≤5 с | 7 дней |
| Ожидание задания | доля времени, когда oldest age каждой очереди ≤10 мин | ≥99% | 7 дней |
| Надёжность стадий | completed / (completed + terminal errors) | ≥99% | 7 дней |
| Стандартный документ | p95 суммы parse+translate+export/index для PDF ≤50 страниц | ≤15 мин | 7 дней |
| Числа | unconfirmed / protected после entity guard | ≤1% | 7 дней |

Латентность считается по стадиям без `document_id` label, чтобы не создавать
высокую кардинальность и не раскрывать идентификаторы документов. Разрез одного
документа остаётся в журнале статусов/трейсе Langfuse; Prometheus хранит только
агрегаты stage/queue/service.

`rag_arq_*` и numerical-метрики относятся к заданиям после включения split.
До набора репрезентативного окна отсутствие ряда не считается выполнением SLO.
DLQ больше нуля — инцидент, а не допустимый error budget: запись сопоставляется с
логом по job ID, после устранения причины создаётся новая штатная операция.

## Основные запросы

```promql
# API error ratio
sum(rate(http_requests_total{status=~"5.."}[30d]))
/ clamp_min(sum(rate(http_requests_total[30d])), 0.001)

# p95 стадии
histogram_quantile(0.95,
  sum by (le, stage) (rate(rag_document_stage_duration_seconds_bucket[7d])))

# terminal error ratio
sum(rate(rag_arq_job_errors_total[7d]))
/ clamp_min(sum(rate(rag_arq_jobs_completed_total[7d]))
  + sum(rate(rag_arq_job_errors_total[7d])), 0.001)
```
