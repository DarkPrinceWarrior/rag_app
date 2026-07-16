# Селективная проверка цитат: квалификация и выпуск

Дата: **15 июля 2026 года**. Контур реализует пункт 12 в редакции § 5.3
`improvement_plan_2026-07-15.md`. Он не скачивает веса из приложения и не
разрешает внешние конечные точки: проверяющая модель обслуживается отдельным
локальным процессом.

## 1. Контракт проверяющего сервиса

Приложение отправляет пакет только на loopback URL:

```json
{
  "backend": "hhem",
  "model": "/models/hhem-2.1-open",
  "pairs": [
    {"claim": "Давление равно 5 МПа", "evidence": "...", "language": "ru"}
  ]
}
```

Ответ:

```json
{"scores": [0.97]}
```

Допустим также список чисел или объектов `{"score": 0.97}`. Число означает
вероятность поддержки утверждения источником и обязано лежать в $[0,1]$.
Одинаковый контракт используется тонкими адаптерами HHEM и LettuceDetect.
Сервис не должен логировать `claim` и `evidence`.

Перед квалификацией проверить:

```bash
curl -fsS http://127.0.0.1:8011/score \
  -H 'content-type: application/json' \
  -d '{"backend":"hhem","model":"local","pairs":[{"claim":"Давление 5 МПа","evidence":"Давление 5 МПа","language":"ru"}]}'
```

Ожидаются HTTP 200, один score и отсутствие исходного текста в журнале сервиса.
Веса заранее размещаются администратором в локальном каталоге A100; запуск с
идентификатором Hugging Face, который может инициировать сетевое скачивание, для
квалификации запрещён. Для HHEM проверить Apache-2.0 снимок и SHA-256; для
LettuceDetect — MIT снимок и SHA-256. Русский вариант допускается только после
дообучения по зафиксированному локальному снимку и отдельной оценки RU.

## 2. Переменные кандидата

```bash
export RAG_CITATION_VERIFICATION_MODE=off
export RAG_CITATION_VERIFIER_BACKEND=hhem
export RAG_CITATION_VERIFIER_URL=http://127.0.0.1:8011/score
export RAG_CITATION_VERIFIER_MODEL=/models/hhem-2.1-open
export RAG_CITATION_VERIFIER_THRESHOLD=0.70
export RAG_CITATION_VERIFIER_TIMEOUT_S=15
```

`off` нужен на первом прогоне, чтобы собрать неизменённые ответы. `shadow`
проверяет тем же дешёвым backend, но не меняет ответ. `selective` буферизует
ответ, удаляет только неподтверждённые утверждения и отказывает целиком лишь
когда не осталось ни одного поддержанного факта. Старый `enforce` сохранён для
воспроизводимости прежней квалификации и по-прежнему использует LLM-repair.

## 3. Сбор наблюдений на 236 кейсах

Запускать в отдельной `tmux`-сессии из серверной рабочей копии. Пути ниже должны
быть вне репозитория либо под игнорируемым `.private/`; артефакт имеет режим
`0600` и не содержит текста утверждений — только case id, язык, gold-флаг и
числовые scores.

```bash
uv run python scripts/evaluate_rag_gold_set.py \
  .private/rag-gold.json .private/rag-sidecar.json \
  --mode candidate --concurrency 1 \
  --report .private/rag-selective-off-report.json \
  --citation-calibration-observations .private/citation-scores-236.json
```

При сборе текущий строгий LLM-verifier служит teacher-меткой, а локальный
HHEM/Lettuce backend даёт score той же claim. Артефакт принимается только при
покрытии всех 236 кейсов.

## 4. Кривая «риск — покрытие» и порог

```bash
uv run python scripts/qualify_selective_citations.py \
  .private/citation-scores-236.json \
  --output .private/citation-calibration-report.json \
  --expected-cases 236 \
  --threshold-start 0.30 --threshold-stop 0.95 --threshold-step 0.01 \
  --answerability-target 0.85 \
  --semantic-precision-target 0.90
```

Скрипт выбирает порог с максимальным покрытием среди точек, где одновременно
$answerability\ accuracy \ge 0.85$ и $semantic\ precision \ge 0.90$. Код выхода
`2` означает NO-GO. Сверить общую кривую и отдельно RU/EN/ZH; общий GO не
перекрывает провал отдельного языка.

## 5. Shadow и выпуск

1. Зафиксировать выбранный порог, снимок весов, лицензию и хеш отчёта.
2. Запустить API с `RAG_CITATION_VERIFICATION_MODE=shadow`; проверить долю ошибок
   backend, задержку и распределение удаляемых claims без изменения ответов.
3. Повторить полный baseline. Требования: ошибок verifier нет, answerability не
   ниже $0.85$, semantic precision не ниже $0.90$ на 236 кейсах и без провала
   RU/EN/ZH.
4. Переключить на `selective`, повторить baseline и smoke вопросов с частично
   неподтверждённым ответом. В отчёте поле `selective_qualification_passed`
   обязано быть `true`.
5. Только после этого менять production-конфигурацию. Откат — режим `off`;
   старый `enforce` не использовать как аварийный режим, поскольку он уже дал
   неприемлемую долю полных отказов.

## 6. Фактическое решение 16.07.2026

Generated-answer shadow завершен на 236/236 закрытых Gold-кейсах. Qwen3.5
выполнил 474 запроса без транспортных ошибок и повторов, извлечено 889
утверждений: 620 teacher-supported и 269 teacher-unsupported. Получены следующие
ROC AUC по языкам:

| Backend | EN | RU | ZH |
| --- | ---: | ---: | ---: |
| HHEM-2.1-Open | 0,247 | 0,624 | 0,224 |
| LettuceDetect router | 0,188 | 0,474 | 0,380 |

Ни для одного языка и backend не найден порог, одновременно сохраняющий
answerability не ниже 0,85 и semantic precision не ниже 0,90. Решение:
**NO-GO**, production-режим остается `off`; пункты 2–4 раздела «Shadow и
выпуск» для этих кандидатов не выполняются, потому что базовый модельный шлюз
уже провален.

Приватный отчет: SHA-256
`838af752727e0ffc77b239e9861ddb04a9d1a6e89ce4c220869df8f62c81c44b`.
Каталог A100 — `0700`, все результаты и журнал — `0600`. Сохраненные JSON
содержат только непрозрачные case id, язык, булеву teacher-метку и score; ключи
с вопросами, ответами, утверждениями и evidence отсутствуют. Same-runtime Qwen
teacher допускается только как консервативный отрицательный шлюз: для будущего
GO обязательны независимая разметка и отдельная квалификация новой либо
дообученной RU/EN/ZH-модели.
