# Миграция ARQ без потери legacy-очереди

По умолчанию `RAG_QUEUE_ROLLOUT_MODE=legacy`: `JobRouter` сохраняет все новые
задания в `arq:queue`, поведение прежнее. Split-workers устанавливаются отдельно
и не входят в `rag-app.target`.

## Фазы

1. **Установка, legacy.** Установить `rag-queue-worker@.service` и
   `rag-split-workers.target`, оставить mode `legacy`. Старый `rag-worker`
   продолжает выполнять все функции и cron. Проверить `rag-pipeline:9108`.
2. **Пустые consumers.** Запустить `rag-split-workers.target`. Четыре worker
   слушают `arq:parse`, `arq:translate`, `arq:export-index`, `arq:memory`, но
   постановщик ещё пишет в legacy. Убедиться, что health-check keys обновляются.
3. **Split новых job.** В отдельном окне изменить только
   `RAG_QUEUE_ROLLOUT_MODE=split` и перезапустить API плюс workers. Уже лежащие
   job остаются в `arq:queue` и исполняются старым worker. Если legacy parse
   заканчивается после переключения, следующий translate уже маршрутизируется
   в профильную очередь.
4. **Drain.** Дождаться `rag_arq_queue_depth{queue="arq:queue"}=0`, отсутствия
   in-progress legacy job и хранения результатов дольше часа. Старый worker
   пока оставить: он единственный запускает recovery/consolidation cron.
5. **Закрепление.** После недели без DLQ/регрессий можно вынести cron в отдельный
   control-worker и только затем погасить legacy worker. Текущий scaffold этого
   шага намеренно не делает — иначе rolling migration дублировала бы cron.

## Идемпотентность и ошибки

Маршрутизатор не переписывает заданные `_job_id`. Основная цепочка использует
стабильные пары `document_id + parse_revision`; memory extraction —
`session_id + message_id`. Повторные пользовательские операции, которым нужен
новый запуск, сохраняют собственный run suffix.

Новые split-workers делают не более трёх попыток. Автоповтор разрешён только для
транспортных ошибок с экспоненциальной задержкой 15–300 секунд; бизнес-ошибка
сразу попадает в bounded DLQ. DLQ хранит только job ID, имя функции, номер
попытки и тип исключения — аргументы/текст документа не записываются. Максимум
1000 записей на очередь.

## Откат

Вернуть mode `legacy` и перезапустить постановщики. Новые job снова пойдут в
`arq:queue`; уже поставленные split job безопасно дренируют профильные workers.
Не перемещать Redis job вручную и не запускать одну job в двух очередях.
