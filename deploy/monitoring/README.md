# Наблюдаемость

Prometheus скрейпит API, exporter очередей/качества, все vLLM/parser endpoints,
DCGM, PostgreSQL, Redis и встроенный exporter MinIO. Grafana автоматически
загружает dashboards API и operations; правила находятся в `alerts.yml`.

Перед запуском:

1. Создать отдельного read-only пользователя PostgreSQL и передать его DSN как
   `RAG_METRICS_POSTGRES_DSN`. Exporter не должен использовать owner/API/worker
   роль. Создание роли относится к серверному окну и здесь не выполняется.
2. Получить bearer token встроенного MinIO metrics endpoint, сохранить только в
   `deploy/monitoring/secrets/minio_metrics_token` с правами `0600`. Каталог
   исключён из git.
3. Проверить доступность GPU внутри LXC. Если nested Docker не получает
   `/dev/nvidia*`, запустить тот же pinned `dcgm-exporter` напрямую на хосте/LXC;
   Prometheus target `127.0.0.1:9400` не меняется.
4. Выполнить `promtool check config prometheus.yml` и `promtool check rules
   alerts.yml`, затем поднять compose. Снаружи порты не публикуются.

Шаги 1–2 выполняет `sudo ./bootstrap_runtime.sh`: скрипт создаёт/ротирует только
роль `rag_metrics` (`CONNECT` + предопределенная read-only роль `pg_monitor`),
записывает DSN в игнорируемый `.env` и MinIO token в игнорируемый `secrets/`.
Значения не выводятся; оба файла имеют режим `0600`. Каталог/token принадлежит
runtime UID/GID `65534:65534` из pinned-образа Prometheus, чтобы не расширять
права чтения. Затем безопасный запуск без DCGM внутри LXC:

```bash
docker compose up -d prometheus postgres-exporter redis-exporter grafana
```

`dcgm-exporter` включается отдельно только после успешной проверки Docker GPU
runtime. Если runtime не зарегистрирован, сервис не запускать. Экспортёр на том
же LXC может сохранить target `127.0.0.1:9400`; host-level экспортёр на Proxmox
должен слушать доступный LXC адрес bridge, закрытый firewall для остальных, а
target в `prometheus.yml` должен быть изменён на этот адрес.
В unprivileged LXC одного `user: 0:0` недостаточно: ошибка DCGM
`Host engine is running as non-root` означает NO-GO внутри контейнера и требует
host-level DCGM, а не расширения привилегий всего monitoring compose.

MinIO использует собственный exporter, отдельный контейнер ему не нужен. Ни тела
заданий, ни имена файлов, ни document/user IDs не становятся Prometheus labels.
