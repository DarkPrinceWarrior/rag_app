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

MinIO использует собственный exporter, отдельный контейнер ему не нужен. Ни тела
заданий, ни имена файлов, ни document/user IDs не становятся Prometheus labels.
