# API и worker под systemd

`rag-api.service` и `rag-worker.service` заменяют только tmux-процессы приложения;
модельные сервисы и Docker Compose не затрагиваются. Оба unit-файла сначала
читают закрытый корневой `.env`, затем отслеживаемый `deploy/rag-runtime.env`.
Поэтому принятые несекретные режимы бюджета контекста, visual retrieval и
теневого парсинга одинаковы после интерактивного и автоматического рестарта.

## Окно переключения

1. Синхронизировать код и `.venv`, собрать SPA, проверить `.env` и выполнить
   `REPO_DIR=/root/projects/rag_app deploy/install_rag_app_services.sh --check`.
2. Дождаться окончания длинных ARQ-задач или явно зафиксировать активные job ID.
   Установить без запуска: `deploy/install_rag_app_services.sh --install`.
3. Остановить только tmux-сессии `rag_api` и `rag_worker`; убедиться, что порта
   `8100` и основного ARQ worker больше нет. Затем `systemctl start rag-app.target`.
4. Проверить `systemctl status rag-api rag-worker`, `/healthz`, effective flags в
   journal старта, постановку одного тестового задания и отсутствие двух worker,
конкурирующих за одну очередь.
5. После smoke сохранить journal и включить обычный мониторинг unit state.

Worker получает `SIGINT` и до 3600 секунд на graceful shutdown, чтобы systemd не
обрывал штатный длинный parse/translate при обычном restart. Это не отменяет
сторож зависших статусов: аварийный SIGKILL/перезагрузка хоста всё ещё должны
восстанавливаться прикладным контуром.

## Откат

`systemctl disable --now rag-app.target`, затем запустить прежние команды в tmux
с тем же `.env` и обязательно экспортировать четыре значения из
`deploy/rag-runtime.env`. Unit-файлы не меняют БД, Redis или MinIO; откат не
требует миграции. Не держать systemd и tmux одновременно.

Разделение очередей вводится отдельным циклом после этого cutover. Полный
порядок legacy → split → drain и откат описан в `deploy/QUEUE_ROLLOUT.md`;
`rag-split-workers.target` намеренно не входит в `rag-app.target`.
