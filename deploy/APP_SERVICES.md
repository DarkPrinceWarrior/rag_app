# API и worker под systemd

`rag-api.service` и `rag-worker.service` заменяют только tmux-процессы приложения;
модельные сервисы и Docker Compose не затрагиваются. Installer объединяет
действующий общий `.env` **без** owner-значения `RAG_DATABASE_URL` с
`.env.api.local`/`.env.worker.local`; итоговые `/etc/docragenslate/api.env` и
`worker.env` содержат ровно URL роли `rag_api` либо `rag_worker`. Поэтому auth,
S3, Langfuse и остальная production-конфигурация не теряются, а owner-роль `rag`
и compose-only `RAG_PG_*` не попадают в runtime. `PYTHON_DOTENV_DISABLED=1`
запрещает `observability.py` повторно подхватить корневой `.env`. Затем
загружается отслеживаемый `deploy/rag-runtime.env`.

Installer преобразует текущие `.env`, `.env.api.local` и `.env.worker.local` в
systemd-совместимый формат: удаляет только префикс `export`, отклоняет произвольный
shell-синтаксис, проверяет имя роли в `RAG_DATABASE_URL`, пишет через временный
файл и атомарный `mv`. Каталог имеет режим `0700`, файлы — `0600`; значения в
stdout/journal не выводятся. Owner URL из общего `.env` приложению не передаётся.
Exporter метрик получает только loopback-настройки Redis, запускается вне каталога
репозитория (чтобы `Settings.env_file=.env` не подхватил owner-env неявно) и не
читает DB/S3/OIDC env.

## Окно переключения

1. Синхронизировать код и `.venv`, собрать SPA, проверить role-specific env и выполнить
   `REPO_DIR=/root/projects/rag_app deploy/install_rag_app_services.sh --check`.
   Для отдельной подготовки закрытых файлов без установки units доступен режим
   `--prepare-env`; он также выполняется автоматически при `--install/--activate`.
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
с `.env.api.local` для API и `.env.worker.local` для worker и обязательно
экспортировать значения из
`deploy/rag-runtime.env`. Unit-файлы не меняют БД, Redis или MinIO; откат не
требует миграции. Не держать systemd и tmux одновременно.

Разделение очередей вводится отдельным циклом после этого cutover. Полный
порядок legacy → split → drain и откат описан в `deploy/QUEUE_ROLLOUT.md`;
`rag-split-workers.target` намеренно не входит в `rag-app.target`.
