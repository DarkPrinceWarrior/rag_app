# Резервное копирование и проверка восстановления

Контур состоит из трех независимых копий: PostgreSQL+WAL через pgBackRest,
versioned mirror MinIO на второй приемник и offline-export realm Keycloak.
`RAG_BACKUP_ROOT` должен быть отдельной точкой монтирования; скрипты прекращают
работу, если это обычный каталог на production-диске.

## Ввод в эксплуатацию

1. Смонтировать отдельный носитель, задать `RAG_BACKUP_ROOT` и собрать PostgreSQL
   с оверлеем: `docker compose -f docker-compose.yml -f
   deploy/backup/docker-compose.backup.yml build postgres`. Первый restart,
   `stanza-create` и `check` выполнять в окне; откат — исходный pinned image и
   удаление WAL-параметров оверлея, данные `pg_data` не меняются. Каталог
   `$RAG_BACKUP_ROOT/pgbackrest` заранее создать с владельцем UID/GID пользователя
   `postgres` внутри контейнера и правами `0750`.
2. Полная копия еженедельно: `pgbackrest_backup.sh full`; разностная ежедневно:
   `pgbackrest_backup.sh diff`. После каждой — `check` и внешний мониторинг
   возраста последней успешной копии/WAL.
3. Установить `/etc/docragenslate/backup.env` с правами `0600`, включить
   `minio-mirror.timer`. Источник и приемник получают versioning; mirror не
   распространяет удаления.
4. Раз в неделю в согласованное окно выполнять
   `keycloak_export.sh --maintenance-window`. База Keycloak также попадает в
   PostgreSQL-копию; realm-export — независимый переносимый слой конфигурации.
5. Не реже раза в квартал выполнять `restore_drill.sh` на отдельном хосте/сети и
   сохранять заполненное свидетельство. Критерий приемки — восстановление одного
   документа, его сегментов, индекса и файлов, успешные `/healthz` и RAG smoke в
   согласованные RPO/RTO. Целевые значения утверждает владелец инфраструктуры;
   scaffold намеренно не подменяет их придуманными числами.

Ни один скрипт не делает restore поверх production. Realm-export кратко
останавливает Keycloak и поэтому требует явного `--maintenance-window`.
