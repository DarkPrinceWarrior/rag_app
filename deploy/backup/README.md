# Резервное копирование и проверка восстановления

Контур состоит из трех независимых копий: PostgreSQL+WAL через pgBackRest,
versioned mirror MinIO на второй приемник и offline-export realm Keycloak.
`RAG_BACKUP_ROOT` должен быть отдельной точкой монтирования; скрипты прекращают
работу, если это `/`, symlink, обычный каталог или bind той же файловой системы,
что production root. `tmpfs`, `ramfs`, Docker overlay и другие недолговременные
псевдо-ФС также отклоняются. После host-level подключения рекомендуется
зафиксировать необязательную аттестацию `RAG_BACKUP_EXPECTED_SOURCE` и
`RAG_BACKUP_EXPECTED_FSTYPE` по выводу `findmnt`; любое последующее расхождение
останавливает копирование.

## Ввод в эксплуатацию

1. Смонтировать отдельный носитель, задать `RAG_BACKUP_ROOT` и собрать PostgreSQL
   с оверлеем: `docker compose -f docker-compose.yml -f
   deploy/backup/docker-compose.backup.yml build postgres`. Первый restart,
   `stanza-create` и `check` выполнять в окне; откат — исходный pinned image и
   удаление WAL-параметров оверлея, данные `pg_data` не меняются. Каталог
   `$RAG_BACKUP_ROOT/pgbackrest` заранее создать с владельцем UID/GID пользователя
   `postgres` внутри контейнера и правами `0750`.
2. Установить `/etc/docragenslate/backup-storage.env` из примера, затем unit-файлы
   `pgbackrest-backup@.service`, `pgbackrest-backup-full.timer`,
   `pgbackrest-backup-diff.timer` и `pgbackrest-wal-check.timer`. Полная копия
   планируется по воскресеньям, разностная — с понедельника по субботу, проверка
   stanza/WAL — ежечасно. **Не включать таймеры до** успешных ручных
   `pgbackrest_backup.sh full` и `pgbackrest_backup.sh check`.
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

## Установка расписания после появления независимого носителя

Следующие команды только устанавливают unit-файлы; запуск и включение выделены
отдельно, чтобы mount/pgBackRest нельзя было активировать случайно:

```bash
install -m 0644 deploy/backup/pgbackrest-backup@.service \
  deploy/backup/pgbackrest-backup-full.timer \
  deploy/backup/pgbackrest-backup-diff.timer \
  deploy/backup/pgbackrest-wal-check.timer /etc/systemd/system/
systemctl daemon-reload

set -a
source /etc/docragenslate/backup-storage.env
set +a
deploy/backup/pgbackrest_backup.sh full
deploy/backup/pgbackrest_backup.sh check

systemctl enable --now pgbackrest-backup-full.timer \
  pgbackrest-backup-diff.timer pgbackrest-wal-check.timer
```

Проверить `systemctl list-timers`, `pgbackrest info --output=json`, появление
`rag_backup_last_success_timestamp_seconds` на exporter `:9108` и отсутствие
backup-alerts после первого успешного цикла. Скрипты пишут только атомарные
epoch-маркеры в `/var/lib/docragenslate/backup-metrics`; содержимое БД, имена
файлов и секреты туда не попадают. Marker не обновляется при любой ошибке
preflight, `check`, backup, mirror или export, поэтому возраст неизбежно достигает
порога alert.

Новые alert rules загружать только после первичных успешных full/check,
MinIO-mirror и Keycloak-export: до появления marker timestamp равен нулю, что
намеренно трактуется как просроченная копия, а не как «нет данных».
