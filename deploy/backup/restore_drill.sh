#!/usr/bin/env bash
# Fail-closed preflight восстановления на ОТДЕЛЬНОМ контуре.
# Сам production не меняет; печатает следующие команды только после проверок.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/projects/rag_app}"
source "$REPO_DIR/deploy/backup/backup_common.sh"
: "${RESTORE_PGDATA:?задайте пустой каталог тестового PostgreSQL}"
: "${RESTORE_MINIO_URL:?задайте адрес тестового MinIO}"
: "${RESTORE_DOCUMENT_ID:?задайте UUID контрольного документа}"

case "$RESTORE_PGDATA" in
  /var/lib/docker/*|/root/projects/rag_app/*)
    echo "отказ: RESTORE_PGDATA похож на production-путь" >&2
    exit 1
    ;;
esac
case "$RESTORE_MINIO_URL" in
  *127.0.0.1:9000*|*localhost:9000*)
    echo "отказ: RESTORE_MINIO_URL указывает на production MinIO" >&2
    exit 1
    ;;
esac
validate_backup_root
test -d "$RAG_BACKUP_ROOT/pgbackrest"
test -d "$RESTORE_PGDATA"
[[ -z "$(ls -A "$RESTORE_PGDATA")" ]] || { echo "RESTORE_PGDATA не пуст" >&2; exit 1; }

cat <<EOF
Preflight пройден. На отдельном restore-host выполнить:
  pgbackrest --stanza=rag-app --repo1-path='$RAG_BACKUP_ROOT/pgbackrest' \\
    --pg1-path='$RESTORE_PGDATA' --type=immediate restore
Затем поднять временные PostgreSQL/MinIO/Keycloak без внешней публикации портов,
зеркалировать versioned bucket'ы с backup-приемника в RESTORE_MINIO_URL и проверить:
  1) миграция БД совпадает с production-снимком;
  2) документ $RESTORE_DOCUMENT_ID и его segments/chunks читаются;
  3) оригинал и экспорт документа скачиваются и совпадают по SHA-256;
  4) realm импортируется из keycloak/*/rag-app-realm.json;
  5) /healthz и один RAG-запрос проходят.
Результаты, RPO/RTO и SHA-256 приложить к deploy/backup/restore-evidence.example.md.
EOF
