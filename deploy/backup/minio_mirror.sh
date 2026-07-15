#!/usr/bin/env bash
# Включает versioning и зеркалирует все прикладные bucket'ы на второй MinIO/S3.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/projects/rag_app}"
BACKUP_ENV="${BACKUP_ENV:-/etc/docragenslate/backup.env}"
test -r "$BACKUP_ENV" || { echo "нет $BACKUP_ENV" >&2; exit 1; }
set -a
# shellcheck disable=SC1090
source "$BACKUP_ENV"
set +a
: "${MINIO_BACKUP_URL:?}"
: "${MINIO_BACKUP_ACCESS_KEY:?}"
: "${MINIO_BACKUP_SECRET_KEY:?}"

compose=(docker compose -f "$REPO_DIR/docker-compose.yml")
mc=("${compose[@]}" exec -T minio mc)
"${compose[@]}" exec -T minio sh -c \
  'mc alias set source http://127.0.0.1:9000 "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD"'
"${mc[@]}" alias set backup "$MINIO_BACKUP_URL" "$MINIO_BACKUP_ACCESS_KEY" "$MINIO_BACKUP_SECRET_KEY"

for bucket in originals artifacts translated exports; do
  "${mc[@]}" version enable "source/$bucket"
  "${mc[@]}" mb --ignore-existing "backup/$bucket"
  "${mc[@]}" version enable "backup/$bucket"
  # Без --remove: удаление/компрометация источника не стирает резервную копию.
  "${mc[@]}" mirror --overwrite "source/$bucket" "backup/$bucket"
done
