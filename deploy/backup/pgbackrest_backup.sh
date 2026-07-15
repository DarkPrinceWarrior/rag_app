#!/usr/bin/env bash
# Создает full/diff pgBackRest-копию. Требует отдельный смонтированный носитель.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/projects/rag_app}"
BACKUP_TYPE="${1:-diff}"
: "${RAG_BACKUP_ROOT:?задайте путь отдельного backup-носителя}"

case "$BACKUP_TYPE" in full|diff) ;; *) echo "usage: $0 [full|diff]" >&2; exit 2 ;; esac
mountpoint -q "$RAG_BACKUP_ROOT" || {
  echo "отказ: RAG_BACKUP_ROOT не является отдельной точкой монтирования" >&2
  exit 1
}

compose=(docker compose -f "$REPO_DIR/docker-compose.yml" -f "$REPO_DIR/deploy/backup/docker-compose.backup.yml")
pgbackrest=("${compose[@]}" exec -T --user postgres postgres pgbackrest --stanza=rag-app)
"${pgbackrest[@]}" stanza-create
"${pgbackrest[@]}" check
"${pgbackrest[@]}" --type="$BACKUP_TYPE" backup
"${pgbackrest[@]}" info --output=json
