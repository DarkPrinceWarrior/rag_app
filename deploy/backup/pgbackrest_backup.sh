#!/usr/bin/env bash
# Создает full/diff pgBackRest-копию. Требует отдельный смонтированный носитель.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/projects/rag_app}"
BACKUP_TYPE="${1:-diff}"
source "$REPO_DIR/deploy/backup/backup_common.sh"

case "$BACKUP_TYPE" in
  full|diff|check) ;;
  *) echo "usage: $0 {full|diff|check}" >&2; exit 2 ;;
esac
validate_backup_root

compose=(docker compose -f "$REPO_DIR/docker-compose.yml" -f "$REPO_DIR/deploy/backup/docker-compose.backup.yml")
pgbackrest=("${compose[@]}" exec -T --user postgres postgres pgbackrest --stanza=rag-app)
if [[ "$BACKUP_TYPE" == check ]]; then
  "${pgbackrest[@]}" check
  "${pgbackrest[@]}" info --output=json
  record_backup_success wal_archive_check
else
  "${pgbackrest[@]}" stanza-create
  "${pgbackrest[@]}" check
  record_backup_success wal_archive_check
  "${pgbackrest[@]}" --type="$BACKUP_TYPE" backup
  "${pgbackrest[@]}" info --output=json
  record_backup_success "pgbackrest_$BACKUP_TYPE"
fi
