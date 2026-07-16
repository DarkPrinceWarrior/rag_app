#!/usr/bin/env bash
# Offline-export realm без пользователей. Требует явного окна и кратко гасит SSO.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/projects/rag_app}"
source "$REPO_DIR/deploy/backup/backup_common.sh"
[[ "${1:-}" == "--maintenance-window" ]] || {
  echo "usage: $0 --maintenance-window" >&2
  exit 2
}
validate_backup_root

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out="$RAG_BACKUP_ROOT/keycloak/$stamp"
mkdir -p "$out"
chmod 0700 "$out"
compose=(docker compose -f "$REPO_DIR/docker-compose.yml")

restart_keycloak() { "${compose[@]}" up -d keycloak; }
trap restart_keycloak EXIT
"${compose[@]}" stop keycloak
"${compose[@]}" run --rm --no-deps -v "$out:/backup" keycloak \
  export --realm rag-app --users skip --file "/backup/rag-app-realm.json"
(cd "$out" && sha256sum rag-app-realm.json > SHA256SUMS)
chmod 0600 "$out/rag-app-realm.json" "$out/SHA256SUMS"
record_backup_success keycloak_export
