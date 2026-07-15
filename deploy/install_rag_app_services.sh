#!/usr/bin/env bash
# Устанавливает декларативные unit-файлы API/worker. По умолчанию только
# валидирует; --prepare-env только готовит role-specific env, --install также
# копирует и включает units, --activate дополнительно запускает.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/projects/rag_app}"
SERVICE_ENV_DIR="${SERVICE_ENV_DIR:-/etc/docragenslate}"
COMMON_ENV_SOURCE="${COMMON_ENV_SOURCE:-$REPO_DIR/.env}"
API_ENV_SOURCE="${API_ENV_SOURCE:-$REPO_DIR/.env.api.local}"
WORKER_ENV_SOURCE="${WORKER_ENV_SOURCE:-$REPO_DIR/.env.worker.local}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv/bin/python}"
mode="${1:---check}"
units=(
  rag-api.service
  rag-worker.service
  rag-metrics-exporter.service
  rag-queue-worker@.service
  rag-split-workers.target
  rag-app.target
)

case "$mode" in
  --check|--prepare-env|--install|--activate) ;;
  *) echo "usage: $0 [--check|--prepare-env|--install|--activate]" >&2; exit 2 ;;
esac

render_role_env() {
  local source="$1" destination="$2" expected_role="$3" tmp
  test -r "$COMMON_ENV_SOURCE" || {
    echo "нет общей production-конфигурации" >&2
    return 1
  }
  test -r "$source" || {
    echo "нет role-specific env для $expected_role" >&2
    return 1
  }
  tmp="$(mktemp "${destination}.tmp.XXXXXX")"
  # Сохраняем действующую production-конфигурацию, но никогда не переносим
  # owner URL базы: единственный RAG_DATABASE_URL приходит из role-specific env.
  if ! awk '
    /^[[:space:]]*($|#)/ { print; next }
    {
      line = $0
      sub(/^[[:space:]]*export[[:space:]]+/, "", line)
      sub(/^[[:space:]]+/, "", line)
      if (line !~ /^[A-Za-z_][A-Za-z0-9_]*=/) exit 42
      if (line ~ /^RAG_DATABASE_URL=/) next
      print line
    }
  ' "$COMMON_ENV_SOURCE" >"$tmp"; then
    rm -f "$tmp"
    echo "общая production-конфигурация имеет неподдерживаемый синтаксис" >&2
    return 1
  fi
  if ! awk '
    /^[[:space:]]*($|#)/ { print; next }
    {
      line = $0
      sub(/^[[:space:]]*export[[:space:]]+/, "", line)
      sub(/^[[:space:]]+/, "", line)
      if (line !~ /^[A-Za-z_][A-Za-z0-9_]*=/) exit 42
      print line
    }
  ' "$source" >>"$tmp"; then
    rm -f "$tmp"
    echo "role-specific env имеет неподдерживаемый синтаксис" >&2
    return 1
  fi
  if ! "$PYTHON_BIN" - "$tmp" "$expected_role" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

path = Path(sys.argv[1])
expected_role = sys.argv[2]
matches: list[str] = []
for line in path.read_text(encoding="utf-8").splitlines():
    match = re.match(r"^RAG_DATABASE_URL=(.*)$", line)
    if match:
        matches.append(match.group(1).strip())
if len(matches) != 1:
    raise SystemExit("RAG_DATABASE_URL must occur exactly once")
value = matches[0]
if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
    value = value[1:-1]
url = urlsplit(value.replace("postgresql+asyncpg://", "postgresql://", 1))
if url.scheme != "postgresql" or url.username != expected_role:
    raise SystemExit("unexpected database role")
PY
  then
    rm -f "$tmp"
    echo "role-specific env не прошёл проверку роли $expected_role" >&2
    return 1
  fi
  chmod 0600 "$tmp"
  mv -f "$tmp" "$destination"
}

validate_role_sources() {
  local tmp_dir
  tmp_dir="$(mktemp -d)"
  if ! render_role_env "$API_ENV_SOURCE" "$tmp_dir/api.env" rag_api \
    || ! render_role_env "$WORKER_ENV_SOURCE" "$tmp_dir/worker.env" rag_worker; then
    rm -rf "$tmp_dir"
    return 1
  fi
  rm -rf "$tmp_dir"
}

prepare_role_envs() {
  local staging
  install -d -m 0700 "$SERVICE_ENV_DIR"
  staging="$(mktemp -d "$SERVICE_ENV_DIR/.env-stage.XXXXXX")"
  if ! render_role_env "$API_ENV_SOURCE" "$staging/api.env" rag_api \
    || ! render_role_env "$WORKER_ENV_SOURCE" "$staging/worker.env" rag_worker; then
    rm -rf "$staging"
    return 1
  fi
  chmod 0600 "$staging/api.env" "$staging/worker.env"
  mv -f "$staging/api.env" "$SERVICE_ENV_DIR/api.env"
  mv -f "$staging/worker.env" "$SERVICE_ENV_DIR/worker.env"
  rmdir "$staging"
  echo "role-specific env подготовлены (значения не выводились)"
}

test -x "$PYTHON_BIN"
validate_role_sources

if [[ "$mode" == "--prepare-env" ]]; then
  prepare_role_envs
  exit 0
fi

for unit in "${units[@]}"; do
  test -f "$REPO_DIR/deploy/$unit"
done
test -f "$REPO_DIR/deploy/rag-runtime.env"
test -x "$REPO_DIR/.venv/bin/uvicorn"
test -x "$REPO_DIR/.venv/bin/arq"
systemd-analyze verify "${units[@]/#/$REPO_DIR/deploy/}"

if [[ "$mode" == "--check" ]]; then
  echo "unit-файлы и role-specific env валидны; установка не выполнялась"
  exit 0
fi

prepare_role_envs
install -m 0644 "${units[@]/#/$REPO_DIR/deploy/}" /etc/systemd/system/
systemctl daemon-reload
systemctl enable rag-app.target

if [[ "$mode" == "--activate" ]]; then
  systemctl restart rag-app.target
  systemctl --no-pager --full status rag-api.service rag-worker.service
else
  echo "unit-файлы установлены и включены; текущие tmux-процессы не остановлены"
fi
