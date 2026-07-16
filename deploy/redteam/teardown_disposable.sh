#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd -P)"
STATE="$HERE/.state"
RUNTIME="$STATE/runtime.env"
COMPOSE="$HERE/docker-compose.disposable.yml"

[[ "${RAG_REDTEAM_CONFIRM_DISPOSABLE:-}" == "YES" ]] || {
  echo "отказ: teardown требует RAG_REDTEAM_CONFIRM_DISPOSABLE=YES" >&2
  exit 2
}
[[ -f "$RUNTIME" && ! -L "$STATE" && ! -L "$RUNTIME" ]] || {
  echo "отказ: валидное состояние одноразового стенда не найдено" >&2
  exit 2
}
[[ "$(realpath -e "$STATE")" == "$HERE/.state" ]] || {
  echo "отказ: каталог состояния вышел за deploy/redteam" >&2
  exit 2
}

# shellcheck disable=SC1090 -- файл создан setup с %q и доступен только владельцу.
source "$RUNTIME"
[[ "$RAG_REDTEAM_MARKER" == "docragenslate-disposable-redteam-v1" ]] || {
  echo "отказ: неверный marker стенда" >&2
  exit 2
}
[[ "$RAG_REDTEAM_RUN_ID" =~ ^[0-9a-f]{12}$ ]] || {
  echo "отказ: неверный run id" >&2
  exit 2
}
[[ "$RAG_REDTEAM_COMPOSE_PROJECT" == "docragenslate-redteam-$RAG_REDTEAM_RUN_ID" ]] || {
  echo "отказ: compose project не соответствует run id" >&2
  exit 2
}
[[ "$RAG_REDTEAM_PG_DATABASE" == "docragenslate_redteam_disposable" ]] || {
  echo "отказ: teardown разрешён только для disposable database" >&2
  exit 2
}

stop_pid() {
  local pid_file="$1" marker="$2" pid environment
  [[ -f "$pid_file" ]] || return 0
  pid="$(<"$pid_file")"
  [[ "$pid" =~ ^[0-9]+$ ]] || { echo "отказ: неверный PID в $pid_file" >&2; exit 2; }
  if kill -0 "$pid" 2>/dev/null; then
    environment="$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null || true)"
    [[ "$environment" == *"RAG_REDTEAM_PROCESS_MARKER=$marker"* ]] || {
      echo "отказ: PID $pid не принадлежит стенду ($marker)" >&2
      exit 2
    }
    kill "$pid"
    for _ in {1..30}; do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.1
    done
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL "$pid"
    fi
  fi
  return 0
}

stop_pid "$STATE/api.pid" "docragenslate-redteam-api-$RAG_REDTEAM_RUN_ID"
stop_pid "$STATE/jwks.pid" "docragenslate-redteam-jwks-$RAG_REDTEAM_RUN_ID"

docker compose --env-file "$STATE/compose.env" -p "$RAG_REDTEAM_COMPOSE_PROJECT" \
  -f "$COMPOSE" down --volumes --remove-orphans --timeout 15

if [[ -d "$STATE/evidence" && ! -L "$STATE/evidence" ]]; then
  if [[ -d "$STATE/evidence/logs" && ! -L "$STATE/evidence/logs" ]]; then
    rm -f -- "$STATE/evidence/logs/"*
    rmdir -- "$STATE/evidence/logs"
  fi
  rm -f -- "$STATE/evidence/"*
  rmdir -- "$STATE/evidence"
fi
# STATE уже проверен через realpath, не является symlink и содержит только
# одноразовые ключи/canary/logs. Удаляем дерево одним retry-safe шагом: случайный
# вложенный файл не должен оставлять полуживой каталог без marker/runtime.
rm -rf --one-file-system -- "$STATE"
[[ ! -e "$STATE" ]] || { echo "отказ: каталог состояния не удалён" >&2; exit 1; }
echo "одноразовый red-team стенд уничтожен: $RAG_REDTEAM_RUN_ID"
