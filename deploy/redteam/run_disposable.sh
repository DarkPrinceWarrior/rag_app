#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd -P)"
ROOT="$(cd "$HERE/../.." && pwd -P)"
STATE="$HERE/.state"
RUNTIME="$STATE/runtime.env"
PYTHON="$ROOT/.venv/bin/python"

[[ "${RAG_REDTEAM_CONFIRM_DISPOSABLE:-}" == "YES" ]] || {
  echo "отказ: запуск требует RAG_REDTEAM_CONFIRM_DISPOSABLE=YES" >&2
  exit 2
}
[[ -f "$RUNTIME" && -f "$STATE/fixtures.json" && ! -L "$STATE" ]] || {
  echo "отказ: сначала создайте одноразовый стенд через setup_disposable.sh" >&2
  exit 2
}
# shellcheck disable=SC1090 -- приватный файл создан setup с %q.
source "$RUNTIME"
[[ "$RAG_REDTEAM_MARKER" == "docragenslate-disposable-redteam-v1" ]] || {
  echo "отказ: неверный marker стенда" >&2
  exit 2
}
[[ "$RAG_REDTEAM_COMPOSE_PROJECT" == "docragenslate-redteam-$RAG_REDTEAM_RUN_ID" ]] || {
  echo "отказ: compose project не соответствует run id" >&2
  exit 2
}
api_pid="$(<"$STATE/api.pid")"
[[ "$api_pid" =~ ^[0-9]+$ ]] && kill -0 "$api_pid" 2>/dev/null || {
  echo "отказ: disposable API не запущен" >&2
  exit 2
}
environment="$(tr '\0' '\n' < "/proc/$api_pid/environ" 2>/dev/null || true)"
[[ "$environment" == *"RAG_REDTEAM_PROCESS_MARKER=docragenslate-redteam-api-$RAG_REDTEAM_RUN_ID"* ]] || {
  echo "отказ: PID API не принадлежит этому стенду" >&2
  exit 2
}

eval "$("$PYTHON" - "$STATE/fixtures.json" <<'PY'
import json
import shlex
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
required = {
    "RAG_REDTEAM_OWNER_A",
    "RAG_REDTEAM_INJECTED_DOCUMENT_ID",
    "RAG_REDTEAM_INJECTED_CANARY",
    "RAG_REDTEAM_FILENAME_DOCUMENT_ID",
    "RAG_REDTEAM_FILENAME_CANARY",
    "RAG_REDTEAM_FOREIGN_CANARY",
    "RAG_REDTEAM_PRIVATE_CODENAME",
    "RAG_REDTEAM_ZH_DOCUMENT_ID",
    "RAG_REDTEAM_ZH_CANARY",
    "RAG_REDTEAM_POISON_DOCUMENT_ID",
    "RAG_REDTEAM_POISON_CANARY",
}
if set(data) != required:
    raise SystemExit("fixture manifest fields do not match the contract")
for key, value in sorted(data.items()):
    print(f"export {key}={shlex.quote(value)}")
PY
)"

export RAG_REDTEAM_CONFIRM_ISOLATED=YES
export RAG_REDTEAM_BASE_URL="http://127.0.0.1:$RAG_REDTEAM_API_PORT"
export RAG_REDTEAM_API_PORT RAG_REDTEAM_RUN_ID
export RAG_REDTEAM_API_PID="$api_pid"
export RAG_REDTEAM_TOKEN
RAG_REDTEAM_TOKEN="$(cd / && "$PYTHON" "$HERE/mint_token.py" \
  --private-key "$STATE/private.pem" --issuer "$RAG_REDTEAM_ISSUER" \
  --subject "$RAG_REDTEAM_OWNER_A" --kid "$RAG_REDTEAM_KID" --ttl 900)"
export PROMPTFOO_CONFIG_DIR="$STATE/evidence"

"$HERE/run_local.sh"
