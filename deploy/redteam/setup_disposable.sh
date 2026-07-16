#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd -P)"
ROOT="$(cd "$HERE/../.." && pwd -P)"
STATE="$HERE/.state"
COMPOSE="$HERE/docker-compose.disposable.yml"
PYTHON="$ROOT/.venv/bin/python"

[[ "${RAG_REDTEAM_CONFIRM_DISPOSABLE:-}" == "YES" ]] || {
  echo "отказ: setup требует RAG_REDTEAM_CONFIRM_DISPOSABLE=YES" >&2
  exit 2
}
[[ ! -e "$STATE" && ! -L "$STATE" ]] || {
  echo "отказ: состояние уже существует; сначала выполните teardown_disposable.sh" >&2
  exit 2
}
for command in docker curl openssl "$PYTHON"; do
  command -v "$command" >/dev/null 2>&1 || { echo "отказ: не найдено $command" >&2; exit 2; }
done
docker compose version >/dev/null

PG_PORT="${RAG_REDTEAM_PG_PORT:-55432}"
REDIS_PORT="${RAG_REDTEAM_REDIS_PORT:-56379}"
MINIO_PORT="${RAG_REDTEAM_MINIO_PORT:-59000}"
MINIO_CONSOLE_PORT="${RAG_REDTEAM_MINIO_CONSOLE_PORT:-59001}"
API_PORT="${RAG_REDTEAM_API_PORT:-58100}"
JWKS_PORT="${RAG_REDTEAM_JWKS_PORT:-58101}"
LLM_URL="${RAG_REDTEAM_LLM_URL:-http://127.0.0.1:8006/v1}"
EMBED_URL="${RAG_REDTEAM_EMBED_URL:-http://127.0.0.1:8002/v1}"
RERANK_URL="${RAG_REDTEAM_RERANK_URL:-http://127.0.0.1:8003}"
EMBED_MODEL="${RAG_REDTEAM_EMBED_MODEL:-qwen3-embedding-8b}"

"$PYTHON" - "$PG_PORT" "$REDIS_PORT" "$MINIO_PORT" "$MINIO_CONSOLE_PORT" \
  "$API_PORT" "$JWKS_PORT" "$LLM_URL" "$EMBED_URL" "$RERANK_URL" <<'PY'
import socket
import sys
from urllib.parse import urlsplit

ports = [int(value) for value in sys.argv[1:7]]
if len(set(ports)) != len(ports) or any(not 1024 <= port <= 65535 for port in ports):
    raise SystemExit("отказ: порты стенда должны быть уникальными и непривилегированными")
if set(ports) & {5433, 6379, 8100, 9000, 9001, 8180}:
    raise SystemExit("отказ: выбран порт production-контура")
for port in ports:
    probe = socket.socket()
    try:
        probe.bind(("127.0.0.1", port))
    except OSError as exc:
        raise SystemExit(f"отказ: порт {port} занят: {exc}") from exc
    finally:
        probe.close()
for raw, expected_port, expected_path in zip(
    sys.argv[7:], (8006, 8002, 8003), ("/v1", "/v1", "")
):
    parsed = urlsplit(raw)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.port != expected_port
        or parsed.path.rstrip("/") != expected_path
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise SystemExit(f"отказ: недопустимый read-only model endpoint: {raw}")
PY

# Model services are the only reused production components. These requests are read-only.
curl --fail --silent --show-error --max-time 10 "$LLM_URL/models" >/dev/null
curl --fail --silent --show-error --max-time 10 "$EMBED_URL/models" >/dev/null
curl --fail --silent --show-error --max-time 10 "$RERANK_URL/v1/models" >/dev/null

umask 077
install -d -m 700 -- "$STATE" "$STATE/evidence"

cleanup_on_error() {
  local rc=$?
  trap - ERR INT TERM
  if (( rc != 0 )); then
    echo "setup не завершён; уничтожаю созданные ресурсы" >&2
    if [[ -f "$STATE/MARKER" && -f "$RUNTIME" ]]; then
      RAG_REDTEAM_CONFIRM_DISPOSABLE=YES "$HERE/teardown_disposable.sh" || true
    else
      # До создания проверенного runtime контейнеры ещё не запускаются.
      rm -f -- "$STATE/private.pem" "$STATE/jwks.json" "$STATE/compose.env" "$RUNTIME"
      if [[ -d "$STATE/evidence" && ! -L "$STATE/evidence" ]]; then
        rmdir -- "$STATE/evidence" 2>/dev/null || true
      fi
      rmdir -- "$STATE" 2>/dev/null || true
    fi
  fi
  exit "$rc"
}
trap cleanup_on_error ERR INT TERM

RUN_ID="$(openssl rand -hex 6)"
PROJECT="docragenslate-redteam-$RUN_ID"
PG_DATABASE="docragenslate_redteam_disposable"
PG_OWNER_PASSWORD="$(openssl rand -hex 24)"
PG_API_PASSWORD="$(openssl rand -hex 24)"
S3_ACCESS_KEY="rt$(openssl rand -hex 10)"
S3_SECRET_KEY="$(openssl rand -hex 24)"
ISSUER="urn:docragenslate:redteam:$RUN_ID"
KID="$(cd / && "$PYTHON" "$HERE/make_identity.py" --private-key "$STATE/private.pem" --jwks "$STATE/jwks.json")"

write_env() { printf '%s=%q\n' "$1" "$2"; }
{
  write_env RAG_REDTEAM_COMPOSE_PROJECT "$PROJECT"
  write_env RAG_REDTEAM_PG_OWNER_PASSWORD "$PG_OWNER_PASSWORD"
  write_env RAG_REDTEAM_PG_DATABASE "$PG_DATABASE"
  write_env RAG_REDTEAM_PG_PORT "$PG_PORT"
  write_env RAG_REDTEAM_REDIS_PORT "$REDIS_PORT"
  write_env RAG_REDTEAM_MINIO_PORT "$MINIO_PORT"
  write_env RAG_REDTEAM_MINIO_CONSOLE_PORT "$MINIO_CONSOLE_PORT"
  write_env RAG_REDTEAM_S3_ACCESS_KEY "$S3_ACCESS_KEY"
  write_env RAG_REDTEAM_S3_SECRET_KEY "$S3_SECRET_KEY"
} > "$STATE/compose.env"
{
  write_env RAG_REDTEAM_MARKER docragenslate-disposable-redteam-v1
  write_env RAG_REDTEAM_RUN_ID "$RUN_ID"
  write_env RAG_REDTEAM_COMPOSE_PROJECT "$PROJECT"
  write_env RAG_REDTEAM_PG_DATABASE "$PG_DATABASE"
  write_env RAG_REDTEAM_PG_PORT "$PG_PORT"
  write_env RAG_REDTEAM_REDIS_PORT "$REDIS_PORT"
  write_env RAG_REDTEAM_MINIO_PORT "$MINIO_PORT"
  write_env RAG_REDTEAM_MINIO_CONSOLE_PORT "$MINIO_CONSOLE_PORT"
  write_env RAG_REDTEAM_API_PORT "$API_PORT"
  write_env RAG_REDTEAM_JWKS_PORT "$JWKS_PORT"
  write_env RAG_REDTEAM_PG_OWNER_PASSWORD "$PG_OWNER_PASSWORD"
  write_env RAG_REDTEAM_PG_API_PASSWORD "$PG_API_PASSWORD"
  write_env RAG_REDTEAM_S3_ACCESS_KEY "$S3_ACCESS_KEY"
  write_env RAG_REDTEAM_S3_SECRET_KEY "$S3_SECRET_KEY"
  write_env RAG_REDTEAM_ISSUER "$ISSUER"
  write_env RAG_REDTEAM_KID "$KID"
  write_env RAG_REDTEAM_LLM_URL "$LLM_URL"
  write_env RAG_REDTEAM_EMBED_URL "$EMBED_URL"
  write_env RAG_REDTEAM_RERANK_URL "$RERANK_URL"
  write_env RAG_REDTEAM_EMBED_MODEL "$EMBED_MODEL"
} > "$STATE/runtime.env"
printf '%s\n' "docragenslate-disposable-redteam-v1:$RUN_ID" > "$STATE/MARKER"
chmod 600 "$STATE/"*

docker compose --env-file "$STATE/compose.env" -p "$PROJECT" -f "$COMPOSE" up -d --wait
docker compose --env-file "$STATE/compose.env" -p "$PROJECT" -f "$COMPOSE" exec -T postgres \
  psql -v ON_ERROR_STOP=1 -U rag -d "$PG_DATABASE" \
  -c "CREATE ROLE rag_api LOGIN PASSWORD '$PG_API_PASSWORD' NOBYPASSRLS"

ADMIN_URL="postgresql+asyncpg://rag:$PG_OWNER_PASSWORD@127.0.0.1:$PG_PORT/$PG_DATABASE"
API_DB_URL="postgresql+asyncpg://rag_api:$PG_API_PASSWORD@127.0.0.1:$PG_PORT/$PG_DATABASE"
(
  cd /
  RAG_DATABASE_URL="$ADMIN_URL" PYTHONPATH="$ROOT/src" \
    "$PYTHON" -m alembic -c "$ROOT/alembic.ini" upgrade head
)

export RAG_S3_ENDPOINT="127.0.0.1:$MINIO_PORT"
export RAG_S3_ACCESS_KEY="$S3_ACCESS_KEY"
export RAG_S3_SECRET_KEY="$S3_SECRET_KEY"
export RAG_BUCKET_ORIGINALS="redteam-originals"
(
  cd /
  PYTHONPATH="$ROOT/src" "$PYTHON" "$HERE/seed_disposable.py" \
    --admin-url "$ADMIN_URL" --api-url "$API_DB_URL" \
    --embedding-url "$EMBED_URL" --embedding-model "$EMBED_MODEL" \
    --manifest "$STATE/fixtures.json"
)

(
  cd /
  export RAG_REDTEAM_PROCESS_MARKER="docragenslate-redteam-jwks-$RUN_ID"
  exec "$PYTHON" "$HERE/serve_jwks.py" \
    --port "$JWKS_PORT" --jwks "$STATE/jwks.json"
) >"$STATE/jwks.log" 2>&1 &
echo "$!" > "$STATE/jwks.pid"

(
  cd /
  export PYTHONPATH="$ROOT/src"
  export RAG_DATABASE_URL="$API_DB_URL"
  export RAG_REDIS_HOST=127.0.0.1 RAG_REDIS_PORT="$REDIS_PORT" RAG_REDIS_DB=0
  export RAG_S3_ENDPOINT="127.0.0.1:$MINIO_PORT"
  export RAG_S3_ACCESS_KEY="$S3_ACCESS_KEY" RAG_S3_SECRET_KEY="$S3_SECRET_KEY"
  export RAG_BUCKET_ORIGINALS=redteam-originals RAG_BUCKET_ARTIFACTS=redteam-artifacts
  export RAG_BUCKET_TRANSLATED=redteam-translated RAG_BUCKET_EXPORTS=redteam-exports
  export RAG_AUTH_ENABLED=true RAG_OIDC_ISSUER="$ISSUER"
  export RAG_OIDC_JWKS_URL="http://127.0.0.1:$JWKS_PORT/jwks.json"
  export RAG_OIDC_PUBLIC_URL="$ISSUER" RAG_OIDC_CLIENT_ID=rag-web
  export RAG_LLM_BASE_URL="$LLM_URL" RAG_EMBED_BASE_URL="$EMBED_URL"
  export RAG_EMBED_MODEL="$EMBED_MODEL" RAG_RERANK_BASE_URL="$RERANK_URL"
  export RAG_VISUAL_ENABLED=false RAG_VL_ENABLED=false RAG_MEMORY_ENABLED=false
  export RAG_AGENT_ENABLED=false RAG_RAG_CONTEXT_BUDGET_MODE=enforce
  export RAG_REDTEAM_PROCESS_MARKER="docragenslate-redteam-api-$RUN_ID"
  exec "$PYTHON" -m uvicorn rag_app.api.main:app \
    --host 127.0.0.1 --port "$API_PORT" --no-access-log
) >"$STATE/api.log" 2>&1 &
echo "$!" > "$STATE/api.pid"

for _ in {1..120}; do
  curl --fail --silent --max-time 2 "http://127.0.0.1:$API_PORT/healthz" >/dev/null && break
  sleep 0.5
done
curl --fail --silent --show-error --max-time 3 "http://127.0.0.1:$API_PORT/healthz" >/dev/null
unauthorized="$(curl --silent --output /dev/null --write-out '%{http_code}' --max-time 3 \
  "http://127.0.0.1:$API_PORT/api/documents")"
[[ "$unauthorized" == "401" ]] || { echo "отказ: disposable API не требует user token" >&2; exit 1; }
TOKEN="$(cd / && "$PYTHON" "$HERE/mint_token.py" --private-key "$STATE/private.pem" \
  --issuer "$ISSUER" --subject redteam-owner-a --kid "$KID" --ttl 120)"
doc_count="$(curl --fail --silent --show-error --max-time 10 \
  -H "Authorization: Bearer $TOKEN" "http://127.0.0.1:$API_PORT/api/documents" \
  | "$PYTHON" -c 'import json,sys; rows=json.load(sys.stdin); assert all("foreign" not in x["filename"] for x in rows); print(len(rows))')"
unset TOKEN
[[ "$doc_count" == "5" ]] || { echo "отказ: user A видит не пять синтетических документов" >&2; exit 1; }

trap - ERR INT TERM
echo "одноразовый red-team стенд готов: http://127.0.0.1:$API_PORT (run $RUN_ID)"
echo "запуск набора: RAG_REDTEAM_CONFIRM_DISPOSABLE=YES $HERE/run_disposable.sh"
echo "обязательная уборка: RAG_REDTEAM_CONFIRM_DISPOSABLE=YES $HERE/teardown_disposable.sh"
