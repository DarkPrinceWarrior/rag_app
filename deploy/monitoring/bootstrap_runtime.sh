#!/usr/bin/env bash
set -euo pipefail

# Создаёт только эксплуатационные секреты monitoring и отдельную роль БД.
# Значения не печатаются и не передаются аргументами внешних процессов.

SCRIPT_DIR="${MONITORING_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
POSTGRES_CONTAINER="${POSTGRES_CONTAINER:-rag-app-postgres-1}"
MINIO_CONTAINER="${MINIO_CONTAINER:-rag-app-minio-1}"
POSTGRES_ADMIN_USER="${POSTGRES_ADMIN_USER:-rag}"
POSTGRES_DATABASE="${POSTGRES_DATABASE:-rag_app}"
POSTGRES_HOST="${POSTGRES_HOST:-127.0.0.1}"
POSTGRES_PORT="${POSTGRES_PORT:-5433}"
METRICS_ROLE="rag_metrics"
PROMETHEUS_UID="${PROMETHEUS_UID:-65534}"
PROMETHEUS_GID="${PROMETHEUS_GID:-65534}"

if [[ ${EUID} -ne 0 ]]; then
    echo "bootstrap_runtime.sh должен выполняться от root" >&2
    exit 1
fi

for command in docker openssl sed; do
    command -v "${command}" >/dev/null || {
        echo "не найден обязательный инструмент: ${command}" >&2
        exit 1
    }
done

docker inspect "${POSTGRES_CONTAINER}" >/dev/null
docker inspect "${MINIO_CONTAINER}" >/dev/null

umask 077
mkdir -p "${SCRIPT_DIR}/secrets"
chmod 0700 "${SCRIPT_DIR}/secrets"

metrics_password="$(openssl rand -hex 32)"
env_tmp="$(mktemp "${SCRIPT_DIR}/.env.tmp.XXXXXX")"
token_tmp="$(mktemp "${SCRIPT_DIR}/secrets/minio_metrics_token.tmp.XXXXXX")"
cleanup() {
    unset metrics_password
    rm -f -- "${env_tmp}" "${token_tmp}"
}
trap cleanup EXIT

{
    printf "\\set metrics_password '%s'\n" "${metrics_password}"
    cat <<'SQL'
DO $bootstrap$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rag_metrics') THEN
        CREATE ROLE rag_metrics LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE INHERIT;
    END IF;
END
$bootstrap$;
ALTER ROLE rag_metrics
    WITH LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE INHERIT
    PASSWORD :'metrics_password';
GRANT CONNECT ON DATABASE :"database_name" TO rag_metrics;
GRANT pg_monitor TO rag_metrics;
SQL
} | docker exec -i "${POSTGRES_CONTAINER}" \
    psql -X -q -v ON_ERROR_STOP=1 \
    -v database_name="${POSTGRES_DATABASE}" \
    -U "${POSTGRES_ADMIN_USER}" -d "${POSTGRES_DATABASE}" >/dev/null

printf '%s\n' \
    "RAG_METRICS_POSTGRES_DSN=postgresql://${METRICS_ROLE}:${metrics_password}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DATABASE}?sslmode=disable" \
    >"${env_tmp}"
# Bootstrap управляет только DSN exporter-а и не должен сбрасывать уже заданный
# локальный пароль Grafana при повторной ротации runtime-учётных данных.
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
    sed -n '/^RAG_GRAFANA_PASSWORD=/p' "${SCRIPT_DIR}/.env" >>"${env_tmp}"
fi
chmod 0600 "${env_tmp}"

docker exec "${MINIO_CONTAINER}" sh -c '
    export MC_HOST_local="http://${MINIO_ROOT_USER}:${MINIO_ROOT_PASSWORD}@127.0.0.1:9000"
    /usr/bin/mc admin prometheus generate local --json
' | sed -n 's/.*"bearerToken":"\([^"]*\)".*/\1/p' >"${token_tmp}"

if [[ ! -s "${token_tmp}" ]]; then
    echo "MinIO не вернул bearer token" >&2
    exit 1
fi
chmod 0600 "${token_tmp}"

mv -f -- "${env_tmp}" "${SCRIPT_DIR}/.env"
mv -f -- "${token_tmp}" "${SCRIPT_DIR}/secrets/minio_metrics_token"
chown "${PROMETHEUS_UID}:${PROMETHEUS_GID}" \
    "${SCRIPT_DIR}/secrets" "${SCRIPT_DIR}/secrets/minio_metrics_token"
trap - EXIT
unset metrics_password

echo "monitoring runtime подготовлен: роль ${METRICS_ROLE}, .env и MinIO token"
