#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CI="${CI:-true}"
export UV_PYTHON_DOWNLOADS="${UV_PYTHON_DOWNLOADS:-never}"
export UV_NO_PROGRESS="${UV_NO_PROGRESS:-true}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${RUNNER_TEMP:-/tmp}/rag-app-uv-cache}"
export PNPM_STORE_DIR="${PNPM_STORE_DIR:-${RUNNER_TEMP:-/tmp}/rag-app-pnpm-store}"
export npm_config_store_dir="$PNPM_STORE_DIR"
mkdir -p "$UV_CACHE_DIR" "$PNPM_STORE_DIR"

die() {
  printf 'CI error: %s\n' "$*" >&2
  exit 1
}

preflight() {
  command -v uv >/dev/null || die "uv is required"
  command -v node >/dev/null || die "Node.js is required"
  command -v pnpm >/dev/null || die "pnpm is required"

  local python_bin node_version pnpm_version
  python_bin="$(uv python find 3.13)"
  "$python_bin" scripts/check_ci_policy.py
  node_version="$(node --version)"
  pnpm_version="$(pnpm --version)"
  [[ "$node_version" == v22.* ]] || die "Node.js 22 is required, got $node_version"
  [[ "$pnpm_version" == 11.12.0 ]] || die "pnpm 11.12.0 is required, got $pnpm_version"
  uv lock --check
}

python_sync() {
  uv sync --locked --group dev --python 3.13
}

pytest_check() {
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --locked pytest -q
}

ruff_check() {
  uv run --locked ruff check src tests alembic scripts/check_ci_policy.py
}

mypy_check() {
  uv run --locked mypy src
}

alembic_check() {
  local head_revision heads sql_file
  heads="$(uv run --locked alembic heads)"
  [[ "$(printf '%s\n' "$heads" | sed '/^[[:space:]]*$/d' | wc -l)" -eq 1 ]] || {
    printf '%s\n' "$heads" >&2
    die "Alembic must have exactly one head"
  }
  [[ "$heads" == *"(head)"* ]] || die "Alembic head marker is missing"
  head_revision="${heads%% *}"
  [[ "$head_revision" =~ ^[A-Za-z0-9_]+$ ]] || die "invalid Alembic head revision"

  sql_file="$(mktemp "${RUNNER_TEMP:-/tmp}/rag-app-alembic.XXXXXX.sql")"
  trap 'rm -f "$sql_file"' RETURN
  uv run --locked alembic upgrade head --sql >"$sql_file"
  [[ -s "$sql_file" ]] || die "Alembic offline upgrade emitted no SQL"
  grep -q "CREATE TABLE alembic_version" "$sql_file" || die "offline SQL lacks baseline"
  grep -Fq "version_num='$head_revision'" "$sql_file" || {
    die "offline SQL does not reach head $head_revision"
  }
  rm -f "$sql_file"
  trap - RETURN
}

web_check() {
  pnpm --dir web install --frozen-lockfile --store-dir "$PNPM_STORE_DIR"
  pnpm --dir web lint
  pnpm --dir web build
}

extension_check() {
  pnpm --dir extension install --frozen-lockfile --store-dir "$PNPM_STORE_DIR"
  pnpm --dir extension compile
  pnpm --dir extension build
}

all_checks() {
  preflight
  python_sync
  pytest_check
  ruff_check
  mypy_check
  alembic_check
  web_check
  extension_check
  git diff --check
}

case "${1:-all}" in
  preflight) preflight ;;
  python-sync) python_sync ;;
  pytest) pytest_check ;;
  ruff) ruff_check ;;
  mypy) mypy_check ;;
  alembic) alembic_check ;;
  web) web_check ;;
  extension) extension_check ;;
  all) all_checks ;;
  *) die "unknown target: $1" ;;
esac
