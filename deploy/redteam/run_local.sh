#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
: "${RAG_REDTEAM_TOKEN:?получите отдельный короткоживущий тестовый token}"
export PROMPTFOO_DISABLE_TELEMETRY=1
export PROMPTFOO_DISABLE_UPDATE=1
export PROMPTFOO_CONFIG_DIR="${PROMPTFOO_CONFIG_DIR:-/tmp/docragenslate-promptfoo}"
exec npx --offline promptfoo eval -c promptfooconfig.yaml --no-cache
