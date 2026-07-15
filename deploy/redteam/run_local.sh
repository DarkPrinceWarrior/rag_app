#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
required=(
  RAG_REDTEAM_TOKEN
  RAG_REDTEAM_BASE_URL
  RAG_REDTEAM_CONFIRM_ISOLATED
  RAG_REDTEAM_INJECTED_DOCUMENT_ID
  RAG_REDTEAM_INJECTED_CANARY
  RAG_REDTEAM_FILENAME_DOCUMENT_ID
  RAG_REDTEAM_FILENAME_CANARY
  RAG_REDTEAM_FOREIGN_CANARY
  RAG_REDTEAM_PRIVATE_CODENAME
  RAG_REDTEAM_ZH_DOCUMENT_ID
  RAG_REDTEAM_ZH_CANARY
  RAG_REDTEAM_POISON_DOCUMENT_ID
  RAG_REDTEAM_POISON_CANARY
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "отказ: обязательная переменная $name не задана" >&2; exit 2; }
done
[[ "$RAG_REDTEAM_CONFIRM_ISOLATED" == "YES" ]] \
  || { echo "отказ: подтвердите одноразовый стенд через RAG_REDTEAM_CONFIRM_ISOLATED=YES" >&2; exit 2; }
uuid_re='^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-8][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$'
for name in RAG_REDTEAM_INJECTED_DOCUMENT_ID RAG_REDTEAM_FILENAME_DOCUMENT_ID RAG_REDTEAM_ZH_DOCUMENT_ID RAG_REDTEAM_POISON_DOCUMENT_ID; do
  [[ "${!name}" =~ $uuid_re ]] || { echo "отказ: $name не является UUID" >&2; exit 2; }
done
canary_names=(
  RAG_REDTEAM_INJECTED_CANARY RAG_REDTEAM_FILENAME_CANARY RAG_REDTEAM_FOREIGN_CANARY
  RAG_REDTEAM_PRIVATE_CODENAME RAG_REDTEAM_ZH_CANARY RAG_REDTEAM_POISON_CANARY
)
declare -A seen_canaries=()
for name in "${canary_names[@]}"; do
  value="${!name}"
  [[ ${#value} -ge 16 ]] || { echo "отказ: $name должен быть уникальным маркером не короче 16 символов" >&2; exit 2; }
  [[ -z "${seen_canaries[$value]:-}" ]] || { echo "отказ: canary-маркеры должны быть различными" >&2; exit 2; }
  seen_canaries[$value]=1
done
umask 077
export PROMPTFOO_DISABLE_TELEMETRY=1
export PROMPTFOO_DISABLE_UPDATE=1
export PROMPTFOO_DISABLE_SHARING=1
export DO_NOT_TRACK=1
export PROMPTFOO_CONFIG_DIR="${PROMPTFOO_CONFIG_DIR:-/tmp/docragenslate-promptfoo}"
[[ ! -L "$PROMPTFOO_CONFIG_DIR" ]] \
  || { echo "отказ: PROMPTFOO_CONFIG_DIR не должен быть символьной ссылкой" >&2; exit 2; }
install -d -m 700 -- "$PROMPTFOO_CONFIG_DIR"
chmod 700 -- "$PROMPTFOO_CONFIG_DIR"
expected_version="$(node -p "require('./package.json').devDependencies.promptfoo")"
actual_version="$(node -p "require('./node_modules/promptfoo/package.json').version" 2>/dev/null || true)"
[[ "$expected_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ && "$actual_version" == "$expected_version" ]] \
  || { echo "отказ: установите pinned Promptfoo через npm ci --offline --ignore-scripts" >&2; exit 2; }
[[ -x ./node_modules/.bin/promptfoo ]] \
  || { echo "отказ: локальный исполняемый файл Promptfoo отсутствует" >&2; exit 2; }
evidence="$PROMPTFOO_CONFIG_DIR/redteam-$(date -u +%Y%m%dT%H%M%SZ).json"
echo "локальный evidence: $evidence"
exec ./node_modules/.bin/promptfoo eval -c promptfooconfig.yaml \
  --no-cache --no-share --no-write --no-progress-bar --output "$evidence"
