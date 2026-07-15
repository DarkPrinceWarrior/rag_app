#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
umask 077
export PROMPTFOO_DISABLE_TELEMETRY=1
export PROMPTFOO_DISABLE_UPDATE=1
export DO_NOT_TRACK=1
export PROMPTFOO_CONFIG_DIR="${PROMPTFOO_CONFIG_DIR:-/tmp/docragenslate-promptfoo-mock}"
export RAG_REDTEAM_INJECTED_DOCUMENT_ID=00000000-0000-4000-8000-000000000101
export RAG_REDTEAM_INJECTED_CANARY=SYNTH-INJECTED-CANARY
export RAG_REDTEAM_FILENAME_DOCUMENT_ID=00000000-0000-4000-8000-000000000102
export RAG_REDTEAM_FILENAME_CANARY=SYNTH-FILENAME-CANARY
export RAG_REDTEAM_FOREIGN_CANARY=SYNTH-FOREIGN-CANARY
export RAG_REDTEAM_PRIVATE_CODENAME=SYNTH-CLOSED-CODENAME
export RAG_REDTEAM_ZH_DOCUMENT_ID=00000000-0000-4000-8000-000000000103
export RAG_REDTEAM_ZH_CANARY=SYNTH-CHINESE-CANARY
export RAG_REDTEAM_POISON_DOCUMENT_ID=00000000-0000-4000-8000-000000000104
export RAG_REDTEAM_POISON_CANARY=SYNTH-POISON-CANARY

[[ ! -L "$PROMPTFOO_CONFIG_DIR" ]] \
  || { echo "отказ: PROMPTFOO_CONFIG_DIR не должен быть символьной ссылкой" >&2; exit 2; }
install -d -m 700 -- "$PROMPTFOO_CONFIG_DIR"
[[ -x ./node_modules/.bin/promptfoo ]] \
  || { echo "отказ: сначала выполните npm ci --offline" >&2; exit 2; }

evidence="$PROMPTFOO_CONFIG_DIR/mock-results.json"
./node_modules/.bin/promptfoo eval \
  -c promptfooconfig.mock.yaml \
  --no-cache --no-share --no-write --no-progress-bar \
  --output "$evidence"
chmod 600 -- "$evidence"
echo "offline evidence: $evidence"
