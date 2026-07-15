#!/usr/bin/env bash
# Готовит НОВОЕ окружение vLLM-кандидата; действующие venv и MinerU не меняет.
set -euo pipefail

UV="${UV:-/root/.local/bin/uv}"
VLLM_VERSION="${VLLM_VERSION:-0.24.0}"
role="${1:-}"

case "$role" in
  main) service_dir="${VLLM_MAIN_CANDIDATE_DIR:-/root/services/vllm-main-${VLLM_VERSION}}" ;;
  visual) service_dir="${VLLM_VISUAL_CANDIDATE_DIR:-/root/services/vllm-visual-${VLLM_VERSION}}" ;;
  *) echo "usage: $0 {main|visual}" >&2; exit 2 ;;
esac

case "$service_dir" in
  /root/services/mineru/*|/root/services/mineru|/root/projects/rag_app/.venv*)
    echo "отказ: кандидат нельзя устанавливать в окружение MinerU/приложения" >&2
    exit 1
    ;;
esac

mkdir -p "$service_dir"
if [[ ! -x "$service_dir/.venv/bin/python" ]]; then
  "$UV" venv --python 3.12 "$service_dir/.venv"
fi
"$UV" pip install --python "$service_dir/.venv/bin/python" \
  "vllm==${VLLM_VERSION}" ninja
"$UV" pip freeze --python "$service_dir/.venv/bin/python" > "$service_dir/requirements.freeze.txt"
"$service_dir/.venv/bin/python" -c 'import vllm; print(vllm.__version__)'
echo "кандидат подготовлен: $service_dir (production unit-файлы не изменены)"
