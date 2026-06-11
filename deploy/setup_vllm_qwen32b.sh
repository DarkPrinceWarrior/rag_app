#!/usr/bin/env bash
# Развёртывание vLLM-сервиса Qwen3-32B-AWQ на a100 (запускать на сервере от root).
# Отдельный venv (python 3.12): vLLM пинит свои torch/зависимости,
# окружение приложения (3.13) не трогаем.
set -euo pipefail

UV=/root/.local/bin/uv
SVC_DIR=/root/services/vllm-qwen32b
MODEL_DIR=/root/models/Qwen3-32B-AWQ
REPO_DIR="${REPO_DIR:-/root/projects/rag_app}"

[ -d "$MODEL_DIR" ] || { echo "нет весов: $MODEL_DIR"; exit 1; }

command -v numactl >/dev/null || { apt-get update -qq && apt-get install -y -qq numactl; }

mkdir -p "$SVC_DIR"
cd "$SVC_DIR"
[ -d .venv ] || "$UV" venv --python 3.12 .venv
"$UV" pip install --python .venv/bin/python --upgrade vllm

cp "$REPO_DIR/deploy/vllm-qwen32b.service" /etc/systemd/system/vllm-qwen32b.service
systemctl daemon-reload
systemctl enable vllm-qwen32b
systemctl restart vllm-qwen32b

echo "Ожидание готовности vLLM (до 15 мин — прогрев и компиляция)…"
for i in $(seq 1 180); do
  if curl -sf http://127.0.0.1:8001/v1/models >/dev/null; then
    echo "vLLM готов:"
    curl -s http://127.0.0.1:8001/v1/models
    exit 0
  fi
  sleep 5
done
echo "vLLM не поднялся за 15 мин — смотри journalctl -u vllm-qwen32b"
exit 1
