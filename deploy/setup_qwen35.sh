#!/usr/bin/env bash
# Воркхорс Qwen3.5-35B-A3B-GPTQ-Int4 на GPU3, порт 8006 (от root, на сервере).
# Веса: /root/services/vllm-qwen32b/.venv/bin/hf download \
#         Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 --local-dir /root/models/Qwen3.5-35B-A3B-GPTQ-Int4
# Освободить GPU3: systemctl disable --now vllm-hunyuan (его роль у HY-MT1.5 :8005).
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/projects/rag_app}"
MODEL_DIR=/root/models/Qwen3.5-35B-A3B-GPTQ-Int4

[ -d "$MODEL_DIR" ] || { echo "нет весов: $MODEL_DIR (см. шапку — hf download)"; exit 1; }

cp "$REPO_DIR/deploy/vllm-qwen35.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable vllm-qwen35
systemctl restart vllm-qwen35

echo "Ожидание готовности (до 20 мин — первый старт компилит DeltaNet-ядра)…"
for i in $(seq 1 240); do
  if curl -sf http://127.0.0.1:8006/v1/models >/dev/null; then
    echo "Qwen3.5-35B-A3B готов"
    exit 0
  fi
  sleep 5
done
echo "Не поднялся — journalctl -u vllm-qwen35"
exit 1
