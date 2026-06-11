#!/usr/bin/env bash
# Быстрый контур виджета: Hunyuan-MT-7B на GPU3 (запускать на сервере от root).
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/projects/rag_app}"
MODEL_DIR=/root/models/Hunyuan-MT-7B

[ -d "$MODEL_DIR" ] || { echo "нет весов: $MODEL_DIR"; exit 1; }

cp "$REPO_DIR/deploy/vllm-hunyuan.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable vllm-hunyuan
systemctl restart vllm-hunyuan

echo "Ожидание готовности (до 15 мин)…"
for i in $(seq 1 180); do
  if curl -sf http://127.0.0.1:8004/v1/models >/dev/null; then
    echo "Hunyuan-MT готов"
    exit 0
  fi
  sleep 5
done
echo "Не поднялся — journalctl -u vllm-hunyuan"
exit 1
