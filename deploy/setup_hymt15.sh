#!/usr/bin/env bash
# Кандидат переводного контура (A/B): HY-MT1.5-7B на GPU1, порт 8005.
# Запускать на сервере от root. Веса качаются отдельно:
#   /root/services/vllm-qwen32b/.venv/bin/hf download tencent/HY-MT1.5-7B \
#       --local-dir /root/models/HY-MT1.5-7B
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/projects/rag_app}"
MODEL_DIR=/root/models/HY-MT1.5-7B

[ -d "$MODEL_DIR" ] || { echo "нет весов: $MODEL_DIR (см. шапку скрипта — hf download)"; exit 1; }

cp "$REPO_DIR/deploy/vllm-hymt15.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable vllm-hymt15
systemctl restart vllm-hymt15

echo "Ожидание готовности (до 15 мин)…"
for i in $(seq 1 180); do
  if curl -sf http://127.0.0.1:8005/v1/models >/dev/null; then
    echo "HY-MT1.5-7B готов"
    exit 0
  fi
  sleep 5
done
echo "Не поднялся — journalctl -u vllm-hymt15"
exit 1
