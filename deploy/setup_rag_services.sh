#!/usr/bin/env bash
# Эмбеддер и reranker этапа 3 (GPU4). Запускать на сервере от root.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/projects/rag_app}"

[ -d /root/models/bge-m3 ] || { echo "нет /root/models/bge-m3"; exit 1; }
[ -d /root/models/bge-reranker-v2-m3 ] || { echo "нет /root/models/bge-reranker-v2-m3"; exit 1; }

cp "$REPO_DIR/deploy/vllm-bge-m3.service" /etc/systemd/system/
cp "$REPO_DIR/deploy/vllm-reranker.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable vllm-bge-m3 vllm-reranker
systemctl restart vllm-bge-m3 vllm-reranker

echo "Ожидание готовности (до 10 мин)…"
for i in $(seq 1 120); do
  ok=0
  curl -sf http://127.0.0.1:8002/v1/models >/dev/null && ok=$((ok+1))
  curl -sf http://127.0.0.1:8003/v1/models >/dev/null && ok=$((ok+1))
  if [ "$ok" = 2 ]; then echo "Эмбеддер и reranker готовы"; exit 0; fi
  sleep 5
done
echo "Не поднялись — journalctl -u vllm-bge-m3 -u vllm-reranker"
exit 1
