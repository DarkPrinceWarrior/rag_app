#!/usr/bin/env bash
# Эмбеддер и reranker (GPU4): Qwen3-Embedding-0.6B + Qwen3-Reranker-4B
# (§ 12.1 шаг 1). Выводит из эксплуатации старые bge-юниты, если стояли.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/projects/rag_app}"

[ -d /root/models/Qwen3-Embedding-0.6B ] || { echo "нет /root/models/Qwen3-Embedding-0.6B"; exit 1; }
[ -d /root/models/Qwen3-Reranker-4B ] || { echo "нет /root/models/Qwen3-Reranker-4B"; exit 1; }

# вывод старых bge-юнитов
for old in vllm-bge-m3; do
  if systemctl list-unit-files | grep -q "$old"; then
    systemctl disable --now "$old" 2>/dev/null || true
    rm -f "/etc/systemd/system/$old.service"
    echo "выведен: $old"
  fi
done

cp "$REPO_DIR/deploy/vllm-embedding.service" /etc/systemd/system/
cp "$REPO_DIR/deploy/vllm-reranker.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable vllm-embedding vllm-reranker
systemctl restart vllm-embedding vllm-reranker

echo "Ожидание готовности (до 10 мин)…"
for i in $(seq 1 120); do
  ok=0
  curl -sf http://127.0.0.1:8002/v1/models >/dev/null && ok=$((ok+1))
  curl -sf http://127.0.0.1:8003/v1/models >/dev/null && ok=$((ok+1))
  if [ "$ok" = 2 ]; then echo "Эмбеддер и reranker (Qwen3) готовы"; exit 0; fi
  sleep 5
done
echo "Не поднялись — journalctl -u vllm-embedding -u vllm-reranker"
exit 1
