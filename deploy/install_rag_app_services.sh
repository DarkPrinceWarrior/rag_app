#!/usr/bin/env bash
# Устанавливает декларативные unit-файлы API/worker. По умолчанию только
# валидирует; --install копирует и включает, --activate дополнительно запускает.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/projects/rag_app}"
mode="${1:---check}"
units=(
  rag-api.service
  rag-worker.service
  rag-metrics-exporter.service
  rag-queue-worker@.service
  rag-split-workers.target
  rag-app.target
)

case "$mode" in
  --check|--install|--activate) ;;
  *) echo "usage: $0 [--check|--install|--activate]" >&2; exit 2 ;;
esac

for unit in "${units[@]}"; do
  test -f "$REPO_DIR/deploy/$unit"
done
test -f "$REPO_DIR/deploy/rag-runtime.env"
test -x "$REPO_DIR/.venv/bin/uvicorn"
test -x "$REPO_DIR/.venv/bin/arq"
systemd-analyze verify "${units[@]/#/$REPO_DIR/deploy/}"

if [[ "$mode" == "--check" ]]; then
  echo "unit-файлы валидны; установка не выполнялась"
  exit 0
fi

install -m 0644 "${units[@]/#/$REPO_DIR/deploy/}" /etc/systemd/system/
systemctl daemon-reload
systemctl enable rag-app.target

if [[ "$mode" == "--activate" ]]; then
  systemctl restart rag-app.target
  systemctl --no-pager --full status rag-api.service rag-worker.service
else
  echo "unit-файлы установлены и включены; текущие tmux-процессы не остановлены"
fi
