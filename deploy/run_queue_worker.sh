#!/usr/bin/env bash
set -euo pipefail

profile="${1:-}"
case "$profile" in
  parse) settings=rag_app.workers.queue_workers.ParseWorkerSettings ;;
  translate) settings=rag_app.workers.queue_workers.TranslateWorkerSettings ;;
  export-index) settings=rag_app.workers.queue_workers.ExportIndexWorkerSettings ;;
  memory) settings=rag_app.workers.queue_workers.MemoryWorkerSettings ;;
  *) echo "usage: $0 {parse|translate|export-index|memory}" >&2; exit 2 ;;
esac
exec /root/projects/rag_app/.venv/bin/arq "$settings"
