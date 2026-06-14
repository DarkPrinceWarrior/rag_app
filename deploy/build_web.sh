#!/usr/bin/env bash
# Сборка веб-SPA локально (на сервере нет Node) и доставка на сервер.
# Прод-артефакт без внешних загрузок (всё бандлится). Запуск из корня репо.
#
#   deploy/build_web.sh [ssh-host]   # по умолчанию a100-remote
set -euo pipefail

HOST="${1:-a100-remote}"
REMOTE="/root/projects/rag_app/web/dist"

cd "$(dirname "$0")/../web"
echo "== pnpm install =="
pnpm install --frozen-lockfile
echo "== build (tsc + vite) =="
pnpm build

echo "== deploy dist → $HOST:$REMOTE =="
ssh "$HOST" "mkdir -p $REMOTE && rm -rf $REMOTE/*"
# scp -r всей сборки (assets + index.html + вендоренные ассеты)
scp -rp dist/* "$HOST:$REMOTE/"
echo "== done. Перезапустите API (tmux rag_api), чтобы подхватить web/dist =="
