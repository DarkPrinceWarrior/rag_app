#!/usr/bin/env bash
# BabelDOC (AGPL-3.0) в отдельном venv — изоляция по roadmap § 9:
# вызываем только CLI, исходники не модифицируем, конфигурация снаружи.
set -euo pipefail

UV=/root/.local/bin/uv
SVC_DIR=/root/services/babeldoc

mkdir -p "$SVC_DIR"
cd "$SVC_DIR"
[ -d .venv ] || "$UV" venv --python 3.12 .venv
"$UV" pip install --python .venv/bin/python --upgrade babeldoc

# прогрев: докачивает DocLayout-YOLO, шрифты и пр.
.venv/bin/babeldoc --warmup
echo "BabelDOC готов: $(.venv/bin/babeldoc --version)"
