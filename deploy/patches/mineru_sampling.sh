#!/usr/bin/env bash
# Фикс MinerU VLM repetition-collapse: поднимаем repetition_penalty в КЛИЕНТСКОМ
# sampling пакета mineru_vl_utils. ПОЧЕМУ не флаг в mineru-vllm.service: sampling
# (temperature/top_p/penalties) задаётся per-request клиентом MinerU и содержит
# repetition_penalty=1.0 ЯВНО — это перебивает любой серверный
# --override-generation-config. У mineru CLI нет флага sampling, env тоже не
# читается. Значит правим дефолт клиента. Идемпотентно; ПЕРЕЗАПУСКАТЬ после
# `uv sync`/переустановки mineru_vl_utils. Сервер mineru-vllm перезапускать НЕ
# нужно — sampling берёт клиентский подпроцесс `mineru` при каждом парсинге.
# Применялось 2026-06-24 после A/B deeplearningbook (MinerU коллапсил 4/10 стр.).
set -euo pipefail
VENV="${1:-/root/projects/rag_app/.venv}"
RP="${2:-1.1}"
F="$("$VENV/bin/python" -c 'import mineru_vl_utils,os;print(os.path.dirname(mineru_vl_utils.__file__))')/mineru_client.py"
python3 - "$F" "$RP" <<'PY'
import re,sys
f,rp=sys.argv[1],sys.argv[2]
s=open(f,encoding="utf-8").read()
new=re.sub(r'(repetition_penalty: float \| None = )[0-9.]+,', rf'\g<1>{rp},', s)
if new!=s:
    open(f,"w",encoding="utf-8").write(new); print(f"patched repetition_penalty -> {rp}: {f}")
else:
    print("pattern not found (структура изменилась?):", f)
PY
grep -n "repetition_penalty: float | None" "$F"
