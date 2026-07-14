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
VENV="${1:-/root/services/mineru/current}"
RP="${2:-1.1}"
F="$("$VENV/bin/python" -I -c 'import mineru_vl_utils,os;print(os.path.dirname(mineru_vl_utils.__file__))')/mineru_client.py"
"$VENV/bin/python" -I - "$F" "$RP" <<'PY'
from __future__ import annotations

import os
import re
import stat
import sys
import tempfile
from pathlib import Path


path = Path(sys.argv[1])
replacement = sys.argv[2]
if re.fullmatch(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?", replacement) is None:
    raise SystemExit(f"invalid repetition_penalty: {replacement!r}")

original_stat = path.lstat()
if not stat.S_ISREG(original_stat.st_mode):
    raise SystemExit(f"refusing to patch a non-regular file: {path}")

source = path.read_text(encoding="utf-8")
pattern = re.compile(
    r"(?m)^(?P<prefix>[ \t]*repetition_penalty: float \| None = )"
    r"(?P<value>[0-9]+(?:\.[0-9]+)?)(?P<suffix>,[ \t]*)$"
)
matches = list(pattern.finditer(source))
if len(matches) != 1:
    raise SystemExit(
        f"expected exactly one repetition_penalty default in {path}, found {len(matches)}"
    )

patched = pattern.sub(
    lambda match: f"{match.group('prefix')}{replacement}{match.group('suffix')}",
    source,
    count=1,
)
payload = patched.encode("utf-8")

fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
temporary = Path(temporary_name)
try:
    os.fchmod(fd, stat.S_IMODE(original_stat.st_mode))
    os.fchown(fd, original_stat.st_uid, original_stat.st_gid)
    with os.fdopen(fd, "wb", closefd=True) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    fd = -1
    os.replace(temporary, path)
    pycache = path.parent / "__pycache__"
    invalidated_bytecode = 0
    if pycache.exists():
        pycache_stat = pycache.lstat()
        if not stat.S_ISDIR(pycache_stat.st_mode):
            raise SystemExit(f"refusing to traverse a non-directory bytecode cache: {pycache}")
        for bytecode in pycache.iterdir():
            if not (bytecode.name.startswith(f"{path.stem}.") and bytecode.suffix == ".pyc"):
                continue
            bytecode_stat = bytecode.lstat()
            if not stat.S_ISREG(bytecode_stat.st_mode):
                raise SystemExit(f"refusing to remove a non-regular bytecode file: {bytecode}")
            bytecode.unlink()
            invalidated_bytecode += 1
        pycache_fd = os.open(pycache, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(pycache_fd)
        finally:
            os.close(pycache_fd)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
finally:
    if fd >= 0:
        os.close(fd)
    temporary.unlink(missing_ok=True)

final_source = path.read_text(encoding="utf-8")
final_matches = list(pattern.finditer(final_source))
final_stat = path.lstat()
expected_metadata = (
    stat.S_IMODE(original_stat.st_mode),
    original_stat.st_uid,
    original_stat.st_gid,
)
actual_metadata = (
    stat.S_IMODE(final_stat.st_mode),
    final_stat.st_uid,
    final_stat.st_gid,
)
if len(final_matches) != 1 or final_matches[0].group("value") != replacement:
    raise SystemExit(f"post-patch value verification failed: {path}")
if actual_metadata != expected_metadata:
    raise SystemExit(
        f"post-patch metadata verification failed: expected {expected_metadata}, got {actual_metadata}"
    )
if final_stat.st_ino == original_stat.st_ino:
    raise SystemExit(f"post-patch hardlink detachment failed: {path}")

print(
    f"patched repetition_penalty -> {replacement}: {path} "
    f"(inode {original_stat.st_ino} -> {final_stat.st_ino}, mode {actual_metadata[0]:#o}, "
    f"invalidated bytecode: {invalidated_bytecode})"
)
PY
