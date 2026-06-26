"""Идемпотентный патч prometheus_fastapi_instrumentator (paddle-venv).

Баг версии 8.0.0 со starlette 1.3.x: в routing.py `route_name = route.path`
падает на `'_IncludedRouter' object has no attribute 'path'` — а это middleware
метрик vLLM OpenAI-сервера, поэтому КАЖДЫЙ запрос к genai-серверу отдаёт 500.
Заменяем на `getattr(route, "path", "")`. Запускается ExecStartPre перед стартом
paddle genai-сервера (paddle-genai.service) — переживает переустановку venv.
"""

from __future__ import annotations

import prometheus_fastapi_instrumentator.routing as r

_OLD = "route_name = route.path"
_NEW = 'route_name = getattr(route, "path", "")'


def main() -> None:
    path = r.__file__
    src = open(path, encoding="utf-8").read()
    patched = src.replace(_OLD, _NEW)
    if src != patched:
        open(path, "w", encoding="utf-8").write(patched)
        print(f"patched {path}")
    else:
        print("already patched")


if __name__ == "__main__":
    main()
