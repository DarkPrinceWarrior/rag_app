#!/usr/bin/env python3
"""Минимальный сетевой smoke vLLM-кандидата без зависимостей проекта."""

from __future__ import annotations

import json
import math
import sys
import urllib.request
from typing import Any

PROFILES: dict[str, tuple[str, str, dict[str, Any]]] = {
    "qwen35": (
        "http://127.0.0.1:18006/v1/chat/completions",
        "chat",
        {
            "model": "qwen3.5-35b-a3b",
            "messages": [{"role": "user", "content": "Ответь одним словом: готов"}],
            "max_tokens": 16,
            "temperature": 0,
        },
    ),
    "hymt2": (
        "http://127.0.0.1:18005/v1/chat/completions",
        "chat",
        {
            "model": "hy-mt2-7b",
            "messages": [{"role": "user", "content": "Translate into Russian: pressure valve"}],
            "max_tokens": 32,
            "temperature": 0,
        },
    ),
    "embedding": (
        "http://127.0.0.1:18002/v1/embeddings",
        "embedding",
        {"model": "qwen3-embedding-8b", "input": ["pressure valve", "pressure valve"]},
    ),
    "reranker": (
        "http://127.0.0.1:18003/v1/rerank",
        "rerank",
        {
            "model": "qwen3-reranker-4b",
            "query": "pressure valve",
            "documents": ["pressure valve specification", "employee vacation schedule"],
        },
    ),
}


def main() -> int:
    profile = sys.argv[1] if len(sys.argv) == 2 else ""
    if profile not in PROFILES:
        print(f"usage: {sys.argv[0]} {{{'|'.join(PROFILES)}}}", file=sys.stderr)
        return 2
    url, kind, payload = PROFILES[profile]
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        body = json.load(response)
    if kind == "chat":
        assert body["choices"][0]["message"]["content"].strip()
    elif kind == "embedding":
        vectors = [row["embedding"] for row in body["data"]]
        assert len(vectors) == 2 and len(vectors[0]) == len(vectors[1]) >= 1024
        assert all(math.isfinite(value) for vector in vectors for value in vector)
    else:
        scores = [float(row["relevance_score"]) for row in body["results"]]
        assert len(scores) == 2 and scores[0] > scores[1]
    print(f"{profile}: smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
