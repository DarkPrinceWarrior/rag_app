#!/usr/bin/env python3
"""Минимальный сетевой smoke vLLM-кандидата без зависимостей проекта."""

from __future__ import annotations

import json
import math
import sys
import urllib.request
from dataclasses import dataclass
from typing import Any

from rag_app.llm.embeddings import build_rerank_payload  # type: ignore[import-untyped]

_MIN_RERANK_GAP = 0.05


@dataclass(frozen=True)
class RerankProbe:
    name: str
    query: str
    documents: tuple[str, str]
    relevant_index: int


RERANK_PROBES = (
    RerankProbe(
        name="en",
        query="What is the maximum operating pressure?",
        documents=(
            "The employee vacation calendar is approved every December.",
            "The maximum operating pressure of the valve is 16 MPa.",
        ),
        relevant_index=1,
    ),
    RerankProbe(
        name="ru",
        query="Какова рабочая температура клапана?",
        documents=(
            "Рабочая температура клапана составляет от −40 °C до +120 °C.",
            "График отпусков работников утверждается в декабре.",
        ),
        relevant_index=0,
    ),
    RerankProbe(
        name="zh",
        query="设备的额定电压是多少？",
        documents=(
            "食堂周五的菜单已经更新。",
            "设备的额定电压为10千伏。",
        ),
        relevant_index=1,
    ),
    RerankProbe(
        name="numeric",
        query="Which line has a design pressure of exactly 16 MPa?",
        documents=(
            "Line P-101 has a design pressure of exactly 16 MPa.",
            "Line P-102 has a design pressure of 10 MPa.",
        ),
        relevant_index=0,
    ),
)

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
        {"model": "qwen3-reranker-4b"},
    ),
}


def _request_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        body: dict[str, Any] = json.load(response)
    return body


def _rerank_payload(model: str, probe: RerankProbe) -> dict[str, Any]:
    payload = build_rerank_payload(probe.query, list(probe.documents))
    payload["model"] = model
    return payload


def _validate_rerank_response(body: dict[str, Any], probe: RerankProbe) -> float:
    results = body.get("results")
    if not isinstance(results, list) or len(results) != len(probe.documents):
        raise AssertionError(f"{probe.name}: incomplete rerank result set")

    scores_by_index: dict[int, float] = {}
    for row in results:
        if not isinstance(row, dict):
            raise AssertionError(f"{probe.name}: invalid rerank result row")
        index = row.get("index")
        if not isinstance(index, int) or isinstance(index, bool):
            raise AssertionError(f"{probe.name}: result index is not an integer")
        if index in scores_by_index or not 0 <= index < len(probe.documents):
            raise AssertionError(f"{probe.name}: duplicate or out-of-range result index {index}")
        score = float(row.get("relevance_score", math.nan))
        if not math.isfinite(score):
            raise AssertionError(f"{probe.name}: non-finite relevance score")
        scores_by_index[index] = score

    if set(scores_by_index) != set(range(len(probe.documents))):
        raise AssertionError(f"{probe.name}: result indexes do not cover all documents")
    relevant_score = scores_by_index[probe.relevant_index]
    irrelevant_score = scores_by_index[1 - probe.relevant_index]
    gap = relevant_score - irrelevant_score
    if gap < _MIN_RERANK_GAP:
        raise AssertionError(
            f"{probe.name}: semantic ranking failed: relevant index={probe.relevant_index}, "
            f"gap={gap:.6f}, required>={_MIN_RERANK_GAP:.2f}"
        )
    return gap


def _run_reranker_smoke(url: str, model: str) -> None:
    gaps = [
        _validate_rerank_response(_request_json(url, _rerank_payload(model, probe)), probe)
        for probe in RERANK_PROBES
    ]
    print(
        "reranker semantic probes: "
        + ", ".join(
            f"{probe.name}=+{gap:.4f}" for probe, gap in zip(RERANK_PROBES, gaps, strict=True)
        )
    )


def main() -> int:
    profile = sys.argv[1] if len(sys.argv) == 2 else ""
    if profile not in PROFILES:
        print(f"usage: {sys.argv[0]} {{{'|'.join(PROFILES)}}}", file=sys.stderr)
        return 2
    url, kind, payload = PROFILES[profile]
    if kind == "rerank":
        _run_reranker_smoke(url, str(payload["model"]))
        print(f"{profile}: smoke ok")
        return 0

    body = _request_json(url, payload)
    if kind == "chat":
        assert body["choices"][0]["message"]["content"].strip()
    elif kind == "embedding":
        vectors = [row["embedding"] for row in body["data"]]
        assert len(vectors) == 2 and len(vectors[0]) == len(vectors[1]) >= 1024
        assert all(math.isfinite(value) for vector in vectors for value in vector)
    print(f"{profile}: smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
