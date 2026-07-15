from __future__ import annotations

import asyncio
from typing import Any

import pytest

from rag_app.config import settings
from rag_app.llm import embeddings


class _Response:
    def __init__(self, scores: list[float]) -> None:
        self._scores = scores

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {
            "results": [
                {"index": index, "relevance_score": score}
                for index, score in enumerate(self._scores)
            ]
        }


class _Client:
    def __init__(self, snapshots: list[list[float]]) -> None:
        self._snapshots = iter(snapshots)
        self.calls: list[dict[str, Any]] = []

    async def __aenter__(self) -> _Client:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def post(self, url: str, *, json: dict[str, Any]) -> _Response:
        self.calls.append({"url": url, "json": json})
        return _Response(next(self._snapshots))


def test_reranker_averages_repeated_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client([[0.9, 0.1], [0.6, 0.4], [0.3, 0.7]])
    monkeypatch.setattr(settings, "rerank_score_repeats", 3)
    monkeypatch.setattr(embeddings.httpx, "AsyncClient", lambda **_: client)

    scores = asyncio.run(embeddings.Reranker().rerank("pressure", ["first", "second"]))

    assert scores == pytest.approx([0.6, 0.4])
    assert len(client.calls) == 3
    assert all(call["json"]["query"] == client.calls[0]["json"]["query"] for call in client.calls)
    assert all(call["json"]["documents"] == client.calls[0]["json"]["documents"] for call in client.calls)


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"results": [{"index": 0, "relevance_score": float("nan")}]}, 1),
        ({"results": [{"index": 0, "relevance_score": 0.5}]}, 2),
        (
            {
                "results": [
                    {"index": 0, "relevance_score": 0.5},
                    {"index": 0, "relevance_score": 0.4},
                ]
            },
            2,
        ),
    ],
)
def test_reranker_rejects_invalid_score_coverage(payload: object, expected: int) -> None:
    with pytest.raises(ValueError, match="reranker"):
        embeddings._validated_rerank_scores(payload, expected)
