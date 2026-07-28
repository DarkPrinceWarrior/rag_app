from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from rag_app.config import Settings, settings
from rag_app.llm.embeddings import Embedder


class _FakeEmbeddings:
    def __init__(self, *, reverse_response: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self.reverse_response = reverse_response

    async def create(self, *, model: str, input: list[str]) -> SimpleNamespace:  # noqa: A002
        self.calls.append({"model": model, "input": input})
        data = [
            SimpleNamespace(
                index=index,
                embedding=(
                    [1.0, 0.0]
                    if self.reverse_response and index % 2 == 0
                    else [0.0, 1.0]
                    if self.reverse_response
                    else [3.0, 4.0]
                ),
            )
            for index, _text in enumerate(input)
        ]
        if self.reverse_response:
            data.reverse()
        return SimpleNamespace(data=data)


def _embedder_with_fake_client(
    *, reverse_response: bool = False
) -> tuple[Embedder, _FakeEmbeddings]:
    embedder = Embedder()
    fake = _FakeEmbeddings(reverse_response=reverse_response)
    embedder.client = SimpleNamespace(embeddings=fake)  # type: ignore[assignment]
    return embedder, fake


def test_embed_input_profile_defaults_to_qwen3() -> None:
    assert Settings.model_fields["embed_input_profile"].default == "qwen3"


def test_embed_input_profile_rejects_unknown_value() -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate({"embed_input_profile": "other"})


@pytest.mark.asyncio
async def test_qwen3_profile_preserves_current_input_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "embed_input_profile", "qwen3")
    monkeypatch.setattr(settings, "embed_query_instruction", "Find relevant passages")
    monkeypatch.setattr(settings, "embed_dim", 2)
    embedder, fake = _embedder_with_fake_client()

    document_vectors = await embedder.embed(["  document body  ", "   "])
    query_vector = await embedder.embed_query("  user question  ")

    assert fake.calls == [
        {
            "model": settings.embed_model,
            "input": ["document body", "."],
        },
        {
            "model": settings.embed_model,
            "input": ["Instruct: Find relevant passages\nQuery: user question"],
        },
    ]
    assert document_vectors == [[0.6, 0.8], [0.6, 0.8]]
    assert query_vector == [0.6, 0.8]


@pytest.mark.asyncio
async def test_embed_empty_input_does_not_call_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "embed_input_profile", "qwen3")
    embedder, fake = _embedder_with_fake_client()

    assert await embedder.embed([]) == []
    assert fake.calls == []


@pytest.mark.asyncio
async def test_embed_uses_multiple_batches_and_orders_each_response_by_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "embed_input_profile", "nemotron3")
    monkeypatch.setattr(settings, "embed_dim", 2)
    embedder, fake = _embedder_with_fake_client(reverse_response=True)

    vectors = await embedder.embed(["a", "b", "c", "d", "e"], batch=2)

    assert [call["input"] for call in fake.calls] == [
        ["passage: a", "passage: b"],
        ["passage: c", "passage: d"],
        ["passage: e"],
    ]
    assert vectors == [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
    ]


@pytest.mark.asyncio
async def test_qwen3_empty_instruction_sends_plain_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "embed_input_profile", "qwen3")
    monkeypatch.setattr(settings, "embed_query_instruction", "")
    monkeypatch.setattr(settings, "embed_dim", 2)
    embedder, fake = _embedder_with_fake_client()

    await embedder.embed_query("  plain query  ")

    assert fake.calls[0]["input"] == ["plain query"]


@pytest.mark.asyncio
async def test_nemotron3_profile_formats_queries_and_passages_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "embed_input_profile", "nemotron3")
    monkeypatch.setattr(settings, "embed_query_instruction", "must be ignored")
    monkeypatch.setattr(settings, "embed_dim", 2)
    embedder, fake = _embedder_with_fake_client()

    await embedder.embed(["  document body  ", "passage: already prefixed", "PASSAGE:"])
    await embedder.embed_query(" query: already prefixed ")

    assert fake.calls == [
        {
            "model": settings.embed_model,
            "input": [
                "passage: document body",
                "passage: already prefixed",
                "passage: .",
            ],
        },
        {
            "model": settings.embed_model,
            "input": ["query: already prefixed"],
        },
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["qwen3", "nemotron3"])
async def test_embedding_body_is_truncated_before_profile_prefix(
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
) -> None:
    monkeypatch.setattr(settings, "embed_input_profile", profile)
    monkeypatch.setattr(settings, "embed_query_instruction", "instruction")
    monkeypatch.setattr(settings, "embed_dim", 2)
    embedder, fake = _embedder_with_fake_client()
    body = "x" * 8001

    await embedder.embed([body])
    await embedder.embed_query(body)

    document_input = fake.calls[0]["input"][0]
    query_input = fake.calls[1]["input"][0]
    if profile == "nemotron3":
        assert document_input == f"passage: {'x' * 8000}"
        assert query_input == f"query: {'x' * 8000}"
    else:
        assert document_input == "x" * 8000
        assert query_input == f"Instruct: instruction\nQuery: {'x' * 8000}"


@pytest.mark.asyncio
async def test_qwen3_profile_does_not_duplicate_full_existing_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "embed_input_profile", "qwen3")
    monkeypatch.setattr(settings, "embed_query_instruction", "instruction")
    monkeypatch.setattr(settings, "embed_dim", 2)
    embedder, fake = _embedder_with_fake_client()
    prefix = "Instruct: instruction\nQuery: "
    prefixed = f"{prefix}{'x' * 8001}"

    await embedder.embed_query(prefixed)

    assert fake.calls[0]["input"] == [f"{prefix}{'x' * 8000}"]


@pytest.mark.asyncio
async def test_nemotron3_existing_prefix_is_excluded_from_body_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "embed_input_profile", "nemotron3")
    monkeypatch.setattr(settings, "embed_dim", 2)
    embedder, fake = _embedder_with_fake_client()
    body = "x" * 8001

    await embedder.embed([f"passage: {body}"])
    await embedder.embed_query(f"QUERY: {body}")

    assert fake.calls[0]["input"] == [f"passage: {'x' * 8000}"]
    assert fake.calls[1]["input"] == [f"query: {'x' * 8000}"]
