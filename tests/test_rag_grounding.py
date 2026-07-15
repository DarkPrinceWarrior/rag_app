from __future__ import annotations

import asyncio
import uuid
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock

import pytest

from rag_app.rag.chat import ChatEngine
from rag_app.rag.grounding import (
    CitationGuardResult,
    CitationVerification,
    CitationVerifier,
    ClaimCheck,
    GroundingError,
    TokenCount,
    compress_low_priority_chunks,
    extractive_compress,
    normalize_verification,
    tokenizer_url,
)
from rag_app.rag.retrieve import RetrievedChunk


def _chunk(index: int, text: str, *, kind: str = "paragraph") -> RetrievedChunk:
    return RetrievedChunk(
        id=uuid.UUID(int=index + 1),
        document_id=uuid.UUID(int=100 + index),
        filename=f"doc-{index}.pdf",
        heading_path=f"Раздел {index}",
        kind=kind,
        page_start=index,
        page_end=index,
        text_en="",
        text_ru=text,
        meta={},
        score=1.0 - index / 100,
    )


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_tokenizer_url_uses_vllm_server_root() -> None:
    assert tokenizer_url("http://127.0.0.1:8006/v1") == "http://127.0.0.1:8006/tokenize"
    assert tokenizer_url("http://vllm.local") == "http://vllm.local/tokenize"


def test_extractive_compression_preserves_literal_evidence() -> None:
    text = (
        "Общее введение без существенных сведений. "
        "Расчётное давление равно 12.5 MPa при 80 °C. "
        "Проверить давление следует по методике. "
        "Ссылка на чертёж: https://example.local/drawing-42. "
        "Длинное заключение, которое не относится к вопросу и может быть исключено."
    )

    compressed = extractive_compress(text, "Какое расчётное давление?", 150)

    assert "12.5 MPa" in compressed
    assert "80 °C" in compressed
    assert "https://example.local/drawing-42" in compressed
    assert compressed == "\n".join(unit for unit in compressed.splitlines())
    assert "Длинное заключение" not in compressed


def test_only_low_priority_chunks_are_compressed() -> None:
    long_text = "Первое предложение. " + "Нерелевантный текст. " * 100 + "Давление 7 MPa."
    chunks = [_chunk(index, long_text) for index in range(5)]

    compressed, changed = compress_low_priority_chunks(
        chunks,
        "Какое давление?",
        after_rank=3,
        max_chars=120,
    )

    assert changed == 2
    assert compressed[:3] == chunks[:3]
    assert compressed[3].id == chunks[3].id
    assert len(compressed[3].text_ru) < len(chunks[3].text_ru)
    assert "7 MPa" in compressed[3].text_ru


def test_verification_requires_valid_citation_and_exact_literals() -> None:
    chunks = [_chunk(0, "Рабочее давление составляет 5 MPa."), _chunk(1, "Температура 80 °C.")]
    report = CitationVerification(
        claims=[
            ClaimCheck(
                claim="Рабочее давление составляет 5 MPa [1].",
                citation_numbers=[1],
                verdict="supported",
            ),
            ClaimCheck(
                claim="Температура составляет 90 °C [2].",
                citation_numbers=[2],
                verdict="supported",
            ),
            ClaimCheck(
                claim="Срок равен 30 дням.",
                citation_numbers=[],
                verdict="supported",
            ),
        ]
    )

    answer = (
        "Рабочее давление составляет 5 MPa [1]. "
        "Температура составляет 90 °C [2]. "
        "Срок равен 30 дням."
    )
    normalized = normalize_verification(report, answer, chunks)

    assert [claim.verdict for claim in normalized.claims] == [
        "supported",
        "unsupported",
        "unsupported",
    ]
    assert normalized.citation_precision == pytest.approx(1 / 3)


def test_verification_cannot_invent_a_citation_number() -> None:
    report = CitationVerification(
        claims=[
            ClaimCheck(
                claim="Рабочее давление составляет 5 MPa",
                citation_numbers=[1],
                verdict="supported",
            )
        ]
    )

    normalized = normalize_verification(
        report,
        "Рабочее давление составляет 5 MPa.",
        [_chunk(0, "Рабочее давление составляет 5 MPa.")],
    )

    assert normalized.claims[0].verdict == "unsupported"
    assert "не совпали" in normalized.claims[0].reason


def test_empty_verification_is_fail_closed() -> None:
    normalized = normalize_verification(
        CitationVerification(claims=[]),
        "Насос развивает 5 MPa [1].",
        [_chunk(0, "Насос развивает 5 MPa.")],
    )

    assert len(normalized.unsupported_claims) == 1


def test_citation_guard_allows_one_repair(monkeypatch: pytest.MonkeyPatch) -> None:
    verifier = CitationVerifier(AsyncMock(), model="test", max_tokens=100)
    bad = CitationVerification(claims=[ClaimCheck(claim="Давление 9 MPa.", verdict="unsupported")])
    good = CitationVerification(
        claims=[
            ClaimCheck(
                claim="Давление 5 MPa [1].",
                citation_numbers=[1],
                verdict="supported",
            )
        ]
    )
    verify = AsyncMock(side_effect=[bad, good])
    repair = AsyncMock(return_value="Давление 5 MPa [1].")
    monkeypatch.setattr(verifier, "verify", verify)
    monkeypatch.setattr(verifier, "_repair", repair)

    result = _run(verifier.guard("Давление 9 MPa [1].", [_chunk(0, "5 MPa")], mode="enforce"))

    assert isinstance(result, CitationGuardResult)
    assert result.answer == "Давление 5 MPa [1]."
    assert result.repaired is True
    assert result.failed_closed is False
    assert verify.await_count == 2
    repair.assert_awaited_once()


def test_citation_guard_fails_closed_after_failed_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = CitationVerifier(AsyncMock(), model="test", max_tokens=100)
    bad = CitationVerification(claims=[ClaimCheck(claim="Давление 9 MPa.", verdict="unsupported")])
    monkeypatch.setattr(verifier, "verify", AsyncMock(side_effect=[bad, bad]))
    monkeypatch.setattr(verifier, "_repair", AsyncMock(return_value="Всё ещё 9 MPa [1]."))

    result = _run(verifier.guard("9 MPa [1].", [_chunk(0, "5 MPa")], mode="enforce"))

    assert result.failed_closed is True
    assert result.answer == "Не удалось подтвердить ответ по найденным фрагментам документов."


def test_exact_budget_drops_history_then_lowest_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = ChatEngine()
    counts: Iterator[TokenCount] = iter(
        [TokenCount(15000, 16384), TokenCount(15000, 16384), TokenCount(1000, 16384)]
    )

    async def fake_count(**_: Any) -> TokenCount:
        return next(counts)

    monkeypatch.setattr("rag_app.rag.chat.count_chat_tokens", fake_count)
    chunks = [_chunk(0, "Источник один."), _chunk(1, "Источник два.")]
    history = [{"role": "user", "content": "Старый вопрос"}]

    prepared = _run(
        engine.prepare_answer(
            "Текущий вопрос",
            chunks,
            history,
            budget_mode="enforce",
        )
    )

    assert prepared.budget_audit is not None
    assert prepared.budget_audit.dropped_history == 1
    assert prepared.budget_audit.dropped_sources == 1
    assert [chunk.id for chunk in prepared.chunks] == [chunks[0].id]
    assert "Источник два" not in prepared.messages[-1]["content"]


def test_enforce_budget_rejects_unavailable_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = ChatEngine()

    async def unavailable(**_: Any) -> TokenCount:
        raise GroundingError("offline")

    monkeypatch.setattr("rag_app.rag.chat.count_chat_tokens", unavailable)

    with pytest.raises(GroundingError, match="offline"):
        _run(
            engine.prepare_answer(
                "Вопрос",
                [_chunk(0, "Источник")],
                [],
                budget_mode="enforce",
            )
        )
