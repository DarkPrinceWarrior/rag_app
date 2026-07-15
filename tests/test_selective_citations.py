from __future__ import annotations

import asyncio
import uuid
from collections.abc import Sequence
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from rag_app.config import settings
from rag_app.eval.citation_calibration import (
    CalibrationCase,
    CalibrationClaim,
    calibrate_threshold,
)
from rag_app.rag.chat import ChatEngine
from rag_app.rag.grounding import CitationGuardResult, CitationVerification, CitationVerifier
from rag_app.rag.retrieve import RetrievedChunk
from rag_app.rag.selective_citations import (
    ClaimPair,
    LocalHttpClaimScoreBackend,
    extract_claim_spans,
)


def _chunk(index: int, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        id=uuid.UUID(int=index + 1),
        document_id=uuid.UUID(int=100 + index),
        filename=f"doc-{index}.pdf",
        heading_path=None,
        kind="paragraph",
        page_start=0,
        page_end=0,
        text_en="",
        text_ru=text,
        meta={},
        score=1.0,
    )


class _Backend:
    def __init__(self, scores: list[float] | None = None, error: Exception | None = None) -> None:
        self.scores = scores or []
        self.error = error
        self.pairs: list[ClaimPair] = []

    async def score(self, pairs: Sequence[ClaimPair]) -> list[float]:
        self.pairs = list(pairs)
        if self.error is not None:
            raise self.error
        return self.scores


def test_claim_spans_detect_ru_en_zh_and_preserve_offsets() -> None:
    answer = "Давление равно 5 МПа [1]. Pressure is stable [2]. 压力稳定 [3]。"

    spans = extract_claim_spans(answer)

    assert [span.language for span in spans] == ["ru", "en", "zh"]
    assert [span.citation_numbers for span in spans] == [(1,), (2,), (3,)]
    assert [answer[span.start : span.end] for span in spans] == [span.raw for span in spans]


def test_selective_guard_keeps_chinese_citations_separate_without_spaces() -> None:
    backend = _Backend([0.95, 0.96])
    verifier = CitationVerifier(
        AsyncMock(),
        model="unused",
        max_tokens=10,
        selective_backend=backend,
        selective_threshold=0.7,
    )

    result = asyncio.run(
        verifier.guard(
            "压力稳定 [1]。温度正常 [2]。",
            [_chunk(0, "压力稳定。"), _chunk(1, "温度正常。")],
            mode="selective",
        )
    )

    assert result.answer == "压力稳定 [1]。温度正常 [2]。"
    assert result.failed_closed is False
    assert result.verification is not None
    assert [claim.citation_numbers for claim in result.verification.claims] == [[1], [2]]


def test_selective_guard_removes_only_unsupported_claim() -> None:
    backend = _Backend([0.98, 0.12])
    verifier = CitationVerifier(
        AsyncMock(),
        model="unused",
        max_tokens=10,
        selective_backend=backend,
        selective_threshold=0.7,
    )
    answer = "Рабочее давление стабильно [1]. Срок поставки составляет 10 дней [2]."

    result = asyncio.run(
        verifier.guard(
            answer,
            [_chunk(0, "Рабочее давление стабильно."), _chunk(1, "Срок не определён.")],
            mode="selective",
        )
    )

    assert result.answer == "Рабочее давление стабильно [1]."
    assert result.repaired is True
    assert result.failed_closed is False
    assert result.removed_claims == 1
    assert result.verification is not None
    assert [claim.verdict for claim in result.verification.claims] == ["supported"]
    assert [pair.language for pair in backend.pairs] == ["ru", "ru"]


def test_selective_guard_refuses_only_when_no_supported_fact_survives() -> None:
    verifier = CitationVerifier(
        AsyncMock(),
        model="unused",
        max_tokens=10,
        selective_backend=_Backend([0.1]),
        selective_threshold=0.7,
    )

    result = asyncio.run(
        verifier.guard("Неподтверждённый факт [1].", [_chunk(0, "Другой факт.")], mode="selective")
    )

    assert result.failed_closed is True
    assert result.answer.startswith("Не удалось подтвердить ответ")


def test_selective_guard_fails_closed_on_backend_error() -> None:
    verifier = CitationVerifier(
        AsyncMock(),
        model="unused",
        max_tokens=10,
        selective_backend=_Backend(error=RuntimeError("offline")),
    )

    result = asyncio.run(
        verifier.guard("Факт [1].", [_chunk(0, "Факт.")], mode="selective")
    )

    assert result.failed_closed is True
    assert result.error is not None
    assert result.answer.startswith("Не удалось подтвердить ответ")


def test_local_backend_rejects_non_loopback_endpoint() -> None:
    with pytest.raises(ValueError, match="loopback"):
        LocalHttpClaimScoreBackend("https://example.com/score", model="hhem")


def test_local_backend_uses_hhem_lettuce_batch_contract() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = __import__("json").loads(request.content)
        assert payload["backend"] == "lettuce"
        assert payload["pairs"][0]["language"] == "zh"
        return httpx.Response(200, json={"scores": [{"score": 0.91}]})

    backend = LocalHttpClaimScoreBackend(
        "http://127.0.0.1:8011/score",
        model="local-lettuce",
        adapter="lettuce",
        transport=httpx.MockTransport(handler),
    )

    scores = asyncio.run(
        backend.score([ClaimPair(claim="压力稳定", evidence="压力稳定", language="zh")])
    )

    assert scores == [0.91]


def test_selective_stream_buffers_draft_before_yielding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = ChatEngine()
    prepared = SimpleNamespace(chunks=[_chunk(0, "Поддержано")])
    monkeypatch.setattr(settings, "rag_citation_verification_mode", "selective")
    monkeypatch.setattr(engine, "prepare_answer", AsyncMock(return_value=prepared))

    async def stream(*args: object, **kwargs: object):
        yield "Поддержано [1]. "
        yield "Лишнее [1]."

    monkeypatch.setattr(engine, "stream_prepared", stream)
    verify = AsyncMock(
        return_value=CitationGuardResult(
            answer="Поддержано [1].",
            verification=CitationVerification(claims=[]),
            repaired=True,
        )
    )
    monkeypatch.setattr(engine, "verify_answer", verify)

    async def collect() -> list[str]:
        return [part async for part in engine.stream_answer("Вопрос", prepared.chunks, [])]

    assert asyncio.run(collect()) == ["Поддержано [1]."]
    verify.assert_awaited_once()


def test_risk_coverage_calibration_selects_highest_coverage_qualified_threshold() -> None:
    cases = [
        CalibrationCase(
            case_id="a",
            answerable=True,
            language="ru",
            claims=(
                CalibrationClaim(score=0.9, supported=True),
                CalibrationClaim(score=0.7, supported=False),
            ),
        ),
        CalibrationCase(
            case_id="b",
            answerable=True,
            language="en",
            claims=(CalibrationClaim(score=0.8, supported=True),),
        ),
        CalibrationCase(
            case_id="c",
            answerable=False,
            language="zh",
            claims=(CalibrationClaim(score=0.4, supported=False),),
        ),
    ]

    result = calibrate_threshold(
        cases,
        [0.4, 0.65, 0.75, 0.85],
        answerability_target=0.85,
        semantic_precision_target=0.90,
    )

    assert result.qualified is True
    assert result.selected_threshold == 0.75
    selected = next(point for point in result.curve if point.threshold == 0.75)
    assert selected.answerability_accuracy == 1.0
    assert selected.semantic_precision == 1.0
