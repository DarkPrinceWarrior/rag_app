"""Exact context budgeting and claim-level citation verification for RAG chat."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field

from rag_app.rag.retrieve import RetrievedChunk
from rag_app.rag.selective_citations import (
    ClaimPair,
    ClaimScoreBackend,
    ScoredClaim,
    detect_claim_language,
    filter_answer,
    score_claims,
)

logger = logging.getLogger(__name__)

GroundingMode = Literal["off", "shadow", "enforce", "selective"]
ClaimVerdict = Literal["supported", "unsupported", "contradicted", "non_factual"]

_CITATION = re.compile(r"\[(\d{1,3})\]")
_UNIT_SPLIT = re.compile(r"(?<=[.!?;])\s+|(?<=[。！？；])\s*|\n+")
_TERM = re.compile(r"[^\W_]{3,}", re.UNICODE)
_NUMBER = re.compile(r"(?<![\w.])[+-]?\d+(?:[.,]\d+)*(?:\s?(?:%|°[CF]|[A-Za-zА-Яа-я]{1,8}))?")
_URL = re.compile(r"https?://[^\s)\]}]+", re.IGNORECASE)
_SAFE_REFUSAL = "Не удалось подтвердить ответ по найденным фрагментам документов."


class GroundingError(RuntimeError):
    """A fail-closed grounding operation could not be completed."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClaimCheck(_StrictModel):
    claim: str = Field(
        min_length=1,
        max_length=4000,
        description="Точный текст атомарного утверждения без маркеров цитат [n].",
    )
    citation_numbers: list[int] = Field(
        default_factory=list,
        max_length=20,
        description="Все номера [n], стоящие рядом с утверждением в ответе.",
    )
    verdict: ClaimVerdict = Field(
        description="Поддержано, не поддержано, противоречит или не является фактом."
    )
    reason: str = Field(default="", max_length=2000)
    score: float | None = Field(default=None, ge=0, le=1)


class CitationVerification(_StrictModel):
    claims: list[ClaimCheck] = Field(default_factory=list, max_length=100)

    @property
    def factual_claims(self) -> list[ClaimCheck]:
        return [claim for claim in self.claims if claim.verdict != "non_factual"]

    @property
    def unsupported_claims(self) -> list[ClaimCheck]:
        return [claim for claim in self.claims if claim.verdict in {"unsupported", "contradicted"}]

    @property
    def citation_precision(self) -> float | None:
        factual = self.factual_claims
        if not factual:
            return None
        supported = sum(claim.verdict == "supported" for claim in factual)
        return supported / len(factual)


@dataclass(frozen=True)
class TokenCount:
    count: int
    max_model_len: int


@dataclass(frozen=True)
class ContextBudgetAudit:
    mode: GroundingMode
    exact_tokens: int | None
    input_limit: int | None
    max_model_len: int | None
    dropped_history: int = 0
    dropped_sources: int = 0
    compressed_sources: int = 0
    tokenizer_error: str | None = None


@dataclass(frozen=True)
class CitationGuardResult:
    answer: str
    verification: CitationVerification | None
    repaired: bool = False
    failed_closed: bool = False
    error: str | None = None
    removed_claims: int = 0


def tokenizer_url(openai_base_url: str) -> str:
    base = openai_base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return f"{base}/tokenize"


async def count_chat_tokens(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: Sequence[dict[str, Any]],
    timeout: float = 30.0,
) -> TokenCount:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    payload = {
        "model": model,
        "messages": list(messages),
        "add_generation_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        async with httpx.AsyncClient(timeout=timeout, headers=headers, trust_env=False) as client:
            response = await client.post(tokenizer_url(base_url), json=payload)
            response.raise_for_status()
        body = response.json()
        count = body["count"]
        max_model_len = body["max_model_len"]
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        raise GroundingError("точный tokenizer vLLM недоступен или вернул неверный ответ") from exc
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in (count, max_model_len)
    ):
        raise GroundingError("точный tokenizer vLLM вернул неверные границы")
    return TokenCount(count=count, max_model_len=max_model_len)


def extractive_compress(text: str, question: str, max_chars: int) -> str:
    """Select original text units; never rewrite numbers, formulas, or URLs."""
    if len(text) <= max_chars:
        return text
    units = [unit.strip() for unit in _UNIT_SPLIT.split(text) if unit.strip()]
    if len(units) <= 1:
        return text
    query_terms = {term.casefold() for term in _TERM.findall(question)}
    must_keep: set[int] = {0}
    scored: list[tuple[int, int]] = []
    for index, unit in enumerate(units):
        if _NUMBER.search(unit) or _URL.search(unit):
            must_keep.add(index)
        overlap = len(query_terms & {term.casefold() for term in _TERM.findall(unit)})
        scored.append((overlap, index))

    selected = set(must_keep)
    size = sum(len(units[index]) + 1 for index in selected)
    for overlap, index in sorted(scored, key=lambda item: (-item[0], item[1])):
        if index in selected or overlap == 0:
            continue
        candidate_size = size + len(units[index]) + 1
        if candidate_size <= max_chars:
            selected.add(index)
            size = candidate_size
    if len(selected) == len(units):
        return text
    return "\n".join(units[index] for index in sorted(selected))


def compress_low_priority_chunks(
    chunks: Sequence[RetrievedChunk],
    question: str,
    *,
    after_rank: int,
    max_chars: int,
) -> tuple[list[RetrievedChunk], int]:
    compressed: list[RetrievedChunk] = []
    source_rank = 0
    changed = 0
    for chunk in chunks:
        if chunk.kind == "catalog":
            compressed.append(chunk)
            continue
        source_rank += 1
        if source_rank <= after_rank:
            compressed.append(chunk)
            continue
        field = "text_ru" if chunk.text_ru else "text_en"
        original = getattr(chunk, field)
        reduced = extractive_compress(original, question, max_chars)
        if reduced != original:
            chunk = replace(chunk, text_ru=reduced) if field == "text_ru" else replace(chunk, text_en=reduced)
            changed += 1
        compressed.append(chunk)
    return compressed, changed


def _source_texts(chunks: Sequence[RetrievedChunk]) -> dict[int, str]:
    return {
        index: chunk.text_ru or chunk.text_en
        for index, chunk in enumerate((c for c in chunks if c.kind != "catalog"), 1)
    }


def _literal_tokens(text: str) -> list[str]:
    without_citations = _CITATION.sub("", text)
    return _NUMBER.findall(without_citations) + _URL.findall(without_citations)


def _actual_citations_for_claim(answer: str, claim: str) -> list[int] | None:
    claim_text = " ".join(_CITATION.sub("", claim).split()).casefold()
    if not claim_text:
        return None
    for unit in _UNIT_SPLIT.split(answer):
        unit_text = " ".join(_CITATION.sub("", unit).split()).casefold()
        if claim_text in unit_text:
            return [int(match.group(1)) for match in _CITATION.finditer(unit)]
    return None


def normalize_verification(
    report: CitationVerification,
    answer: str,
    chunks: Sequence[RetrievedChunk],
) -> CitationVerification:
    sources = _source_texts(chunks)
    normalized: list[ClaimCheck] = []
    for claim in report.claims:
        verdict = claim.verdict
        reason = claim.reason
        declared = list(dict.fromkeys(claim.citation_numbers))
        actual = _actual_citations_for_claim(answer, claim.claim)
        if actual is None:
            verdict = "unsupported"
            reason = "Проверяемое утверждение не найдено дословно в ответе."
            actual = []
        elif actual != declared:
            verdict = "unsupported"
            reason = "Номера цитат проверяющей модели не совпали с ответом."
        cited = [number for number in dict.fromkeys(actual) if number in sources]
        if verdict == "supported" and len(cited) != len(set(actual)):
            verdict = "unsupported"
            reason = "Утверждение содержит номер несуществующего источника."
        if verdict == "supported" and not cited:
            verdict = "unsupported"
            reason = "У фактического утверждения нет допустимой цитаты."
        if verdict == "supported":
            evidence = "\n".join(sources[number] for number in cited).casefold()
            missing = [token for token in _literal_tokens(claim.claim) if token.casefold() not in evidence]
            if missing:
                verdict = "unsupported"
                reason = "В процитированном тексте нет точных значений или ссылок: " + ", ".join(missing[:8])
        normalized.append(
            claim.model_copy(update={"citation_numbers": cited, "verdict": verdict, "reason": reason})
        )
    if not normalized and answer.strip() and answer.strip() != _SAFE_REFUSAL:
        normalized.append(
            ClaimCheck(
                claim=answer[:4000],
                citation_numbers=[],
                verdict="unsupported",
                reason="Проверяющая модель не выделила ни одного утверждения.",
            )
        )
    return CitationVerification(claims=normalized)


def _verification_prompt(answer: str, chunks: Sequence[RetrievedChunk]) -> str:
    sources = _source_texts(chunks)
    source_block = "\n\n".join(f"[{number}]\n{text}" for number, text in sources.items())
    return f"""Проверь ответ на уровне атомарных утверждений.

Для каждого проверяемого фактического утверждения верни отдельный элемент:
- claim: точный текст утверждения из ответа БЕЗ маркеров [n], без пересказа;
- citation_numbers: ОБЯЗАТЕЛЬНО скопируй все целые n из маркеров [n], стоящих рядом с ним;
  например, для «Давление 5 МПа [1]» верни citation_numbers=[1];
- verdict: проверяй утверждение ТОЛЬКО по перечисленным источникам.

Значения, единицы, формулы, отрицания и условия должны совпадать. Если цитаты нет, номер
неверен или источник лишь тематически похож, verdict=unsupported. При прямом конфликте —
contradicted. Заголовки, оговорки и фразы без проверяемого факта — non_factual. Даже для
unsupported/contradicted всё равно скопируй фактически указанные citation_numbers.

ОТВЕТ:
{answer}

ИСТОЧНИКИ:
{source_block}
"""


class CitationVerifier:
    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        model: str,
        max_tokens: int,
        selective_backend: ClaimScoreBackend | None = None,
        selective_threshold: float = 0.7,
    ) -> None:
        if not 0.0 <= selective_threshold <= 1.0:
            raise ValueError("selective citation threshold must be in [0, 1]")
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._selective_backend = selective_backend
        self._selective_threshold = selective_threshold

    async def verify(self, answer: str, chunks: Sequence[RetrievedChunk]) -> CitationVerification:
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "citation_verification",
                "strict": True,
                "schema": CitationVerification.model_json_schema(),
            },
        }
        try:
            completion = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": "Ты строгий проверяющий доказательности корпоративного RAG.",
                    },
                    {"role": "user", "content": _verification_prompt(answer, chunks)},
                ],
                temperature=0.0,
                max_tokens=self._max_tokens,
                response_format=cast(Any, response_format),
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            content = completion.choices[0].message.content
            if not content:
                raise ValueError("empty verifier response")
            report = CitationVerification.model_validate_json(content)
        except Exception as exc:  # noqa: BLE001 - SDK errors vary across releases
            # OpenAI transport exceptions do not share one stable base across SDK releases.
            raise GroundingError("проверка цитат не завершилась") from exc
        return normalize_verification(report, answer, chunks)

    async def _verify_selective(
        self,
        answer: str,
        chunks: Sequence[RetrievedChunk],
    ) -> tuple[CitationVerification, list[ScoredClaim]]:
        if self._selective_backend is None:
            raise GroundingError("селективный backend проверки цитат не настроен")
        scored = await score_claims(
            answer,
            _source_texts(chunks),
            self._selective_backend,
            threshold=self._selective_threshold,
        )
        report = CitationVerification(
            claims=[
                ClaimCheck(
                    claim=item.span.claim,
                    citation_numbers=list(item.span.citation_numbers),
                    verdict=(
                        "non_factual"
                        if item.span.non_factual
                        else "supported"
                        if item.supported
                        else "unsupported"
                    ),
                    reason=item.reason,
                    score=item.score,
                )
                for item in scored
            ]
        )
        return normalize_verification(report, answer, chunks), scored

    async def score_report_claims(
        self,
        report: CitationVerification,
        chunks: Sequence[RetrievedChunk],
    ) -> list[float]:
        """Оценить LLM/human-labeled claims дешёвым backend для калибровки."""

        if self._selective_backend is None:
            raise GroundingError("селективный backend проверки цитат не настроен")
        sources = _source_texts(chunks)
        factual = report.factual_claims
        scores = [0.0] * len(factual)
        pairs: list[ClaimPair] = []
        indexes: list[int] = []
        for index, claim in enumerate(factual):
            evidence = [
                sources[number]
                for number in claim.citation_numbers
                if number in sources
            ]
            if not evidence or len(evidence) != len(claim.citation_numbers):
                continue
            pairs.append(
                ClaimPair(
                    claim=claim.claim,
                    evidence="\n".join(evidence),
                    language=detect_claim_language(claim.claim),
                )
            )
            indexes.append(index)
        values = await self._selective_backend.score(pairs)
        for index, value in zip(indexes, values, strict=True):
            scores[index] = value
        return scores

    async def _repair(
        self,
        answer: str,
        report: CitationVerification,
        chunks: Sequence[RetrievedChunk],
    ) -> str:
        unsupported = "\n".join(
            f"- {claim.claim[:500]}: {claim.reason[:300]}"
            for claim in report.unsupported_claims[:20]
        )
        prompt = f"""Исправь ответ один раз. Удали неподтверждённые утверждения или замени их
на подтверждённые указанными источниками. Не добавляй новые факты. Сохрани корректные части,
числа, единицы и ссылки [n]. Верни только итоговый ответ.

НЕПОДТВЕРЖДЁННОЕ:
{unsupported}

{_verification_prompt(answer, chunks)}
"""
        completion = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=self._max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        content = completion.choices[0].message.content
        if not content:
            raise GroundingError("исправление ответа вернуло пустой текст")
        return content.strip()

    async def guard(
        self,
        answer: str,
        chunks: Sequence[RetrievedChunk],
        *,
        mode: GroundingMode,
    ) -> CitationGuardResult:
        if mode == "off" or not chunks:
            return CitationGuardResult(answer=answer, verification=None)
        try:
            if mode == "selective":
                report, scored = await self._verify_selective(answer, chunks)
                supported = [claim for claim in report.factual_claims if claim.verdict == "supported"]
                if not supported:
                    return CitationGuardResult(
                        answer=_SAFE_REFUSAL,
                        verification=CitationVerification(claims=[]),
                        repaired=True,
                        failed_closed=True,
                        removed_claims=len(report.unsupported_claims),
                    )
                keep = [
                    claim.verdict in {"supported", "non_factual"}
                    for claim in report.claims
                ]
                filtered = filter_answer(answer, scored, keep)
                retained = CitationVerification(
                    claims=[
                        claim
                        for claim in report.claims
                        if claim.verdict in {"supported", "non_factual"}
                    ]
                )
                if not filtered:
                    return CitationGuardResult(
                        answer=_SAFE_REFUSAL,
                        verification=CitationVerification(claims=[]),
                        repaired=True,
                        failed_closed=True,
                        removed_claims=len(report.unsupported_claims),
                    )
                return CitationGuardResult(
                    answer=filtered,
                    verification=retained,
                    repaired=filtered != answer,
                    removed_claims=len(report.unsupported_claims),
                )

            if mode == "shadow" and self._selective_backend is not None:
                report, _ = await self._verify_selective(answer, chunks)
                return CitationGuardResult(answer=answer, verification=report)

            report = await self.verify(answer, chunks)
            if mode == "shadow" or not report.unsupported_claims:
                return CitationGuardResult(answer=answer, verification=report)
            repaired = await self._repair(answer, report, chunks)
            repaired_report = await self.verify(repaired, chunks)
            if repaired_report.unsupported_claims:
                return CitationGuardResult(
                    answer=_SAFE_REFUSAL,
                    verification=CitationVerification(claims=[]),
                    repaired=True,
                    failed_closed=True,
                )
            return CitationGuardResult(
                answer=repaired,
                verification=repaired_report,
                repaired=True,
            )
        except Exception as exc:  # noqa: BLE001 - transport/model errors vary
            logger.warning("citation verification failed: %s", exc)
            if mode in {"enforce", "selective"}:
                return CitationGuardResult(
                    answer=_SAFE_REFUSAL,
                    verification=None,
                    failed_closed=True,
                    error=str(exc),
                )
            return CitationGuardResult(
                answer=answer,
                verification=None,
                error=str(exc),
            )
