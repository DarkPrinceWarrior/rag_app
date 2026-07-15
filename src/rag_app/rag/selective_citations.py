"""Селективная claim-level проверка цитат через локальный entailment backend."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol
from urllib.parse import urlsplit

import httpx

Language = Literal["ru", "en", "zh"]

_CITATION = re.compile(r"\[(\d{1,3})\]")
_CLAIM_SPAN = re.compile(r"[^\n.!?。！？;；]+(?:[.!?。！？;；]+|(?=\n)|$)")
_MARKDOWN_PREFIX = re.compile(r"^\s*(?:#{1,6}\s+|[-*•]\s+|\d{1,3}[.)]\s+)")
_MULTISPACE = re.compile(r"[ \t]{2,}")
_BLANK_LINES = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)+")


@dataclass(frozen=True, slots=True)
class ClaimSpan:
    start: int
    end: int
    raw: str
    claim: str
    citation_numbers: tuple[int, ...]
    language: Language
    non_factual: bool = False


@dataclass(frozen=True, slots=True)
class ClaimPair:
    claim: str
    evidence: str
    language: Language


@dataclass(frozen=True, slots=True)
class ScoredClaim:
    span: ClaimSpan
    score: float | None
    supported: bool
    reason: str


class ClaimScoreBackend(Protocol):
    async def score(self, pairs: Sequence[ClaimPair]) -> list[float]: ...


def detect_claim_language(text: str) -> Language:
    if re.search(r"[\u3400-\u9fff]", text):
        return "zh"
    if re.search(r"[А-Яа-яЁё]", text):
        return "ru"
    return "en"


def extract_claim_spans(answer: str) -> list[ClaimSpan]:
    """Разбить RU/EN/ZH ответ на проверяемые предложения с точными offsets."""

    spans: list[ClaimSpan] = []
    for match in _CLAIM_SPAN.finditer(answer):
        raw = match.group(0)
        visible = raw.strip()
        if not visible:
            continue
        citations = tuple(dict.fromkeys(int(value) for value in _CITATION.findall(visible)))
        without_citations = _CITATION.sub("", visible).strip()
        claim = _MARKDOWN_PREFIX.sub("", without_citations).strip()
        claim = claim.rstrip(".!?。！？;").strip()
        if not claim:
            continue
        non_factual = (
            visible.lstrip().startswith("#")
            or (claim.endswith(":") and not citations)
            or claim.casefold() in {"источники", "sources", "来源"}
        )
        spans.append(
            ClaimSpan(
                start=match.start(),
                end=match.end(),
                raw=raw,
                claim=claim,
                citation_numbers=citations,
                language=detect_claim_language(claim),
                non_factual=non_factual,
            )
        )
    return spans


class LocalHttpClaimScoreBackend:
    """Batch HTTP-клиент для локальных HHEM/Lettuce adapters.

    Контракт: POST ``endpoint`` с ``{"model", "pairs": [{claim,evidence,language}]}``;
    ответ — ``{"scores": [0..1]}`` либо список чисел/объектов ``{"score": ...}``.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        model: str,
        adapter: Literal["hhem", "lettuce"] = "hhem",
        timeout_s: float = 15.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        parsed = urlsplit(endpoint)
        if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise ValueError("citation verifier endpoint must be loopback HTTP(S)")
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.adapter = adapter
        self.timeout_s = timeout_s
        self.transport = transport

    async def score(self, pairs: Sequence[ClaimPair]) -> list[float]:
        if not pairs:
            return []
        payload = {
            "backend": self.adapter,
            "model": self.model,
            "pairs": [
                {"claim": pair.claim, "evidence": pair.evidence, "language": pair.language}
                for pair in pairs
            ],
        }
        async with httpx.AsyncClient(timeout=self.timeout_s, transport=self.transport) as client:
            response = await client.post(self.endpoint, json=payload)
            response.raise_for_status()
            body = response.json()
        raw_scores = body.get("scores") if isinstance(body, dict) else body
        if not isinstance(raw_scores, list) or len(raw_scores) != len(pairs):
            raise ValueError("citation verifier returned an invalid score batch")
        scores: list[float] = []
        for item in raw_scores:
            raw = item.get("score") if isinstance(item, dict) else item
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError("citation verifier score is not numeric")
            value = float(raw)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("citation verifier score is outside [0, 1]")
            scores.append(value)
        return scores


async def score_claims(
    answer: str,
    sources: dict[int, str],
    backend: ClaimScoreBackend,
    *,
    threshold: float,
) -> list[ScoredClaim]:
    spans = extract_claim_spans(answer)
    pairs: list[ClaimPair] = []
    pair_indexes: list[int] = []
    results: list[ScoredClaim | None] = [None] * len(spans)
    for index, span in enumerate(spans):
        if span.non_factual:
            results[index] = ScoredClaim(span, None, True, "non-factual heading")
            continue
        cited = [sources[number] for number in span.citation_numbers if number in sources]
        if not cited or len(cited) != len(span.citation_numbers):
            results[index] = ScoredClaim(span, 0.0, False, "missing or invalid citation")
            continue
        pairs.append(ClaimPair(span.claim, "\n".join(cited), span.language))
        pair_indexes.append(index)
    scores = await backend.score(pairs)
    for index, score in zip(pair_indexes, scores, strict=True):
        results[index] = ScoredClaim(
            spans[index],
            score,
            score >= threshold,
            f"entailment={score:.6f}; threshold={threshold:.6f}",
        )
    return [result for result in results if result is not None]


def filter_answer(answer: str, claims: Sequence[ScoredClaim], keep: Sequence[bool]) -> str:
    """Удалить только offsets отклонённых factual claim, сохранив остальные."""

    if len(claims) != len(keep):
        raise ValueError("claim/filter length mismatch")
    filtered = answer
    for item, retained in reversed(list(zip(claims, keep, strict=True))):
        if not retained and not item.span.non_factual:
            filtered = filtered[: item.span.start] + filtered[item.span.end :]
    filtered = _MULTISPACE.sub(" ", filtered)
    filtered = _BLANK_LINES.sub("\n\n", filtered)
    return filtered.strip()
