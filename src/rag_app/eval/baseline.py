"""Deterministic scoring core for the private production RAG baseline."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import re
import statistics
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Literal, Protocol
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy.engine import make_url

from rag_app.eval.gold_set import GoldRecord, gold_record_case_sha256
from rag_app.eval.private_sidecar import PrivateSidecarRecord
from rag_app.eval.rag_metrics import (
    citation_metrics,
    mrr_at_k,
    ndcg_at_k,
    quantity_unit_metrics,
    recall_at_k,
)

_CITATION_RE = re.compile(r"\[(\d{1,4})\]")
_ABSTENTION_MARKERS = (
    "в библиотеке не нашлось",
    "в документах ответа не нашлось",
    "недостаточно данных",
    "нет достаточной информации",
    "не могу предоставить",
    "insufficient data",
    "insufficient evidence",
    "not enough information",
    "cannot provide",
    "没有足够的信息",
    "信息不足",
    "无法提供",
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class RetrievedUnit(_StrictModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID


class BaselineObservation(_StrictModel):
    """Private runtime result. It must never be serialized by the report writer."""

    case_id: str
    gold_case_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scope_id: str = Field(pattern=r"^scope-sha256:[0-9a-f]{64}$")
    answer: str = Field(max_length=32_000)
    retrieved: tuple[RetrievedUnit, ...] = Field(max_length=64)
    retrieval_ms: float = Field(ge=0)
    generation_ms: float = Field(ge=0)
    total_ms: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_observation(self) -> BaselineObservation:
        chunk_ids = [item.chunk_id for item in self.retrieved]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("retrieved chunk IDs must be unique")
        if self.total_ms + 1e-6 < self.retrieval_ms + self.generation_ms:
            raise ValueError("total latency cannot be below component latency")
        return self


class RankedScores(_StrictModel):
    recall: dict[str, Any]
    mrr: dict[str, Any]
    ndcg: dict[str, Any]


class BaselineCaseMetrics(_StrictModel):
    case_id: str
    answerable: bool
    answerability_correct: bool
    abstained: bool
    ranked: dict[str, RankedScores]
    citation: dict[str, Any]
    quantities: dict[str, Any]
    retrieval_ms: float
    generation_ms: float
    total_ms: float


class BaselineModelIdentifiers(_StrictModel):
    llm: str = Field(min_length=1, max_length=256)
    embedding: str = Field(min_length=1, max_length=256)
    reranker: str = Field(min_length=1, max_length=256)
    visual_embedding: str | None = Field(default=None, max_length=256)
    visual_reranker: str | None = Field(default=None, max_length=256)


class RuntimeModelRevision(_StrictModel):
    endpoint_metadata_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_version_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_process_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    local_config_manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    weight_manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    weight_file_count: int = Field(ge=0)
    weight_bytes: int = Field(ge=0)
    declared_revision: str | None = Field(default=None, min_length=1, max_length=256)


class BaselineModelRevisions(_StrictModel):
    llm: RuntimeModelRevision
    embedding: RuntimeModelRevision
    reranker: RuntimeModelRevision
    visual_embedding: RuntimeModelRevision | None = None
    visual_reranker: RuntimeModelRevision | None = None


class BaselineConfiguration(_StrictModel):
    top_k: int = Field(ge=10, le=64)
    dense_top_k: int = Field(ge=1)
    sparse_top_k: int = Field(ge=1)
    # Optional preserves verification of v1 reports produced before point 9.
    sparse_backend: Literal["postgres_fts", "pg_textsearch"] | None = None
    rrf_k: int | None = Field(default=None, ge=1)
    rerank_top_k: int = Field(ge=1)
    rerank_min_score: float
    embedding_dim: int = Field(ge=1)
    visual_enabled: bool
    context_max_chars: int = Field(ge=1)
    context_window_tokens: int = Field(ge=1)
    output_tokens: int = Field(ge=1)
    answer_route: Literal["doc_only"]
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    temperature: float = Field(ge=0, le=2)
    top_p: float = Field(gt=0, le=1)
    seed_namespace: int = Field(ge=0)
    seed_strategy: Literal["case-id-sha256-v1"]
    enable_thinking: Literal[False]


class BaselineProvenance(_StrictModel):
    runner: Literal["retrieval_direct_answer"]
    evaluation_mode: Literal["candidate", "release"]
    evaluated_at: datetime
    git_sha: str | None = Field(default=None, pattern=r"^[0-9a-f]{40,64}$")
    git_dirty: bool | None
    gold_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sidecar_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_corpus_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scope_count: int = Field(ge=1)
    document_snapshot_count: int = Field(ge=1)
    models: BaselineModelIdentifiers
    model_revisions: BaselineModelRevisions
    configuration: BaselineConfiguration
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_evaluated_at(self) -> BaselineProvenance:
        if self.evaluated_at.tzinfo is None or self.evaluated_at.utcoffset() is None:
            raise ValueError("evaluated_at must be timezone-aware")
        configuration_payloads = (
            self.configuration.model_dump(mode="json", exclude_none=True),
            self.configuration.model_dump(mode="json"),
        )
        expected_configuration_sha256 = {
            hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            for payload in configuration_payloads
        }
        if self.configuration_sha256 not in expected_configuration_sha256:
            raise ValueError("configuration_sha256 does not match configuration")
        return self


class BaselineReport(_StrictModel):
    schema_version: Literal["rag-baseline-report-v1"]
    provenance: BaselineProvenance
    case_count: int
    answerable_count: int
    no_answer_count: int
    answerability_accuracy: float
    mean_recall: dict[str, float | None]
    mean_mrr: dict[str, float | None]
    mean_ndcg: dict[str, float | None]
    mean_citation_precision: float | None
    mean_citation_recall: float | None
    mean_quantity_unit_accuracy: float | None
    mean_quantity_unit_recall: float | None
    unsupported_number_rate: float
    latency_ms: dict[str, float]
    cases: tuple[BaselineCaseMetrics, ...]


class BaselineRunner(Protocol):
    async def run_case(
        self,
        record: GoldRecord,
        sidecar: PrivateSidecarRecord,
    ) -> BaselineObservation: ...


class BaselineEvaluationError(ValueError):
    """Fail-closed error that does not include questions, answers or quotes."""


def _loopback_host(host: str | None, *, name: str) -> str:
    if not host:
        raise BaselineEvaluationError(f"{name} has no host")
    if host.casefold() == "localhost":
        return host
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        raise BaselineEvaluationError(f"{name} must use a literal loopback host") from None
    if not address.is_loopback:
        raise BaselineEvaluationError(f"{name} must be loopback-only")
    return host


def require_loopback_url(value: str, *, name: str) -> str:
    parsed = urlparse(value.strip())
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise BaselineEvaluationError(
            f"{name} must be a credential-free HTTP(S) URL without query or fragment"
        )
    _loopback_host(parsed.hostname, name=name)
    return value


def require_loopback_endpoint(value: str, *, name: str) -> str:
    parsed = urlparse(value if "://" in value else f"http://{value}")
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise BaselineEvaluationError(
            f"{name} must be a credential-free HTTP(S) endpoint without query or fragment"
        )
    _loopback_host(parsed.hostname, name=name)
    return value


def require_loopback_database_url(value: str) -> str:
    parsed = make_url(value)
    if parsed.drivername != "postgresql+asyncpg":
        raise BaselineEvaluationError("database must use postgresql+asyncpg")
    _loopback_host(parsed.host, name="database")
    return value


def is_abstention(answer: str) -> bool:
    normalized = " ".join(answer.casefold().split())
    return not normalized or any(marker in normalized for marker in _ABSTENTION_MARKERS)


def _quantity_payload(sidecar: PrivateSidecarRecord, name: Literal["expected", "supported"]):
    values = getattr(sidecar.quantities, name)
    return [{"value": item.value, "unit": item.unit} for item in values]


def score_observation(
    record: GoldRecord,
    sidecar: PrivateSidecarRecord,
    observation: BaselineObservation,
    *,
    ks: Sequence[int] = (1, 5, 10),
) -> BaselineCaseMetrics:
    if observation.case_id != record.case_id:
        raise BaselineEvaluationError("runtime case binding mismatch")
    if observation.gold_case_sha256 != gold_record_case_sha256(record):
        raise BaselineEvaluationError("runtime gold hash mismatch")
    if observation.scope_id != record.scope_id or sidecar.scope_id != record.scope_id:
        raise BaselineEvaluationError("runtime scope mismatch")

    evidence_by_chunk = {item.chunk_id: item.evidence_id for item in sidecar.exact_evidence}
    ranked_refs: list[str] = []
    ranked_groups: list[set[str]] = []
    for rank, item in enumerate(observation.retrieved, start=1):
        evidence_id = evidence_by_chunk.get(item.chunk_id)
        ranked_refs.append(evidence_id or f"unmatched-rank:{rank}")
        ranked_groups.append({evidence_id} if evidence_id else set())
    relevance = {item.evidence_id: item.relevance_grade for item in record.evidence}
    ranked: dict[str, RankedScores] = {}
    for k in ks:
        ranked[str(k)] = RankedScores(
            recall=dict(recall_at_k(ranked_refs, relevance, k, answerable=record.answerable)),
            mrr=dict(mrr_at_k(ranked_refs, relevance, k, answerable=record.answerable)),
            ndcg=dict(ndcg_at_k(ranked_refs, relevance, k, answerable=record.answerable)),
        )

    citation_ranks = [int(match.group(1)) for match in _CITATION_RE.finditer(observation.answer)]
    citation = citation_metrics(
        citation_ranks,
        ranked_groups,
        set(relevance),
        answerable=record.answerable,
    )
    quantities = quantity_unit_metrics(
        observation.answer,
        _quantity_payload(sidecar, "expected"),
        supported_quantities=_quantity_payload(sidecar, "supported"),
        answerable=record.answerable,
    )
    abstained = is_abstention(observation.answer)
    answerability_correct = abstained if not record.answerable else not abstained
    return BaselineCaseMetrics(
        case_id=record.case_id,
        answerable=record.answerable,
        answerability_correct=answerability_correct,
        abstained=abstained,
        ranked=ranked,
        citation=dict(citation),
        quantities=dict(quantities),
        retrieval_ms=observation.retrieval_ms,
        generation_ms=observation.generation_ms,
        total_ms=observation.total_ms,
    )


def _mean_eligible(values: Sequence[float | None]) -> float | None:
    eligible = [value for value in values if value is not None]
    return None if not eligible else sum(eligible) / len(eligible)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = math.ceil(percentile * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def aggregate_metrics(
    cases: Sequence[BaselineCaseMetrics],
    *,
    provenance: BaselineProvenance,
) -> BaselineReport:
    if not cases:
        raise BaselineEvaluationError("cannot aggregate an empty evaluation")
    ks = sorted({key for case in cases for key in case.ranked}, key=int)

    def metric_values(name: Literal["recall", "mrr", "ndcg"], k: str) -> list[float | None]:
        values: list[float | None] = []
        for case in cases:
            payload = getattr(case.ranked[k], name)
            value = payload["value"]
            values.append(float(value) if isinstance(value, (int, float)) else None)
        return values

    total_mentions = sum(int(case.quantities["mentioned_number_count"]) for case in cases)
    unsupported = sum(int(case.quantities["unsupported_number_count"]) for case in cases)
    totals = [case.total_ms for case in cases]
    retrieval = [case.retrieval_ms for case in cases]
    generation = [case.generation_ms for case in cases]
    return BaselineReport(
        schema_version="rag-baseline-report-v1",
        provenance=provenance,
        case_count=len(cases),
        answerable_count=sum(case.answerable for case in cases),
        no_answer_count=sum(not case.answerable for case in cases),
        answerability_accuracy=sum(case.answerability_correct for case in cases) / len(cases),
        mean_recall={k: _mean_eligible(metric_values("recall", k)) for k in ks},
        mean_mrr={k: _mean_eligible(metric_values("mrr", k)) for k in ks},
        mean_ndcg={k: _mean_eligible(metric_values("ndcg", k)) for k in ks},
        mean_citation_precision=_mean_eligible(
            [
                float(value)
                if isinstance(value := case.citation["citation_precision"], (int, float))
                else None
                for case in cases
            ]
        ),
        mean_citation_recall=_mean_eligible(
            [
                float(value) if isinstance(value := case.citation["citation_recall"], (int, float)) else None
                for case in cases
            ]
        ),
        mean_quantity_unit_accuracy=_mean_eligible(
            [
                float(value)
                if isinstance(value := case.quantities["quantity_unit_accuracy"], (int, float))
                else None
                for case in cases
            ]
        ),
        mean_quantity_unit_recall=_mean_eligible(
            [
                float(value)
                if isinstance(value := case.quantities["quantity_unit_recall"], (int, float))
                else None
                for case in cases
            ]
        ),
        unsupported_number_rate=unsupported / total_mentions if total_mentions else 0.0,
        latency_ms={
            "retrieval_mean": statistics.fmean(retrieval),
            "generation_mean": statistics.fmean(generation),
            "total_mean": statistics.fmean(totals),
            "total_p50": statistics.median(totals),
            "total_p95": _percentile(totals, 0.95),
        },
        cases=tuple(cases),
    )


async def evaluate_baseline(
    records: Sequence[GoldRecord],
    sidecars: Mapping[str, PrivateSidecarRecord],
    runner: BaselineRunner,
    *,
    provenance: BaselineProvenance,
    ks: Sequence[int] = (1, 5, 10),
) -> BaselineReport:
    results: list[BaselineCaseMetrics] = []
    for record in records:
        try:
            sidecar = sidecars[record.case_id]
            observation = await runner.run_case(record, sidecar)
            results.append(score_observation(record, sidecar, observation, ks=ks))
        except (KeyError, ValueError) as error:
            raise BaselineEvaluationError(f"evaluation failed closed ({type(error).__name__})") from None
    return aggregate_metrics(results, provenance=provenance)


__all__ = [
    "BaselineCaseMetrics",
    "BaselineEvaluationError",
    "BaselineObservation",
    "BaselineConfiguration",
    "BaselineModelIdentifiers",
    "BaselineModelRevisions",
    "BaselineProvenance",
    "BaselineReport",
    "BaselineRunner",
    "RetrievedUnit",
    "RuntimeModelRevision",
    "aggregate_metrics",
    "evaluate_baseline",
    "is_abstention",
    "require_loopback_database_url",
    "require_loopback_endpoint",
    "require_loopback_url",
    "score_observation",
]
