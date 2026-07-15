#!/usr/bin/env python3
"""Fail-closed paired qualification for hierarchical retrieval.

The quality phase collects the baseline and hierarchical candidate pools once,
scores their union with several independent frozen reranker snapshots, and
reuses each snapshot for both variants.  Serialized artifacts contain hashes,
stable identifiers, labels, rankings and metrics, but never questions or
document text.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import json
import math
import random
import time
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator
from sqlalchemy import text as sql
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from rag_app.config import settings
from rag_app.db.engine import create_sessionmaker
from rag_app.db.rls import assert_api_rls_role, reset_principal, set_principal
from rag_app.eval.baseline import (
    require_loopback_database_url,
    require_loopback_endpoint,
    require_loopback_url,
)
from rag_app.eval.gold_set import GoldRecord, parse_gold_set_bytes
from rag_app.eval.private_artifacts import read_private_bytes, read_private_json, write_private_json_fresh
from rag_app.eval.private_sidecar import bind_gold_sidecar, parse_private_sidecar_bytes
from rag_app.eval.report_attestation import (
    atomic_write_private_artifact_attestation,
    create_private_artifact_attestation,
    load_hmac_key,
)
from rag_app.eval.retrieval_gate import canonical_sha256
from rag_app.llm.embeddings import Embedder, Reranker
from rag_app.rag.retrieve import RetrievedChunk, Retriever
from rag_app.storage.s3 import Storage

try:
    from scripts.evaluate_retrieval_bm25 import (
        ModelEndpointRevision,
        _CollectionReranker,
        _CorpusVerifier,
        _git_state,
        _load_model_revision,
        _PairedQueryEmbedder,
        _private_dir,
        _statistical_cluster_ids,
        build_case_bindings,
        stratified_cluster_split,
    )
except ModuleNotFoundError:  # direct ``python scripts/...`` execution
    from evaluate_retrieval_bm25 import (  # type: ignore[no-redef,import-not-found]
        ModelEndpointRevision,
        _CollectionReranker,
        _CorpusVerifier,
        _git_state,
        _load_model_revision,
        _PairedQueryEmbedder,
        _private_dir,
        _statistical_cluster_ids,
        build_case_bindings,
        stratified_cluster_split,
    )

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "hierarchical-retrieval-report-v1"
EXPECTED_RELEASE_CASES: Literal[236] = 236
EXPECTED_NO_ANSWER_CASES: Literal[67] = 67
MIN_LOCKED_CASES = 200
MAX_TUNING_CASES = 36
MIN_SCORE_SNAPSHOTS = 3
MIN_LOAD_CONCURRENCY = 10
MIN_LOAD_REQUESTS_PER_VARIANT = 200
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 2026071511
GLOBAL_NONINFERIORITY_MARGIN = 0.01
TARGET_NDCG_GAIN = 0.02
TARGET_FULL_EVIDENCE_GAIN = 0.03
TARGET_STRUCTURED_RECALL_GAIN = 0.02
MIN_ROUTE_RECALL_AT_5 = 0.95
MIN_EVIDENCE_RECALL_AT_20 = 0.95
MAX_P95_RATIO = 1.10
MAX_P95_INCREASE_MS = 250.0
MIN_THROUGHPUT_RATIO = 0.90
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_IMAGE_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"
_CASE_PATTERN = r"^ragq-[a-z0-9][a-z0-9._-]{7,63}$"
_SCOPE_PATTERN = r"^scope-sha256:[0-9a-f]{64}$"
_CLUSTER_PATTERN = r"^cluster-sha256:[0-9a-f]{64}$"
_SCORE_PATTERN = r"^0x[0-9a-f]+(?:\.[0-9a-f]+)?p[+-][0-9]+$"
_ARTIFACT_HMAC_DOMAIN = b"docragenslate/hierarchical-retrieval/v1\0"
_ATTESTED_SOURCES = (
    "scripts/evaluate_hierarchical_retrieval.py",
    "scripts/evaluate_retrieval_bm25.py",
    "src/rag_app/config.py",
    "src/rag_app/db/rls.py",
    "src/rag_app/eval/gold_set.py",
    "src/rag_app/eval/private_artifacts.py",
    "src/rag_app/eval/private_sidecar.py",
    "src/rag_app/eval/report_attestation.py",
    "src/rag_app/eval/retrieval_gate.py",
    "src/rag_app/llm/embeddings.py",
    "src/rag_app/rag/chunking.py",
    "src/rag_app/rag/hierarchy_backfill.py",
    "src/rag_app/rag/retrieve.py",
    "src/rag_app/rag/tree_chunking.py",
    "scripts/backfill_chunk_hierarchy.py",
    "uv.lock",
)

RunMode = Literal["debug", "qualification"]
VariantName = Literal["baseline", "candidate"]


class HierarchicalEvaluationError(RuntimeError):
    """The run cannot produce trustworthy hierarchical retrieval evidence."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        str_strip_whitespace=True,
    )


class FrozenScoreEntry(_StrictModel):
    question_sha256: str = Field(pattern=_SHA256_PATTERN)
    text_sha256: str = Field(pattern=_SHA256_PATTERN)
    score_hex: str = Field(pattern=_SCORE_PATTERN)

    @model_validator(mode="after")
    def validate_score(self) -> FrozenScoreEntry:
        value = float.fromhex(self.score_hex)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("frozen reranker score is outside [0, 1]")
        return self


class FrozenScoreSnapshot(_StrictModel):
    schema_version: Literal["hierarchical-frozen-scores-v1"] = "hierarchical-frozen-scores-v1"
    snapshot_index: int = Field(ge=0, le=63)
    pair_count: int = Field(ge=1, le=1_000_000)
    entries: tuple[FrozenScoreEntry, ...] = Field(min_length=1, max_length=1_000_000)
    pair_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    scoring_seconds: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_entries(self) -> FrozenScoreSnapshot:
        keys = [(item.question_sha256, item.text_sha256) for item in self.entries]
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise ValueError("frozen score pairs must be unique and canonically ordered")
        if self.pair_count != len(self.entries):
            raise ValueError("frozen score pair_count is inconsistent")
        if self.pair_manifest_sha256 != canonical_sha256(self.entries):
            raise ValueError("frozen score manifest hash mismatch")
        return self


class ArtifactHmac(_StrictModel):
    schema_version: Literal["hierarchical-artifact-hmac-v1"] = "hierarchical-artifact-hmac-v1"
    artifact_type: str = Field(min_length=1, max_length=128)
    artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    key_id: str = Field(pattern=_SHA256_PATTERN)
    signature: str = Field(pattern=_SHA256_PATTERN)


class VariantMetrics(_StrictModel):
    recall_at_5: float = Field(ge=0.0, le=1.0)
    recall_at_10: float = Field(ge=0.0, le=1.0)
    ndcg_at_10: float = Field(ge=0.0, le=1.0)
    full_evidence_at_10: float = Field(ge=0.0, le=1.0)
    returned_count: int = Field(ge=0, le=1_000)
    abstained: StrictBool

    @model_validator(mode="after")
    def validate_abstention(self) -> VariantMetrics:
        if self.abstained != (self.returned_count == 0):
            raise ValueError("abstention must match returned_count")
        return self


class SnapshotCaseMetrics(_StrictModel):
    snapshot_index: int = Field(ge=0, le=63)
    baseline: VariantMetrics
    candidate: VariantMetrics
    baseline_order_sha256: str = Field(pattern=_SHA256_PATTERN)
    candidate_order_sha256: str = Field(pattern=_SHA256_PATTERN)


class HierarchicalCaseResult(_StrictModel):
    case_id: str = Field(pattern=_CASE_PATTERN)
    gold_case_sha256: str = Field(pattern=_SHA256_PATTERN)
    question_sha256: str = Field(pattern=_SHA256_PATTERN)
    query_embedding_sha256: str = Field(pattern=_SHA256_PATTERN)
    scope_id: str = Field(pattern=_SCOPE_PATTERN)
    cluster_id: str = Field(pattern=_CLUSTER_PATTERN)
    split: Literal["tuning", "locked"]
    language: Literal["ru", "en", "zh"]
    hop_type: Literal["single", "multi", "cross_document"]
    content_types: tuple[Literal["text", "table", "formula", "figure", "scan"], ...]
    challenge_tags: tuple[Literal["numbers", "units", "standards", "prompt_injection", "leakage"], ...]
    answerable: StrictBool
    relevant_count: int = Field(ge=0, le=16)
    route_recall_at_5: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_recall_at_20: float | None = Field(default=None, ge=0.0, le=1.0)
    baseline_pool_count: int = Field(ge=0, le=1_000)
    candidate_pool_count: int = Field(ge=0, le=1_000)
    hierarchical_added: int = Field(ge=0, le=1_000)
    hierarchy_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    scope_violation_count: Literal[0]
    hierarchical_fallback: Literal[False]
    snapshots: tuple[SnapshotCaseMetrics, ...] = Field(min_length=MIN_SCORE_SNAPSHOTS, max_length=64)

    @model_validator(mode="after")
    def validate_case(self) -> HierarchicalCaseResult:
        indexes = [item.snapshot_index for item in self.snapshots]
        if indexes != list(range(len(indexes))):
            raise ValueError("case score snapshots are incomplete or unordered")
        if self.answerable != (self.relevant_count > 0):
            raise ValueError("answerability and relevant_count are inconsistent")
        if self.answerable != (self.route_recall_at_5 is not None):
            raise ValueError("only answerable cases have route metrics")
        if self.answerable != (self.evidence_recall_at_20 is not None):
            raise ValueError("only answerable cases have evidence metrics")
        if self.candidate_pool_count != self.baseline_pool_count + self.hierarchical_added:
            raise ValueError("hierarchical candidate count is inconsistent")
        return self


class MetricDecision(_StrictModel):
    metric: str = Field(min_length=2, max_length=128)
    eligible_case_count: int = Field(ge=1, le=500)
    cluster_count: int = Field(ge=2, le=500)
    baseline: float = Field(ge=0.0, le=1.0)
    candidate: float = Field(ge=0.0, le=1.0)
    improvement: float = Field(ge=-1.0, le=1.0)
    ci_low: float = Field(ge=-1.0, le=1.0)
    ci_high: float = Field(ge=-1.0, le=1.0)
    minimum_improvement: float = Field(ge=-1.0, le=1.0)
    passed: StrictBool


class AbsoluteDecision(_StrictModel):
    metric: str = Field(min_length=2, max_length=128)
    eligible_case_count: int = Field(ge=1, le=500)
    value: float = Field(ge=0.0, le=1.0)
    minimum: float = Field(ge=0.0, le=1.0)
    passed: StrictBool


class HierarchicalQualityDecision(_StrictModel):
    global_metrics: tuple[MetricDecision, ...] = Field(min_length=5)
    slice_metrics: tuple[MetricDecision, ...] = Field(min_length=1)
    absolutes: tuple[AbsoluteDecision, ...] = Field(min_length=2)
    per_snapshot_noninferiority: tuple[StrictBool, ...] = Field(min_length=MIN_SCORE_SNAPSHOTS)
    no_answer: MetricDecision
    failure_codes: tuple[str, ...]
    accepted: StrictBool

    @model_validator(mode="after")
    def validate_decision(self) -> HierarchicalQualityDecision:
        if self.accepted == bool(self.failure_codes):
            raise ValueError("quality acceptance and failure codes are inconsistent")
        if len(self.failure_codes) != len(set(self.failure_codes)):
            raise ValueError("quality failure codes must be unique")
        return self


class RlsEvidence(_StrictModel):
    schema_version: Literal["hierarchical-rls-evidence-v1"]
    principal_count: int = Field(ge=10, le=100)
    owner_scope_count: int = Field(ge=2, le=100)
    probe_count: int = Field(ge=10)
    admin_foreign_truth_count: int = Field(ge=1)
    anonymous_visible_count: Literal[0]
    leak_count: Literal[0]
    scope_violation_count: Literal[0]
    passed: Literal[True]


class LoadRequestObservation(_StrictModel):
    request_id: str = Field(pattern=_SHA256_PATTERN)
    pair_index: int = Field(ge=0)
    order_in_pair: Literal[0, 1]
    case_id: str = Field(pattern=_CASE_PATTERN)
    variant: VariantName
    started_at: datetime
    completed_at: datetime
    latency_ms: float = Field(ge=0.0)
    returned_count: int = Field(ge=0, le=1_000)
    order_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    success: StrictBool
    error_code: str | None = Field(default=None, pattern=r"^[A-Za-z][A-Za-z0-9_]{0,127}$")

    @model_validator(mode="after")
    def validate_outcome(self) -> LoadRequestObservation:
        if self.started_at.tzinfo is None or self.completed_at.tzinfo is None:
            raise ValueError("load timestamps must be timezone-aware")
        if self.completed_at < self.started_at:
            raise ValueError("load completion predates start")
        if self.success != (self.error_code is None and self.order_sha256 is not None):
            raise ValueError("load outcome fields are inconsistent")
        return self


class LoadAggregate(_StrictModel):
    request_count: int = Field(ge=1)
    completed_count: int = Field(ge=0)
    error_count: int = Field(ge=0)
    p95_latency_ms: float = Field(gt=0.0)
    throughput_rps: float = Field(gt=0.0)


class LoadEvidence(_StrictModel):
    schema_version: Literal["hierarchical-load-evidence-v1"] = "hierarchical-load-evidence-v1"
    concurrency: int = Field(ge=1)
    observed_peak_concurrency: int = Field(ge=1)
    requests_per_variant: int = Field(ge=1)
    warmups_per_variant: int = Field(ge=0)
    raw_observations_sha256: str = Field(pattern=_SHA256_PATTERN)
    baseline: LoadAggregate
    candidate: LoadAggregate
    maximum_candidate_p95_ms: float = Field(gt=0.0)
    throughput_ratio: float = Field(gt=0.0)
    passed: StrictBool

    @model_validator(mode="after")
    def validate_load(self) -> LoadEvidence:
        if self.observed_peak_concurrency != self.concurrency:
            raise ValueError("load did not reach declared concurrency")
        for value in (self.baseline, self.candidate):
            if value.request_count != self.requests_per_variant:
                raise ValueError("load request count is incomplete")
            if value.completed_count + value.error_count != value.request_count:
                raise ValueError("load aggregate counts do not add up")
        expected = self.candidate.throughput_rps / self.baseline.throughput_rps
        if not math.isclose(self.throughput_ratio, expected, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("load throughput ratio is inconsistent")
        expected_passed = (
            self.baseline.error_count == 0
            and self.candidate.error_count == 0
            and self.candidate.p95_latency_ms <= self.maximum_candidate_p95_ms
            and self.throughput_ratio >= MIN_THROUGHPUT_RATIO
        )
        if self.passed != expected_passed:
            raise ValueError("load decision is inconsistent")
        return self


class DatabaseEvidence(_StrictModel):
    image_digest: str = Field(pattern=_IMAGE_DIGEST_PATTERN)
    server_version_num: int = Field(ge=170_000, lt=180_000)
    extensions: dict[str, str]
    extension_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    chunk_index_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)


class ModelEvidence(_StrictModel):
    embedding: ModelEndpointRevision
    reranker: ModelEndpointRevision
    embedding_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    reranker_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)


class HierarchicalProvenance(_StrictModel):
    repository_sha: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    repository_dirty: StrictBool
    gold_sha256: str = Field(pattern=_SHA256_PATTERN)
    sidecar_sha256: str = Field(pattern=_SHA256_PATTERN)
    case_count: Literal[236]
    no_answer_case_count: Literal[67]
    split_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    runtime_corpus_sha256_before: str = Field(pattern=_SHA256_PATTERN)
    runtime_corpus_sha256_after: str = Field(pattern=_SHA256_PATTERN)
    hierarchy_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    query_vector_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    score_snapshot_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    score_snapshot_count: int = Field(ge=MIN_SCORE_SNAPSHOTS, le=64)
    retrieval_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    database: DatabaseEvidence
    models: ModelEvidence
    rls_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    load_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_corpus_binding(self) -> HierarchicalProvenance:
        if self.runtime_corpus_sha256_before != self.runtime_corpus_sha256_after:
            raise ValueError("runtime corpus changed during qualification")
        return self


class HierarchicalReport(_StrictModel):
    schema_version: Literal["hierarchical-retrieval-report-v1"] = "hierarchical-retrieval-report-v1"
    run_id: str = Field(pattern=_SHA256_PATTERN)
    mode: RunMode
    evaluated_at: datetime
    provenance: HierarchicalProvenance
    cases: tuple[HierarchicalCaseResult, ...] = Field(
        min_length=EXPECTED_RELEASE_CASES,
        max_length=EXPECTED_RELEASE_CASES,
    )
    score_snapshots: tuple[FrozenScoreSnapshot, ...] = Field(
        min_length=MIN_SCORE_SNAPSHOTS,
        max_length=64,
    )
    rls: RlsEvidence
    load: LoadEvidence
    quality: HierarchicalQualityDecision
    release_accepted: StrictBool

    @model_validator(mode="after")
    def validate_report(self) -> HierarchicalReport:
        if self.evaluated_at.tzinfo is None or self.evaluated_at.utcoffset() is None:
            raise ValueError("report timestamp must be timezone-aware")
        case_ids = [item.case_id for item in self.cases]
        if case_ids != sorted(case_ids) or len(case_ids) != len(set(case_ids)):
            raise ValueError("report cases must be unique and sorted")
        indexes = [item.snapshot_index for item in self.score_snapshots]
        if indexes != list(range(len(indexes))):
            raise ValueError("report score snapshots are incomplete or unordered")
        expected = (
            self.mode == "qualification"
            and self.quality.accepted
            and self.rls.passed
            and self.load.passed
        )
        if self.release_accepted != expected:
            raise ValueError("release decision is inconsistent")
        return self


@dataclass(frozen=True, slots=True)
class _CollectedCase:
    binding: Any
    scope: Any
    baseline: tuple[RetrievedChunk, ...]
    candidate: tuple[RetrievedChunk, ...]
    hierarchy_manifest_sha256: str
    hierarchy_added: int


@dataclass(frozen=True, slots=True)
class _PairedValue:
    cluster_id: str
    baseline: float
    candidate: float


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _chunk_text(chunk: RetrievedChunk) -> str:
    return (chunk.text_ru or chunk.text_en)[:4000]


def _chunk_manifest(chunk: RetrievedChunk) -> dict[str, Any]:
    return {
        "id": str(chunk.id),
        "document_id": str(chunk.document_id),
        "kind": chunk.kind,
        "heading_path_sha256": _text_sha256(chunk.heading_path),
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "text_en_sha256": _text_sha256(chunk.text_en),
        "text_ru_sha256": _text_sha256(chunk.text_ru),
        "meta_sha256": _sha256_json(chunk.meta),
    }


def _order_sha256(chunks: Sequence[RetrievedChunk]) -> str:
    return _sha256_json([str(chunk.id) for chunk in chunks])


def _artifact_signature(artifact_type: str, raw: bytes, key: bytes) -> str:
    payload = _ARTIFACT_HMAC_DOMAIN + artifact_type.encode() + b"\0" + raw
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def _write_signed_private_json(
    path: Path,
    *,
    artifact_type: str,
    payload: BaseModel | Mapping[str, Any] | Sequence[Any],
    key: bytes,
) -> tuple[bytes, str]:
    value = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
    raw = _canonical_bytes(value)
    artifact = write_private_json_fresh(path, raw, max_bytes=256 * 1024 * 1024)
    envelope = ArtifactHmac(
        artifact_type=artifact_type,
        artifact_sha256=artifact.sha256,
        key_id=_sha256_bytes(key),
        signature=_artifact_signature(artifact_type, raw, key),
    )
    signature_path = path.with_name(f"{path.stem}.hmac.json")
    write_private_json_fresh(signature_path, _canonical_bytes(envelope.model_dump(mode="json")))
    reread = read_private_bytes(path, max_bytes=256 * 1024 * 1024)
    if reread.sha256 != artifact.sha256 or not hmac.compare_digest(
        envelope.signature,
        _artifact_signature(artifact_type, reread.raw_bytes, key),
    ):
        raise HierarchicalEvaluationError("private artifact HMAC readback failed")
    return raw, artifact.sha256


def _metric_recall(
    ranked: Sequence[RetrievedChunk],
    relevance: Mapping[uuid.UUID, int],
    k: int,
) -> float:
    if not relevance:
        return 0.0
    return len({chunk.id for chunk in ranked[:k]} & set(relevance)) / len(relevance)


def _route_document_recall(
    ranked: Sequence[RetrievedChunk],
    relevant_document_ids: set[uuid.UUID],
    k: int,
) -> float:
    if not relevant_document_ids:
        return 0.0
    returned = {chunk.document_id for chunk in ranked[:k]}
    return len(returned & relevant_document_ids) / len(relevant_document_ids)


def _metric_ndcg(
    ranked: Sequence[RetrievedChunk],
    relevance: Mapping[uuid.UUID, int],
    k: int,
) -> float:
    if not relevance:
        return 0.0
    gains = [float(relevance.get(chunk.id, 0)) for chunk in ranked[:k]]
    actual = math.fsum((2.0**gain - 1.0) / math.log2(index + 2.0) for index, gain in enumerate(gains))
    ideal_gains = sorted((float(value) for value in relevance.values()), reverse=True)[:k]
    ideal = math.fsum((2.0**gain - 1.0) / math.log2(index + 2.0) for index, gain in enumerate(ideal_gains))
    return actual / ideal if ideal else 0.0


def _variant_metrics(
    ranked: Sequence[RetrievedChunk],
    relevance: Mapping[uuid.UUID, int],
    *,
    answerable: bool,
) -> VariantMetrics:
    if answerable:
        relevant = set(relevance)
        full = float(relevant.issubset({chunk.id for chunk in ranked[:10]}))
        recall_5 = _metric_recall(ranked, relevance, 5)
        recall_10 = _metric_recall(ranked, relevance, 10)
        ndcg_10 = _metric_ndcg(ranked, relevance, 10)
    else:
        recall_5 = recall_10 = ndcg_10 = full = 0.0
    return VariantMetrics(
        recall_at_5=recall_5,
        recall_at_10=recall_10,
        ndcg_at_10=ndcg_10,
        full_evidence_at_10=full,
        returned_count=len(ranked),
        abstained=not ranked,
    )


def _score_map(snapshot: FrozenScoreSnapshot) -> dict[tuple[str, str], float]:
    return {
        (item.question_sha256, item.text_sha256): float.fromhex(item.score_hex) for item in snapshot.entries
    }


def _rank_with_frozen_scores(
    chunks: Sequence[RetrievedChunk],
    *,
    question_sha256: str,
    scores: Mapping[tuple[str, str], float],
    top_k: int,
    minimum_score: float,
) -> tuple[RetrievedChunk, ...]:
    if not chunks:
        return ()
    ranked: list[tuple[RetrievedChunk, float]] = []
    for chunk in chunks:
        key = (question_sha256, _text_sha256(_chunk_text(chunk)))
        if key not in scores:
            raise HierarchicalEvaluationError("frozen reranker score cache missed a candidate pair")
        score = scores[key]
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise HierarchicalEvaluationError("frozen reranker score is invalid")
        ranked.append((chunk, score))
    ranked.sort(key=lambda item: (-item[1], item[0].id.int))
    if ranked[0][1] < minimum_score:
        return ()
    return tuple(chunk for chunk, _ in ranked[:top_k])


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise HierarchicalEvaluationError("bootstrap sample is empty")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _paired_bootstrap(
    rows: Sequence[_PairedValue],
    *,
    seed: int,
    samples: int = BOOTSTRAP_SAMPLES,
) -> tuple[float, float, float, float, float, int]:
    if not rows or not all(math.isfinite(value) for row in rows for value in (row.baseline, row.candidate)):
        raise HierarchicalEvaluationError("paired bootstrap input is empty or non-finite")
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row.cluster_id].append(row.candidate - row.baseline)
    if len(grouped) < 2:
        raise HierarchicalEvaluationError("paired bootstrap requires at least two clusters")
    clusters = tuple((math.fsum(values), len(values)) for _, values in sorted(grouped.items()))
    rng = random.Random(seed)
    bootstrapped: list[float] = []
    for _ in range(samples):
        total = 0.0
        count = 0
        for _ in range(len(clusters)):
            cluster_sum, cluster_count = clusters[rng.randrange(len(clusters))]
            total += cluster_sum
            count += cluster_count
        bootstrapped.append(total / count)
    baseline = math.fsum(row.baseline for row in rows) / len(rows)
    candidate = math.fsum(row.candidate for row in rows) / len(rows)
    improvement = candidate - baseline
    return (
        baseline,
        candidate,
        improvement,
        _quantile(bootstrapped, 0.025),
        _quantile(bootstrapped, 0.975),
        len(clusters),
    )


def _case_metric(case: HierarchicalCaseResult, variant: VariantName, metric: str) -> float:
    values = [float(getattr(getattr(snapshot, variant), metric)) for snapshot in case.snapshots]
    return math.fsum(values) / len(values)


def _metric_decision(
    cases: Sequence[HierarchicalCaseResult],
    *,
    metric: str,
    minimum_improvement: float,
    seed_label: str,
) -> MetricDecision:
    rows = [
        _PairedValue(
            cluster_id=case.cluster_id,
            baseline=_case_metric(case, "baseline", metric),
            candidate=_case_metric(case, "candidate", metric),
        )
        for case in cases
    ]
    baseline, candidate, improvement, ci_low, ci_high, clusters = _paired_bootstrap(
        rows,
        seed=BOOTSTRAP_SEED ^ int.from_bytes(hashlib.sha256(seed_label.encode()).digest()[:8], "big"),
    )
    return MetricDecision(
        metric=seed_label,
        eligible_case_count=len(rows),
        cluster_count=clusters,
        baseline=baseline,
        candidate=candidate,
        improvement=improvement,
        ci_low=ci_low,
        ci_high=ci_high,
        minimum_improvement=minimum_improvement,
        passed=ci_low >= minimum_improvement,
    )


def evaluate_quality(cases: Sequence[HierarchicalCaseResult]) -> HierarchicalQualityDecision:
    locked = [case for case in cases if case.split == "locked"]
    answerable = [case for case in locked if case.answerable]
    if len(locked) < MIN_LOCKED_CASES or not answerable:
        raise HierarchicalEvaluationError("locked evaluation set is incomplete")
    failures: list[str] = []
    global_metrics = [
        _metric_decision(
            answerable,
            metric="recall_at_5",
            minimum_improvement=-GLOBAL_NONINFERIORITY_MARGIN,
            seed_label="global:recall_at_5",
        ),
        _metric_decision(
            answerable,
            metric="recall_at_10",
            minimum_improvement=-GLOBAL_NONINFERIORITY_MARGIN,
            seed_label="global:recall_at_10",
        ),
        _metric_decision(
            answerable,
            metric="ndcg_at_10",
            minimum_improvement=-GLOBAL_NONINFERIORITY_MARGIN,
            seed_label="global:ndcg_at_10_noninferiority",
        ),
        _metric_decision(
            answerable,
            metric="ndcg_at_10",
            minimum_improvement=TARGET_NDCG_GAIN,
            seed_label="global:ndcg_at_10_target",
        ),
    ]
    multi = [case for case in answerable if case.hop_type in {"multi", "cross_document"}]
    structured = [case for case in answerable if set(case.content_types) & {"table", "figure"}]
    global_metrics.append(
        _metric_decision(
            multi,
            metric="full_evidence_at_10",
            minimum_improvement=TARGET_FULL_EVIDENCE_GAIN,
            seed_label="multi_cross:full_evidence_at_10_target",
        )
    )
    global_metrics.append(
        _metric_decision(
            structured,
            metric="recall_at_10",
            minimum_improvement=TARGET_STRUCTURED_RECALL_GAIN,
            seed_label="structured:recall_at_10_target",
        )
    )
    for item in global_metrics:
        if not item.passed:
            failures.append(f"metric:{item.metric}")

    labels: dict[str, list[HierarchicalCaseResult]] = defaultdict(list)
    for case in answerable:
        case_labels = {
            f"language:{case.language}",
            f"hop:{case.hop_type}",
            f"scope:{case.scope_id}",
            *(f"content:{value}" for value in case.content_types),
        }
        for label in case_labels:
            labels[label].append(case)
    slice_metrics: list[MetricDecision] = []
    for label, selected in sorted(labels.items()):
        for metric in ("recall_at_10", "ndcg_at_10"):
            try:
                decision = _metric_decision(
                    selected,
                    metric=metric,
                    minimum_improvement=-GLOBAL_NONINFERIORITY_MARGIN,
                    seed_label=f"slice:{label}:{metric}",
                )
            except HierarchicalEvaluationError:
                failures.append(f"slice_insufficient:{label}:{metric}")
                continue
            slice_metrics.append(decision)
            if not decision.passed:
                failures.append(f"slice_regression:{label}:{metric}")
    if not slice_metrics:
        raise HierarchicalEvaluationError("quality gate produced no valid slices")

    route_values = [cast(float, case.route_recall_at_5) for case in answerable]
    evidence_values = [cast(float, case.evidence_recall_at_20) for case in answerable]
    absolutes = (
        AbsoluteDecision(
            metric="route_recall_at_5",
            eligible_case_count=len(route_values),
            value=math.fsum(route_values) / len(route_values),
            minimum=MIN_ROUTE_RECALL_AT_5,
            passed=math.fsum(route_values) / len(route_values) >= MIN_ROUTE_RECALL_AT_5,
        ),
        AbsoluteDecision(
            metric="evidence_recall_at_20",
            eligible_case_count=len(evidence_values),
            value=math.fsum(evidence_values) / len(evidence_values),
            minimum=MIN_EVIDENCE_RECALL_AT_20,
            passed=math.fsum(evidence_values) / len(evidence_values) >= MIN_EVIDENCE_RECALL_AT_20,
        ),
    )
    for absolute in absolutes:
        if not absolute.passed:
            failures.append(f"absolute:{absolute.metric}")

    snapshot_count = len(locked[0].snapshots)
    per_snapshot: list[StrictBool] = []
    for snapshot_index in range(snapshot_count):
        deltas = [
            (
                case.snapshots[snapshot_index].candidate.recall_at_10
                - case.snapshots[snapshot_index].baseline.recall_at_10,
                case.snapshots[snapshot_index].candidate.ndcg_at_10
                - case.snapshots[snapshot_index].baseline.ndcg_at_10,
            )
            for case in answerable
        ]
        passed = all(
            math.fsum(value[index] for value in deltas) / len(deltas) >= -GLOBAL_NONINFERIORITY_MARGIN
            for index in (0, 1)
        )
        per_snapshot.append(cast(StrictBool, passed))
        if not passed:
            failures.append(f"snapshot_noninferiority:{snapshot_index}")

    no_answer_cases = [case for case in locked if not case.answerable]
    if not no_answer_cases:
        raise HierarchicalEvaluationError("locked set has no no-answer cases")
    no_answer_rows = [
        _PairedValue(
            case.cluster_id,
            math.fsum(float(item.baseline.abstained) for item in case.snapshots) / len(case.snapshots),
            math.fsum(float(item.candidate.abstained) for item in case.snapshots) / len(case.snapshots),
        )
        for case in no_answer_cases
    ]
    baseline, candidate, improvement, ci_low, ci_high, clusters = _paired_bootstrap(
        no_answer_rows,
        seed=BOOTSTRAP_SEED ^ int.from_bytes(hashlib.sha256(b"no-answer").digest()[:8], "big"),
    )
    no_answer = MetricDecision(
        metric="no_answer:abstention",
        eligible_case_count=len(no_answer_rows),
        cluster_count=clusters,
        baseline=baseline,
        candidate=candidate,
        improvement=improvement,
        ci_low=ci_low,
        ci_high=ci_high,
        minimum_improvement=-GLOBAL_NONINFERIORITY_MARGIN,
        passed=ci_low >= -GLOBAL_NONINFERIORITY_MARGIN,
    )
    if not no_answer.passed:
        failures.append("no_answer_abstention_regression")
    security_cases = [
        case for case in no_answer_cases if set(case.challenge_tags) & {"leakage", "prompt_injection"}
    ]
    if not security_cases or any(
        not snapshot.candidate.abstained for case in security_cases for snapshot in case.snapshots
    ):
        failures.append("security_no_answer_false_positive")
    if sum(case.hierarchical_added for case in cases) == 0:
        failures.append("hierarchical_candidate_added_nothing")

    unique = tuple(dict.fromkeys(failures))
    return HierarchicalQualityDecision(
        global_metrics=tuple(global_metrics),
        slice_metrics=tuple(slice_metrics),
        absolutes=absolutes,
        per_snapshot_noninferiority=tuple(per_snapshot),
        no_answer=no_answer,
        failure_codes=unique,
        accepted=not unique,
    )


async def _collect_case(
    *,
    retriever: Retriever,
    sessionmaker: async_sessionmaker[AsyncSession],
    binding: Any,
    scope: Any,
) -> _CollectedCase:
    token = set_principal(scope.owner_sub, False)
    try:
        async with sessionmaker() as session, session.begin():
            trace = await retriever.retrieve_with_trace(
                session,
                binding.record.question,
                top_k=max(10, settings.rag_context_top_k),
                owner_sub=scope.owner_sub,
                allow_rerank_fallback=False,
                hierarchical_mode="shadow",
                allow_hierarchical_fallback=False,
            )
    finally:
        reset_principal(token)
    if trace.hierarchical_mode != "shadow" or trace.hierarchical_fallback or trace.reranker_fallback:
        raise HierarchicalEvaluationError("candidate collection used a fallback")
    baseline = trace.hybrid_pre_rerank
    candidate = trace.hierarchical_pre_rerank
    baseline_ids = [chunk.id for chunk in baseline]
    candidate_ids = [chunk.id for chunk in candidate]
    if (
        len(baseline_ids) != len(set(baseline_ids))
        or len(candidate_ids) != len(set(candidate_ids))
        or candidate_ids[: len(baseline_ids)] != baseline_ids
        or trace.hierarchical_added != len(candidate_ids) - len(baseline_ids)
    ):
        raise HierarchicalEvaluationError("hierarchical candidate union contract is invalid")
    if any(chunk.document_id not in scope.document_ids for chunk in candidate):
        raise HierarchicalEvaluationError("hierarchical candidate escaped the verified owner scope")
    hierarchy_manifest = [_chunk_manifest(chunk) for chunk in candidate]
    return _CollectedCase(
        binding=binding,
        scope=scope,
        baseline=baseline,
        candidate=candidate,
        hierarchy_manifest_sha256=_sha256_json(hierarchy_manifest),
        hierarchy_added=trace.hierarchical_added,
    )


async def _freeze_scores(
    collected: Sequence[_CollectedCase],
    *,
    snapshot_count: int,
) -> tuple[FrozenScoreSnapshot, ...]:
    if snapshot_count < MIN_SCORE_SNAPSHOTS:
        raise HierarchicalEvaluationError("at least three frozen score snapshots are required")
    reranker = Reranker()
    snapshots: list[FrozenScoreSnapshot] = []
    for snapshot_index in range(snapshot_count):
        started = time.perf_counter()
        entries: dict[tuple[str, str], FrozenScoreEntry] = {}
        for item in collected:
            question = item.binding.record.question
            question_sha256 = item.binding.record.question_sha256
            texts_by_hash: dict[str, str] = {}
            for chunk in item.candidate:
                value = _chunk_text(chunk)
                digest = _text_sha256(value)
                previous = texts_by_hash.setdefault(digest, value)
                if previous != value:
                    raise HierarchicalEvaluationError("reranker text hash collision")
            ordered = sorted(texts_by_hash.items())
            if not ordered:
                continue
            scores = await reranker.rerank(question, [value for _, value in ordered])
            if len(scores) != len(ordered) or not all(
                math.isfinite(score) and 0.0 <= score <= 1.0 for score in scores
            ):
                raise HierarchicalEvaluationError("live reranker returned invalid frozen scores")
            for (text_sha256, _), score in zip(ordered, scores, strict=True):
                key = (question_sha256, text_sha256)
                entry = FrozenScoreEntry(
                    question_sha256=question_sha256,
                    text_sha256=text_sha256,
                    score_hex=float(score).hex(),
                )
                previous_entry = entries.setdefault(key, entry)
                if previous_entry != entry:
                    raise HierarchicalEvaluationError("frozen score pair changed within one snapshot")
        ordered_entries = tuple(entries[key] for key in sorted(entries))
        snapshots.append(
            FrozenScoreSnapshot(
                snapshot_index=snapshot_index,
                pair_count=len(ordered_entries),
                entries=ordered_entries,
                pair_manifest_sha256=canonical_sha256(ordered_entries),
                scoring_seconds=time.perf_counter() - started,
            )
        )
    return tuple(snapshots)


def _case_result(
    item: _CollectedCase,
    *,
    snapshots: Sequence[FrozenScoreSnapshot],
    split: Literal["tuning", "locked"],
    cluster_id: str,
    query_embedding_sha256: str,
    top_k: int,
    minimum_score: float,
) -> HierarchicalCaseResult:
    record: GoldRecord = item.binding.record
    relevance: Mapping[uuid.UUID, int] = item.binding.relevance
    relevant_document_ids = {evidence.document_id for evidence in item.binding.sidecar.exact_evidence}
    snapshot_results: list[SnapshotCaseMetrics] = []
    evidence_recall_at_20: list[float] = []
    for snapshot in snapshots:
        scores = _score_map(snapshot)
        baseline = _rank_with_frozen_scores(
            item.baseline,
            question_sha256=record.question_sha256,
            scores=scores,
            top_k=top_k,
            minimum_score=minimum_score,
        )
        candidate = _rank_with_frozen_scores(
            item.candidate,
            question_sha256=record.question_sha256,
            scores=scores,
            top_k=top_k,
            minimum_score=minimum_score,
        )
        candidate_at_20 = _rank_with_frozen_scores(
            item.candidate,
            question_sha256=record.question_sha256,
            scores=scores,
            top_k=20,
            minimum_score=minimum_score,
        )
        if record.answerable:
            evidence_recall_at_20.append(_metric_recall(candidate_at_20, relevance, 20))
        snapshot_results.append(
            SnapshotCaseMetrics(
                snapshot_index=snapshot.snapshot_index,
                baseline=_variant_metrics(baseline, relevance, answerable=record.answerable),
                candidate=_variant_metrics(candidate, relevance, answerable=record.answerable),
                baseline_order_sha256=_order_sha256(baseline),
                candidate_order_sha256=_order_sha256(candidate),
            )
        )
    return HierarchicalCaseResult(
        case_id=record.case_id,
        gold_case_sha256=item.binding.sidecar.gold_case_sha256,
        question_sha256=record.question_sha256,
        query_embedding_sha256=query_embedding_sha256,
        scope_id=record.scope_id,
        cluster_id=cluster_id,
        split=split,
        language=record.language,
        hop_type=record.hop_type,
        content_types=tuple(record.content_types),
        challenge_tags=tuple(record.challenge_tags),
        answerable=record.answerable,
        relevant_count=len(relevance),
        route_recall_at_5=(
            _route_document_recall(item.baseline, relevant_document_ids, 5) if record.answerable else None
        ),
        evidence_recall_at_20=(
            math.fsum(evidence_recall_at_20) / len(evidence_recall_at_20)
            if evidence_recall_at_20
            else None
        ),
        baseline_pool_count=len(item.baseline),
        candidate_pool_count=len(item.candidate),
        hierarchical_added=item.hierarchy_added,
        hierarchy_manifest_sha256=item.hierarchy_manifest_sha256,
        scope_violation_count=0,
        hierarchical_fallback=False,
        snapshots=tuple(snapshot_results),
    )


def _request_id(
    *,
    case_id: str,
    variant: VariantName,
    pair_index: int,
    order_in_pair: Literal[0, 1],
) -> str:
    return _sha256_json(
        {
            "case_id": case_id,
            "variant": variant,
            "pair_index": pair_index,
            "order_in_pair": order_in_pair,
        }
    )


async def _load_request(
    *,
    retriever: Retriever,
    sessionmaker: async_sessionmaker[AsyncSession],
    semaphore: asyncio.Semaphore,
    item: _CollectedCase,
    variant: VariantName,
    pair_index: int,
    order_in_pair: Literal[0, 1],
    top_k: int,
) -> LoadRequestObservation:
    order_sha256: str | None = None
    returned_count = 0
    error_code: str | None = None
    async with semaphore:
        started_at = datetime.now(UTC)
        started = time.perf_counter()
        token = set_principal(item.scope.owner_sub, False)
        try:
            async with sessionmaker() as session, session.begin():
                trace = await retriever.retrieve_with_trace(
                    session,
                    item.binding.record.question,
                    owner_sub=item.scope.owner_sub,
                    top_k=top_k,
                    allow_rerank_fallback=False,
                    hierarchical_mode="off" if variant == "baseline" else "active",
                    allow_hierarchical_fallback=False,
                )
            if trace.reranker_fallback or trace.hierarchical_fallback:
                raise HierarchicalEvaluationError("load request used a fallback")
            expected_mode = "off" if variant == "baseline" else "active"
            if trace.hierarchical_mode != expected_mode:
                raise HierarchicalEvaluationError("load request changed hierarchical mode")
            if any(chunk.document_id not in item.scope.document_ids for chunk in trace.final):
                raise HierarchicalEvaluationError("load request escaped owner scope")
            returned_count = len(trace.final)
            order_sha256 = _order_sha256(trace.final)
        except Exception as error:  # noqa: BLE001 - only the sanitized type is persisted
            error_code = type(error).__name__
        finally:
            reset_principal(token)
        completed_at = datetime.now(UTC)
        latency_ms = (time.perf_counter() - started) * 1000.0
    return LoadRequestObservation(
        request_id=_request_id(
            case_id=item.binding.record.case_id,
            variant=variant,
            pair_index=pair_index,
            order_in_pair=order_in_pair,
        ),
        pair_index=pair_index,
        order_in_pair=order_in_pair,
        case_id=item.binding.record.case_id,
        variant=variant,
        started_at=started_at,
        completed_at=completed_at,
        latency_ms=latency_ms,
        returned_count=returned_count,
        order_sha256=order_sha256,
        success=error_code is None,
        error_code=error_code,
    )


def _observed_peak_concurrency(observations: Sequence[LoadRequestObservation]) -> int:
    events: dict[datetime, list[int]] = defaultdict(lambda: [0, 0])
    for item in observations:
        events[item.started_at][0] += 1
        events[item.completed_at][1] += 1
    active = 0
    peak = 0
    for timestamp in sorted(events):
        starts, completions = events[timestamp]
        active += starts
        peak = max(peak, active)
        active -= completions
    if active != 0:
        raise HierarchicalEvaluationError("load concurrency timeline is incomplete")
    return peak


def _nearest_rank(values: Sequence[float], probability: float) -> float:
    if not values:
        raise HierarchicalEvaluationError("load latency sample is empty")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def _load_aggregate(
    observations: Sequence[LoadRequestObservation],
    variant: VariantName,
) -> LoadAggregate:
    selected = [item for item in observations if item.variant == variant]
    completed = sum(item.success for item in selected)
    errors = len(selected) - completed
    duration = (
        max(item.completed_at for item in selected) - min(item.started_at for item in selected)
    ).total_seconds()
    if duration <= 0:
        raise HierarchicalEvaluationError("load duration is invalid")
    return LoadAggregate(
        request_count=len(selected),
        completed_count=completed,
        error_count=errors,
        p95_latency_ms=_nearest_rank([item.latency_ms for item in selected], 0.95),
        throughput_rps=completed / duration,
    )


async def generate_load_evidence(
    *,
    retriever: Retriever,
    sessionmaker: async_sessionmaker[AsyncSession],
    collected: Sequence[_CollectedCase],
    concurrency: int,
    requests_per_variant: int,
    warmups_per_variant: int,
    top_k: int,
) -> tuple[LoadEvidence, tuple[LoadRequestObservation, ...]]:
    if concurrency < 1 or requests_per_variant < 1 or top_k < 10 or not collected:
        raise HierarchicalEvaluationError("load configuration is invalid")
    semaphore = asyncio.Semaphore(concurrency)
    for index in range(warmups_per_variant):
        item = collected[index % len(collected)]
        for variant in ("baseline", "candidate"):
            observation = await _load_request(
                retriever=retriever,
                sessionmaker=sessionmaker,
                semaphore=semaphore,
                item=item,
                variant=variant,
                pair_index=index,
                order_in_pair=0,
                top_k=top_k,
            )
            if not observation.success:
                raise HierarchicalEvaluationError("load warmup failed")

    async def run_pair(pair_index: int) -> tuple[LoadRequestObservation, LoadRequestObservation]:
        item = collected[pair_index % len(collected)]
        variants: tuple[VariantName, VariantName] = (
            ("baseline", "candidate") if pair_index % 2 == 0 else ("candidate", "baseline")
        )
        rows = []
        for order_in_pair, variant in enumerate(variants):
            rows.append(
                await _load_request(
                    retriever=retriever,
                    sessionmaker=sessionmaker,
                    semaphore=semaphore,
                    item=item,
                    variant=variant,
                    pair_index=pair_index,
                    order_in_pair=cast(Literal[0, 1], order_in_pair),
                    top_k=top_k,
                )
            )
        return rows[0], rows[1]

    paired = await asyncio.gather(*(run_pair(index) for index in range(requests_per_variant)))
    observations = tuple(item for pair in paired for item in pair)
    peak = _observed_peak_concurrency(observations)
    baseline = _load_aggregate(observations, "baseline")
    candidate = _load_aggregate(observations, "candidate")
    maximum_p95 = min(
        baseline.p95_latency_ms * MAX_P95_RATIO,
        baseline.p95_latency_ms + MAX_P95_INCREASE_MS,
    )
    throughput_ratio = candidate.throughput_rps / baseline.throughput_rps
    passed = (
        peak == concurrency
        and baseline.error_count == candidate.error_count == 0
        and candidate.p95_latency_ms <= maximum_p95
        and throughput_ratio >= MIN_THROUGHPUT_RATIO
    )
    return (
        LoadEvidence(
            concurrency=concurrency,
            observed_peak_concurrency=peak,
            requests_per_variant=requests_per_variant,
            warmups_per_variant=warmups_per_variant,
            raw_observations_sha256=canonical_sha256(observations),
            baseline=baseline,
            candidate=candidate,
            maximum_candidate_p95_ms=maximum_p95,
            throughput_ratio=throughput_ratio,
            passed=passed,
        ),
        observations,
    )


async def _database_evidence(engine: AsyncEngine, image_digest: str) -> DatabaseEvidence:
    async with engine.connect() as connection:
        server_version = int(
            (await connection.execute(sql("SELECT current_setting('server_version_num')"))).scalar_one()
        )
        extension_rows = (
            await connection.execute(sql("SELECT extname, extversion FROM pg_extension ORDER BY extname"))
        ).all()
        index_rows = (
            await connection.execute(
                sql(
                    "SELECT indexname, indexdef FROM pg_indexes "
                    "WHERE schemaname='public' AND tablename='chunks' ORDER BY indexname"
                )
            )
        ).all()
    extensions = {str(row.extname): str(row.extversion) for row in extension_rows}
    return DatabaseEvidence(
        image_digest=image_digest,
        server_version_num=server_version,
        extensions=extensions,
        extension_manifest_sha256=_sha256_json(extensions),
        chunk_index_manifest_sha256=_sha256_json(
            [{"name": row.indexname, "definition": row.indexdef} for row in index_rows]
        ),
    )


def _load_rls_evidence(path: Path) -> tuple[RlsEvidence, str]:
    artifact = read_private_json(
        path,
        parser=lambda raw: RlsEvidence.model_validate_json(raw, strict=True),
    )
    return artifact.value, artifact.sha256


def _retrieval_config_sha256(*, top_k: int, minimum_score: float) -> str:
    return _sha256_json(
        {
            "dense_backend": settings.rag_dense_backend,
            "sparse_backend": settings.rag_sparse_backend,
            "dense_top_k": settings.rag_dense_top_k,
            "sparse_top_k": settings.rag_sparse_top_k,
            "rrf_k": settings.rag_rrf_k,
            "rerank_top_k": settings.rag_rerank_top_k,
            "rerank_min_score": minimum_score,
            "final_top_k": top_k,
            "hierarchical_anchor_top_k": settings.rag_hierarchical_anchor_top_k,
            "hierarchical_page_radius": settings.rag_hierarchical_page_radius,
            "hierarchical_per_anchor_k": settings.rag_hierarchical_per_anchor_k,
            "hierarchical_max_candidates": settings.rag_hierarchical_max_candidates,
        }
    )


def _assert_output_inside_work_dir(path: Path, work_dir: Path) -> None:
    try:
        path.parent.resolve(strict=True).relative_to(work_dir.resolve(strict=True))
    except (OSError, ValueError):
        raise HierarchicalEvaluationError("private outputs must stay inside the work directory") from None


async def run(args: argparse.Namespace) -> HierarchicalReport:
    if args.top_k < 10:
        raise HierarchicalEvaluationError("top_k must be at least 10 for Recall@10 qualification")
    mode = cast(RunMode, args.mode)
    work_dir = _private_dir(args.work_dir.expanduser())
    output = args.output.expanduser() if args.output else work_dir / "report.json"
    _assert_output_inside_work_dir(output, work_dir)
    if any(work_dir.iterdir()):
        raise HierarchicalEvaluationError("hierarchical evaluation requires a fresh empty work directory")

    gold_artifact = read_private_bytes(args.gold, max_bytes=256 * 1024 * 1024)
    sidecar_artifact = read_private_bytes(args.sidecar, max_bytes=256 * 1024 * 1024)
    records, _ = parse_gold_set_bytes(gold_artifact.raw_bytes, mode="release")
    sidecars = bind_gold_sidecar(records, parse_private_sidecar_bytes(sidecar_artifact.raw_bytes))
    bindings = build_case_bindings(records, sidecars)
    repository_sha, repository_dirty = _git_state()
    if mode == "qualification" and repository_dirty:
        raise HierarchicalEvaluationError("qualification requires a clean Git repository")
    if (
        len(records) != EXPECTED_RELEASE_CASES
        or sum(not item.answerable for item in records) != EXPECTED_NO_ANSWER_CASES
    ):
        raise HierarchicalEvaluationError("qualification Gold cardinality is invalid")
    if args.snapshots < MIN_SCORE_SNAPSHOTS:
        raise HierarchicalEvaluationError("qualification requires at least three score snapshots")
    if args.hmac_key is None:
        raise HierarchicalEvaluationError("hierarchical evaluation requires an HMAC key")
    key = load_hmac_key(args.hmac_key, REPOSITORY_ROOT)
    if args.rls_evidence is None:
        raise HierarchicalEvaluationError("hierarchical evaluation requires RLS evidence")
    rls, rls_artifact_sha256 = _load_rls_evidence(args.rls_evidence)
    if mode == "qualification" and (
        args.load_concurrency < MIN_LOAD_CONCURRENCY
        or args.load_requests_per_variant < MIN_LOAD_REQUESTS_PER_VARIANT
        or args.embedding_revision_evidence is None
        or args.reranker_revision_evidence is None
        or args.database_image_digest is None
    ):
        raise HierarchicalEvaluationError("qualification provenance or load configuration is incomplete")

    require_loopback_url(settings.embed_base_url, name="embedding endpoint")
    require_loopback_url(settings.rerank_base_url, name="reranker endpoint")
    require_loopback_endpoint(settings.s3_endpoint, name="MinIO endpoint")
    database_url = require_loopback_database_url(args.database_url)
    engine = create_async_engine(
        database_url,
        pool_pre_ping=True,
        connect_args={
            "server_settings": {
                "application_name": "rag_hierarchical_qualification",
                "default_transaction_read_only": "on",
            }
        },
    )
    embedder = Embedder()
    load_embedder: Embedder | None = None
    try:
        await assert_api_rls_role(engine, required=True)
        sessions = create_sessionmaker(engine)
        verifier = _CorpusVerifier(sessions, Storage())
        corpus_before, scopes = await verifier.verify(records, sidecars)
        if len(scopes) < 2:
            raise HierarchicalEvaluationError("qualification needs at least two owner scopes")
        split = stratified_cluster_split(records, seed=BOOTSTRAP_SEED, locked_fraction=0.85)
        if len(split.locked_case_ids) < MIN_LOCKED_CASES or len(split.tuning_case_ids) > MAX_TUNING_CASES:
            raise HierarchicalEvaluationError("qualification split cardinality is invalid")
        locked_ids = set(split.locked_case_ids)
        cluster_ids = _statistical_cluster_ids(bindings)

        paired_embedder = _PairedQueryEmbedder(embedder)
        await paired_embedder.preload([record.question for record in records])
        collector = Retriever(paired_embedder, _CollectionReranker())
        collected: list[_CollectedCase] = []
        for case_id in sorted(bindings):
            binding = bindings[case_id]
            collected.append(
                await _collect_case(
                    retriever=collector,
                    sessionmaker=sessions,
                    binding=binding,
                    scope=scopes[binding.record.scope_id],
                )
            )
        score_snapshots = await _freeze_scores(collected, snapshot_count=args.snapshots)
        scores_dir = _private_dir(work_dir / "scores")
        score_artifact_hashes: list[str] = []
        for snapshot in score_snapshots:
            _, artifact_hash = _write_signed_private_json(
                scores_dir / f"snapshot-{snapshot.snapshot_index}.json",
                artifact_type="hierarchical-frozen-scores-v1",
                payload=snapshot,
                key=key,
            )
            score_artifact_hashes.append(artifact_hash)

        cases = tuple(
            _case_result(
                item,
                snapshots=score_snapshots,
                split="locked" if item.binding.record.case_id in locked_ids else "tuning",
                cluster_id=cluster_ids[item.binding.record.case_id],
                query_embedding_sha256=paired_embedder.vector_sha256(item.binding.record.question),
                top_k=args.top_k,
                minimum_score=args.rerank_min_score,
            )
            for item in collected
        )
        quality = evaluate_quality(cases)

        load_embedder = Embedder()
        load_retriever = Retriever(load_embedder, Reranker())
        locked_collected = [item for item in collected if item.binding.record.case_id in locked_ids]
        load, raw_load = await generate_load_evidence(
            retriever=load_retriever,
            sessionmaker=sessions,
            collected=locked_collected,
            concurrency=args.load_concurrency,
            requests_per_variant=args.load_requests_per_variant,
            warmups_per_variant=args.load_warmups_per_variant,
            top_k=args.top_k,
        )
        _, load_artifact_sha256 = _write_signed_private_json(
            work_dir / "load.raw.json",
            artifact_type="hierarchical-load-raw-v1",
            payload=[item.model_dump(mode="json") for item in raw_load],
            key=key,
        )
        corpus_after, scopes_after = await verifier.verify(records, sidecars)
        if corpus_after != corpus_before or {key: value.evidence for key, value in scopes_after.items()} != {
            key: value.evidence for key, value in scopes.items()
        }:
            raise HierarchicalEvaluationError("runtime corpus changed during evaluation")

        if args.embedding_revision_evidence is None or args.reranker_revision_evidence is None:
            raise HierarchicalEvaluationError("model revision evidence is required")
        embedding_revision, embedding_artifact_sha256 = _load_model_revision(
            args.embedding_revision_evidence,
            expected_model=settings.embed_model,
        )
        reranker_revision, reranker_artifact_sha256 = _load_model_revision(
            args.reranker_revision_evidence,
            expected_model=settings.rerank_model,
        )
        models = ModelEvidence(
            embedding=embedding_revision,
            reranker=reranker_revision,
            embedding_artifact_sha256=embedding_artifact_sha256,
            reranker_artifact_sha256=reranker_artifact_sha256,
        )
        if args.database_image_digest is None:
            raise HierarchicalEvaluationError("database image digest is required")
        database = await _database_evidence(engine, args.database_image_digest)
        hierarchy_manifest_sha256 = _sha256_json(
            [
                {"case_id": item.binding.record.case_id, "sha256": item.hierarchy_manifest_sha256}
                for item in collected
            ]
        )
        query_vector_manifest_sha256 = _sha256_json(
            [
                {
                    "case_id": item.case_id,
                    "question_sha256": item.question_sha256,
                    "query_embedding_sha256": item.query_embedding_sha256,
                }
                for item in cases
            ]
        )
        retrieval_config_sha256 = _retrieval_config_sha256(
            top_k=args.top_k,
            minimum_score=args.rerank_min_score,
        )
        split_manifest_sha256 = _sha256_json(split.model_dump(mode="json"))
        score_snapshot_manifest_sha256 = _sha256_json(score_artifact_hashes)
        run_id = _sha256_json(
            {
                "repository_sha": repository_sha,
                "gold_sha256": gold_artifact.sha256,
                "sidecar_sha256": sidecar_artifact.sha256,
                "corpus_sha256": corpus_before,
                "hierarchy_manifest_sha256": hierarchy_manifest_sha256,
                "query_vector_manifest_sha256": query_vector_manifest_sha256,
                "score_snapshot_manifest_sha256": score_snapshot_manifest_sha256,
                "retrieval_config_sha256": retrieval_config_sha256,
                "rls_artifact_sha256": rls_artifact_sha256,
                "load_artifact_sha256": load_artifact_sha256,
            }
        )
        report = HierarchicalReport(
            run_id=run_id,
            mode=mode,
            evaluated_at=datetime.now(UTC),
            provenance=HierarchicalProvenance(
                repository_sha=repository_sha,
                repository_dirty=repository_dirty,
                gold_sha256=gold_artifact.sha256,
                sidecar_sha256=sidecar_artifact.sha256,
                case_count=EXPECTED_RELEASE_CASES,
                no_answer_case_count=EXPECTED_NO_ANSWER_CASES,
                split_manifest_sha256=split_manifest_sha256,
                runtime_corpus_sha256_before=corpus_before,
                runtime_corpus_sha256_after=corpus_after,
                hierarchy_manifest_sha256=hierarchy_manifest_sha256,
                query_vector_manifest_sha256=query_vector_manifest_sha256,
                score_snapshot_manifest_sha256=score_snapshot_manifest_sha256,
                score_snapshot_count=len(score_snapshots),
                retrieval_config_sha256=retrieval_config_sha256,
                database=database,
                models=models,
                rls_artifact_sha256=rls_artifact_sha256,
                load_artifact_sha256=load_artifact_sha256,
            ),
            cases=tuple(sorted(cases, key=lambda item: item.case_id)),
            score_snapshots=score_snapshots,
            rls=rls,
            load=load,
            quality=quality,
            release_accepted=(
                mode == "qualification" and quality.accepted and rls.passed and load.passed
            ),
        )
        report_bytes, _ = _write_signed_private_json(
            output,
            artifact_type=SCHEMA_VERSION,
            payload=report,
            key=key,
        )
        if mode == "qualification":
            attestation = create_private_artifact_attestation(
                artifact_bytes=report_bytes,
                artifact_type=SCHEMA_VERSION,
                key=key,
                repository_root=REPOSITORY_ROOT,
                source_paths=_ATTESTED_SOURCES,
            )
            atomic_write_private_artifact_attestation(
                output.with_name(f"{output.stem}.attestation.json"),
                attestation,
            )
        return report
    finally:
        await embedder.client.close()
        if load_embedder is not None:
            await load_embedder.client.close()
        await engine.dispose()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=("debug", "qualification"), default="debug")
    parser.add_argument("--database-url", default=settings.database_url)
    parser.add_argument("--database-image-digest")
    parser.add_argument("--embedding-revision-evidence", type=Path)
    parser.add_argument("--reranker-revision-evidence", type=Path)
    parser.add_argument("--rls-evidence", type=Path)
    parser.add_argument("--hmac-key", type=Path)
    parser.add_argument("--snapshots", type=int, default=MIN_SCORE_SNAPSHOTS)
    parser.add_argument("--top-k", type=int, default=max(10, settings.rag_context_top_k))
    parser.add_argument("--rerank-min-score", type=float, default=settings.rag_rerank_min_score)
    parser.add_argument("--load-concurrency", type=int, default=MIN_LOAD_CONCURRENCY)
    parser.add_argument(
        "--load-requests-per-variant",
        type=int,
        default=MIN_LOAD_REQUESTS_PER_VARIANT,
    )
    parser.add_argument("--load-warmups-per-variant", type=int, default=20)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        report = asyncio.run(run(args))
    except (HierarchicalEvaluationError, RuntimeError, ValueError) as error:
        raise SystemExit(f"hierarchical retrieval evaluation failed: {error}") from None
    print(
        json.dumps(
            {
                "run_id": report.run_id,
                "release_accepted": report.release_accepted,
                "failure_codes": report.quality.failure_codes,
            },
            sort_keys=True,
        )
    )
    return 0 if report.mode == "debug" or report.release_accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
