#!/usr/bin/env python3
"""Paired retrieval-only A/B for PostgreSQL FTS versus pg_textsearch.

The runner consumes the reviewed release Gold and its private sidecar. Runtime
artifacts contain only stable identifiers, labels, hashes, rankings and metrics;
questions, excerpts and document text are never serialized.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import itertools
import json
import math
import os
import random
import re
import stat
import subprocess
import time
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, cast

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
from rag_app.eval.gold_set import (
    DocumentSnapshot,
    GoldRecord,
    bytes_sha256,
    make_document_ref,
    make_scope_id,
    parse_gold_set_bytes,
    parsed_chunks_sha256,
    text_sha256,
)
from rag_app.eval.private_artifacts import (
    PrivateArtifactError,
    PrivateArtifactFormatError,
    read_private_bytes,
    read_private_json,
    write_private_json_fresh,
)
from rag_app.eval.private_sidecar import (
    PrivateSidecarRecord,
    RetrievalProbe,
    bind_gold_sidecar,
    parse_private_sidecar_bytes,
)
from rag_app.eval.rag_metrics import mrr_at_k, ndcg_at_k, recall_at_k
from rag_app.eval.report_attestation import (
    atomic_write_private_artifact_attestation,
    create_private_artifact_attestation,
    load_hmac_key,
    load_private_artifact_attestation,
    verify_private_artifact_attestation,
)
from rag_app.eval.retrieval_gate import (
    LoadEvidence,
    MetricDecision,
    OperationalEvidence,
    OwnerScopeProvenance,
    PgTextsearchProvenance,
    RetrievalCaseMetrics,
    RetrievalCaseResult,
    RetrievalConfiguration,
    RetrievalGateDecision,
    RetrievalGatePolicy,
    RetrievalModelRevisions,
    RetrievalProvenance,
    RetrievalReport,
    RlsPrincipalEvidence,
    RuntimeModelRevision,
    SparseEngineProvenance,
    SparseIndexDefinition,
    canonical_sha256,
    evaluate_retrieval_gate,
)
from rag_app.llm.embeddings import Embedder, Reranker
from rag_app.rag.retrieve import (
    RetrievalTrace,
    Retriever,
    SparseBackend,
    SparseEngine,
    sparse_query_plan,
)
from rag_app.storage.s3 import Storage

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_GIT_SHA_PATTERN = r"^[0-9a-f]{40,64}$"
_CASE_ID_PATTERN = r"^ragq-[a-z0-9][a-z0-9._-]{7,63}$"
_CONTAINER_NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$"
_EVIDENCE_KINDS = ("rls", "load", "update", "delete", "restart")
_CASE_HMAC_DOMAIN = b"docragenslate/retrieval-bm25-case/v2\0"
_BUILD_RECIPE = REPOSITORY_ROOT / "deploy/postgres-bm25/Dockerfile"
_PREPARE_SQL = REPOSITORY_ROOT / "deploy/postgres-bm25/prepare_candidate.sql"
_MAX_RERANK_REPEAT_DELTA = 0.01
_SCORE_EPSILON = 1e-12
_QUERY_EMBEDDING_PROTOCOL: Literal["single-live-vector-per-question-v1"] = (
    "single-live-vector-per-question-v1"
)
_LOAD_EMBEDDING_PROTOCOL: Literal["live-per-request-v1"] = "live-per-request-v1"

VariantName = Literal["baseline", "candidate"]
SplitName = Literal["tuning", "locked"]
PoolName = Literal["dense", "sparse", "hybrid", "final"]
RunMode = Literal["dev", "qualification"]


class RetrievalEvaluationError(RuntimeError):
    """The paired evaluation cannot produce trustworthy evidence."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class RetrievalConfig(_StrictModel):
    dense_top_k: int = Field(ge=10, le=1000)
    sparse_top_k: int = Field(ge=10, le=1000)
    rrf_k: int = Field(ge=1, le=10_000)
    rerank_top_k: int = Field(ge=10, le=1000)
    rerank_min_score: float = Field(ge=0.0, le=1.0)
    final_top_k: int = Field(default=10, ge=10, le=100)

    @model_validator(mode="after")
    def validate_cutoffs(self) -> RetrievalConfig:
        if self.rerank_top_k > self.dense_top_k + self.sparse_top_k:
            raise ValueError("rerank_top_k exceeds the maximum hybrid pool")
        if self.final_top_k > self.rerank_top_k:
            raise ValueError("final_top_k must not exceed rerank_top_k")
        return self

    @property
    def fingerprint(self) -> str:
        return _sha256_json(self.model_dump(mode="json"))


class RankedMetrics(_StrictModel):
    recall: dict[str, float | None]
    mrr: dict[str, float | None]
    ndcg: dict[str, float | None]
    no_answer_false_positive: bool | None


class PoolObservation(_StrictModel):
    ranked_chunk_ids: tuple[uuid.UUID, ...]
    order_sha256: str = Field(pattern=_SHA256_PATTERN)
    latency_ms: float = Field(ge=0.0)
    metrics: RankedMetrics


class VariantObservation(_StrictModel):
    variant: VariantName
    requested_sparse_backend: Literal["postgres_fts", "pg_textsearch"]
    sparse_engine: str = Field(min_length=1, max_length=64)
    config_sha256: str = Field(pattern=_SHA256_PATTERN)
    pools: dict[PoolName, PoolObservation]
    repeat_order_sha256: tuple[str, ...] = Field(min_length=2, max_length=20)
    deterministic: StrictBool
    reranker_consensus_applied: StrictBool
    reranker_max_score_delta: float = Field(ge=0.0)
    reranker_all_max_score_delta: float = Field(ge=0.0)
    reranker_fallback: StrictBool
    returned_count: int = Field(ge=0, le=1000)
    abstained: StrictBool

    @model_validator(mode="after")
    def validate_abstention(self) -> VariantObservation:
        if self.abstained != (self.returned_count == 0):
            raise ValueError("abstention must match returned_count")
        hashes_stable = len(set(self.repeat_order_sha256)) == 1
        if self.reranker_consensus_applied:
            if (
                hashes_stable
                or self.reranker_max_score_delta > _MAX_RERANK_REPEAT_DELTA + _SCORE_EPSILON
            ):
                raise ValueError("reranker consensus evidence is inconsistent")
        elif not hashes_stable:
            raise ValueError("non-consensus repeat ordering must be identical")
        if self.reranker_max_score_delta > self.reranker_all_max_score_delta:
            raise ValueError("output reranker delta exceeds the full candidate delta")
        return self


class CaseArtifact(_StrictModel):
    schema_version: Literal["retrieval-bm25-case-v2"] = "retrieval-bm25-case-v2"
    run_id: str = Field(pattern=_SHA256_PATTERN)
    split: SplitName
    case_id: str = Field(pattern=_CASE_ID_PATTERN)
    gold_case_sha256: str = Field(pattern=_SHA256_PATTERN)
    question_sha256: str = Field(pattern=_SHA256_PATTERN)
    query_embedding_protocol: Literal["single-live-vector-per-question-v1"]
    query_embedding_sha256: str = Field(pattern=_SHA256_PATTERN)
    scope_id: str
    language: Literal["ru", "en", "zh"]
    hop_type: Literal["single", "multi", "cross_document"]
    content_types: tuple[str, ...]
    challenge_tags: tuple[str, ...]
    answerable: StrictBool
    relevant_chunk_ids: tuple[uuid.UUID, ...]
    relevance_grades: dict[str, int]
    observation: VariantObservation


class CaseArtifactHmac(_StrictModel):
    schema_version: Literal["retrieval-bm25-case-hmac-v2"] = "retrieval-bm25-case-hmac-v2"
    key_id: str = Field(pattern=_SHA256_PATTERN)
    artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    run_id: str = Field(pattern=_SHA256_PATTERN)
    config_sha256: str = Field(pattern=_SHA256_PATTERN)
    case_id: str = Field(pattern=_CASE_ID_PATTERN)
    signature: str = Field(pattern=_SHA256_PATTERN)


class SplitManifest(_StrictModel):
    schema_version: Literal["retrieval-split-v1"] = "retrieval-split-v1"
    seed: int
    locked_fraction: float = Field(gt=0.0, lt=1.0)
    tuning_case_ids: tuple[str, ...] = Field(min_length=1)
    locked_case_ids: tuple[str, ...] = Field(min_length=1)
    tuning_cluster_ids: tuple[str, ...] = Field(min_length=1)
    locked_cluster_ids: tuple[str, ...] = Field(min_length=1)
    distribution: dict[str, dict[str, int]]
    manifest_sha256: str = Field(pattern=_SHA256_PATTERN)


class EvidenceReference(_StrictModel):
    kind: Literal["rls", "load", "update", "delete", "restart", "operational"]
    schema_version: str = Field(min_length=1, max_length=128)
    artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    corpus_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    passed: Literal[True]


class _ExternalEvidenceBase(_StrictModel):
    schema_version: str = Field(min_length=1, max_length=128)
    kind: Literal["rls", "load", "update", "delete", "restart"]
    passed: Literal[True]
    corpus_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)


class _RlsEvidence(_ExternalEvidenceBase):
    schema_version: Literal["retrieval-rls-evidence-v1"]
    kind: Literal["rls"]
    principals: tuple[RlsPrincipalEvidence, ...] = Field(min_length=1, max_length=100)


class _LoadEvidence(_ExternalEvidenceBase):
    schema_version: Literal["retrieval-load-evidence-v2"]
    kind: Literal["load"]
    config_sha256: str = Field(pattern=_SHA256_PATTERN)
    locked_case_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    raw_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    embedding_protocol: Literal["live-per-request-v1"]
    baseline: LoadEvidence
    candidate: LoadEvidence


class LoadRequestObservation(_StrictModel):
    request_id: str = Field(pattern=_SHA256_PATTERN)
    pair_index: int = Field(ge=0)
    order_in_pair: Literal[0, 1]
    case_id: str = Field(pattern=_CASE_ID_PATTERN)
    backend: Literal["postgres_fts", "pg_textsearch"]
    sparse_engine: str | None = Field(default=None, max_length=64)
    started_at: datetime
    completed_at: datetime
    latency_ms: float = Field(ge=0)
    returned_count: int = Field(ge=0, le=1000)
    order_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    success: StrictBool
    error_code: str | None = Field(default=None, pattern=r"^[A-Za-z][A-Za-z0-9_]{0,127}$")

    @model_validator(mode="after")
    def validate_outcome(self) -> LoadRequestObservation:
        if self.started_at.tzinfo is None or self.completed_at.tzinfo is None:
            raise ValueError("load timestamps must be timezone-aware")
        if self.completed_at < self.started_at:
            raise ValueError("load completion predates start")
        if self.success != (self.error_code is None):
            raise ValueError("load success and error code are inconsistent")
        if self.success != (self.order_sha256 is not None):
            raise ValueError("successful load observation requires an order hash")
        return self


class RawLoadEvidence(_StrictModel):
    schema_version: Literal["retrieval-load-raw-v2"] = "retrieval-load-raw-v2"
    corpus_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    config_sha256: str = Field(pattern=_SHA256_PATTERN)
    locked_case_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    embedding_protocol: Literal["live-per-request-v1"]
    concurrency: int = Field(ge=1, le=1000)
    requests_per_backend: int = Field(ge=1, le=100_000)
    started_at: datetime
    completed_at: datetime
    observations: tuple[LoadRequestObservation, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_observations(self) -> RawLoadEvidence:
        if self.started_at.tzinfo is None or self.completed_at.tzinfo is None:
            raise ValueError("raw load timestamps must be timezone-aware")
        expected = self.requests_per_backend * 2
        if len(self.observations) != expected:
            raise ValueError("raw load observation count is incomplete")
        request_ids = [item.request_id for item in self.observations]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("raw load request IDs must be unique")
        for backend in ("postgres_fts", "pg_textsearch"):
            if sum(item.backend == backend for item in self.observations) != self.requests_per_backend:
                raise ValueError("raw load backend count is incomplete")
        pairs: dict[int, list[LoadRequestObservation]] = defaultdict(list)
        for item in self.observations:
            pairs[item.pair_index].append(item)
        if set(pairs) != set(range(self.requests_per_backend)):
            raise ValueError("raw load pair indexes are incomplete")
        for pair_index, pair in pairs.items():
            ordered = sorted(pair, key=lambda item: item.order_in_pair)
            expected_backends = (
                ("postgres_fts", "pg_textsearch")
                if pair_index % 2 == 0
                else ("pg_textsearch", "postgres_fts")
            )
            if (
                len(ordered) != 2
                or tuple(item.order_in_pair for item in ordered) != (0, 1)
                or tuple(item.backend for item in ordered) != expected_backends
                or ordered[1].started_at < ordered[0].completed_at
            ):
                raise ValueError("raw load pair order is invalid")
        return self


class _UpdateEvidence(_ExternalEvidenceBase):
    schema_version: Literal["retrieval-update-evidence-v1"]
    kind: Literal["update"]
    visible: Literal[True]
    visibility_seconds: float = Field(ge=0)
    raw_evidence_sha256: str = Field(pattern=_SHA256_PATTERN)


class _DeleteEvidence(_ExternalEvidenceBase):
    schema_version: Literal["retrieval-delete-evidence-v1"]
    kind: Literal["delete"]
    hidden: Literal[True]
    visibility_seconds: float = Field(ge=0)
    raw_evidence_sha256: str = Field(pattern=_SHA256_PATTERN)


class _RestartEvidence(_ExternalEvidenceBase):
    schema_version: Literal["retrieval-restart-evidence-v1"]
    kind: Literal["restart"]
    recovered: Literal[True]
    recovery_seconds: float = Field(ge=0)
    rollback_succeeded: Literal[True]
    rollback_seconds: float = Field(ge=0)
    rollback_backend: Literal["postgres_fts"]
    rollback_index_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    candidate_image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    candidate_index_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    raw_evidence_sha256: str = Field(pattern=_SHA256_PATTERN)


class ModelRevisionEvidence(_StrictModel):
    embedding: ModelEndpointRevision
    reranker: ModelEndpointRevision
    embedding_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    reranker_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)


class ModelEndpointRevision(_StrictModel):
    model: str = Field(min_length=1, max_length=256)
    declared_revision: str = Field(min_length=1, max_length=256)
    endpoint_metadata_sha256: str = Field(pattern=_SHA256_PATTERN)
    runtime_version_sha256: str = Field(pattern=_SHA256_PATTERN)
    weight_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    model_config_sha256: str = Field(pattern=_SHA256_PATTERN)


class DatabaseEvidence(_StrictModel):
    image_ref: str = Field(min_length=1, max_length=512)
    image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    server_version_num: int = Field(ge=170_000, lt=180_000)
    extensions: dict[str, str]
    index_definitions: dict[str, str]
    index_definitions_sha256: dict[str, str]
    extension_binary_sha256: str = Field(pattern=_SHA256_PATTERN)
    extension_binary_bytes: int = Field(ge=1)
    extension_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    extension_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    package_sha256: str = Field(pattern=_SHA256_PATTERN)
    base_image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    build_recipe_sha256: str = Field(pattern=_SHA256_PATTERN)
    prepare_sql_sha256: str = Field(pattern=_SHA256_PATTERN)
    baseline_index_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    candidate_index_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)


class SparseEngineCaseEvidence(_StrictModel):
    case_id: str = Field(pattern=_CASE_ID_PATTERN)
    split: SplitName
    variant: VariantName
    config_sha256: str = Field(pattern=_SHA256_PATTERN)
    sparse_engine: Literal["postgres_fts", "pg_textsearch_ru", "pg_textsearch_en"]
    artifact_sha256: str = Field(pattern=_SHA256_PATTERN)


class ScopeEvidence(_StrictModel):
    scope_id: str = Field(pattern=r"^scope-sha256:[0-9a-f]{64}$")
    case_count: int = Field(ge=1, le=500)
    document_count: int = Field(ge=1)
    chunk_count: int = Field(ge=1)
    corpus_sha256: str = Field(pattern=_SHA256_PATTERN)


class QueryEmbeddingEvidence(_StrictModel):
    protocol: Literal["single-live-vector-per-question-v1"]
    cache_scope: Literal["run"]
    reuse_scope: Literal["tuning+locked+variants+repeats"]
    preloaded: Literal[True]
    unique_question_count: int = Field(ge=1, le=500)
    live_call_count: int = Field(ge=1, le=500)
    vector_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    config_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_live_calls(self) -> QueryEmbeddingEvidence:
        if self.live_call_count != self.unique_question_count:
            raise ValueError("query embedding live-call count must equal unique questions")
        return self


class RunManifest(_StrictModel):
    schema_version: Literal["retrieval-bm25-run-v2"] = "retrieval-bm25-run-v2"
    run_id: str = Field(pattern=_SHA256_PATTERN)
    mode: RunMode
    repository_sha: str = Field(pattern=_GIT_SHA_PATTERN)
    repository_dirty: StrictBool
    gold_sha256: str = Field(pattern=_SHA256_PATTERN)
    sidecar_sha256: str = Field(pattern=_SHA256_PATTERN)
    gold_corpus_sha256: str = Field(pattern=_SHA256_PATTERN)
    runtime_corpus_sha256: str = Field(pattern=_SHA256_PATTERN)
    policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    owner_scopes: tuple[ScopeEvidence, ...] = Field(min_length=2, max_length=100)
    split: SplitManifest
    control_config: RetrievalConfig
    sweep_configs: tuple[RetrievalConfig, ...] = Field(min_length=1, max_length=10_000)
    model_revisions: ModelRevisionEvidence
    database: DatabaseEvidence
    external_evidence: tuple[EvidenceReference, ...]
    query_embedding: QueryEmbeddingEvidence
    load_embedding_protocol: Literal["live-per-request-v1"]
    repeat_count: int = Field(ge=2, le=20)
    created_at: datetime


class AggregateMetrics(_StrictModel):
    answerable_cases: int = Field(ge=0)
    no_answer_cases: int = Field(ge=0)
    recall: dict[str, float | None]
    mrr: dict[str, float | None]
    ndcg: dict[str, float | None]
    no_answer_false_positive_rate: float | None
    returned_count: int = Field(ge=0)
    abstained_count: int = Field(ge=0)
    no_answer_returned_count: int = Field(ge=0)
    no_answer_abstained_count: int = Field(ge=0)
    latency_ms: dict[str, float]


class FinalReport(_StrictModel):
    schema_version: Literal["retrieval-bm25-report-v2"] = "retrieval-bm25-report-v2"
    run_id: str = Field(pattern=_SHA256_PATTERN)
    manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    selected_candidate_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    query_embedding: QueryEmbeddingEvidence
    load_embedding_protocol: Literal["live-per-request-v1"]
    tuning_results: dict[str, dict[PoolName, AggregateMetrics]]
    locked_results: dict[VariantName, dict[PoolName, AggregateMetrics]]
    locked_slices: dict[VariantName, dict[str, dict[PoolName, AggregateMetrics]]]
    baseline_gate_report_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    candidate_gate_report_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    gate_decision: RetrievalGateDecision | None = None
    locked_decision: LockedDecision
    release_accepted: StrictBool | None = None
    deterministic: Literal[True]
    sparse_engine_evidence: tuple[SparseEngineCaseEvidence, ...]
    runtime_corpus_sha256_after: str = Field(pattern=_SHA256_PATTERN)
    completed_at: datetime


class LockedNoAnswerDecision(_StrictModel):
    eligible_case_count: int = Field(ge=1, le=500)
    cluster_count: int = Field(ge=2)
    baseline_abstention_rate: float = Field(ge=0, le=1)
    candidate_abstention_rate: float = Field(ge=0, le=1)
    improvement: float = Field(ge=-1, le=1)
    ci_low: float = Field(ge=-1, le=1)
    ci_high: float = Field(ge=-1, le=1)
    noninferiority_margin: float = Field(ge=0, le=1)
    passed: StrictBool


class LockedDecision(_StrictModel):
    schema_version: Literal["retrieval-locked-decision-v1"] = "retrieval-locked-decision-v1"
    case_count: int = Field(ge=1, le=500)
    tuning_case_count: int = Field(ge=1, le=500)
    metrics: tuple[MetricDecision, ...] = Field(min_length=7, max_length=7)
    no_answer: LockedNoAnswerDecision
    accepted: StrictBool
    failure_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _CaseBinding:
    record: GoldRecord
    sidecar: PrivateSidecarRecord
    relevance: dict[uuid.UUID, int]


@dataclass(frozen=True, slots=True)
class _Cluster:
    cluster_id: str
    case_ids: tuple[str, ...]
    labels: frozenset[str]


@dataclass(frozen=True, slots=True)
class _VerifiedScope:
    owner_sub: str
    document_refs: dict[uuid.UUID, str]
    document_ids: frozenset[uuid.UUID]
    evidence: ScopeEvidence


class TraceRetriever(Protocol):
    async def retrieve_with_trace(
        self,
        session: AsyncSession,
        query: str,
        document_id: uuid.UUID | None = None,
        folder_id: uuid.UUID | None = None,
        top_k: int | None = None,
        document_ids: list[uuid.UUID] | None = None,
        owner_sub: str | None = None,
        allow_rerank_fallback: bool = True,
        sparse_backend: SparseBackend | None = None,
        *,
        dense_top_k: int | None = None,
        sparse_top_k: int | None = None,
        rrf_k: int | None = None,
        rerank_top_k: int | None = None,
        rerank_min_score: float | None = None,
    ) -> RetrievalTrace: ...


class _PairedQueryEmbedder(Embedder):
    """Pin one live vector per question so sparse A/B shares an exact dense input."""

    def __init__(self, delegate: Embedder) -> None:
        self._delegate = delegate
        self._query_vectors: dict[str, tuple[float, ...]] = {}
        self._allowed_questions: frozenset[str] | None = None
        self._live_call_count = 0

    async def embed(self, texts: list[str], batch: int | None = None) -> list[list[float]]:
        return await self._delegate.embed(texts, batch)

    async def embed_query(self, query: str) -> list[float]:
        cached = self._query_vectors.get(query)
        if cached is None:
            if self._allowed_questions is not None and query not in self._allowed_questions:
                raise RetrievalEvaluationError("query embedding cache received an unknown question")
            vector = await self._delegate.embed_query(query)
            if (
                len(vector) != settings.embed_dim
                or not all(math.isfinite(value) for value in vector)
                or math.fsum(value * value for value in vector) <= 0.0
            ):
                raise RetrievalEvaluationError("query embedding is invalid")
            cached = tuple(float(value) for value in vector)
            self._query_vectors[query] = cached
            self._live_call_count += 1
        return list(cached)

    async def preload(self, questions: Sequence[str]) -> None:
        if self._allowed_questions is not None or self._query_vectors:
            raise RetrievalEvaluationError("query embedding cache was already initialized")
        unique = frozenset(questions)
        if not unique:
            raise RetrievalEvaluationError("query embedding preload set is empty")
        self._allowed_questions = unique
        for question in sorted(unique):
            await self.embed_query(question)
        if self._live_call_count != len(unique):
            raise RetrievalEvaluationError("query embedding preload count is inconsistent")

    def vector_sha256(self, query: str) -> str:
        vector = self._query_vectors.get(query)
        if vector is None:
            raise RetrievalEvaluationError("query embedding was not preloaded")
        return _sha256_json(list(vector))

    def evidence(
        self,
        records: Sequence[GoldRecord],
        revision: ModelEndpointRevision,
    ) -> QueryEmbeddingEvidence:
        if self._allowed_questions is None or set(self._query_vectors) != set(self._allowed_questions):
            raise RetrievalEvaluationError("query embedding preload is incomplete")
        vector_rows = [
            {
                "case_id": record.case_id,
                "question_sha256": record.question_sha256,
                "query_embedding_sha256": self.vector_sha256(record.question),
            }
            for record in sorted(records, key=lambda item: item.case_id)
        ]
        config_sha256 = _sha256_json(
            {
                "model": settings.embed_model,
                "declared_revision": revision.declared_revision,
                "model_config_sha256": revision.model_config_sha256,
                "instruction": settings.embed_query_instruction,
                "input_truncation_chars": 8000,
                "embed_dim": settings.embed_dim,
            }
        )
        return QueryEmbeddingEvidence(
            protocol=_QUERY_EMBEDDING_PROTOCOL,
            cache_scope="run",
            reuse_scope="tuning+locked+variants+repeats",
            preloaded=True,
            unique_question_count=len(self._allowed_questions),
            live_call_count=self._live_call_count,
            vector_manifest_sha256=_sha256_json(vector_rows),
            config_sha256=config_sha256,
        )


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256_order(values: Sequence[uuid.UUID]) -> str:
    return _sha256_json([str(value) for value in values])


def _nearest_rank(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(math.ceil(quantile * len(ordered)) - 1, len(ordered) - 1))
    return ordered[index]


def _case_labels(record: GoldRecord) -> frozenset[str]:
    labels = {
        f"language:{record.language}",
        f"hop:{record.hop_type}",
        f"answerable:{str(record.answerable).lower()}",
    }
    labels.update(f"content:{value}" for value in record.content_types)
    labels.update(f"challenge:{value}" for value in record.challenge_tags)
    if not record.challenge_tags:
        labels.add("challenge:none")
    return frozenset(labels)


def _retrieval_no_answer_count(records: Sequence[GoldRecord]) -> int:
    return sum(not record.answerable for record in records)


def _union_find_clusters(records: Sequence[GoldRecord]) -> list[_Cluster]:
    parent = {record.case_id: record.case_id for record in records}

    def root(case_id: str) -> str:
        while parent[case_id] != case_id:
            parent[case_id] = parent[parent[case_id]]
            case_id = parent[case_id]
        return case_id

    def union(left: str, right: str) -> None:
        left_root, right_root = root(left), root(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    documents: dict[str, str] = {}
    by_id = {record.case_id: record for record in records}
    for record in records:
        for snapshot in record.document_scope:
            previous = documents.setdefault(snapshot.document_ref, record.case_id)
            union(previous, record.case_id)

    grouped: dict[str, list[str]] = defaultdict(list)
    for case_id in sorted(parent):
        grouped[root(case_id)].append(case_id)
    clusters = []
    for case_ids in grouped.values():
        labels = frozenset(label for case_id in case_ids for label in _case_labels(by_id[case_id]))
        cluster_id = _sha256_json(case_ids)
        clusters.append(_Cluster(cluster_id, tuple(case_ids), labels))
    return sorted(clusters, key=lambda item: item.cluster_id)


def _split_distribution(records: Sequence[GoldRecord], case_ids: set[str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        if record.case_id in case_ids:
            counts.update(_case_labels(record))
    return dict(sorted(counts.items()))


def stratified_cluster_split(
    records: Sequence[GoldRecord],
    *,
    seed: int,
    locked_fraction: float,
) -> SplitManifest:
    if not 0.0 < locked_fraction < 1.0:
        raise RetrievalEvaluationError("locked_fraction must be between zero and one")
    clusters = _union_find_clusters(records)
    if len(clusters) < 2:
        raise RetrievalEvaluationError("at least two independent scope/document clusters are required")

    label_clusters: dict[str, set[str]] = defaultdict(set)
    for cluster in clusters:
        for label in cluster.labels:
            label_clusters[label].add(cluster.cluster_id)
    total_cases = sum(len(cluster.case_ids) for cluster in clusters)
    target = max(1, min(round(total_cases * locked_fraction), total_cases - 1))
    all_labels = sorted(label_clusters)

    def score(indices: tuple[int, ...]) -> tuple[float, float, str] | None:
        selected = {clusters[index].cluster_id for index in indices}
        if not selected or len(selected) == len(clusters):
            return None
        locked_size = sum(len(clusters[index].case_ids) for index in indices)
        tuning_ids = {cluster.cluster_id for cluster in clusters} - selected
        if any(
            not values & selected or not values & tuning_ids
            for values in label_clusters.values()
            if len(values) >= 2
        ):
            return None
        locked_counts = Counter(label for index in indices for label in clusters[index].labels)
        distribution_error = sum(
            abs(locked_counts[label] / len(indices) - len(label_clusters[label]) / len(clusters))
            for label in all_labels
        )
        tie = hashlib.sha256(f"{seed}:{','.join(sorted(selected))}".encode()).hexdigest()
        return abs(locked_size - target), distribution_error, tie

    candidates: Iterable[tuple[int, ...]]
    if len(clusters) <= 18:
        candidates = itertools.chain.from_iterable(
            itertools.combinations(range(len(clusters)), size) for size in range(1, len(clusters))
        )
    else:
        ordered = sorted(
            range(len(clusters)),
            key=lambda index: hashlib.sha256(f"{seed}:{clusters[index].cluster_id}".encode()).hexdigest(),
        )
        candidates = (tuple(ordered[:size]) for size in range(1, len(clusters)))

    best_indices: tuple[int, ...] | None = None
    best_score: tuple[float, float, str] | None = None
    for indices in candidates:
        candidate_score = score(indices)
        if candidate_score is not None and (best_score is None or candidate_score < best_score):
            best_indices, best_score = indices, candidate_score
    if best_indices is None:
        raise RetrievalEvaluationError("unable to produce a leakage-free stratified split")

    locked_clusters = {clusters[index].cluster_id for index in best_indices}
    locked_cases = {
        case_id
        for cluster in clusters
        if cluster.cluster_id in locked_clusters
        for case_id in cluster.case_ids
    }
    all_case_ids = {record.case_id for record in records}
    tuning_cases = all_case_ids - locked_cases
    distribution = {
        "all": _split_distribution(records, all_case_ids),
        "tuning": _split_distribution(records, tuning_cases),
        "locked": _split_distribution(records, locked_cases),
    }
    tuning_case_ids = tuple(sorted(tuning_cases))
    locked_case_ids = tuple(sorted(locked_cases))
    tuning_cluster_ids = tuple(
        sorted(cluster.cluster_id for cluster in clusters if cluster.cluster_id not in locked_clusters)
    )
    locked_cluster_ids = tuple(sorted(locked_clusters))
    payload = {
        "seed": seed,
        "locked_fraction": locked_fraction,
        "tuning_case_ids": tuning_case_ids,
        "locked_case_ids": locked_case_ids,
        "tuning_cluster_ids": tuning_cluster_ids,
        "locked_cluster_ids": locked_cluster_ids,
        "distribution": distribution,
    }
    return SplitManifest(
        seed=seed,
        locked_fraction=locked_fraction,
        tuning_case_ids=tuning_case_ids,
        locked_case_ids=locked_case_ids,
        tuning_cluster_ids=tuning_cluster_ids,
        locked_cluster_ids=locked_cluster_ids,
        distribution=distribution,
        manifest_sha256=_sha256_json(payload),
    )


def build_case_bindings(
    records: Sequence[GoldRecord],
    sidecars: Mapping[str, PrivateSidecarRecord],
) -> dict[str, _CaseBinding]:
    bindings: dict[str, _CaseBinding] = {}
    for record in records:
        sidecar = sidecars[record.case_id]
        grade_by_evidence = {item.evidence_id: item.relevance_grade for item in record.evidence}
        relevance = {item.chunk_id: grade_by_evidence[item.evidence_id] for item in sidecar.exact_evidence}
        if record.answerable and not relevance:
            raise RetrievalEvaluationError("answerable case lacks exact-evidence chunk ground truth")
        if not record.answerable and relevance:
            raise RetrievalEvaluationError("no-answer case declares relevant chunks")
        if len(relevance) != len(sidecar.exact_evidence):
            raise RetrievalEvaluationError("exact-evidence chunk IDs are not one-to-one")
        bindings[record.case_id] = _CaseBinding(record, sidecar, relevance)
    return bindings


def score_ranking(
    ranked_chunk_ids: Sequence[uuid.UUID],
    relevance: Mapping[uuid.UUID, int],
    *,
    answerable: bool,
    ks: Sequence[int] = (1, 5, 10, 20, 50),
) -> RankedMetrics:
    ranked = [str(value) for value in ranked_chunk_ids]
    if len(ranked) != len(set(ranked)):
        raise RetrievalEvaluationError("ranked chunk IDs must be unique")
    grades = {str(chunk_id): grade for chunk_id, grade in relevance.items()}
    recall: dict[str, float | None] = {}
    mrr: dict[str, float | None] = {}
    ndcg: dict[str, float | None] = {}
    for k in ks:
        recall[str(k)] = cast(float | None, recall_at_k(ranked, grades, k, answerable=answerable)["value"])
        mrr[str(k)] = cast(float | None, mrr_at_k(ranked, grades, k, answerable=answerable)["value"])
        ndcg[str(k)] = cast(float | None, ndcg_at_k(ranked, grades, k, answerable=answerable)["value"])
    return RankedMetrics(
        recall=recall,
        mrr=mrr,
        ndcg=ndcg,
        no_answer_false_positive=bool(ranked) if not answerable else None,
    )


def aggregate_pool(
    cases: Sequence[CaseArtifact],
    *,
    pool: PoolName = "final",
) -> AggregateMetrics:
    if not cases:
        raise RetrievalEvaluationError("cannot aggregate an empty case set")
    answerable = [case for case in cases if case.answerable]
    no_answer = [case for case in cases if not case.answerable]

    def mean_metric(name: Literal["recall", "mrr", "ndcg"], k: str) -> float | None:
        values = [
            value
            for case in answerable
            if (value := getattr(case.observation.pools[pool].metrics, name)[k]) is not None
        ]
        return sum(values) / len(values) if values else None

    latencies = [case.observation.pools[pool].latency_ms for case in cases]
    fpr_values = [bool(case.observation.pools[pool].metrics.no_answer_false_positive) for case in no_answer]
    returned_counts = [case.observation.returned_count for case in cases]
    abstained_count = sum(case.observation.abstained for case in cases)
    return AggregateMetrics(
        answerable_cases=len(answerable),
        no_answer_cases=len(no_answer),
        recall={k: mean_metric("recall", k) for k in ("1", "5", "10")},
        mrr={k: mean_metric("mrr", k) for k in ("1", "5", "10")},
        ndcg={k: mean_metric("ndcg", k) for k in ("1", "5", "10")},
        no_answer_false_positive_rate=(sum(fpr_values) / len(fpr_values) if fpr_values else None),
        returned_count=sum(returned_counts),
        abstained_count=abstained_count,
        no_answer_returned_count=sum(case.observation.returned_count for case in no_answer),
        no_answer_abstained_count=sum(case.observation.abstained for case in no_answer),
        latency_ms={
            "mean": sum(latencies) / len(latencies),
            "p50": _nearest_rank(latencies, 0.50),
            "p95": _nearest_rank(latencies, 0.95),
        },
    )


def aggregate_all_pools(cases: Sequence[CaseArtifact]) -> dict[PoolName, AggregateMetrics]:
    return {pool: aggregate_pool(cases, pool=pool) for pool in ("dense", "sparse", "hybrid", "final")}


def aggregate_slices(
    cases: Sequence[CaseArtifact],
) -> dict[str, dict[PoolName, AggregateMetrics]]:
    slices: dict[str, list[CaseArtifact]] = defaultdict(list)
    for case in cases:
        labels = {
            "overall",
            f"language:{case.language}",
            f"hop:{case.hop_type}",
            f"scope:{case.scope_id}",
            *(f"content:{value}" for value in case.content_types),
            *(f"challenge:{value}" for value in case.challenge_tags),
        }
        if not case.challenge_tags:
            labels.add("challenge:none")
        for label in labels:
            slices[label].append(case)
    return {label: aggregate_all_pools(selected) for label, selected in sorted(slices.items())}


def sweep_configs(
    control: RetrievalConfig,
    *,
    sparse_top_k: Sequence[int],
    rrf_k: Sequence[int],
    rerank_min_score: Sequence[float],
) -> tuple[RetrievalConfig, ...]:
    configs = {
        RetrievalConfig(
            dense_top_k=control.dense_top_k,
            sparse_top_k=sparse,
            rrf_k=rrf,
            rerank_top_k=control.rerank_top_k,
            rerank_min_score=threshold,
            final_top_k=control.final_top_k,
        ).fingerprint: RetrievalConfig(
            dense_top_k=control.dense_top_k,
            sparse_top_k=sparse,
            rrf_k=rrf,
            rerank_top_k=control.rerank_top_k,
            rerank_min_score=threshold,
            final_top_k=control.final_top_k,
        )
        for sparse, rrf, threshold in itertools.product(sparse_top_k, rrf_k, rerank_min_score)
    }
    if not configs:
        raise RetrievalEvaluationError("sweep must contain at least one configuration")
    return tuple(configs[key] for key in sorted(configs))


def select_tuning_config(results: Mapping[str, AggregateMetrics]) -> str:
    if not results:
        raise RetrievalEvaluationError("tuning results are empty")

    def objective(item: tuple[str, AggregateMetrics]) -> tuple[float, float, float, str]:
        fingerprint, metrics = item
        ndcg = metrics.ndcg["10"]
        recall = metrics.recall["5"]
        if ndcg is None or recall is None:
            raise RetrievalEvaluationError("tuning set has no eligible answerable metrics")
        return (-ndcg, -recall, metrics.latency_ms["p95"], fingerprint)

    return min(results.items(), key=objective)[0]


def _gold_corpus_fingerprint(records: Sequence[GoldRecord]) -> str:
    scopes: dict[str, dict[str, Any]] = defaultdict(dict)
    for record in records:
        for snapshot in record.document_scope:
            payload = snapshot.model_dump(mode="json")
            previous = scopes[record.scope_id].setdefault(snapshot.document_ref, payload)
            if previous != payload:
                raise RetrievalEvaluationError("gold corpus snapshot is internally inconsistent")
    canonical = [
        {"scope_id": scope, "documents": [docs[key] for key in sorted(docs)]}
        for scope, docs in sorted(scopes.items())
    ]
    return _sha256_json(canonical)


def _git_state() -> tuple[str, bool]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        raise RetrievalEvaluationError("unable to capture repository state") from None
    if not revision or not all(value in "0123456789abcdef" for value in revision):
        raise RetrievalEvaluationError("repository revision is invalid")
    return revision, bool(status.strip())


def _private_dir(path: Path) -> Path:
    path.mkdir(mode=0o700, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise RetrievalEvaluationError("private work path must be a regular directory")
    if path.stat().st_mode & 0o077:
        raise RetrievalEvaluationError("private work directory must not be group/world accessible")
    return path


def _case_artifact_path(
    work_dir: Path,
    *,
    split: SplitName,
    variant: VariantName,
    config_sha256: str,
    case_id: str,
) -> Path:
    directory = _private_dir(work_dir)
    for part in ("cases", split, variant, config_sha256):
        directory = _private_dir(directory / part)
    return directory / f"{case_id}.json"


def _case_hmac_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.hmac.json")


def _case_hmac_signature(artifact_bytes: bytes, key: bytes) -> str:
    return hmac.new(key, _CASE_HMAC_DOMAIN + artifact_bytes, hashlib.sha256).hexdigest()


def _case_hmac(artifact: CaseArtifact, artifact_bytes: bytes, key: bytes) -> CaseArtifactHmac:
    return CaseArtifactHmac(
        key_id=hashlib.sha256(key).hexdigest(),
        artifact_sha256=hashlib.sha256(artifact_bytes).hexdigest(),
        run_id=artifact.run_id,
        config_sha256=artifact.observation.config_sha256,
        case_id=artifact.case_id,
        signature=_case_hmac_signature(artifact_bytes, key),
    )


def _verify_case_hmac(
    path: Path,
    *,
    artifact: CaseArtifact,
    artifact_bytes: bytes,
    key: bytes,
) -> None:
    signature_path = _case_hmac_path(path)
    if not signature_path.exists():
        raise RetrievalEvaluationError("qualification resume artifact is missing its case HMAC")
    try:
        actual = read_private_json(
            signature_path,
            parser=lambda raw: CaseArtifactHmac.model_validate_json(raw, strict=True),
        ).value
    except (PrivateArtifactFormatError, ValueError):
        raise RetrievalEvaluationError("qualification case HMAC is invalid") from None
    expected = _case_hmac(artifact, artifact_bytes, key)
    if actual.model_dump(exclude={"signature"}) != expected.model_dump(
        exclude={"signature"}
    ) or not hmac.compare_digest(actual.signature, expected.signature):
        raise RetrievalEvaluationError("qualification case HMAC verification failed")


def _write_or_validate_case_hmac(
    path: Path,
    *,
    artifact: CaseArtifact,
    artifact_bytes: bytes,
    key: bytes,
) -> None:
    signature_path = _case_hmac_path(path)
    expected = _case_hmac(artifact, artifact_bytes, key)
    if signature_path.exists():
        _verify_case_hmac(
            path,
            artifact=artifact,
            artifact_bytes=artifact_bytes,
            key=key,
        )
        return
    write_private_json_fresh(
        signature_path,
        _canonical_bytes(expected.model_dump(mode="json")),
    )


def load_resumed_case(
    path: Path,
    *,
    run_id: str,
    split: SplitName,
    variant: VariantName,
    config_sha256: str,
    query_embedding_sha256: str,
    binding: _CaseBinding,
    hmac_key: bytes | None = None,
) -> CaseArtifact | None:
    if not path.exists():
        return None
    loaded = read_private_json(
        path,
        parser=lambda raw: CaseArtifact.model_validate_json(raw, strict=True),
    )
    artifact = loaded.value
    record = binding.record
    expected_identity = (
        run_id,
        split,
        variant,
        config_sha256,
        record.case_id,
        binding.sidecar.gold_case_sha256,
        record.question_sha256,
        _QUERY_EMBEDDING_PROTOCOL,
        query_embedding_sha256,
        record.scope_id,
    )
    actual_identity = (
        artifact.run_id,
        artifact.split,
        artifact.observation.variant,
        artifact.observation.config_sha256,
        artifact.case_id,
        artifact.gold_case_sha256,
        artifact.question_sha256,
        artifact.query_embedding_protocol,
        artifact.query_embedding_sha256,
        artifact.scope_id,
    )
    if actual_identity != expected_identity:
        raise RetrievalEvaluationError("resume artifact identity does not match this run")
    if hmac_key is not None:
        _verify_case_hmac(
            path,
            artifact=artifact,
            artifact_bytes=loaded.raw_bytes,
            key=hmac_key,
        )
    return artifact


def load_or_write_case(
    path: Path,
    expected: CaseArtifact,
    *,
    hmac_key: bytes | None = None,
) -> CaseArtifact:
    artifact_bytes: bytes
    if path.exists():
        loaded = read_private_json(
            path,
            parser=lambda raw: CaseArtifact.model_validate_json(raw, strict=True),
        )
        artifact = loaded.value
        if artifact != expected:
            raise RetrievalEvaluationError("resume artifact does not match the current observation")
        artifact_bytes = loaded.raw_bytes
    else:
        artifact_bytes = _canonical_bytes(expected.model_dump(mode="json"))
        write_private_json_fresh(path, artifact_bytes)
    if hmac_key is not None:
        _write_or_validate_case_hmac(
            path,
            artifact=expected,
            artifact_bytes=artifact_bytes,
            key=hmac_key,
        )
    return expected


def _read_external_evidence(
    path: Path,
    *,
    expected_kind: str,
) -> tuple[Any, _ExternalEvidenceBase]:
    parsers: dict[str, type[_ExternalEvidenceBase]] = {
        "rls": _RlsEvidence,
        "load": _LoadEvidence,
        "update": _UpdateEvidence,
        "delete": _DeleteEvidence,
        "restart": _RestartEvidence,
    }
    parser = parsers.get(expected_kind)
    if parser is None:
        raise RetrievalEvaluationError("external evidence kind is unsupported")
    try:
        artifact = read_private_json(
            path,
            parser=lambda raw: parser.model_validate_json(raw, strict=True),
        )
    except (PrivateArtifactFormatError, ValueError):
        raise RetrievalEvaluationError(f"{expected_kind} evidence schema is invalid") from None
    return artifact, artifact.value


def _parse_external_evidence(
    path: Path,
    *,
    expected_kind: str,
    corpus_snapshot_sha256: str,
    policy: RetrievalGatePolicy | None = None,
) -> EvidenceReference:
    artifact, value = _read_external_evidence(path, expected_kind=expected_kind)
    if value.kind != expected_kind:
        raise RetrievalEvaluationError(f"{expected_kind} evidence kind mismatch")
    if value.corpus_snapshot_sha256 != corpus_snapshot_sha256:
        raise RetrievalEvaluationError(f"{expected_kind} evidence corpus binding mismatch")
    if policy is not None:
        if isinstance(value, _RlsEvidence):
            if len(value.principals) < policy.min_rls_principal_count or any(
                item.leak_count for item in value.principals
            ):
                raise RetrievalEvaluationError("RLS evidence does not satisfy policy")
        elif isinstance(value, _LoadEvidence):
            for measurement in (value.baseline, value.candidate):
                if (
                    measurement.concurrency < policy.min_load_concurrency
                    or measurement.request_count < policy.min_load_requests
                    or measurement.error_count
                    or measurement.completed_count != measurement.request_count
                ):
                    raise RetrievalEvaluationError("load evidence does not satisfy policy")
        elif isinstance(value, (_UpdateEvidence, _DeleteEvidence)):
            if value.visibility_seconds > policy.max_operational_seconds:
                raise RetrievalEvaluationError("visibility evidence exceeds policy duration")
        elif isinstance(value, _RestartEvidence):
            if (
                value.recovery_seconds > policy.max_operational_seconds
                or value.rollback_seconds > policy.max_operational_seconds
                or value.rollback_index_manifest_sha256 != policy.required_baseline_index_manifest_sha256
                or value.candidate_image_digest != policy.required_candidate_image_digest
                or value.candidate_index_manifest_sha256 != policy.required_candidate_index_manifest_sha256
            ):
                raise RetrievalEvaluationError("restart/rollback evidence does not satisfy policy")
    return EvidenceReference(
        kind=cast(Any, expected_kind),
        schema_version=value.schema_version,
        artifact_sha256=artifact.sha256,
        corpus_snapshot_sha256=corpus_snapshot_sha256,
        passed=True,
    )


def _parse_operational_evidence(
    path: Path,
    *,
    corpus_snapshot_sha256: str,
    policy: RetrievalGatePolicy,
) -> tuple[EvidenceReference, OperationalEvidence]:
    try:
        artifact = read_private_json(
            path,
            parser=lambda raw: OperationalEvidence.model_validate_json(raw, strict=True),
        )
    except (PrivateArtifactFormatError, ValueError):
        raise RetrievalEvaluationError("operational evidence schema is invalid") from None
    value = artifact.value
    _validate_operational_raw_evidence(path, value=value, policy=policy)
    if (
        value.corpus_snapshot_sha256 != corpus_snapshot_sha256
        or value.candidate_image_digest != policy.required_candidate_image_digest
        or value.candidate_index_manifest_sha256 != policy.required_candidate_index_manifest_sha256
        or len(value.rls_principals) < policy.min_rls_principal_count
        or any(item.leak_count for item in value.rls_principals)
        or not value.update_visible
        or not value.delete_hidden
        or not value.restart_recovered
        or not value.rollback_succeeded
        or value.determinism_replays < policy.min_determinism_replays
        or value.determinism_mismatches
        or value.rollback_backend != "postgres_fts"
        or value.rollback_index_manifest_sha256 != policy.required_baseline_index_manifest_sha256
        or any(
            duration > policy.max_operational_seconds
            for duration in (
                value.update_visibility_seconds,
                value.delete_visibility_seconds,
                value.restart_recovery_seconds,
                value.determinism_seconds,
                value.rollback_seconds,
            )
        )
    ):
        raise RetrievalEvaluationError("operational evidence does not satisfy policy")
    return (
        EvidenceReference(
            kind="operational",
            schema_version="rag-operational-evidence-v3",
            artifact_sha256=artifact.sha256,
            corpus_snapshot_sha256=corpus_snapshot_sha256,
            passed=True,
        ),
        value,
    )


def _operational_component(
    directory: Path,
    *,
    name: str,
    expected_sha256: str,
) -> Mapping[str, Any]:
    filename = {
        "lifecycle": "lifecycle_raw.json",
        "restart": "restart_raw.json",
        "rollback": "rollback_raw.json",
        "determinism": "determinism_raw.json",
        "gold_scope_rls": "gold_scope_rls_raw.json",
    }[name]
    try:
        artifact = read_private_json(directory / filename)
    except PrivateArtifactError:
        raise RetrievalEvaluationError("operational component cannot be read safely") from None
    if artifact.sha256 != expected_sha256 or not isinstance(artifact.value, dict):
        raise RetrievalEvaluationError("operational component hash/schema mismatch")
    return cast(Mapping[str, Any], artifact.value)


def _validate_operational_raw_evidence(
    path: Path,
    *,
    value: OperationalEvidence,
    policy: RetrievalGatePolicy,
) -> None:
    raw_path = path.with_name("operational_raw.json")
    try:
        raw_artifact = read_private_json(raw_path)
    except PrivateArtifactError:
        raise RetrievalEvaluationError("operational raw evidence is invalid") from None
    if raw_artifact.sha256 != value.raw_evidence_sha256 or not isinstance(raw_artifact.value, dict):
        raise RetrievalEvaluationError("operational raw evidence hash mismatch")
    raw = cast(Mapping[str, Any], raw_artifact.value)
    component_hashes = raw.get("component_sha256")
    if not isinstance(component_hashes, dict):
        raise RetrievalEvaluationError("operational component manifest is invalid")
    expected_names = {
        "lifecycle",
        "restart",
        "rollback",
        "determinism",
        "gold_scope_rls",
    }
    if set(component_hashes) != expected_names or any(
        not isinstance(component_hashes[name], str)
        or re.fullmatch(_SHA256_PATTERN, component_hashes[name]) is None
        for name in expected_names
    ):
        raise RetrievalEvaluationError("operational component manifest is incomplete")
    components = {
        name: _operational_component(
            path.parent,
            name=name,
            expected_sha256=component_hashes[name],
        )
        for name in expected_names
    }
    lifecycle = components["lifecycle"]
    restart = components["restart"]
    rollback = components["rollback"]
    determinism = components["determinism"]
    gold_scope_rls = components["gold_scope_rls"]
    update = lifecycle.get("update")
    delete = lifecycle.get("delete")
    raw_principals = raw.get("rls_principals")
    if (
        not isinstance(update, dict)
        or not isinstance(delete, dict)
        or not isinstance(raw_principals, list)
        or any(not isinstance(item, dict) for item in raw_principals)
    ):
        raise RetrievalEvaluationError("operational raw observations are incomplete")
    principal_projection = tuple(
        sorted(
            (
                item.get("principal_ref"),
                item.get("probe_count"),
                item.get("leak_count"),
                item.get("evidence_sha256"),
            )
            for item in raw_principals
            if isinstance(item, dict)
        )
    )
    expected_principals = tuple(
        (
            item.principal_ref,
            item.probe_count,
            item.leak_count,
            item.evidence_sha256,
        )
        for item in value.rls_principals
    )
    restart_checks = (
        "extension_ready",
        "indexes_valid_ready_live",
        "postgres_ready",
        "restart_command_succeeded",
        "shared_preload_ready",
        "representative_bm25_en_nonempty",
        "representative_bm25_ru_nonempty",
    )
    rollback_negative = rollback.get("negative_cases")
    gold_scopes = gold_scope_rls.get("scopes")
    raw_matches = (
        raw.get("schema_version") == "rag-retrieval-operational-raw-v3"
        and raw.get("target") == "isolated_clone_only"
        and raw.get("production_changed") is False
        and raw.get("candidate_ready_for_gold") is True
        and raw.get("corpus_snapshot_sha256") == value.corpus_snapshot_sha256
        and raw.get("candidate_image_digest") == value.candidate_image_digest
        and raw.get("candidate_index_manifest_sha256") == value.candidate_index_manifest_sha256
        and principal_projection == expected_principals
        and gold_scope_rls.get("all_gold_scopes_covered") is True
        and gold_scope_rls.get("reviewed_case_count") == policy.expected_case_count
        and gold_scope_rls.get("total_leak_count") == 0
        and isinstance(gold_scopes, list)
        and len(gold_scopes) >= policy.min_owner_scope_count
        and value.update_visible
        == all(
            update.get(key) is True
            for key in ("committed", "new_en_visible", "new_ru_visible", "old_en_hidden", "old_ru_hidden")
        )
        and value.update_visibility_seconds == update.get("visibility_seconds")
        and value.delete_hidden
        == all(
            delete.get(key) is True
            for key in ("committed", "chunk_hidden", "document_hidden", "new_en_hidden", "new_ru_hidden")
        )
        and value.delete_visibility_seconds == delete.get("visibility_seconds")
        and value.restart_recovered == all(restart.get(key) is True for key in restart_checks)
        and value.restart_recovery_seconds == restart.get("recovery_seconds")
        and value.determinism_replays == determinism.get("replays")
        and value.determinism_mismatches == determinism.get("mismatches")
        and value.determinism_seconds == determinism.get("duration_seconds")
        and value.rollback_succeeded
        == (
            rollback.get("rollback_correct") is True
            and rollback.get("gin_valid_ready_live") is True
            and rollback.get("owner_mismatches") == 0
            and isinstance(rollback_negative, list)
            and all(isinstance(item, dict) and item.get("returned_count") == 0 for item in rollback_negative)
        )
        and value.rollback_seconds == rollback.get("rollback_seconds")
        and value.rollback_backend == rollback.get("rollback_backend")
        and value.rollback_index_manifest_sha256 == rollback.get("baseline_index_manifest_sha256")
    )
    if not raw_matches:
        raise RetrievalEvaluationError("operational summary does not match raw observations")


def _value_sha256(value: Any) -> str:
    if isinstance(value, (dict, list)):
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    else:
        encoded = str(value if value is not None else "").encode()
    return hashlib.sha256(encoded).hexdigest()


def _runtime_scope_digest(
    document_rows: Sequence[Any],
    chunk_rows: Sequence[Any],
    page_rows: Sequence[Any],
) -> str:
    payload = {
        "documents": [
            {
                "id": str(row.id),
                "s3_key_sha256": _value_sha256(row.s3_key_original),
                "page_count": row.page_count,
                "chunk_count": row.chunk_count,
                "indexed_at": row.indexed_at.isoformat() if row.indexed_at else None,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            }
            for row in document_rows
        ],
        "chunks": [
            {
                "id": str(row.id),
                "document_id": str(row.document_id),
                "idx": row.idx,
                "kind": row.kind,
                "heading_path_sha256": _value_sha256(row.heading_path or ""),
                "page_start": row.page_start,
                "page_end": row.page_end,
                "text_en_sha256": _value_sha256(row.text_en or ""),
                "body_sha256": _value_sha256(row.body or ""),
                "emb_en_sha256": _value_sha256(row.emb_en_text),
                "emb_ru_sha256": _value_sha256(row.emb_ru_text),
                "meta_sha256": _value_sha256(row.meta or {}),
            }
            for row in chunk_rows
        ],
        "page_embeddings": [
            {
                "id": str(row.id),
                "document_id": str(row.document_id),
                "page_idx": row.page_idx,
                "embedding_sha256": _value_sha256(row.emb_text),
                "meta_sha256": _value_sha256(row.meta or {}),
            }
            for row in page_rows
        ],
    }
    return _sha256_json(payload)


def _retrieval_probe_matches(
    probe: RetrievalProbe,
    row: Any,
    document_refs: Mapping[uuid.UUID, str],
    snapshot: DocumentSnapshot,
) -> bool:
    body = (row.body or "").strip()
    expected_page = min(max(int(row.page_start or 0) + 1, 1), snapshot.page_count)
    return (
        row.document_id == probe.document_id
        and document_refs.get(row.document_id) == probe.document_ref
        and row.page_start == probe.page_start
        and row.page_end == probe.page_end
        and probe.page == expected_page
        and text_sha256(body) == probe.content_sha256
    )


class _CorpusVerifier:
    def __init__(
        self,
        sessionmaker: async_sessionmaker[AsyncSession],
        storage: Storage,
    ) -> None:
        self.sessionmaker = sessionmaker
        self.storage = storage

    async def _load_scope_rows(
        self,
        sidecar: PrivateSidecarRecord,
    ) -> tuple[str, list[Any], list[Any], list[Any]]:
        source_ids = [item.document_id for item in sidecar.source_documents]
        token = set_principal("retrieval-corpus-verifier", True)
        try:
            async with self.sessionmaker() as session, session.begin():
                owner_rows = (
                    await session.execute(
                        sql(
                            "SELECT id, owner_sub FROM documents "
                            "WHERE id = ANY(CAST(:ids AS uuid[])) AND status = 'done'"
                        ),
                        {"ids": source_ids},
                    )
                ).all()
                if len(owner_rows) != len(source_ids):
                    raise RetrievalEvaluationError("source document resolution mismatch")
                owners = {row.owner_sub for row in owner_rows}
                if len(owners) != 1:
                    raise RetrievalEvaluationError("source documents do not resolve to one owner")
                owner_sub = str(next(iter(owners)))
                if make_scope_id(owner_sub) != sidecar.scope_id:
                    raise RetrievalEvaluationError("resolved owner scope hash mismatch")
                document_rows = (
                    await session.execute(
                        sql(
                            "SELECT id, s3_key_original, page_count, chunk_count, "
                            "indexed_at, updated_at FROM documents "
                            "WHERE owner_sub = :owner AND status = 'done' ORDER BY id"
                        ),
                        {"owner": owner_sub},
                    )
                ).all()
                document_ids = [row.id for row in document_rows]
                chunk_rows = (
                    await session.execute(
                        sql(
                            "SELECT id, document_id, idx, kind, heading_path, page_start, "
                            "page_end, text_en, COALESCE(NULLIF(text_ru, ''), text_en) AS body, "
                            "meta, emb_en::text AS emb_en_text, emb_ru::text AS emb_ru_text "
                            "FROM chunks WHERE document_id = ANY(CAST(:ids AS uuid[])) "
                            "ORDER BY document_id, idx, id"
                        ),
                        {"ids": document_ids},
                    )
                ).all()
                page_rows = (
                    await session.execute(
                        sql(
                            "SELECT id, document_id, page_idx, emb::text AS emb_text, meta "
                            "FROM page_embeddings WHERE document_id = ANY(CAST(:ids AS uuid[])) "
                            "ORDER BY document_id, page_idx, id"
                        ),
                        {"ids": document_ids},
                    )
                ).all()
        finally:
            reset_principal(token)
        return owner_sub, list(document_rows), list(chunk_rows), list(page_rows)

    async def _verify_case_evidence(
        self,
        record: GoldRecord,
        sidecar: PrivateSidecarRecord,
        document_refs: Mapping[uuid.UUID, str],
    ) -> None:
        chunk_ids = [item.chunk_id for item in sidecar.exact_evidence]
        chunk_ids.extend(item.chunk_id for item in sidecar.retrieval_probe)
        if not chunk_ids:
            return
        token = set_principal("retrieval-corpus-verifier", True)
        try:
            async with self.sessionmaker() as session, session.begin():
                rows = (
                    await session.execute(
                        sql(
                            "SELECT id, document_id, idx, kind, heading_path, page_start, "
                            "page_end, COALESCE(NULLIF(text_ru, ''), text_en) AS body "
                            "FROM chunks WHERE id = ANY(CAST(:ids AS uuid[]))"
                        ),
                        {"ids": list(dict.fromkeys(chunk_ids))},
                    )
                ).all()
        finally:
            reset_principal(token)
        by_id = {row.id: row for row in rows}
        if len(by_id) != len(set(chunk_ids)):
            raise RetrievalEvaluationError("sidecar chunk resolution mismatch")
        for item in sidecar.exact_evidence:
            row = by_id[item.chunk_id]
            body = (row.body or "").strip()
            if (
                row.document_id != item.document_id
                or document_refs.get(row.document_id) != item.document_ref
                or row.idx != item.chunk_index
                or row.kind != item.kind
                or (row.heading_path or "") != item.heading_path
                or row.page_start != item.page_start
                or row.page_end != item.page_end
                or text_sha256(body) != item.text_sha256
                or item.exact_quote not in body
                or text_sha256(item.exact_quote) != item.content_sha256
            ):
                raise RetrievalEvaluationError("production evidence hash/locator mismatch")
        snapshots = {item.document_ref: item for item in record.document_scope}
        for probe in sidecar.retrieval_probe:
            row = by_id[probe.chunk_id]
            snapshot = snapshots.get(probe.document_ref)
            if snapshot is None or not _retrieval_probe_matches(
                probe,
                row,
                document_refs,
                snapshot,
            ):
                raise RetrievalEvaluationError("production retrieval probe mismatch")

    async def verify(
        self,
        records: Sequence[GoldRecord],
        sidecars: Mapping[str, PrivateSidecarRecord],
    ) -> tuple[str, dict[str, _VerifiedScope]]:
        records_by_scope: dict[str, list[GoldRecord]] = defaultdict(list)
        for record in records:
            records_by_scope[record.scope_id].append(record)
        verified: dict[str, _VerifiedScope] = {}
        for scope_id, scope_records in sorted(records_by_scope.items()):
            first = min(scope_records, key=lambda item: item.case_id)
            first_sidecar = sidecars[first.case_id]
            owner_sub, documents, chunks, pages = await self._load_scope_rows(first_sidecar)
            chunks_by_document: dict[uuid.UUID, list[Any]] = defaultdict(list)
            for row in chunks:
                chunks_by_document[row.document_id].append(row)
            actual_snapshots: dict[str, tuple[str, int]] = {}
            document_refs: dict[uuid.UUID, str] = {}
            for document in documents:
                rows = chunks_by_document.get(document.id, [])
                if not rows:
                    raise RetrievalEvaluationError("scope document has no indexed chunks")
                source_bytes = await self.storage.get_bytes(
                    settings.bucket_originals,
                    document.s3_key_original,
                )
                document_ref = make_document_ref(bytes_sha256(source_bytes))
                page_count = max(
                    int(document.page_count or 0),
                    max(int(row.page_end or 0) + 1 for row in rows),
                    1,
                )
                parsed_sha256 = parsed_chunks_sha256(
                    [
                        {
                            "idx": row.idx,
                            "kind": row.kind,
                            "heading_path": row.heading_path or "",
                            "page_start": row.page_start,
                            "page_end": row.page_end,
                            "text": (row.text_en or "").strip(),
                        }
                        for row in rows
                    ]
                )
                actual_snapshots[document_ref] = (parsed_sha256, page_count)
                document_refs[document.id] = document_ref
            expected = {
                item.document_ref: (item.parsed_content_sha256, item.page_count)
                for item in first.document_scope
            }
            if actual_snapshots != expected:
                raise RetrievalEvaluationError("production corpus hash snapshot mismatch")
            for record in scope_records:
                record_expected = {
                    item.document_ref: (item.parsed_content_sha256, item.page_count)
                    for item in record.document_scope
                }
                if record_expected != expected:
                    raise RetrievalEvaluationError("Gold scope snapshot changed between cases")
                sidecar = sidecars[record.case_id]
                if any(
                    document_refs.get(item.document_id) != item.document_ref
                    for item in sidecar.source_documents
                ):
                    raise RetrievalEvaluationError("sidecar document mapping mismatch")
                await self._verify_case_evidence(record, sidecar, document_refs)
            digest = _runtime_scope_digest(documents, chunks, pages)
            evidence = ScopeEvidence(
                scope_id=scope_id,
                case_count=len(scope_records),
                document_count=len(documents),
                chunk_count=len(chunks),
                corpus_sha256=digest,
            )
            verified[scope_id] = _VerifiedScope(
                owner_sub=owner_sub,
                document_refs=document_refs,
                document_ids=frozenset(document_refs),
                evidence=evidence,
            )

        case_bindings = []
        for record in sorted(records, key=lambda item: item.case_id):
            sidecar = sidecars[record.case_id]
            case_bindings.append(
                {
                    "gold_case_sha256": sidecar.gold_case_sha256,
                    "evidence": sorted(
                        (
                            str(item.chunk_id),
                            item.text_sha256,
                            item.content_sha256,
                            item.page,
                            item.page_start,
                            item.page_end,
                        )
                        for item in sidecar.exact_evidence
                    ),
                    "retrieval_probe": sorted(
                        (
                            str(item.chunk_id),
                            item.content_sha256,
                            item.page,
                            item.page_start,
                            item.page_end,
                        )
                        for item in sidecar.retrieval_probe
                    ),
                }
            )
        snapshot = _sha256_json(
            {
                "scopes": [
                    {
                        "scope_id": scope_id,
                        "runtime_sha256": verified[scope_id].evidence.corpus_sha256,
                    }
                    for scope_id in sorted(verified)
                ],
                "case_bindings": case_bindings,
            }
        )
        return snapshot, verified


def _read_regular_bytes(path: Path, *, max_bytes: int = 16 * 1024 * 1024) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path.expanduser(), flags)
    except OSError:
        raise RetrievalEvaluationError("unable to open required input") from None
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or not 0 < info.st_size <= max_bytes:
            raise RetrievalEvaluationError("required input is not a bounded regular file")
        chunks: list[bytes] = []
        remaining = info.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise RetrievalEvaluationError("required input changed while reading")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RetrievalEvaluationError("required input changed while reading")
        after = os.fstat(descriptor)
        if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
        ):
            raise RetrievalEvaluationError("required input changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _tracked_repository_source(path: Path) -> str:
    try:
        repository = REPOSITORY_ROOT.resolve(strict=True)
        source = path.expanduser().resolve(strict=True)
        relative = source.relative_to(repository).as_posix()
    except (OSError, ValueError):
        raise RetrievalEvaluationError("qualification policy must be a tracked repository source") from None
    try:
        result = subprocess.run(  # noqa: S603
            ["git", "ls-files", "--error-unmatch", "--", relative],  # noqa: S607
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        raise RetrievalEvaluationError("qualification policy must be a tracked repository source") from None
    if result.stdout.strip() != relative:
        raise RetrievalEvaluationError("qualification policy must resolve to one tracked repository source")
    return relative


def _load_policy(
    path: Path,
    *,
    require_tracked: bool = False,
) -> tuple[RetrievalGatePolicy, bytes, str, str | None]:
    raw = _read_regular_bytes(path)
    try:
        policy = RetrievalGatePolicy.model_validate_json(raw, strict=True)
    except ValueError:
        raise RetrievalEvaluationError("retrieval policy schema is invalid") from None
    source = _tracked_repository_source(path) if require_tracked else None
    return policy, raw, hashlib.sha256(raw).hexdigest(), source


def _inspect_database_container(container: str) -> tuple[str, str]:
    if re.fullmatch(_CONTAINER_NAME_PATTERN, container) is None:
        raise RetrievalEvaluationError("database container name is invalid")
    try:
        result = subprocess.run(  # noqa: S603
            [
                "docker",
                "inspect",
                "--type",
                "container",
                "--format",
                "{{.Config.Image}}\n{{.Image}}",
                container,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        raise RetrievalEvaluationError("unable to inspect the database container image") from None
    lines = result.stdout.splitlines()
    if len(lines) != 2 or not lines[0] or re.fullmatch(r"sha256:[0-9a-f]{64}", lines[1]) is None:
        raise RetrievalEvaluationError("database container image evidence is invalid")
    return lines[0], lines[1]


def _runtime_database_evidence(
    *,
    container: str,
    extension_binary_path: Path,
    policy: RetrievalGatePolicy,
) -> DatabaseEvidence:
    if not extension_binary_path.expanduser().is_absolute():
        raise RetrievalEvaluationError("extension binary path must be absolute")
    image_ref, image_digest = _inspect_database_container(container)
    binary = _read_regular_bytes(extension_binary_path, max_bytes=256 * 1024 * 1024)
    recipe = _read_regular_bytes(_BUILD_RECIPE)
    prepare_sql = _read_regular_bytes(_PREPARE_SQL)
    binary_sha256 = hashlib.sha256(binary).hexdigest()
    recipe_sha256 = hashlib.sha256(recipe).hexdigest()
    prepare_sha256 = hashlib.sha256(prepare_sql).hexdigest()
    try:
        recipe_text = recipe.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise RetrievalEvaluationError("database build recipe is not valid UTF-8") from None
    base_match = re.search(r"^FROM\s+\S+@(sha256:[0-9a-f]{64})$", recipe_text, re.MULTILINE)
    version_match = re.search(
        r"^ARG PG_TEXTSEARCH_VERSION=([0-9]+\.[0-9]+\.[0-9]+)$",
        recipe_text,
        re.MULTILINE,
    )
    package_match = re.search(
        r"^ARG PG_TEXTSEARCH_SHA256=([0-9a-f]{64})$",
        recipe_text,
        re.MULTILINE,
    )
    if (
        base_match is None
        or version_match is None
        or package_match is None
        or image_digest != policy.required_candidate_image_digest
        or binary_sha256 != policy.required_extension_binary_sha256
        or recipe_sha256 != policy.required_build_recipe_sha256
        or prepare_sha256 != policy.required_prepare_sql_sha256
        or base_match.group(1) != policy.required_base_postgres_image_digest
        or version_match.group(1) != policy.required_pg_textsearch_version
        or package_match.group(1) != policy.required_pg_textsearch_package_sha256
        or policy.required_pg_textsearch_commit[:8] not in image_ref
    ):
        raise RetrievalEvaluationError("database runtime provenance does not match policy")
    return DatabaseEvidence(
        image_ref=image_ref,
        image_digest=image_digest,
        server_version_num=170_000,
        extensions={},
        index_definitions={},
        index_definitions_sha256={},
        extension_binary_sha256=binary_sha256,
        extension_binary_bytes=len(binary),
        extension_version=version_match.group(1),
        extension_commit=policy.required_pg_textsearch_commit,
        package_sha256=package_match.group(1),
        base_image_digest=base_match.group(1),
        build_recipe_sha256=recipe_sha256,
        prepare_sql_sha256=prepare_sha256,
        baseline_index_manifest_sha256=policy.required_baseline_index_manifest_sha256,
        candidate_index_manifest_sha256=policy.required_candidate_index_manifest_sha256,
    )


def _load_model_revision(
    path: Path,
    *,
    expected_model: str,
) -> tuple[ModelEndpointRevision, str]:
    try:
        artifact = read_private_json(
            path,
            parser=lambda raw: ModelEndpointRevision.model_validate_json(raw, strict=True),
        )
    except (PrivateArtifactFormatError, ValueError):
        raise RetrievalEvaluationError("model revision evidence is invalid") from None
    if artifact.value.model != expected_model:
        raise RetrievalEvaluationError("model revision evidence model does not match runtime")
    return artifact.value, artifact.sha256


async def _database_evidence(
    engine: AsyncEngine,
    *,
    policy: RetrievalGatePolicy,
    runtime: DatabaseEvidence,
) -> DatabaseEvidence:
    names = [item.name for item in policy.required_baseline_indexes]
    names.extend(item.name for item in policy.required_candidate_indexes)
    async with engine.connect() as connection:
        version = int((await connection.execute(sql("SHOW server_version_num"))).scalar_one())
        extension_rows = (
            await connection.execute(sql("SELECT extname, extversion FROM pg_extension ORDER BY extname"))
        ).all()
        index_rows = (
            await connection.execute(
                sql(
                    "SELECT indexname, indexdef FROM pg_indexes "
                    "WHERE schemaname = current_schema() "
                    "AND indexname = ANY(CAST(:names AS text[])) ORDER BY indexname"
                ),
                {"names": names},
            )
        ).all()
    if version < 170_000 or version >= 180_000:
        raise RetrievalEvaluationError("retrieval qualification requires PostgreSQL 17")
    definitions = {str(row.indexname): str(row.indexdef) for row in index_rows}
    if set(definitions) != set(names):
        raise RetrievalEvaluationError("required sparse indexes are missing")
    expected_definitions = {item.name: item.canonical_definition for item in policy.required_baseline_indexes}
    expected_definitions.update(
        {item.name: item.canonical_definition for item in policy.required_candidate_indexes}
    )
    if definitions != expected_definitions:
        raise RetrievalEvaluationError("sparse index definition does not match policy")
    extensions = {str(row.extname): str(row.extversion) for row in extension_rows}
    if extensions.get("pg_textsearch") != policy.required_pg_textsearch_version:
        raise RetrievalEvaluationError("pg_textsearch version does not match policy")
    definition_hashes = {
        name: hashlib.sha256(definition.encode()).hexdigest()
        for name, definition in sorted(definitions.items())
    }
    actual_baseline_indexes = tuple(
        SparseIndexDefinition(
            name=item.name,
            access_method="gin",
            canonical_definition=definitions[item.name],
            definition_sha256=definition_hashes[item.name],
        )
        for item in policy.required_baseline_indexes
    )
    actual_candidate_indexes = tuple(
        SparseIndexDefinition(
            name=item.name,
            access_method="bm25",
            text_config=item.text_config,
            k1=item.k1,
            b=item.b,
            canonical_definition=definitions[item.name],
            definition_sha256=definition_hashes[item.name],
        )
        for item in policy.required_candidate_indexes
    )
    return DatabaseEvidence(
        image_ref=runtime.image_ref,
        image_digest=runtime.image_digest,
        server_version_num=version,
        extensions=extensions,
        index_definitions=definitions,
        index_definitions_sha256=definition_hashes,
        extension_binary_sha256=runtime.extension_binary_sha256,
        extension_binary_bytes=runtime.extension_binary_bytes,
        extension_version=extensions["pg_textsearch"],
        extension_commit=runtime.extension_commit,
        package_sha256=runtime.package_sha256,
        base_image_digest=runtime.base_image_digest,
        build_recipe_sha256=runtime.build_recipe_sha256,
        prepare_sql_sha256=runtime.prepare_sql_sha256,
        baseline_index_manifest_sha256=canonical_sha256(actual_baseline_indexes),
        candidate_index_manifest_sha256=canonical_sha256(actual_candidate_indexes),
    )


def _pool_rows(trace: RetrievalTrace) -> dict[PoolName, tuple[Any, ...]]:
    return {
        "dense": trace.dense,
        "sparse": trace.sparse,
        "hybrid": trace.hybrid_pre_rerank,
        "final": trace.final,
    }


def _pool_latency(trace: RetrievalTrace, pool: PoolName) -> float:
    latency = trace.stage_latency_ms
    if pool == "dense":
        return latency.get("embedding", 0.0) + latency.get("dense_sql", 0.0)
    if pool == "sparse":
        return latency.get("sparse_sql", 0.0)
    if pool == "hybrid":
        return sum(latency.get(name, 0.0) for name in ("embedding", "dense_sql", "sparse_sql", "fusion"))
    return latency.get("total", 0.0)


def _expected_sparse_engine(record: GoldRecord, backend: SparseBackend) -> SparseEngine:
    planned = sparse_query_plan(record.question, backend).engine
    if backend == "postgres_fts":
        return "postgres_fts"
    if record.language == "ru":
        expected: SparseEngine = "pg_textsearch_ru"
    elif record.language == "en":
        expected = "pg_textsearch_en"
    else:
        expected = "postgres_fts"
    if planned != expected:
        raise RetrievalEvaluationError("Gold language and production sparse script routing disagree")
    return expected


async def _execute_case(
    *,
    retriever: TraceRetriever,
    sessionmaker: async_sessionmaker[AsyncSession],
    binding: _CaseBinding,
    scope: _VerifiedScope,
    config: RetrievalConfig,
    backend: SparseBackend,
    variant: VariantName,
    split: SplitName,
    run_id: str,
    repeat_count: int,
    query_embedding_sha256: str,
) -> CaseArtifact:
    pool_orders: dict[PoolName, list[tuple[uuid.UUID, ...]]] = defaultdict(list)
    pool_latencies: dict[PoolName, list[float]] = defaultdict(list)
    repeat_hashes: list[str] = []
    rerank_score_snapshots: list[dict[uuid.UUID, float]] = []
    sparse_engines: set[str] = set()
    reranker_fallback = False
    record = binding.record
    expected_sparse_engine = _expected_sparse_engine(record, backend)
    for _ in range(repeat_count):
        token = set_principal(scope.owner_sub, False)
        try:
            async with sessionmaker() as session, session.begin():
                trace = await retriever.retrieve_with_trace(
                    session,
                    record.question,
                    top_k=config.final_top_k,
                    owner_sub=scope.owner_sub,
                    allow_rerank_fallback=False,
                    sparse_backend=backend,
                    dense_top_k=config.dense_top_k,
                    sparse_top_k=config.sparse_top_k,
                    rrf_k=config.rrf_k,
                    rerank_top_k=config.rerank_top_k,
                    rerank_min_score=config.rerank_min_score,
                )
        finally:
            reset_principal(token)
        if trace.requested_sparse_backend != backend:
            raise RetrievalEvaluationError("retriever used an unexpected sparse backend")
        if trace.sparse_engine != expected_sparse_engine:
            raise RetrievalEvaluationError("retriever used an unexpected sparse engine")
        if any(
            chunk.document_id not in scope.document_ids
            for rows in _pool_rows(trace).values()
            for chunk in rows
        ):
            raise RetrievalEvaluationError("retriever escaped the verified owner scope")
        sparse_engines.add(trace.sparse_engine)
        reranker_fallback = reranker_fallback or trace.reranker_fallback
        reranked_order = tuple(chunk.id for chunk in trace.reranked)
        if len(reranked_order) != len(set(reranked_order)):
            raise RetrievalEvaluationError("reranker returned duplicate chunk IDs")
        rerank_score_snapshots.append({chunk.id: chunk.score for chunk in trace.reranked})
        order_payload: dict[str, list[str]] = {}
        for pool, rows in _pool_rows(trace).items():
            order = tuple(chunk.id for chunk in rows)
            if len(order) != len(set(order)):
                raise RetrievalEvaluationError("retrieval stage returned duplicate chunk IDs")
            pool_orders[pool].append(order)
            pool_latencies[pool].append(_pool_latency(trace, pool))
            order_payload[pool] = [str(chunk_id) for chunk_id in order]
        final_order = pool_orders["final"][-1]
        if final_order != reranked_order[: len(final_order)]:
            raise RetrievalEvaluationError("final retrieval results are not a reranked prefix")
        repeat_hashes.append(_sha256_json(order_payload))
    deterministic = len(set(repeat_hashes)) == 1
    reranked_sets_stable = len({frozenset(snapshot) for snapshot in rerank_score_snapshots}) == 1
    if not reranked_sets_stable:
        raise RetrievalEvaluationError("reranker candidate universe changed between repeats")
    shared_reranked = set(rerank_score_snapshots[0])
    all_score_delta = max(
        (
            max(snapshot[chunk_id] for snapshot in rerank_score_snapshots)
            - min(snapshot[chunk_id] for snapshot in rerank_score_snapshots)
            for chunk_id in shared_reranked
        ),
        default=0.0,
    )
    output_ids = set().union(*(set(order) for order in pool_orders["final"]))
    max_score_delta = max(
        (
            max(snapshot[chunk_id] for snapshot in rerank_score_snapshots)
            - min(snapshot[chunk_id] for snapshot in rerank_score_snapshots)
            for chunk_id in output_ids
        ),
        default=0.0,
    )
    consensus_applied = False
    if not deterministic:
        unstable = ",".join(pool for pool, orders in sorted(pool_orders.items()) if len(set(orders)) > 1)
        final_sets_stable = len({frozenset(order) for order in pool_orders["final"]}) == 1
        final_lengths = {len(order) for order in pool_orders["final"]}
        final_count = next(iter(final_lengths)) if len(final_lengths) == 1 else 0
        can_apply_consensus = (
            unstable == "final"
            and len(final_lengths) == 1
            and final_count > 0
            and max_score_delta <= _MAX_RERANK_REPEAT_DELTA + _SCORE_EPSILON
        )
        if not can_apply_consensus:
            raise RetrievalEvaluationError(
                "repeated retrieval ordering is not deterministic "
                f"(stages={unstable},final_set_stable={str(final_sets_stable).lower()},"
                f"max_score_delta={max_score_delta:.6f},"
                f"all_reranker_score_delta={all_score_delta:.6f})"
            )
        canonical_final = tuple(
            sorted(
                output_ids,
                key=lambda chunk_id: (
                    -math.fsum(snapshot[chunk_id] for snapshot in rerank_score_snapshots)
                    / len(rerank_score_snapshots),
                    chunk_id.int,
                ),
            )[:final_count]
        )
        pool_orders["final"] = [canonical_final] * repeat_count
        consensus_applied = True
        deterministic = True
    if reranker_fallback:
        raise RetrievalEvaluationError("reranker fallback occurred during qualification")
    if len(sparse_engines) != 1:
        raise RetrievalEvaluationError("sparse engine changed between deterministic repeats")

    pools: dict[PoolName, PoolObservation] = {}
    for pool in ("dense", "sparse", "hybrid", "final"):
        first = pool_orders[pool][0]
        pools[pool] = PoolObservation(
            ranked_chunk_ids=first,
            order_sha256=_sha256_order(first),
            latency_ms=sum(pool_latencies[pool]) / len(pool_latencies[pool]),
            metrics=score_ranking(
                first,
                binding.relevance,
                answerable=record.answerable,
            ),
        )
    final_count = len(pools["final"].ranked_chunk_ids)
    return CaseArtifact(
        run_id=run_id,
        split=split,
        case_id=record.case_id,
        gold_case_sha256=binding.sidecar.gold_case_sha256,
        question_sha256=record.question_sha256,
        query_embedding_protocol=_QUERY_EMBEDDING_PROTOCOL,
        query_embedding_sha256=query_embedding_sha256,
        scope_id=record.scope_id,
        language=record.language,
        hop_type=record.hop_type,
        content_types=tuple(record.content_types),
        challenge_tags=tuple(record.challenge_tags),
        answerable=record.answerable,
        relevant_chunk_ids=tuple(sorted(binding.relevance, key=lambda value: value.int)),
        relevance_grades={
            str(chunk_id): binding.relevance[chunk_id]
            for chunk_id in sorted(binding.relevance, key=lambda value: value.int)
        },
        observation=VariantObservation(
            variant=variant,
            requested_sparse_backend=backend,
            sparse_engine=next(iter(sparse_engines)),
            config_sha256=config.fingerprint,
            pools=pools,
            repeat_order_sha256=tuple(repeat_hashes),
            deterministic=deterministic,
            reranker_consensus_applied=consensus_applied,
            reranker_max_score_delta=max_score_delta,
            reranker_all_max_score_delta=all_score_delta,
            reranker_fallback=False,
            returned_count=final_count,
            abstained=final_count == 0,
        ),
    )


async def _run_or_resume_case(
    *,
    retriever: TraceRetriever,
    sessionmaker: async_sessionmaker[AsyncSession],
    work_dir: Path,
    binding: _CaseBinding,
    scope: _VerifiedScope,
    config: RetrievalConfig,
    backend: SparseBackend,
    variant: VariantName,
    split: SplitName,
    run_id: str,
    repeat_count: int,
    query_embedding_sha256: str,
    hmac_key: bytes | None = None,
) -> CaseArtifact:
    path = _case_artifact_path(
        work_dir,
        split=split,
        variant=variant,
        config_sha256=config.fingerprint,
        case_id=binding.record.case_id,
    )
    resumed = load_resumed_case(
        path,
        run_id=run_id,
        split=split,
        variant=variant,
        config_sha256=config.fingerprint,
        query_embedding_sha256=query_embedding_sha256,
        binding=binding,
        hmac_key=hmac_key,
    )
    if resumed is not None:
        return resumed
    artifact = await _execute_case(
        retriever=retriever,
        sessionmaker=sessionmaker,
        binding=binding,
        scope=scope,
        config=config,
        backend=backend,
        variant=variant,
        split=split,
        run_id=run_id,
        repeat_count=repeat_count,
        query_embedding_sha256=query_embedding_sha256,
    )
    return load_or_write_case(path, artifact, hmac_key=hmac_key)


async def _load_request(
    *,
    retriever: TraceRetriever,
    sessionmaker: async_sessionmaker[AsyncSession],
    semaphore: asyncio.Semaphore,
    binding: _CaseBinding,
    scope: _VerifiedScope,
    config: RetrievalConfig,
    backend: SparseBackend,
    pair_index: int,
    order_in_pair: Literal[0, 1],
) -> LoadRequestObservation:
    request_id = _load_request_id(
        case_id=binding.record.case_id,
        backend=backend,
        pair_index=pair_index,
        order_in_pair=order_in_pair,
        config_sha256=config.fingerprint,
    )
    async with semaphore:
        started_at = datetime.now(UTC)
        started = time.perf_counter()
        sparse_engine: str | None = None
        returned_count = 0
        order_sha256: str | None = None
        error_code: str | None = None
        token = set_principal(scope.owner_sub, False)
        try:
            async with sessionmaker() as session, session.begin():
                trace = await retriever.retrieve_with_trace(
                    session,
                    binding.record.question,
                    top_k=config.final_top_k,
                    owner_sub=scope.owner_sub,
                    allow_rerank_fallback=False,
                    sparse_backend=backend,
                    dense_top_k=config.dense_top_k,
                    sparse_top_k=config.sparse_top_k,
                    rrf_k=config.rrf_k,
                    rerank_top_k=config.rerank_top_k,
                    rerank_min_score=config.rerank_min_score,
                )
            sparse_engine = trace.sparse_engine
            if trace.requested_sparse_backend != backend or trace.reranker_fallback:
                raise RetrievalEvaluationError("load request changed retrieval execution mode")
            if trace.sparse_engine != _expected_sparse_engine(binding.record, backend):
                raise RetrievalEvaluationError("load request used an unexpected sparse engine")
            if any(chunk.document_id not in scope.document_ids for chunk in trace.final):
                raise RetrievalEvaluationError("load request escaped the verified owner scope")
            order = tuple(chunk.id for chunk in trace.final)
            if len(order) != len(set(order)):
                raise RetrievalEvaluationError("load request returned duplicate chunk IDs")
            returned_count = len(order)
            order_sha256 = _sha256_order(order)
        except Exception as error:  # noqa: BLE001 - the raw artifact records sanitized failure type
            error_code = type(error).__name__
        finally:
            reset_principal(token)
        completed_at = datetime.now(UTC)
        return LoadRequestObservation(
            request_id=request_id,
            pair_index=pair_index,
            order_in_pair=order_in_pair,
            case_id=binding.record.case_id,
            backend=backend,
            sparse_engine=sparse_engine,
            started_at=started_at,
            completed_at=completed_at,
            latency_ms=(time.perf_counter() - started) * 1000,
            returned_count=returned_count,
            order_sha256=order_sha256,
            success=error_code is None,
            error_code=error_code,
        )


def _load_request_id(
    *,
    case_id: str,
    backend: SparseBackend,
    pair_index: int,
    order_in_pair: Literal[0, 1],
    config_sha256: str,
) -> str:
    return _sha256_json(
        {
            "case_id": case_id,
            "backend": backend,
            "pair_index": pair_index,
            "order_in_pair": order_in_pair,
            "config_sha256": config_sha256,
        }
    )


def _observed_peak_concurrency(
    observations: Sequence[LoadRequestObservation],
) -> int:
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
        if active < 0:
            raise RetrievalEvaluationError("raw load concurrency timeline is invalid")
    if active != 0 or peak < 1:
        raise RetrievalEvaluationError("raw load concurrency timeline is incomplete")
    return peak


def _validate_raw_load_evidence(
    raw: RawLoadEvidence,
    *,
    config: RetrievalConfig,
    locked_case_ids: Sequence[str],
    corpus_snapshot_sha256: str,
    concurrency: int | None = None,
    requests_per_backend: int | None = None,
) -> int:
    locked_ids = tuple(sorted(locked_case_ids))
    if not locked_ids:
        raise RetrievalEvaluationError("load evidence requires locked cases")
    if (
        raw.corpus_snapshot_sha256 != corpus_snapshot_sha256
        or raw.config_sha256 != config.fingerprint
        or raw.locked_case_manifest_sha256 != _sha256_json(locked_ids)
        or (concurrency is not None and raw.concurrency != concurrency)
        or (requests_per_backend is not None and raw.requests_per_backend != requests_per_backend)
    ):
        raise RetrievalEvaluationError("raw load evidence does not match this run")
    for item in raw.observations:
        expected_case_id = locked_ids[item.pair_index % len(locked_ids)]
        expected_id = _load_request_id(
            case_id=expected_case_id,
            backend=item.backend,
            pair_index=item.pair_index,
            order_in_pair=item.order_in_pair,
            config_sha256=config.fingerprint,
        )
        if item.case_id != expected_case_id or item.request_id != expected_id:
            raise RetrievalEvaluationError("raw load request binding is invalid")
    observed_peak = _observed_peak_concurrency(raw.observations)
    if observed_peak != raw.concurrency:
        raise RetrievalEvaluationError("raw load did not reach its declared concurrency")
    return observed_peak


def _aggregate_load_backend(
    raw: RawLoadEvidence,
    backend: SparseBackend,
    *,
    observed_peak_concurrency: int,
) -> LoadEvidence:
    observations = [item for item in raw.observations if item.backend == backend]
    completed = sum(item.success for item in observations)
    errors = len(observations) - completed
    if errors or completed != raw.requests_per_backend:
        raise RetrievalEvaluationError("load evidence contains failed or missing requests")
    started_at = min(item.started_at for item in observations)
    completed_at = max(item.completed_at for item in observations)
    duration = (completed_at - started_at).total_seconds()
    if duration <= 0:
        raise RetrievalEvaluationError("load evidence duration is invalid")
    return LoadEvidence(
        concurrency=observed_peak_concurrency,
        request_count=len(observations),
        completed_count=completed,
        error_count=errors,
        duration_seconds=duration,
        p95_latency_ms=_nearest_rank(
            [item.latency_ms for item in observations],
            0.95,
        ),
        throughput_rps=completed / duration,
        raw_observations_sha256=_sha256_json([item.model_dump(mode="json") for item in observations]),
    )


async def generate_load_evidence(
    *,
    output: Path,
    retriever: TraceRetriever,
    sessionmaker: async_sessionmaker[AsyncSession],
    bindings: Mapping[str, _CaseBinding],
    scopes: Mapping[str, _VerifiedScope],
    locked_case_ids: Sequence[str],
    config: RetrievalConfig,
    corpus_snapshot_sha256: str,
    concurrency: int,
    requests_per_backend: int,
) -> _LoadEvidence:
    if concurrency < 1 or requests_per_backend < 1:
        raise RetrievalEvaluationError("load concurrency and request count must be positive")
    if not locked_case_ids:
        raise RetrievalEvaluationError("load evidence requires locked cases")
    locked_ids = tuple(sorted(locked_case_ids))
    locked_manifest = _sha256_json(locked_ids)
    raw_path = output.with_name(f"{output.stem}.raw.json")
    if output.exists():
        return _validate_load_evidence_binding(
            output,
            config=config,
            locked_case_ids=locked_ids,
            corpus_snapshot_sha256=corpus_snapshot_sha256,
            concurrency=concurrency,
            requests_per_backend=requests_per_backend,
        )

    if raw_path.exists():
        raw_artifact = read_private_json(
            raw_path,
            parser=lambda value: RawLoadEvidence.model_validate_json(value, strict=True),
        )
        raw = raw_artifact.value
        raw_artifact_sha256 = raw_artifact.sha256
        _validate_raw_load_evidence(
            raw,
            config=config,
            locked_case_ids=locked_ids,
            corpus_snapshot_sha256=corpus_snapshot_sha256,
            concurrency=concurrency,
            requests_per_backend=requests_per_backend,
        )
    else:
        semaphore = asyncio.Semaphore(concurrency)
        started_at = datetime.now(UTC)

        async def run_pair(pair_index: int) -> tuple[LoadRequestObservation, ...]:
            case_id = locked_ids[pair_index % len(locked_ids)]
            binding = bindings[case_id]
            scope = scopes[binding.record.scope_id]
            backends: tuple[SparseBackend, SparseBackend] = (
                ("postgres_fts", "pg_textsearch")
                if pair_index % 2 == 0
                else ("pg_textsearch", "postgres_fts")
            )
            pair_observations = []
            for order_in_pair, backend in enumerate(backends):
                pair_observations.append(
                    await _load_request(
                        retriever=retriever,
                        sessionmaker=sessionmaker,
                        semaphore=semaphore,
                        binding=binding,
                        scope=scope,
                        config=config,
                        backend=backend,
                        pair_index=pair_index,
                        order_in_pair=cast(Literal[0, 1], order_in_pair),
                    )
                )
            return tuple(pair_observations)

        paired = await asyncio.gather(*(run_pair(pair_index) for pair_index in range(requests_per_backend)))
        observations = tuple(item for pair in paired for item in pair)
        raw = RawLoadEvidence(
            corpus_snapshot_sha256=corpus_snapshot_sha256,
            config_sha256=config.fingerprint,
            locked_case_manifest_sha256=locked_manifest,
            embedding_protocol=_LOAD_EMBEDDING_PROTOCOL,
            concurrency=concurrency,
            requests_per_backend=requests_per_backend,
            started_at=started_at,
            completed_at=datetime.now(UTC),
            observations=observations,
        )
        write_private_json_fresh(
            raw_path,
            _canonical_bytes(raw.model_dump(mode="json")),
        )
        raw_artifact_sha256 = read_private_bytes(
            raw_path,
            max_bytes=256 * 1024 * 1024,
        ).sha256
    observed_peak = _validate_raw_load_evidence(
        raw,
        config=config,
        locked_case_ids=locked_ids,
        corpus_snapshot_sha256=corpus_snapshot_sha256,
        concurrency=concurrency,
        requests_per_backend=requests_per_backend,
    )
    baseline = _aggregate_load_backend(
        raw,
        "postgres_fts",
        observed_peak_concurrency=observed_peak,
    )
    candidate = _aggregate_load_backend(
        raw,
        "pg_textsearch",
        observed_peak_concurrency=observed_peak,
    )
    envelope = _LoadEvidence(
        schema_version="retrieval-load-evidence-v2",
        kind="load",
        passed=True,
        corpus_snapshot_sha256=corpus_snapshot_sha256,
        config_sha256=config.fingerprint,
        locked_case_manifest_sha256=locked_manifest,
        raw_artifact_sha256=raw_artifact_sha256,
        embedding_protocol=_LOAD_EMBEDDING_PROTOCOL,
        baseline=baseline,
        candidate=candidate,
    )
    write_private_json_fresh(
        output,
        _canonical_bytes(envelope.model_dump(mode="json")),
    )
    return envelope


def _statistical_cluster_ids(
    bindings: Mapping[str, _CaseBinding],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for case_id, binding in bindings.items():
        evidence_documents = sorted({item.document_ref for item in binding.sidecar.exact_evidence})
        payload: dict[str, Any]
        if binding.record.answerable:
            payload = {
                "scope_id": binding.record.scope_id,
                "documents": evidence_documents,
            }
        else:
            payload = {
                "scope_id": binding.record.scope_id,
                "source_documents": sorted({item.document_ref for item in binding.sidecar.source_documents}),
                "generation_model": binding.sidecar.generation.model,
                "generation_seed": binding.sidecar.generation.seed,
            }
        result[case_id] = f"cluster-sha256:{_sha256_json(payload)}"
    return result


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise RetrievalEvaluationError("bootstrap sample is empty")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _locked_bootstrap(
    rows: Sequence[tuple[str, float, float]],
    *,
    samples: int,
    seed: int,
    confidence_level: float,
) -> tuple[float, float, float, float, float, int]:
    grouped: dict[str, list[float]] = defaultdict(list)
    baseline_values: list[float] = []
    candidate_values: list[float] = []
    for cluster_id, baseline, candidate in rows:
        grouped[cluster_id].append(candidate - baseline)
        baseline_values.append(baseline)
        candidate_values.append(candidate)
    if len(grouped) < 2:
        raise RetrievalEvaluationError("locked decision requires at least two statistical clusters")
    clusters = tuple((sum(values), len(values)) for _, values in sorted(grouped.items()))
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
    tail = (1 - confidence_level) / 2
    baseline_mean = math.fsum(baseline_values) / len(baseline_values)
    candidate_mean = math.fsum(candidate_values) / len(candidate_values)
    return (
        baseline_mean,
        candidate_mean,
        candidate_mean - baseline_mean,
        _quantile(bootstrapped, tail),
        _quantile(bootstrapped, 1 - tail),
        len(clusters),
    )


def _locked_metric_values(artifact: CaseArtifact) -> dict[str, float]:
    if not artifact.answerable:
        return {}
    case = _gate_case_result(artifact, cluster_id=f"cluster-sha256:{'0' * 64}")
    if case.metrics is None:
        raise RetrievalEvaluationError("locked answerable case lacks metrics")
    return {
        name: float(getattr(case.metrics, name))
        for name in (
            "recall_at_5",
            "recall_at_10",
            "mrr_at_10",
            "ndcg_at_10",
            "lexical_recall_at_5",
            "lexical_recall_at_50",
            "hybrid_union_recall_at_20",
        )
    }


def _assert_query_embedding_pairing(
    baseline: Sequence[CaseArtifact],
    candidate: Sequence[CaseArtifact],
    *,
    evidence: QueryEmbeddingEvidence | None = None,
) -> None:
    baseline_by_id = {item.case_id: item for item in baseline}
    candidate_by_id = {item.case_id: item for item in candidate}
    if (
        len(baseline_by_id) != len(baseline)
        or len(candidate_by_id) != len(candidate)
        or baseline_by_id.keys() != candidate_by_id.keys()
    ):
        raise RetrievalEvaluationError("paired query-embedding case IDs differ")
    rows: list[dict[str, str]] = []
    for case_id in sorted(baseline_by_id):
        left = baseline_by_id[case_id]
        right = candidate_by_id[case_id]
        if left.observation.variant != "baseline" or right.observation.variant != "candidate":
            raise RetrievalEvaluationError("paired query-embedding variants are invalid")
        if (
            left.query_embedding_protocol != _QUERY_EMBEDDING_PROTOCOL
            or right.query_embedding_protocol != _QUERY_EMBEDDING_PROTOCOL
            or left.question_sha256 != right.question_sha256
            or left.query_embedding_sha256 != right.query_embedding_sha256
        ):
            raise RetrievalEvaluationError("paired query-embedding evidence differs")
        rows.append(
            {
                "case_id": case_id,
                "question_sha256": left.question_sha256,
                "query_embedding_sha256": left.query_embedding_sha256,
            }
        )
    if evidence is not None and (
        len({row["question_sha256"] for row in rows}) != evidence.unique_question_count
        or _sha256_json(rows) != evidence.vector_manifest_sha256
    ):
        raise RetrievalEvaluationError("query-embedding manifest does not match case artifacts")


def evaluate_locked_decision(
    baseline: Sequence[CaseArtifact],
    candidate: Sequence[CaseArtifact],
    *,
    tuning_case_count: int,
    cluster_ids: Mapping[str, str],
    policy: RetrievalGatePolicy,
) -> LockedDecision:
    _assert_query_embedding_pairing(baseline, candidate)
    baseline_by_id = {item.case_id: item for item in baseline}
    candidate_by_id = {item.case_id: item for item in candidate}
    if baseline_by_id.keys() != candidate_by_id.keys():
        raise RetrievalEvaluationError("locked paired case IDs differ")
    failure_codes: list[str] = []
    decisions: list[MetricDecision] = []
    metric_names = (
        "recall_at_5",
        "recall_at_10",
        "mrr_at_10",
        "ndcg_at_10",
        "lexical_recall_at_5",
        "lexical_recall_at_50",
        "hybrid_union_recall_at_20",
    )
    for metric in metric_names:
        rows = []
        for case_id in sorted(baseline_by_id):
            left = baseline_by_id[case_id]
            if not left.answerable:
                continue
            right = candidate_by_id[case_id]
            rows.append(
                (
                    cluster_ids[case_id],
                    _locked_metric_values(left)[metric],
                    _locked_metric_values(right)[metric],
                )
            )
        baseline_mean, candidate_mean, improvement, ci_low, ci_high, clusters = _locked_bootstrap(
            rows,
            samples=policy.bootstrap_samples,
            seed=policy.bootstrap_seed
            ^ int.from_bytes(hashlib.sha256(f"locked:{metric}".encode()).digest()[:8], "big"),
            confidence_level=policy.confidence_level,
        )
        noninferiority = ci_low >= -policy.global_max_regression
        target = (
            policy.target_lexical_recall_at_5_gain
            if metric == "lexical_recall_at_5"
            else policy.target_ndcg_at_10_gain
            if metric == "ndcg_at_10"
            else None
        )
        target_passed = target is None or ci_low >= target
        passed = noninferiority and target_passed
        if not noninferiority:
            failure_codes.append(f"locked_regression:{metric}")
        if not target_passed:
            failure_codes.append(f"locked_target_gain:{metric}")
        decisions.append(
            MetricDecision(
                metric=cast(Any, metric),
                eligible_case_count=len(rows),
                cluster_count=clusters,
                baseline=baseline_mean,
                candidate=candidate_mean,
                improvement=improvement,
                ci_low=ci_low,
                ci_high=ci_high,
                noninferiority_passed=noninferiority,
                target_gain=target,
                target_passed=target_passed,
                passed=passed,
            )
        )
    no_answer_rows = [
        (
            cluster_ids[case_id],
            float(left.observation.abstained),
            float(candidate_by_id[case_id].observation.abstained),
        )
        for case_id, left in sorted(baseline_by_id.items())
        if not left.answerable
    ]
    if not no_answer_rows:
        raise RetrievalEvaluationError("locked set has no no-answer cases")
    baseline_rate, candidate_rate, improvement, ci_low, ci_high, clusters = _locked_bootstrap(
        no_answer_rows,
        samples=policy.bootstrap_samples,
        seed=policy.bootstrap_seed ^ int.from_bytes(hashlib.sha256(b"locked:no_answer").digest()[:8], "big"),
        confidence_level=policy.confidence_level,
    )
    no_answer_passed = ci_low >= -policy.global_max_regression
    if not no_answer_passed:
        failure_codes.append("locked_no_answer_abstention_regression")
    no_answer = LockedNoAnswerDecision(
        eligible_case_count=len(no_answer_rows),
        cluster_count=clusters,
        baseline_abstention_rate=baseline_rate,
        candidate_abstention_rate=candidate_rate,
        improvement=improvement,
        ci_low=ci_low,
        ci_high=ci_high,
        noninferiority_margin=policy.global_max_regression,
        passed=no_answer_passed,
    )
    return LockedDecision(
        case_count=len(baseline_by_id),
        tuning_case_count=tuning_case_count,
        metrics=tuple(decisions),
        no_answer=no_answer,
        accepted=not failure_codes,
        failure_codes=tuple(failure_codes),
    )


def _gate_case_result(
    artifact: CaseArtifact,
    *,
    cluster_id: str,
) -> RetrievalCaseResult:
    metrics: RetrievalCaseMetrics | None = None
    if artifact.answerable:
        final = artifact.observation.pools["final"].metrics
        sparse = artifact.observation.pools["sparse"].metrics
        hybrid = artifact.observation.pools["hybrid"].metrics

        def required(values: Mapping[str, float | None], key: str) -> float:
            value = values[key]
            if value is None:
                raise RetrievalEvaluationError("answerable gate metric is unexpectedly ineligible")
            return value

        metrics = RetrievalCaseMetrics(
            recall_at_5=required(final.recall, "5"),
            recall_at_10=required(final.recall, "10"),
            mrr_at_10=required(final.mrr, "10"),
            ndcg_at_10=required(final.ndcg, "10"),
            lexical_recall_at_5=required(sparse.recall, "5"),
            lexical_recall_at_50=required(sparse.recall, "50"),
            hybrid_union_recall_at_20=required(hybrid.recall, "20"),
        )
    return RetrievalCaseResult(
        case_id=artifact.case_id,
        gold_case_sha256=artifact.gold_case_sha256,
        reviewed=True,
        scope_id=artifact.scope_id,
        cluster_id=cluster_id,
        sparse_engine=cast(Any, artifact.observation.sparse_engine),
        language=artifact.language,
        content_types=cast(Any, artifact.content_types),
        answerable=artifact.answerable,
        metrics=metrics,
        returned_count=artifact.observation.returned_count,
        abstained=artifact.observation.abstained,
        retrieval_ms=artifact.observation.pools["final"].latency_ms,
    )


def _sparse_engine_case_evidence(
    artifacts: Sequence[CaseArtifact],
) -> tuple[SparseEngineCaseEvidence, ...]:
    values = [
        SparseEngineCaseEvidence(
            case_id=item.case_id,
            split=item.split,
            variant=item.observation.variant,
            config_sha256=item.observation.config_sha256,
            sparse_engine=cast(Any, item.observation.sparse_engine),
            artifact_sha256=hashlib.sha256(_canonical_bytes(item.model_dump(mode="json"))).hexdigest(),
        )
        for item in artifacts
    ]
    return tuple(
        sorted(
            values,
            key=lambda item: (
                item.case_id,
                item.split,
                item.variant,
                item.config_sha256,
            ),
        )
    )


def _gate_sparse_engine(
    backend: SparseBackend,
    *,
    policy: RetrievalGatePolicy,
    database: DatabaseEvidence,
) -> SparseEngineProvenance:
    if backend == "postgres_fts":
        indexes = tuple(
            SparseIndexDefinition(
                name=item.name,
                access_method="gin",
                canonical_definition=database.index_definitions[item.name],
                definition_sha256=database.index_definitions_sha256[item.name],
            )
            for item in policy.required_baseline_indexes
        )
        extension = None
    else:
        indexes = tuple(
            SparseIndexDefinition(
                name=item.name,
                access_method="bm25",
                text_config=item.text_config,
                k1=item.k1,
                b=item.b,
                canonical_definition=database.index_definitions[item.name],
                definition_sha256=database.index_definitions_sha256[item.name],
            )
            for item in policy.required_candidate_indexes
        )
        extension = PgTextsearchProvenance(
            version=database.extension_version,
            extension_commit=database.extension_commit,
            package_sha256=database.package_sha256,
            extension_binary_sha256=database.extension_binary_sha256,
            extension_binary_path="/usr/lib/postgresql/17/lib/pg_textsearch.so",
            extension_binary_bytes=database.extension_binary_bytes,
            container_image_digest=database.image_digest,
            base_postgres_image_digest=database.base_image_digest,
            build_recipe_sha256=database.build_recipe_sha256,
            prepare_sql_sha256=database.prepare_sql_sha256,
            legacy_fts_index_manifest_sha256=(database.baseline_index_manifest_sha256),
            spdx_license="PostgreSQL",
        )
    return SparseEngineProvenance(
        backend=backend,
        indexes=indexes,
        index_manifest_sha256=(
            database.baseline_index_manifest_sha256
            if backend == "postgres_fts"
            else database.candidate_index_manifest_sha256
        ),
        pg_textsearch=extension,
    )


def _gate_operations(
    evidence_paths: Mapping[str, Path],
    *,
    case_artifacts: Sequence[CaseArtifact],
    corpus_snapshot_sha256: str,
    policy: RetrievalGatePolicy,
) -> tuple[LoadEvidence, LoadEvidence, OperationalEvidence]:
    if set(evidence_paths) != {"operational", "load"}:
        raise RetrievalEvaluationError(
            "qualification requires consolidated operational and raw load evidence"
        )
    _, load_value = _read_external_evidence(evidence_paths["load"], expected_kind="load")
    if not isinstance(load_value, _LoadEvidence):
        raise RetrievalEvaluationError("load evidence type mismatch")
    _, operations = _parse_operational_evidence(
        evidence_paths["operational"],
        corpus_snapshot_sha256=corpus_snapshot_sha256,
        policy=policy,
    )
    if any(not item.observation.deterministic for item in case_artifacts):
        raise RetrievalEvaluationError("runner determinism evidence is inconsistent")
    return load_value.baseline, load_value.candidate, operations


def _gate_report(
    *,
    artifacts: Sequence[CaseArtifact],
    backend: SparseBackend,
    evaluated_at: datetime,
    policy: RetrievalGatePolicy,
    manifest: RunManifest,
    selected: RetrievalConfig,
    cluster_ids: Mapping[str, str],
    load: LoadEvidence,
    operations: OperationalEvidence,
) -> RetrievalReport:
    cases = tuple(
        _gate_case_result(item, cluster_id=cluster_ids[item.case_id])
        for item in sorted(artifacts, key=lambda value: value.case_id)
    )
    scope_provenance = tuple(
        OwnerScopeProvenance(**item.model_dump(mode="python")) for item in manifest.owner_scopes
    )
    models = RetrievalModelRevisions(
        embedding=RuntimeModelRevision(**manifest.model_revisions.embedding.model_dump()),
        reranker=RuntimeModelRevision(**manifest.model_revisions.reranker.model_dump()),
    )
    configuration = RetrievalConfiguration(
        dense_top_k=selected.dense_top_k,
        sparse_top_k=selected.sparse_top_k,
        rrf_k=selected.rrf_k,
        rerank_top_k=selected.rerank_top_k,
        final_top_k=selected.final_top_k,
        rerank_min_score=selected.rerank_min_score,
        embedding_dim=settings.embed_dim,
        visual_enabled=False,
    )
    provenance = RetrievalProvenance(
        repo_sha=manifest.repository_sha,
        git_dirty=False,
        gold_artifact_sha256=manifest.gold_sha256,
        sidecar_artifact_sha256=manifest.sidecar_sha256,
        corpus_snapshot_sha256=manifest.runtime_corpus_sha256,
        postgres_server_version_num=manifest.database.server_version_num,
        reviewed_case_count=len(cases),
        owner_scopes=scope_provenance,
        owner_scope_manifest_sha256=canonical_sha256(scope_provenance),
        models=models,
        configuration=configuration,
        configuration_sha256=canonical_sha256(configuration),
        sparse_engine=_gate_sparse_engine(
            backend,
            policy=policy,
            database=manifest.database,
        ),
    )
    case_manifest = [
        item.model_dump(
            mode="json",
            exclude={"metrics", "returned_count", "abstained", "retrieval_ms"},
        )
        for item in cases
    ]
    return RetrievalReport(
        evaluated_at=evaluated_at,
        provenance=provenance,
        case_count=len(cases),
        case_manifest_sha256=canonical_sha256(case_manifest),
        cases=cases,
        load=load,
        operations=operations,
    )


def _write_or_validate_model(path: Path, value: BaseModel) -> tuple[bytes, str]:
    raw = _canonical_bytes(value.model_dump(mode="json"))
    if path.exists():
        artifact = read_private_bytes(path, max_bytes=64 * 1024 * 1024)
        if artifact.raw_bytes != raw:
            raise RetrievalEvaluationError("existing gate artifact does not match this run")
        return artifact.raw_bytes, artifact.sha256
    write_private_json_fresh(path, raw)
    return raw, hashlib.sha256(raw).hexdigest()


def _evidence_paths_from_args(
    values: Sequence[str],
    *,
    mode: RunMode,
    operational_evidence: Path | None,
    load_evidence: Path | None,
    generated_load_evidence: Path | None,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for value in values:
        kind, separator, raw_path = value.partition("=")
        if not separator or kind not in _EVIDENCE_KINDS or not raw_path:
            raise RetrievalEvaluationError(
                "evidence must use one of rls|load|update|delete|restart=/absolute/path"
            )
        if kind in paths:
            raise RetrievalEvaluationError("external evidence kind is duplicated")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            raise RetrievalEvaluationError("external evidence path must be absolute")
        paths[kind] = path
    if operational_evidence is not None:
        if any(kind != "load" for kind in paths):
            raise RetrievalEvaluationError(
                "consolidated operational evidence cannot be mixed with split operation evidence"
            )
        paths["operational"] = operational_evidence.expanduser()
        if load_evidence is not None:
            paths["load"] = load_evidence.expanduser()
    elif load_evidence is not None:
        paths["load"] = load_evidence.expanduser()
    if any(not path.is_absolute() for path in paths.values()):
        raise RetrievalEvaluationError("external evidence path must be absolute")
    complete = set(paths) == {"operational", "load"}
    if generated_load_evidence is not None and "load" not in paths:
        complete = set(paths) == {"operational"}
    if mode == "qualification" and not complete:
        raise RetrievalEvaluationError(
            "qualification requires consolidated operational plus raw load evidence"
        )
    return paths


def _external_evidence_from_args(
    paths: Mapping[str, Path],
    *,
    corpus_snapshot_sha256: str,
    policy: RetrievalGatePolicy,
) -> tuple[EvidenceReference, ...]:
    references = [
        _parse_external_evidence(
            paths[kind],
            expected_kind=kind,
            corpus_snapshot_sha256=corpus_snapshot_sha256,
            policy=policy,
        )
        for kind in _EVIDENCE_KINDS
        if kind in paths
    ]
    if "operational" in paths:
        reference, _ = _parse_operational_evidence(
            paths["operational"],
            corpus_snapshot_sha256=corpus_snapshot_sha256,
            policy=policy,
        )
        references.append(reference)
    return tuple(sorted(references, key=lambda item: item.kind))


def _validate_load_evidence_binding(
    path: Path,
    *,
    config: RetrievalConfig,
    locked_case_ids: Sequence[str],
    corpus_snapshot_sha256: str,
    concurrency: int | None = None,
    requests_per_backend: int | None = None,
) -> _LoadEvidence:
    _, value = _read_external_evidence(path, expected_kind="load")
    if not isinstance(value, _LoadEvidence):
        raise RetrievalEvaluationError("load evidence type mismatch")
    raw_path = path.with_name(f"{path.stem}.raw.json")
    try:
        raw_artifact = read_private_json(
            raw_path,
            parser=lambda raw: RawLoadEvidence.model_validate_json(raw, strict=True),
            max_bytes=256 * 1024 * 1024,
        )
    except (PrivateArtifactError, ValueError):
        raise RetrievalEvaluationError("raw load evidence schema is invalid") from None
    raw = raw_artifact.value
    observed_peak = _validate_raw_load_evidence(
        raw,
        config=config,
        locked_case_ids=locked_case_ids,
        corpus_snapshot_sha256=corpus_snapshot_sha256,
        concurrency=concurrency,
        requests_per_backend=requests_per_backend,
    )
    expected_baseline = _aggregate_load_backend(
        raw,
        "postgres_fts",
        observed_peak_concurrency=observed_peak,
    )
    expected_candidate = _aggregate_load_backend(
        raw,
        "pg_textsearch",
        observed_peak_concurrency=observed_peak,
    )
    if (
        value.config_sha256 != config.fingerprint
        or value.locked_case_manifest_sha256 != _sha256_json(tuple(sorted(locked_case_ids)))
        or value.corpus_snapshot_sha256 != corpus_snapshot_sha256
        or value.raw_artifact_sha256 != raw_artifact.sha256
        or value.baseline != expected_baseline
        or value.candidate != expected_candidate
    ):
        raise RetrievalEvaluationError("load evidence does not bind this locked run")
    return value


def _manifest_path(work_dir: Path) -> Path:
    return work_dir / "manifest.json"


def _load_or_write_manifest(path: Path, expected: RunManifest) -> RunManifest:
    if path.exists():
        actual = read_private_json(
            path,
            parser=lambda raw: RunManifest.model_validate_json(raw, strict=True),
        ).value
        if actual.model_dump(mode="json", exclude={"created_at"}) != expected.model_dump(
            mode="json",
            exclude={"created_at"},
        ):
            raise RetrievalEvaluationError("resume manifest does not match current inputs")
        return actual
    write_private_json_fresh(path, _canonical_bytes(expected.model_dump(mode="json")))
    return expected


def _load_or_write_report(path: Path, expected: FinalReport) -> tuple[FinalReport, bytes]:
    raw = _canonical_bytes(expected.model_dump(mode="json"))
    if path.exists():
        artifact = read_private_json(
            path,
            parser=lambda value: FinalReport.model_validate_json(value, strict=True),
        )
        if artifact.value.model_dump(mode="json", exclude={"completed_at"}) != expected.model_dump(
            mode="json",
            exclude={"completed_at"},
        ):
            raise RetrievalEvaluationError("existing final report does not match completed run")
        return artifact.value, artifact.raw_bytes
    write_private_json_fresh(path, raw)
    return expected, raw


def _assert_output_inside_work_dir(path: Path, work_dir: Path) -> None:
    try:
        path.parent.resolve(strict=True).relative_to(work_dir.resolve(strict=True))
    except (OSError, ValueError):
        raise RetrievalEvaluationError("private outputs must stay inside the work directory") from None


async def run(args: argparse.Namespace) -> FinalReport | None:
    mode = cast(RunMode, args.mode)
    work_dir = _private_dir(args.work_dir.expanduser())
    output = args.output.expanduser() if args.output else work_dir / "report.json"
    attestation_output = (
        args.attestation_output.expanduser()
        if args.attestation_output
        else work_dir / "report.attestation.json"
    )
    _assert_output_inside_work_dir(output, work_dir)
    _assert_output_inside_work_dir(attestation_output, work_dir)
    if args.stop_after_load_evidence and args.generate_load_evidence is None:
        raise RetrievalEvaluationError("--stop-after-load-evidence requires --generate-load-evidence")

    gold_artifact = read_private_bytes(args.gold, max_bytes=256 * 1024 * 1024)
    sidecar_artifact = read_private_bytes(args.sidecar, max_bytes=256 * 1024 * 1024)
    records, _ = parse_gold_set_bytes(gold_artifact.raw_bytes, mode="release")
    sidecars = bind_gold_sidecar(
        records,
        parse_private_sidecar_bytes(sidecar_artifact.raw_bytes),
    )
    bindings = build_case_bindings(records, sidecars)
    policy, _, policy_sha256, policy_source = _load_policy(
        args.policy,
        require_tracked=mode == "qualification",
    )
    repository_sha, repository_dirty = _git_state()
    if mode == "qualification" and repository_dirty:
        raise RetrievalEvaluationError("qualification requires a clean Git repository")
    if mode == "qualification":
        if args.hmac_key is None:
            raise RetrievalEvaluationError("qualification requires an HMAC key")
        case_hmac_key = load_hmac_key(args.hmac_key, REPOSITORY_ROOT)
        if gold_artifact.sha256 != policy.gold_artifact_sha256:
            raise RetrievalEvaluationError("Gold artifact does not match qualification policy")
        if sidecar_artifact.sha256 != policy.sidecar_artifact_sha256:
            raise RetrievalEvaluationError("sidecar artifact does not match qualification policy")
        if len(records) != policy.expected_case_count:
            raise RetrievalEvaluationError("reviewed case count does not match policy")
        if _retrieval_no_answer_count(records) != policy.expected_no_answer_case_count:
            raise RetrievalEvaluationError("no-answer case count does not match policy")
    else:
        case_hmac_key = None
    if settings.visual_enabled:
        raise RetrievalEvaluationError("sparse retrieval qualification requires visual retrieval off")
    require_loopback_url(settings.embed_base_url, name="embedding endpoint")
    require_loopback_url(settings.rerank_base_url, name="reranker endpoint")
    require_loopback_endpoint(settings.s3_endpoint, name="MinIO endpoint")

    api_database_url = require_loopback_database_url(args.database_url)
    provenance_database_url = require_loopback_database_url(args.provenance_database_url)
    api_engine = create_async_engine(
        api_database_url,
        pool_pre_ping=True,
        connect_args={
            "server_settings": {
                "application_name": "rag_bm25_qualification_api_rls",
                "default_transaction_read_only": "on",
            }
        },
    )
    provenance_engine = create_async_engine(
        provenance_database_url,
        pool_pre_ping=True,
        connect_args={
            "server_settings": {
                "application_name": "rag_bm25_qualification_provenance",
                "default_transaction_read_only": "on",
            }
        },
    )
    embedder = Embedder()
    try:
        await assert_api_rls_role(api_engine, required=True)
        api_sessions = create_sessionmaker(api_engine)
        provenance_sessions = create_sessionmaker(provenance_engine)
        verifier = _CorpusVerifier(provenance_sessions, Storage())
        runtime_corpus_sha256, scopes = await verifier.verify(records, sidecars)
        if len(scopes) < 2:
            raise RetrievalEvaluationError("at least two independently owned scopes are required")
        if mode == "qualification":
            if runtime_corpus_sha256 != policy.corpus_snapshot_sha256:
                raise RetrievalEvaluationError("runtime corpus does not match qualification policy")
            if len(scopes) < policy.min_owner_scope_count:
                raise RetrievalEvaluationError("owner-scope coverage is below policy")

        database_runtime = _runtime_database_evidence(
            container=args.database_container,
            extension_binary_path=args.extension_binary_path,
            policy=policy,
        )
        database = await _database_evidence(
            api_engine,
            policy=policy,
            runtime=database_runtime,
        )
        embedding_revision, embedding_artifact_sha256 = _load_model_revision(
            args.embedding_revision_evidence,
            expected_model=settings.embed_model,
        )
        reranker_revision, reranker_artifact_sha256 = _load_model_revision(
            args.reranker_revision_evidence,
            expected_model=settings.rerank_model,
        )
        models = ModelRevisionEvidence(
            embedding=embedding_revision,
            reranker=reranker_revision,
            embedding_artifact_sha256=embedding_artifact_sha256,
            reranker_artifact_sha256=reranker_artifact_sha256,
        )
        evidence_paths = _evidence_paths_from_args(
            args.evidence,
            mode=mode,
            operational_evidence=args.operational_evidence,
            load_evidence=args.load_evidence,
            generated_load_evidence=args.generate_load_evidence,
        )
        _external_evidence_from_args(
            evidence_paths,
            corpus_snapshot_sha256=runtime_corpus_sha256,
            policy=policy,
        )
        split = stratified_cluster_split(
            records,
            seed=args.split_seed,
            locked_fraction=args.locked_fraction,
        )
        if mode == "qualification" and (len(split.locked_case_ids) < 200 or len(split.tuning_case_ids) > 36):
            raise RetrievalEvaluationError(
                "qualification requires at least 200 locked cases and at most 36 tuning cases"
            )
        control = RetrievalConfig(
            dense_top_k=settings.rag_dense_top_k,
            sparse_top_k=settings.rag_sparse_top_k,
            rrf_k=settings.rag_rrf_k,
            rerank_top_k=settings.rag_rerank_top_k,
            rerank_min_score=settings.rag_rerank_min_score,
            final_top_k=max(10, settings.rag_context_top_k),
        )
        sweep = sweep_configs(
            control,
            sparse_top_k=args.sparse_top_k,
            rrf_k=args.rrf_k,
            rerank_min_score=args.rerank_min_score,
        )
        repeat_count = args.repeats
        if mode == "qualification" and repeat_count < policy.min_determinism_replays:
            raise RetrievalEvaluationError("determinism replay count is below policy")
        paired_embedder = _PairedQueryEmbedder(embedder)
        await paired_embedder.preload([record.question for record in records])
        query_embedding = paired_embedder.evidence(records, embedding_revision)
        manifest_identity = {
            "mode": mode,
            "repository_sha": repository_sha,
            "repository_dirty": repository_dirty,
            "gold_sha256": gold_artifact.sha256,
            "sidecar_sha256": sidecar_artifact.sha256,
            "gold_corpus_sha256": _gold_corpus_fingerprint(records),
            "runtime_corpus_sha256": runtime_corpus_sha256,
            "policy_sha256": policy_sha256,
            "owner_scopes": [scopes[key].evidence.model_dump(mode="json") for key in sorted(scopes)],
            "split": split.model_dump(mode="json"),
            "control_config": control.model_dump(mode="json"),
            "sweep_configs": [item.model_dump(mode="json") for item in sweep],
            "model_revisions": models.model_dump(mode="json"),
            "database": database.model_dump(mode="json"),
            "query_embedding": query_embedding.model_dump(mode="json"),
            "load_embedding_protocol": _LOAD_EMBEDDING_PROTOCOL,
            "repeat_count": repeat_count,
        }
        run_id = _sha256_json(manifest_identity)
        retriever = Retriever(paired_embedder, Reranker())
        load_retriever = Retriever(embedder, Reranker())
        tuning_set = set(split.tuning_case_ids)
        locked_set = set(split.locked_case_ids)
        tuning_cases: dict[str, list[CaseArtifact]] = {}
        for config in sweep:
            observations = []
            for case_id in sorted(tuning_set):
                binding = bindings[case_id]
                observations.append(
                    await _run_or_resume_case(
                        retriever=retriever,
                        sessionmaker=api_sessions,
                        work_dir=work_dir,
                        binding=binding,
                        scope=scopes[binding.record.scope_id],
                        config=config,
                        backend="pg_textsearch",
                        variant="candidate",
                        split="tuning",
                        run_id=run_id,
                        repeat_count=repeat_count,
                        query_embedding_sha256=paired_embedder.vector_sha256(
                            binding.record.question
                        ),
                        hmac_key=case_hmac_key,
                    )
                )
            tuning_cases[config.fingerprint] = observations
        tuning_results = {
            fingerprint: aggregate_all_pools(cases) for fingerprint, cases in sorted(tuning_cases.items())
        }
        selected_fingerprint = select_tuning_config(
            {fingerprint: metrics["final"] for fingerprint, metrics in tuning_results.items()}
        )
        selected = next(config for config in sweep if config.fingerprint == selected_fingerprint)
        locked_cases: dict[VariantName, list[CaseArtifact]] = {
            "baseline": [],
            "candidate": [],
        }
        for case_id in sorted(locked_set):
            binding = bindings[case_id]
            variants: list[tuple[VariantName, SparseBackend]] = [
                ("baseline", "postgres_fts"),
                ("candidate", "pg_textsearch"),
            ]
            if int(hashlib.sha256(case_id.encode()).hexdigest(), 16) % 2:
                variants.reverse()
            for variant, backend in variants:
                locked_cases[variant].append(
                    await _run_or_resume_case(
                        retriever=retriever,
                        sessionmaker=api_sessions,
                        work_dir=work_dir,
                        binding=binding,
                        scope=scopes[binding.record.scope_id],
                        config=selected,
                        backend=backend,
                        variant=variant,
                        split="locked",
                        run_id=run_id,
                        repeat_count=repeat_count,
                        query_embedding_sha256=paired_embedder.vector_sha256(
                            binding.record.question
                        ),
                        hmac_key=case_hmac_key,
                    )
                )

        baseline_tuning: list[CaseArtifact] = []
        if mode == "qualification":
            for case_id in sorted(tuning_set):
                binding = bindings[case_id]
                baseline_tuning.append(
                    await _run_or_resume_case(
                        retriever=retriever,
                        sessionmaker=api_sessions,
                        work_dir=work_dir,
                        binding=binding,
                        scope=scopes[binding.record.scope_id],
                        config=selected,
                        backend="postgres_fts",
                        variant="baseline",
                        split="tuning",
                        run_id=run_id,
                        repeat_count=repeat_count,
                        query_embedding_sha256=paired_embedder.vector_sha256(
                            binding.record.question
                        ),
                        hmac_key=case_hmac_key,
                    )
                )

        if args.generate_load_evidence is not None:
            generated_load_path = args.generate_load_evidence.expanduser()
            if not generated_load_path.is_absolute():
                raise RetrievalEvaluationError("generated load evidence path must be absolute")
            _assert_output_inside_work_dir(generated_load_path, work_dir)
            if "load" in evidence_paths and evidence_paths["load"] != generated_load_path:
                raise RetrievalEvaluationError("generated and ingested load evidence are mutually exclusive")
            if mode == "qualification" and (
                args.load_concurrency < policy.min_load_concurrency
                or args.load_requests_per_backend < policy.min_load_requests
            ):
                raise RetrievalEvaluationError("generated load parameters are below policy")
            await generate_load_evidence(
                output=generated_load_path,
                retriever=load_retriever,
                sessionmaker=api_sessions,
                bindings=bindings,
                scopes=scopes,
                locked_case_ids=split.locked_case_ids,
                config=selected,
                corpus_snapshot_sha256=runtime_corpus_sha256,
                concurrency=args.load_concurrency,
                requests_per_backend=args.load_requests_per_backend,
            )
            evidence_paths["load"] = generated_load_path
        evidence = _external_evidence_from_args(
            evidence_paths,
            corpus_snapshot_sha256=runtime_corpus_sha256,
            policy=policy,
        )
        if mode == "qualification" and not any(item.kind == "load" for item in evidence):
            raise RetrievalEvaluationError("qualification load evidence is missing")
        if "load" in evidence_paths:
            _validate_load_evidence_binding(
                evidence_paths["load"],
                config=selected,
                locked_case_ids=split.locked_case_ids,
                corpus_snapshot_sha256=runtime_corpus_sha256,
            )
        manifest = _load_or_write_manifest(
            _manifest_path(work_dir),
            RunManifest(
                run_id=run_id,
                mode=mode,
                repository_sha=repository_sha,
                repository_dirty=repository_dirty,
                gold_sha256=gold_artifact.sha256,
                sidecar_sha256=sidecar_artifact.sha256,
                gold_corpus_sha256=_gold_corpus_fingerprint(records),
                runtime_corpus_sha256=runtime_corpus_sha256,
                policy_sha256=policy_sha256,
                owner_scopes=tuple(scopes[key].evidence for key in sorted(scopes)),
                split=split,
                control_config=control,
                sweep_configs=sweep,
                model_revisions=models,
                database=database,
                external_evidence=evidence,
                query_embedding=query_embedding,
                load_embedding_protocol=_LOAD_EMBEDDING_PROTOCOL,
                repeat_count=repeat_count,
                created_at=datetime.now(UTC),
            ),
        )
        if args.stop_after_load_evidence:
            runtime_after_load, scopes_after_load = await verifier.verify(records, sidecars)
            if runtime_after_load != runtime_corpus_sha256 or {
                key: value.evidence for key, value in scopes_after_load.items()
            } != {key: value.evidence for key, value in scopes.items()}:
                raise RetrievalEvaluationError("runtime corpus changed during load generation")
            return None
        cluster_ids = _statistical_cluster_ids(bindings)
        locked_decision = evaluate_locked_decision(
            locked_cases["baseline"],
            locked_cases["candidate"],
            tuning_case_count=len(tuning_set),
            cluster_ids=cluster_ids,
            policy=policy,
        )
        baseline_gate_report_sha256: str | None = None
        candidate_gate_report_sha256: str | None = None
        gate_decision: RetrievalGateDecision | None = None
        report_engine_artifacts = [
            *locked_cases["baseline"],
            *locked_cases["candidate"],
            *tuning_cases[selected_fingerprint],
        ]
        if mode == "qualification":
            all_baseline = [*baseline_tuning, *locked_cases["baseline"]]
            all_candidate = [
                *tuning_cases[selected_fingerprint],
                *locked_cases["candidate"],
            ]
            _assert_query_embedding_pairing(
                all_baseline,
                all_candidate,
                evidence=query_embedding,
            )
            report_engine_artifacts = [*all_baseline, *all_candidate]
            baseline_load, candidate_load, operations = _gate_operations(
                evidence_paths,
                case_artifacts=all_candidate,
                corpus_snapshot_sha256=runtime_corpus_sha256,
                policy=policy,
            )
            evaluated_at = datetime.now(UTC)
            baseline_gate = _gate_report(
                artifacts=all_baseline,
                backend="postgres_fts",
                evaluated_at=evaluated_at,
                policy=policy,
                manifest=manifest,
                selected=selected,
                cluster_ids=cluster_ids,
                load=baseline_load,
                operations=operations,
            )
            candidate_gate = _gate_report(
                artifacts=all_candidate,
                backend="pg_textsearch",
                evaluated_at=evaluated_at,
                policy=policy,
                manifest=manifest,
                selected=selected,
                cluster_ids=cluster_ids,
                load=candidate_load,
                operations=operations,
            )
            gate_decision = evaluate_retrieval_gate(
                baseline_gate,
                candidate_gate,
                policy,
                evaluated_at=evaluated_at,
            )
            gate_dir = _private_dir(work_dir / "gate")
            _, baseline_artifact_sha256 = _write_or_validate_model(
                gate_dir / "baseline.json",
                baseline_gate,
            )
            _, candidate_artifact_sha256 = _write_or_validate_model(
                gate_dir / "candidate.json",
                candidate_gate,
            )
            _write_or_validate_model(gate_dir / "decision.json", gate_decision)
            if (
                baseline_artifact_sha256
                != hashlib.sha256(_canonical_bytes(baseline_gate.model_dump(mode="json"))).hexdigest()
                or candidate_artifact_sha256
                != hashlib.sha256(_canonical_bytes(candidate_gate.model_dump(mode="json"))).hexdigest()
            ):
                raise RetrievalEvaluationError("gate report artifact hash mismatch")
            baseline_gate_report_sha256 = gate_decision.baseline_report_sha256
            candidate_gate_report_sha256 = gate_decision.candidate_report_sha256

        runtime_after, scopes_after = await verifier.verify(records, sidecars)
        if runtime_after != runtime_corpus_sha256 or {
            key: value.evidence for key, value in scopes_after.items()
        } != {key: value.evidence for key, value in scopes.items()}:
            raise RetrievalEvaluationError("runtime corpus changed during the paired evaluation")
        manifest_sha256 = _sha256_json(manifest.model_dump(mode="json"))
        report = FinalReport(
            run_id=run_id,
            manifest_sha256=manifest_sha256,
            policy_sha256=policy_sha256,
            selected_candidate_config_sha256=selected_fingerprint,
            query_embedding=query_embedding,
            load_embedding_protocol=_LOAD_EMBEDDING_PROTOCOL,
            tuning_results=tuning_results,
            locked_results={variant: aggregate_all_pools(cases) for variant, cases in locked_cases.items()},
            locked_slices={variant: aggregate_slices(cases) for variant, cases in locked_cases.items()},
            baseline_gate_report_sha256=baseline_gate_report_sha256,
            candidate_gate_report_sha256=candidate_gate_report_sha256,
            gate_decision=gate_decision,
            locked_decision=locked_decision,
            release_accepted=(
                locked_decision.accepted and gate_decision.accepted if gate_decision is not None else None
            ),
            deterministic=True,
            sparse_engine_evidence=_sparse_engine_case_evidence(report_engine_artifacts),
            runtime_corpus_sha256_after=runtime_after,
            completed_at=datetime.now(UTC),
        )
        report, report_bytes = _load_or_write_report(output, report)
        if mode == "qualification":
            if case_hmac_key is None or policy_source is None:
                raise RetrievalEvaluationError("qualification attestation inputs are missing")
            if attestation_output.exists():
                artifact = read_private_bytes(attestation_output, max_bytes=8 * 1024 * 1024)
                attestation = load_private_artifact_attestation(artifact.raw_bytes)
                verify_private_artifact_attestation(
                    attestation,
                    artifact_bytes=report_bytes,
                    expected_artifact_type="rag-retrieval-bm25-report-v2",
                    key=case_hmac_key,
                    repository_root=REPOSITORY_ROOT,
                )
            else:
                attestation = create_private_artifact_attestation(
                    artifact_bytes=report_bytes,
                    artifact_type="rag-retrieval-bm25-report-v2",
                    key=case_hmac_key,
                    repository_root=REPOSITORY_ROOT,
                    source_paths=(
                        policy_source,
                        "deploy/postgres-bm25/Dockerfile",
                        "deploy/postgres-bm25/prepare_candidate.sql",
                        "pyproject.toml",
                        "scripts/evaluate_retrieval_bm25.py",
                        "src/rag_app/db/rls.py",
                        "src/rag_app/eval/gold_set.py",
                        "src/rag_app/eval/private_artifacts.py",
                        "src/rag_app/eval/private_sidecar.py",
                        "src/rag_app/eval/report_attestation.py",
                        "src/rag_app/eval/retrieval_gate.py",
                        "src/rag_app/llm/embeddings.py",
                        "src/rag_app/rag/retrieve.py",
                        "src/rag_app/storage/s3.py",
                        "uv.lock",
                    ),
                )
                atomic_write_private_artifact_attestation(attestation_output, attestation)
        return report
    finally:
        await embedder.client.close()
        await api_engine.dispose()
        await provenance_engine.dispose()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--attestation-output", type=Path)
    parser.add_argument("--mode", choices=("dev", "qualification"), default="dev")
    parser.add_argument("--database-url", default=settings.database_url)
    parser.add_argument("--provenance-database-url", default=settings.database_url)
    parser.add_argument("--database-container", required=True)
    parser.add_argument("--extension-binary-path", type=Path, required=True)
    parser.add_argument("--embedding-revision-evidence", type=Path, required=True)
    parser.add_argument("--reranker-revision-evidence", type=Path, required=True)
    parser.add_argument("--hmac-key", type=Path)
    parser.add_argument("--evidence", action="append", default=[])
    parser.add_argument("--operational-evidence", type=Path)
    parser.add_argument("--load-evidence", type=Path)
    parser.add_argument("--generate-load-evidence", type=Path)
    parser.add_argument("--stop-after-load-evidence", action="store_true")
    parser.add_argument("--load-concurrency", type=int, default=10)
    parser.add_argument("--load-requests-per-backend", type=int, default=200)
    parser.add_argument("--split-seed", type=int, default=2026071409)
    parser.add_argument("--locked-fraction", type=float, default=0.85)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sparse-top-k", type=int, action="append", default=[50, 100])
    parser.add_argument("--rrf-k", type=int, action="append", default=[30, 60])
    parser.add_argument(
        "--rerank-min-score",
        type=float,
        action="append",
        default=[0.05, 0.10],
    )
    return parser


def _result_exit_code(report: FinalReport, mode: RunMode) -> int:
    if mode == "qualification" and report.release_accepted is not True:
        return 2
    return 0


def main() -> int:
    args = _parser().parse_args()
    try:
        report = asyncio.run(run(args))
    except (RetrievalEvaluationError, PrivateArtifactFormatError, ValueError, RuntimeError) as error:
        raise SystemExit(f"retrieval BM25 evaluation failed: {error}") from None
    if report is None:
        print(json.dumps({"load_evidence_generated": True}, sort_keys=True))
        return 0
    print(
        json.dumps(
            {
                "run_id": report.run_id,
                "selected_candidate_config_sha256": report.selected_candidate_config_sha256,
                "deterministic": report.deterministic,
                "release_accepted": report.release_accepted,
            },
            sort_keys=True,
        )
    )
    return _result_exit_code(report, cast(RunMode, args.mode))


if __name__ == "__main__":
    raise SystemExit(main())
