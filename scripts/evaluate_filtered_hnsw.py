#!/usr/bin/env python3
"""Fail-closed filtered-HNSW production preflight.

The preflight may reject a candidate, but it never authorizes a production
cutover. A corpus where PostgreSQL naturally selects both HNSW indexes must
still pass the full paired quality and concurrency qualification.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import subprocess
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import text as sql
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from rag_app.config import settings
from rag_app.db.engine import create_engine, create_sessionmaker
from rag_app.db.rls import reset_principal, set_principal
from rag_app.eval.gold_set import bytes_sha256, load_gold_set
from rag_app.eval.private_sidecar import PrivateSidecarRecord, bind_gold_sidecar, load_private_sidecar
from rag_app.eval.report_attestation import (
    ReportAttestationError,
    atomic_write_private_artifact_attestation,
    create_private_artifact_attestation,
    load_hmac_key,
)
from rag_app.rag.retrieve import configure_hnsw_transaction, dense_query_plan

type ScopeKind = Literal["document", "document_ids", "folder", "owner"]
type RunMode = Literal["debug", "qualification"]
type PreflightDecision = Literal["no_go", "requires_full_ab"]

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "filtered-hnsw-preflight-v2"
EXPECTED_RELEASE_CASES = 236
EXPECTED_PGVECTOR_VERSION = "0.8.5"
EXPECTED_DATABASE_IMAGE_DIGEST = (
    "sha256:d2ef61f42ef767baa5a1475393303cc235bcd92febd9d7014eddb48b41f3bad0"
)
REQUIRED_HNSW_INDEXES = frozenset({"ix_chunks_emb_en", "ix_chunks_emb_ru"})
ATTESTED_SOURCES = (
    "alembic/versions/0026_pgvector_085.py",
    "docker-compose.yml",
    "scripts/evaluate_filtered_hnsw.py",
    "src/rag_app/config.py",
    "src/rag_app/db/models.py",
    "src/rag_app/rag/retrieve.py",
    "uv.lock",
)


class QualificationError(RuntimeError):
    """The run cannot produce trustworthy preflight evidence."""


@dataclass(frozen=True, slots=True)
class DocumentMeta:
    owner_sub: str | None
    folder_id: uuid.UUID | None
    status: str


@dataclass(frozen=True, slots=True)
class Scope:
    kind: ScopeKind
    owner_sub: str
    params: dict[str, Any]

    @property
    def ref(self) -> str:
        payload = {
            "kind": self.kind,
            "owner_ref": _sha256_text(self.owner_sub),
            "doc_id": str(self.params["doc_id"]) if self.params["doc_id"] else None,
            "doc_ids": sorted(str(value) for value in self.params["doc_ids"] or ()),
            "folder_id": str(self.params["folder_id"]) if self.params["folder_id"] else None,
        }
        return f"scope-sha256:{_sha256_json(payload)}"


@dataclass(frozen=True, slots=True)
class PlanObservation:
    scope_ref: str
    scope_kind: ScopeKind
    natural_indexes: frozenset[str]
    forced_indexes: frozenset[str]
    exact_indexes: frozenset[str]


@dataclass(frozen=True, slots=True)
class BenchmarkObservation:
    scope_ref: str
    scope_kind: ScopeKind
    exact_latency_ms: float
    off_latency_ms: float
    strict_latency_ms: float
    off_oracle_recall: float
    strict_oracle_recall: float
    off_filled: bool
    strict_filled: bool
    scope_violation_count: int


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _sha256_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _git_state() -> tuple[str, bool]:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return sha, dirty


def _scope_params(
    *,
    doc_id: uuid.UUID | None = None,
    doc_ids: Sequence[uuid.UUID] | None = None,
    folder_id: uuid.UUID | None = None,
    owner: str,
) -> dict[str, Any]:
    return {
        "doc_id": doc_id,
        "doc_ids": list(doc_ids) if doc_ids else None,
        "folder_id": folder_id,
        "owner": owner,
    }


def _case_scopes(
    sidecar: PrivateSidecarRecord,
    metadata: Mapping[uuid.UUID, DocumentMeta],
) -> tuple[Scope, ...]:
    doc_ids = tuple(item.document_id for item in sidecar.source_documents)
    rows = [metadata[item] for item in doc_ids]
    owners = {item.owner_sub for item in rows}
    if len(owners) != 1 or None in owners:
        raise QualificationError("Gold scope does not have exactly one authenticated owner")
    owner = next(iter(owners))
    assert owner is not None
    scopes = [Scope("document_ids", owner, _scope_params(doc_ids=doc_ids, owner=owner))]
    if len(doc_ids) == 1:
        scopes.append(Scope("document", owner, _scope_params(doc_id=doc_ids[0], owner=owner)))
    folders = {item.folder_id for item in rows}
    if len(folders) == 1 and None not in folders:
        folder_id = next(iter(folders))
        assert folder_id is not None
        scopes.append(Scope("folder", owner, _scope_params(folder_id=folder_id, owner=owner)))
    scopes.append(Scope("owner", owner, _scope_params(owner=owner)))
    return tuple(scopes)


def _unique_scopes(
    sidecars: Sequence[PrivateSidecarRecord],
    metadata: Mapping[uuid.UUID, DocumentMeta],
) -> tuple[Scope, ...]:
    by_ref: dict[str, Scope] = {}
    for sidecar in sidecars:
        for scope in _case_scopes(sidecar, metadata):
            by_ref.setdefault(scope.ref, scope)
    return tuple(by_ref[key] for key in sorted(by_ref))


def _index_names(plan: Any) -> frozenset[str]:
    names: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            name = value.get("Index Name")
            if isinstance(name, str):
                names.add(name)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(plan)
    return frozenset(names)


async def _explain_indexes(
    sessionmaker: async_sessionmaker[AsyncSession],
    *,
    scope: Scope,
    query_vector: str,
    backend: Literal["exact", "hnsw"],
    force_hnsw: bool,
    k: int,
) -> frozenset[str]:
    token = set_principal(scope.owner_sub, False)
    try:
        async with sessionmaker() as session:
            if backend == "hnsw":
                await configure_hnsw_transaction(
                    session,
                    iterative_scan="strict_order",
                    ef_search=100,
                    max_scan_tuples=20_000,
                    scan_mem_multiplier=2.0,
                )
            if force_hnsw:
                await session.execute(sql("SET LOCAL enable_seqscan = off"))
            statement = "EXPLAIN (FORMAT JSON) " + dense_query_plan(backend).statement
            plan = (
                await session.execute(
                    sql(statement),
                    {**scope.params, "qe": query_vector, "k": k},
                )
            ).scalar_one()
    finally:
        reset_principal(token)
    return _index_names(plan)


async def _plan_observation(
    sessionmaker: async_sessionmaker[AsyncSession],
    scope: Scope,
    *,
    query_vector: str,
    k: int,
) -> PlanObservation:
    natural = await _explain_indexes(
        sessionmaker,
        scope=scope,
        query_vector=query_vector,
        backend="hnsw",
        force_hnsw=False,
        k=k,
    )
    forced = await _explain_indexes(
        sessionmaker,
        scope=scope,
        query_vector=query_vector,
        backend="hnsw",
        force_hnsw=True,
        k=k,
    )
    exact = await _explain_indexes(
        sessionmaker,
        scope=scope,
        query_vector=query_vector,
        backend="exact",
        force_hnsw=False,
        k=k,
    )
    return PlanObservation(scope.ref, scope.kind, natural, forced, exact)


async def _load_runtime_state(
    sessionmaker: async_sessionmaker[AsyncSession],
    document_ids: Sequence[uuid.UUID],
) -> tuple[dict[uuid.UUID, DocumentMeta], dict[str, Any], dict[str, int], str]:
    token = set_principal("filtered-hnsw-preflight-admin", True)
    try:
        async with sessionmaker() as session:
            document_rows = (
                await session.execute(
                    sql(
                        "SELECT id, owner_sub, folder_id, status::text AS status FROM documents"
                    )
                )
            ).all()
            chunk_rows = (
                await session.execute(
                    sql(
                        "SELECT c.id, c.document_id, d.owner_sub, d.folder_id, d.status::text AS status, "
                        "c.emb_en::text AS emb_en, c.emb_ru::text AS emb_ru, "
                        "(c.emb_en IS NOT NULL) AS has_en, (c.emb_ru IS NOT NULL) AS has_ru "
                        "FROM chunks c JOIN documents d ON d.id=c.document_id ORDER BY c.id"
                    )
                )
            ).all()
            database = (
                await session.execute(
                    sql(
                        "SELECT current_setting('server_version_num')::int AS server_version_num, "
                        "(SELECT extversion FROM pg_extension WHERE extname='vector') AS vector_version, "
                        "current_setting('hnsw.ef_search')::int AS hnsw_ef_search, "
                        "current_setting('hnsw.iterative_scan') AS hnsw_iterative_scan, "
                        "current_setting('hnsw.max_scan_tuples')::int AS hnsw_max_scan_tuples, "
                        "current_setting('hnsw.scan_mem_multiplier')::float8 AS hnsw_scan_mem_multiplier, "
                        "(SELECT version_num FROM alembic_version LIMIT 1) AS alembic_revision, "
                        "format_type((SELECT atttypid FROM pg_attribute WHERE attrelid='chunks'::regclass "
                        "AND attname='emb_en'), (SELECT atttypmod FROM pg_attribute "
                        "WHERE attrelid='chunks'::regclass AND attname='emb_en')) AS emb_en_type, "
                        "format_type((SELECT atttypid FROM pg_attribute WHERE attrelid='chunks'::regclass "
                        "AND attname='emb_ru'), (SELECT atttypmod FROM pg_attribute "
                        "WHERE attrelid='chunks'::regclass AND attname='emb_ru')) AS emb_ru_type, "
                        "row_security_active('chunks'::regclass) AS rls_active, "
                        "(SELECT rolbypassrls FROM pg_roles WHERE rolname=current_user) AS bypass_rls, "
                        "pg_has_role(current_user, (SELECT relowner FROM pg_class "
                        "WHERE oid='chunks'::regclass), 'member') AS owns_chunks"
                    )
                )
            ).one()
            index_rows = (
                await session.execute(
                    sql(
                        "SELECT ci.relname AS name, i.indisvalid AS valid, i.indisready AS ready, "
                        "pg_get_indexdef(i.indexrelid) AS definition "
                        "FROM pg_index i JOIN pg_class ci ON ci.oid=i.indexrelid "
                        "WHERE ci.relname = ANY(CAST(:names AS text[])) ORDER BY ci.relname"
                    ),
                    {"names": sorted(REQUIRED_HNSW_INDEXES)},
                )
            ).all()
    finally:
        reset_principal(token)
    metadata = {
        row.id: DocumentMeta(row.owner_sub, row.folder_id, row.status) for row in document_rows
    }
    if not set(document_ids).issubset(metadata):
        raise QualificationError("Gold sidecar documents do not match the runtime database")
    owner_counts: dict[str, int] = {}
    manifest_rows: list[dict[str, Any]] = []
    for row in chunk_rows:
        if row.owner_sub is not None:
            owner_counts[row.owner_sub] = owner_counts.get(row.owner_sub, 0) + 1
        manifest_rows.append(
            {
                "chunk_id": str(row.id),
                "document_id": str(row.document_id),
                "owner_ref": _sha256_text(row.owner_sub or ""),
                "folder_ref": _sha256_text(str(row.folder_id) if row.folder_id else ""),
                "status": row.status,
                "has_en": bool(row.has_en),
                "has_ru": bool(row.has_ru),
                "emb_en_sha256": _sha256_text(row.emb_en or ""),
                "emb_ru_sha256": _sha256_text(row.emb_ru or ""),
            }
        )
    evidence = {
        "server_version_num": int(database.server_version_num),
        "pgvector_version": str(database.vector_version),
        "chunk_count": len(chunk_rows),
        "embedded_chunk_count": sum(row.has_en or row.has_ru for row in chunk_rows),
        "owner_with_chunks_count": len(owner_counts),
        "alembic_revision": str(database.alembic_revision),
        "emb_en_type": str(database.emb_en_type),
        "emb_ru_type": str(database.emb_ru_type),
        "embedding_values_finite": all(
            marker not in (row.emb_en or "").lower() and marker not in (row.emb_ru or "").lower()
            for row in chunk_rows
            for marker in ("nan", "infinity", "-infinity")
        ),
        "hnsw_ef_search": int(database.hnsw_ef_search),
        "hnsw_iterative_scan": str(database.hnsw_iterative_scan),
        "hnsw_max_scan_tuples": int(database.hnsw_max_scan_tuples),
        "hnsw_scan_mem_multiplier": float(database.hnsw_scan_mem_multiplier),
        "hnsw_index_count": len(index_rows),
        "hnsw_indexes_valid": len(index_rows) == len(REQUIRED_HNSW_INDEXES)
        and all(row.valid and row.ready for row in index_rows),
        "hnsw_index_manifest_sha256": _sha256_json(
            [
                {
                    "name": row.name,
                    "valid": bool(row.valid),
                    "ready": bool(row.ready),
                    "definition": row.definition,
                }
                for row in index_rows
            ]
        ),
        "rls_active": bool(database.rls_active),
        "role_bypass_rls": bool(database.bypass_rls),
        "role_owns_chunks": bool(database.owns_chunks),
    }
    return metadata, evidence, owner_counts, _sha256_json(manifest_rows)


async def _rls_evidence(
    sessionmaker: async_sessionmaker[AsyncSession],
    owner_counts: Mapping[str, int],
) -> tuple[dict[str, Any], bool]:
    owners = sorted(owner_counts)
    if len(owners) < 2:
        return {"principal_count": len(owners), "reason": "foreign_canary_unavailable"}, False
    checks: list[dict[str, Any]] = []
    for index, owner in enumerate(owners):
        foreign_owner = owners[(index + 1) % len(owners)]
        admin_token = set_principal("filtered-hnsw-preflight-admin", True)
        try:
            async with sessionmaker() as session:
                canary_id = (
                    await session.execute(
                        sql(
                            "SELECT c.id FROM chunks c JOIN documents d ON d.id=c.document_id "
                            "WHERE d.owner_sub=:owner ORDER BY c.id LIMIT 1"
                        ),
                        {"owner": foreign_owner},
                    )
                ).scalar_one()
        finally:
            reset_principal(admin_token)
        token = set_principal(owner, False)
        try:
            async with sessionmaker() as session:
                visible = (await session.execute(sql("SELECT count(*) FROM chunks"))).scalar_one()
                foreign_visible = (
                    await session.execute(
                        sql("SELECT count(*) FROM chunks WHERE id=:chunk_id"),
                        {"chunk_id": canary_id},
                    )
                ).scalar_one()
                wrong_owner = (
                    await session.execute(
                        sql(
                            "SELECT count(*) FROM chunks c JOIN documents d ON d.id=c.document_id "
                            "WHERE d.owner_sub IS DISTINCT FROM :owner"
                        ),
                        {"owner": owner},
                    )
                ).scalar_one()
        finally:
            reset_principal(token)
        checks.append(
            {
                "principal_ref": f"principal-sha256:{_sha256_text(owner)}",
                "expected_visible": owner_counts[owner],
                "visible": int(visible),
                "foreign_canary_visible": int(foreign_visible),
                "wrong_owner_visible": int(wrong_owner),
            }
        )
    async with sessionmaker() as session:
        anonymous_visible = (await session.execute(sql("SELECT count(*) FROM chunks"))).scalar_one()
    safe = all(
        item["visible"] == item["expected_visible"]
        and item["foreign_canary_visible"] == 0
        and item["wrong_owner_visible"] == 0
        for item in checks
    ) and anonymous_visible == 0
    return {
        "principal_count": len(checks),
        "admin_foreign_truth_count": sum(owner_counts.values()),
        "anonymous_visible_count": int(anonymous_visible),
        "checks_sha256": _sha256_json(checks),
    }, safe


def _plan_evidence(observations: Sequence[PlanObservation]) -> dict[str, Any]:
    if not observations:
        raise QualificationError("no filter scopes were produced")
    natural_ok = sum(REQUIRED_HNSW_INDEXES.issubset(item.natural_indexes) for item in observations)
    forced_ok = sum(REQUIRED_HNSW_INDEXES.issubset(item.forced_indexes) for item in observations)
    exact_ann = sum(bool(REQUIRED_HNSW_INDEXES & item.exact_indexes) for item in observations)
    rows = [
        {
            "scope_ref": item.scope_ref,
            "scope_kind": item.scope_kind,
            "natural": sorted(item.natural_indexes),
            "forced": sorted(item.forced_indexes),
            "exact": sorted(item.exact_indexes),
        }
        for item in observations
    ]
    return {
        "scope_count": len(observations),
        "scope_kind_counts": {
            kind: sum(item.scope_kind == kind for item in observations)
            for kind in sorted({item.scope_kind for item in observations})
        },
        "natural_dual_hnsw_count": natural_ok,
        "forced_dual_hnsw_count": forced_ok,
        "exact_ann_count": exact_ann,
        "natural_dual_hnsw_coverage": natural_ok / len(observations),
        "forced_dual_hnsw_coverage": forced_ok / len(observations),
        "manifest_sha256": _sha256_json(rows),
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0 or any(not math.isfinite(value) for value in values):
        raise QualificationError("benchmark latency evidence is invalid")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _oracle_recall(returned: Sequence[uuid.UUID], oracle: Sequence[uuid.UUID]) -> float:
    if not oracle:
        return 1.0 if not returned else 0.0
    return len(set(returned) & set(oracle)) / len(oracle)


def _scope_allows_document(
    scope: Scope,
    document_id: uuid.UUID,
    metadata: Mapping[uuid.UUID, DocumentMeta],
) -> bool:
    document = metadata.get(document_id)
    if document is None or document.owner_sub != scope.owner_sub or document.status != "done":
        return False
    if scope.params["doc_id"] is not None and document_id != scope.params["doc_id"]:
        return False
    if scope.params["doc_ids"] is not None and document_id not in scope.params["doc_ids"]:
        return False
    if scope.params["folder_id"] is not None and document.folder_id != scope.params["folder_id"]:
        return False
    return True


async def _execute_dense(
    session: AsyncSession,
    *,
    scope: Scope,
    query_vector: str,
    k: int,
    backend: Literal["exact", "hnsw"],
    iterative_scan: Literal["off", "strict_order"] | None = None,
) -> tuple[tuple[tuple[uuid.UUID, uuid.UUID], ...], float]:
    started = time.perf_counter()
    if backend == "hnsw":
        await configure_hnsw_transaction(
            session,
            iterative_scan=iterative_scan,
            ef_search=100,
            max_scan_tuples=20_000,
            scan_mem_multiplier=2.0,
        )
    rows = (
        await session.execute(
            sql(dense_query_plan(backend).statement),
            {**scope.params, "qe": query_vector, "k": k},
        )
    ).all()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    result = tuple((row.id, row.document_id) for row in rows)
    if len({row[0] for row in result}) != len(result):
        raise QualificationError("dense benchmark returned duplicate chunk IDs")
    return result, elapsed_ms


async def _benchmark_scope(
    sessionmaker: async_sessionmaker[AsyncSession],
    scope: Scope,
    *,
    query_vector: str,
    k: int,
    repeats: int,
    metadata: Mapping[uuid.UUID, DocumentMeta],
) -> tuple[BenchmarkObservation, ...]:
    token = set_principal(scope.owner_sub, False)
    try:
        async with sessionmaker() as session, session.begin():
            await _execute_dense(
                session, scope=scope, query_vector=query_vector, k=k, backend="exact"
            )
            await _execute_dense(
                session,
                scope=scope,
                query_vector=query_vector,
                k=k,
                backend="hnsw",
                iterative_scan="off",
            )
            await _execute_dense(
                session,
                scope=scope,
                query_vector=query_vector,
                k=k,
                backend="hnsw",
                iterative_scan="strict_order",
            )
            observations: list[BenchmarkObservation] = []
            for repeat in range(repeats):
                variants = ["exact", "off", "strict"]
                shift = (repeat + int(scope.ref[-2:], 16)) % len(variants)
                variants = variants[shift:] + variants[:shift]
                results: dict[str, tuple[tuple[tuple[uuid.UUID, uuid.UUID], ...], float]] = {}
                for variant in variants:
                    if variant == "exact":
                        results[variant] = await _execute_dense(
                            session,
                            scope=scope,
                            query_vector=query_vector,
                            k=k,
                            backend="exact",
                        )
                    else:
                        results[variant] = await _execute_dense(
                            session,
                            scope=scope,
                            query_vector=query_vector,
                            k=k,
                            backend="hnsw",
                            iterative_scan="off" if variant == "off" else "strict_order",
                        )
                oracle_rows, exact_latency = results["exact"]
                off_rows, off_latency = results["off"]
                strict_rows, strict_latency = results["strict"]
                oracle = tuple(row[0] for row in oracle_rows)
                off = tuple(row[0] for row in off_rows)
                strict = tuple(row[0] for row in strict_rows)
                scope_violations = sum(
                    not _scope_allows_document(scope, document_id, metadata)
                    for rows in (oracle_rows, off_rows, strict_rows)
                    for _, document_id in rows
                )
                observations.append(
                    BenchmarkObservation(
                        scope_ref=scope.ref,
                        scope_kind=scope.kind,
                        exact_latency_ms=exact_latency,
                        off_latency_ms=off_latency,
                        strict_latency_ms=strict_latency,
                        off_oracle_recall=_oracle_recall(off, oracle),
                        strict_oracle_recall=_oracle_recall(strict, oracle),
                        off_filled=len(off) >= len(oracle),
                        strict_filled=len(strict) >= len(oracle),
                        scope_violation_count=scope_violations,
                    )
                )
    finally:
        reset_principal(token)
    return tuple(observations)


async def _run_concurrent_benchmark(
    sessionmaker: async_sessionmaker[AsyncSession],
    scopes: Sequence[Scope],
    *,
    query_vector: str,
    k: int,
    repeats: int,
    concurrency: int,
    metadata: Mapping[uuid.UUID, DocumentMeta],
) -> tuple[BenchmarkObservation, ...]:
    semaphore = asyncio.Semaphore(concurrency)
    start = asyncio.Event()

    async def run_one(scope: Scope) -> tuple[BenchmarkObservation, ...]:
        await start.wait()
        async with semaphore:
            return await _benchmark_scope(
                sessionmaker,
                scope,
                query_vector=query_vector,
                k=k,
                repeats=repeats,
                metadata=metadata,
            )

    tasks = [asyncio.create_task(run_one(scope)) for scope in scopes]
    start.set()
    nested = await asyncio.gather(*tasks)
    return tuple(item for group in nested for item in group)


def _benchmark_evidence(observations: Sequence[BenchmarkObservation]) -> dict[str, Any]:
    if not observations:
        raise QualificationError("benchmark produced no observations")
    numeric = [
        value
        for item in observations
        for value in (
            item.exact_latency_ms,
            item.off_latency_ms,
            item.strict_latency_ms,
            item.off_oracle_recall,
            item.strict_oracle_recall,
        )
    ]
    if any(not math.isfinite(value) or value < 0.0 for value in numeric):
        raise QualificationError("benchmark contains non-finite or negative values")
    rows = [
        {
            "scope_ref": item.scope_ref,
            "scope_kind": item.scope_kind,
            "off_recall": item.off_oracle_recall,
            "strict_recall": item.strict_oracle_recall,
            "off_filled": item.off_filled,
            "strict_filled": item.strict_filled,
            "scope_violations": item.scope_violation_count,
        }
        for item in observations
    ]
    count = len(observations)
    exact_p95 = _quantile([item.exact_latency_ms for item in observations], 0.95)
    off_p95 = _quantile([item.off_latency_ms for item in observations], 0.95)
    strict_p95 = _quantile([item.strict_latency_ms for item in observations], 0.95)
    off_recall = math.fsum(item.off_oracle_recall for item in observations) / count
    strict_recall = math.fsum(item.strict_oracle_recall for item in observations) / count
    result = {
        "observation_count": count,
        "scope_count": len({item.scope_ref for item in observations}),
        "exact_latency_p95_ms": exact_p95,
        "off_latency_p95_ms": off_p95,
        "strict_latency_p95_ms": strict_p95,
        "off_mean_oracle_recall": off_recall,
        "strict_mean_oracle_recall": strict_recall,
        "off_fill_rate": sum(item.off_filled for item in observations) / count,
        "strict_fill_rate": sum(item.strict_filled for item in observations) / count,
        "scope_violation_count": sum(item.scope_violation_count for item in observations),
        "manifest_sha256": _sha256_json(rows),
    }
    result["iterative_recall_gain"] = strict_recall - off_recall
    result["strict_p95_ratio_to_exact"] = strict_p95 / exact_p95 if exact_p95 > 0 else math.inf
    return result


def evaluate_preflight(
    *,
    release_eligible: bool,
    database: Mapping[str, Any],
    plan: Mapping[str, Any],
    benchmark: Mapping[str, Any],
    database_image_digest: str,
    corpus_stable: bool,
    rls_safe: bool,
) -> tuple[PreflightDecision, list[str]]:
    reasons: list[str] = []
    if not release_eligible:
        reasons.append("debug_or_incomplete_run_is_release_ineligible")
    if database.get("pgvector_version") != EXPECTED_PGVECTOR_VERSION:
        reasons.append("pgvector_version_is_not_0_8_5")
    if database.get("alembic_revision") != "0026":
        reasons.append("database_schema_revision_is_not_0026")
    if database_image_digest != EXPECTED_DATABASE_IMAGE_DIGEST:
        reasons.append("database_image_digest_is_not_pinned_pgvector_0_8_5")
    if database.get("embedded_chunk_count") != database.get("chunk_count"):
        reasons.append("chunks_without_dense_embeddings_exist")
    if not database.get("embedding_values_finite"):
        reasons.append("non_finite_dense_embeddings_exist")
    if database.get("emb_en_type") != "vector(1024)" or database.get("emb_ru_type") != "vector(1024)":
        reasons.append("dense_embedding_column_type_is_not_vector_1024")
    if not database.get("hnsw_indexes_valid"):
        reasons.append("required_hnsw_indexes_are_missing_or_invalid")
    if not database.get("rls_active") or database.get("role_bypass_rls") or database.get("role_owns_chunks"):
        reasons.append("api_database_role_or_rls_is_unsafe")
    if not rls_safe:
        reasons.append("rls_cross_owner_or_anonymous_probe_failed")
    if not corpus_stable:
        reasons.append("corpus_changed_during_preflight")
    if plan.get("exact_ann_count") != 0:
        reasons.append("exact_oracle_used_ann_index")
    if plan.get("forced_dual_hnsw_coverage") != 1.0:
        reasons.append("dual_language_query_is_not_hnsw_indexable_for_every_scope")
    benchmark_values = (
        benchmark.get("off_mean_oracle_recall"),
        benchmark.get("strict_mean_oracle_recall"),
        benchmark.get("off_fill_rate"),
        benchmark.get("strict_fill_rate"),
        benchmark.get("iterative_recall_gain"),
        benchmark.get("strict_p95_ratio_to_exact"),
    )
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in benchmark_values):
        reasons.append("benchmark_metrics_are_not_finite")
    else:
        if benchmark["strict_mean_oracle_recall"] < 0.99:
            reasons.append("strict_hnsw_mean_oracle_recall_below_0_99")
        if benchmark["strict_fill_rate"] < 0.99:
            reasons.append("strict_hnsw_fill_rate_below_0_99")
        if benchmark["strict_p95_ratio_to_exact"] > 1.05:
            reasons.append("strict_hnsw_p95_regressed_over_5_percent")
        if benchmark["iterative_recall_gain"] < 0.005:
            reasons.append("iterative_scan_has_no_measurable_recall_gain")
    if benchmark.get("scope_violation_count") != 0:
        reasons.append("dense_results_violated_owner_or_filter_scope")
    if plan.get("natural_dual_hnsw_coverage") == 0.0:
        reasons.append("production_planner_never_chooses_dual_hnsw")
    if reasons:
        return "no_go", reasons
    return "requires_full_ab", ["preflight_passed_full_paired_quality_and_concurrency_ab_required"]


async def run(args: argparse.Namespace) -> dict[str, Any]:
    git_sha, git_dirty = _git_state()
    if args.mode == "qualification" and git_dirty:
        raise QualificationError("qualification requires a clean repository including untracked files")
    if args.mode == "qualification" and args.limit_cases is not None:
        raise QualificationError("qualification cannot limit Gold cases")
    gold_path = Path(args.gold).expanduser().resolve(strict=True)
    sidecar_path = Path(args.sidecar).expanduser().resolve(strict=True)
    records, _ = load_gold_set(gold_path, mode="release")
    all_sidecars = bind_gold_sidecar(records, load_private_sidecar(sidecar_path))
    if args.limit_cases is not None:
        if args.limit_cases < 1:
            raise QualificationError("debug case limit must be positive")
        records = records[: args.limit_cases]
    release_eligible = (
        args.mode == "qualification"
        and not git_dirty
        and args.limit_cases is None
        and len(records) == EXPECTED_RELEASE_CASES
    )
    selected_sidecars = [all_sidecars[record.case_id] for record in records]
    document_ids = sorted(
        {item.document_id for sidecar in selected_sidecars for item in sidecar.source_documents},
        key=str,
    )
    engine = create_engine().execution_options(isolation_level="REPEATABLE READ")
    sessionmaker = create_sessionmaker(engine)
    try:
        metadata, database, owner_counts, corpus_before = await _load_runtime_state(
            sessionmaker, document_ids
        )
        scopes = _unique_scopes(selected_sidecars, metadata)
        rls, rls_safe = await _rls_evidence(sessionmaker, owner_counts)
        unit_vector = [0.0] * settings.embed_dim
        unit_vector[0] = 1.0
        query_vector = str(unit_vector)
        observations = [
            await _plan_observation(sessionmaker, scope, query_vector=query_vector, k=args.k)
            for scope in scopes
        ]
        benchmark_observations = await _run_concurrent_benchmark(
            sessionmaker,
            scopes,
            query_vector=query_vector,
            k=args.k,
            repeats=args.repeats,
            concurrency=args.concurrency,
            metadata=metadata,
        )
        _, database_after, owner_counts_after, corpus_after = await _load_runtime_state(
            sessionmaker, document_ids
        )
    finally:
        await engine.dispose()
    corpus_stable = (
        corpus_before == corpus_after
        and database == database_after
        and owner_counts == owner_counts_after
    )
    plan = _plan_evidence(observations)
    benchmark = _benchmark_evidence(benchmark_observations)
    decision, reasons = evaluate_preflight(
        release_eligible=release_eligible,
        database=database,
        plan=plan,
        benchmark=benchmark,
        database_image_digest=args.database_image_digest,
        corpus_stable=corpus_stable,
        rls_safe=rls_safe,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "mode": args.mode,
        "release_eligible": release_eligible,
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "database_image_digest": args.database_image_digest,
        "protocol_source_sha256": bytes_sha256(Path(__file__).read_bytes()),
        "gold_sha256": bytes_sha256(gold_path.read_bytes()),
        "sidecar_sha256": bytes_sha256(sidecar_path.read_bytes()),
        "case_count": len(records),
        "k": args.k,
        "repeats": args.repeats,
        "concurrency": args.concurrency,
        "candidate_config": {
            "backend": "hnsw",
            "iterative_scan": "strict_order",
            "ef_search": 100,
            "max_scan_tuples": 20_000,
            "scan_mem_multiplier": 2.0,
            "config_sha256": _sha256_json(
                {
                    "backend": "hnsw",
                    "iterative_scan": "strict_order",
                    "ef_search": 100,
                    "max_scan_tuples": 20_000,
                    "scan_mem_multiplier": 2.0,
                }
            ),
        },
        "planner_probe_vector_sha256": bytes_sha256(query_vector.encode()),
        "database": database,
        "corpus_manifest_sha256": corpus_before,
        "corpus_stable": corpus_stable,
        "rls": rls,
        "rls_safe": rls_safe,
        "plan_evidence": plan,
        "benchmark_evidence": benchmark,
        "decision": decision,
        "decision_reasons": reasons,
    }


def _report_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def _write_private_bytes(path: Path, raw: bytes) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--attestation-output", required=True, type=Path)
    parser.add_argument("--hmac-key", required=True, type=Path)
    parser.add_argument("--database-image-digest", required=True)
    parser.add_argument("--mode", choices=("debug", "qualification"), default="debug")
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--limit-cases", type=int)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not 1 <= args.k <= 1000 or not 1 <= args.repeats <= 20 or not 1 <= args.concurrency <= 32:
        raise SystemExit("invalid k, repeats or concurrency")
    try:
        report = asyncio.run(run(args))
        raw = _report_bytes(report)
        key = load_hmac_key(args.hmac_key, REPOSITORY_ROOT)
        attestation = create_private_artifact_attestation(
            artifact_bytes=raw,
            artifact_type="filtered-hnsw-preflight-v2",
            key=key,
            repository_root=REPOSITORY_ROOT,
            source_paths=ATTESTED_SOURCES,
        )
        if attestation.repository_git_sha != report["git_sha"]:
            raise QualificationError("repository changed while the preflight was running")
        _write_private_bytes(args.output, raw)
        atomic_write_private_artifact_attestation(args.attestation_output, attestation)
    except (
        OSError,
        QualificationError,
        ReportAttestationError,
        SQLAlchemyError,
        subprocess.CalledProcessError,
        ValueError,
    ) as error:
        print(f"filtered HNSW preflight failed: {type(error).__name__}")
        return 1
    print(json.dumps({"decision": report["decision"], "reasons": report["decision_reasons"]}))
    return 2 if report["decision"] == "no_go" else 3


if __name__ == "__main__":
    raise SystemExit(main())
