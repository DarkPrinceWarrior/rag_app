"""Проверяемый runtime-профиль для воспроизводимой квалификации реранкера."""

from __future__ import annotations

import hashlib
import math
import shlex
from datetime import datetime
from typing import Literal
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_app.llm.embeddings import build_rerank_payload, reranker_template_sha256

Precision = Literal["float32", "float16"]


class RerankerRuntimeEvidence(BaseModel):
    """Private artifact proving the temporary vLLM profile and template probe."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["reranker-runtime-evidence-v1"] = "reranker-runtime-evidence-v1"
    model: str = Field(min_length=1, max_length=256)
    endpoint: str = Field(min_length=1, max_length=512)
    max_num_seqs: Literal[1]
    precision: Precision
    strictly_sequential: Literal[True]
    client_template_protocol: Literal["manual-qwen3-reranker-template-v1"]
    server_chat_template_disabled: Literal[True]
    template_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    process_argv_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    relevant_score: float = Field(ge=0, le=1)
    irrelevant_score: float = Field(ge=0, le=1)
    minimum_probe_gap: float = Field(default=0.2, ge=0.1, le=1)
    captured_at: datetime

    @model_validator(mode="after")
    def validate_runtime(self) -> RerankerRuntimeEvidence:
        parsed = urlsplit(self.endpoint)
        if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise ValueError("reranker qualification endpoint must be loopback")
        if self.template_sha256 != reranker_template_sha256():
            raise ValueError("reranker client template hash does not match current code")
        if self.relevant_score - self.irrelevant_score < self.minimum_probe_gap:
            raise ValueError("reranker template probe separation is below its floor")
        if self.captured_at.tzinfo is None:
            raise ValueError("reranker runtime capture timestamp must be timezone-aware")
        return self


def _flag_value(argv: list[str], name: str) -> str | None:
    for index, value in enumerate(argv):
        if value == name:
            return argv[index + 1] if index + 1 < len(argv) else None
        prefix = f"{name}="
        if value.startswith(prefix):
            return value[len(prefix) :]
    return None


def parse_vllm_runtime_argv(raw: bytes) -> tuple[Literal[1], Precision, str]:
    """Проверить точный argv временного vLLM; принимает /proc NUL или shell-строку."""

    if not raw or len(raw) > 1024 * 1024:
        raise ValueError("vLLM argv evidence is empty or too large")
    argv = (
        [part.decode(errors="strict") for part in raw.split(b"\0") if part]
        if b"\0" in raw
        else shlex.split(raw.decode(errors="strict"))
    )
    max_num_seqs = _flag_value(argv, "--max-num-seqs")
    if max_num_seqs != "1":
        raise ValueError("temporary reranker must use --max-num-seqs 1")
    raw_dtype = (_flag_value(argv, "--dtype") or "").casefold()
    if raw_dtype in {"float", "float32"}:
        precision: Precision = "float32"
    elif raw_dtype in {"half", "float16"}:
        precision = "float16"
    else:
        raise ValueError("temporary reranker dtype must be float32 or float16")
    if _flag_value(argv, "--chat-template") is not None or "--chat-template" in argv:
        raise ValueError("server chat template would double-apply the client template")
    return 1, precision, hashlib.sha256(raw).hexdigest()


async def _single_score(endpoint: str, query: str, document: str) -> float:
    payload = build_rerank_payload(query, [document])
    async with httpx.AsyncClient(timeout=60.0, trust_env=False) as client:
        response = await client.post(f"{endpoint.rstrip('/')}/v1/rerank", json=payload)
        response.raise_for_status()
        body = response.json()
    results = body.get("results") if isinstance(body, dict) else None
    if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
        raise ValueError("reranker template probe returned an invalid result")
    raw_score = results[0].get("relevance_score")
    if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
        raise ValueError("reranker template probe score is not numeric")
    score = float(raw_score)
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("reranker template probe score is outside [0, 1]")
    return score


async def capture_runtime_evidence(
    process_argv: bytes,
    *,
    endpoint: str,
    model: str,
) -> RerankerRuntimeEvidence:
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        raise ValueError("reranker qualification endpoint must be loopback")
    max_num_seqs, precision, argv_sha256 = parse_vllm_runtime_argv(process_argv)
    query = "What is the maximum working pressure?"
    relevant = await _single_score(
        endpoint,
        query,
        "The maximum working pressure is 10 MPa.",
    )
    irrelevant = await _single_score(
        endpoint,
        query,
        "The delivery address is stated in Appendix B.",
    )
    from datetime import UTC

    return RerankerRuntimeEvidence(
        model=model,
        endpoint=endpoint,
        max_num_seqs=max_num_seqs,
        precision=precision,
        strictly_sequential=True,
        client_template_protocol="manual-qwen3-reranker-template-v1",
        server_chat_template_disabled=True,
        template_sha256=reranker_template_sha256(),
        process_argv_sha256=argv_sha256,
        relevant_score=relevant,
        irrelevant_score=irrelevant,
        captured_at=datetime.now(UTC),
    )
