from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from rag_app.config import settings
from rag_app.eval.reranker_runtime import RerankerRuntimeEvidence, parse_vllm_runtime_argv
from rag_app.llm.embeddings import build_rerank_payload, reranker_template_sha256


def test_qwen3_reranker_payload_contains_manual_template() -> None:
    payload = build_rerank_payload("pressure?", ["10 MPa"])

    assert payload["model"] == settings.rerank_model
    assert "<Instruct>:" in payload["query"]
    assert "<Query>: pressure?" in payload["query"]
    assert payload["documents"][0].startswith("<Document>: 10 MPa")
    assert payload["documents"][0].endswith("<think>\n\n</think>\n\n")


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [("float32", "float32"), ("float", "float32"), ("float16", "float16"), ("half", "float16")],
)
def test_runtime_argv_requires_single_sequence_and_supported_precision(
    dtype: str,
    expected: str,
) -> None:
    raw = f"vllm serve /models/qwen --max-num-seqs=1 --dtype {dtype}".encode()

    max_num_seqs, precision, digest = parse_vllm_runtime_argv(raw)

    assert max_num_seqs == 1
    assert precision == expected
    assert len(digest) == 64


@pytest.mark.parametrize(
    "argv",
    [
        b"vllm serve /models/qwen --dtype float32",
        b"vllm serve /models/qwen --max-num-seqs 2 --dtype float32",
        b"vllm serve /models/qwen --max-num-seqs 1 --dtype bfloat16",
        b"vllm serve /models/qwen --max-num-seqs 1 --dtype float16 --chat-template qwen.jinja",
    ],
)
def test_runtime_argv_rejects_nonqualification_profiles(argv: bytes) -> None:
    with pytest.raises(ValueError):
        parse_vllm_runtime_argv(argv)


def test_runtime_evidence_checks_template_hash_and_probe_gap() -> None:
    payload = {
        "model": settings.rerank_model,
        "endpoint": settings.rerank_base_url,
        "max_num_seqs": 1,
        "precision": "float32",
        "strictly_sequential": True,
        "client_template_protocol": "manual-qwen3-reranker-template-v1",
        "server_chat_template_disabled": True,
        "template_sha256": reranker_template_sha256(),
        "process_argv_sha256": "a" * 64,
        "relevant_score": 0.95,
        "irrelevant_score": 0.05,
        "minimum_probe_gap": 0.2,
        "captured_at": datetime.now(UTC),
    }

    evidence = RerankerRuntimeEvidence.model_validate(payload, strict=True)

    assert evidence.precision == "float32"
    with pytest.raises(ValidationError, match="template hash"):
        RerankerRuntimeEvidence.model_validate(
            {**payload, "template_sha256": "0" * 64},
            strict=True,
        )
    with pytest.raises(ValidationError, match="probe separation"):
        RerankerRuntimeEvidence.model_validate(
            {**payload, "relevant_score": 0.2, "irrelevant_score": 0.1},
            strict=True,
        )
