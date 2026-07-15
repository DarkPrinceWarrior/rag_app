"""Deterministic quantity/unit support audit for grounded RAG answers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypedDict

from rag_app.eval.rag_metrics import extract_quantity_mentions
from rag_app.rag.retrieve import RetrievedChunk


class QuantityGuardResult(TypedDict):
    """Aggregate-only result: safe to log without document or answer content."""

    mentioned_count: int
    supported_count: int
    unsupported_count: int
    unsupported_pair_count: int
    unsupported_value_count: int
    invalid_unit_count: int
    unsupported_rate: float


class PrivateQuantityGuardArtifact(TypedDict):
    """Per-case private-evaluator record with no answer or evidence payload."""

    schema_version: Literal["rag-quantity-guard/v1"]
    case_id: str
    mentioned_count: int
    supported_count: int
    unsupported_count: int
    unsupported_pair_count: int
    unsupported_value_count: int
    invalid_unit_count: int
    unsupported_rate: float


def evaluate_quantity_support(
    answer: str,
    chunks: Sequence[RetrievedChunk],
) -> QuantityGuardResult:
    """Compare answer quantities with the exact retrieved chunks used for generation.

    Catalog pseudo-chunks are not claim evidence. RU/EN unit aliases and decimal
    comma/dot spellings are normalized by the shared evaluation extractor.
    """

    answer_mentions = extract_quantity_mentions(answer, comma_policy="decimal")
    evidence_mentions = [
        mention
        for chunk in chunks
        if chunk.kind != "catalog"
        for text in (chunk.text_en, chunk.text_ru)
        if text
        for mention in extract_quantity_mentions(text, comma_policy="decimal")
    ]
    supported_pairs = {
        (mention["value"], mention["unit"])
        for mention in evidence_mentions
        if mention["unit"] is not None
    }
    supported_values = {mention["value"] for mention in evidence_mentions}
    unsupported_pair_count = sum(
        mention["unit"] is None
        or (mention["value"], mention["unit"]) not in supported_pairs
        for mention in answer_mentions
    )
    unsupported_value_count = sum(
        mention["value"] not in supported_values for mention in answer_mentions
    )
    mentioned_count = len(answer_mentions)
    supported_count = mentioned_count - unsupported_pair_count
    return {
        "mentioned_count": mentioned_count,
        "supported_count": supported_count,
        "unsupported_count": unsupported_pair_count,
        "unsupported_pair_count": unsupported_pair_count,
        "unsupported_value_count": unsupported_value_count,
        "invalid_unit_count": sum(not mention["unit_valid"] for mention in answer_mentions),
        "unsupported_rate": unsupported_pair_count / mentioned_count if mentioned_count else 0.0,
    }


def private_quantity_guard_artifact(
    case_id: str,
    result: QuantityGuardResult,
) -> PrivateQuantityGuardArtifact:
    """Build the content-free artifact contract for private Gold evaluation."""

    if not case_id.strip():
        raise ValueError("quantity guard case_id must be non-empty")
    return {
        "schema_version": "rag-quantity-guard/v1",
        "case_id": case_id,
        **result,
    }


__all__ = [
    "PrivateQuantityGuardArtifact",
    "QuantityGuardResult",
    "evaluate_quantity_support",
    "private_quantity_guard_artifact",
]
