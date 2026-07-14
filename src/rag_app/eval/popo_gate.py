"""Content-free A/B qualification gate for MinerU-Popo post-processing.

The gate deliberately evaluates Popo as a post-processor, not as an OCR model.
Gold and variant artifacts contain only stable block identifiers and hashes; raw
document text is neither required nor emitted in the report.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Language = Literal["ru", "en", "zh"]
BlockKind = Literal["heading", "text", "table", "image", "other"]
RelationTask = Literal["text_continuation", "table_continuation", "image_association"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SourceRef(_StrictModel):
    document_ref: str = Field(min_length=1)
    block_id: str = Field(min_length=1)

    def key(self) -> tuple[str, str]:
        return self.document_ref, self.block_id


class GradedSourceRef(SourceRef):
    grade: int = Field(ge=1, le=3)


class GoldBlock(_StrictModel):
    block_id: str = Field(min_length=1)
    page: int = Field(ge=1)
    order: int = Field(ge=0)
    kind: BlockKind
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class HeadingEdge(_StrictModel):
    child_id: str = Field(min_length=1)
    parent_id: str | None = None

    @model_validator(mode="after")
    def validate_not_self_referential(self) -> HeadingEdge:
        if self.child_id == self.parent_id:
            raise ValueError("heading edge cannot be self-referential")
        return self


def _validate_heading_graph(edges: list[HeadingEdge]) -> None:
    children = [edge.child_id for edge in edges]
    if len(children) != len(set(children)):
        raise ValueError("heading graph assigns more than one parent to a child")

    parents = {edge.child_id: edge.parent_id for edge in edges}
    for child_id in parents:
        visited: set[str] = set()
        current_id: str | None = child_id
        while current_id is not None and current_id in parents:
            if current_id in visited:
                raise ValueError("heading graph contains a cycle")
            visited.add(current_id)
            current_id = parents[current_id]


class OrderPair(_StrictModel):
    before_id: str = Field(min_length=1)
    after_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_not_self_referential(self) -> OrderPair:
        if self.before_id == self.after_id:
            raise ValueError("order pair cannot be self-referential")
        return self


class StructuralRelation(_StrictModel):
    task: RelationTask
    source_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_not_self_referential(self) -> StructuralRelation:
        if self.source_id == self.target_id:
            raise ValueError("structural relation cannot be self-referential")
        return self


def _validate_relation_semantics(
    relation: StructuralRelation, block_by_id: dict[str, GoldBlock]
) -> None:
    source = block_by_id[relation.source_id]
    target = block_by_id[relation.target_id]
    if relation.task == "text_continuation" and {source.kind, target.kind} != {"text"}:
        raise ValueError("text continuation must connect text blocks")
    if relation.task == "table_continuation" and {source.kind, target.kind} != {"table"}:
        raise ValueError("table continuation must connect table blocks")
    if relation.task == "image_association" and (
        (source.kind == "image") == (target.kind == "image")
    ):
        raise ValueError("image association must connect one image to one non-image block")
    if relation.task in {"text_continuation", "table_continuation"} and (
        source.page >= target.page
    ):
        raise ValueError("continuation relation must move to a later page")


class GoldDocument(_StrictModel):
    document_ref: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    language: Language
    page_count: int = Field(ge=1)
    blocks: list[GoldBlock] = Field(min_length=1)
    heading_edges: list[HeadingEdge] = Field(default_factory=list)
    order_pairs: list[OrderPair] = Field(default_factory=list)
    relations: list[StructuralRelation] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_annotations(self) -> GoldDocument:
        block_ids = [block.block_id for block in self.blocks]
        if len(block_ids) != len(set(block_ids)):
            raise ValueError("gold document has duplicate block identifiers")
        block_orders = [block.order for block in self.blocks]
        if len(block_orders) != len(set(block_orders)):
            raise ValueError("gold document has duplicate block order values")
        if any(block.page > self.page_count for block in self.blocks):
            raise ValueError("gold block page exceeds document page count")

        block_by_id = {block.block_id: block for block in self.blocks}
        _validate_heading_graph(self.heading_edges)
        for edge in self.heading_edges:
            if edge.child_id not in block_by_id or (
                edge.parent_id is not None and edge.parent_id not in block_by_id
            ):
                raise ValueError("heading edge references an unknown block")
            if block_by_id[edge.child_id].kind != "heading" or (
                edge.parent_id is not None and block_by_id[edge.parent_id].kind != "heading"
            ):
                raise ValueError("heading edge must reference heading blocks")

        order_pairs = [(pair.before_id, pair.after_id) for pair in self.order_pairs]
        if len(order_pairs) != len(set(order_pairs)):
            raise ValueError("gold document has duplicate order pairs")
        for pair in self.order_pairs:
            if pair.before_id not in block_by_id or pair.after_id not in block_by_id:
                raise ValueError("order pair references an unknown block")
            if block_by_id[pair.before_id].order >= block_by_id[pair.after_id].order:
                raise ValueError("gold order pair contradicts block order")

        relations = [
            (relation.task, relation.source_id, relation.target_id) for relation in self.relations
        ]
        if len(relations) != len(set(relations)):
            raise ValueError("gold document has duplicate structural relations")
        for relation in self.relations:
            if relation.source_id not in block_by_id or relation.target_id not in block_by_id:
                raise ValueError("structural relation references an unknown block")
            _validate_relation_semantics(relation, block_by_id)
        return self


class GoldDownstreamCase(_StrictModel):
    case_id: str = Field(min_length=1)
    language: Language
    relevant: list[GradedSourceRef] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_relevant(self) -> GoldDownstreamCase:
        keys = [item.key() for item in self.relevant]
        if len(keys) != len(set(keys)):
            raise ValueError("downstream gold has duplicate evidence references")
        return self


class PopoGold(_StrictModel):
    schema_version: Literal["popo-gold-v1"]
    source_revision: str = Field(min_length=7)
    documents: list[GoldDocument] = Field(min_length=1)
    downstream_cases: list[GoldDownstreamCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_corpus(self) -> PopoGold:
        document_refs = [document.document_ref for document in self.documents]
        if len(document_refs) != len(set(document_refs)):
            raise ValueError("gold corpus has duplicate document references")
        case_ids = [case.case_id for case in self.downstream_cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("gold corpus has duplicate downstream case identifiers")

        evidence = {
            (document.document_ref, block.block_id)
            for document in self.documents
            for block in document.blocks
        }
        for case in self.downstream_cases:
            if any(item.key() not in evidence for item in case.relevant):
                raise ValueError("downstream gold references evidence outside the source inventory")
        return self


class NodeSourceMap(_StrictModel):
    node_id: str = Field(min_length=1)
    source_block_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_source_blocks(self) -> NodeSourceMap:
        if len(self.source_block_ids) != len(set(self.source_block_ids)):
            raise ValueError("node source map contains duplicate source block identifiers")
        return self


class VariantDocument(_StrictModel):
    document_ref: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    node_source_map: list[NodeSourceMap] = Field(min_length=1)
    heading_edges: list[HeadingEdge] = Field(default_factory=list)
    block_order: list[str] = Field(min_length=1)
    relations: list[StructuralRelation] = Field(default_factory=list)
    latency_ms: float = Field(gt=0)
    peak_vram_mib: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_internal_references(self) -> VariantDocument:
        node_ids = [node.node_id for node in self.node_source_map]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("variant document has duplicate node identifiers")
        source_ids = [
            block_id for node in self.node_source_map for block_id in node.source_block_ids
        ]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("variant document maps a source block more than once")
        if len(self.block_order) != len(set(self.block_order)):
            raise ValueError("variant document block order contains duplicate identifiers")
        if set(self.block_order) != set(source_ids):
            raise ValueError("variant block order must exactly cover its source mapping")

        source_inventory = set(source_ids)
        _validate_heading_graph(self.heading_edges)
        heading_refs = {
            block_id
            for edge in self.heading_edges
            for block_id in (edge.child_id, edge.parent_id)
            if block_id is not None
        }
        if not heading_refs <= source_inventory:
            raise ValueError("variant heading graph references an unmapped source block")

        relations = [
            (relation.task, relation.source_id, relation.target_id) for relation in self.relations
        ]
        if len(relations) != len(set(relations)):
            raise ValueError("variant document has duplicate structural relations")
        relation_refs = {
            block_id
            for relation in self.relations
            for block_id in (relation.source_id, relation.target_id)
        }
        if not relation_refs <= source_inventory:
            raise ValueError("variant relation references an unmapped source block")
        return self


class VariantDownstreamResult(_StrictModel):
    case_id: str = Field(min_length=1)
    ranked: list[SourceRef]
    cited: list[SourceRef]

    @model_validator(mode="after")
    def validate_unique_refs(self) -> VariantDownstreamResult:
        ranked = [item.key() for item in self.ranked]
        cited = [item.key() for item in self.cited]
        if len(ranked) != len(set(ranked)) or len(cited) != len(set(cited)):
            raise ValueError("downstream result contains duplicate evidence references")
        if not set(cited) <= set(ranked):
            raise ValueError("cited evidence must be present in the ranked evidence")
        return self


class PopoVariant(_StrictModel):
    schema_version: Literal["popo-variant-v1"]
    source_revision: str = Field(min_length=7)
    variant_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=7)
    code_revision: str = Field(min_length=7)
    seed: int = Field(ge=0)
    documents: list[VariantDocument] = Field(min_length=1)
    downstream_results: list[VariantDownstreamResult] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_keys(self) -> PopoVariant:
        document_refs = [document.document_ref for document in self.documents]
        case_ids = [case.case_id for case in self.downstream_results]
        if len(document_refs) != len(set(document_refs)):
            raise ValueError("variant has duplicate document references")
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("variant has duplicate downstream case identifiers")
        return self


@dataclass(frozen=True, slots=True)
class GatePolicy:
    """Production qualification thresholds for the private complex-document corpus."""

    min_documents_per_language: int = 1
    min_pages_per_language: int = 5
    min_heading_edges_per_language: int = 10
    min_order_pairs_per_language: int = 20
    min_relations_per_task: int = 10
    min_relations_per_task_per_language: int = 3
    min_downstream_cases_per_language: int = 20
    max_structure_regression: float = 0.005
    min_structure_gain: float = 0.01
    max_downstream_regression: float = 0.01
    max_p95_latency_ratio: float = 2.0
    max_peak_vram_mib: float = 36_000.0

    def __post_init__(self) -> None:
        count_fields = (
            self.min_documents_per_language,
            self.min_pages_per_language,
            self.min_heading_edges_per_language,
            self.min_order_pairs_per_language,
            self.min_relations_per_task,
            self.min_relations_per_task_per_language,
            self.min_downstream_cases_per_language,
        )
        if any(value < 1 for value in count_fields):
            raise ValueError("evidence minimums must be positive")
        bounded_fields = (
            self.max_structure_regression,
            self.min_structure_gain,
            self.max_downstream_regression,
        )
        if any(value < 0 or value > 1 for value in bounded_fields):
            raise ValueError("quality thresholds must be between zero and one")
        if self.max_p95_latency_ratio < 1:
            raise ValueError("latency ratio must be at least one")
        if self.max_peak_vram_mib <= 0:
            raise ValueError("VRAM budget must be positive")


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def source_inventory_sha256(blocks: list[GoldBlock]) -> str:
    inventory = [
        {
            "block_id": block.block_id,
            "content_sha256": block.content_sha256,
            "kind": block.kind,
            "order": block.order,
            "page": block.page,
        }
        for block in sorted(blocks, key=lambda item: item.block_id)
    ]
    return canonical_sha256(inventory)


def _set_metrics(
    gold: set[tuple[Any, ...]], predicted: set[tuple[Any, ...]]
) -> dict[str, float | int]:
    true_positive = len(gold & predicted)
    precision = true_positive / len(predicted) if predicted else float(not gold)
    recall = true_positive / len(gold) if gold else float(not predicted)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "gold": len(gold),
        "predicted": len(predicted),
        "true_positive": true_positive,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _ndcg_at_10(relevant: dict[tuple[str, str], int], ranked: list[tuple[str, str]]) -> float:
    dcg = sum(
        (2 ** relevant.get(item, 0) - 1) / math.log2(rank + 2)
        for rank, item in enumerate(ranked[:10])
    )
    ideal = sorted(relevant.values(), reverse=True)[:10]
    idcg = sum((2**grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def _mean(rows: list[dict[str, float]], key: str) -> float:
    return sum(row[key] for row in rows) / len(rows) if rows else 0.0


def _validate_pair_linkage(gold: PopoGold, baseline: PopoVariant, candidate: PopoVariant) -> None:
    if baseline.variant_id == candidate.variant_id:
        raise ValueError("baseline and candidate variant identifiers must differ")
    if len({gold.source_revision, baseline.source_revision, candidate.source_revision}) != 1:
        raise ValueError("gold and variants must use the same source revision")
    if baseline.seed != candidate.seed:
        raise ValueError("baseline and candidate must use the same deterministic seed")

    gold_document_by_ref = {document.document_ref: document for document in gold.documents}
    gold_documents = {
        document_ref: document.source_sha256
        for document_ref, document in gold_document_by_ref.items()
    }
    gold_cases = {case.case_id for case in gold.downstream_cases}
    for variant in (baseline, candidate):
        variant_documents = {
            document.document_ref: document.source_sha256 for document in variant.documents
        }
        variant_cases = {case.case_id for case in variant.downstream_results}
        if variant_documents != gold_documents:
            raise ValueError("variant document identities or source hashes do not match gold")
        if variant_cases != gold_cases:
            raise ValueError("variant downstream case set does not match gold")
        for document in variant.documents:
            gold_document = gold_document_by_ref[document.document_ref]
            block_by_id = {block.block_id: block for block in gold_document.blocks}
            for edge in document.heading_edges:
                if edge.child_id not in block_by_id or (
                    edge.parent_id is not None and edge.parent_id not in block_by_id
                ):
                    raise ValueError("variant heading edge references an unknown gold block")
                if block_by_id[edge.child_id].kind != "heading" or (
                    edge.parent_id is not None and block_by_id[edge.parent_id].kind != "heading"
                ):
                    raise ValueError("variant heading edge must reference heading blocks")
            for relation in document.relations:
                if relation.source_id not in block_by_id or relation.target_id not in block_by_id:
                    raise ValueError("variant relation references an unknown gold block")
                _validate_relation_semantics(relation, block_by_id)


def _mapping_metrics(gold: GoldDocument, variant: VariantDocument) -> dict[str, float | int | bool]:
    expected = {block.block_id for block in gold.blocks}
    assigned = [block_id for node in variant.node_source_map for block_id in node.source_block_ids]
    counts = Counter(assigned)
    known = set(assigned) & expected
    duplicate_count = sum(count - 1 for block_id, count in counts.items() if block_id in expected)
    unknown_count = sum(count for block_id, count in counts.items() if block_id not in expected)
    expected_digest = source_inventory_sha256(gold.blocks)
    inventory_match = variant.source_inventory_sha256 == expected_digest
    coverage = len(known) / len(expected)
    order_counts = Counter(variant.block_order)
    order_unknown_count = sum(
        count for block_id, count in order_counts.items() if block_id not in expected
    )
    order_missing_count = len(expected - set(variant.block_order))
    order_duplicate_count = sum(count - 1 for count in order_counts.values() if count > 1)
    structural_refs = {
        block_id
        for edge in variant.heading_edges
        for block_id in (edge.child_id, edge.parent_id)
        if block_id is not None
    } | {
        block_id
        for relation in variant.relations
        for block_id in (relation.source_id, relation.target_id)
    }
    structural_unknown_count = len(structural_refs - expected)
    reference_valid = (
        order_unknown_count == 0
        and order_missing_count == 0
        and order_duplicate_count == 0
        and structural_unknown_count == 0
    )
    return {
        "coverage": coverage,
        "missing_count": len(expected - known),
        "duplicate_count": duplicate_count,
        "unknown_count": unknown_count,
        "inventory_match": inventory_match,
        "order_unknown_count": order_unknown_count,
        "order_missing_count": order_missing_count,
        "order_duplicate_count": order_duplicate_count,
        "structural_unknown_count": structural_unknown_count,
        "reference_valid": reference_valid,
        "valid": coverage == 1.0
        and duplicate_count == 0
        and unknown_count == 0
        and inventory_match,
        "source_integrity_valid": coverage == 1.0
        and duplicate_count == 0
        and unknown_count == 0
        and inventory_match
        and reference_valid,
    }


def _document_structure_metrics(
    gold: GoldDocument, variant: VariantDocument
) -> dict[str, Any]:
    heading_gold = {(edge.child_id, edge.parent_id) for edge in gold.heading_edges}
    heading_predicted = {(edge.child_id, edge.parent_id) for edge in variant.heading_edges}

    positions = {block_id: index for index, block_id in enumerate(variant.block_order)}
    duplicate_order_ids = len(variant.block_order) - len(positions)
    covered_pairs = [
        pair
        for pair in gold.order_pairs
        if pair.before_id in positions and pair.after_id in positions
    ]
    correct_pairs = sum(
        positions[pair.before_id] < positions[pair.after_id] for pair in covered_pairs
    )
    order_total = len(gold.order_pairs)

    relations: dict[str, dict[str, float | int]] = {}
    for task in ("text_continuation", "table_continuation", "image_association"):
        relation_gold = {
            (relation.source_id, relation.target_id)
            for relation in gold.relations
            if relation.task == task
        }
        relation_predicted = {
            (relation.source_id, relation.target_id)
            for relation in variant.relations
            if relation.task == task
        }
        relations[task] = _set_metrics(relation_gold, relation_predicted)

    return {
        "mapping": _mapping_metrics(gold, variant),
        "heading": _set_metrics(heading_gold, heading_predicted),
        "order": {
            "gold": order_total,
            "covered": len(covered_pairs),
            "correct": correct_pairs,
            "coverage": len(covered_pairs) / order_total if order_total else 1.0,
            "accuracy": correct_pairs / len(covered_pairs) if covered_pairs else float(not order_total),
            "score": correct_pairs / order_total if order_total else 1.0,
            "duplicate_ids": duplicate_order_ids,
        },
        "relations": relations,
    }


def _aggregate_structure(
    rows: list[tuple[Language, dict[str, Any]]], language: Language | None = None
) -> dict[str, Any]:
    selected = [metrics for row_language, metrics in rows if language is None or row_language == language]
    heading_gold = sum(int(row["heading"]["gold"]) for row in selected)
    heading_predicted = sum(int(row["heading"]["predicted"]) for row in selected)
    heading_tp = sum(int(row["heading"]["true_positive"]) for row in selected)
    heading_precision = heading_tp / heading_predicted if heading_predicted else float(not heading_gold)
    heading_recall = heading_tp / heading_gold if heading_gold else float(not heading_predicted)
    heading_f1 = (
        2 * heading_precision * heading_recall / (heading_precision + heading_recall)
        if heading_precision + heading_recall
        else 0.0
    )

    order_gold = sum(int(row["order"]["gold"]) for row in selected)
    order_covered = sum(int(row["order"]["covered"]) for row in selected)
    order_correct = sum(int(row["order"]["correct"]) for row in selected)
    relation_totals: dict[str, dict[str, float | int]] = {}
    relation_gold_total = relation_predicted_total = relation_tp_total = 0
    for task in ("text_continuation", "table_continuation", "image_association"):
        task_gold = sum(int(row["relations"][task]["gold"]) for row in selected)
        task_predicted = sum(int(row["relations"][task]["predicted"]) for row in selected)
        task_tp = sum(int(row["relations"][task]["true_positive"]) for row in selected)
        relation_totals[task] = _set_metrics(
            {(task, index) for index in range(task_gold)},
            {(task, index) for index in range(task_tp)}
            | {(f"predicted-{task}", index) for index in range(task_predicted - task_tp)},
        )
        relation_gold_total += task_gold
        relation_predicted_total += task_predicted
        relation_tp_total += task_tp

    relation_precision = (
        relation_tp_total / relation_predicted_total
        if relation_predicted_total
        else float(not relation_gold_total)
    )
    relation_recall = (
        relation_tp_total / relation_gold_total
        if relation_gold_total
        else float(not relation_predicted_total)
    )
    relation_f1 = (
        2 * relation_precision * relation_recall / (relation_precision + relation_recall)
        if relation_precision + relation_recall
        else 0.0
    )

    return {
        "documents": len(selected),
        "mapping_valid": all(bool(row["mapping"]["source_integrity_valid"]) for row in selected),
        "mapping_coverage": min(
            (float(row["mapping"]["coverage"]) for row in selected), default=0.0
        ),
        "mapping_failures": sum(
            not bool(row["mapping"]["source_integrity_valid"]) for row in selected
        ),
        "heading": {
            "gold": heading_gold,
            "predicted": heading_predicted,
            "true_positive": heading_tp,
            "precision": heading_precision,
            "recall": heading_recall,
            "f1": heading_f1,
        },
        "order": {
            "gold": order_gold,
            "covered": order_covered,
            "correct": order_correct,
            "coverage": order_covered / order_gold if order_gold else 1.0,
            "accuracy": order_correct / order_covered if order_covered else float(not order_gold),
            "score": order_correct / order_gold if order_gold else 1.0,
            "duplicate_ids": sum(int(row["order"]["duplicate_ids"]) for row in selected),
        },
        "relations": relation_totals,
        "relation_micro_f1": relation_f1,
    }


def _downstream_metrics(gold: PopoGold, variant: PopoVariant) -> dict[str, Any]:
    results = {result.case_id: result for result in variant.downstream_results}
    source_inventory = {
        (document.document_ref, block.block_id)
        for document in gold.documents
        for block in document.blocks
    }
    rows: list[tuple[Language, dict[str, float]]] = []
    unknown_reference_count = 0
    for case in gold.downstream_cases:
        result = results[case.case_id]
        relevant = {item.key(): item.grade for item in case.relevant}
        relevant_keys = set(relevant)
        ranked = [item.key() for item in result.ranked]
        cited = {item.key() for item in result.cited}
        unknown_reference_count += sum(item not in source_inventory for item in ranked)
        unknown_reference_count += sum(item not in source_inventory for item in cited)
        retrieved = set(ranked[:5])
        recall_at_5 = len(retrieved & relevant_keys) / len(relevant_keys)
        citation_precision = len(cited & relevant_keys) / len(cited) if cited else 0.0
        citation_recall = len(cited & relevant_keys) / len(relevant_keys)
        citation_f1 = (
            2 * citation_precision * citation_recall / (citation_precision + citation_recall)
            if citation_precision + citation_recall
            else 0.0
        )
        rows.append(
            (
                case.language,
                {
                    "recall_at_5": recall_at_5,
                    "ndcg_at_10": _ndcg_at_10(relevant, ranked),
                    "citation_precision": citation_precision,
                    "citation_recall": citation_recall,
                    "citation_f1": citation_f1,
                },
            )
        )

    def aggregate(language: Language | None = None) -> dict[str, float | int]:
        selected = [row for row_language, row in rows if language is None or row_language == language]
        return {
            "cases": len(selected),
            "recall_at_5": _mean(selected, "recall_at_5"),
            "ndcg_at_10": _mean(selected, "ndcg_at_10"),
            "citation_precision": _mean(selected, "citation_precision"),
            "citation_recall": _mean(selected, "citation_recall"),
            "citation_f1": _mean(selected, "citation_f1"),
        }

    return {
        "overall": aggregate(),
        "by_language": {language: aggregate(language) for language in ("ru", "en", "zh")},
        "source_references_valid": unknown_reference_count == 0,
        "unknown_source_references": unknown_reference_count,
    }


def _evaluate_variant(gold: PopoGold, variant: PopoVariant) -> dict[str, Any]:
    gold_documents = {document.document_ref: document for document in gold.documents}
    structure_rows = [
        (
            gold_documents[document.document_ref].language,
            _document_structure_metrics(gold_documents[document.document_ref], document),
        )
        for document in variant.documents
    ]
    latencies = [document.latency_ms for document in variant.documents]
    vrams = [document.peak_vram_mib for document in variant.documents]
    return {
        "variant_id": variant.variant_id,
        "model_revision": variant.model_revision,
        "code_revision": variant.code_revision,
        "seed": variant.seed,
        "structure": {
            "overall": _aggregate_structure(structure_rows),
            "by_language": {
                language: _aggregate_structure(structure_rows, language)
                for language in ("ru", "en", "zh")
            },
        },
        "downstream": _downstream_metrics(gold, variant),
        "runtime": {
            "documents": len(latencies),
            "latency_mean_ms": sum(latencies) / len(latencies),
            "latency_p95_ms": _percentile(latencies, 0.95),
            "latency_max_ms": max(latencies),
            "peak_vram_mib": max(vrams),
        },
    }


def _evidence_issues(gold: PopoGold, policy: GatePolicy) -> list[str]:
    issues: list[str] = []
    documents_by_language: Counter[str] = Counter(document.language for document in gold.documents)
    pages_by_language: Counter[str] = Counter()
    headings_by_language: Counter[str] = Counter()
    orders_by_language: Counter[str] = Counter()
    relations_by_task: Counter[str] = Counter()
    relations_by_language_task: Counter[tuple[str, str]] = Counter()
    cases_by_language: Counter[str] = Counter(case.language for case in gold.downstream_cases)
    for document in gold.documents:
        pages_by_language[document.language] += document.page_count
        headings_by_language[document.language] += len(document.heading_edges)
        orders_by_language[document.language] += len(document.order_pairs)
        relations_by_task.update(relation.task for relation in document.relations)
        relations_by_language_task.update(
            (document.language, relation.task) for relation in document.relations
        )

    for language in ("ru", "en", "zh"):
        if documents_by_language[language] < policy.min_documents_per_language:
            issues.append(f"insufficient_documents:{language}")
        if pages_by_language[language] < policy.min_pages_per_language:
            issues.append(f"insufficient_pages:{language}")
        if headings_by_language[language] < policy.min_heading_edges_per_language:
            issues.append(f"insufficient_heading_edges:{language}")
        if orders_by_language[language] < policy.min_order_pairs_per_language:
            issues.append(f"insufficient_order_pairs:{language}")
        if cases_by_language[language] < policy.min_downstream_cases_per_language:
            issues.append(f"insufficient_downstream_cases:{language}")
        for task in ("text_continuation", "table_continuation", "image_association"):
            if (
                relations_by_language_task[(language, task)]
                < policy.min_relations_per_task_per_language
            ):
                issues.append(f"insufficient_relations:{language}:{task}")
    for task in ("text_continuation", "table_continuation", "image_association"):
        if relations_by_task[task] < policy.min_relations_per_task:
            issues.append(f"insufficient_relations:{task}")
    return issues


def _decision(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    policy: GatePolicy,
    evidence_issues: list[str],
) -> dict[str, Any]:
    failures = list(evidence_issues)
    if not baseline["structure"]["overall"]["mapping_valid"]:
        failures.append("baseline_source_mapping_invalid")
    if not candidate["structure"]["overall"]["mapping_valid"]:
        failures.append("candidate_source_mapping_invalid")
    if not baseline["downstream"]["source_references_valid"]:
        failures.append("baseline_downstream_source_references_invalid")
    if not candidate["downstream"]["source_references_valid"]:
        failures.append("candidate_downstream_source_references_invalid")

    structure_metrics = {
        "heading_parent_f1": ("heading", "f1"),
        "order_score": ("order", "score"),
    }
    improvements: dict[str, float] = {}
    for scope in ("overall", "ru", "en", "zh"):
        baseline_scope = (
            baseline["structure"]["overall"]
            if scope == "overall"
            else baseline["structure"]["by_language"][scope]
        )
        candidate_scope = (
            candidate["structure"]["overall"]
            if scope == "overall"
            else candidate["structure"]["by_language"][scope]
        )
        for metric_name, path in structure_metrics.items():
            delta = float(candidate_scope[path[0]][path[1]]) - float(
                baseline_scope[path[0]][path[1]]
            )
            improvements[f"{scope}:{metric_name}"] = delta
            if delta < -policy.max_structure_regression:
                failures.append(f"structure_regression:{scope}:{metric_name}")
        relation_delta = float(candidate_scope["relation_micro_f1"]) - float(
            baseline_scope["relation_micro_f1"]
        )
        improvements[f"{scope}:relation_micro_f1"] = relation_delta
        if relation_delta < -policy.max_structure_regression:
            failures.append(f"structure_regression:{scope}:relation_micro_f1")
    if max(improvements.values(), default=0.0) < policy.min_structure_gain:
        failures.append("no_material_structure_gain")

    for scope in ("overall", "ru", "en", "zh"):
        baseline_scope = (
            baseline["downstream"]["overall"]
            if scope == "overall"
            else baseline["downstream"]["by_language"][scope]
        )
        candidate_scope = (
            candidate["downstream"]["overall"]
            if scope == "overall"
            else candidate["downstream"]["by_language"][scope]
        )
        for metric in ("recall_at_5", "ndcg_at_10", "citation_f1"):
            if float(candidate_scope[metric]) < (
                float(baseline_scope[metric]) - policy.max_downstream_regression
            ):
                failures.append(f"downstream_regression:{scope}:{metric}")

    latency_ratio = float(candidate["runtime"]["latency_p95_ms"]) / float(
        baseline["runtime"]["latency_p95_ms"]
    )
    if latency_ratio > policy.max_p95_latency_ratio:
        failures.append("latency_budget_exceeded")
    if float(candidate["runtime"]["peak_vram_mib"]) > policy.max_peak_vram_mib:
        failures.append("vram_budget_exceeded")

    eligible = not evidence_issues
    unique_failures = sorted(set(failures))
    return {
        "eligible": eligible,
        "accepted": eligible and not unique_failures,
        "failures": unique_failures,
        "structure_deltas": improvements,
        "latency_p95_ratio": latency_ratio,
    }


def evaluate_popo_pair(
    gold_payload: PopoGold | dict[str, Any],
    baseline_payload: PopoVariant | dict[str, Any],
    candidate_payload: PopoVariant | dict[str, Any],
    *,
    policy: GatePolicy | None = None,
) -> dict[str, Any]:
    """Evaluate a deterministic raw-MinerU versus MinerU+Popo pair."""

    gold = (
        gold_payload if isinstance(gold_payload, PopoGold) else PopoGold.model_validate(gold_payload)
    )
    baseline = (
        baseline_payload
        if isinstance(baseline_payload, PopoVariant)
        else PopoVariant.model_validate(baseline_payload)
    )
    candidate = (
        candidate_payload
        if isinstance(candidate_payload, PopoVariant)
        else PopoVariant.model_validate(candidate_payload)
    )
    selected_policy = policy or GatePolicy()
    _validate_pair_linkage(gold, baseline, candidate)

    baseline_metrics = _evaluate_variant(gold, baseline)
    candidate_metrics = _evaluate_variant(gold, candidate)
    evidence_issues = _evidence_issues(gold, selected_policy)
    report: dict[str, Any] = {
        "schema_version": "popo-ab-report-v1",
        "gold_sha256": canonical_sha256(gold.model_dump(mode="json")),
        "source_revision": gold.source_revision,
        "policy": asdict(selected_policy),
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "decision": _decision(
            baseline_metrics, candidate_metrics, selected_policy, evidence_issues
        ),
    }
    report["report_sha256"] = canonical_sha256(report)
    return report
