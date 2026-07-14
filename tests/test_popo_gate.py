from __future__ import annotations

import json
import stat
import subprocess
import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from rag_app.eval.popo_gate import GatePolicy, evaluate_popo_pair, source_inventory_sha256

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _gold() -> dict[str, Any]:
    documents = []
    cases = []
    for language, digest in (("ru", SHA_A), ("en", SHA_B), ("zh", SHA_C)):
        document_ref = f"doc-{language}"
        blocks = [
            {
                "block_id": "h1",
                "page": 1,
                "order": 0,
                "kind": "heading",
                "content_sha256": "1" * 64,
            },
            {
                "block_id": "t1",
                "page": 1,
                "order": 1,
                "kind": "text",
                "content_sha256": "2" * 64,
            },
            {
                "block_id": "h2",
                "page": 2,
                "order": 2,
                "kind": "heading",
                "content_sha256": "3" * 64,
            },
            {
                "block_id": "t2",
                "page": 2,
                "order": 3,
                "kind": "text",
                "content_sha256": "4" * 64,
            },
            {
                "block_id": "table1",
                "page": 2,
                "order": 4,
                "kind": "table",
                "content_sha256": "5" * 64,
            },
            {
                "block_id": "table2",
                "page": 3,
                "order": 5,
                "kind": "table",
                "content_sha256": "6" * 64,
            },
            {
                "block_id": "image1",
                "page": 3,
                "order": 6,
                "kind": "image",
                "content_sha256": "7" * 64,
            },
        ]
        documents.append(
            {
                "document_ref": document_ref,
                "source_sha256": digest,
                "language": language,
                "page_count": 3,
                "blocks": blocks,
                "heading_edges": [
                    {"child_id": "h1", "parent_id": None},
                    {"child_id": "h2", "parent_id": "h1"},
                ],
                "order_pairs": [
                    {"before_id": "h1", "after_id": "t1"},
                    {"before_id": "t1", "after_id": "h2"},
                    {"before_id": "table1", "after_id": "table2"},
                ],
                "relations": [
                    {"task": "text_continuation", "source_id": "t1", "target_id": "t2"},
                    {
                        "task": "table_continuation",
                        "source_id": "table1",
                        "target_id": "table2",
                    },
                    {"task": "image_association", "source_id": "image1", "target_id": "t2"},
                ],
            }
        )
        cases.append(
            {
                "case_id": f"case-{language}",
                "language": language,
                "relevant": [
                    {"document_ref": document_ref, "block_id": "t2", "grade": 3}
                ],
            }
        )
    return {
        "schema_version": "popo-gold-v1",
        "source_revision": "source-revision-1",
        "documents": documents,
        "downstream_cases": cases,
    }


def _variant(gold: dict[str, Any], *, candidate: bool) -> dict[str, Any]:
    documents = []
    results = []
    for gold_document in gold["documents"]:
        block_ids = [block["block_id"] for block in gold_document["blocks"]]
        from rag_app.eval.popo_gate import GoldBlock

        blocks = [GoldBlock.model_validate(block) for block in gold_document["blocks"]]
        documents.append(
            {
                "document_ref": gold_document["document_ref"],
                "source_sha256": gold_document["source_sha256"],
                "source_inventory_sha256": source_inventory_sha256(blocks),
                "node_source_map": [
                    {"node_id": f"node-{block_id}", "source_block_ids": [block_id]}
                    for block_id in block_ids
                ],
                "heading_edges": (
                    gold_document["heading_edges"]
                    if candidate
                    else [{"child_id": "h1", "parent_id": None}]
                ),
                "block_order": (
                    block_ids
                    if candidate
                    else ["h1", "h2", "t1", "t2", "table2", "table1", "image1"]
                ),
                "relations": gold_document["relations"] if candidate else [],
                "latency_ms": 150.0 if candidate else 100.0,
                "peak_vram_mib": 12_000.0 if candidate else 1_000.0,
            }
        )
        results.append(
            {
                "case_id": f"case-{gold_document['language']}",
                "ranked": [
                    {
                        "document_ref": gold_document["document_ref"],
                        "block_id": "t2",
                    }
                ],
                "cited": [
                    {
                        "document_ref": gold_document["document_ref"],
                        "block_id": "t2",
                    }
                ],
            }
        )
    return {
        "schema_version": "popo-variant-v1",
        "source_revision": gold["source_revision"],
        "variant_id": "popo" if candidate else "raw-mineru",
        "model_revision": "model-revision-1" if candidate else "raw-parser-revision-1",
        "code_revision": "code-revision-1",
        "seed": 17,
        "documents": documents,
        "downstream_results": results,
    }


def _test_policy(**overrides: Any) -> GatePolicy:
    values: dict[str, Any] = {
        "min_documents_per_language": 1,
        "min_pages_per_language": 1,
        "min_heading_edges_per_language": 1,
        "min_order_pairs_per_language": 1,
        "min_relations_per_task": 1,
        "min_relations_per_task_per_language": 1,
        "min_downstream_cases_per_language": 1,
    }
    values.update(overrides)
    return GatePolicy(**values)


def test_accepts_structural_gain_with_noninferior_downstream() -> None:
    gold = _gold()
    report = evaluate_popo_pair(
        gold, _variant(gold, candidate=False), _variant(gold, candidate=True), policy=_test_policy()
    )

    assert report["decision"]["eligible"] is True
    assert report["decision"]["accepted"] is True
    assert report["candidate"]["structure"]["overall"]["mapping_valid"] is True
    assert report["candidate"]["structure"]["overall"]["heading"]["f1"] == 1.0
    assert report["candidate"]["downstream"]["by_language"]["zh"]["recall_at_5"] == 1.0
    assert len(report["report_sha256"]) == 64


@pytest.mark.parametrize("failure", ["missing", "unknown", "digest"])
def test_source_mapping_fails_closed(failure: str) -> None:
    gold = _gold()
    candidate = _variant(gold, candidate=True)
    mapping = candidate["documents"][0]["node_source_map"]
    if failure == "missing":
        mapping.pop()
        candidate["documents"][0]["block_order"].remove("image1")
        candidate["documents"][0]["relations"] = [
            relation
            for relation in candidate["documents"][0]["relations"]
            if relation["task"] != "image_association"
        ]
    elif failure == "unknown":
        mapping[-1]["source_block_ids"] = ["not-in-source"]
        candidate["documents"][0]["block_order"][-1] = "not-in-source"
        candidate["documents"][0]["relations"] = [
            relation
            for relation in candidate["documents"][0]["relations"]
            if relation["task"] != "image_association"
        ]
    else:
        candidate["documents"][0]["source_inventory_sha256"] = "f" * 64

    report = evaluate_popo_pair(
        gold, _variant(gold, candidate=False), candidate, policy=_test_policy()
    )

    assert report["decision"]["accepted"] is False
    assert "candidate_source_mapping_invalid" in report["decision"]["failures"]


def test_rejects_cross_page_gold_continuation_on_one_page() -> None:
    gold = _gold()
    gold["documents"][0]["blocks"][3]["page"] = 1

    with pytest.raises(ValidationError, match="continuation relation must move to a later page"):
        evaluate_popo_pair(
            gold,
            _variant(_gold(), candidate=False),
            _variant(_gold(), candidate=True),
            policy=_test_policy(),
        )


def test_rejects_downstream_regression_in_single_language_slice() -> None:
    gold = _gold()
    candidate = _variant(gold, candidate=True)
    zh_result = next(row for row in candidate["downstream_results"] if row["case_id"] == "case-zh")
    zh_result["ranked"] = []
    zh_result["cited"] = []

    report = evaluate_popo_pair(
        gold, _variant(gold, candidate=False), candidate, policy=_test_policy()
    )

    assert report["decision"]["accepted"] is False
    assert "downstream_regression:zh:recall_at_5" in report["decision"]["failures"]
    assert "downstream_regression:zh:citation_f1" in report["decision"]["failures"]


def test_unknown_citation_reference_fails_closed() -> None:
    gold = _gold()
    candidate = _variant(gold, candidate=True)
    unknown = {"document_ref": "doc-ru", "block_id": "not-in-source"}
    candidate["downstream_results"][0]["ranked"] = [unknown]
    candidate["downstream_results"][0]["cited"] = [unknown]

    report = evaluate_popo_pair(
        gold, _variant(gold, candidate=False), candidate, policy=_test_policy()
    )

    assert "candidate_downstream_source_references_invalid" in report["decision"]["failures"]


def test_marks_incomplete_multilingual_gold_ineligible() -> None:
    gold = _gold()
    gold["documents"] = [document for document in gold["documents"] if document["language"] != "zh"]
    gold["downstream_cases"] = [case for case in gold["downstream_cases"] if case["language"] != "zh"]

    report = evaluate_popo_pair(
        gold, _variant(gold, candidate=False), _variant(gold, candidate=True), policy=_test_policy()
    )

    assert report["decision"]["eligible"] is False
    assert "insufficient_documents:zh" in report["decision"]["failures"]
    assert "insufficient_downstream_cases:zh" in report["decision"]["failures"]


def test_runtime_budgets_reject_candidate() -> None:
    gold = _gold()
    candidate = _variant(gold, candidate=True)
    for document in candidate["documents"]:
        document["latency_ms"] = 500.0
        document["peak_vram_mib"] = 37_000.0

    report = evaluate_popo_pair(
        gold,
        _variant(gold, candidate=False),
        candidate,
        policy=_test_policy(max_p95_latency_ratio=2.0, max_peak_vram_mib=36_000.0),
    )

    assert "latency_budget_exceeded" in report["decision"]["failures"]
    assert "vram_budget_exceeded" in report["decision"]["failures"]


def test_rejects_unlinked_source_revision() -> None:
    gold = _gold()
    candidate = _variant(gold, candidate=True)
    candidate["source_revision"] = "different-source-revision"

    with pytest.raises(ValueError, match="same source revision"):
        evaluate_popo_pair(
            gold, _variant(gold, candidate=False), candidate, policy=_test_policy()
        )


def test_report_is_deterministic_for_identical_artifacts() -> None:
    gold = _gold()
    baseline = _variant(gold, candidate=False)
    candidate = _variant(gold, candidate=True)

    first = evaluate_popo_pair(gold, baseline, candidate, policy=_test_policy())
    second = evaluate_popo_pair(
        deepcopy(gold), deepcopy(baseline), deepcopy(candidate), policy=_test_policy()
    )

    assert first == second


def test_policy_rejects_disabled_evidence_minimums() -> None:
    with pytest.raises(ValueError, match="evidence minimums must be positive"):
        GatePolicy(min_relations_per_task=0)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate_source", "duplicate source block identifiers"),
        ("duplicate_node", "duplicate node identifiers"),
        ("duplicate_order", "block order contains duplicate identifiers"),
        ("duplicate_heading_child", "more than one parent"),
        ("heading_cycle", "heading graph contains a cycle"),
        ("duplicate_relation", "duplicate structural relations"),
    ],
)
def test_variant_rejects_internal_duplicates_and_invalid_graphs(
    mutation: str, message: str
) -> None:
    gold = _gold()
    candidate = _variant(gold, candidate=True)
    document = candidate["documents"][0]
    if mutation == "duplicate_source":
        document["node_source_map"][0]["source_block_ids"].append("h1")
    elif mutation == "duplicate_node":
        document["node_source_map"][1]["node_id"] = document["node_source_map"][0]["node_id"]
    elif mutation == "duplicate_order":
        document["block_order"][-1] = document["block_order"][0]
    elif mutation == "duplicate_heading_child":
        document["heading_edges"].append({"child_id": "h2", "parent_id": None})
    elif mutation == "heading_cycle":
        document["heading_edges"][0]["parent_id"] = "h2"
    else:
        document["relations"].append(deepcopy(document["relations"][0]))

    with pytest.raises(ValidationError, match=message):
        evaluate_popo_pair(
            gold, _variant(gold, candidate=False), candidate, policy=_test_policy()
        )


@pytest.mark.parametrize("failure", ["multiple_parents", "cycle"])
def test_gold_rejects_invalid_heading_graph(failure: str) -> None:
    gold = _gold()
    edges = gold["documents"][0]["heading_edges"]
    if failure == "multiple_parents":
        edges.append({"child_id": "h2", "parent_id": None})
        message = "more than one parent"
    else:
        edges[0]["parent_id"] = "h2"
        message = "heading graph contains a cycle"

    with pytest.raises(ValidationError, match=message):
        evaluate_popo_pair(
            gold,
            _variant(_gold(), candidate=False),
            _variant(_gold(), candidate=True),
            policy=_test_policy(),
        )


@pytest.mark.parametrize("failure", ["duplicate", "contradiction"])
def test_gold_rejects_invalid_order_pairs(failure: str) -> None:
    gold = _gold()
    pairs = gold["documents"][0]["order_pairs"]
    if failure == "duplicate":
        pairs.append(deepcopy(pairs[0]))
        message = "duplicate order pairs"
    else:
        pairs.append({"before_id": "t1", "after_id": "h1"})
        message = "contradicts block order"

    with pytest.raises(ValidationError, match=message):
        evaluate_popo_pair(
            gold,
            _variant(_gold(), candidate=False),
            _variant(_gold(), candidate=True),
            policy=_test_policy(),
        )


def test_variant_heading_edges_must_reference_heading_blocks() -> None:
    gold = _gold()
    candidate = _variant(gold, candidate=True)
    candidate["documents"][0]["heading_edges"].append(
        {"child_id": "t1", "parent_id": "h1"}
    )

    with pytest.raises(ValueError, match="must reference heading blocks"):
        evaluate_popo_pair(
            gold, _variant(gold, candidate=False), candidate, policy=_test_policy()
        )


def test_variant_relations_must_match_block_kinds() -> None:
    gold = _gold()
    candidate = _variant(gold, candidate=True)
    candidate["documents"][0]["relations"][0]["target_id"] = "table2"

    with pytest.raises(ValueError, match="must connect text blocks"):
        evaluate_popo_pair(
            gold, _variant(gold, candidate=False), candidate, policy=_test_policy()
        )


def test_variant_continuations_must_move_forward_across_pages() -> None:
    gold = _gold()
    candidate = _variant(gold, candidate=True)
    candidate["documents"][0]["relations"][0] = {
        "task": "text_continuation",
        "source_id": "t2",
        "target_id": "t1",
    }

    with pytest.raises(ValueError, match="must move to a later page"):
        evaluate_popo_pair(
            gold, _variant(gold, candidate=False), candidate, policy=_test_policy()
        )


def test_citation_must_be_present_in_ranked_evidence() -> None:
    gold = _gold()
    candidate = _variant(gold, candidate=True)
    candidate["downstream_results"][0]["cited"] = [
        {"document_ref": "doc-ru", "block_id": "h1"}
    ]

    with pytest.raises(ValidationError, match="present in the ranked evidence"):
        evaluate_popo_pair(
            gold, _variant(gold, candidate=False), candidate, policy=_test_policy()
        )


def test_atomic_report_is_private(tmp_path: Path) -> None:
    project_root = Path(__file__).parents[1]
    output = tmp_path / "report.json"
    gold_path = tmp_path / "gold.json"
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    policy_path = tmp_path / "policy.json"
    gold = _gold()
    for path, payload in (
        (gold_path, gold),
        (baseline_path, _variant(gold, candidate=False)),
        (candidate_path, _variant(gold, candidate=True)),
        (policy_path, asdict(_test_policy())),
    ):
        path.write_text(json.dumps(payload), encoding="utf-8")
    output.write_text("old", encoding="utf-8")
    output.chmod(0o644)

    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "evaluate_popo_gate.py"),
            "--gold",
            str(gold_path),
            "--baseline",
            str(baseline_path),
            "--candidate",
            str(candidate_path),
            "--policy",
            str(policy_path),
            "--output",
            str(output),
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert not list(tmp_path.glob(".report.json.*.tmp"))
