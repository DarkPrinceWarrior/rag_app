from __future__ import annotations

import copy
import hashlib
import json
import runpy
from pathlib import Path
from typing import Any

import pytest

_MODULE = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "compare_parser_benchmarks.py"))
ParserComparisonError = _MODULE["ParserComparisonError"]
compare_summaries = _MODULE["compare_summaries"]
two_sided_sign_test = _MODULE["two_sided_sign_test"]


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _result(index: int, *, candidate: bool, latency: float | None = None) -> dict[str, Any]:
    category = ("table", "chart", "layout")[index % 3]
    benchmark: dict[str, Any] = {
        "bbox_valid_ratio": 1.0,
        "reading_order_score": 0.95,
    }
    if category == "table":
        benchmark.update(
            table_detected=True,
            table_cell_count_ratio=0.9,
            table_row_count_ratio=1.0,
        )
    if category == "chart":
        benchmark["visual_region_preserved"] = True
    return {
        "status": "ok",
        "latency_s": latency if latency is not None else (8.0 if candidate else 10.0),
        "quality": {
            "score": 0.92,
            "raw_parser": {"score": 0.88},
            "backfilled_page_ratio": 0.05,
        },
        "bbox_segments": 3,
        "reading_order_evidence": True,
        "benchmark": benchmark,
        "source_chars": 100,
        "text_sha256": _sha(f"{'candidate' if candidate else 'baseline'}-{index}"),
        "adjacent_duplicate_character_ratio": 0.01,
        "raw_stats": {
            "bbox_segments": 3,
            "reading_order_evidence": True,
            "source_chars": 100,
            "text_sha256": _sha(f"raw-{'candidate' if candidate else 'baseline'}-{index}"),
            "adjacent_duplicate_character_ratio": 0.012,
            "bbox_valid_ratio": 1.0,
            "reading_order_score": 0.95,
        },
    }


def _summary(*, candidate: bool, page_count: int = 24) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for index in range(page_count):
        category = ("table", "chart", "layout")[index % 3]
        filename = f"page-{index:03d}.pdf"
        results[filename] = {
            "category": category,
            "selection": {"hard": True, "index": index},
            "source_sha256": _sha(filename),
            "mineru": _result(index, candidate=candidate),
        }
    return {
        "benchmark_schema_version": 2,
        "run_label": "mineru-3.4.4" if candidate else "mineru-3.3.1",
        "runtime_provenance": {
            "client": {"version": "3.4.4" if candidate else "3.3.1"},
            "server": {
                "started_at": ("2026-07-14T10:00:00+00:00" if candidate else "2026-07-14T09:00:00+00:00"),
                "endpoint": "http://127.0.0.1:30011" if candidate else "http://127.0.0.1:30010",
                "vllm_version": "0.21.0",
                "mineru_version": "3.4.4" if candidate else "3.3.1",
            },
            "model": {
                "snapshot_sha": "a" * 40,
                "manifest_sha256": _sha("model-manifest"),
            },
            "evidence": {
                "path": "/runtime/candidate.json" if candidate else "/runtime/baseline.json",
                "sha256": _sha("candidate-runtime" if candidate else "baseline-runtime"),
            },
            "controlled": {
                "parser_backend": "vlm-http-client",
                "table_enable": False,
                "server_inference_args": ["--gpu-memory-utilization", "0.30"],
                "repetition_penalty": 1.1,
                "sampling_patch_sha256": _sha("mineru-sampling-patch"),
            },
        },
        "source": "test/ParseBench",
        "source_revision": "pinned-revision",
        "backends": ["mineru"],
        "results": results,
    }


def test_accepts_complete_nonregressing_candidate_with_significant_speedup() -> None:
    report = compare_summaries(_summary(candidate=False), _summary(candidate=True))

    assert report["status"] == "accepted"
    assert report["paired_errors"] == {
        "both_ok": 24,
        "baseline_only_error": 0,
        "candidate_only_error": 0,
        "both_error": 0,
    }
    assert report["latency_s"]["baseline"] == {"mean": 10.0, "median": 10.0, "p95": 10.0}
    assert report["latency_s"]["candidate"] == {"mean": 8.0, "median": 8.0, "p95": 8.0}
    assert report["latency_s"]["speedup_baseline_over_candidate"]["median"] == 1.25
    assert report["latency_s"]["two_sided_sign_test"]["p_value"] < 0.05
    assert report["quality"]["raw_parser"]["delta"] == 0.0
    assert report["backfill_ratio"]["delta"] == 0.0
    assert report["text_hash_identity"] == {
        "final": {
            "paired_count": 24,
            "identical_count": 0,
            "changed_count": 24,
            "identity_rate": 0.0,
        },
        "raw_parser": {
            "paired_count": 24,
            "identical_count": 0,
            "changed_count": 24,
            "identity_rate": 0.0,
        },
    }
    assert report["runtime_provenance"]["baseline"]["client"]["version"] == "3.3.1"
    assert report["runtime_provenance"]["candidate"]["client"]["version"] == "3.4.4"
    assert report["adjacent_duplicate_characters"]["raw_parser"]["delta"] == 0.0
    assert report["gate"]["regressions"] == []
    assert report["gate"]["missing_evidence"] == []
    assert "latency_improvement" in report["gate"]["improvements"]


def test_rejects_new_candidate_error_even_when_remaining_pages_are_faster() -> None:
    baseline = _summary(candidate=False)
    candidate = _summary(candidate=True)
    candidate["results"]["page-000.pdf"]["mineru"] = {
        "status": "error",
        "error": "timeout",
    }

    report = compare_summaries(baseline, candidate)

    assert report["status"] == "rejected"
    assert report["paired_errors"]["candidate_only_error"] == 1
    assert "new_candidate_errors" in report["gate"]["regressions"]


def test_rejects_raw_quality_and_reading_order_regressions() -> None:
    baseline = _summary(candidate=False)
    candidate = _summary(candidate=True)
    for page in candidate["results"].values():
        page["mineru"]["quality"]["raw_parser"]["score"] = 0.80
        page["mineru"]["benchmark"]["reading_order_score"] = 0.80

    report = compare_summaries(baseline, candidate)

    assert report["status"] == "rejected"
    assert "raw_quality_regression" in report["gate"]["regressions"]
    assert "reading_order_score_regression" in report["gate"]["regressions"]


def test_inconclusive_when_evidence_is_too_small_or_structural_fields_are_missing() -> None:
    baseline = _summary(candidate=False, page_count=8)
    candidate = _summary(candidate=True, page_count=8)
    for summary in (baseline, candidate):
        for page in summary["results"].values():
            page["mineru"]["benchmark"].pop("bbox_valid_ratio")

    report = compare_summaries(baseline, candidate)

    assert report["status"] == "inconclusive"
    assert "insufficient_paired_pages" in report["gate"]["missing_evidence"]
    assert "insufficient_bbox_valid_ratio_evidence" in report["gate"]["missing_evidence"]


def test_inconclusive_when_raw_hash_or_raw_structure_is_missing() -> None:
    baseline = _summary(candidate=False)
    candidate = _summary(candidate=True)
    for page in candidate["results"].values():
        page["mineru"]["raw_stats"].pop("text_sha256")
        page["mineru"]["raw_stats"].pop("bbox_valid_ratio")

    report = compare_summaries(baseline, candidate)

    assert report["status"] == "inconclusive"
    assert "missing_raw_parser_text_hashes" in report["gate"]["missing_evidence"]
    assert "missing_raw_bbox_valid_ratio" in report["gate"]["missing_evidence"]


def test_inconclusive_when_bbox_and_reading_order_have_no_real_evidence() -> None:
    baseline = _summary(candidate=False)
    candidate = _summary(candidate=True)
    for summary in (baseline, candidate):
        for page in summary["results"].values():
            result = page["mineru"]
            result["bbox_segments"] = 0
            result["reading_order_evidence"] = False
            result["raw_stats"]["bbox_segments"] = 0
            result["raw_stats"]["reading_order_evidence"] = False

    report = compare_summaries(baseline, candidate)

    assert report["status"] == "inconclusive"
    assert "insufficient_bbox_valid_ratio_evidence" in report["gate"]["missing_evidence"]
    assert "insufficient_reading_order_score_evidence" in report["gate"]["missing_evidence"]


def test_rejects_candidate_that_loses_baseline_bbox_evidence() -> None:
    baseline = _summary(candidate=False)
    candidate = _summary(candidate=True)
    for page in candidate["results"].values():
        result = page["mineru"]
        result["bbox_segments"] = 0
        result["raw_stats"]["bbox_segments"] = 0

    report = compare_summaries(baseline, candidate)

    assert report["status"] == "rejected"
    assert report["structure"]["bbox_valid_ratio"]["delta"] == -1.0
    assert "bbox_valid_ratio_regression" in report["gate"]["regressions"]
    assert "raw_bbox_valid_ratio_regression" in report["gate"]["regressions"]


def test_unannotated_tables_do_not_require_cell_or_row_coverage() -> None:
    baseline = _summary(candidate=False)
    candidate = _summary(candidate=True)
    for summary in (baseline, candidate):
        for page in summary["results"].values():
            if page["category"] == "table":
                page["mineru"]["benchmark"].pop("table_cell_count_ratio")
                page["mineru"]["benchmark"].pop("table_row_count_ratio")

    report = compare_summaries(baseline, candidate)

    assert report["status"] == "accepted"
    assert report["structure"]["table_detection_rate"]["paired_count"] == 8
    assert report["structure"]["table_cell_coverage"]["paired_count"] == 0
    assert report["structure"]["table_row_coverage"]["paired_count"] == 0
    assert "missing_table_cell_coverage" not in report["gate"]["missing_evidence"]
    assert "missing_table_row_coverage" not in report["gate"]["missing_evidence"]


def test_annotated_tables_require_matching_cell_and_row_evidence() -> None:
    baseline = _summary(candidate=False)
    candidate = _summary(candidate=True)
    for summary in (baseline, candidate):
        for page in summary["results"].values():
            if page["category"] == "table":
                page["selection"].update(cells=20, rows=4)
    for page in candidate["results"].values():
        if page["category"] == "table":
            page["mineru"]["benchmark"].pop("table_cell_count_ratio")
            page["mineru"]["benchmark"].pop("table_row_count_ratio")

    report = compare_summaries(baseline, candidate)

    assert report["status"] == "inconclusive"
    assert "missing_table_cell_coverage" in report["gate"]["missing_evidence"]
    assert "missing_table_row_coverage" in report["gate"]["missing_evidence"]


def test_rejects_table_oversegmentation_instead_of_clipping_it_to_perfect() -> None:
    baseline = _summary(candidate=False)
    candidate = _summary(candidate=True)
    for summary in (baseline, candidate):
        for page in summary["results"].values():
            if page["category"] == "table":
                page["selection"].update(cells=20, rows=4)
    for page in baseline["results"].values():
        if page["category"] == "table":
            page["mineru"]["benchmark"]["table_cell_count_ratio"] = 1.0
            page["mineru"]["benchmark"]["table_row_count_ratio"] = 1.0
    for page in candidate["results"].values():
        if page["category"] == "table":
            page["mineru"]["benchmark"]["table_cell_count_ratio"] = 2.0
            page["mineru"]["benchmark"]["table_row_count_ratio"] = 2.0

    report = compare_summaries(baseline, candidate)

    assert report["status"] == "rejected"
    assert "table_cell_coverage_regression" in report["gate"]["regressions"]
    assert "table_row_coverage_regression" in report["gate"]["regressions"]


def test_rejects_p95_latency_regression_even_when_median_is_faster() -> None:
    baseline = _summary(candidate=False)
    candidate = _summary(candidate=True)
    candidate["results"]["page-022.pdf"]["mineru"]["latency_s"] = 30.0
    candidate["results"]["page-023.pdf"]["mineru"]["latency_s"] = 30.0

    report = compare_summaries(baseline, candidate)

    assert report["latency_s"]["speedup_baseline_over_candidate"]["median"] == 1.25
    assert report["latency_s"]["speedup_baseline_over_candidate"]["p95"] < 1.0
    assert report["status"] == "rejected"
    assert "latency_regression" in report["gate"]["regressions"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("category", "unknown", "invalid category"),
        ("selection", None, "selection must be an object"),
    ],
)
def test_rejects_invalid_page_contract(field: str, value: Any, message: str) -> None:
    baseline = _summary(candidate=False)
    candidate = _summary(candidate=True)
    candidate["results"]["page-000.pdf"][field] = value

    with pytest.raises(ParserComparisonError, match=message):
        compare_summaries(baseline, candidate)


def test_inconclusive_when_both_versions_fail_the_same_pages() -> None:
    baseline = _summary(candidate=False)
    candidate = _summary(candidate=True)
    for summary in (baseline, candidate):
        summary["results"]["page-002.pdf"]["mineru"] = {
            "status": "error",
            "error": "timeout",
        }

    report = compare_summaries(baseline, candidate)

    assert report["status"] == "inconclusive"
    assert "unresolved_paired_errors" in report["gate"]["missing_evidence"]


def test_rejects_adjacent_duplicate_character_regression_in_final_and_raw_text() -> None:
    baseline = _summary(candidate=False)
    candidate = _summary(candidate=True)
    for page in candidate["results"].values():
        page["mineru"]["adjacent_duplicate_character_ratio"] = 0.03
        page["mineru"]["raw_stats"]["adjacent_duplicate_character_ratio"] = 0.04

    report = compare_summaries(baseline, candidate)

    assert report["status"] == "rejected"
    assert "final_adjacent_duplicate_character_regression" in report["gate"]["regressions"]
    assert "raw_parser_adjacent_duplicate_character_regression" in report["gate"]["regressions"]


def test_duplicate_gate_excludes_pairs_where_either_parser_has_no_text() -> None:
    baseline = _summary(candidate=False)
    candidate = _summary(candidate=True)
    for index in range(4):
        baseline_result = baseline["results"][f"page-{index:03d}.pdf"]["mineru"]
        candidate_result = candidate["results"][f"page-{index:03d}.pdf"]["mineru"]
        baseline_result["source_chars"] = 0
        baseline_result["raw_stats"]["source_chars"] = 0
        candidate_result["adjacent_duplicate_character_ratio"] = 0.5
        candidate_result["raw_stats"]["adjacent_duplicate_character_ratio"] = 0.5

    report = compare_summaries(baseline, candidate)

    assert report["status"] == "accepted"
    assert report["adjacent_duplicate_characters"]["final"] == {
        "paired_count": 20,
        "baseline_mean": 0.01,
        "candidate_mean": 0.01,
        "delta": 0.0,
        "eligible_paired_count": 20,
        "excluded_empty_text_pairs": 4,
    }
    assert "final_adjacent_duplicate_character_regression" not in report["gate"]["regressions"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "runtime_provenance"),
        ("same_client", "client versions"),
        ("mismatched_client_server", "client and server MinerU versions"),
        ("different_vllm", "vLLM version"),
        ("different_model", "model manifest"),
        ("bad_model_snapshot", "model.snapshot_sha"),
        ("same_server_identity", "server runtime identity"),
        ("same_evidence", "runtime evidence SHA256"),
        ("bad_started_at", "ISO 8601"),
        ("bad_evidence_hash", "evidence.sha256"),
        ("different_controlled", "controlled runtime settings"),
    ],
)
def test_runtime_provenance_fails_closed(mutation: str, message: str) -> None:
    baseline = _summary(candidate=False)
    candidate = _summary(candidate=True)
    runtime = candidate["runtime_provenance"]
    if mutation == "missing":
        candidate.pop("runtime_provenance")
    elif mutation == "same_client":
        runtime["client"]["version"] = "3.3.1"
    elif mutation == "mismatched_client_server":
        runtime["client"]["version"] = "3.4.5"
    elif mutation == "different_vllm":
        runtime["server"]["vllm_version"] = "0.22.0"
    elif mutation == "different_model":
        runtime["model"]["manifest_sha256"] = _sha("other-model")
    elif mutation == "bad_model_snapshot":
        runtime["model"]["snapshot_sha"] = "main"
    elif mutation == "same_server_identity":
        runtime["server"]["started_at"] = baseline["runtime_provenance"]["server"]["started_at"]
        runtime["server"]["endpoint"] = baseline["runtime_provenance"]["server"]["endpoint"]
    elif mutation == "same_evidence":
        runtime["evidence"]["sha256"] = baseline["runtime_provenance"]["evidence"]["sha256"]
    elif mutation == "bad_started_at":
        runtime["server"]["started_at"] = "today"
    elif mutation == "different_controlled":
        runtime["controlled"]["table_enable"] = True
    else:
        runtime["evidence"]["sha256"] = "not-a-digest"

    with pytest.raises(ParserComparisonError, match=message):
        compare_summaries(baseline, candidate)


def test_runtime_evidence_files_are_verified_when_requested(tmp_path: Path) -> None:
    baseline = _summary(candidate=False)
    candidate = _summary(candidate=True)
    baseline_evidence = tmp_path / "baseline.txt"
    candidate_evidence = tmp_path / "candidate.txt"
    baseline_evidence.write_text("baseline runtime", encoding="utf-8")
    candidate_evidence.write_text("candidate runtime", encoding="utf-8")
    for summary, evidence in (
        (baseline, baseline_evidence),
        (candidate, candidate_evidence),
    ):
        summary["runtime_provenance"]["evidence"] = {
            "path": str(evidence),
            "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
        }

    report = compare_summaries(baseline, candidate, verify_runtime_evidence=True)
    assert report["status"] == "accepted"

    candidate_evidence.write_text("tampered runtime", encoding="utf-8")
    with pytest.raises(ParserComparisonError, match="evidence SHA256 mismatch"):
        compare_summaries(baseline, candidate, verify_runtime_evidence=True)


def test_runtime_evidence_rejects_relative_paths_and_symlinks(tmp_path: Path) -> None:
    baseline = _summary(candidate=False)
    candidate = _summary(candidate=True)
    baseline["runtime_provenance"]["evidence"]["path"] = "relative.txt"
    with pytest.raises(ParserComparisonError, match="path must be absolute"):
        compare_summaries(baseline, candidate, verify_runtime_evidence=True)

    target = tmp_path / "candidate.txt"
    target.write_text("candidate runtime", encoding="utf-8")
    link = tmp_path / "candidate-link.txt"
    link.symlink_to(target)
    baseline_evidence = tmp_path / "baseline.txt"
    baseline_evidence.write_text("baseline runtime", encoding="utf-8")
    baseline["runtime_provenance"]["evidence"] = {
        "path": str(baseline_evidence),
        "sha256": hashlib.sha256(baseline_evidence.read_bytes()).hexdigest(),
    }
    candidate["runtime_provenance"]["evidence"] = {
        "path": str(link),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
    with pytest.raises(ParserComparisonError, match="regular non-symlink"):
        compare_summaries(baseline, candidate, verify_runtime_evidence=True)


@pytest.mark.parametrize("mutation", ["sha", "selection", "pages", "revision"])
def test_rejects_incomparable_corpus(mutation: str) -> None:
    baseline = _summary(candidate=False)
    candidate = _summary(candidate=True)
    if mutation == "sha":
        candidate["results"]["page-000.pdf"]["source_sha256"] = "f" * 64
    elif mutation == "selection":
        candidate["results"]["page-000.pdf"]["selection"] = {"hard": False}
    elif mutation == "pages":
        candidate["results"].pop("page-000.pdf")
    else:
        candidate["source_revision"] = "different"

    with pytest.raises(ParserComparisonError):
        compare_summaries(baseline, candidate)


def test_sign_test_excludes_ties_and_uses_exact_two_sided_probability() -> None:
    result = two_sided_sign_test(
        [10.0, 10.0, 10.0, 10.0, 10.0],
        [8.0, 8.0, 8.0, 12.0, 10.0],
    )

    assert result == {
        "candidate_faster": 3,
        "candidate_slower": 1,
        "ties": 1,
        "non_tied_count": 4,
        "p_value": 0.625,
    }


def test_default_backend_and_json_artifact_round_trip(tmp_path: Path) -> None:
    baseline = _summary(candidate=False)
    candidate = _summary(candidate=True)
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    loaded_baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    loaded_candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    report = compare_summaries(loaded_baseline, loaded_candidate)

    assert report["backend"] == "mineru"
    assert report["corpus"]["page_count"] == 24
    assert len(report["corpus"]["manifest_sha256"]) == 64


def test_inputs_are_not_mutated() -> None:
    baseline = _summary(candidate=False)
    candidate = _summary(candidate=True)
    original_baseline = copy.deepcopy(baseline)
    original_candidate = copy.deepcopy(candidate)

    compare_summaries(baseline, candidate)

    assert baseline == original_baseline
    assert candidate == original_candidate
