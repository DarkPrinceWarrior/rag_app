from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_filtered_hnsw.py"
_SPEC = importlib.util.spec_from_file_location("evaluate_filtered_hnsw", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
runner = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = runner
_SPEC.loader.exec_module(runner)


def _database(**changes: object) -> dict[str, object]:
    result: dict[str, object] = {
        "pgvector_version": "0.8.5",
        "alembic_revision": "0026",
        "chunk_count": 292,
        "embedded_chunk_count": 292,
        "emb_en_type": "vector(1024)",
        "emb_ru_type": "vector(1024)",
        "embedding_values_finite": True,
        "hnsw_indexes_valid": True,
        "rls_active": True,
        "role_bypass_rls": False,
        "role_owns_chunks": False,
    }
    result.update(changes)
    return result


def _plan(**changes: object) -> dict[str, object]:
    result: dict[str, object] = {
        "exact_ann_count": 0,
        "forced_dual_hnsw_coverage": 1.0,
        "natural_dual_hnsw_coverage": 1.0,
    }
    result.update(changes)
    return result


def _benchmark(**changes: object) -> dict[str, object]:
    result: dict[str, object] = {
        "off_mean_oracle_recall": 0.98,
        "strict_mean_oracle_recall": 0.995,
        "off_fill_rate": 0.98,
        "strict_fill_rate": 1.0,
        "iterative_recall_gain": 0.015,
        "strict_p95_ratio_to_exact": 1.0,
        "scope_violation_count": 0,
    }
    result.update(changes)
    return result


def test_preflight_rejects_current_small_corpus_planner() -> None:
    decision, reasons = runner.evaluate_preflight(
        release_eligible=True,
        database=_database(),
        plan=_plan(natural_dual_hnsw_coverage=0.0),
        benchmark=_benchmark(),
        database_image_digest=runner.EXPECTED_DATABASE_IMAGE_DIGEST,
        corpus_stable=True,
        rls_safe=True,
    )

    assert decision == "no_go"
    assert "production_planner_never_chooses_dual_hnsw" in reasons


def test_preflight_never_authorizes_cutover() -> None:
    assert runner.evaluate_preflight(
        release_eligible=True,
        database=_database(),
        plan=_plan(),
        benchmark=_benchmark(),
        database_image_digest=runner.EXPECTED_DATABASE_IMAGE_DIGEST,
        corpus_stable=True,
        rls_safe=True,
    ) == (
        "requires_full_ab",
        ["preflight_passed_full_paired_quality_and_concurrency_ab_required"],
    )


@pytest.mark.parametrize(
    ("database", "plan", "corpus_stable", "rls_safe", "reason"),
    [
        (_database(pgvector_version="0.8.2"), _plan(), True, True, "pgvector_version_is_not_0_8_5"),
        (
            _database(embedded_chunk_count=291),
            _plan(),
            True,
            True,
            "chunks_without_dense_embeddings_exist",
        ),
        (
            _database(),
            _plan(forced_dual_hnsw_coverage=0.5),
            True,
            True,
            "dual_language_query_is_not_hnsw_indexable_for_every_scope",
        ),
        (_database(), _plan(exact_ann_count=1), True, True, "exact_oracle_used_ann_index"),
        (_database(), _plan(), False, True, "corpus_changed_during_preflight"),
        (_database(), _plan(), True, False, "rls_cross_owner_or_anonymous_probe_failed"),
    ],
)
def test_preflight_fails_closed_for_invalid_evidence(
    database: dict[str, object],
    plan: dict[str, object],
    corpus_stable: bool,
    rls_safe: bool,
    reason: str,
) -> None:
    decision, reasons = runner.evaluate_preflight(
        release_eligible=True,
        database=database,
        plan=plan,
        benchmark=_benchmark(),
        database_image_digest=runner.EXPECTED_DATABASE_IMAGE_DIGEST,
        corpus_stable=corpus_stable,
        rls_safe=rls_safe,
    )

    assert decision == "no_go"
    assert reason in reasons


def test_debug_or_limited_run_is_release_ineligible() -> None:
    decision, reasons = runner.evaluate_preflight(
        release_eligible=False,
        database=_database(),
        plan=_plan(),
        benchmark=_benchmark(),
        database_image_digest=runner.EXPECTED_DATABASE_IMAGE_DIGEST,
        corpus_stable=True,
        rls_safe=True,
    )

    assert decision == "no_go"
    assert "debug_or_incomplete_run_is_release_ineligible" in reasons


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"strict_mean_oracle_recall": float("nan")}, "benchmark_metrics_are_not_finite"),
        ({"strict_mean_oracle_recall": 0.98}, "strict_hnsw_mean_oracle_recall_below_0_99"),
        ({"strict_fill_rate": 0.98}, "strict_hnsw_fill_rate_below_0_99"),
        ({"strict_p95_ratio_to_exact": 1.06}, "strict_hnsw_p95_regressed_over_5_percent"),
        ({"iterative_recall_gain": 0.0}, "iterative_scan_has_no_measurable_recall_gain"),
        ({"scope_violation_count": 1}, "dense_results_violated_owner_or_filter_scope"),
    ],
)
def test_benchmark_metrics_fail_closed(changes: dict[str, object], reason: str) -> None:
    decision, reasons = runner.evaluate_preflight(
        release_eligible=True,
        database=_database(),
        plan=_plan(),
        benchmark=_benchmark(**changes),
        database_image_digest=runner.EXPECTED_DATABASE_IMAGE_DIGEST,
        corpus_stable=True,
        rls_safe=True,
    )

    assert decision == "no_go"
    assert reason in reasons


def test_git_state_includes_untracked_files(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        output = "a" * 40 if command[1] == "rev-parse" else "?? new.py\n"
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    assert runner._git_state() == ("a" * 40, True)
    assert ["git", "status", "--porcelain", "--untracked-files=all"] in calls
