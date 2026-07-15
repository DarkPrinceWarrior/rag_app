from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "qualify_direct_reranker.py"
    spec = importlib.util.spec_from_file_location("qualify_direct_reranker", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_SCRIPT = _load_script()
GoldCase = _SCRIPT.GoldCase
_aggregate = _SCRIPT._aggregate
build_candidates = _SCRIPT.build_candidates
load_gold_cases = _SCRIPT.load_gold_cases


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_load_gold_cases_keeps_only_answerable_exact_evidence(tmp_path: Path) -> None:
    gold = tmp_path / "gold.jsonl"
    sidecar = tmp_path / "sidecar.jsonl"
    _write_jsonl(
        gold,
        [
            {"case_id": "kept", "answerable": True, "language": "ru", "question": "Q"},
            {"case_id": "skipped", "answerable": False, "language": "en", "question": "Q"},
        ],
    )
    _write_jsonl(
        sidecar,
        [
            {
                "case_id": "kept",
                "exact_evidence": [{"exact_quote": "  evidence  "}],
                "quantities": {"expected": ["24.75 MPa"]},
            },
            {"case_id": "skipped", "exact_evidence": [{"exact_quote": "other"}]},
        ],
    )

    assert load_gold_cases(gold, sidecar) == [
        GoldCase(
            case_id="kept",
            language="ru",
            question="Q",
            positive="evidence",
            positive_sha256="ee8250fb76e094b34b471f13a73dbbe51d1ae142e9df59d7c0d31ec20f0a0a8e",
            numeric=True,
        )
    ]


def test_build_candidates_is_deterministic_and_preserves_positive() -> None:
    pool = [
        GoldCase(str(index), "ru", f"q{index}", f"e{index}", f"sha{index}", False)
        for index in range(8)
    ]

    first = build_candidates(pool[0], pool, 5)
    second = build_candidates(pool[0], list(reversed(pool)), 5)

    assert first == second
    documents, positive_index = first
    assert len(documents) == 6
    assert documents[positive_index] == "e0"


def test_aggregate_reports_rank_metrics() -> None:
    assert _aggregate([{"rank": 1}, {"rank": 2}, {"rank": 4}]) == {
        "cases": 3,
        "top1": 1 / 3,
        "recall_at_3": 2 / 3,
        "mrr": (1 + 1 / 2 + 1 / 4) / 3,
    }
