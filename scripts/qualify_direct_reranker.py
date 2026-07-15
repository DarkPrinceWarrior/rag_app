"""Rank-only qualification for the deterministic Qwen3 reranker runtime.

The evidence artifact intentionally excludes questions, excerpts and raw model
scores.  It contains only source hashes, stable case identifiers, ranks and
aggregate metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from rag_app.llm.embeddings import build_rerank_payload, reranker_template_sha256


@dataclass(frozen=True, slots=True)
class GoldCase:
    case_id: str
    language: str
    question: str
    positive: str
    positive_sha256: str
    numeric: bool


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_gold_cases(gold_path: Path, sidecar_path: Path) -> list[GoldCase]:
    sidecars = {row["case_id"]: row for row in _load_jsonl(sidecar_path)}
    cases: list[GoldCase] = []
    for record in _load_jsonl(gold_path):
        sidecar = sidecars.get(record["case_id"])
        if not record.get("answerable") or not sidecar or not sidecar.get("exact_evidence"):
            continue
        quotes = [
            item["exact_quote"].strip()
            for item in sidecar["exact_evidence"]
            if item.get("exact_quote", "").strip()
        ]
        if not quotes:
            continue
        positive = "\n".join(quotes)[:4000]
        cases.append(
            GoldCase(
                case_id=record["case_id"],
                language=record["language"],
                question=record["question"],
                positive=positive,
                positive_sha256=hashlib.sha256(positive.encode()).hexdigest(),
                numeric=bool(sidecar.get("quantities", {}).get("expected")),
            )
        )
    return sorted(cases, key=lambda item: item.case_id)


def _stable_order(case_id: str, value: str) -> str:
    return hashlib.sha256(f"{case_id}:{value}".encode()).hexdigest()


def build_candidates(case: GoldCase, pool: list[GoldCase], count: int) -> tuple[list[str], int]:
    distractors = sorted(
        (
            item
            for item in pool
            if item.case_id != case.case_id and item.positive_sha256 != case.positive_sha256
        ),
        key=lambda item: _stable_order(case.case_id, item.case_id),
    )[:count]
    if len(distractors) != count:
        raise RuntimeError(f"not enough {case.language} distractors for {case.case_id}")
    tagged = [(case.positive, True), *((item.positive, False) for item in distractors)]
    tagged.sort(key=lambda item: _stable_order(case.case_id, hashlib.sha256(item[0].encode()).hexdigest()))
    return [text for text, _ in tagged], next(i for i, (_, positive) in enumerate(tagged) if positive)


def rerank(endpoint: str, query: str, documents: list[str], timeout_s: float) -> tuple[int, ...]:
    payload = build_rerank_payload(query, documents)
    with httpx.Client(timeout=timeout_s) as client:
        response = client.post(f"{endpoint.rstrip('/')}/v1/rerank", json=payload)
        response.raise_for_status()
    body = response.json()
    ranking = tuple(int(item["index"]) for item in body["results"])
    if sorted(ranking) != list(range(len(documents))):
        raise RuntimeError("reranker returned an invalid permutation")
    return ranking


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        return {"cases": 0, "top1": 0.0, "recall_at_3": 0.0, "mrr": 0.0}
    return {
        "cases": len(rows),
        "top1": sum(row["rank"] == 1 for row in rows) / len(rows),
        "recall_at_3": sum(row["rank"] <= 3 for row in rows) / len(rows),
        "mrr": sum(1.0 / row["rank"] for row in rows) / len(rows),
    }


def _synthetic_cases() -> list[tuple[str, str, list[str], int]]:
    return [
        (
            "syn-en",
            "Which hydrostatic test pressure is required for the vessel?",
            [
                "The coating dry film thickness shall be 320 micrometres.",
                "Hydrostatic testing shall be performed at 24.75 MPa for 30 minutes.",
                "The design temperature is minus 40 degrees Celsius.",
                "All welds require full radiographic examination.",
            ],
            1,
        ),
        (
            "syn-ru",
            "Какое давление требуется при гидравлическом испытании сосуда?",
            [
                "Толщина сухой плёнки покрытия должна составлять 320 мкм.",
                "Все сварные соединения подлежат радиографическому контролю.",
                "Гидравлическое испытание выполняют при 24,75 МПа в течение 30 минут.",
                "Расчётная температура составляет минус 40 градусов Цельсия.",
            ],
            2,
        ),
        (
            "syn-zh",
            "容器水压试验需要多大压力？",
            [
                "设计温度为零下40摄氏度。",
                "所有焊缝均需进行射线检测。",
                "涂层干膜厚度应为320微米。",
                "水压试验应在24.75兆帕下持续30分钟。",
            ],
            3,
        ),
        (
            "syn-numeric",
            "При каком давлении проводят испытание: 16,5 или 24,75 МПа?",
            [
                "Максимально допустимое рабочее давление составляет 16,5 МПа.",
                "Испытательное давление равно 24,75 МПа; выдержка 30 минут.",
                "Давление настройки предохранительного клапана равно 17,2 МПа.",
                "Расчётное давление трубопровода составляет 10,0 МПа.",
            ],
            1,
        ),
    ]


def qualify(args: argparse.Namespace) -> dict[str, Any]:
    gold_path = Path(args.gold)
    sidecar_path = Path(args.sidecar)
    cases = load_gold_cases(gold_path, sidecar_path)
    by_language: dict[str, list[GoldCase]] = defaultdict(list)
    for case in cases:
        by_language[case.language].append(case)

    rows: list[dict[str, Any]] = []
    deterministic = True
    for case in cases:
        documents, positive_index = build_candidates(
            case, by_language[case.language], args.distractors
        )
        first = rerank(args.endpoint, case.question, documents, args.timeout)
        second = rerank(args.endpoint, case.question, documents, args.timeout)
        deterministic = deterministic and first == second
        rank = first.index(positive_index) + 1
        rows.append(
            {
                "case_id": case.case_id,
                "language": case.language,
                "numeric": case.numeric,
                "rank": rank,
                "ranking_sha256": hashlib.sha256(
                    json.dumps(first, separators=(",", ":")).encode()
                ).hexdigest(),
            }
        )

    synthetic_rows = []
    for case_id, query, documents, positive_index in _synthetic_cases():
        rankings = [rerank(args.endpoint, query, documents, args.timeout) for _ in range(3)]
        deterministic = deterministic and rankings[0] == rankings[1] == rankings[2]
        synthetic_rows.append({"case_id": case_id, "rank": rankings[0].index(positive_index) + 1})

    grouped = {
        language: _aggregate([row for row in rows if row["language"] == language])
        for language in sorted(by_language)
    }
    grouped["numeric"] = _aggregate([row for row in rows if row["numeric"]])
    aggregate = _aggregate(rows)
    synthetic_top1 = sum(row["rank"] == 1 for row in synthetic_rows)
    gates = {
        "rank_deterministic": deterministic,
        "gold_top1_at_least_0_75": aggregate["top1"] >= 0.75,
        "gold_recall_at_3_at_least_0_95": aggregate["recall_at_3"] >= 0.95,
        "synthetic_top1_4_of_4": synthetic_top1 == 4,
    }
    return {
        "schema_version": "direct-qwen3-reranker-qualification-v1",
        "status": "GO" if all(gates.values()) else "NO-GO",
        "endpoint": args.endpoint,
        "method": {
            "runtime": "transformers-direct-causal-lm",
            "micro_batch_size": 1,
            "comparison": "ranks-only",
            "repeats_gold": 2,
            "repeats_synthetic": 3,
            "distractors_per_gold_case": args.distractors,
            "template_sha256": reranker_template_sha256(),
        },
        "sources": {
            "gold_sha256": _sha256_file(gold_path),
            "sidecar_sha256": _sha256_file(sidecar_path),
        },
        "metrics": {
            "gold": aggregate,
            "by_slice": grouped,
            "synthetic_top1": synthetic_top1,
            "synthetic_cases": len(synthetic_rows),
        },
        "gates": gates,
        "cases": rows,
        "synthetic": synthetic_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8003")
    parser.add_argument("--output", required=True)
    parser.add_argument("--distractors", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    if args.distractors < 1:
        parser.error("--distractors must be positive")
    result = qualify(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output.chmod(0o600)
    summary = {"status": result["status"], "metrics": result["metrics"], "gates": result["gates"]}
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
