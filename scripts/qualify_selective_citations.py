#!/usr/bin/env python3
"""Калибровка selective verifier по обезличенному claim-score JSON корпуса 236."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from rag_app.eval.citation_calibration import CalibrationCase, calibrate_threshold
from rag_app.eval.gold_set import ensure_private_gold_path
from rag_app.eval.private_artifacts import read_private_bytes, write_private_json_fresh

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold-start", type=float, default=0.30)
    parser.add_argument("--threshold-stop", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--answerability-target", type=float, default=0.85)
    parser.add_argument("--semantic-precision-target", type=float, default=0.90)
    parser.add_argument("--expected-cases", type=int, default=236)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    observations_path = ensure_private_gold_path(args.observations, REPOSITORY_ROOT)
    output_path = ensure_private_gold_path(args.output, REPOSITORY_ROOT)
    payload: Any = json.loads(
        read_private_bytes(observations_path, max_bytes=64 * 1024 * 1024).raw_bytes
    )
    raw_cases = payload.get("cases") if isinstance(payload, dict) else payload
    cases = TypeAdapter(list[CalibrationCase]).validate_python(raw_cases)
    if len(cases) != args.expected_cases:
        raise SystemExit(f"expected {args.expected_cases} cases, got {len(cases)}")
    if args.threshold_step <= 0 or args.threshold_stop < args.threshold_start:
        raise SystemExit("invalid threshold range")
    steps = int(round((args.threshold_stop - args.threshold_start) / args.threshold_step))
    thresholds = [args.threshold_start + index * args.threshold_step for index in range(steps + 1)]
    result = calibrate_threshold(
        cases,
        thresholds,
        answerability_target=args.answerability_target,
        semantic_precision_target=args.semantic_precision_target,
    )
    write_private_json_fresh(
        output_path,
        (
            json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).encode(),
    )
    print(json.dumps({"qualified": result.qualified, "threshold": result.selected_threshold}))
    return 0 if result.qualified else 2


if __name__ == "__main__":
    raise SystemExit(main())
