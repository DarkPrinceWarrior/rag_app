#!/usr/bin/env python3
"""Зафиксировать private runtime-evidence временного реранкера для BM25 A/B."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from rag_app.config import settings
from rag_app.eval.private_artifacts import read_private_bytes, write_private_json_fresh
from rag_app.eval.reranker_runtime import capture_runtime_evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--process-argv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default=settings.rerank_base_url)
    parser.add_argument("--model", default=settings.rerank_model)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.model != settings.rerank_model:
        raise SystemExit("runtime evidence model must match RERANK_MODEL")
    raw_argv = read_private_bytes(args.process_argv, max_bytes=1024 * 1024).raw_bytes
    evidence = asyncio.run(
        capture_runtime_evidence(
            raw_argv,
            endpoint=args.endpoint,
            model=args.model,
        )
    )
    write_private_json_fresh(
        args.output,
        (
            json.dumps(evidence.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).encode(),
    )
    print(
        json.dumps(
            {
                "precision": evidence.precision,
                "template_probe_gap": evidence.relevant_score - evidence.irrelevant_score,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
