#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import stat
from pathlib import Path
from typing import Any

from rag_app.eval.gold_set import load_gold_set
from rag_app.eval.parser_rag_linkage import (
    ParserRagLinkageError,
    build_linkage_report,
    gold_document_snapshot,
    parser_corpus_snapshot,
)
from rag_app.eval.private_sidecar import bind_gold_sidecar, load_private_sidecar

_MAX_REPORT_BYTES = 64 * 1024 * 1024


class _DuplicateKeyError(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _load_report(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ParserRagLinkageError("parser report must not be a symlink")
    try:
        info = path.stat()
    except OSError as error:
        raise ParserRagLinkageError(
            f"unable to stat parser report ({type(error).__name__})"
        ) from None
    if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_REPORT_BYTES:
        raise ParserRagLinkageError("parser report is not a bounded regular file")
    try:
        payload = json.loads(
            path.read_bytes(),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, _DuplicateKeyError, ValueError) as error:
        raise ParserRagLinkageError(
            f"unable to load parser report ({type(error).__name__})"
        ) from None
    if not isinstance(payload, dict):
        raise ParserRagLinkageError("parser report must be a JSON object")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-closed preflight for parser A/B downstream-RAG eligibility."
    )
    parser.add_argument("baseline_report", type=Path)
    parser.add_argument("candidate_report", type=Path)
    parser.add_argument("gold", type=Path)
    parser.add_argument("sidecar", type=Path)
    parser.add_argument("--backend", default="mineru")
    parser.add_argument("--gold-mode", choices=("candidate", "release"), default="release")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        records, _ = load_gold_set(args.gold, mode=args.gold_mode, repository_root=Path.cwd())
        sidecars = load_private_sidecar(args.sidecar, repository_root=Path.cwd())
        bind_gold_sidecar(records, sidecars)
        baseline = parser_corpus_snapshot(_load_report(args.baseline_report), backend=args.backend)
        candidate = parser_corpus_snapshot(_load_report(args.candidate_report), backend=args.backend)
        report = build_linkage_report(baseline, candidate, gold_document_snapshot(records))
    except (ParserRagLinkageError, ValueError) as error:
        print(json.dumps({"schema_version": "parser-rag-linkage-v1", "eligible": False, "error": str(error)}))
        return 3
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
