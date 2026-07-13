"""Validate a private RAG gold-set JSONL or export its public JSON Schema."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag_app.eval.gold_set import GoldSetValidationError, gold_record_json_schema, load_gold_set


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, help="private JSONL outside git or below .private/")
    parser.add_argument("--mode", choices=("candidate", "release"), default="candidate")
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--schema-out", type=Path, help="write the non-sensitive per-record JSON Schema")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.schema_out is not None:
        args.schema_out.parent.mkdir(parents=True, exist_ok=True)
        args.schema_out.write_text(
            json.dumps(gold_record_json_schema(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(args.schema_out)
        return 0
    if args.path is None:
        raise SystemExit("path is required unless --schema-out is used")
    try:
        _, report = load_gold_set(
            args.path,
            mode=args.mode,
            repository_root=args.repository_root,
        )
    except GoldSetValidationError as error:
        print(f"gold set rejected: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
