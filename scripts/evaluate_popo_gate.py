#!/usr/bin/env python3
"""Run the content-free MinerU versus MinerU+Popo A/B qualification gate."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from rag_app.eval.popo_gate import GatePolicy, evaluate_popo_pair


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        os.chmod(path, 0o600)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qualify deterministic raw-MinerU and MinerU+Popo artifacts."
    )
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--policy",
        type=Path,
        help="Optional JSON object overriding all GatePolicy fields.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    policy_payload = _read_json(args.policy) if args.policy else {}
    policy = GatePolicy(**policy_payload)
    report = evaluate_popo_pair(
        _read_json(args.gold),
        _read_json(args.baseline),
        _read_json(args.candidate),
        policy=policy,
    )
    _write_json_atomic(args.output, report)
    decision = report["decision"]
    print(
        json.dumps(
            {
                "accepted": decision["accepted"],
                "eligible": decision["eligible"],
                "failures": decision["failures"],
                "report_sha256": report["report_sha256"],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0 if decision["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
