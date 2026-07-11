"""Скачать воспроизводимый набор сложных форм VAREX с JSON Schema и эталоном."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

_DATASET = "ibm-research/VAREX"
_ROWS_URL = "https://datasets-server.huggingface.co/rows"
_PDF_BASE = "https://huggingface.co/datasets/ibm-research/VAREX/resolve/main/pdfs/"
_PAGE_SIZE = 100
_MAX_RETRIES = 8
_SAFE_FILENAME = re.compile(r"[^0-9A-Za-z._-]+")


def _open_url(request: Request, timeout_s: float) -> Any:
    for attempt in range(_MAX_RETRIES):
        try:
            return urlopen(request, timeout=timeout_s)
        except HTTPError as exc:
            if exc.code < 500 or attempt == _MAX_RETRIES - 1:
                raise
        except (TimeoutError, URLError):
            if attempt == _MAX_RETRIES - 1:
                raise
        time.sleep(min(30.0, 2.0**attempt))
    raise RuntimeError("unreachable retry state")


def _json_url(url: str, timeout_s: float) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": "DocRAGenslate-eval/1.0"})
    with _open_url(request, timeout_s) as response:
        return json.load(response)


def _download(url: str, destination: Path, timeout_s: float) -> None:
    request = Request(url, headers={"User-Agent": "DocRAGenslate-eval/1.0"})
    part = destination.with_suffix(destination.suffix + ".part")
    with _open_url(request, timeout_s) as response, part.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    part.replace(destination)


def _schema_complexity(value: Any, depth: int = 0) -> tuple[int, int, int]:
    if not isinstance(value, dict):
        return 0, depth, 0
    properties = value.get("properties")
    leaves = 0
    max_depth = depth
    arrays = int(value.get("type") == "array")
    if isinstance(properties, dict):
        for child in properties.values():
            child_leaves, child_depth, child_arrays = _schema_complexity(child, depth + 1)
            leaves += child_leaves or 1
            max_depth = max(max_depth, child_depth)
            arrays += child_arrays
    items = value.get("items")
    if isinstance(items, dict):
        child_leaves, child_depth, child_arrays = _schema_complexity(items, depth + 1)
        leaves += child_leaves
        max_depth = max(max_depth, child_depth)
        arrays += child_arrays
    return leaves, max_depth, arrays


def _selection_score(row: dict[str, Any]) -> tuple[int, int, int, int]:
    schema = json.loads(row["schema"])
    ground_truth = json.loads(row["ground_truth"])
    leaves, depth, arrays = _schema_complexity(schema)
    return leaves + depth * 3 + arrays * 5, leaves, depth, len(json.dumps(ground_truth))


def _rows_page(offset: int, timeout_s: float) -> dict[str, Any]:
    query = urlencode(
        {
            "dataset": _DATASET,
            "config": "default",
            "split": "benchmark",
            "offset": offset,
            "length": _PAGE_SIZE,
        }
    )
    return _json_url(f"{_ROWS_URL}?{query}", timeout_s)


def _select_rows(
    *,
    splits: set[str],
    per_split: int,
    timeout_s: float,
) -> list[dict[str, Any]]:
    candidates: dict[str, list[dict[str, Any]]] = {split: [] for split in splits}
    offset = 0
    while True:
        page = _rows_page(offset, timeout_s)
        for item in page["rows"]:
            row = item["row"]
            if row["split"] not in splits:
                continue
            candidates[row["split"]].append(
                {
                    "row_idx": item["row_idx"],
                    "row": row,
                    "score": _selection_score(row),
                }
            )
        offset += len(page["rows"])
        if offset >= page["num_rows_total"] or not page["rows"]:
            break

    selected: list[dict[str, Any]] = []
    for split in sorted(splits):
        ranked = sorted(
            candidates[split],
            key=lambda item: (item["score"], item["row"]["doc_id"]),
            reverse=True,
        )
        selected.extend(ranked[:per_split])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--per-split", type=int, default=3)
    parser.add_argument("--splits", nargs="+", default=["Nested", "Table"])
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    if args.per_split < 1:
        raise ValueError("per_split must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = _select_rows(
        splits=set(args.splits),
        per_split=args.per_split,
        timeout_s=args.timeout,
    )
    manifest_pages: list[dict[str, Any]] = []
    for index, item in enumerate(selected, start=1):
        row = item["row"]
        doc_id = row["doc_id"]
        stem = f"{index:02d}_{row['split'].lower()}_{_SAFE_FILENAME.sub('_', doc_id)}"
        image_path = args.output_dir / f"{stem}.jpg"
        pdf_path = args.output_dir / f"{stem}.pdf"
        if not image_path.exists():
            _download(row["image"]["src"], image_path, args.timeout)
        if not pdf_path.exists():
            _download(_PDF_BASE + quote(f"{doc_id}.pdf"), pdf_path, args.timeout)

        image_bytes = image_path.read_bytes()
        pdf_bytes = pdf_path.read_bytes()
        manifest_pages.append(
            {
                "row_idx": item["row_idx"],
                "doc_id": doc_id,
                "split": row["split"],
                "selection_score": list(item["score"]),
                "image_file": image_path.name,
                "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                "pdf_file": pdf_path.name,
                "pdf_sha256": hashlib.sha256(pdf_bytes).hexdigest(),
                "schema": json.loads(row["schema"]),
                "ground_truth": json.loads(row["ground_truth"]),
            }
        )
        print(f"{index}/{len(selected)} {doc_id}: {row['split']} score={item['score']}")

    manifest = {
        "source": _DATASET,
        "source_license": "cdla-permissive-2.0",
        "selection_policy": "highest schema complexity per requested structural split",
        "pages": manifest_pages,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
