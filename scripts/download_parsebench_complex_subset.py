"""Скачать небольшой воспроизводимый набор сложных страниц ParseBench."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

_BASE_URL = "https://huggingface.co/datasets/llamaindex/ParseBench/resolve/main/"
_SOURCE_REVISION = "main"
_PAGES: tuple[dict[str, Any], ...] = (
    {
        "category": "table",
        "path": "docs/table/637951191e7b35ae07dbd4c76a2e6690c370_pg2_pg1_page1.pdf",
        "selection": {"tags": ["hard"], "rows": 67, "cells": 270, "spans": 80},
    },
    {
        "category": "table",
        "path": "docs/table/SERFF_TX_random_pages 1_page1056.pdf",
        "selection": {"tags": ["hard"], "rows": 111, "cells": 1821, "spans": 46},
    },
    {
        "category": "table",
        "path": "docs/table/BRWS-134565917_page1123.pdf",
        "selection": {"tags": ["hard"], "rows": 615, "cells": 1845, "spans": 0},
    },
    {
        "category": "chart",
        "path": "docs/chart/Renewables2025_1_p36.pdf",
        "selection": {"tags": ["3d_chart", "need_estimate"], "rules": 11},
    },
    {
        "category": "chart",
        "path": "docs/chart/2024_healthatglance_rep_en_p119.pdf",
        "selection": {"tags": [], "rules": 11},
    },
    {
        "category": "layout",
        "path": "docs/layout/076523s007lbl_p2.pdf",
        "selection": {"tags": ["hard"], "rules": 198},
    },
    {
        "category": "layout",
        "path": "docs/layout/2024-Ford-Integrated-Sustainability-and-Financial-Report_Final_p21.pdf",
        "selection": {"tags": ["hard"], "rules": 116},
    },
    {
        "category": "text_formatting",
        "path": "docs/text/text_ocr__paper8col.pdf",
        "selection": {"tags": ["hard", "ocr"], "rules": 109},
    },
    {
        "category": "text_formatting",
        "path": "docs/text/text_dense__legalnotices.pdf",
        "selection": {"tags": ["dense", "hard"], "rules": 98},
    },
)


def _download(url: str, destination: Path, timeout_s: float) -> None:
    request = Request(url, headers={"User-Agent": "DocRAGenslate-eval/1.0"})
    part = destination.with_suffix(destination.suffix + ".part")
    with urlopen(request, timeout=timeout_s) as response, part.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    if not part.read_bytes().startswith(b"%PDF"):
        part.unlink(missing_ok=True)
        raise RuntimeError(f"источник не вернул PDF: {url}")
    part.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    for index, page in enumerate(_PAGES, start=1):
        source_path = str(page["path"])
        filename = f"{index:02d}_{page['category']}_{Path(source_path).name}"
        destination = args.output_dir / filename
        url = _BASE_URL + quote(source_path, safe="/")
        if not destination.is_file():
            _download(url, destination, args.timeout)
        payload = destination.read_bytes()
        manifest.append(
            {
                **page,
                "file": filename,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "url": url,
            }
        )
        print(f"{index}/{len(_PAGES)} {filename}: {len(payload)} bytes")

    output = {
        "source": "llamaindex/ParseBench",
        "source_revision": _SOURCE_REVISION,
        "source_license": "Apache-2.0",
        "selection_policy": "top hard pages by rules/table structure; not a random smoke set",
        "pages": manifest,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
