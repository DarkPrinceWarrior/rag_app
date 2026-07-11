"""A/B schema-guided sidecar-моделей на сложном открытом поднаборе VAREX."""

from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path
from typing import Any

import httpx

_PROTOCOL_VERSION = 2


def _extraction_prompt(schema: dict[str, Any]) -> str:
    return (
        "Extract structured data from this document.\n"
        "Return a JSON object matching this schema:\n\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        "Return null for fields you cannot find.\n"
        "Return ONLY valid JSON.\n"
        "Return an instance of the JSON with extracted values, not the schema itself."
    )


def _parse_json_output(value: str) -> dict[str, Any]:
    stripped = value.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("model output must be a JSON object")
    return parsed


def _load_summary(
    path: Path,
    *,
    expected: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    if not resume or not path.exists():
        return {**expected, "results": {}}
    existing = json.loads(path.read_text(encoding="utf-8"))
    for key, value in expected.items():
        if existing.get(key) != value:
            raise ValueError(
                f"resume metadata mismatch for {key}: {existing.get(key)!r} != {value!r}"
            )
    if not isinstance(existing.get("results"), dict):
        raise ValueError("resume summary must contain a results object")
    return existing


def _prediction_complete(path: Path, result: Any) -> bool:
    if not isinstance(result, dict) or result.get("status") != "ok" or not path.exists():
        return False
    try:
        prediction = _parse_json_output(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return "_error" not in prediction


def _request(
    client: httpx.Client,
    *,
    endpoint: str,
    model: str,
    image_path: Path,
    schema: dict[str, Any],
    max_tokens: int,
) -> tuple[dict[str, Any], str]:
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    response = client.post(
        endpoint.rstrip("/") + "/chat/completions",
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": _extraction_prompt(schema)},
                    ],
                },
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        },
    )
    response.raise_for_status()
    payload = response.json()
    content = payload["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise ValueError("model response content must be a string")
    return _parse_json_output(content), content


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    manifest = json.loads((args.corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    prediction_dir = args.output_dir / args.name / "image"
    data_dir = args.output_dir / "_data"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    expected_summary: dict[str, Any] = {
        "protocol_version": _PROTOCOL_VERSION,
        "source": manifest["source"],
        "model": args.model,
        "name": args.name,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "prompt_role": "user_multimodal",
    }
    summary_path = args.output_dir / f"{args.name}.summary.json"
    summary = _load_summary(
        summary_path,
        expected=expected_summary,
        resume=args.resume,
    )

    with httpx.Client(timeout=args.timeout) as client:
        for page in manifest["pages"]:
            doc_id = page["doc_id"]
            image_path = args.corpus_dir / page["image_file"]
            gt_dir = data_dir / doc_id
            gt_dir.mkdir(exist_ok=True)
            (gt_dir / "ground_truth.json").write_text(
                json.dumps(page["ground_truth"], ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            prediction_path = prediction_dir / f"{doc_id}.pred.json"
            if args.resume and _prediction_complete(
                prediction_path,
                summary["results"].get(doc_id),
            ):
                print(f"{doc_id}: {args.name} resume=skip", flush=True)
                continue

            print(f"{doc_id}: {args.name}", flush=True)
            started = time.monotonic()
            raw_output = ""
            try:
                prediction, raw_output = _request(
                    client,
                    endpoint=args.endpoint,
                    model=args.model,
                    image_path=image_path,
                    schema=page["schema"],
                    max_tokens=args.max_tokens,
                )
                status = "ok"
                error = None
            except Exception as exc:
                prediction = {"_error": str(exc)}
                status = "error"
                error = str(exc)
            latency_s = round(time.monotonic() - started, 3)
            prediction_path.write_text(
                json.dumps(prediction, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            (prediction_dir / f"{doc_id}.raw.txt").write_text(raw_output, encoding="utf-8")
            summary["results"][doc_id] = {
                "split": page["split"],
                "status": status,
                "latency_s": latency_s,
                "error": error,
            }
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(summary["results"][doc_id], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
