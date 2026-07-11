"""A/B schema-guided sidecar-моделей на сложном открытом поднаборе VAREX."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

_PROTOCOL_VERSION = 3
_NUEXTRACT3_REVISION = "2e9fca82ee641e6bb6e1f5d905241e994be27a07"


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


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
    parsed = json.loads(stripped, parse_constant=_reject_constant)
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
            raise ValueError(f"resume metadata mismatch for {key}: {existing.get(key)!r} != {value!r}")
    if not isinstance(existing.get("results"), dict):
        raise ValueError("resume summary must contain a results object")
    return existing


def _prediction_complete(path: Path, result: Any) -> bool:
    if (
        not isinstance(result, dict)
        or result.get("status") != "ok"
        or result.get("finish_reason") != "stop"
        or path.is_symlink()
        or not path.is_file()
    ):
        return False
    try:
        prediction = _parse_json_output(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return "_error" not in prediction


def _nuextract_contract(
    schema: dict[str, Any],
    *,
    converter: Callable[..., tuple[Any, list[Any]]] | None = None,
    description_getter: Callable[[dict[str, Any]], str] | None = None,
) -> dict[str, str]:
    if converter is None:
        from numind.nuextract_utils import (  # type: ignore[import-not-found]
            convert_json_schema_to_nuextract_template,
        )

        converter = convert_json_schema_to_nuextract_template
    if description_getter is None:
        from numind.nuextract_utils.json_schema import (  # type: ignore[import-not-found]
            get_description_json_schema_nodes,
        )

        description_getter = get_description_json_schema_nodes
    assert converter is not None
    assert description_getter is not None
    template, dropped = converter(schema, omit_unsupported_branches=False)
    if dropped:
        raise ValueError(f"NuExtract schema conversion dropped branches: {dropped!r}")
    descriptions = description_getter(schema)
    if not isinstance(descriptions, str) or not descriptions.strip():
        raise ValueError("NuExtract schema descriptions must be non-empty")
    instructions = (
        "Return null for fields you cannot find.\n"
        "Return ONLY valid JSON.\n"
        "Return an instance of the JSON with extracted values, not the schema itself.\n"
        + descriptions.strip()
    )
    return {
        "template": json.dumps(template, ensure_ascii=False, separators=(",", ":")),
        "instructions": instructions,
    }


def _request_payload(
    *,
    model: str,
    image_b64: str,
    schema: dict[str, Any],
    max_tokens: int,
    profile: str,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}", "detail": "high"},
        }
    ]
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "seed": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    if profile == "generic":
        content.append({"type": "text", "text": _extraction_prompt(schema)})
    elif profile == "nuextract3":
        contract = _nuextract_contract(schema)
        payload["chat_template_kwargs"] = {
            **contract,
            "enable_thinking": False,
        }
    else:
        raise ValueError(f"unknown request profile: {profile}")
    return payload


def _request(
    client: httpx.Client,
    *,
    endpoint: str,
    model: str,
    image_path: Path,
    schema: dict[str, Any],
    max_tokens: int,
    profile: str,
) -> tuple[str, str | None]:
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    response = client.post(
        endpoint.rstrip("/") + "/chat/completions",
        json=_request_payload(
            model=model,
            image_b64=image_b64,
            schema=schema,
            max_tokens=max_tokens,
            profile=profile,
        ),
    )
    response.raise_for_status()
    payload = response.json()
    choice = payload["choices"][0]
    content = choice["message"]["content"]
    if not isinstance(content, str):
        raise ValueError("model response content must be a string")
    finish_reason = choice.get("finish_reason")
    return content, finish_reason if isinstance(finish_reason, str) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--profile", choices=("generic", "nuextract3"), default="generic")
    parser.add_argument("--model-revision", default="")
    parser.add_argument("--runtime", default="")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_tokens == 0:
        args.max_tokens = 8192 if args.profile == "nuextract3" else 4096
    if args.max_tokens < 1 or args.max_tokens > 16_384:
        parser.error("--max-tokens must be in 1..16384")
    if args.profile == "nuextract3":
        if not args.model_revision:
            args.model_revision = _NUEXTRACT3_REVISION
        if args.model_revision != _NUEXTRACT3_REVISION:
            parser.error("NuExtract3 benchmark requires the pinned revision")
        if not args.runtime.strip():
            parser.error("NuExtract3 benchmark requires --runtime with exact package versions")

    manifest_path = args.corpus_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prediction_dir = args.output_dir / args.name / "image"
    data_dir = args.output_dir / "_data"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    expected_summary: dict[str, Any] = {
        "protocol_version": _PROTOCOL_VERSION,
        "source": manifest["source"],
        "model": args.model,
        "model_revision": args.model_revision or None,
        "runtime": args.runtime or None,
        "name": args.name,
        "profile": args.profile,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "seed": 0,
        "prompt_role": "user_multimodal",
        "corpus_manifest_sha256": _sha256_file(manifest_path),
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
            if (
                image_path.is_symlink()
                or not image_path.is_file()
                or _sha256_file(image_path) != page.get("image_sha256")
            ):
                raise ValueError(f"VAREX image missing or SHA256 mismatch: {image_path.name}")
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
            finish_reason = None
            try:
                raw_output, finish_reason = _request(
                    client,
                    endpoint=args.endpoint,
                    model=args.model,
                    image_path=image_path,
                    schema=page["schema"],
                    max_tokens=args.max_tokens,
                    profile=args.profile,
                )
                if finish_reason != "stop":
                    raise ValueError(f"model finish_reason must be stop, got {finish_reason!r}")
                prediction = _parse_json_output(raw_output)
                status = "ok"
                error = None
            except Exception as exc:
                prediction = {"_error": str(exc)}
                status = "error"
                error = str(exc)
            latency_s = round(time.monotonic() - started, 3)
            _atomic_write_json(prediction_path, prediction)
            (prediction_dir / f"{doc_id}.raw.txt").write_text(raw_output, encoding="utf-8")
            summary["results"][doc_id] = {
                "split": page["split"],
                "status": status,
                "latency_s": latency_s,
                "error": error,
                "finish_reason": finish_reason,
            }
            _atomic_write_json(summary_path, summary)
            print(json.dumps(summary["results"][doc_id], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
