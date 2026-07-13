"""Append checkpointed coverage cases to an existing private RAG candidate set."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import stat
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from generate_private_rag_eval import (
    CaseTarget,
    GeneratedCase,
    PrivateRagGenerator,
    Stratum,
    UniqueCaseRegistry,
    build_checkpoint_identity,
    build_checkpoint_targets,
    build_document_snapshots,
    corpus_fingerprint,
    create_readonly_sessionmaker,
    ensure_private_directory,
    generate_stratum,
    language_schedule,
    load_corpus,
    plan_source_sets,
    preflight_generated_cases,
)

from rag_app.eval.automated_review import require_private_input_0600
from rag_app.eval.gold_set import GoldRecord, ensure_private_gold_path, load_gold_set
from rag_app.eval.private_checkpoint import (
    PrivateCheckpointStore,
    RunIdentity,
    canonical_sha256,
    checkpoint_lineage_entry,
)
from rag_app.eval.private_sidecar import (
    PrivateSidecarRecord,
    bind_gold_sidecar,
    load_private_sidecar,
)
from rag_app.storage.s3 import Storage

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_CONTRACT_VERSION = "rag-gold-coverage-supplement-v1"
_STRATA: tuple[Stratum, ...] = (
    "single_hop",
    "multi_hop",
    "cross_document",
    "no_answer",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_private_group(payloads: tuple[tuple[Path, bytes], ...]) -> None:
    """Idempotently publish a fresh 0600 group while preserving existing files."""

    raw_paths = [path for path, _ in payloads]
    if any(path.is_symlink() for path in raw_paths):
        raise ValueError("private output must not be a symlink")
    paths = [path.resolve(strict=False) for path in raw_paths]
    if len(paths) != len(set(paths)):
        raise ValueError("private output paths must be distinct")
    missing: list[tuple[Path, bytes]] = []
    for path, content in zip(paths, (content for _, content in payloads), strict=True):
        if not path.exists():
            missing.append((path, content))
            continue
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
            raise ValueError("existing private output is not a regular 0600 file")
        if path.read_bytes() != content:
            raise FileExistsError("existing private output differs from expected content")

    staged: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for path, content in missing:
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            path.parent.chmod(0o700)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.",
                dir=path.parent,
            )
            temporary = Path(temporary_name)
            os.fchmod(descriptor, 0o600)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
            staged.append((path, temporary))
        for path, temporary in staged:
            os.link(temporary, path)
            published.append(path)
            _fsync_parent(path)
    except BaseException:
        for path in reversed(published):
            path.unlink(missing_ok=True)
            _fsync_parent(path)
        raise
    finally:
        for _, temporary in staged:
            temporary.unlink(missing_ok=True)


def checkpoint_base_link(
    base_hashes: dict[str, str],
    identity: RunIdentity,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "checkpoint_identity_sha256": canonical_sha256(identity),
        "base_artifacts": dict(sorted(base_hashes.items())),
        "base_artifacts_sha256": canonical_sha256(base_hashes),
    }


def require_checkpoint_base_link(
    path: Path,
    expected: dict[str, Any],
) -> None:
    require_private_input_0600(path, name="checkpoint base link")
    try:
        actual = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("checkpoint base link is not valid UTF-8 JSON") from None
    if actual != expected:
        raise ValueError("checkpoint base link does not match base artifacts")


def supplement_artifact_payloads(
    output_dir: Path,
    *,
    seed: int,
    records: list[dict[str, Any]],
    sidecars: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> tuple[tuple[Path, bytes], ...]:
    stem = f"private_rag_eval_seed_{seed}"
    gold_path = output_dir / f"{stem}.jsonl"
    sidecar_path = output_dir / f"{stem}.generator.jsonl"
    manifest_path = output_dir / f"{stem}.manifest.json"
    gold_bytes = b"".join(_canonical_json(item) + b"\n" for item in records)
    sidecar_bytes = b"".join(_canonical_json(item) + b"\n" for item in sidecars)
    final_manifest = {
        **manifest,
        "gold_artifact_sha256": hashlib.sha256(gold_bytes).hexdigest(),
        "generator_artifact_sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
    }
    manifest_bytes = (
        json.dumps(final_manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    return (
        (sidecar_path, sidecar_bytes),
        (gold_path, gold_bytes),
        (manifest_path, manifest_bytes),
    )


def load_bound_base(
    gold_path: Path,
    sidecar_path: Path,
    manifest_path: Path,
) -> tuple[list[GoldRecord], list[PrivateSidecarRecord], dict[str, Any], dict[str, str]]:
    paths = {
        "gold": require_private_input_0600(gold_path, name="base gold set"),
        "sidecar": require_private_input_0600(sidecar_path, name="base private sidecar"),
        "manifest": require_private_input_0600(manifest_path, name="base manifest"),
    }
    records, _ = load_gold_set(paths["gold"], mode="candidate", repository_root=REPOSITORY_ROOT)
    sidecars = load_private_sidecar(paths["sidecar"], repository_root=REPOSITORY_ROOT)
    bind_gold_sidecar(records, sidecars)
    try:
        manifest = json.loads(paths["manifest"].read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("base manifest is not valid UTF-8 JSON") from None
    if not isinstance(manifest, dict):
        raise ValueError("base manifest must be a JSON object")
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    if manifest.get("schema_version") != 1 or manifest.get("purpose") != "private_rag_candidate_evaluation":
        raise ValueError("base manifest has an unsupported identity")
    if (
        manifest.get("gold_artifact_sha256") != hashes["gold"]
        or manifest.get("generator_artifact_sha256") != hashes["sidecar"]
    ):
        raise ValueError("base manifest artifact hashes do not match")
    return records, sidecars, manifest, hashes


def challenge_targets(
    seed: int,
    *,
    standards_count: int,
    prompt_injection_count: int,
    leakage_count: int,
) -> dict[Stratum, tuple[CaseTarget, ...]]:
    counts = (standards_count, prompt_injection_count, leakage_count)
    if any(not 0 <= count <= 50 for count in counts) or not any(counts):
        raise ValueError("challenge counts must be in [0, 50] with at least one positive")
    standards = tuple(
        CaseTarget(language=language, challenge_tag="standards")
        for language in language_schedule(seed, standards_count)
    )
    prompt_injection = tuple(
        CaseTarget(language=language, challenge_tag="prompt_injection")
        for language in language_schedule(seed + 1, prompt_injection_count)
    )
    leakage = tuple(
        CaseTarget(language=language, challenge_tag="leakage")
        for language in language_schedule(seed + 2, leakage_count)
    )
    return {
        "single_hop": (*standards, *prompt_injection),
        "multi_hop": (),
        "cross_document": (),
        "no_answer": leakage,
    }


def seed_registry(
    records: list[GoldRecord],
    checkpoint: PrivateCheckpointStore,
) -> UniqueCaseRegistry:
    registry = UniqueCaseRegistry(checkpoint.iter_slots())
    for record in records:
        duplicate = registry.claim(record)
        if duplicate is not None:
            raise ValueError("base and supplement checkpoint contain a duplicate case")
    return registry


def merge_and_validate(
    base_records: list[GoldRecord],
    base_sidecars: list[PrivateSidecarRecord],
    supplement: list[GeneratedCase],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, int]]]:
    supplement_records = [
        GoldRecord.model_validate_json(
            json.dumps(item.record, ensure_ascii=False), strict=True
        )
        for item in supplement
    ]
    supplement_sidecars = [
        PrivateSidecarRecord.model_validate_json(
            json.dumps(item.metadata, ensure_ascii=False), strict=True
        )
        for item in supplement
    ]
    records = [*base_records, *supplement_records]
    sidecars = [*base_sidecars, *supplement_sidecars]
    bind_gold_sidecar(records, sidecars)
    generated = [
        GeneratedCase(
            record=record.model_dump(mode="json"),
            metadata=sidecar.model_dump(mode="json"),
        )
        for record, sidecar in zip(records, sidecars, strict=True)
    ]
    preflight = preflight_generated_cases(generated, trial=False)
    return (
        [item.record for item in generated],
        [item.metadata for item in generated],
        preflight,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-gold", type=Path, required=True)
    parser.add_argument("--base-sidecar", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026071314)
    parser.add_argument("--standards-count", type=int, default=8)
    parser.add_argument("--prompt-injection-count", type=int, default=10)
    parser.add_argument("--leakage-count", type=int, default=10)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=2, choices=(1, 2))
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--llm-base-url", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser


async def async_main(args: argparse.Namespace) -> int:
    if args.min_chars <= 0 or args.max_attempts <= 0:
        raise ValueError("min-chars and max-attempts must be positive")
    case_targets = challenge_targets(
        args.seed,
        standards_count=args.standards_count,
        prompt_injection_count=args.prompt_injection_count,
        leakage_count=args.leakage_count,
    )
    total_targets = sum(len(targets) for targets in case_targets.values())
    base_gold = ensure_private_gold_path(args.base_gold, REPOSITORY_ROOT)
    base_sidecar = ensure_private_gold_path(args.base_sidecar, REPOSITORY_ROOT)
    base_manifest = ensure_private_gold_path(args.base_manifest, REPOSITORY_ROOT)
    base_records, base_sidecars, base_metadata, base_hashes = load_bound_base(
        base_gold,
        base_sidecar,
        base_manifest,
    )
    if len(base_records) + total_targets > 500:
        raise ValueError("combined candidate set would exceed 500 records")

    engine, sessionmaker = create_readonly_sessionmaker()
    generator: PrivateRagGenerator | None = None
    try:
        corpus = await load_corpus(sessionmaker)
        current_fingerprint = corpus_fingerprint(corpus)
        base_corpus = base_metadata.get("corpus")
        if (
            not isinstance(base_corpus, dict)
            or base_corpus.get("fingerprint_sha256") != current_fingerprint
        ):
            raise RuntimeError("base candidate corpus differs from current production corpus")
        eligible = [
            chunk
            for chunk in corpus
            if (
                len(chunk.text) >= args.min_chars
                or (chunk.kind in {"table", "image"} and len(chunk.text) >= 12)
            )
        ]
        if not eligible:
            raise RuntimeError("the corpus has no eligible chunks")
        plans = plan_source_sets(
            eligible,
            seed=args.seed,
            pool_per_stratum=max(total_targets * 8, len(eligible) ** 2),
        )
        checkpoint_targets = build_checkpoint_targets(_STRATA, plans, case_targets)
        snapshots = await build_document_snapshots(corpus, Storage())
        identity = build_checkpoint_identity(
            seed=args.seed,
            corpus=corpus,
            snapshots=snapshots,
            checkpoint_targets=checkpoint_targets,
            model=args.model,
            model_revision=args.model_revision,
            per_stratum=total_targets,
            min_chars=args.min_chars,
            trial=True,
            generator_contract_version=_CONTRACT_VERSION,
        )
        output_dir = ensure_private_directory(args.output_dir)
        checkpoint_root = output_dir / f".private_rag_supplement_seed_{args.seed}.checkpoint"
        base_link_path = checkpoint_root.with_name(f"{checkpoint_root.name}.base-link.json")
        expected_base_link = checkpoint_base_link(base_hashes, identity)
        if args.resume:
            require_checkpoint_base_link(base_link_path, expected_base_link)
            checkpoint = PrivateCheckpointStore.resume(
                checkpoint_root,
                identity,
                max_attempts=args.max_attempts,
            )
        else:
            if base_link_path.exists() or base_link_path.is_symlink():
                raise FileExistsError("checkpoint base link already exists")
            publish_private_group(
                ((base_link_path, _canonical_json(expected_base_link) + b"\n"),)
            )
            try:
                checkpoint = PrivateCheckpointStore.create(
                    checkpoint_root,
                    identity,
                    max_attempts=args.max_attempts,
                )
            except BaseException:
                base_link_path.unlink(missing_ok=True)
                _fsync_parent(base_link_path)
                raise
        registry = seed_registry(base_records, checkpoint)
        generator = PrivateRagGenerator(
            sessionmaker,
            model=args.model,
            base_url=args.llm_base_url,
            seed=args.seed,
            concurrency=args.concurrency,
            corpus=corpus,
            snapshots=snapshots,
        )
        generated_results = await asyncio.gather(
            *(
                generate_stratum(
                    generator,
                    plans[stratum],
                    target=len(case_targets[stratum]),
                    max_attempts_per_source=args.max_attempts,
                    case_targets=case_targets[stratum],
                    checkpoint_targets=checkpoint_targets[stratum],
                    checkpoint=checkpoint,
                    unique_cases=registry,
                    breadth_first=True,
                )
                for stratum in ("single_hop", "no_answer")
                if case_targets[stratum]
            )
        )
        supplement = [item for accepted, _ in generated_results for item in accepted]
        rejected = sum(len(cursor.rejects) for cursor in checkpoint.iter_cursors())
        if len(supplement) != total_targets:
            raise RuntimeError("coverage supplement generation is incomplete")
        records, sidecars, preflight = merge_and_validate(
            base_records,
            base_sidecars,
            supplement,
        )
        post_corpus = await load_corpus(sessionmaker)
        if corpus_fingerprint(post_corpus) != current_fingerprint:
            raise RuntimeError("production corpus changed during supplement generation")
        if {name: file_sha256(path) for name, path in {
            "gold": base_gold,
            "sidecar": base_sidecar,
            "manifest": base_manifest,
        }.items()} != base_hashes:
            raise RuntimeError("base artifacts changed during supplement generation")

        manifest: dict[str, Any] = {
            "schema_version": 1,
            "purpose": "private_rag_candidate_evaluation_supplement",
            "privacy": {
                "database_mode": "default_transaction_read_only",
                "model_network_scope": "loopback_only",
                "artifact_mode": "0600",
            },
            "base": {
                "gold_artifact_sha256": base_hashes["gold"],
                "generator_artifact_sha256": base_hashes["sidecar"],
                "manifest_sha256": base_hashes["manifest"],
                "record_count": len(base_records),
                "corpus_fingerprint_sha256": base_metadata["corpus"]["fingerprint_sha256"],
            },
            "corpus": {
                "documents": len({chunk.document_id for chunk in corpus}),
                "chunks_total": len(corpus),
                "chunks_eligible": len(eligible),
                "fingerprint_sha256": corpus_fingerprint(corpus),
                "owner_scopes": len({chunk.scope_id for chunk in corpus}),
            },
            "generation": {
                "contract_version": _CONTRACT_VERSION,
                "model": args.model,
                "model_revision": args.model_revision,
                "seed": args.seed,
                "temperature": 0.0,
                "concurrency": args.concurrency,
                "challenge_counts": {
                    "standards": args.standards_count,
                    "prompt_injection": args.prompt_injection_count,
                    "leakage": args.leakage_count,
                },
                "requested": total_targets,
                "accepted": len(supplement),
                "rejected_attempts": rejected,
                "checkpoint": checkpoint_lineage_entry(checkpoint).model_dump(mode="json"),
                "checkpoint_base_link_sha256": file_sha256(base_link_path),
            },
            "combined": {
                "record_count": len(records),
                "language_counts": dict(
                    sorted(Counter(record["language"] for record in records).items())
                ),
                "preflight": preflight,
            },
        }
        payloads = supplement_artifact_payloads(
            output_dir,
            seed=args.seed,
            records=records,
            sidecars=sidecars,
            manifest=manifest,
        )
        publish_private_group(payloads)
        checkpoint.cleanup_after_success(final_artifacts_written=True)
        base_link_path.unlink()
        _fsync_parent(base_link_path)
        paths = tuple(path for path, _ in payloads)
        print(
            "private RAG coverage supplement generated: "
            f"base={len(base_records)} supplement={len(supplement)} total={len(records)} "
            f"rejected={rejected}"
        )
        print(f"artifact={paths[1]}")
        print(f"generator_metadata={paths[0]}")
        print(f"manifest={paths[2]}")
        return 0
    finally:
        if generator is not None:
            await generator.close()
        await engine.dispose()


def main() -> int:
    return asyncio.run(async_main(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
