from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _module():
    scripts = Path(__file__).parents[1] / "scripts"
    _load_module("generate_private_rag_eval", scripts / "generate_private_rag_eval.py")
    return _load_module(
        "generate_private_rag_supplement",
        scripts / "generate_private_rag_supplement.py",
    )


def _private_file(path: Path, content: bytes) -> Path:
    path.write_bytes(content)
    path.chmod(0o600)
    return path


def _identity(module):
    return module.RunIdentity(
        seed=7,
        corpus_sha256="1" * 64,
        snapshots_sha256="2" * 64,
        plan_sha256="3" * 64,
        model="local-model",
        model_revision="revision-1",
        generator_contract_version="supplement-v1",
        per_stratum=8,
        min_chars=200,
        trial=True,
    )


def test_challenge_targets_are_balanced_and_deterministic() -> None:
    module = _module()

    first = module.challenge_targets(
        2026071314,
        standards_count=8,
        prompt_injection_count=10,
        leakage_count=10,
    )
    second = module.challenge_targets(
        2026071314,
        standards_count=8,
        prompt_injection_count=10,
        leakage_count=10,
    )
    all_targets = (*first["single_hop"], *first["no_answer"])

    assert first == second
    assert len(first["single_hop"]) == 18
    assert len(first["no_answer"]) == 10
    assert {item.challenge_tag for item in all_targets} == {
        "standards",
        "prompt_injection",
        "leakage",
    }
    for challenge in ("standards", "prompt_injection", "leakage"):
        counts = {
            language: sum(
                item.language == language and item.challenge_tag == challenge
                for item in all_targets
            )
            for language in ("ru", "en", "zh")
        }
        assert max(counts.values()) - min(counts.values()) <= 1
    with pytest.raises(ValueError, match=r"\[0, 50\]"):
        module.challenge_targets(
            1,
            standards_count=0,
            prompt_injection_count=0,
            leakage_count=0,
        )


def test_base_manifest_is_hash_bound_to_private_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    gold = _private_file(tmp_path / "gold.jsonl", b"gold\n")
    sidecar = _private_file(tmp_path / "sidecar.jsonl", b"sidecar\n")
    manifest_payload = {
        "schema_version": 1,
        "purpose": "private_rag_candidate_evaluation",
        "gold_artifact_sha256": hashlib.sha256(gold.read_bytes()).hexdigest(),
        "generator_artifact_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
    }
    manifest = _private_file(
        tmp_path / "manifest.json",
        json.dumps(manifest_payload).encode(),
    )
    monkeypatch.setattr(module, "load_gold_set", lambda *args, **kwargs: ([object()], object()))
    monkeypatch.setattr(module, "load_private_sidecar", lambda *args, **kwargs: [object()])
    monkeypatch.setattr(module, "bind_gold_sidecar", lambda *args, **kwargs: {})

    records, sidecars, loaded, hashes = module.load_bound_base(gold, sidecar, manifest)

    assert len(records) == len(sidecars) == 1
    assert loaded == manifest_payload
    assert hashes["manifest"] == hashlib.sha256(manifest.read_bytes()).hexdigest()

    manifest_payload["gold_artifact_sha256"] = "0" * 64
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    manifest.chmod(0o600)
    with pytest.raises(ValueError, match="artifact hashes"):
        module.load_bound_base(gold, sidecar, manifest)


def test_base_manifest_requires_exact_private_permissions(
    tmp_path: Path,
) -> None:
    module = _module()
    paths = [
        _private_file(tmp_path / "gold.jsonl", b"gold\n"),
        _private_file(tmp_path / "sidecar.jsonl", b"sidecar\n"),
        _private_file(tmp_path / "manifest.json", b"{}"),
    ]
    paths[2].chmod(0o640)

    with pytest.raises(ValueError, match="0600"):
        module.load_bound_base(*paths)


def test_private_group_is_idempotent_and_preserves_conflicts(tmp_path: Path) -> None:
    module = _module()
    first = _private_file(tmp_path / "first.json", b"first")
    second = tmp_path / "second.json"
    payloads = ((first, b"first"), (second, b"second"))

    module.publish_private_group(payloads)
    module.publish_private_group(payloads)

    assert first.read_bytes() == b"first"
    assert second.read_bytes() == b"second"
    assert first.stat().st_mode & 0o777 == second.stat().st_mode & 0o777 == 0o600

    second.write_bytes(b"pre-existing")
    second.chmod(0o600)
    with pytest.raises(FileExistsError, match="differs"):
        module.publish_private_group(payloads)
    assert first.read_bytes() == b"first"
    assert second.read_bytes() == b"pre-existing"


def test_private_group_rejects_symlink_without_touching_target(tmp_path: Path) -> None:
    module = _module()
    target = _private_file(tmp_path / "target.json", b"sentinel")
    link = tmp_path / "link.json"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        module.publish_private_group(((link, b"replacement"),))
    assert target.read_bytes() == b"sentinel"


def test_private_group_rolls_back_new_files_on_publish_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    real_link = module.os.link
    calls = 0

    def fail_second(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic link failure")
        real_link(source, destination)

    monkeypatch.setattr(module.os, "link", fail_second)
    with pytest.raises(OSError, match="synthetic link failure"):
        module.publish_private_group(((first, b"first"), (second, b"second")))
    assert not first.exists()
    assert not second.exists()


def test_checkpoint_base_link_fails_closed_on_base_change(tmp_path: Path) -> None:
    module = _module()
    identity = _identity(module)
    base_hashes = {"gold": "4" * 64, "sidecar": "5" * 64, "manifest": "6" * 64}
    expected = module.checkpoint_base_link(base_hashes, identity)
    path = _private_file(
        tmp_path / "base-link.json",
        module._canonical_json(expected) + b"\n",
    )

    module.require_checkpoint_base_link(path, expected)
    changed = module.checkpoint_base_link({**base_hashes, "gold": "7" * 64}, identity)
    with pytest.raises(ValueError, match="does not match"):
        module.require_checkpoint_base_link(path, changed)
