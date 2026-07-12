#!/usr/bin/env python3
"""Validate pinned self-hosted CI and package-manager security policy."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PNPM = "pnpm@11.12.0"
MINIMUM_RELEASE_AGE_MINUTES = 1440


def _fail(message: str) -> None:
    raise SystemExit(f"CI policy error: {message}")


def _package_manager(path: Path) -> str:
    value = json.loads(path.read_text(encoding="utf-8"))
    manager = value.get("packageManager")
    if not isinstance(manager, str):
        _fail(f"{path.relative_to(ROOT)} lacks packageManager")
    return manager


def _yaml_scalar(text: str, key: str, *, path: Path) -> str:
    match = re.search(rf"(?m)^{re.escape(key)}:\s*([^#\s]+)\s*$", text)
    if match is None:
        _fail(f"{path.relative_to(ROOT)} lacks {key}")
    return match.group(1)


def _allow_builds(text: str, *, path: Path) -> dict[str, bool]:
    lines = text.splitlines()
    try:
        start = lines.index("allowBuilds:") + 1
    except ValueError:
        _fail(f"{path.relative_to(ROOT)} lacks allowBuilds")
    result: dict[str, bool] = {}
    for line in lines[start:]:
        if not line.startswith("  "):
            break
        match = re.fullmatch(r"  ([A-Za-z0-9@/_.-]+): (true|false)", line)
        if match is None:
            _fail(f"invalid allowBuilds entry in {path.relative_to(ROOT)}: {line!r}")
        result[match.group(1)] = match.group(2) == "true"
    return result


def _check_workspace(name: str, *, expected_builds: dict[str, bool]) -> None:
    directory = ROOT / name
    package_path = directory / "package.json"
    workspace_path = directory / "pnpm-workspace.yaml"
    lock_path = directory / "pnpm-lock.yaml"
    if _package_manager(package_path) != EXPECTED_PNPM:
        _fail(f"{name}/package.json must pin {EXPECTED_PNPM}")
    workspace = workspace_path.read_text(encoding="utf-8")
    age = int(_yaml_scalar(workspace, "minimumReleaseAge", path=workspace_path))
    if age < MINIMUM_RELEASE_AGE_MINUTES:
        _fail(f"{name} minimumReleaseAge must be at least one day")
    if _yaml_scalar(workspace, "enableGlobalVirtualStore", path=workspace_path) != "false":
        _fail(f"{name} must isolate node_modules from the global virtual store")
    builds = _allow_builds(workspace, path=workspace_path)
    if builds != expected_builds:
        _fail(f"{name} allowBuilds must be exactly {expected_builds}, got {builds}")
    lock = lock_path.read_text(encoding="utf-8")
    if not re.search(r"(?m)^lockfileVersion: ['\"]9\.0['\"]$", lock):
        _fail(f"{name}/pnpm-lock.yaml must use lockfileVersion 9.0")


def _check_workflow() -> None:
    path = ROOT / ".github" / "workflows" / "ci.yml"
    workflow = path.read_text(encoding="utf-8")
    required = (
        "runs-on: [self-hosted, linux, x64]",
        "persist-credentials: false",
        "UV_PYTHON_DOWNLOADS: never",
        "git diff --check",
    )
    for value in required:
        if value not in workflow:
            _fail(f"CI workflow lacks {value!r}")
    forbidden = (
        "upload-artifact",
        "actions/cache",
        "scp ",
        "ssh ",
        "a100",
        "doc-rag-translate",
        "bucket_originals",
    )
    lowered = workflow.lower()
    for value in forbidden:
        if value in lowered:
            _fail(f"CI workflow contains forbidden external/production operation {value!r}")


def main() -> None:
    if sys.version_info[:2] != (3, 13):
        _fail(f"policy checker requires Python 3.13, got {sys.version.split()[0]}")
    if (ROOT / ".python-version").read_text(encoding="utf-8").strip() != "3.13":
        _fail(".python-version must remain pinned to 3.13")
    _check_workspace("web", expected_builds={"esbuild": True})
    _check_workspace(
        "extension",
        expected_builds={"esbuild": True, "spawn-sync": False},
    )
    _check_workflow()
    print("CI policy: OK")


if __name__ == "__main__":
    main()
