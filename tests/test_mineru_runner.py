from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest

from rag_app.pipeline import parse


class _FailedProcess:
    returncode = 1

    async def communicate(self) -> tuple[bytes, None]:
        return b"server unavailable", None

    def kill(self) -> None:
        return None


def test_run_mineru_can_disable_pipeline_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commands: list[tuple[Any, ...]] = []

    async def create_subprocess_exec(*command: Any, **_kwargs: Any) -> _FailedProcess:
        commands.append(command)
        return _FailedProcess()

    monkeypatch.setattr(parse.asyncio, "create_subprocess_exec", create_subprocess_exec)

    with pytest.raises(RuntimeError, match="код 1"):
        asyncio.run(
            parse.run_mineru(
                tmp_path / "input.pdf",
                tmp_path / "out",
                backend="vlm-http-client",
                allow_fallback=False,
            )
        )

    assert len(commands) == 1
    assert "vlm-http-client" in commands[0]

    commands.clear()
    with pytest.raises(RuntimeError, match="код 1"):
        asyncio.run(
            parse.run_mineru(
                tmp_path / "input.pdf",
                tmp_path / "fallback-out",
                backend="vlm-http-client",
            )
        )

    assert len(commands) == 2
    assert "pipeline" in commands[1]


def test_run_mineru_uses_selected_runtime_bin_first_in_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_bin = tmp_path / "runtime" / "bin" / "mineru"
    runtime_bin.parent.mkdir(parents=True)
    runtime_bin.write_text("", encoding="utf-8")
    environments: list[dict[str, str]] = []

    async def create_subprocess_exec(*_command: Any, **kwargs: Any) -> _FailedProcess:
        environments.append(kwargs["env"])
        return _FailedProcess()

    monkeypatch.setattr(parse.settings, "mineru_bin", str(runtime_bin))
    monkeypatch.setattr(parse.asyncio, "create_subprocess_exec", create_subprocess_exec)

    with pytest.raises(RuntimeError, match="код 1"):
        asyncio.run(
            parse.run_mineru(
                tmp_path / "input.pdf",
                tmp_path / "out",
                backend="vlm-http-client",
                allow_fallback=False,
            )
        )

    assert environments[0]["PATH"].split(os.pathsep, 1)[0] == str(runtime_bin.parent)


def test_run_mineru_defaults_to_pinned_runtime_when_installed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pinned_bin = tmp_path / "services" / "mineru" / "current" / "bin" / "mineru"
    pinned_bin.parent.mkdir(parents=True)
    pinned_bin.write_text("", encoding="utf-8")
    commands: list[tuple[Any, ...]] = []

    async def create_subprocess_exec(*command: Any, **_kwargs: Any) -> _FailedProcess:
        commands.append(command)
        return _FailedProcess()

    monkeypatch.setattr(parse.settings, "mineru_bin", None)
    monkeypatch.setattr(parse, "_PINNED_MINERU_BIN", pinned_bin)
    monkeypatch.setattr(parse.asyncio, "create_subprocess_exec", create_subprocess_exec)

    with pytest.raises(RuntimeError, match="код 1"):
        asyncio.run(
            parse.run_mineru(
                tmp_path / "input.pdf",
                tmp_path / "out",
                backend="vlm-http-client",
                allow_fallback=False,
            )
        )

    assert commands[0][0] == str(pinned_bin)
