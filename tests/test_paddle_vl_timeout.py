from __future__ import annotations

import asyncio
from pathlib import Path

from rag_app.pipeline import paddle_vl


class _TimedOutProcess:
    returncode = None

    def __init__(self) -> None:
        self.killed = False
        self.waited = False

    async def communicate(self):
        raise TimeoutError

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> int:
        self.waited = True
        return -9


def test_timeout_kills_and_reaps_subprocess(tmp_path: Path, monkeypatch) -> None:
    process = _TimedOutProcess()

    async def fake_subprocess(*args, **kwargs):
        del args, kwargs
        return process

    source = tmp_path / "source.pdf"
    source.write_bytes(b"pdf")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_subprocess)

    try:
        asyncio.run(paddle_vl.run_paddle(source, tmp_path / "out", timeout_s=1))
    except RuntimeError as exc:
        assert "таймаут 1s" in str(exc)
    else:
        raise AssertionError("timeout must fail")

    assert process.killed
    assert process.waited
