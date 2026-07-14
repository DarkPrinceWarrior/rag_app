from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SWITCH_SCRIPT = PROJECT_ROOT / "deploy" / "switch_mineru_runtime.sh"
SERVICE_UNIT = PROJECT_ROOT / "deploy" / "mineru-vllm.service"
SAMPLING_PATCH = PROJECT_ROOT / "deploy" / "patches" / "mineru_sampling.sh"


def test_switch_script_has_valid_bash_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", str(SWITCH_SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_switch_script_rejects_unsupported_version_before_touching_runtime() -> None:
    result = subprocess.run(
        ["bash", str(SWITCH_SCRIPT), "3.4.5"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "unsupported argument: 3.4.5" in result.stderr


def test_switch_script_defaults_to_no_restart_and_verifies_runtime() -> None:
    source = SWITCH_SCRIPT.read_text(encoding="utf-8")

    assert 'restart=false' in source
    assert 'if [[ "$restart" == true ]]' in source
    assert 'version("mineru")' in source
    assert '[[ -x "$server_bin" ]]' in source
    assert '[[ -x "$client_bin" ]]' in source
    assert 'case "$runtime_real" in' in source
    assert 'mv -Tf -- "$temporary_link" "$CURRENT_LINK"' in source
    assert 'wait_for_health' in source
    assert 'rollback_to_previous_runtime' in source
    assert "readonly HEALTH_TIMEOUT_SECONDS=180" in source
    assert "SECONDS + HEALTH_TIMEOUT_SECONDS" in source
    assert "up to three minutes" in source


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def _switch_test_harness(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    service_root = tmp_path / "services" / "mineru"
    runtimes_root = service_root / "runtimes"
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    runtimes_root.mkdir(parents=True)

    for version in ("3.3.1", "3.4.4"):
        bin_dir = runtimes_root / version / "bin"
        bin_dir.mkdir(parents=True)
        _write_executable(bin_dir / "python", f"#!/usr/bin/env bash\nprintf %s {version!r}\n")
        _write_executable(bin_dir / "mineru", "#!/usr/bin/env bash\nexit 0\n")
        _write_executable(bin_dir / "mineru-vllm-server", "#!/usr/bin/env bash\nexit 0\n")

    systemctl_log = tmp_path / "systemctl.log"
    _write_executable(
        fake_bin / "systemctl",
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$*" >>"$SWITCH_SYSTEMCTL_LOG"\n'
        'if [[ "$1" == restart && -n "${SWITCH_FAIL_RESTART_ONCE:-}" '
        '&& ! -e "$SWITCH_RESTART_FAILED_MARKER" ]]; then\n'
        '  : >"$SWITCH_RESTART_FAILED_MARKER"\n'
        "  exit 1\n"
        "fi\n"
        "exit 0\n",
    )
    _write_executable(
        fake_bin / "curl",
        "#!/usr/bin/env bash\n"
        'target="$(realpath -e -- "$SWITCH_CURRENT_LINK")"\n'
        '[[ "$target" != *"/${SWITCH_UNHEALTHY_VERSION:-__none__}" ]]\n',
    )

    source = SWITCH_SCRIPT.read_text(encoding="utf-8")
    source = source.replace(
        'readonly SERVICE_ROOT="/root/services/mineru"',
        f'readonly SERVICE_ROOT="{service_root}"',
    )
    source = source.replace(
        "readonly HEALTH_TIMEOUT_SECONDS=180",
        "readonly HEALTH_TIMEOUT_SECONDS=0",
    )
    source = source.replace(
        "readonly HEALTH_POLL_INTERVAL_SECONDS=1",
        "readonly HEALTH_POLL_INTERVAL_SECONDS=0",
    )
    script = tmp_path / "switch_mineru_runtime.sh"
    _write_executable(script, source)

    current_link = service_root / "current"
    current_link.symlink_to(runtimes_root / "3.3.1")
    return script, current_link, fake_bin, systemctl_log


def _switch_environment(
    *,
    current_link: Path,
    fake_bin: Path,
    systemctl_log: Path,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "SWITCH_CURRENT_LINK": str(current_link),
            "SWITCH_RESTART_FAILED_MARKER": str(current_link.parent / "restart-failed"),
            "SWITCH_SYSTEMCTL_LOG": str(systemctl_log),
        }
    )
    return environment


def test_switch_restart_rolls_back_when_candidate_never_becomes_healthy(
    tmp_path: Path,
) -> None:
    script, current_link, fake_bin, systemctl_log = _switch_test_harness(tmp_path)
    environment = _switch_environment(
        current_link=current_link,
        fake_bin=fake_bin,
        systemctl_log=systemctl_log,
    )
    environment["SWITCH_UNHEALTHY_VERSION"] = "3.4.4"

    result = subprocess.run(
        ["bash", str(script), "3.4.4", "--restart"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode != 0
    assert current_link.resolve().name == "3.3.1"
    assert "restored healthy runtime" in result.stderr
    assert systemctl_log.read_text(encoding="utf-8").splitlines().count(
        "restart mineru-vllm.service"
    ) == 2


def test_switch_restart_rolls_back_after_restart_failure(tmp_path: Path) -> None:
    script, current_link, fake_bin, systemctl_log = _switch_test_harness(tmp_path)
    environment = _switch_environment(
        current_link=current_link,
        fake_bin=fake_bin,
        systemctl_log=systemctl_log,
    )
    environment["SWITCH_FAIL_RESTART_ONCE"] = "1"

    result = subprocess.run(
        ["bash", str(script), "3.4.4", "--restart"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode != 0
    assert current_link.resolve().name == "3.3.1"
    assert "restored healthy runtime" in result.stderr
    assert systemctl_log.read_text(encoding="utf-8").splitlines().count(
        "restart mineru-vllm.service"
    ) == 2


def test_switch_restart_keeps_healthy_candidate(tmp_path: Path) -> None:
    script, current_link, fake_bin, systemctl_log = _switch_test_harness(tmp_path)
    environment = _switch_environment(
        current_link=current_link,
        fake_bin=fake_bin,
        systemctl_log=systemctl_log,
    )

    result = subprocess.run(
        ["bash", str(script), "3.4.4", "--restart"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert current_link.resolve().name == "3.4.4"
    assert systemctl_log.read_text(encoding="utf-8").splitlines().count(
        "restart mineru-vllm.service"
    ) == 1


def test_mineru_service_uses_pinned_runtime_only() -> None:
    source = SERVICE_UNIT.read_text(encoding="utf-8")

    assert "Environment=PATH=/root/services/mineru/current/bin:" in source
    assert "/root/services/mineru/current/bin/mineru-vllm-server" in source
    assert "/root/projects/rag_app/.venv/bin" not in source


@pytest.fixture
def fake_mineru_runtime(tmp_path: Path) -> tuple[Path, Path]:
    runtime = tmp_path / "runtime"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(runtime)],
        check=True,
        capture_output=True,
        text=True,
    )
    site_packages = next((runtime / "lib").glob("python*/site-packages"))
    package = site_packages / "mineru_vl_utils"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    client = package / "mineru_client.py"
    return runtime, client


@pytest.mark.parametrize("initial_value", ["1.0", "1.1"])
def test_sampling_patch_detaches_hardlink_and_preserves_metadata(
    fake_mineru_runtime: tuple[Path, Path],
    tmp_path: Path,
    initial_value: str,
) -> None:
    runtime, client = fake_mineru_runtime
    client.write_text(
        "class Sampling:\n"
        "    def __init__(\n"
        f"        repetition_penalty: float | None = {initial_value},\n"
        "    ): ...\n",
        encoding="utf-8",
    )
    client.chmod(0o640)
    cache_sibling = tmp_path / "cached_mineru_client.py"
    cache_sibling.hardlink_to(client)
    subprocess.run(
        [
            str(runtime / "bin" / "python"),
            "-I",
            "-c",
            "import mineru_vl_utils.mineru_client",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    bytecode = list((client.parent / "__pycache__").glob("mineru_client.*.pyc"))
    assert bytecode
    original_inode = client.stat().st_ino
    original_cache = cache_sibling.read_bytes()

    result = subprocess.run(
        ["bash", str(SAMPLING_PATCH), str(runtime), "1.1"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "repetition_penalty: float | None = 1.1," in client.read_text(encoding="utf-8")
    assert client.stat().st_ino != original_inode
    assert client.stat().st_nlink == 1
    assert client.stat().st_mode & 0o777 == 0o640
    assert cache_sibling.stat().st_ino == original_inode
    assert cache_sibling.read_bytes() == original_cache
    assert not list((client.parent / "__pycache__").glob("mineru_client.*.pyc"))
    imported_default = subprocess.run(
        [
            str(runtime / "bin" / "python"),
            "-I",
            "-c",
            "import inspect; from mineru_vl_utils.mineru_client import Sampling; "
            "print(inspect.signature(Sampling.__init__).parameters['repetition_penalty'].default)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert imported_default.stdout.strip() == "1.1"


def test_sampling_patch_fails_closed_on_ambiguous_pattern(
    fake_mineru_runtime: tuple[Path, Path],
) -> None:
    runtime, client = fake_mineru_runtime
    source = (
        "repetition_penalty: float | None = 1.0,\n"
        "repetition_penalty: float | None = 1.0,\n"
    )
    client.write_text(source, encoding="utf-8")
    original_inode = client.stat().st_ino

    result = subprocess.run(
        ["bash", str(SAMPLING_PATCH), str(runtime), "1.1"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "expected exactly one repetition_penalty default" in result.stderr
    assert client.stat().st_ino == original_inode
    assert client.read_text(encoding="utf-8") == source


def test_sampling_patch_defaults_to_pinned_runtime() -> None:
    source = SAMPLING_PATCH.read_text(encoding="utf-8")

    assert 'VENV="${1:-/root/services/mineru/current}"' in source
