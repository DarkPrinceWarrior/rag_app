"""Статические гарантии эксплуатационного scaffold без обращения к production."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_runtime_flags_are_declarative_and_non_secret() -> None:
    lines = {
        line
        for raw in _read("deploy/rag-runtime.env").splitlines()
        if (line := raw.strip()) and not line.startswith("#")
    }
    assert lines == {
        "RAG_RAG_CONTEXT_BUDGET_MODE=enforce",
        "RAG_RAG_QUANTITY_GUARD_MODE=warn",
        "RAG_VISUAL_ENABLED=true",
        "RAG_PARSER_QUALITY_SHADOW_ENABLED=true",
        "RAG_PARSER_PAGE_ROUTER_MODE=shadow",
        "RAG_TRANSLATION_ENTITY_GUARD_MODE=shadow",
        "RAG_TRANSLATION_MEMORY_MODE=enforce",
        "RAG_TRANSLATION_MEMORY_NEAREST_TOP_K=0",
        "RAG_QUEUE_ROLLOUT_MODE=split",
    }


def test_app_units_load_tracked_flags_after_role_specific_env() -> None:
    expected = {
        "deploy/rag-api.service": "/etc/docragenslate/api.env",
        "deploy/rag-worker.service": "/etc/docragenslate/worker.env",
        "deploy/rag-queue-worker@.service": "/etc/docragenslate/worker.env",
    }
    for unit, role_env in expected.items():
        text = _read(unit)
        local_env = text.index(f"EnvironmentFile={role_env}")
        runtime_env = text.index("EnvironmentFile=/root/projects/rag_app/deploy/rag-runtime.env")
        assert local_env < runtime_env
        assert "EnvironmentFile=-/root/projects/rag_app/.env" not in text
        assert "Environment=PYTHON_DOTENV_DISABLED=1" in text
        assert "Restart=on-failure" in text
    assert "TimeoutStopSec=3600" in _read("deploy/rag-worker.service")


def test_metrics_exporter_receives_only_loopback_redis_env() -> None:
    text = _read("deploy/rag-metrics-exporter.service")
    environment = "\n".join(
        line for line in text.splitlines() if line.startswith("Environment=")
    )
    assert "EnvironmentFile=" not in text
    assert "WorkingDirectory=/\n" in text
    assert "WorkingDirectory=/root/projects/rag_app" not in text
    assert "Environment=RAG_REDIS_HOST=127.0.0.1" in environment
    assert "Environment=RAG_REDIS_PORT=6379" in environment
    assert "Environment=RAG_REDIS_DB=0" in environment
    assert "DATABASE" not in environment
    assert "S3_" not in environment
    assert "OIDC" not in environment


def _run_env_preparation(
    tmp_path: Path,
    api_source: str,
    worker_source: str,
    common_source: str = (
        "RAG_DATABASE_URL=postgresql+asyncpg://rag:owner-secret@db/rag_app\n"
        "RAG_AUTH_ENABLED=true\n"
        "RAG_S3_SECRET_KEY=s3-secret\n"
        "RAG_PG_USER=rag\n"
        "RAG_PG_PASSWORD=compose-owner-secret\n"
        "RAG_PG_DB=rag_app\n"
    ),
) -> subprocess.CompletedProcess[str]:
    common = tmp_path / "common.source"
    api = tmp_path / "api.source"
    worker = tmp_path / "worker.source"
    common.write_text(common_source, encoding="utf-8")
    api.write_text(api_source, encoding="utf-8")
    worker.write_text(worker_source, encoding="utf-8")
    env = {
        **os.environ,
        "REPO_DIR": str(ROOT),
        "SERVICE_ENV_DIR": str(tmp_path / "service-env"),
        "COMMON_ENV_SOURCE": str(common),
        "API_ENV_SOURCE": str(api),
        "WORKER_ENV_SOURCE": str(worker),
        "PYTHON_BIN": str(ROOT / ".venv/bin/python"),
    }
    return subprocess.run(
        [str(ROOT / "deploy/install_rag_app_services.sh"), "--prepare-env"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_installer_normalizes_role_env_without_printing_secrets(tmp_path: Path) -> None:
    result = _run_env_preparation(
        tmp_path,
        "export RAG_DATABASE_URL=postgresql+asyncpg://rag_api:api-secret@db/rag_app\n"
        "RAG_VISUAL_ENABLED=true\n",
        "export RAG_DATABASE_URL=postgresql+asyncpg://rag_worker:worker-secret@db/rag_app\n",
    )
    assert result.returncode == 0, result.stderr
    assert "api-secret" not in result.stdout + result.stderr
    assert "worker-secret" not in result.stdout + result.stderr
    assert "owner-secret" not in result.stdout + result.stderr
    assert "s3-secret" not in result.stdout + result.stderr
    assert "compose-owner-secret" not in result.stdout + result.stderr
    api = tmp_path / "service-env/api.env"
    worker = tmp_path / "service-env/worker.env"
    assert "RAG_AUTH_ENABLED=true" in api.read_text()
    assert "RAG_AUTH_ENABLED=true" in worker.read_text()
    assert "RAG_S3_SECRET_KEY=s3-secret" in api.read_text()
    assert "rag:owner-secret" not in api.read_text() + worker.read_text()
    assert "RAG_PG_USER=" not in api.read_text() + worker.read_text()
    assert "RAG_PG_PASSWORD=" not in api.read_text() + worker.read_text()
    assert "RAG_PG_DB=" not in api.read_text() + worker.read_text()
    assert api.read_text().count("RAG_DATABASE_URL=") == 1
    assert worker.read_text().count("RAG_DATABASE_URL=") == 1
    assert "export " not in api.read_text() + worker.read_text()
    assert stat.S_IMODE(api.stat().st_mode) == 0o600
    assert stat.S_IMODE(worker.stat().st_mode) == 0o600
    assert stat.S_IMODE(api.parent.stat().st_mode) == 0o700


def test_installer_rejects_owner_or_swapped_database_roles(tmp_path: Path) -> None:
    result = _run_env_preparation(
        tmp_path,
        "RAG_DATABASE_URL=postgresql+asyncpg://rag:secret@db/rag_app\n",
        "RAG_DATABASE_URL=postgresql+asyncpg://rag_api:secret@db/rag_app\n",
    )
    assert result.returncode != 0
    assert "не прошёл проверку роли rag_api" in result.stderr
    assert not (tmp_path / "service-env").exists()


def test_installer_rejects_shell_commands_in_role_env(tmp_path: Path) -> None:
    result = _run_env_preparation(
        tmp_path,
        "source /tmp/unsafe\n"
        "RAG_DATABASE_URL=postgresql+asyncpg://rag_api:secret@db/rag_app\n",
        "RAG_DATABASE_URL=postgresql+asyncpg://rag_worker:secret@db/rag_app\n",
    )
    assert result.returncode != 0
    assert "неподдерживаемый синтаксис" in result.stderr
    assert not (tmp_path / "service-env").exists()


def test_installer_rejects_shell_commands_in_common_env(tmp_path: Path) -> None:
    result = _run_env_preparation(
        tmp_path,
        "RAG_DATABASE_URL=postgresql+asyncpg://rag_api:secret@db/rag_app\n",
        "RAG_DATABASE_URL=postgresql+asyncpg://rag_worker:secret@db/rag_app\n",
        common_source="source /tmp/unsafe\nRAG_AUTH_ENABLED=true\n",
    )
    assert result.returncode != 0
    assert "общая production-конфигурация имеет неподдерживаемый синтаксис" in result.stderr
    assert not (tmp_path / "service-env").exists()


def test_installer_requires_exactly_one_database_url_per_role(tmp_path: Path) -> None:
    result = _run_env_preparation(
        tmp_path,
        "RAG_DATABASE_URL=postgresql+asyncpg://rag_api:secret@db/rag_app\n"
        "RAG_DATABASE_URL=postgresql+asyncpg://rag_api:other@db/rag_app\n",
        "RAG_DATABASE_URL=postgresql+asyncpg://rag_worker:secret@db/rag_app\n",
    )
    assert result.returncode != 0
    assert "не прошёл проверку роли rag_api" in result.stderr
    assert "secret" not in result.stdout + result.stderr
    assert "other" not in result.stdout + result.stderr
    assert not (tmp_path / "service-env").exists()


def test_split_workers_are_installed_but_not_enabled_with_legacy_target() -> None:
    app_target = _read("deploy/rag-app.target")
    split_target = _read("deploy/rag-split-workers.target")
    installer = _read("deploy/install_rag_app_services.sh")
    wrapper = _read("deploy/run_queue_worker.sh")
    assert "rag-split-workers.target" not in app_target
    assert all(
        f"rag-queue-worker@{profile}.service" in split_target
        for profile in ("parse", "translate", "export-index", "memory")
    )
    assert "systemctl enable rag-app.target" in installer
    assert "systemctl enable rag-split-workers.target" not in installer
    assert '*) echo "usage: $0 {parse|translate|export-index|memory}"' in wrapper


def test_vllm_candidate_profiles_cannot_target_parser_environments() -> None:
    prepare = _read("deploy/vllm/prepare_candidate_env.sh")
    runner = _read("deploy/vllm/run_candidate.sh")
    assert "/root/services/mineru" in prepare
    assert "отказ: кандидат нельзя устанавливать" in prepare
    assert "mineru)" not in runner
    assert "paddle)" not in runner


def test_gpu4_profiles_use_pinned_runtime_and_direct_reranker() -> None:
    embedding = _read("deploy/vllm-embedding.service")
    reranker = _read("deploy/vllm-reranker.service")
    runner = _read("deploy/vllm/run_candidate.sh")

    assert "/root/services/vllm-main-0.24.0/.venv/bin/vllm" in embedding
    assert "--enforce-eager" in embedding
    assert "--dtype float16" in embedding
    assert "/root/services/vllm-main-0.24.0/.venv/bin/uvicorn" in reranker
    assert "direct_qwen3_reranker_server:app" in reranker
    assert "DIRECT_RERANK_DTYPE=bfloat16" in reranker
    assert "--runner pooling" not in reranker
    assert "direct-reranker" in runner
    assert "systemctl is-active --quiet" in runner


def test_backup_and_redteam_yaml_are_valid() -> None:
    for relative in (
        "deploy/backup/docker-compose.backup.yml",
        "deploy/redteam/promptfooconfig.yaml",
        "deploy/redteam/cases.yaml",
    ):
        assert yaml.safe_load(_read(relative)) is not None


def test_backup_root_guard_rejects_root_symlink_and_volatile_filesystem(tmp_path: Path) -> None:
    common = ROOT / "deploy/backup/backup_common.sh"
    symlink = tmp_path / "backup-link"
    symlink.symlink_to("/dev/shm", target_is_directory=True)
    for candidate, expected in (
        ("/", "корень системы"),
        (str(ROOT), "не является отдельной точкой монтирования"),
        (str(symlink), "не должен быть symlink"),
        ("/dev/shm", "не является долговременным носителем"),
    ):
        result = subprocess.run(
            ["bash", "-c", 'source "$1"; validate_backup_root', "backup-test", str(common)],
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "RAG_BACKUP_ROOT": candidate},
        )
        assert result.returncode != 0
        assert expected in result.stderr


def test_backup_schedule_is_tracked_but_not_installed_by_app_installer() -> None:
    service = _read("deploy/backup/pgbackrest-backup@.service")
    full = _read("deploy/backup/pgbackrest-backup-full.timer")
    diff = _read("deploy/backup/pgbackrest-backup-diff.timer")
    wal = _read("deploy/backup/pgbackrest-wal-check.timer")
    installer = _read("deploy/install_rag_app_services.sh")
    assert "RequiresMountsFor=/backup" in service
    assert "EnvironmentFile=-/etc/docragenslate/backup-storage.env" in service
    assert "OnCalendar=Sun *-*-* 02:00:00" in full
    assert "OnCalendar=Mon..Sat *-*-* 02:00:00" in diff
    assert "OnCalendar=hourly" in wal
    assert "pgbackrest-backup" not in installer


def test_backup_scripts_share_fail_closed_guard_and_content_free_markers() -> None:
    common = _read("deploy/backup/backup_common.sh")
    assert '"$resolved" != /' in common
    assert '"$backup_device" != "$root_device"' in common
    assert "tmpfs|ramfs|overlay|overlayfs" in common
    assert "RAG_BACKUP_EXPECTED_SOURCE" in common
    assert "RAG_BACKUP_EXPECTED_FSTYPE" in common
    for relative in (
        "deploy/backup/pgbackrest_backup.sh",
        "deploy/backup/keycloak_export.sh",
        "deploy/backup/restore_drill.sh",
    ):
        text = _read(relative)
        assert "backup_common.sh" in text
        assert "validate_backup_root" in text
    assert "record_backup_success minio_mirror" in _read("deploy/backup/minio_mirror.sh")
    assert "record_backup_success keycloak_export" in _read("deploy/backup/keycloak_export.sh")
    assert "record_backup_success wal_archive_check" in _read(
        "deploy/backup/pgbackrest_backup.sh"
    )


def test_restore_drill_refuses_production_targets() -> None:
    text = _read("deploy/backup/restore_drill.sh")
    assert "/var/lib/docker/*|/root/projects/rag_app/*" in text
    assert "127.0.0.1:9000" in text


def test_agent_instruction_files_remain_identical() -> None:
    assert (ROOT / "CLAUDE.md").read_bytes() == (ROOT / "AGENTS.md").read_bytes()
