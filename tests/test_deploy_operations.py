"""Статические гарантии эксплуатационного scaffold без обращения к production."""

from __future__ import annotations

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
        "RAG_VISUAL_ENABLED=true",
        "RAG_PARSER_QUALITY_SHADOW_ENABLED=true",
        "RAG_PARSER_PAGE_ROUTER_MODE=shadow",
        "RAG_QUEUE_ROLLOUT_MODE=legacy",
    }


def test_app_units_load_tracked_flags_after_local_env() -> None:
    for unit in (
        "deploy/rag-api.service",
        "deploy/rag-worker.service",
        "deploy/rag-metrics-exporter.service",
        "deploy/rag-queue-worker@.service",
    ):
        text = _read(unit)
        local_env = text.index("EnvironmentFile=-/root/projects/rag_app/.env")
        runtime_env = text.index("EnvironmentFile=/root/projects/rag_app/deploy/rag-runtime.env")
        assert local_env < runtime_env
        assert "Restart=on-failure" in text
    assert "TimeoutStopSec=3600" in _read("deploy/rag-worker.service")


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
    assert "systemctl is-active --quiet" in runner


def test_backup_and_redteam_yaml_are_valid() -> None:
    for relative in (
        "deploy/backup/docker-compose.backup.yml",
        "deploy/redteam/promptfooconfig.yaml",
        "deploy/redteam/cases.yaml",
    ):
        assert yaml.safe_load(_read(relative)) is not None


def test_restore_drill_refuses_production_targets() -> None:
    text = _read("deploy/backup/restore_drill.sh")
    assert "/var/lib/docker/*|/root/projects/rag_app/*" in text
    assert "127.0.0.1:9000" in text


def test_agent_instruction_files_remain_identical() -> None:
    assert (ROOT / "CLAUDE.md").read_bytes() == (ROOT / "AGENTS.md").read_bytes()
