"""Статические гарантии безопасного bootstrap контура monitoring."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_monitoring_bootstrap_uses_dedicated_least_privilege_role() -> None:
    text = _read("deploy/monitoring/bootstrap_runtime.sh")
    assert 'METRICS_ROLE="rag_metrics"' in text
    assert "GRANT pg_monitor TO rag_metrics" in text
    assert 'GRANT CONNECT ON DATABASE :"database_name" TO rag_metrics' in text
    assert '-v database_name="${POSTGRES_DATABASE}"' in text
    assert "NOSUPERUSER NOCREATEDB NOCREATEROLE INHERIT" in text
    assert "GRANT ALL" not in text
    assert "rag_api" not in text
    assert "rag_worker" not in text


def test_monitoring_bootstrap_keeps_secrets_out_of_output_and_git() -> None:
    text = _read("deploy/monitoring/bootstrap_runtime.sh")
    gitignore = _read(".gitignore")
    assert "deploy/monitoring/secrets/" in gitignore
    assert ".env" in gitignore
    assert 'umask 077' in text
    assert 'chmod 0600 "${env_tmp}"' in text
    assert 'chmod 0600 "${token_tmp}"' in text
    assert 'PROMETHEUS_UID="${PROMETHEUS_UID:-65534}"' in text
    assert 'PROMETHEUS_GID="${PROMETHEUS_GID:-65534}"' in text
    assert 'chown "${PROMETHEUS_UID}:${PROMETHEUS_GID}"' in text
    assert "'/^RAG_GRAFANA_PASSWORD=/p'" in text
    assert "set -x" not in text
    assert "echo \"${metrics_password}" not in text
    assert "cat \"${SCRIPT_DIR}/.env\"" not in text


def test_monitoring_compose_keeps_public_ports_closed() -> None:
    compose = yaml.safe_load(_read("deploy/monitoring/docker-compose.yml"))
    for service in compose["services"].values():
        assert "ports" not in service
        assert service.get("network_mode") == "host"
    assert compose["services"]["dcgm-exporter"]["cap_add"] == ["SYS_ADMIN"]
    assert compose["services"]["dcgm-exporter"]["profiles"] == ["dcgm"]
