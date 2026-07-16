"""Метрики очередей и качества не раскрывают document/user identifiers."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from prometheus_client import CollectorRegistry, generate_latest

from rag_app.metrics_exporter import (
    _backup_marker_timestamp,
    _backup_metrics,
    _number_guard_counts,
)
from rag_app.workers.queueing import _document_number_guard_counts

ROOT = Path(__file__).resolve().parents[1]


def test_number_guard_counts_use_absolute_document_snapshots() -> None:
    protected, unconfirmed = _number_guard_counts(
        {
            b"doc-a": b"10|2",
            b"doc-b": b"5|0",
            b"invalid": b"broken",
            b"invalid-number": b"ten|two",
        }
    )
    assert protected == 15
    assert unconfirmed == 2


def test_backup_markers_fail_closed_and_export_only_fixed_operations(tmp_path: Path) -> None:
    (tmp_path / "pgbackrest_full.timestamp").write_text("150\n", encoding="ascii")
    (tmp_path / "pgbackrest_diff.timestamp").write_text("not-a-time\n", encoding="ascii")
    (tmp_path / "wal_archive_check.timestamp").write_text("999999\n", encoding="ascii")
    assert _backup_marker_timestamp(tmp_path / "pgbackrest_full.timestamp", now=200) == 150
    assert _backup_marker_timestamp(tmp_path / "pgbackrest_diff.timestamp", now=200) == 0
    assert _backup_marker_timestamp(tmp_path / "wal_archive_check.timestamp", now=200) == 0

    registry = CollectorRegistry()
    _backup_metrics(registry, state_dir=tmp_path, now=200)
    payload = generate_latest(registry).decode()

    assert 'rag_backup_last_success_timestamp_seconds{operation="pgbackrest_full"} 150.0' in payload
    assert 'rag_backup_age_seconds{operation="pgbackrest_full"} 50.0' in payload
    assert 'rag_backup_last_success_timestamp_seconds{operation="pgbackrest_diff"} 0.0' in payload
    assert 'rag_backup_age_seconds{operation="pgbackrest_diff"} -1.0' in payload
    assert "not-a-time" not in payload


def test_document_number_guard_includes_quantities_with_units() -> None:
    protected, unconfirmed = _document_number_guard_counts(
        [
            {
                "entity_guard": {
                    "protected": {"number": 2, "measurement": 3, "standard": 1},
                    "unconfirmed": {"number": 1, "measurement": 2, "standard": 1},
                }
            },
            None,
            {"entity_guard": "invalid"},
        ]
    )

    assert protected == 5
    assert unconfirmed == 3


def test_prometheus_config_has_all_required_targets() -> None:
    config = yaml.safe_load((ROOT / "deploy/monitoring/prometheus.yml").read_text())
    jobs = {job["job_name"]: job for job in config["scrape_configs"]}
    assert {
        "rag-api",
        "rag-pipeline",
        "vllm",
        "dcgm",
        "postgres-exporter",
        "redis-exporter",
        "minio",
    } <= jobs.keys()
    vllm_targets = {
        target
        for group in jobs["vllm"]["static_configs"]
        for target in group["targets"]
    }
    assert {
        "127.0.0.1:8002",
        "127.0.0.1:8003",
        "127.0.0.1:8005",
        "127.0.0.1:8006",
        "127.0.0.1:8007",
        "127.0.0.1:8118",
        "127.0.0.1:8120",
        "127.0.0.1:30010",
    } == vllm_targets


def test_alerts_and_dashboard_are_valid_and_cover_dlq_numbers_gpu() -> None:
    alerts = yaml.safe_load((ROOT / "deploy/monitoring/alerts.yml").read_text())
    names = {
        rule["alert"]
        for group in alerts["groups"]
        for rule in group["rules"]
    }
    assert {
        "RagDeadLetterQueueNotEmpty",
        "RagUnconfirmedNumbers",
        "RagGpuTemperatureHigh",
        "RagDatabaseBackupStale",
        "RagWalArchiveCheckStale",
        "RagMinioMirrorStale",
        "RagKeycloakExportStale",
    } <= names
    dashboard = json.loads(
        (ROOT / "deploy/monitoring/grafana/dashboards/rag-operations.json").read_text()
    )
    expressions = "\n".join(
        target["expr"]
        for panel in dashboard["panels"]
        for target in panel.get("targets", [])
    )
    assert "rag_arq_queue_depth" in expressions
    assert "rag_translation_number_unconfirmed_ratio" in expressions
    assert "DCGM_FI_DEV_FB_USED" in expressions


def test_prometheus_labels_do_not_include_sensitive_ids() -> None:
    exporter = (ROOT / "src/rag_app/metrics_exporter.py").read_text()
    assert '["document_id"]' not in exporter
    assert '["user_id"]' not in exporter
    assert '["owner_sub"]' not in exporter
