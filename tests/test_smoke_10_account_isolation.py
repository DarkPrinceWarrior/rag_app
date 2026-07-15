from __future__ import annotations

import asyncio
import importlib.util
import json
import stat
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "smoke_10_account_isolation.py"
    spec = importlib.util.spec_from_file_location("smoke_10_account_isolation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_script()


class _Result:
    def all(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(document_id=uuid.UUID(int=2))]


class _Session:
    def __init__(self) -> None:
        self.parameters: dict[str, object] | None = None
        self.statement = ""

    async def execute(self, statement: object, parameters: dict[str, object]) -> _Result:
        self.statement = str(statement)
        self.parameters = parameters
        return _Result()


def test_hierarchy_probe_reuses_production_sql_and_explicit_scope() -> None:
    session = _Session()
    anchor_id = uuid.UUID(int=1)

    rows = asyncio.run(
        runner._hierarchy_rows(
            session,
            anchor_ids=[anchor_id],
            owner_sub="keycloak-subject",
        )
    )

    assert len(rows) == 1
    assert "anchor_input AS MATERIALIZED" in session.statement
    assert session.parameters == {
        "doc_id": None,
        "doc_ids": None,
        "folder_id": None,
        "owner": "keycloak-subject",
        "anchor_ids": [anchor_id],
        "page_radius": runner.settings.rag_hierarchical_page_radius,
        "per_anchor_k": runner.settings.rag_hierarchical_per_anchor_k,
        "expansion_k": runner.settings.rag_hierarchical_max_candidates,
    }


def test_private_hierarchy_evidence_is_written_with_mode_0600(tmp_path: Path) -> None:
    path = tmp_path / "hierarchical-rls.json"
    payload = {"schema_version": "hierarchical-rls-evidence-v1", "passed": True}

    runner._write_private_report(path, payload)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_text(encoding="utf-8")) == payload
