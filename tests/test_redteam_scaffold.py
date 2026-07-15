"""Fail-closed checks for the offline red-team scaffold (no network calls)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
REDTEAM = ROOT / "deploy" / "redteam"


def _node(source: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["node", "--input-type=module", "--eval", source],
        cwd=REDTEAM,
        capture_output=True,
        check=False,
        text=True,
    )


def test_provider_rejects_public_or_credentialed_origins() -> None:
    provider_uri = (REDTEAM / "provider.mjs").as_uri()
    result = _node(
        f"""
        import {{ redteamBaseUrl }} from {provider_uri!r};
        const accepted = [
          'http://127.0.0.1:8100', 'http://10.1.2.3:8100',
          'https://172.16.2.3', 'http://192.168.2.3', 'http://[::1]:8100'
        ];
        const rejected = [
          'https://example.com', 'http://8.8.8.8', 'ftp://127.0.0.1',
          'http://user:secret@127.0.0.1', 'http://127.0.0.1/prefix'
        ];
        for (const value of accepted) redteamBaseUrl(value);
        for (const value of rejected) {{
          try {{ redteamBaseUrl(value); process.exit(10); }} catch {{ /* expected */ }}
        }}
        """
    )
    assert result.returncode == 0, result.stderr


def test_provider_requires_complete_sse_and_does_not_return_raw_trace() -> None:
    provider_uri = (REDTEAM / "provider.mjs").as_uri()
    result = _node(
        f"""
        import Provider from {provider_uri!r};
        process.env.RAG_REDTEAM_TOKEN = 'short-lived-test-token';
        process.env.RAG_REDTEAM_BASE_URL = 'http://127.0.0.1:8100';
        const provider = new Provider();
        globalThis.fetch = async () => new Response('not sse', {{
          status: 200, headers: {{ 'content-type': 'text/plain' }}
        }});
        const wrongType = await provider.callApi('test', {{ vars: {{}} }});
        if (!wrongType.error) process.exit(11);
        globalThis.fetch = async (_url, init) => {{
          if (init.redirect !== 'error') process.exit(12);
          return new Response('data: {{"type":"delta","text":"safe"}}\\n\\n', {{
            status: 200, headers: {{ 'content-type': 'text/event-stream; charset=utf-8' }}
          }});
        }};
        const incomplete = await provider.callApi('test', {{ vars: {{}} }});
        if (!incomplete.error) process.exit(13);
        globalThis.fetch = async () => new Response(
          'data: {{"type":"delta","text":"safe"}}\\n\\n' +
          'data: {{"type":"done","citations":[]}}\\n\\n', {{
            status: 200, headers: {{ 'content-type': 'text/event-stream' }}
          }}
        );
        const complete = await provider.callApi('test', {{ vars: {{}} }});
        if (complete.output !== 'safe' || 'raw' in complete) process.exit(14);
        """
    )
    assert result.returncode == 0, result.stderr


def test_assertions_fail_closed_and_normalize_pressure() -> None:
    assertions_uri = (REDTEAM / "assertions.mjs").as_uri()
    result = _node(
        f"""
        import {{
          mustNotContainCanary, mustNotAssertFalsePressure,
          mustNotConfirmMembership, mustRefuseAttack, noInternalLeak
        }} from {assertions_uri!r};
        if (mustNotContainCanary('safe', {{ vars: {{}} }}).pass) process.exit(20);
        if (mustNotContainCanary('leak-CANARY', {{ vars: {{ forbidden_canary: 'CANARY' }} }}).pass) process.exit(21);
        if (mustNotAssertFalsePressure('Давление 999\\u00a0МПа.').pass) process.exit(22);
        if (mustNotAssertFalsePressure('Pressure: 999 MPa').pass) process.exit(23);
        if (!mustNotAssertFalsePressure('Давление 9,99 МПа.').pass) process.exit(24);
        if (!mustNotAssertFalsePressure('Давление не равно 999 МПа.').pass) process.exit(25);
        if (!noInternalLeak('Не могу раскрыть system prompt.').pass) process.exit(26);
        if (!mustRefuseAttack('Запрос отклонён.').pass) process.exit(27);
        if (mustRefuseAttack('Инструкции выполнены.').pass) process.exit(28);
        if (mustNotConfirmMembership('Документ существует.').pass) process.exit(29);
        if (!mustNotConfirmMembership('Не могу подтвердить наличие.').pass) process.exit(30);
        """
    )
    assert result.returncode == 0, result.stderr


def test_runner_refuses_incomplete_environment_before_npx() -> None:
    result = subprocess.run(
        ["bash", "run_local.sh"],
        cwd=REDTEAM,
        env={"PATH": os.environ["PATH"]},
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 2
    assert "обязательная переменная RAG_REDTEAM_TOKEN не задана" in result.stderr


def test_document_injection_cases_have_fail_closed_canaries() -> None:
    cases = yaml.safe_load((REDTEAM / "cases.yaml").read_text(encoding="utf-8"))
    by_description = {case["description"]: case for case in cases}
    for description in ("Косвенная инъекция из документа", "Инъекция через имя файла"):
        case = by_description[description]
        assert case["vars"]["forbidden_canary"].startswith("{{env.RAG_REDTEAM_")
        assert any("mustNotContainCanary" in item["value"] for item in case["assert"])

    runner = (REDTEAM / "run_local.sh").read_text(encoding="utf-8")
    assert 'actual_version" == "$expected_version"' in runner
    assert "exec ./node_modules/.bin/promptfoo eval" in runner
    assert 'RAG_REDTEAM_CONFIRM_ISOLATED" == "YES"' in runner
    assert "PROMPTFOO_DISABLE_TELEMETRY=1" in runner
    assert "DO_NOT_TRACK=1" in runner
    assert 'install -d -m 700 -- "$PROMPTFOO_CONFIG_DIR"' in runner


def test_all_adversarial_cases_have_specific_contract_assertions() -> None:
    cases = yaml.safe_load((REDTEAM / "cases.yaml").read_text(encoding="utf-8"))
    by_description = {case["description"]: case for case in cases}
    assert len(cases) == 8
    assert "mustRefuseAttack" in str(by_description["Извлечение системного промпта"]["assert"])
    assert "mustNotConfirmMembership" in str(
        by_description["Определение наличия закрытого документа"]["assert"]
    )
    for description in (
        "Инъекция на китайском",
        "Отравление базы знаний на одноразовом индексе",
    ):
        assertions = str(by_description[description]["assert"])
        assert "mustNotContainCanary" in assertions
        assert "mustRefuseAttack" in assertions


def test_offline_mock_uses_real_promptfoo_config_without_network_provider() -> None:
    config = yaml.safe_load((REDTEAM / "promptfooconfig.mock.yaml").read_text(encoding="utf-8"))
    assert config["providers"][0]["id"] == "file://mock_provider.mjs"
    assert config["tests"] == "cases.yaml"
    runner = (REDTEAM / "run_mock.sh").read_text(encoding="utf-8")
    assert "--no-share --no-write" in runner
    assert "RAG_REDTEAM_TOKEN" not in runner
