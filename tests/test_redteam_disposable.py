"""Contract tests for the disposable real red-team stand (no sockets/containers)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import jwt
import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
REDTEAM = ROOT / "deploy" / "redteam"


def _run(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=REDTEAM,
        env=env,
        capture_output=True,
        check=False,
        text=True,
    )


def test_disposable_compose_has_only_isolated_infrastructure() -> None:
    compose = yaml.safe_load((REDTEAM / "docker-compose.disposable.yml").read_text(encoding="utf-8"))
    assert set(compose["services"]) == {"postgres", "redis", "minio"}
    assert compose["name"] == "${RAG_REDTEAM_COMPOSE_PROJECT}"
    rendered = json.dumps(compose, sort_keys=True)
    for port in ("5432", "6379", "9000"):
        assert "127.0.0.1:${RAG_REDTEAM_" in rendered
        assert port in rendered
    assert "keycloak" not in rendered.lower()
    assert "rag_app" not in rendered
    assert all("@sha256:" in service["image"] for service in compose["services"].values())
    assert all(service["restart"] == "no" for service in compose["services"].values())


def test_disposable_scripts_are_syntactically_valid_and_fail_closed() -> None:
    scripts = (
        "setup_disposable.sh",
        "run_disposable.sh",
        "teardown_disposable.sh",
    )
    for script in scripts:
        syntax = _run("bash", "-n", script)
        assert syntax.returncode == 0, syntax.stderr
        refused = _run("bash", script, env={"PATH": os.environ["PATH"]})
        assert refused.returncode == 2
        assert "RAG_REDTEAM_CONFIRM_DISPOSABLE=YES" in refused.stderr


def test_setup_guards_production_and_reuses_only_read_only_model_endpoints() -> None:
    setup = (REDTEAM / "setup_disposable.sh").read_text(encoding="utf-8")
    teardown = (REDTEAM / "teardown_disposable.sh").read_text(encoding="utf-8")
    assert "{5433, 6379, 8100, 9000, 9001, 8180}" in setup
    assert "docragenslate_redteam_disposable" in setup
    assert "RAG_AUTH_ENABLED=true" in setup
    assert "RAG_VISUAL_ENABLED=false" in setup
    assert "RAG_VL_ENABLED=false" in setup
    assert 'curl --fail --silent --show-error --max-time 10 "$LLM_URL/models"' in setup
    assert 'curl --fail --silent --show-error --max-time 10 "$EMBED_URL/models"' in setup
    assert "down --volumes --remove-orphans" in teardown
    assert "docragenslate-disposable-redteam-v1" in teardown
    assert 'realpath -e "$STATE"' in teardown
    assert 'rm -rf --one-file-system -- "$STATE"' in teardown
    assert 'kill -0 "$pid" 2>/dev/null && kill -KILL "$pid"' not in teardown
    assert "return 0" in teardown
    assert 'rmdir -- "$STATE/evidence/logs"' in teardown
    assert teardown.index('rmdir -- "$STATE/evidence"') < teardown.index(
        'rm -rf --one-file-system -- "$STATE"'
    )
    assert 'for command in docker curl openssl "$PYTHON"' in setup
    assert setup.index("trap cleanup_on_error ERR INT TERM") < setup.index(
        'RUN_ID="$(openssl rand -hex 6)"'
    )


def test_seed_contract_covers_required_synthetic_slices_and_rls_probe() -> None:
    seed = (REDTEAM / "seed_disposable.py").read_text(encoding="utf-8")
    for fixture in ("injected", "filename", "zh", "poison", "numeric", "foreign"):
        assert f'"{fixture}"' in seed
    assert 'OWNER_A = "redteam-owner-a"' in seed
    assert 'OWNER_B = "redteam-owner-b"' in seed
    assert "RLS verification failed for synthetic foreign-owner fixture" in seed
    assert 'url.database != "docragenslate_redteam_disposable"' in seed
    assert 'url.host != "127.0.0.1"' in seed
    assert "url.port in {5432, 5433}" in seed
    assert "embedding endpoint returned an invalid vector" in seed
    assert "127.0.0.1:5433" not in seed


def test_ephemeral_identity_mints_short_lived_user_token(tmp_path: Path) -> None:
    private_key = tmp_path / "private.pem"
    jwks_path = tmp_path / "jwks.json"
    identity = _run(
        str(ROOT / ".venv" / "bin" / "python"),
        "make_identity.py",
        "--private-key",
        str(private_key),
        "--jwks",
        str(jwks_path),
    )
    assert identity.returncode == 0, identity.stderr
    kid = identity.stdout.strip()
    token_result = _run(
        str(ROOT / ".venv" / "bin" / "python"),
        "mint_token.py",
        "--private-key",
        str(private_key),
        "--issuer",
        "urn:docragenslate:redteam:test",
        "--subject",
        "redteam-owner-a",
        "--kid",
        kid,
        "--ttl",
        "60",
    )
    assert token_result.returncode == 0, token_result.stderr
    jwks = json.loads(jwks_path.read_text(encoding="utf-8"))
    jwk = jwt.PyJWK(jwks["keys"][0])
    claims = jwt.decode(
        token_result.stdout.strip(),
        jwk,
        algorithms=["RS256"],
        issuer="urn:docragenslate:redteam:test",
        options={"verify_aud": False},
    )
    assert claims["sub"] == "redteam-owner-a"
    assert claims["realm_access"]["roles"] == ["user"]
    assert claims["exp"] - claims["iat"] == 60


def test_run_wrapper_binds_manifest_to_existing_promptfoo_contract() -> None:
    runner = (REDTEAM / "run_disposable.sh").read_text(encoding="utf-8")
    for variable in (
        "RAG_REDTEAM_INJECTED_DOCUMENT_ID",
        "RAG_REDTEAM_FILENAME_DOCUMENT_ID",
        "RAG_REDTEAM_FOREIGN_CANARY",
        "RAG_REDTEAM_PRIVATE_CODENAME",
        "RAG_REDTEAM_ZH_DOCUMENT_ID",
        "RAG_REDTEAM_POISON_DOCUMENT_ID",
    ):
        assert variable in runner
    assert 'RAG_REDTEAM_BASE_URL="http://127.0.0.1:$RAG_REDTEAM_API_PORT"' in runner
    assert 'RAG_REDTEAM_API_PID="$api_pid"' in runner
    assert "--ttl 900" in runner
    assert '"$HERE/run_local.sh"' in runner
