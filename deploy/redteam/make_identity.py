"""Create ephemeral RSA identity material for the disposable stand."""

from __future__ import annotations

import argparse
import base64
import json
import secrets
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa


def _b64uint(value: int) -> str:
    raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-key", required=True, type=Path)
    parser.add_argument("--jwks", required=True, type=Path)
    args = parser.parse_args()
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    kid = f"redteam-{secrets.token_hex(8)}"
    args.private_key.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    args.private_key.chmod(0o600)
    numbers = key.public_key().public_numbers()
    args.jwks.write_text(
        json.dumps(
            {
                "keys": [
                    {
                        "kty": "RSA",
                        "kid": kid,
                        "use": "sig",
                        "alg": "RS256",
                        "n": _b64uint(numbers.n),
                        "e": _b64uint(numbers.e),
                    }
                ]
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    args.jwks.chmod(0o600)
    print(kid)


if __name__ == "__main__":
    main()
