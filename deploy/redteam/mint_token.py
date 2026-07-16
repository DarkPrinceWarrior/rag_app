"""Mint a short-lived user-A JWT from disposable stand key material."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jwt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-key", required=True, type=Path)
    parser.add_argument("--issuer", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--kid", required=True)
    parser.add_argument("--ttl", default=600, type=int)
    args = parser.parse_args()
    if not 60 <= args.ttl <= 900:
        raise SystemExit("token TTL must be between 60 and 900 seconds")
    now = int(time.time())
    token = jwt.encode(
        {
            "iss": args.issuer,
            "sub": args.subject,
            "preferred_username": "redteam-owner-a",
            "azp": "rag-web",
            "realm_access": {"roles": ["user"]},
            "iat": now,
            "exp": now + args.ttl,
        },
        args.private_key.read_bytes(),
        algorithm="RS256",
        headers={"kid": args.kid},
    )
    print(token)


if __name__ == "__main__":
    main()
