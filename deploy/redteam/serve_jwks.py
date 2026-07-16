"""Minimal loopback-only JWKS server for the disposable red-team API."""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class JwksHandler(BaseHTTPRequestHandler):
    jwks: bytes = b""

    def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
        if self.path != "/jwks.json":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(self.jwks)))
        self.end_headers()
        self.wfile.write(self.jwks)

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--jwks", required=True, type=Path)
    args = parser.parse_args()
    payload = json.loads(args.jwks.read_text(encoding="utf-8"))
    if not isinstance(payload.get("keys"), list) or len(payload["keys"]) != 1:
        raise SystemExit("JWKS must contain exactly one key")
    JwksHandler.jwks = json.dumps(payload, separators=(",", ":")).encode()
    ThreadingHTTPServer(("127.0.0.1", args.port), JwksHandler).serve_forever()


if __name__ == "__main__":
    main()
