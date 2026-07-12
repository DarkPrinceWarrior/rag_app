from __future__ import annotations

from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from prometheus_fastapi_instrumentator import Instrumentator


def test_instrumentator_handles_included_router() -> None:
    router = APIRouter()

    @router.get("/probe")
    async def probe() -> dict[str, bool]:
        return {"ok": True}

    app = FastAPI()
    app.include_router(router, prefix="/api")
    Instrumentator().instrument(app)

    response = TestClient(app, raise_server_exceptions=False).get("/api/probe")

    assert response.status_code == 200
    assert response.json() == {"ok": True}
