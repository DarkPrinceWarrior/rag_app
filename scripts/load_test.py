"""Нагрузочный тест этапа 5: N одновременных документов (roadmap § 11: N=20).

Заливает N копий тестового PDF одновременно, ждёт done/error всех,
печатает тайминги и горлышки. Запуск (на сервере):
  uv run python scripts/load_test.py [N=20] [страниц=10] [api=http://127.0.0.1:8100]
"""

from __future__ import annotations

import asyncio
import io
import statistics
import sys
import time

import httpx

sys.path.insert(0, "scripts")
from make_test_pdf import build_story_pdf  # noqa: E402


async def upload_and_wait(client: httpx.AsyncClient, api: str, pdf: bytes, i: int) -> dict:
    t0 = time.monotonic()
    resp = await client.post(
        f"{api}/api/documents",
        files={"file": (f"load_{i:02d}.pdf", io.BytesIO(pdf), "application/pdf")},
    )
    resp.raise_for_status()
    doc_id = resp.json()["id"]
    status = "uploaded"
    while status not in ("done", "error"):
        await asyncio.sleep(5)
        status = (await client.get(f"{api}/api/documents/{doc_id}")).json()["status"]
    return {"i": i, "id": doc_id, "status": status, "sec": time.monotonic() - t0}


async def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    pages = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    api = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8100"

    pdf = build_story_pdf(pages)
    print(f"PDF: {pages} стр., {len(pdf)} байт; документов: {n}")
    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=120.0) as client:
        results = await asyncio.gather(
            *(upload_and_wait(client, api, pdf, i) for i in range(n))
        )
    wall = time.monotonic() - t0

    ok = [r for r in results if r["status"] == "done"]
    failed = [r for r in results if r["status"] != "done"]
    times = sorted(r["sec"] for r in ok)
    print(f"\nИТОГО: {len(ok)}/{n} done за {wall:.0f} с (стена)")
    if times:
        print(
            f"per-doc: min {times[0]:.0f} с · медиана {statistics.median(times):.0f} с · "
            f"p90 {times[int(len(times) * 0.9) - 1]:.0f} с · max {times[-1]:.0f} с"
        )
    for r in failed:
        print(f"  FAIL {r['i']}: {r['id']}")


if __name__ == "__main__":
    asyncio.run(main())
