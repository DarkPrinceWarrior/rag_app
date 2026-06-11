// Смоук собранного расширения: выделение → кнопка → тултип перевода.
// Требует: собранный .output/chrome-mv3 и доступный API на localhost:8100.
// Запуск: node e2e/smoke.mjs
import { chromium } from 'playwright';
import { fileURLToPath } from 'node:url';
import http from 'node:http';
import path from 'node:path';
import fs from 'node:fs';

const root = path.dirname(fileURLToPath(import.meta.url));
const extPath = path.resolve(root, '../.output/chrome-mv3');
const outDir = path.resolve(root, 'out');
fs.mkdirSync(outDir, { recursive: true });

const TEST_HTML = `<!doctype html><html><body style="font: 16px sans-serif; padding: 40px; max-width: 700px">
  <h1>Pressure Vessel Specification</h1>
  <p id="p1">The maximum allowable working pressure shall not exceed 16.5 MPa at a design temperature of 120 degrees.</p>
  <p id="p2">All welded joints shall be subject to radiographic examination.</p>
</body></html>`;

// content script не работает на data:/file: URL — поднимаем локальный http
const server = http.createServer((_req, res) => {
  res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
  res.end(TEST_HTML);
});
await new Promise((r) => server.listen(8123, '127.0.0.1', r));

const ctx = await chromium.launchPersistentContext('', {
  // старый headless не грузит расширения — нужен новый (--headless=new)
  headless: false,
  args: ['--headless=new', `--disable-extensions-except=${extPath}`, `--load-extension=${extPath}`],
});
try {
  const page = await ctx.newPage();
  await page.goto('http://127.0.0.1:8123/');
  await page.waitForTimeout(700); // content script инициализируется

  // выделяем абзац тройным кликом (selection + mouseup)
  await page.click('#p1', { clickCount: 3 });
  await page.waitForTimeout(400);
  await page.screenshot({ path: path.join(outDir, '1-button.png') });

  // кнопка «Перевести» — в closed shadow DOM: кликаем по координатам под выделением
  const rect = await page.evaluate(() => {
    const r = window.getSelection().getRangeAt(0).getBoundingClientRect();
    return { x: r.left + r.width / 2 - 36 + 30, y: r.bottom + 8 + 12 };
  });
  await page.mouse.click(rect.x, rect.y);

  // ждём тултип с переводом (кириллица появляется в скриншоте)
  await page.waitForTimeout(4000);
  await page.screenshot({ path: path.join(outDir, '2-tooltip.png') });

  // полностраничный перевод: команда content script'у через SW расширения
  const [sw] = ctx.serviceWorkers();
  await sw.evaluate(async () => {
    const [tab] = await chrome.tabs.query({ active: true });
    await chrome.tabs.sendMessage(tab.id, { type: 'translate-page' });
  });
  await page.waitForTimeout(1500);
  await page.screenshot({ path: path.join(outDir, '3-page-translated.png') });
  console.log('OK: скриншоты в', outDir);
} finally {
  await ctx.close();
  server.close();
}
