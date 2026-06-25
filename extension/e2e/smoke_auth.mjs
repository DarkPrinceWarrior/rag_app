// Смоук собранного расширения С АВТОРИЗАЦИЕЙ: впрыскиваем валидный токен в
// storage расширения (headless-вход через Keycloak невозможен), затем
// проверяем перевод выделения (тултип) и перевод всей страницы (замена текста).
// Запуск: ACCESS=... REFRESH=... EXPIRES=1800 node e2e/smoke_auth.mjs
import { chromium } from 'playwright';
import { fileURLToPath } from 'node:url';
import http from 'node:http';
import path from 'node:path';
import fs from 'node:fs';

const root = path.dirname(fileURLToPath(import.meta.url));
const extPath = path.resolve(root, '../.output/chrome-mv3');
const outDir = path.resolve(root, 'out');
fs.mkdirSync(outDir, { recursive: true });

const TEST_HTML = `<!doctype html><html><head><meta charset="utf-8"></head>
<body style="font: 16px sans-serif; padding: 40px; max-width: 700px">
  <h1>Pressure Vessel Specification</h1>
  <p id="p1">The maximum allowable working pressure shall not exceed 16.5 MPa at a design temperature of 120 degrees.</p>
  <p id="p2">All welded joints shall be subject to radiographic examination before commissioning.</p>
</body></html>`;

const CYR = /[А-Яа-яЁё]/;
const token = {
  access_token: process.env.ACCESS,
  refresh_token: process.env.REFRESH,
  exp: Date.now() + (Number(process.env.EXPIRES || 1800) - 30) * 1000,
};

const server = http.createServer((_req, res) => {
  res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
  res.end(TEST_HTML);
});
await new Promise((r) => server.listen(8123, '127.0.0.1', r));

const ctx = await chromium.launchPersistentContext('', {
  headless: false,
  args: ['--headless=new', `--disable-extensions-except=${extPath}`, `--load-extension=${extPath}`],
});

const result = { config: null, token_injected: false, selection: null, page_translate: null, errors: [] };
try {
  // дождаться service worker расширения
  let [sw] = ctx.serviceWorkers();
  if (!sw) sw = await ctx.waitForEvent('serviceworker', { timeout: 15000 });

  // 1) задать API-базу и впрыснуть токен в storage расширения
  await sw.evaluate(async (tok) => {
    await chrome.storage.sync.set({ apiBase: 'http://localhost:8100' });
    await chrome.storage.local.set({ oidc_tokens: tok });
  }, token);
  result.token_injected = true;

  // проверим, что расширение видит конфиг бэкенда (auth_enabled и т.п.)
  result.config = await sw.evaluate(async () => {
    const r = await fetch('http://localhost:8100/api/config');
    return await r.json();
  });

  const page = await ctx.newPage();
  page.on('console', (m) => { if (m.type() === 'error') result.errors.push('page: ' + m.text()); });
  await page.goto('http://127.0.0.1:8123/');
  await page.waitForTimeout(900);

  // 2) перевод выделения: тройной клик по абзацу → кнопка под выделением → тултип
  await page.click('#p1', { clickCount: 3 });
  await page.waitForTimeout(400);
  await page.screenshot({ path: path.join(outDir, 'a1-selection-button.png') });
  const rect = await page.evaluate(() => {
    const r = window.getSelection().getRangeAt(0).getBoundingClientRect();
    return { x: r.left + r.width / 2 - 36 + 30, y: r.bottom + 8 + 12 };
  });
  await page.mouse.click(rect.x, rect.y);
  await page.waitForTimeout(5000);
  await page.screenshot({ path: path.join(outDir, 'a2-selection-tooltip.png') });
  result.selection = 'screenshot a2-selection-tooltip.png (тултип в closed shadow DOM)';

  // 3) перевод всей страницы через SW → текст #p1/#p2 заменяется на месте
  const before = await page.evaluate(() => ({
    p1: document.getElementById('p1').textContent,
    p2: document.getElementById('p2').textContent,
  }));
  await sw.evaluate(async () => {
    const [tab] = await chrome.tabs.query({ active: true });
    await chrome.tabs.sendMessage(tab.id, { type: 'translate-page' });
  });
  // ждём, пока появится кириллица (до 25 c)
  let after = before;
  for (let i = 0; i < 25; i++) {
    await page.waitForTimeout(1000);
    after = await page.evaluate(() => ({
      p1: document.getElementById('p1').textContent,
      p2: document.getElementById('p2').textContent,
    }));
    if (CYR.test(after.p1) || CYR.test(after.p2)) break;
  }
  await page.screenshot({ path: path.join(outDir, 'a3-page-translated.png') });
  result.page_translate = {
    before, after,
    p1_translated: CYR.test(after.p1) && after.p1 !== before.p1,
    p2_translated: CYR.test(after.p2) && after.p2 !== before.p2,
  };

  // 4) попап расширения: статус входа, кнопки, история переводов
  const extId = process.env.EXT_ID || '';
  if (extId) {
    const pop = await ctx.newPage();
    await pop.setViewportSize({ width: 400, height: 600 });
    await pop.goto(`chrome-extension://${extId}/popup.html`);
    await pop.waitForTimeout(2500);
    await pop.screenshot({ path: path.join(outDir, 'a4-popup.png') });
    result.popup = 'screenshot a4-popup.png';
  }
} catch (e) {
  result.errors.push('fatal: ' + (e && e.message ? e.message : String(e)));
} finally {
  await ctx.close();
  server.close();
}
console.log(JSON.stringify(result, null, 2));
