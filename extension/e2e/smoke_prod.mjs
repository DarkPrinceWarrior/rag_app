import { chromium } from 'playwright';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const extensionDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const extensionPath = path.resolve(extensionDir, '.output/chrome-mv3');
const manifest = JSON.parse(fs.readFileSync(path.join(extensionPath, 'manifest.json'), 'utf8'));
const expectedId = 'mhjdfiggibjmaiomlggloepafgphgdpa';
const expectedApi = 'https://doc-rag-translate.ds-mind-lab.ru';
const playwrightCache = path.join(os.homedir(), '.cache/ms-playwright');
const cachedChromium = fs.existsSync(playwrightCache)
  ? fs.readdirSync(playwrightCache)
      .filter((name) => /^chromium-\d+$/.test(name))
      .sort((left, right) => Number(right.split('-')[1]) - Number(left.split('-')[1]))
      .map((name) => path.join(playwrightCache, name, 'chrome-linux64/chrome'))
      .find((candidate) => fs.existsSync(candidate))
  : undefined;
const browserExecutable = process.env.CHROME_BIN || cachedChromium || '/usr/bin/google-chrome';
const testServer = http.createServer((_request, response) => {
  response.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
  response.end('<!doctype html><p id="sample">Attention is all you need for this engineering document.</p>');
});
await new Promise((resolve) => testServer.listen(0, '127.0.0.1', resolve));
const serverAddress = testServer.address();
if (!serverAddress || typeof serverAddress === 'string') throw new Error('Не удалось определить порт тестового сервера');
const testUrl = `http://127.0.0.1:${serverAddress.port}/`;
let context;

try {
  context = await chromium.launchPersistentContext('', {
    headless: false,
    executablePath: browserExecutable,
    args: ['--headless=new', `--disable-extensions-except=${extensionPath}`, `--load-extension=${extensionPath}`],
  });
  let [worker] = context.serviceWorkers();
  if (!worker) worker = await context.waitForEvent('serviceworker', { timeout: 15000 });
  const runtimeId = await worker.evaluate(() => chrome.runtime.id);
  if (runtimeId !== expectedId) throw new Error(`ID ${runtimeId}, ожидался ${expectedId}`);

  const popup = await context.newPage();
  await popup.goto(`chrome-extension://${runtimeId}/popup.html`);
  await worker.evaluate(() => chrome.storage.sync.set({ apiBase: 'http://localhost:8100' }));
  await popup.reload();
  const apiBase = await popup.locator('#api-base').inputValue();
  if (apiBase !== expectedApi) throw new Error(`API ${apiBase}, ожидался ${expectedApi}`);
  const login = popup.getByRole('button', { name: 'Войти' });
  await login.waitFor({ state: 'visible', timeout: 15000 });

  const config = await worker.evaluate(async (base) => {
    const response = await fetch(`${base}/api/config`);
    return { status: response.status, body: await response.json() };
  }, expectedApi);
  if (config.status !== 200 || !config.body.auth_enabled) throw new Error(`Неверный /api/config: ${JSON.stringify(config)}`);

  const startedAt = Date.now();
  const authFailure = await popup.evaluate(() =>
    chrome.runtime.sendMessage({ type: 'selection', text: 'Attention is all you need.' }),
  );
  if (!authFailure?.error?.includes('Нужен вход')) {
    throw new Error(`Runtime message не вернул ожидаемую ошибку входа: ${JSON.stringify(authFailure)}`);
  }
  const authFailureMs = Date.now() - startedAt;
  if (authFailureMs > 10_000) throw new Error(`Ошибка авторизации возвращалась слишком долго: ${authFailureMs} мс`);

  const page = await context.newPage();
  await page.goto(testUrl);
  await page.locator('#rag-app-widget-host').waitFor({ state: 'attached' });
  await page.locator('#sample').selectText();
  await page.dispatchEvent('#sample', 'mouseup');
  await page.waitForTimeout(100);
  const selectionRect = await page.evaluate(() => {
    const range = window.getSelection()?.getRangeAt(0);
    if (!range) throw new Error('Тест не создал выделение');
    const rect = range.getBoundingClientRect();
    return { x: rect.left + rect.width / 2, y: rect.bottom + 22 };
  });
  await page.mouse.click(selectionRect.x, selectionRect.y);
  await page.waitForFunction(
    () => document.querySelector('#rag-app-widget-host')?.getAttribute('data-rag-status') === 'error',
    undefined,
    { timeout: 10_000 },
  );

  let observedAuthorization = '';
  await context.route(`${expectedApi}/api/selection/translate`, (route) => {
    observedAuthorization = route.request().headers().authorization ?? '';
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ text: 'Внимание — это всё, что вам нужно.', engine: 'hy-mt2-7b', ms: 250 }),
    });
  });
  await worker.evaluate(() =>
    chrome.storage.local.set({
      oidc_tokens: { access_token: 'smoke-token', refresh_token: 'smoke-refresh', exp: Date.now() + 60_000 },
    }),
  );
  await page.locator('#sample').selectText();
  await page.dispatchEvent('#sample', 'mouseup');
  await page.waitForTimeout(100);
  await page.mouse.click(selectionRect.x, selectionRect.y);
  await page.waitForFunction(
    () => document.querySelector('#rag-app-widget-host')?.getAttribute('data-rag-status') === 'done',
    undefined,
    { timeout: 10_000 },
  );
  if (observedAuthorization !== 'Bearer smoke-token') {
    throw new Error(`Неверный Authorization в запросе перевода: ${observedAuthorization || '<пусто>'}`);
  }

  console.log(
    JSON.stringify(
      {
        runtimeId,
        apiBase,
        authEnabled: config.body.auth_enabled,
        loginVisible: true,
        authFailureMs,
        selectionErrorVisible: true,
        selectionSuccessVisible: true,
      },
      null,
      2,
    ),
  );
} finally {
  await context?.close();
  await new Promise((resolve) => testServer.close(resolve));
}
