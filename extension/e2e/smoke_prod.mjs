import { chromium } from 'playwright';
import fs from 'node:fs';
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

const context = await chromium.launchPersistentContext('', {
  headless: false,
  executablePath: browserExecutable,
  args: ['--headless=new', `--disable-extensions-except=${extensionPath}`, `--load-extension=${extensionPath}`],
});

try {
  let [worker] = context.serviceWorkers();
  if (!worker) worker = await context.waitForEvent('serviceworker', { timeout: 15000 });
  const runtimeId = await worker.evaluate(() => chrome.runtime.id);
  if (runtimeId !== expectedId) throw new Error(`ID ${runtimeId}, ожидался ${expectedId}`);

  const popup = await context.newPage();
  await popup.goto(`chrome-extension://${runtimeId}/popup.html`);
  const apiBase = await popup.locator('#api-base').inputValue();
  if (apiBase !== expectedApi) throw new Error(`API ${apiBase}, ожидался ${expectedApi}`);
  const login = popup.getByRole('button', { name: 'Войти' });
  await login.waitFor({ state: 'visible', timeout: 15000 });

  const config = await worker.evaluate(async (base) => {
    const response = await fetch(`${base}/api/config`);
    return { status: response.status, body: await response.json() };
  }, expectedApi);
  if (config.status !== 200 || !config.body.auth_enabled) throw new Error(`Неверный /api/config: ${JSON.stringify(config)}`);

  console.log(JSON.stringify({ runtimeId, apiBase, authEnabled: config.body.auth_enabled, loginVisible: true }, null, 2));
} finally {
  await context.close();
}
