import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const extensionDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const source = await fs.readFile(path.resolve(extensionDir, '../web/public/favicon.svg'), 'utf8');
const outputDir = path.resolve(extensionDir, 'public/icon');
await fs.mkdir(outputDir, { recursive: true });

const browser = await chromium.launch({
  headless: true,
  executablePath: process.env.CHROME_BIN || '/usr/bin/google-chrome',
});
try {
  for (const size of [16, 32, 48, 128]) {
    const page = await browser.newPage({ viewport: { width: size, height: size }, deviceScaleFactor: 1 });
    await page.setContent(`<style>*{box-sizing:border-box}html,body{margin:0;width:100%;height:100%}svg{width:100%;height:100%;display:block}</style>${source}`);
    await page.screenshot({ path: path.join(outputDir, `${size}.png`), omitBackground: true });
    await page.close();
  }
} finally {
  await browser.close();
}

console.log(`Иконки расширения обновлены: ${outputDir}`);
