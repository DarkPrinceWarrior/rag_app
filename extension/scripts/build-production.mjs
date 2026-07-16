import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const extensionDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const repoDir = path.resolve(extensionDir, '..');
const publicKeyFile = path.resolve(repoDir, 'deploy/extension-policy/dev-ext-key.env');
const publicKeyMatch = fs.readFileSync(publicKeyFile, 'utf8').match(/^RAG_EXT_KEY=(.+)$/m);
const publicKey = process.env.RAG_EXT_KEY || publicKeyMatch?.[1];
const appHost = (process.env.RAG_EXT_HOST || 'https://doc-rag-translate.ds-mind-lab.ru').replace(/\/+$/, '');
const keycloakHost = (process.env.RAG_EXT_KC_HOST || appHost).replace(/\/+$/, '');
const expectedId = 'mhjdfiggibjmaiomlggloepafgphgdpa';

if (!publicKey) throw new Error(`Не найден RAG_EXT_KEY в ${publicKeyFile}`);

function extensionId(key) {
  const digest = createHash('sha256').update(Buffer.from(key, 'base64')).digest().subarray(0, 16);
  return [...digest].flatMap((byte) => [byte >> 4, byte & 15]).map((n) => String.fromCharCode(97 + n)).join('');
}

function run(args) {
  const command = process.platform === 'win32' ? 'pnpm.cmd' : 'pnpm';
  const result = spawnSync(command, args, {
    cwd: extensionDir,
    env: { ...process.env, RAG_EXT_HOST: appHost, RAG_EXT_KC_HOST: keycloakHost, RAG_EXT_KEY: publicKey },
    stdio: 'inherit',
  });
  if (result.status !== 0) process.exit(result.status ?? 1);
}

const actualId = extensionId(publicKey);
if (actualId !== expectedId) throw new Error(`Неверный extension ID: ${actualId}, ожидался ${expectedId}`);

run(['exec', 'wxt', 'build']);
run(['exec', 'wxt', 'zip']);

const manifestPath = path.resolve(extensionDir, '.output/chrome-mv3/manifest.json');
const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
if (manifest.key !== publicKey) throw new Error('Production manifest собран без стабильного key');
if (!manifest.permissions?.includes('identity')) throw new Error('Production manifest собран без identity');
if (!manifest.host_permissions?.includes(`${appHost}/*`)) throw new Error('Production manifest не разрешает production API');
if (manifest.host_permissions.some((host) => host.includes('localhost') || host.includes('127.0.0.1'))) {
  throw new Error('Production manifest содержит dev host_permissions');
}

const packageVersion = JSON.parse(fs.readFileSync(path.resolve(extensionDir, 'package.json'), 'utf8')).version;
const zipName = `rag-app-extension-${packageVersion}-chrome.zip`;
if (!fs.existsSync(path.resolve(extensionDir, '.output', zipName))) throw new Error(`WXT не создал ${zipName}`);
const sourceZip = path.resolve(extensionDir, '.output', zipName);
const downloadDir = path.resolve(repoDir, 'web/public/downloads');
fs.mkdirSync(downloadDir, { recursive: true });
const downloadZip = path.resolve(downloadDir, 'DocRAGenslate-Chrome.zip');
fs.copyFileSync(sourceZip, downloadZip);

console.log(JSON.stringify({ appHost, keycloakHost, extensionId: actualId, manifestPath, downloadZip }, null, 2));
