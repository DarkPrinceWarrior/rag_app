// OIDC PKCE через chrome.identity.launchWebAuthFlow → Keycloak (roadmap § 9, этап 5).
// Адрес Keycloak и client_id НЕ зашиты в расширение — приходят с бэкенда
// (/api/config), как и в веб-приложении (static/auth.js). Токены живут в
// storage.local; refresh_token продлевает доступ без повторного окна логина.
import { browser } from 'wxt/browser';
import { getApiBase } from '@/utils/api';

export interface OidcConfig {
  auth_enabled: boolean;
  oidc_authority: string;
  oidc_client_id: string;
}

interface Tokens {
  access_token: string;
  refresh_token?: string;
  exp: number; // ms-epoch с запасом 30 c до фактического истечения
}

const TOKENS_KEY = 'oidc_tokens';
let cfgCache: { base: string; cfg: OidcConfig } | null = null;

const b64url = (buf: ArrayBuffer): string =>
  btoa(String.fromCharCode(...new Uint8Array(buf)))
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=+$/, '');

async function sha256(text: string): Promise<string> {
  return b64url(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text)));
}

const randomString = (): string => b64url(crypto.getRandomValues(new Uint8Array(32)).buffer);

export async function getOidcConfig(): Promise<OidcConfig> {
  const base = await getApiBase();
  if (cfgCache && cfgCache.base === base) return cfgCache.cfg;
  const resp = await fetch(`${base}/api/config`);
  const cfg = (await resp.json()) as OidcConfig;
  cfgCache = { base, cfg };
  return cfg;
}

async function loadTokens(): Promise<Tokens | null> {
  const stored = await browser.storage.local.get(TOKENS_KEY);
  return (stored[TOKENS_KEY] as Tokens) ?? null;
}

async function saveTokens(tokens: Tokens | null): Promise<void> {
  if (tokens) await browser.storage.local.set({ [TOKENS_KEY]: tokens });
  else await browser.storage.local.remove(TOKENS_KEY);
}

async function tokenRequest(cfg: OidcConfig, body: Record<string, string>): Promise<Tokens> {
  const resp = await fetch(`${cfg.oidc_authority}/protocol/openid-connect/token`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams(body),
  });
  if (!resp.ok) throw new Error(`OIDC token endpoint: ${resp.status}`);
  const data = await resp.json();
  const tokens: Tokens = {
    access_token: data.access_token,
    refresh_token: data.refresh_token,
    exp: Date.now() + (data.expires_in - 30) * 1000,
  };
  await saveTokens(tokens);
  return tokens;
}

async function interactiveLogin(cfg: OidcConfig): Promise<Tokens> {
  const verifier = randomString();
  const challenge = await sha256(verifier);
  // https://<extension-id>.chromiumapp.org/ — покрыт redirect-маской realm'а
  const redirectUri = browser.identity.getRedirectURL();
  const authUrl =
    `${cfg.oidc_authority}/protocol/openid-connect/auth?` +
    new URLSearchParams({
      client_id: cfg.oidc_client_id,
      response_type: 'code',
      scope: 'openid',
      redirect_uri: redirectUri,
      code_challenge: challenge,
      code_challenge_method: 'S256',
    });
  const redirect = await browser.identity.launchWebAuthFlow({ url: authUrl, interactive: true });
  if (!redirect) throw new Error('OIDC: окно входа закрыто без ответа');
  const code = new URL(redirect).searchParams.get('code');
  if (!code) throw new Error('OIDC: код авторизации не получен');
  return tokenRequest(cfg, {
    grant_type: 'authorization_code',
    client_id: cfg.oidc_client_id,
    code,
    redirect_uri: redirectUri,
    code_verifier: verifier,
  });
}

/** Токен для запроса. interactive=false — тихо (валидный/refresh, иначе null);
 *  interactive=true — при отсутствии откроет окно логина Keycloak. */
export async function getAccessToken(interactive = false): Promise<string | null> {
  const cfg = await getOidcConfig();
  if (!cfg.auth_enabled) return null;
  const tokens = await loadTokens();
  if (tokens && Date.now() < tokens.exp) return tokens.access_token;
  if (tokens?.refresh_token) {
    try {
      const refreshed = await tokenRequest(cfg, {
        grant_type: 'refresh_token',
        client_id: cfg.oidc_client_id,
        refresh_token: tokens.refresh_token,
      });
      return refreshed.access_token;
    } catch {
      await saveTokens(null);
    }
  }
  if (!interactive) return null;
  return (await interactiveLogin(cfg)).access_token;
}

export async function login(): Promise<void> {
  await interactiveLogin(await getOidcConfig());
}

export async function logout(): Promise<void> {
  await saveTokens(null);
}

export async function authStatus(): Promise<{ enabled: boolean; loggedIn: boolean }> {
  const cfg = await getOidcConfig();
  if (!cfg.auth_enabled) return { enabled: false, loggedIn: false };
  return { enabled: true, loggedIn: Boolean(await loadTokens()) };
}
