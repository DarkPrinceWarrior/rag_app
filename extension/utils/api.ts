// Общий клиент API (фоновый SW — единственный, кто ходит в сеть).
import { browser } from 'wxt/browser';
import { clearAuthTokens, getAccessToken } from '@/utils/auth';
import { fetchWithTimeout } from '@/utils/network';

/** Бросается, когда бэкенд требует вход (auth включён, токена нет/протух). */
export class AuthRequiredError extends Error {
  constructor() {
    super('Нужен вход: откройте попап расширения и нажмите «Войти»');
    this.name = 'AuthRequiredError';
  }
}

export interface NodeItem {
  id: string;
  text: string;
}

export interface HistoryEntry {
  source: string;
  translated: string;
  engine: string;
  ts: number;
}

export const DEFAULT_API_BASE = __RAG_EXT_DEFAULT_API__;
const LEGACY_LOCAL_BASES = new Set(['http://localhost:8100', 'http://127.0.0.1:8100']);

export async function getApiBase(): Promise<string> {
  const { apiBase } = await browser.storage.sync.get({ apiBase: DEFAULT_API_BASE });
  const normalized =
    typeof apiBase === 'string' && apiBase.trim() ? apiBase.trim().replace(/\/+$/, '') : DEFAULT_API_BASE;
  if (DEFAULT_API_BASE.startsWith('https://') && LEGACY_LOCAL_BASES.has(normalized)) {
    await browser.storage.sync.set({ apiBase: DEFAULT_API_BASE });
    return DEFAULT_API_BASE;
  }
  return normalized;
}

class HttpResponseError extends Error {
  constructor(readonly status: number, detail: string) {
    super(`${status}: ${detail}`);
    this.name = 'HttpResponseError';
  }
}

async function postJson<T>(path: string, body: unknown, retries = 0): Promise<T> {
  const base = await getApiBase();
  let lastErr: unknown;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const token = await getAccessToken(false);
      const resp = await fetchWithTimeout(`${base}${path}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify(body),
      });
      if (resp.status === 401) {
        await clearAuthTokens();
        throw new AuthRequiredError(); // не ретраим — нужен новый вход
      }
      if (!resp.ok) {
        const detail = await resp.json().then((d) => d.detail).catch(() => resp.statusText);
        throw new HttpResponseError(resp.status, detail);
      }
      return (await resp.json()) as T;
    } catch (e) {
      if (e instanceof AuthRequiredError) throw e;
      if (e instanceof HttpResponseError && e.status < 500) throw e;
      lastErr = e;
      if (attempt < retries) await new Promise((r) => setTimeout(r, 500 * (attempt + 1)));
    }
  }
  throw lastErr;
}

export function translateSelection(text: string) {
  return postJson<{ text: string; engine: string; ms: number }>('/api/selection/translate', {
    text,
    target_lang: 'ru',
  });
}

export function translateNodes(items: NodeItem[]) {
  return postJson<{ items: NodeItem[]; engine: string; ms: number }>('/api/web/translate', {
    items,
    target_lang: 'ru',
  });
}

export async function pushHistory(entry: HistoryEntry): Promise<void> {
  const { history } = await browser.storage.local.get({ history: [] as HistoryEntry[] });
  const next = [entry, ...(history as HistoryEntry[])].slice(0, 20);
  await browser.storage.local.set({ history: next });
}
