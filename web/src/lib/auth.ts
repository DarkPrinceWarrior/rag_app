// OIDC PKCE-вход через Keycloak без внешних библиотек (порт static/auth.js,
// roadmap § 9: on-prem, без CDN). Конфиг (authority/client_id) приходит с
// бэкенда (/api/config). Токены в sessionStorage; authFetch добавляет Bearer
// к /api и на 401 переинициирует логин.

interface AppConfig {
  auth_enabled: boolean
  oidc_authority: string
  oidc_client_id: string
}

interface Tokens {
  access_token: string
  refresh_token?: string
  exp: number
}

const TOKENS_KEY = 'rag_tokens'
const VERIFIER_KEY = 'rag_pkce_verifier'
let cfg: AppConfig | null = null

const b64url = (buf: ArrayBuffer): string =>
  btoa(String.fromCharCode(...new Uint8Array(buf)))
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=+$/, '')

async function sha256(text: string): Promise<string> {
  return b64url(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text)))
}

const randomString = (): string => b64url(crypto.getRandomValues(new Uint8Array(32)).buffer)

const loadTokens = (): Tokens | null => JSON.parse(sessionStorage.getItem(TOKENS_KEY) || 'null')
const saveTokens = (t: Tokens) => sessionStorage.setItem(TOKENS_KEY, JSON.stringify(t))

async function login(): Promise<void> {
  const verifier = randomString()
  sessionStorage.setItem(VERIFIER_KEY, verifier)
  const challenge = await sha256(verifier)
  const params = new URLSearchParams({
    client_id: cfg!.oidc_client_id,
    response_type: 'code',
    scope: 'openid',
    redirect_uri: location.origin + location.pathname,
    code_challenge: challenge,
    code_challenge_method: 'S256',
  })
  location.assign(`${cfg!.oidc_authority}/protocol/openid-connect/auth?${params}`)
}

async function tokenRequest(body: Record<string, string>): Promise<void> {
  const resp = await fetch(`${cfg!.oidc_authority}/protocol/openid-connect/token`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams(body),
  })
  if (!resp.ok) throw new Error(`token endpoint: ${resp.status}`)
  const data = await resp.json()
  saveTokens({
    access_token: data.access_token,
    refresh_token: data.refresh_token,
    exp: Date.now() + (data.expires_in - 30) * 1000,
  })
}

async function exchangeCode(): Promise<void> {
  const url = new URL(location.href)
  const code = url.searchParams.get('code')
  if (!code) return
  await tokenRequest({
    grant_type: 'authorization_code',
    client_id: cfg!.oidc_client_id,
    code,
    redirect_uri: location.origin + location.pathname,
    code_verifier: sessionStorage.getItem(VERIFIER_KEY) || '',
  })
  for (const k of ['code', 'session_state', 'iss']) url.searchParams.delete(k)
  history.replaceState(null, '', url.toString())
}

async function ensureToken(): Promise<string | null> {
  if (!cfg?.auth_enabled) return null
  const tokens = loadTokens()
  if (tokens && Date.now() < tokens.exp) return tokens.access_token
  if (tokens?.refresh_token) {
    try {
      await tokenRequest({
        grant_type: 'refresh_token',
        client_id: cfg.oidc_client_id,
        refresh_token: tokens.refresh_token,
      })
      return loadTokens()!.access_token
    } catch {
      sessionStorage.removeItem(TOKENS_KEY)
    }
  }
  await login() // редирект; код дальше не выполнится
  return null
}

/** Блокирующая инициализация до рендера приложения. */
export async function initAuth(): Promise<void> {
  cfg = await (await fetch('/api/config')).json()
  if (!cfg!.auth_enabled) return
  await exchangeCode()
  await ensureToken()
}

export function logout(): void {
  sessionStorage.removeItem(TOKENS_KEY)
  void login()
}

export interface CurrentUser {
  username: string
  roles: string[]
  isAdmin: boolean
}

/** Текущий пользователь из access-токена (preferred_username + realm-роли).
 * Без auth (dev) — встроенный local-dev/admin. */
export function currentUser(): CurrentUser {
  const t = loadTokens()
  if (!t?.access_token) return { username: 'local-dev', roles: ['admin'], isAdmin: true }
  try {
    const payload = JSON.parse(
      decodeURIComponent(
        atob(t.access_token.split('.')[1].replace(/-/g, '+').replace(/_/g, '/'))
          .split('')
          .map((c) => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2))
          .join(''),
      ),
    )
    const roles: string[] = (payload.realm_access?.roles ?? []).filter((r: string) =>
      ['user', 'admin'].includes(r),
    )
    return {
      username: payload.preferred_username ?? payload.sub ?? 'пользователь',
      roles,
      isAdmin: roles.includes('admin'),
    }
  } catch {
    return { username: 'пользователь', roles: [], isAdmin: false }
  }
}

/** fetch с авто-Bearer для /api; на 401 — повторный вход. */
export async function authFetch(input: string, init: RequestInit = {}): Promise<Response> {
  if (cfg?.auth_enabled && input.startsWith('/api')) {
    const token = await ensureToken()
    init.headers = { ...(init.headers || {}), Authorization: `Bearer ${token}` }
    const resp = await fetch(input, init)
    if (resp.status === 401) {
      sessionStorage.removeItem(TOKENS_KEY)
      await login()
    }
    return resp
  }
  return fetch(input, init)
}

export async function bearer(): Promise<string | null> {
  return ensureToken()
}
