// PKCE-вход через Keycloak без внешних библиотек (on-prem, без CDN — roadmap § 9).
// Подключается ПЕРВЫМ скриптом на каждой странице: перехватывает fetch к /api/*
// и добавляет Authorization. При RAG_AUTH_ENABLED=false ничего не делает.
(() => {
  const TOKENS_KEY = "rag_tokens";
  const VERIFIER_KEY = "rag_pkce_verifier";
  let cfg = null;

  const b64url = (buf) =>
    btoa(String.fromCharCode(...new Uint8Array(buf))).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");

  async function sha256(text) {
    return b64url(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text)));
  }

  const randomString = () => b64url(crypto.getRandomValues(new Uint8Array(32)));

  const loadTokens = () => JSON.parse(sessionStorage.getItem(TOKENS_KEY) || "null");
  const saveTokens = (t) => sessionStorage.setItem(TOKENS_KEY, JSON.stringify(t));

  async function login() {
    const verifier = randomString();
    sessionStorage.setItem(VERIFIER_KEY, verifier);
    const challenge = await sha256(verifier);
    const params = new URLSearchParams({
      client_id: cfg.oidc_client_id,
      response_type: "code",
      scope: "openid",
      redirect_uri: location.origin + location.pathname + location.search,
      code_challenge: challenge,
      code_challenge_method: "S256",
    });
    location.assign(`${cfg.oidc_authority}/protocol/openid-connect/auth?${params}`);
  }

  async function tokenRequest(body) {
    const resp = await fetch(`${cfg.oidc_authority}/protocol/openid-connect/token`, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams(body),
    });
    if (!resp.ok) throw new Error(`token endpoint: ${resp.status}`);
    const data = await resp.json();
    saveTokens({
      access_token: data.access_token,
      refresh_token: data.refresh_token,
      exp: Date.now() + (data.expires_in - 30) * 1000,
    });
  }

  async function exchangeCode() {
    const url = new URL(location.href);
    const code = url.searchParams.get("code");
    if (!code) return false;
    await tokenRequest({
      grant_type: "authorization_code",
      client_id: cfg.oidc_client_id,
      code,
      redirect_uri: location.origin + location.pathname,
      code_verifier: sessionStorage.getItem(VERIFIER_KEY) || "",
    });
    url.searchParams.delete("code");
    url.searchParams.delete("session_state");
    url.searchParams.delete("iss");
    history.replaceState(null, "", url.toString());
    return true;
  }

  async function ensureToken() {
    let tokens = loadTokens();
    if (tokens && Date.now() < tokens.exp) return tokens.access_token;
    if (tokens?.refresh_token) {
      try {
        await tokenRequest({
          grant_type: "refresh_token",
          client_id: cfg.oidc_client_id,
          refresh_token: tokens.refresh_token,
        });
        return loadTokens().access_token;
      } catch {
        sessionStorage.removeItem(TOKENS_KEY);
      }
    }
    await login(); // редирект; дальше код не выполняется
    return null;
  }

  const origFetch = window.fetch.bind(window);
  window.fetch = async (input, init = {}) => {
    const url = typeof input === "string" ? input : input.url;
    if (cfg?.auth_enabled && url.startsWith("/api")) {
      const token = await ensureToken();
      init.headers = { ...(init.headers || {}), Authorization: `Bearer ${token}` };
      const resp = await origFetch(input, init);
      if (resp.status === 401) {
        sessionStorage.removeItem(TOKENS_KEY);
        await login();
      }
      return resp;
    }
    return origFetch(input, init);
  };

  // Блокирующая инициализация до DOMContentLoaded-кода страниц
  window.authReady = (async () => {
    cfg = await (await origFetch("/api/config")).json();
    if (!cfg.auth_enabled) return;
    await exchangeCode();
    await ensureToken();
  })();
})();
