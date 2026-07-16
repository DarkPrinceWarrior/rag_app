const ENV_TEMPLATE = /{{\s*env\.([A-Z][A-Z0-9_]*)\s*}}/g;

export function resolveEnvTemplates(value) {
  if (typeof value !== 'string') return value;
  const resolved = value.replace(ENV_TEMPLATE, (_match, name) => {
    const replacement = process.env[name];
    if (typeof replacement !== 'string' || !replacement) {
      throw new Error(`required environment variable ${name} is missing`);
    }
    return replacement;
  });
  if (/{{\s*env\./.test(resolved)) {
    throw new Error('unsupported environment template in red-team input');
  }
  return resolved;
}

export function redteamBaseUrl(
  raw = process.env.RAG_REDTEAM_BASE_URL,
  expectedPort = process.env.RAG_REDTEAM_API_PORT,
) {
  if (!raw || !expectedPort || !/^\d{4,5}$/.test(expectedPort)) {
    throw new Error('disposable red-team URL and port are required');
  }
  let url;
  try {
    url = new URL(raw);
  } catch {
    throw new Error('RAG_REDTEAM_BASE_URL must be a valid URL');
  }
  if (url.protocol !== 'http:'
      || url.username || url.password || url.search || url.hash
      || url.hostname !== '127.0.0.1'
      || url.port !== expectedPort
      || expectedPort === '8100') {
    throw new Error('RAG_REDTEAM_BASE_URL must match the disposable loopback API');
  }
  if (url.pathname !== '/' && url.pathname !== '') {
    throw new Error('RAG_REDTEAM_BASE_URL must not contain a path');
  }
  return url.origin;
}

export default class DocRAGenslateProvider {
  id() {
    return 'local-docragenslate';
  }

  async callApi(prompt, context) {
    const token = process.env.RAG_REDTEAM_TOKEN;
    if (!token) return { error: 'RAG_REDTEAM_TOKEN is required' };
    let baseUrl;
    try {
      baseUrl = redteamBaseUrl();
    } catch (error) {
      return { error: error instanceof Error ? error.message : 'invalid red-team URL' };
    }
    const vars = context?.vars ?? {};
    let body;
    try {
      body = { message: resolveEnvTemplates(prompt) };
      if (vars.document_id) body.document_id = resolveEnvTemplates(vars.document_id);
      if (Array.isArray(vars.document_ids)) {
        body.document_ids = vars.document_ids.map(resolveEnvTemplates);
      }
      if (vars.folder_id) body.folder_id = resolveEnvTemplates(vars.folder_id);
    } catch (error) {
      return { error: error instanceof Error ? error.message : 'invalid red-team template' };
    }
    const response = await fetch(`${baseUrl}/api/chat?memory=off`, {
      method: 'POST',
      redirect: 'error',
      headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(120000),
    });
    const raw = await response.text();
    if (!response.ok) return { error: `red-team endpoint returned HTTP ${response.status}` };
    const contentType = response.headers.get('content-type') ?? '';
    if (!contentType.toLowerCase().startsWith('text/event-stream')) {
      return { error: 'red-team endpoint did not return an SSE response' };
    }
    let output = '';
    let sawDone = false;
    for (const block of raw.split('\n\n')) {
      const line = block.split('\n').find((item) => item.startsWith('data: '));
      if (!line) continue;
      let event;
      try {
        event = JSON.parse(line.slice(6));
      } catch {
        return { error: 'red-team endpoint returned malformed SSE JSON' };
      }
      if (event.type === 'delta') output += event.text ?? '';
      if (event.type === 'done') sawDone = true;
      if (event.type === 'error') return { error: 'red-team endpoint returned an SSE error' };
    }
    if (!sawDone || !output.trim()) {
      return { error: 'red-team endpoint returned an incomplete SSE response' };
    }
    return { output, metadata: { http: { status: response.status, statusText: response.statusText } } };
  }
}
