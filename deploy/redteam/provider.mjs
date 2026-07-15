import net from 'node:net';

function isPrivateAddress(hostname) {
  const host = hostname.replace(/^\[|\]$/g, '');
  if (host === 'localhost' || host === '::1') return true;
  if (net.isIP(host) === 4) {
    const octets = host.split('.').map(Number);
    return octets[0] === 127
      || octets[0] === 10
      || (octets[0] === 172 && octets[1] >= 16 && octets[1] <= 31)
      || (octets[0] === 192 && octets[1] === 168);
  }
  if (net.isIP(host) === 6) {
    const normalized = host.toLowerCase();
    return normalized.startsWith('fc') || normalized.startsWith('fd')
      || /^fe[89ab]/.test(normalized);
  }
  return false;
}

export function redteamBaseUrl(raw = process.env.RAG_REDTEAM_BASE_URL ?? 'http://127.0.0.1:8100') {
  let url;
  try {
    url = new URL(raw);
  } catch {
    throw new Error('RAG_REDTEAM_BASE_URL must be a valid URL');
  }
  if (!['http:', 'https:'].includes(url.protocol)
      || url.username || url.password || url.search || url.hash
      || !isPrivateAddress(url.hostname)) {
    throw new Error('RAG_REDTEAM_BASE_URL must use HTTP(S) on loopback or a private IP literal');
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
    const body = { message: prompt };
    if (vars.document_id) body.document_id = vars.document_id;
    if (Array.isArray(vars.document_ids)) body.document_ids = vars.document_ids;
    if (vars.folder_id) body.folder_id = vars.folder_id;
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
