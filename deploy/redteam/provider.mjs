export default class DocRAGenslateProvider {
  id() {
    return 'local-docragenslate';
  }

  async callApi(prompt, context) {
    const baseUrl = process.env.RAG_REDTEAM_BASE_URL ?? 'http://127.0.0.1:8100';
    const token = process.env.RAG_REDTEAM_TOKEN;
    if (!token) return { error: 'RAG_REDTEAM_TOKEN is required' };
    const vars = context?.vars ?? {};
    const body = { message: prompt };
    if (vars.document_id) body.document_id = vars.document_id;
    if (Array.isArray(vars.document_ids)) body.document_ids = vars.document_ids;
    if (vars.folder_id) body.folder_id = vars.folder_id;
    const response = await fetch(`${baseUrl}/api/chat?memory=off`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(120000),
    });
    const raw = await response.text();
    if (!response.ok) return { error: `HTTP ${response.status}: ${raw.slice(0, 500)}` };
    let output = '';
    for (const block of raw.split('\n\n')) {
      const line = block.split('\n').find((item) => item.startsWith('data: '));
      if (!line) continue;
      const event = JSON.parse(line.slice(6));
      if (event.type === 'delta') output += event.text ?? '';
      if (event.type === 'error') return { error: event.detail ?? 'SSE error' };
    }
    return { output, raw, metadata: { http: { status: response.status, statusText: response.statusText } } };
  }
}
