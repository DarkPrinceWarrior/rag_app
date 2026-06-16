import { bearer } from '@/lib/auth'

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type SSEEvent = Record<string, any>

/** POST /api/chat и разбор SSE-потока (data: {...}\n\n) с авто-Bearer. */
export async function streamChat(
  body: unknown,
  onEvent: (ev: SSEEvent) => void,
  signal?: AbortSignal,
  memoryOff = false,
): Promise<void> {
  const token = await bearer()
  const resp = await fetch(memoryOff ? '/api/chat?memory=off' : '/api/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify(body),
    signal,
  })
  if (!resp.ok || !resp.body) {
    throw new Error(`chat: ${resp.status}`)
  }
  const reader = resp.body.getReader()
  const dec = new TextDecoder()
  let buf = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buf += dec.decode(value, { stream: true })
    let i: number
    while ((i = buf.indexOf('\n\n')) >= 0) {
      const line = buf.slice(0, i).trim()
      buf = buf.slice(i + 2)
      if (line.startsWith('data: ')) onEvent(JSON.parse(line.slice(6)))
    }
  }
}
