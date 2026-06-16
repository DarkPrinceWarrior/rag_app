import { authFetch } from '@/lib/auth'

// доступные виды скачивания (kind для /download/{kind}) и их подписи
export const EXPORT_LABELS: Record<string, string> = {
  docx: 'DOCX',
  pdf: 'PDF (RU)',
  pdf_dual: 'PDF (EN+RU)',
  source: 'Исходный формат',
}

export interface Document {
  id: string
  filename: string
  status: string
  kind: string
  size_bytes: number
  page_count: number | null
  segment_count: number
  translated_count: number
  review_count: number
  chunk_count: number
  exports: string[]
  folder_id: string | null
  error: string | null
  created_at: string
}

export interface Folder {
  id: string
  name: string
  documents: number
}

export interface SearchHit {
  chunk_id: string
  document_id: string
  filename: string
  heading_path: string
  kind: string
  page_start: number | null
  page_end: number | null
  snippet: string
  score: number
}

export interface Segment {
  id: string
  idx: number
  page_idx: number | null
  kind: string
  heading_level: number | null
  source_text: string
  translated_text: string | null
  needs_review: boolean
  validation: Record<string, unknown> | null
  bbox?: number[] | null
  page_size?: number[] | null
  table_cells?: TableCell[][] | null
  table_cells_ru?: TableCell[][] | null
  caption?: string | null
  caption_ru?: string | null
}

export interface TableCell {
  text: string
  colspan: number
  rowspan: number
}

export interface Citation {
  n: number
  chunk_id: string
  document_id: string
  filename: string
  heading_path: string
  page_start: number | null
  page_end: number | null
  segment_ids: string[]
  bboxes: number[][]
}

export interface ChatSession {
  id: string
  title: string
  document_id: string | null
  folder_id: string | null
  created_at: string
  updated_at: string
}

export interface ChatHistoryMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  citations: Citation[]
  created_at: string
}

export interface MemoryItem {
  id: string
  scope: string
  kind: string
  content: string
  sensitivity: string
  importance: number
  confidence: number
  project_id: string | null
  document_id: string | null
  thread_id: string | null
  memory_provider: string
  created_at: string
  updated_at: string
}

export interface MemoryCandidate {
  id: string
  action: string
  status: string
  confidence: number
  proposed: Record<string, unknown>
  rationale: string | null
  created_at: string
}

export interface ExtractTable {
  title: string
  columns: string[]
  rows: string[][]
  sources: { n: number; document_id: string; filename: string; heading_path: string; page: number | null; segment_ids: string[] }[]
}

async function jget<T>(path: string): Promise<T> {
  const r = await authFetch(path)
  if (!r.ok) throw new Error(`${r.status}: ${(await r.json().catch(() => ({}))).detail ?? r.statusText}`)
  return r.json() as Promise<T>
}

async function jsend<T>(path: string, method: string, body?: unknown): Promise<T> {
  const r = await authFetch(path, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  if (!r.ok) throw new Error(`${r.status}: ${(await r.json().catch(() => ({}))).detail ?? r.statusText}`)
  return r.json() as Promise<T>
}

async function jdel(path: string): Promise<void> {
  const r = await authFetch(path, { method: 'DELETE' })
  if (!r.ok && r.status !== 204) {
    throw new Error(`${r.status}: ${(await r.json().catch(() => ({}))).detail ?? r.statusText}`)
  }
}

export const api = {
  listDocuments: () => jget<Document[]>('/api/documents'),
  getDocument: (id: string) => jget<Document>(`/api/documents/${id}`),
  getSegments: (id: string) => jget<Segment[]>(`/api/documents/${id}/segments`),
  reexport: (id: string) => jsend<{ status: string }>(`/api/documents/${id}/reexport`, 'POST'),
  retry: (id: string) => jsend<{ status: string }>(`/api/documents/${id}/retry`, 'POST'),
  reparseOcr: (id: string, lang = 'east_slavic') =>
    jsend<{ status: string; ocr_lang: string }>(`/api/documents/${id}/reparse-ocr`, 'POST', { lang }),
  patchSegment: (segId: string, text: string) =>
    jsend<Segment>(`/api/segments/${segId}`, 'PATCH', { translated_text: text }),

  listFolders: () => jget<Folder[]>('/api/folders'),
  createFolder: (name: string) => jsend<Folder>('/api/folders', 'POST', { name }),
  moveDocument: (id: string, folder_id: string | null) =>
    jsend<{ status: string }>(`/api/documents/${id}/folder`, 'PATCH', { folder_id }),

  search: (q: string, opts: { document_id?: string; folder_id?: string } = {}) => {
    const p = new URLSearchParams({ q })
    if (opts.document_id) p.set('document_id', opts.document_id)
    if (opts.folder_id) p.set('folder_id', opts.folder_id)
    return jget<SearchHit[]>(`/api/search?${p}`)
  },

  listSessions: () => jget<ChatSession[]>('/api/chat/sessions'),
  getSessionMessages: (id: string) =>
    jget<ChatHistoryMessage[]>(`/api/chat/sessions/${id}/messages`),

  // Память (docs/MEMORY_rev4_mem0_articles.md §8)
  listMemory: (opts: { scope?: string; project_id?: string; q?: string } = {}) => {
    const p = new URLSearchParams()
    if (opts.scope) p.set('scope', opts.scope)
    if (opts.project_id) p.set('project_id', opts.project_id)
    if (opts.q) p.set('q', opts.q)
    const qs = p.toString()
    return jget<MemoryItem[]>(`/api/memory${qs ? '?' + qs : ''}`)
  },
  createMemory: (body: { scope: string; kind: string; content: string; sensitivity?: string }) =>
    jsend<MemoryItem>('/api/memory', 'POST', body),
  updateMemory: (id: string, body: { content?: string; importance?: number; sensitivity?: string }) =>
    jsend<MemoryItem>(`/api/memory/${id}`, 'PATCH', body),
  deleteMemory: (id: string) => jdel(`/api/memory/${id}`),
  listMemoryCandidates: (status = 'pending') =>
    jget<MemoryCandidate[]>(`/api/memory/candidates?status=${status}`),
  acceptCandidate: (id: string) =>
    jsend<{ item_id: string | null }>(`/api/memory/candidates/${id}/accept`, 'POST'),
  rejectCandidate: (id: string) =>
    jsend<MemoryCandidate>(`/api/memory/candidates/${id}/reject`, 'POST'),
  purgeMemory: () => jsend<{ purged: string; items: number; events: number }>('/api/memory/purge', 'POST', {}),

  extractTable: (query: string, document_id: string | null) =>
    jsend<ExtractTable>('/api/extract/table', 'POST', { query, document_id }),

  translateFragment: (text: string) =>
    jsend<{ text: string; engine: string; ms: number }>('/api/translate/fragment', 'POST', { text }),

  async uploadDocument(file: File): Promise<Document> {
    const fd = new FormData()
    fd.append('file', file)
    const r = await authFetch('/api/documents', { method: 'POST', body: fd })
    if (!r.ok) throw new Error(`${r.status}: ${(await r.json().catch(() => ({}))).detail ?? r.statusText}`)
    return r.json()
  },
}

export const downloadUrl = (docId: string, key: string) => `/api/documents/${docId}/download/${key}`
