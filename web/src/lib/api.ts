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

export const api = {
  listDocuments: () => jget<Document[]>('/api/documents'),
  getDocument: (id: string) => jget<Document>(`/api/documents/${id}`),
  getSegments: (id: string) => jget<Segment[]>(`/api/documents/${id}/segments`),
  reexport: (id: string) => jsend<{ status: string }>(`/api/documents/${id}/reexport`, 'POST'),
  retry: (id: string) => jsend<{ status: string }>(`/api/documents/${id}/retry`, 'POST'),
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
