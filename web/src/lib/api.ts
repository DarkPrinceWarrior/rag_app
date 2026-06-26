import { authFetch } from '@/lib/auth'

// Потолок сегментов, который грузит вьювер за один запрос. Бэкстоп против
// патологических документов (xlsx-дата-дампы на сотни тысяч ячеек вешали
// «Загрузка…»). Должен совпадать с дефолтом `limit` в API list_segments.
export const SEGMENTS_LIMIT = 4000

// виды скачивания ПЕРЕВОДА (kind для /download/{kind}) и их подписи.
// Это всё — переведённый документ в разных форматах; «оригинал» (как загружен)
// скачивается отдельным пунктом меню (kind=original).
export const EXPORT_LABELS: Record<string, string> = {
  docx: 'Перевод — DOCX',
  pdf: 'Перевод — PDF (рус.)',
  pdf_dual: 'Перевод — PDF (англ.+рус.)',
  source: 'Перевод — в исходном формате',
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
  has_view?: boolean // PDF-рендер OOXML готов целиком (оригинал И перевод)
  has_view_orig?: boolean // рендер оригинала готов (рано, после парсинга)
  has_view_ru?: boolean // рендер перевода готов (на экспорте)
  parser_backend?: string | null // движок парсинга pdf_text: mineru | dots_mocr | paddle_vl
  source_lang?: string | null // язык-источник, определён автоматически (ru|en|zh|auto); цель всегда ru
  source_type?: string // file | web (ТЗ §4.7.2)
  project_object?: string | null // объект строительства (ТЗ §4.7.2/§4.7.3)
  created_at: string
}

// история правок перевода сегмента (ТЗ §4.7.2)
export interface SegmentVersion {
  id: string
  old_text: string | null
  new_text: string | null
  editor: string
  created_at: string
}

// фильтры списка/поиска по библиотеке (ТЗ §4.7.3): только тип и даты —
// имя файла и содержимое ищет единый /api/search
export interface DocFilters {
  kind?: string
  date_from?: string
  date_to?: string
}

export interface Folder {
  id: string
  name: string
  documents: number
}

// Доп. перевод документа на язык, отличный от источника (ТЗ §4.3): RU→EN/RU→ZH.
export interface TranslationInfo {
  target_lang: string
  status: string // translating | exporting | done | error
  translated_count: number
  segment_count: number
  needs_review_count: number
  has_export: boolean
  error: string | null
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
  match?: 'filename' | 'content'
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
  image_url?: string | null
  location?: Record<string, number> | null
  // положение в левом (оригинал) и правом (перевод) рендер-PDF — кросс-навигация
  loc_left?: PdfLoc | null
  loc_right?: PdfLoc | null
}

export interface PdfLoc {
  page: number // 0-based
  bbox: number[] // [x0,y0,x1,y1] top-left, pt
  pagesize: number[] // [w,h] pt
}

export interface TableCell {
  text: string
  colspan: number
  rowspan: number
}

// xlsx → интерактивный грид-просмотр (а не office-PDF «принт»): лист = сетка
// строковых значений ячеек, оригинал + перевод.
export interface SheetData {
  name: string
  name_ru?: string // перевод названия листа (для вкладок справа)
  orig: string[][]
  ru: string[][]
  total_rows: number
  total_cols: number
  truncated: boolean
  charts?: string[] // заголовки встроенных диаграмм (в гриде не рисуются)
}
export interface SheetsResponse {
  sheets: SheetData[]
  translated_ready: boolean
}

// pptx → интерактивный просмотр слайдов (а не office-PDF «принт»).
export interface SlideLine {
  orig: string
  ru: string
  level: number
}
export interface SlideBlock {
  type: 'text' | 'table' | 'image'
  lines?: SlideLine[]
  rows?: { orig: string; ru: string }[][]
  shape?: number
}
export interface Slide {
  index: number
  title: string
  title_ru: string
  blocks: SlideBlock[]
}
export interface SlidesResponse {
  slides: Slide[]
  translated_ready: boolean
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
  listDocuments: (filters: DocFilters = {}) => {
    const p = new URLSearchParams()
    for (const [k, v] of Object.entries(filters)) if (v) p.set(k, v)
    const qs = p.toString()
    return jget<Document[]>(`/api/documents${qs ? '?' + qs : ''}`)
  },
  getDocument: (id: string) => jget<Document>(`/api/documents/${id}`),
  listSegmentVersions: (segId: string) =>
    jget<SegmentVersion[]>(`/api/segments/${segId}/versions`),
  deleteDocument: (id: string) => jdel(`/api/documents/${id}`),
  getSegments: (id: string) => jget<Segment[]>(`/api/documents/${id}/segments?limit=${SEGMENTS_LIMIT}`),
  getSheets: (id: string) => jget<SheetsResponse>(`/api/documents/${id}/sheets`),
  getSlides: (id: string) => jget<SlidesResponse>(`/api/documents/${id}/slides`),
  reexport: (id: string) => jsend<{ status: string }>(`/api/documents/${id}/reexport`, 'POST'),
  retry: (id: string) => jsend<{ status: string }>(`/api/documents/${id}/retry`, 'POST'),
  reparseOcr: (id: string, lang = 'east_slavic') =>
    jsend<{ status: string; ocr_lang: string }>(`/api/documents/${id}/reparse-ocr`, 'POST', { lang }),
  reparse: (id: string, backend: string) =>
    jsend<{ status: string; backend: string }>(`/api/documents/${id}/reparse`, 'POST', { backend }),
  patchSegment: (segId: string, text: string) =>
    jsend<Segment>(`/api/segments/${segId}`, 'PATCH', { translated_text: text }),

  listFolders: () => jget<Folder[]>('/api/folders'),
  createFolder: (name: string) => jsend<Folder>('/api/folders', 'POST', { name }),
  deleteFolder: (id: string) => jdel(`/api/folders/${id}`),
  moveDocument: (id: string, folder_id: string | null) =>
    jsend<{ status: string }>(`/api/documents/${id}/folder`, 'PATCH', { folder_id }),

  // доп. переводы документа (ТЗ §4.3)
  listTranslations: (id: string) => jget<TranslationInfo[]>(`/api/documents/${id}/translations`),
  createTranslation: (id: string, target_lang: string) =>
    jsend<{ target_lang: string; status: string }>(
      `/api/documents/${id}/translations`,
      'POST',
      { target_lang },
    ),

  search: (
    q: string,
    opts: {
      document_id?: string
      folder_id?: string
      kind?: string
      date_from?: string
      date_to?: string
    } = {},
  ) => {
    const p = new URLSearchParams({ q })
    for (const k of ['document_id', 'folder_id', 'kind', 'date_from', 'date_to'] as const) {
      const v = opts[k]
      if (v) p.set(k, v)
    }
    return jget<SearchHit[]>(`/api/search?${p}`)
  },

  listSessions: () => jget<ChatSession[]>('/api/chat/sessions'),
  getSessionMessages: (id: string) =>
    jget<ChatHistoryMessage[]>(`/api/chat/sessions/${id}/messages`),
  deleteSession: (id: string) => jdel(`/api/chat/sessions/${id}`),

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

  extractTable: (
    query: string,
    scope: { document_id?: string | null; folder_id?: string; document_ids?: string[] } = {},
  ) => jsend<ExtractTable>('/api/extract/table', 'POST', { query, ...scope }),

  async uploadDocument(file: File): Promise<Document> {
    // Направление перевода не выбирается: язык-источник определяется
    // автоматически, цель всегда русский (ТЗ §4.3).
    const fd = new FormData()
    fd.append('file', file)
    const r = await authFetch('/api/documents', { method: 'POST', body: fd })
    if (!r.ok) throw new Error(`${r.status}: ${(await r.json().catch(() => ({}))).detail ?? r.statusText}`)
    return r.json()
  },
}

export const downloadUrl = (docId: string, key: string) => `/api/documents/${docId}/download/${key}`
export const translationDownloadUrl = (docId: string, lang: string) =>
  `/api/documents/${docId}/translations/${lang}/download`
export const slideImageUrl = (docId: string, slide: number, shape: number) =>
  `/api/documents/${docId}/slide-image/${slide}/${shape}`
