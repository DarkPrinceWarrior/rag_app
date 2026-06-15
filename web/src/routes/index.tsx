import { useRef, useState } from 'react'
import { createFileRoute, Link } from '@tanstack/react-router'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useVirtualizer } from '@tanstack/react-virtual'
import { api, EXPORT_LABELS, downloadUrl, type Document } from '@/lib/api'
import { authFetch } from '@/lib/auth'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { StatusBadge } from '@/components/StatusBadge'

export const Route = createFileRoute('/')({ component: Library })

const inProgress = (d: Document) => !['done', 'error'].includes(d.status)

function Library() {
  const qc = useQueryClient()
  const [folder, setFolder] = useState<string>('') // '' = все
  const fileInput = useRef<HTMLInputElement>(null)

  const docsQ = useQuery({
    queryKey: ['documents'],
    queryFn: api.listDocuments,
    refetchInterval: (q) => (q.state.data?.some(inProgress) ? 2500 : false),
  })
  const foldersQ = useQuery({ queryKey: ['folders'], queryFn: api.listFolders })

  const upload = useMutation({
    mutationFn: api.uploadDocument,
    onSuccess: () => qc.invalidateQueries({ queryKey: ['documents'] }),
  })

  const docs = (docsQ.data ?? []).filter((d) => !folder || d.folder_id === folder)

  return (
    <div className="mx-auto max-w-5xl px-4 py-5">
      <UploadZone
        busy={upload.isPending}
        onFile={(f) => upload.mutate(f)}
        fileInput={fileInput}
      />
      {upload.isError && <p className="mt-2 text-sm text-destructive">Ошибка загрузки: {String(upload.error)}</p>}

      <SearchPanel folder={folder} />

      <div className="mt-5 flex flex-wrap items-center gap-2">
        <FolderChip active={folder === ''} onClick={() => setFolder('')} label="Все" count={docsQ.data?.length} />
        {foldersQ.data?.map((f) => (
          <FolderChip key={f.id} active={folder === f.id} onClick={() => setFolder(f.id)} label={f.name} count={f.documents} />
        ))}
        <NewFolder onCreated={() => qc.invalidateQueries({ queryKey: ['folders'] })} />
      </div>

      {docsQ.isLoading ? (
        <p className="mt-6 text-sm text-muted-foreground">Загрузка…</p>
      ) : docs.length === 0 ? (
        <p className="mt-6 text-sm text-muted-foreground">Пока нет документов. Загрузите PDF/DOCX/XLSX/PPTX выше.</p>
      ) : (
        <DocList docs={docs} />
      )}
    </div>
  )
}

function UploadZone({
  busy,
  onFile,
  fileInput,
}: {
  busy: boolean
  onFile: (f: File) => void
  fileInput: React.RefObject<HTMLInputElement | null>
}) {
  const [drag, setDrag] = useState(false)
  return (
    <div
      onClick={() => fileInput.current?.click()}
      onDragOver={(e) => {
        e.preventDefault()
        setDrag(true)
      }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDrag(false)
        const f = e.dataTransfer.files[0]
        if (f) onFile(f)
      }}
      className={
        'cursor-pointer rounded-lg border-2 border-dashed p-6 text-center text-sm transition-colors ' +
        (drag ? 'border-primary bg-accent' : 'border-border bg-card text-muted-foreground hover:bg-accent/50')
      }
    >
      {busy ? 'Загружаю…' : 'Перетащите документ сюда или кликните — PDF, DOCX, XLSX, PPTX, JPG, PNG, TXT'}
      <input
        ref={fileInput}
        type="file"
        hidden
        accept=".pdf,.docx,.xlsx,.pptx,.jpg,.jpeg,.png,.txt"
        onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])}
      />
    </div>
  )
}

function FolderChip({ active, onClick, label, count }: { active: boolean; onClick: () => void; label: string; count?: number }) {
  return (
    <button
      onClick={onClick}
      className={
        'rounded-full border px-3 py-1 text-sm transition-colors ' +
        (active ? 'border-primary bg-primary text-primary-foreground' : 'bg-card hover:bg-accent')
      }
    >
      {label}
      {count != null && <span className="ml-1.5 opacity-70">{count}</span>}
    </button>
  )
}

function NewFolder({ onCreated }: { onCreated: () => void }) {
  const [open, setOpen] = useState(false)
  const [name, setName] = useState('')
  const create = useMutation({
    mutationFn: () => api.createFolder(name.trim()),
    onSuccess: () => {
      setName('')
      setOpen(false)
      onCreated()
    },
  })
  if (!open)
    return (
      <Button variant="ghost" size="sm" onClick={() => setOpen(true)}>
        + папка
      </Button>
    )
  return (
    <span className="flex items-center gap-1">
      <Input
        autoFocus
        value={name}
        onChange={(e) => setName(e.target.value)}
        onKeyDown={(e) => e.key === 'Enter' && name.trim() && create.mutate()}
        placeholder="название"
        className="h-8 w-36"
      />
      <Button size="sm" disabled={!name.trim()} onClick={() => create.mutate()}>
        ок
      </Button>
    </span>
  )
}

function DocList({ docs }: { docs: Document[] }) {
  const parentRef = useRef<HTMLDivElement>(null)
  const v = useVirtualizer({
    count: docs.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 84,
    overscan: 8,
  })
  return (
    <div ref={parentRef} className="mt-4 max-h-[calc(100vh-260px)] overflow-auto">
      <div style={{ height: v.getTotalSize(), position: 'relative' }}>
        {v.getVirtualItems().map((item) => {
          const d = docs[item.index]
          return (
            <div
              key={d.id}
              style={{ position: 'absolute', top: 0, left: 0, width: '100%', transform: `translateY(${item.start}px)` }}
              className="pb-2"
            >
              <DocRow d={d} />
            </div>
          )
        })}
      </div>
    </div>
  )
}

function DocRow({ d }: { d: Document }) {
  const progress =
    d.status === 'translating' && d.segment_count
      ? `${Math.round((d.translated_count / d.segment_count) * 100)}%`
      : null
  return (
    <div className="flex items-center gap-3 rounded-lg border bg-card px-4 py-3 shadow-sm">
      <div className="min-w-0 flex-1">
        <div className="truncate font-medium">{d.filename}</div>
        <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
          <StatusBadge status={d.status} />
          {progress && <span>{progress}</span>}
          {d.page_count != null && <span>{d.page_count} стр.</span>}
          {d.chunk_count > 0 && <span>{d.chunk_count} чанков</span>}
          {d.review_count > 0 && <span className="text-amber-600">⚠ проверить числа: {d.review_count}</span>}
          {d.error && <span className="text-destructive">{d.error.slice(0, 80)}</span>}
        </div>
      </div>
      <div className="flex shrink-0 flex-wrap items-center justify-end gap-1.5">
        {d.exports.map((k) => (
          <a key={k} href={downloadUrl(d.id, k)} onClick={(e) => downloadAuthed(e, downloadUrl(d.id, k))}>
            <Button variant="outline" size="sm">
              {EXPORT_LABELS[k] ?? k}
            </Button>
          </a>
        ))}
        {d.status === 'done' && (
          <>
            <Link to="/view/$id" params={{ id: d.id }}>
              <Button variant="secondary" size="sm">
                Просмотр
              </Button>
            </Link>
            <Link to="/chat" search={{ doc: d.id }}>
              <Button variant="ghost" size="sm">
                Чат
              </Button>
            </Link>
          </>
        )}
        {d.status === 'error' && (
          <Button variant="ghost" size="sm" onClick={() => void api.retry(d.id)}>
            Повторить
          </Button>
        )}
      </div>
    </div>
  )
}

// Скачивание через authFetch (download-роут за require_user) → blob → клик
async function downloadAuthed(e: React.MouseEvent, url: string) {
  e.preventDefault()
  const r = await authFetch(url)
  if (!r.ok) return
  const blob = await r.blob()
  const cd = r.headers.get('Content-Disposition') || ''
  const name = /filename="?([^"]+)"?/.exec(cd)?.[1] || 'document'
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = name
  a.click()
  URL.revokeObjectURL(a.href)
}

function SearchPanel({ folder }: { folder: string }) {
  const [q, setQ] = useState('')
  const [submitted, setSubmitted] = useState('')
  const searchQ = useQuery({
    queryKey: ['search', submitted, folder],
    queryFn: () => api.search(submitted, folder ? { folder_id: folder } : {}),
    enabled: submitted.length >= 2,
  })
  return (
    <div className="mt-4">
      <div className="flex gap-2">
        <Input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && setSubmitted(q.trim())}
          placeholder="Поиск по библиотеке (гибрид + reranker)…"
        />
        <Button onClick={() => setSubmitted(q.trim())} disabled={q.trim().length < 2}>
          Найти
        </Button>
      </div>
      {searchQ.data && (
        <div className="mt-2 space-y-1.5">
          {searchQ.data.length === 0 && <p className="text-sm text-muted-foreground">Ничего не найдено.</p>}
          {searchQ.data.map((h) => (
            <Link
              key={h.chunk_id}
              to="/view/$id"
              params={{ id: h.document_id }}
              search={{ page: h.page_start != null ? h.page_start + 1 : undefined }}
              className="block rounded-md border bg-card px-3 py-2 text-sm hover:bg-accent"
            >
              <div className="flex justify-between gap-2">
                <span className="truncate font-medium">{h.filename}</span>
                <span className="shrink-0 text-xs text-muted-foreground">{h.heading_path}</span>
              </div>
              <div className="mt-0.5 line-clamp-2 text-muted-foreground">{h.snippet}</div>
            </Link>
          ))}
        </div>
      )}
    </div>
  )
}
