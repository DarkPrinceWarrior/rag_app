import { useState } from 'react'
import { createFileRoute } from '@tanstack/react-router'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api, type MemoryItem } from '@/lib/api'
import { authFetch } from '@/lib/auth'
import { Button } from '@/components/ui/button'

export const Route = createFileRoute('/memory')({ component: MemoryPage })

const SCOPES = ['user', 'project', 'document', 'thread', 'org']
const KINDS = ['preference', 'fact', 'glossary', 'rule', 'task', 'correction', 'summary']

function MemoryPage() {
  const qc = useQueryClient()
  const [q, setQ] = useState('')
  const [scope, setScope] = useState('user')
  const [kind, setKind] = useState('fact')
  const [content, setContent] = useState('')
  const [editId, setEditId] = useState<string | null>(null)
  const [draft, setDraft] = useState('')

  const itemsQ = useQuery({ queryKey: ['memory', q], queryFn: () => api.listMemory({ q: q || undefined }) })
  const candQ = useQuery({ queryKey: ['memory-candidates'], queryFn: () => api.listMemoryCandidates('pending') })

  const invalidate = () => {
    qc.invalidateQueries({ queryKey: ['memory'] })
    qc.invalidateQueries({ queryKey: ['memory-candidates'] })
  }

  const createM = useMutation({
    mutationFn: () => api.createMemory({ scope, kind, content: content.trim() }),
    onSuccess: () => {
      setContent('')
      invalidate()
    },
  })
  const delM = useMutation({ mutationFn: (id: string) => api.deleteMemory(id), onSuccess: invalidate })
  const editM = useMutation({
    mutationFn: (v: { id: string; content: string }) => api.updateMemory(v.id, { content: v.content }),
    onSuccess: () => {
      setEditId(null)
      invalidate()
    },
  })
  const acceptM = useMutation({ mutationFn: (id: string) => api.acceptCandidate(id), onSuccess: invalidate })
  const rejectM = useMutation({ mutationFn: (id: string) => api.rejectCandidate(id), onSuccess: invalidate })
  const purgeM = useMutation({ mutationFn: () => api.purgeMemory(), onSuccess: invalidate })

  async function exportMemory() {
    const r = await authFetch('/api/memory/export')
    if (!r.ok) return
    const blob = new Blob([JSON.stringify(await r.json(), null, 2)], { type: 'application/json' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = 'memory.json'
    a.click()
    URL.revokeObjectURL(a.href)
  }

  return (
    <div className="mx-auto max-w-3xl px-4 py-4">
      <div className="mb-3 flex items-center gap-2">
        <h1 className="text-lg font-semibold">Память</h1>
        <span className="text-xs text-muted-foreground">что приложение помнит о вас и проектах</span>
        <div className="ml-auto flex gap-1.5">
          <Button variant="outline" size="sm" onClick={exportMemory}>
            Экспорт
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              if (confirm('Удалить всю вашу память без возможности восстановления (152-ФЗ)?')) purgeM.mutate()
            }}
          >
            Очистить всё
          </Button>
        </div>
      </div>

      {/* Кандидаты на подтверждение */}
      {(candQ.data?.length ?? 0) > 0 && (
        <div className="mb-4 rounded-lg border border-amber-300/50 bg-amber-50/40 p-3">
          <div className="mb-1.5 text-sm font-medium">На подтверждение ({candQ.data!.length})</div>
          <div className="space-y-1.5">
            {candQ.data!.map((c) => (
              <div key={c.id} className="flex items-center gap-2 text-sm">
                <span className="rounded bg-muted px-1.5 py-0.5 text-xs">{String(c.proposed.kind ?? '')}</span>
                <span className="min-w-0 flex-1 truncate">{String(c.proposed.content ?? '')}</span>
                <span className="text-xs text-muted-foreground">{(c.confidence * 100).toFixed(0)}%</span>
                <Button size="sm" variant="outline" onClick={() => acceptM.mutate(c.id)}>
                  ✓
                </Button>
                <Button size="sm" variant="ghost" onClick={() => rejectM.mutate(c.id)}>
                  ✕
                </Button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Добавление */}
      <div className="mb-4 flex flex-wrap items-end gap-2 rounded-lg border p-3">
        <select value={scope} onChange={(e) => setScope(e.target.value)} className="h-9 rounded-md border bg-card px-2 text-sm">
          {SCOPES.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
        <select value={kind} onChange={(e) => setKind(e.target.value)} className="h-9 rounded-md border bg-card px-2 text-sm">
          {KINDS.map((k) => (
            <option key={k} value={k}>
              {k}
            </option>
          ))}
        </select>
        <input
          value={content}
          onChange={(e) => setContent(e.target.value)}
          placeholder="Например: отчёты присылать в формате XLSX"
          className="h-9 min-w-[16rem] flex-1 rounded-md border bg-card px-3 text-sm"
        />
        <Button size="sm" disabled={!content.trim() || createM.isPending} onClick={() => createM.mutate()}>
          Добавить
        </Button>
      </div>

      {/* Поиск + список */}
      <input
        value={q}
        onChange={(e) => setQ(e.target.value)}
        placeholder="Поиск по памяти…"
        className="mb-2 h-9 w-full rounded-md border bg-card px-3 text-sm"
      />
      <div className="space-y-1.5">
        {itemsQ.data?.length === 0 && <p className="py-6 text-center text-sm text-muted-foreground">Память пуста</p>}
        {itemsQ.data?.map((it: MemoryItem) => (
          <div key={it.id} className="flex items-start gap-2 rounded-lg border px-3 py-2 text-sm">
            <span className="mt-0.5 rounded bg-muted px-1.5 py-0.5 text-xs">{it.kind}/{it.scope}</span>
            {editId === it.id ? (
              <>
                <input
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  className="min-w-0 flex-1 rounded-md border bg-card px-2 py-1 text-sm"
                />
                <Button size="sm" variant="outline" onClick={() => editM.mutate({ id: it.id, content: draft })}>
                  Сохранить
                </Button>
                <Button size="sm" variant="ghost" onClick={() => setEditId(null)}>
                  Отмена
                </Button>
              </>
            ) : (
              <>
                <span className="min-w-0 flex-1 whitespace-pre-wrap">{it.content}</span>
                <button
                  className="text-xs text-muted-foreground hover:text-foreground"
                  onClick={() => {
                    setEditId(it.id)
                    setDraft(it.content)
                  }}
                >
                  ✎
                </button>
                <button className="text-xs text-destructive hover:underline" onClick={() => delM.mutate(it.id)}>
                  Удалить
                </button>
              </>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
