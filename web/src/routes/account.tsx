import { useState } from 'react'
import { createFileRoute } from '@tanstack/react-router'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api, type MemoryItem } from '@/lib/api'
import { authFetch, currentUser, logout } from '@/lib/auth'
import { Button } from '@/components/ui/button'
import { Select } from '@/components/ui/select'

export const Route = createFileRoute('/account')({ component: AccountPage })

const SCOPES = ['user', 'project', 'document', 'thread', 'org']
const KINDS = ['preference', 'fact', 'glossary', 'rule', 'task', 'correction', 'summary']

// русские подписи областей и типов памяти (значения в БД остаются английскими)
const SCOPE_RU: Record<string, string> = {
  user: 'Пользователь',
  project: 'Проект',
  document: 'Документ',
  thread: 'Диалог',
  org: 'Организация',
}
const KIND_RU: Record<string, string> = {
  preference: 'Предпочтение',
  fact: 'Факт',
  glossary: 'Глоссарий',
  rule: 'Правило',
  task: 'Задача',
  correction: 'Исправление',
  summary: 'Сводка',
}
const scopeRu = (s: string) => SCOPE_RU[s] ?? s
const kindRu = (k: string) => KIND_RU[k] ?? k

function AccountPage() {
  const user = currentUser()
  return (
    <div className="mx-auto max-w-3xl px-4 py-5">
      {/* Карточка пользователя */}
      <div className="mb-5 flex items-center gap-3 rounded-xl border bg-card p-4">
        <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-primary/10 text-lg font-semibold text-primary">
          {user.username.slice(0, 1).toUpperCase()}
        </div>
        <div className="min-w-0">
          <div className="text-base font-semibold">{user.username}</div>
          <div className="mt-0.5 flex flex-wrap gap-1">
            {(user.roles.length ? user.roles : ['user']).map((r) => (
              <span key={r} className="rounded bg-muted px-1.5 py-0.5 text-xs text-muted-foreground">
                {r === 'admin' ? 'администратор' : 'пользователь'}
              </span>
            ))}
          </div>
        </div>
        <Button variant="outline" size="sm" className="ml-auto" onClick={logout}>
          Выйти
        </Button>
      </div>

      <MemorySection isAdmin={user.isAdmin} />
    </div>
  )
}

function MemorySection({ isAdmin }: { isAdmin: boolean }) {
  const qc = useQueryClient()
  const [q, setQ] = useState('')
  const [scope, setScope] = useState('user')
  const [kind, setKind] = useState('fact')
  const [content, setContent] = useState('')
  const [editId, setEditId] = useState<string | null>(null)
  const [draft, setDraft] = useState('')
  const [open, setOpen] = useState(false) // «Память» свёрнута по умолчанию (не на весь экран)
  const [fScope, setFScope] = useState('') // фильтр списка по области ('' = все)
  const [fKind, setFKind] = useState('') // фильтр списка по типу ('' = все)

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

  const count = itemsQ.data?.length
  const pending = candQ.data?.length ?? 0
  // список с учётом фильтров области/типа (поиск q идёт на сервер)
  const items = (itemsQ.data ?? []).filter(
    (it) => (!fScope || it.scope === fScope) && (!fKind || it.kind === fKind),
  )

  return (
    <div className="rounded-xl border bg-card">
      {/* Заголовок-переключатель: «Память» свёрнута, разворачивается по клику */}
      <div className="flex items-center gap-2 px-4 py-3">
        <button onClick={() => setOpen((v) => !v)} className="flex min-w-0 flex-1 items-center gap-2 text-left">
          <span className="text-muted-foreground">{open ? '▾' : '▸'}</span>
          <h1 className="text-base font-semibold">Память</h1>
          {pending > 0 && (
            <span className="rounded-full bg-amber-100 px-1.5 py-0.5 text-xs text-amber-800">
              {pending} на подтверждение
            </span>
          )}
          <span className="truncate text-xs text-muted-foreground">
            {count != null ? `${count} запис${count % 10 === 1 && count % 100 !== 11 ? 'ь' : 'ей'} · ` : ''}
            что приложение помнит о вас и проектах{isAdmin ? ' (админ: видны все)' : ''}
          </span>
        </button>
        {open && (
          <div className="flex shrink-0 gap-1.5">
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
        )}
      </div>

      {!open ? null : (
        <div className="border-t px-4 py-4">
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
        <Select
          value={scope}
          onChange={setScope}
          options={SCOPES.map((s) => ({ value: s, label: scopeRu(s) }))}
          className="min-w-[7rem]"
        />
        <Select
          value={kind}
          onChange={setKind}
          options={KINDS.map((k) => ({ value: k, label: kindRu(k) }))}
          className="min-w-[8rem]"
        />
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

      {/* Поиск + фильтры (область / тип) */}
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Поиск по памяти…"
          className="h-9 min-w-[12rem] flex-1 rounded-md border bg-card px-3 text-sm"
        />
        <Select
          value={fScope}
          onChange={setFScope}
          options={[{ value: '', label: 'Все области' }, ...SCOPES.map((s) => ({ value: s, label: scopeRu(s) }))]}
          className="min-w-[9rem]"
        />
        <Select
          value={fKind}
          onChange={setFKind}
          options={[{ value: '', label: 'Все типы' }, ...KINDS.map((k) => ({ value: k, label: kindRu(k) }))]}
          className="min-w-[9rem]"
        />
      </div>
      <div className="space-y-1.5">
        {items.length === 0 && (
          <p className="py-6 text-center text-sm text-muted-foreground">
            {itemsQ.data?.length ? 'Ничего не найдено по фильтрам' : 'Память пуста'}
          </p>
        )}
        {items.map((it: MemoryItem) => (
          <div key={it.id} className="flex items-start gap-2 rounded-lg border px-3 py-2 text-sm">
            <span className="mt-0.5 shrink-0 rounded bg-muted px-1.5 py-0.5 text-xs">
              {kindRu(it.kind)} · {scopeRu(it.scope)}
            </span>
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
      )}
    </div>
  )
}
