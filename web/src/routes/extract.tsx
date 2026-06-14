import { useState } from 'react'
import { createFileRoute, Link } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { api, type ExtractTable } from '@/lib/api'
import { authFetch } from '@/lib/auth'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'

export const Route = createFileRoute('/extract')({ component: Extract })

function Extract() {
  const [docId, setDocId] = useState('')
  const [query, setQuery] = useState('')
  const [table, setTable] = useState<ExtractTable | null>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')

  const docsQ = useQuery({
    queryKey: ['documents'],
    queryFn: api.listDocuments,
    select: (ds) => ds.filter((d) => d.status === 'done'),
  })

  async function run() {
    if (!query.trim()) return
    setBusy(true)
    setErr('')
    try {
      setTable(await api.extractTable(query.trim(), docId || null))
    } catch (e) {
      setErr(String(e))
    }
    setBusy(false)
  }

  async function downloadXlsx() {
    if (!table) return
    const r = await authFetch('/api/extract/xlsx', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(table),
    })
    const blob = await r.blob()
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = (table.title || 'extract').slice(0, 60).replace(/[^\wа-яА-Я -]/g, '') + '.xlsx'
    a.click()
    URL.revokeObjectURL(a.href)
  }

  return (
    <div className="mx-auto max-w-5xl px-4 py-5">
      <p className="mb-3 text-sm text-muted-foreground">
        Запрос вроде «вытащи все спецификации труб в таблицу» или «сведи сроки и штрафы». Результат — в XLSX.
      </p>
      <div className="flex flex-wrap gap-2">
        <select value={docId} onChange={(e) => setDocId(e.target.value)} className="h-9 rounded-md border bg-card px-2 text-sm">
          <option value="">Вся библиотека</option>
          {docsQ.data?.map((d) => (
            <option key={d.id} value={d.id}>
              {d.filename}
            </option>
          ))}
        </select>
        <Input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && run()}
          placeholder="что свести в таблицу…"
          className="min-w-64 flex-1"
        />
        <Button onClick={run} disabled={busy || !query.trim()}>
          {busy ? 'Извлекаю…' : 'Извлечь'}
        </Button>
        <Button variant="secondary" disabled={!table?.rows.length} onClick={downloadXlsx}>
          Скачать XLSX
        </Button>
      </div>
      {err && <p className="mt-2 text-sm text-destructive">Ошибка: {err}</p>}

      {table && (
        <div className="mt-5">
          <h2 className="mb-2 text-base font-semibold">{table.title}</h2>
          {table.rows.length === 0 ? (
            <p className="text-sm text-muted-foreground">Данных по запросу не нашлось.</p>
          ) : (
            <div className="overflow-auto rounded-lg border bg-card shadow-sm">
              <table className="w-full border-collapse text-sm">
                <thead>
                  <tr>
                    {table.columns.map((c) => (
                      <th key={c} className="border-b bg-muted px-3 py-2 text-left font-semibold">
                        {c}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {table.rows.map((r, i) => (
                    <tr key={i} className="border-b last:border-0">
                      {table.columns.map((_, j) => (
                        <td key={j} className="px-3 py-2 align-top">
                          {r[j]}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {table.sources.length > 0 && (
            <div className="mt-3 text-xs text-muted-foreground">
              Источники:{' '}
              {table.sources.map((s, i) => (
                <span key={s.n}>
                  {i > 0 && ' · '}
                  <Link
                    to="/view/$id"
                    params={{ id: s.document_id }}
                    search={{ seg: s.segment_ids?.[0], page: s.page ?? undefined }}
                    className="text-primary hover:underline"
                  >
                    [{s.n}] {s.filename}
                    {s.page ? ` · стр. ${s.page}` : ''}
                  </Link>
                </span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
