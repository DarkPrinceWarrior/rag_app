import { useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api, type SheetData } from '@/lib/api'

// A, B, …, Z, AA, AB, … — буква столбца по индексу (как в Excel).
function colLabel(i: number): string {
  let s = ''
  let n = i + 1
  while (n > 0) {
    const m = (n - 1) % 26
    s = String.fromCharCode(65 + m) + s
    n = Math.floor((n - 1) / 26)
  }
  return s
}

const NUM_RE = /^[-+]?[\d\s.,%()$€₽]+$/

type Sel = { r: number; c: number } | null

function Grid({
  rows,
  label,
  sel,
  onSelect,
  scrollRef,
  onScroll,
}: {
  rows: string[][]
  label: string
  sel: Sel
  onSelect: (r: number, c: number) => void
  scrollRef: React.RefObject<HTMLDivElement | null>
  onScroll: () => void
}) {
  const nCols = rows.reduce((m, r) => Math.max(m, r.length), 0)
  const cols = Array.from({ length: nCols }, (_, c) => c)
  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      <div className="border-b bg-muted/40 px-3 py-1 text-xs font-medium text-muted-foreground">{label}</div>
      <div ref={scrollRef} onScroll={onScroll} className="min-h-0 min-w-0 flex-1 overflow-auto">
        <table className="border-collapse text-xs">
          <thead>
            <tr>
              <th className="sticky left-0 top-0 z-30 min-w-[2.5rem] border border-border bg-muted" />
              {cols.map((c) => (
                <th
                  key={c}
                  className="sticky top-0 z-20 min-w-[5rem] border border-border bg-muted px-2 py-1 font-medium text-muted-foreground"
                >
                  {colLabel(c)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, r) => (
              <tr key={r}>
                <td className="sticky left-0 z-10 min-w-[2.5rem] border border-border bg-muted px-1 text-center align-top text-muted-foreground tabular-nums">
                  {r + 1}
                </td>
                {cols.map((c) => {
                  const v = row[c] ?? ''
                  const on = sel?.r === r && sel?.c === c
                  const num = v !== '' && NUM_RE.test(v)
                  return (
                    <td
                      key={c}
                      onClick={() => onSelect(r, c)}
                      title={v}
                      className={
                        'max-w-[22rem] cursor-cell truncate border border-border px-2 py-1 align-top ' +
                        (num ? 'text-right tabular-nums ' : '') +
                        (on ? 'bg-primary/20 ring-1 ring-inset ring-primary' : 'bg-card hover:bg-accent/40')
                      }
                    >
                      {v}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export function XlsxView({ docId }: { docId: string }) {
  const q = useQuery({ queryKey: ['sheets', docId], queryFn: () => api.getSheets(docId) })
  const [active, setActive] = useState(0)
  const [sel, setSel] = useState<Sel>(null)
  const leftRef = useRef<HTMLDivElement>(null)
  const rightRef = useRef<HTMLDivElement>(null)
  const syncing = useRef(false)

  if (q.isLoading) return <p className="p-6 text-sm text-muted-foreground">Загрузка таблицы…</p>
  if (q.isError || !q.data)
    return <p className="p-6 text-sm text-destructive">Не удалось загрузить таблицу.</p>
  const sheets: SheetData[] = q.data.sheets
  if (!sheets.length) return <p className="p-6 text-sm text-muted-foreground">В книге нет листов.</p>

  const idx = Math.min(active, sheets.length - 1)
  const sheet = sheets[idx]
  const translatedReady = q.data.translated_ready

  // синхронная прокрутка панелей: строка N оригинала ↔ строка N перевода
  function sync(from: 'l' | 'r') {
    if (syncing.current) return
    syncing.current = true
    const a = (from === 'l' ? leftRef : rightRef).current
    const b = (from === 'l' ? rightRef : leftRef).current
    if (a && b) {
      b.scrollTop = a.scrollTop
      b.scrollLeft = a.scrollLeft
    }
    requestAnimationFrame(() => {
      syncing.current = false
    })
  }

  const selOrig = sel ? (sheet.orig[sel.r]?.[sel.c] ?? '') : ''
  const selRu = sel ? (sheet.ru[sel.r]?.[sel.c] ?? '') : ''

  return (
    <div className="flex h-[calc(100vh-97px)] flex-col">
      {/* строка формулы: адрес ячейки + значения оригинала и перевода */}
      <div className="flex items-center gap-3 border-b bg-card px-3 py-1.5 text-xs">
        <span className="rounded border bg-muted px-1.5 py-0.5 font-mono text-muted-foreground">
          {sel ? `${colLabel(sel.c)}${sel.r + 1}` : '—'}
        </span>
        <span className="min-w-0 flex-1 truncate text-muted-foreground">
          ориг.: <span className="text-foreground">{selOrig || '∅'}</span>
        </span>
        <span className="min-w-0 flex-1 truncate text-muted-foreground">
          пер.: <span className="text-foreground">{selRu || '∅'}</span>
        </span>
        {sheet.truncated && (
          <span className="whitespace-nowrap rounded bg-amber-100 px-2 py-0.5 text-amber-800">
            лист обрезан до {Math.min(sheet.total_rows, 1000).toLocaleString('ru')}×
            {Math.min(sheet.total_cols, 60)} из {sheet.total_rows.toLocaleString('ru')}×{sheet.total_cols}
          </span>
        )}
      </div>

      {/* две панели: оригинал | перевод */}
      <div className="flex min-h-0 flex-1">
        <div className="flex w-1/2 min-w-0 border-r">
          <Grid
            rows={sheet.orig}
            label="оригинал"
            sel={sel}
            onSelect={(r, c) => setSel({ r, c })}
            scrollRef={leftRef}
            onScroll={() => sync('l')}
          />
        </div>
        <div className="flex w-1/2 min-w-0">
          <Grid
            rows={sheet.ru}
            label={translatedReady ? 'перевод' : 'перевод · готовится…'}
            sel={sel}
            onSelect={(r, c) => setSel({ r, c })}
            scrollRef={rightRef}
            onScroll={() => sync('r')}
          />
        </div>
      </div>

      {/* вкладки листов — снизу, как в Excel */}
      {sheets.length > 0 && (
        <div className="flex items-center gap-0.5 overflow-x-auto border-t bg-muted/30 px-2 py-1">
          {sheets.map((s, i) => (
            <button
              key={i}
              onClick={() => {
                setActive(i)
                setSel(null)
              }}
              title={`${s.name} · ${s.total_rows.toLocaleString('ru')}×${s.total_cols}`}
              className={
                'max-w-[14rem] truncate whitespace-nowrap rounded-t border-x border-t px-3 py-1 text-xs ' +
                (i === idx
                  ? 'border-border bg-card font-medium text-foreground'
                  : 'border-transparent bg-muted/50 text-muted-foreground hover:bg-muted')
              }
            >
              {s.name}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
