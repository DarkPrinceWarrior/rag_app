import { useEffect, useMemo, useRef, useState } from 'react'
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
const ROW_HEAD_W = 44 // ширина колонки с номерами строк, px
const MIN_COL_W = 48

type Sel = { r: number; c: number } | null

function Grid({
  rows,
  label,
  sel,
  onSelect,
  widths,
  onResize,
  scrollRef,
  onScroll,
}: {
  rows: string[][]
  label: string
  sel: Sel
  onSelect: (r: number, c: number) => void
  widths: number[]
  onResize: (c: number, w: number) => void
  scrollRef: React.RefObject<HTMLDivElement | null>
  onScroll: () => void
}) {
  const nCols = widths.length
  const cols = Array.from({ length: nCols }, (_, c) => c)
  const tableW = ROW_HEAD_W + widths.reduce((a, b) => a + b, 0)

  function startResize(c: number, e: React.PointerEvent) {
    e.preventDefault()
    e.stopPropagation()
    const startX = e.clientX
    const startW = widths[c]
    const move = (ev: PointerEvent) => onResize(c, Math.max(MIN_COL_W, startW + ev.clientX - startX))
    const up = () => {
      document.removeEventListener('pointermove', move)
      document.removeEventListener('pointerup', up)
    }
    document.addEventListener('pointermove', move)
    document.addEventListener('pointerup', up)
  }

  return (
    <div className="flex min-w-0 flex-1 flex-col">
      <div className="border-b bg-muted/40 px-3 py-1 text-xs font-medium text-muted-foreground">{label}</div>
      <div ref={scrollRef} onScroll={onScroll} className="max-h-[calc(100vh-12rem)] overflow-auto">
        <table className="table-fixed border-collapse text-xs" style={{ width: tableW }}>
          <colgroup>
            <col style={{ width: ROW_HEAD_W }} />
            {cols.map((c) => (
              <col key={c} style={{ width: widths[c] }} />
            ))}
          </colgroup>
          <thead>
            <tr>
              <th className="sticky left-0 top-0 z-30 border border-border bg-muted" />
              {cols.map((c) => (
                <th
                  key={c}
                  className="sticky top-0 z-20 select-none border border-border bg-muted px-2 py-1 font-medium text-muted-foreground"
                >
                  <div className="relative">
                    {colLabel(c)}
                    <span
                      onPointerDown={(e) => startResize(c, e)}
                      title="Потянуть — изменить ширину столбца"
                      className="absolute -right-2.5 top-0 z-10 h-6 w-2.5 cursor-col-resize hover:bg-primary/40"
                    />
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, r) => (
              <tr key={r}>
                <td className="sticky left-0 z-10 border border-border bg-muted px-1 text-center align-top text-muted-foreground tabular-nums">
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
                        'cursor-cell overflow-hidden text-ellipsis whitespace-nowrap border border-border px-2 py-1 ' +
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

// начальная ширина столбца по содержимому (px): по самой длинной ячейке листа.
function baseWidths(sheet: SheetData): number[] {
  const n = Math.max(0, ...sheet.orig.map((r) => r.length), ...sheet.ru.map((r) => r.length))
  const w: number[] = []
  for (let c = 0; c < n; c++) {
    let maxLen = colLabel(c).length
    for (const r of sheet.orig) maxLen = Math.max(maxLen, (r[c] ?? '').length)
    for (const r of sheet.ru) maxLen = Math.max(maxLen, (r[c] ?? '').length)
    w.push(Math.min(360, Math.max(64, Math.round(maxLen * 6.6) + 20)))
  }
  return w
}

export function XlsxView({ docId }: { docId: string }) {
  const q = useQuery({ queryKey: ['sheets', docId], queryFn: () => api.getSheets(docId) })
  const [active, setActive] = useState(0)
  const [sel, setSel] = useState<Sel>(null)
  const [override, setOverride] = useState<Record<number, number>>({})
  const leftRef = useRef<HTMLDivElement>(null)
  const rightRef = useRef<HTMLDivElement>(null)
  const syncing = useRef(false)

  const sheets: SheetData[] = q.data?.sheets ?? []
  const idx = Math.min(active, Math.max(0, sheets.length - 1))
  const sheet = sheets[idx]
  // ширины столбцов по содержимому активного листа; ручное растягивание — поверх.
  const base = useMemo(() => (sheet ? baseWidths(sheet) : []), [sheet])
  const widths = base.map((w, c) => override[c] ?? w)
  // сброс ручных ширин и выделения при смене листа
  useEffect(() => {
    setOverride({})
    setSel(null)
  }, [idx])

  if (q.isLoading) return <p className="p-6 text-sm text-muted-foreground">Загрузка таблицы…</p>
  if (q.isError || !q.data) return <p className="p-6 text-sm text-destructive">Не удалось загрузить таблицу.</p>
  if (!sheets.length || !sheet) return <p className="p-6 text-sm text-muted-foreground">В книге нет листов.</p>

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
  function resize(c: number, w: number) {
    setOverride((o) => ({ ...o, [c]: w }))
  }

  const selOrig = sel ? (sheet.orig[sel.r]?.[sel.c] ?? '') : ''
  const selRu = sel ? (sheet.ru[sel.r]?.[sel.c] ?? '') : ''

  return (
    <div className="flex flex-col">
      {/* строка формулы: адрес + ПОЛНЫЙ текст ячейки (оригинал и перевод, с переносом) */}
      <div className="border-b bg-card px-3 py-1.5 text-xs">
        <div className="flex items-start gap-3">
          <span className="mt-0.5 shrink-0 rounded border bg-muted px-1.5 py-0.5 font-mono text-muted-foreground">
            {sel ? `${colLabel(sel.c)}${sel.r + 1}` : '—'}
          </span>
          <div className="grid min-w-0 flex-1 grid-cols-2 gap-4">
            <div className="min-w-0">
              <div className="text-[10px] uppercase tracking-wide text-muted-foreground">оригинал</div>
              <div className="max-h-28 overflow-auto whitespace-pre-wrap break-words">{selOrig || '∅'}</div>
            </div>
            <div className="min-w-0">
              <div className="text-[10px] uppercase tracking-wide text-muted-foreground">перевод</div>
              <div className="max-h-28 overflow-auto whitespace-pre-wrap break-words">{selRu || '∅'}</div>
            </div>
          </div>
          {sheet.truncated && (
            <span className="mt-0.5 shrink-0 whitespace-nowrap rounded bg-amber-100 px-2 py-0.5 text-amber-800">
              лист обрезан до {Math.min(sheet.total_rows, 1000).toLocaleString('ru')}×
              {Math.min(sheet.total_cols, 60)} из {sheet.total_rows.toLocaleString('ru')}×{sheet.total_cols}
            </span>
          )}
        </div>
      </div>

      {/* пометка о встроенных диаграммах: грид показывает ячейки, не рисунки */}
      {sheet.charts && sheet.charts.length > 0 && (
        <div className="border-b bg-sky-50 px-3 py-1 text-xs text-sky-800">
          📊 На листе {sheet.charts.length > 1 ? 'есть диаграммы' : 'есть диаграмма'}:{' '}
          {sheet.charts.map((t) => `«${t}»`).join(', ')} — показывается только в исходном
          файле (в табличном виде диаграммы не отображаются).
        </div>
      )}

      {/* две панели: оригинал | перевод (растягиваемые столбцы, общая ширина) */}
      <div className="flex">
        <div className="flex w-1/2 min-w-0 border-r">
          <Grid
            rows={sheet.orig}
            label="оригинал"
            sel={sel}
            onSelect={(r, c) => setSel({ r, c })}
            widths={widths}
            onResize={resize}
            scrollRef={leftRef}
            onScroll={() => sync('l')}
          />
        </div>
        <div className="flex w-1/2 min-w-0">
          <Grid
            rows={sheet.ru}
            label={q.data.translated_ready ? 'перевод' : 'перевод · готовится…'}
            sel={sel}
            onSelect={(r, c) => setSel({ r, c })}
            widths={widths}
            onResize={resize}
            scrollRef={rightRef}
            onScroll={() => sync('r')}
          />
        </div>
      </div>

      {/* вкладки листов — сразу под таблицей (как в Excel): слева оригинальные
          названия, справа переведённые. Любая вкладка переключает активный лист. */}
      <div className="flex border-t bg-muted/30">
        {([
          ['l', (s: SheetData) => s.name],
          ['r', (s: SheetData) => s.name_ru || s.name],
        ] as const).map(([side, label]) => (
          <div
            key={side}
            className={'flex w-1/2 min-w-0 items-center gap-0.5 overflow-x-auto px-2 py-1 ' + (side === 'l' ? 'border-r' : '')}
          >
            {sheets.map((s, i) => (
              <button
                key={i}
                onClick={() => setActive(i)}
                title={`${s.name}${s.name_ru && s.name_ru !== s.name ? ` · ${s.name_ru}` : ''} · ${s.total_rows.toLocaleString('ru')}×${s.total_cols}`}
                className={
                  'max-w-[14rem] truncate whitespace-nowrap rounded-t border-x border-t px-3 py-1 text-xs ' +
                  (i === idx
                    ? 'border-border bg-card font-medium text-foreground'
                    : 'border-transparent bg-muted/50 text-muted-foreground hover:bg-muted')
                }
              >
                {label(s)}
              </button>
            ))}
          </div>
        ))}
      </div>
    </div>
  )
}
