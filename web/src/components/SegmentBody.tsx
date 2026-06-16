import { cleanMath } from '@/lib/cleanMath'
import type { Segment } from '@/lib/api'

/** Рендер содержимого сегмента: таблица (объединённые ячейки) — настоящей
 *  таблицей, остальное — текстом. Используется в панели источника чата, чтобы
 *  процитированная таблица выглядела как во вьювере, а не «кашей». */
export function SegmentBody({ s }: { s: Segment }) {
  const cells = s.table_cells_ru ?? s.table_cells
  const caption = s.caption_ru ?? s.caption
  if (cells && cells.length) {
    const headerRows = Math.max(1, ...cells[0].map((c) => c.rowspan))
    return (
      <div className="overflow-x-auto">
        {caption && (
          <div className="mb-1 whitespace-pre-line text-xs font-medium text-muted-foreground">
            {cleanMath(caption)}
          </div>
        )}
        <table className="border-collapse text-sm">
          <tbody>
            {cells.map((row, ri) => (
              <tr key={ri} className={ri < headerRows ? 'bg-muted/60 font-medium' : ''}>
                {row.map((c, ci) => (
                  <td
                    key={ci}
                    colSpan={c.colspan > 1 ? c.colspan : undefined}
                    rowSpan={c.rowspan > 1 ? c.rowspan : undefined}
                    className="border border-border px-2.5 py-1 align-top"
                  >
                    {cleanMath(c.text)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    )
  }
  // таблица без структурных ячеек (старый парс): текст вида "a | b | c" по строкам.
  // строки без « | » до таблицы — это подпись/заголовок (рендерим сверху).
  const text = (s.translated_text || s.source_text || '').trim()
  const pipeLines = text.split('\n').filter((l) => l.includes(' | ')).length
  if (s.kind === 'table' && pipeLines >= 1) {
    const head: string[] = []
    const rows: string[][] = []
    for (const l of text.split('\n')) {
      if (!l.trim()) continue
      if (l.includes(' | ')) rows.push(l.split(' | ').map((c) => cleanMath(c)))
      else if (rows.length === 0) head.push(cleanMath(l))
    }
    if (rows.length) {
      const cols = Math.max(...rows.map((r) => r.length))
      return (
        <div className="overflow-x-auto">
          {head.map((h, i) => (
            <div key={i} className="mb-1 whitespace-pre-line text-xs font-medium text-muted-foreground">
              {h}
            </div>
          ))}
          <table className="border-collapse text-sm">
            <tbody>
              {rows.map((r, ri) => (
                <tr key={ri} className={ri === 0 ? 'bg-muted/60 font-medium' : ''}>
                  {Array.from({ length: cols }).map((_, ci) => (
                    <td key={ci} className="border border-border px-2.5 py-1 align-top">
                      {r[ci] ?? ''}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )
    }
  }
  return <div className="whitespace-pre-wrap text-sm text-foreground/90">{text}</div>
}
