import type { Citation } from '@/lib/api'

/** Цитаты на одну и ту же страницу схлопываем в один чип (объединяя segment_ids),
 *  чтобы в подвале не было «стр. 10, стр. 10, стр. 5, стр. 5». Ключ — имя+страница
 *  (а не document_id): один и тот же файл, загруженный дважды, для пользователя
 *  выглядит дублем — показываем один чип. Общий модуль: чат и ассистент. */
export function dedupeCitations(cites: Citation[]): Citation[] {
  const byPage = new Map<string, Citation>()
  for (const c of cites) {
    const key = `${c.filename}|${c.page_start ?? 'x'}`
    const ex = byPage.get(key)
    if (ex) ex.segment_ids = [...new Set([...(ex.segment_ids ?? []), ...(c.segment_ids ?? [])])]
    else byPage.set(key, { ...c, segment_ids: [...(c.segment_ids ?? [])] })
  }
  return [...byPage.values()]
}
