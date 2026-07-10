import type { Document } from '@/lib/api'

export const inProgress = (d: Document) => !['done', 'error'].includes(d.status)

// Бейдж направления перевода: источник определён автоматически, цель всегда RU.
// Русский документ не переводится; "auto" — язык ещё не определён (до перевода).
export const DIRECTION: Record<string, { label: string; cls: string }> = {
  en: { label: 'EN → RU', cls: 'bg-blue-50 text-blue-700' },
  zh: { label: 'ZH → RU', cls: 'bg-rose-50 text-rose-700' },
  ru: { label: 'RU · без перевода', cls: 'bg-muted text-muted-foreground' },
}

export const FORMAT_TONE: Record<string, { badge: string; surface: string }> = {
  DOCX: { badge: 'bg-blue-50 text-[#0a78ff]', surface: 'group-hover:bg-blue-50/60' },
  PDF: { badge: 'bg-red-50 text-[#ff160a]', surface: 'group-hover:bg-red-50/50' },
  PPTX: { badge: 'bg-amber-50 text-[#ff9d0a]', surface: 'group-hover:bg-amber-50/70' },
  XLSX: { badge: 'bg-emerald-50 text-[#008562]', surface: 'group-hover:bg-emerald-50/60' },
  TXT: { badge: 'bg-slate-100 text-slate-700', surface: 'group-hover:bg-slate-100' },
  IMAGE: { badge: 'bg-violet-50 text-violet-700', surface: 'group-hover:bg-violet-50/70' },
}

/** Тип файла по расширению (совпадает с ключами FORMAT_TONE), null если расширения нет. */
export function formatFromFilename(filename: string): string | null {
  const ext = /\.([a-z0-9]+)$/i.exec(filename)?.[1]?.toUpperCase()
  if (ext === 'JPG' || ext === 'JPEG' || ext === 'PNG') return 'IMAGE'
  return ext ?? null
}

export function documentFormat(d: Document): string {
  return formatFromFilename(d.filename) ?? (d.kind.startsWith('pdf') ? 'PDF' : d.kind.toUpperCase())
}

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 Б'
  const units = ['Б', 'КБ', 'МБ', 'ГБ', 'ТБ']
  let value = bytes
  let unit = 0
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024
    unit += 1
  }
  return `${value.toLocaleString('ru-RU', {
    maximumFractionDigits: value >= 10 || unit === 0 ? 0 : 1,
  })} ${units[unit]}`
}

export function formatDate(value: string): string {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return 'дата не указана'
  return new Intl.DateTimeFormat('ru-RU', { day: 'numeric', month: 'long', year: 'numeric' }).format(date)
}

export function formatFileCount(count: number): string {
  const mod10 = count % 10
  const mod100 = count % 100
  const word =
    mod10 === 1 && mod100 !== 11
      ? 'файл'
      : mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)
        ? 'файла'
        : 'файлов'
  return `${count} ${word}`
}

export function formatDocCount(count: number): string {
  const mod10 = count % 10
  const mod100 = count % 100
  const word =
    mod10 === 1 && mod100 !== 11
      ? 'документ'
      : mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)
        ? 'документа'
        : 'документов'
  return `${count} ${word}`
}
