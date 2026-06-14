import { Badge } from '@/components/ui/badge'

const DONE = 'done'
const ERROR = 'error'

const LABEL: Record<string, string> = {
  uploaded: 'загружен',
  parsing: 'парсинг',
  parsed: 'разобран',
  translating: 'перевод',
  translated: 'переведён',
  exporting: 'экспорт',
  done: 'готов',
  error: 'ошибка',
}

export function StatusBadge({ status }: { status: string }) {
  const variant = status === DONE ? 'success' : status === ERROR ? 'default' : 'warning'
  return (
    <Badge variant={variant} className={status === ERROR ? 'bg-destructive text-white' : ''}>
      {LABEL[status] ?? status}
    </Badge>
  )
}
