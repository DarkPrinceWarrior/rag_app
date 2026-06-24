import { useEffect, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { X, AlertTriangle, Info } from 'lucide-react'
import { Button } from '@/components/ui/button'

/** Базовая модалка: затемнённый бэкдроп, центр, закрытие по Esc и клику вне. */
export function Modal({
  open,
  onClose,
  children,
  className = '',
  labelledBy,
}: {
  open: boolean
  onClose: () => void
  children: ReactNode
  className?: string
  labelledBy?: string
}) {
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    document.addEventListener('keydown', onKey)
    const prev = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.removeEventListener('keydown', onKey)
      document.body.style.overflow = prev
    }
  }, [open, onClose])

  if (!open) return null
  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/45 p-4 backdrop-blur-sm"
      onMouseDown={(e) => e.target === e.currentTarget && onClose()}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby={labelledBy}
        className={
          'w-full max-w-md rounded-xl border bg-card shadow-2xl ' +
          'animate-in fade-in zoom-in-95 duration-150 ' +
          className
        }
      >
        {children}
      </div>
    </div>,
    document.body,
  )
}

/** Строгая модалка-подтверждение: иконка + заголовок + «что произойдёт» + действия. */
export function ConfirmDialog({
  open,
  onClose,
  onConfirm,
  title,
  description,
  confirmLabel = 'Подтвердить',
  cancelLabel = 'Отмена',
  tone = 'default',
  busy = false,
}: {
  open: boolean
  onClose: () => void
  onConfirm: () => void
  title: string
  description: ReactNode
  confirmLabel?: string
  cancelLabel?: string
  tone?: 'default' | 'danger'
  busy?: boolean
}) {
  const danger = tone === 'danger'
  return (
    <Modal open={open} onClose={onClose} labelledBy="confirm-title">
      <div className="p-5">
        <div className="flex items-start gap-3">
          <span
            className={
              'flex h-10 w-10 shrink-0 items-center justify-center rounded-full ' +
              (danger ? 'bg-destructive/10 text-destructive' : 'bg-primary/10 text-primary')
            }
          >
            {danger ? <AlertTriangle className="h-5 w-5" /> : <Info className="h-5 w-5" />}
          </span>
          <div className="min-w-0 flex-1">
            <div className="flex items-start justify-between gap-2">
              <h2 id="confirm-title" className="text-base font-semibold leading-6">
                {title}
              </h2>
              <button
                onClick={onClose}
                aria-label="Закрыть"
                className="-mr-1 -mt-1 rounded-md p-1 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            <div className="mt-1.5 text-sm leading-relaxed text-muted-foreground">{description}</div>
          </div>
        </div>
        <div className="mt-5 flex justify-end gap-2">
          <Button variant="outline" size="sm" onClick={onClose} disabled={busy}>
            {cancelLabel}
          </Button>
          <Button
            size="sm"
            variant={danger ? 'destructive' : 'default'}
            onClick={onConfirm}
            disabled={busy}
          >
            {busy ? 'Выполняется…' : confirmLabel}
          </Button>
        </div>
      </div>
    </Modal>
  )
}
