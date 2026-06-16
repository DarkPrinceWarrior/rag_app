import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { cn } from '@/lib/utils'

/** Минимальное выпадающее меню. Содержимое рендерится порталом в body с
 *  position:fixed — иначе его режет overflow-auto родителя (виртуализированный
 *  список библиотеки). Если снизу мало места — раскрывается вверх. Закрытие по
 *  клику вне, Esc и прокрутке (фиксированная позиция оторвалась бы от триггера).
 *  Без внешних UI-зависимостей. children получает close(). */
export function Menu({
  trigger,
  children,
  triggerClassName,
  title,
}: {
  trigger: ReactNode
  children: (close: () => void) => ReactNode
  triggerClassName?: string
  title?: string
}) {
  const [open, setOpen] = useState(false)
  const [pos, setPos] = useState<{ top?: number; bottom?: number; right: number }>({ right: 0 })
  const btnRef = useRef<HTMLButtonElement>(null)
  const menuRef = useRef<HTMLDivElement>(null)

  useLayoutEffect(() => {
    if (!open || !btnRef.current) return
    const r = btnRef.current.getBoundingClientRect()
    const right = Math.max(8, window.innerWidth - r.right)
    const spaceBelow = window.innerHeight - r.bottom
    // мало места снизу → открыть вверх (привязка к нижней грани триггера)
    if (spaceBelow < 240) setPos({ bottom: window.innerHeight - r.top + 4, right })
    else setPos({ top: r.bottom + 4, right })
  }, [open])

  useEffect(() => {
    if (!open) return
    const onDoc = (e: MouseEvent) => {
      if (btnRef.current?.contains(e.target as Node)) return
      if (menuRef.current?.contains(e.target as Node)) return
      setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setOpen(false)
    const close = () => setOpen(false) // скролл/resize — фикс-позиция оторвётся от триггера
    document.addEventListener('mousedown', onDoc)
    document.addEventListener('keydown', onKey)
    window.addEventListener('scroll', close, true)
    window.addEventListener('resize', close)
    return () => {
      document.removeEventListener('mousedown', onDoc)
      document.removeEventListener('keydown', onKey)
      window.removeEventListener('scroll', close, true)
      window.removeEventListener('resize', close)
    }
  }, [open])

  return (
    <>
      <button
        ref={btnRef}
        type="button"
        title={title}
        onClick={() => setOpen((o) => !o)}
        className={cn(
          'inline-flex h-8 w-8 items-center justify-center rounded-md transition-colors hover:bg-accent',
          open && 'bg-accent',
          triggerClassName,
        )}
      >
        {trigger}
      </button>
      {open &&
        createPortal(
          <div
            ref={menuRef}
            style={{ position: 'fixed', top: pos.top, bottom: pos.bottom, right: pos.right }}
            className="z-50 min-w-44 rounded-md border bg-card p-1 shadow-lg"
          >
            {children(() => setOpen(false))}
          </div>,
          document.body,
        )}
    </>
  )
}

export function MenuItem({
  onClick,
  children,
  icon,
  destructive,
  disabled,
}: {
  onClick: () => void
  children: ReactNode
  icon?: ReactNode
  destructive?: boolean
  disabled?: boolean
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      className={cn(
        'flex w-full items-center gap-2 rounded px-2 py-1.5 text-left text-sm transition-colors disabled:opacity-50',
        destructive ? 'text-destructive hover:bg-destructive/10' : 'hover:bg-accent',
      )}
    >
      {icon && <span className="shrink-0 opacity-80">{icon}</span>}
      {children}
    </button>
  )
}

export function MenuLabel({ children }: { children: ReactNode }) {
  return (
    <div className="px-2 pb-0.5 pt-1 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
      {children}
    </div>
  )
}

export function MenuSeparator() {
  return <div className="my-1 h-px bg-border" />
}
