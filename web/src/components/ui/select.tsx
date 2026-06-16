import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { ChevronDown, Check } from 'lucide-react'
import { cn } from '@/lib/utils'

export interface SelectOption {
  value: string
  label: string
}

/** Единый селектор приложения: триггер + выпадающий список в стиле меню ⋮
 *  (портал в body, position:fixed, авто-разворот вверх). Заменяет нативный
 *  <select>, чтобы выпадашка везде выглядела одинаково и фирменно. */
export function Select({
  value,
  onChange,
  options,
  placeholder,
  icon,
  className,
  align = 'start',
}: {
  value: string
  onChange: (v: string) => void
  options: SelectOption[]
  placeholder?: string
  icon?: ReactNode
  className?: string
  align?: 'start' | 'end'
}) {
  const [open, setOpen] = useState(false)
  const [pos, setPos] = useState<{
    top?: number
    bottom?: number
    left?: number
    right?: number
    width: number
  }>({ width: 0 })
  const btnRef = useRef<HTMLButtonElement>(null)
  const menuRef = useRef<HTMLDivElement>(null)
  const current = options.find((o) => o.value === value)

  useLayoutEffect(() => {
    if (!open || !btnRef.current) return
    const r = btnRef.current.getBoundingClientRect()
    const base = {
      width: r.width,
      left: align === 'start' ? r.left : undefined,
      right: align === 'end' ? window.innerWidth - r.right : undefined,
    }
    const spaceBelow = window.innerHeight - r.bottom
    if (spaceBelow < 280) setPos({ ...base, bottom: window.innerHeight - r.top + 4 })
    else setPos({ ...base, top: r.bottom + 4 })
  }, [open, align])

  useEffect(() => {
    if (!open) return
    const onDoc = (e: MouseEvent) => {
      if (btnRef.current?.contains(e.target as Node)) return
      if (menuRef.current?.contains(e.target as Node)) return
      setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setOpen(false)
    const close = () => setOpen(false)
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
        onClick={() => setOpen((o) => !o)}
        className={cn(
          'flex items-center gap-2 rounded-lg border bg-card px-3 py-1.5 text-sm transition-colors hover:bg-accent',
          className,
        )}
      >
        {icon && <span className="shrink-0 text-muted-foreground">{icon}</span>}
        <span className="min-w-0 flex-1 truncate text-left">
          {current?.label ?? placeholder ?? '—'}
        </span>
        <ChevronDown
          className={cn('h-4 w-4 shrink-0 text-muted-foreground transition-transform', open && 'rotate-180')}
        />
      </button>
      {open &&
        createPortal(
          <div
            ref={menuRef}
            style={{
              position: 'fixed',
              top: pos.top,
              bottom: pos.bottom,
              left: pos.left,
              right: pos.right,
              minWidth: pos.width,
            }}
            className="z-50 max-h-72 max-w-[min(92vw,420px)] overflow-auto rounded-md border bg-card p-1 shadow-lg"
          >
            {options.map((o) => (
              <button
                key={o.value}
                type="button"
                title={o.label}
                onClick={() => {
                  onChange(o.value)
                  setOpen(false)
                }}
                className={cn(
                  'flex w-full items-center gap-2 rounded px-2 py-1.5 text-left text-sm transition-colors hover:bg-accent',
                  o.value === value && 'bg-accent/60',
                )}
              >
                <Check className={cn('h-4 w-4 shrink-0', o.value === value ? 'opacity-100' : 'opacity-0')} />
                <span className="truncate">{o.label}</span>
              </button>
            ))}
          </div>,
          document.body,
        )}
    </>
  )
}
