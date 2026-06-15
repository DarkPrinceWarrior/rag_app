import { useState } from 'react'
import { createFileRoute } from '@tanstack/react-router'
import { api } from '@/lib/api'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'

export const Route = createFileRoute('/translate')({ component: TranslateFragment })

function TranslateFragment() {
  const [src, setSrc] = useState('')
  const [out, setOut] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')

  async function run() {
    if (!src.trim()) return
    setBusy(true)
    setErr('')
    try {
      setOut((await api.translateFragment(src.trim())).text)
    } catch (e) {
      setErr(String(e))
    }
    setBusy(false)
  }

  return (
    <div className="mx-auto max-w-5xl px-4 py-5">
      <p className="mb-3 text-sm text-muted-foreground">
        Перевод вставленного фрагмента EN→RU с учётом терминологии (Qwen3 + глоссарий). Для целого
        документа — загрузите его в библиотеку.
      </p>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        <div>
          <div className="mb-1 text-xs text-muted-foreground">Оригинал (EN)</div>
          <Textarea
            value={src}
            onChange={(e) => setSrc(e.target.value)}
            placeholder="вставьте английский текст…"
            className="min-h-72"
          />
        </div>
        <div>
          <div className="mb-1 text-xs text-muted-foreground">Перевод (RU)</div>
          <Textarea
            value={out}
            readOnly
            placeholder="перевод появится здесь…"
            className="min-h-72 bg-muted/40"
          />
        </div>
      </div>
      <div className="mt-3 flex items-center gap-2">
        <Button onClick={run} disabled={busy || !src.trim()}>
          {busy ? 'Перевожу…' : 'Перевести'}
        </Button>
        {out && (
          <Button variant="outline" onClick={() => navigator.clipboard.writeText(out)}>
            Копировать
          </Button>
        )}
        {err && <span className="text-sm text-destructive">Ошибка: {err}</span>}
      </div>
    </div>
  )
}
