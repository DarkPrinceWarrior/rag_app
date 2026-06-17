import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css' // шрифты бандлятся Vite локально (без CDN, § 9)
import { cn } from '@/lib/utils'

/** Рендер ответа модели как Markdown (GFM): таблицы, списки, жирный, код.
 *  До этого ответ показывался как plain-text — таблицы ломались, а **жирный**
 *  светил звёздочками. Стили — под наш дизайн (таблица как во вьювере). */
export function Markdown({ content, className }: { content: string; className?: string }) {
  return (
    <div className={cn('text-sm leading-relaxed [&>:first-child]:mt-0 [&>:last-child]:mb-0', className)}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={{
          p: ({ children }) => <p className="my-2">{children}</p>,
          strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
          em: ({ children }) => <em className="italic">{children}</em>,
          ul: ({ children }) => <ul className="my-2 list-disc space-y-1 pl-5">{children}</ul>,
          ol: ({ children }) => <ol className="my-2 list-decimal space-y-1 pl-5">{children}</ol>,
          li: ({ children }) => <li className="marker:text-muted-foreground">{children}</li>,
          h1: ({ children }) => <h1 className="mb-2 mt-3 text-lg font-bold">{children}</h1>,
          h2: ({ children }) => <h2 className="mb-1.5 mt-3 text-base font-semibold">{children}</h2>,
          h3: ({ children }) => <h3 className="mb-1 mt-2 text-sm font-semibold">{children}</h3>,
          a: ({ href, children }) => (
            <a href={href} target="_blank" rel="noreferrer" className="text-primary underline underline-offset-2">
              {children}
            </a>
          ),
          code: ({ className: cls, children }) =>
            cls ? (
              <code className={cn('font-mono', cls)}>{children}</code>
            ) : (
              <code className="rounded bg-muted px-1 py-0.5 font-mono text-[0.85em]">{children}</code>
            ),
          pre: ({ children }) => (
            <pre className="my-2 overflow-x-auto rounded-md bg-muted p-3 text-[0.85em]">{children}</pre>
          ),
          table: ({ children }) => (
            <div className="my-3 overflow-x-auto">
              <table className="border-collapse text-sm">{children}</table>
            </div>
          ),
          thead: ({ children }) => <thead className="bg-muted/60">{children}</thead>,
          th: ({ children }) => (
            <th className="border border-border px-2.5 py-1.5 text-left font-semibold">{children}</th>
          ),
          td: ({ children }) => <td className="border border-border px-2.5 py-1.5 align-top">{children}</td>,
          blockquote: ({ children }) => (
            <blockquote className="my-2 border-l-2 border-border pl-3 text-muted-foreground">{children}</blockquote>
          ),
          hr: () => <hr className="my-3 border-border" />,
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}
