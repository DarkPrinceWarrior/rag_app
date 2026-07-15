import { readdirSync, statSync } from 'node:fs'
import { join } from 'node:path'

const assetsDir = new URL('../dist/assets/', import.meta.url)
const files = readdirSync(assetsDir).filter((name) => name.endsWith('.js'))
const sizes = new Map(files.map((name) => [name, statSync(join(assetsDir.pathname, name)).size]))

const PDF_WORKER_LIMIT = 1_250_000
const CHUNK_LIMIT = 550_000
const TOTAL_JS_LIMIT = 2_600_000

const failures = []
for (const [name, size] of sizes) {
  const limit = name.startsWith('pdf.worker.') || name.startsWith('pdf.worker.min-')
    ? PDF_WORKER_LIMIT
    : CHUNK_LIMIT
  if (size > limit) failures.push(`${name}: ${size} > ${limit} bytes`)
}

const total = [...sizes.values()].reduce((sum, size) => sum + size, 0)
if (total > TOTAL_JS_LIMIT) failures.push(`all JS chunks: ${total} > ${TOTAL_JS_LIMIT} bytes`)

for (const route of ['chat-', 'upload-', 'view._id-']) {
  if (!files.some((name) => name.startsWith(route))) {
    failures.push(`missing lazy route chunk: ${route}*.js`)
  }
}

if (failures.length) {
  throw new Error(`Bundle budget exceeded:\n${failures.join('\n')}`)
}

console.log(`Bundle budget OK: ${files.length} JS chunks, ${total} bytes total`)
