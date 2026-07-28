import { fileURLToPath } from 'node:url'
import { expect, test, type Route } from '@playwright/test'

const documentId = '00000000-0000-4000-8000-000000000041'
const segmentId = '00000000-0000-4000-8000-000000000441'
const pdfFixture = fileURLToPath(
  new URL('../../docs/DocRAGenslate_Руководство_пользователя.pdf', import.meta.url),
)

const document = {
  id: documentId,
  filename: 'Cross-highlight fixture.pdf',
  status: 'done',
  kind: 'pdf_scan',
  size_bytes: 4096,
  page_count: 15,
  segment_count: 1,
  translated_count: 1,
  review_count: 0,
  chunk_count: 1,
  exports: ['pdf'],
  folder_id: null,
  error: null,
  has_view: false,
  has_view_orig: false,
  has_view_ru: false,
  parser_backend: 'mineru',
  source_lang: 'en',
  created_at: '2026-07-28T10:00:00Z',
}

const segment = {
  id: segmentId,
  idx: 0,
  page_idx: 0,
  kind: 'paragraph',
  heading_level: null,
  source_text: 'MEETING TOPICS',
  translated_text: 'ТЕМЫ ВСТРЕЧИ',
  needs_review: false,
  validation: null,
  bbox: null,
  page_size: null,
  loc_left: {
    page: 0,
    bbox: [100, 100, 300, 180],
    pagesize: [921.6, 518.4],
  },
  loc_right: {
    page: 1,
    bbox: [120, 110, 320, 190],
    pagesize: [921.6, 518.4],
  },
}

async function json(route: Route, body: unknown) {
  await route.fulfill({ contentType: 'application/json', body: JSON.stringify(body) })
}

test('PDF cross-highlight is cleared by every supported dismissal action', async ({ page }) => {
  await page.route('**/api/**', async (route) => {
    const path = new URL(route.request().url()).pathname

    if (path === '/api/config')
      return json(route, { auth_enabled: false, oidc_authority: '', oidc_client_id: '' })
    if (path === `/api/documents/${documentId}`) return json(route, document)
    if (path === `/api/documents/${documentId}/segments`) return json(route, [segment])
    if (
      path === `/api/documents/${documentId}/download/original` ||
      path === `/api/documents/${documentId}/download/pdf`
    )
      return route.fulfill({ contentType: 'application/pdf', path: pdfFixture })
    if (/^\/api\/documents\/[^/]+\/translations$/.test(path)) return json(route, [])
    return json(route, {})
  })

  await page.goto(`/view/${documentId}`)

  const originalPane = page.getByText('оригинал', { exact: true }).locator('..').locator('..')
  const translatedPane = page
    .getByText('перевод · документ — кликните фрагмент, чтобы найти его в оригинале', {
      exact: true,
    })
    .locator('..')
    .locator('..')
  const sourceRegion = originalPane.getByTitle('Найти этот фрагмент на другой стороне')
  const targetHighlight = translatedPane.locator('.pointer-events-none.border-2.border-primary')

  await expect(sourceRegion).toBeVisible()
  await expect(originalPane.getByText('стр. 1 / 15')).toBeVisible()
  await expect(translatedPane.getByText('стр. 1 / 15')).toBeVisible()

  const selectSource = async () => {
    await sourceRegion.click()
    await expect(translatedPane.getByText('стр. 2 / 15')).toBeVisible()
    await expect(targetHighlight).toBeVisible()
  }

  await selectSource()
  await sourceRegion.click()
  await expect(targetHighlight).toHaveCount(0)

  await selectSource()
  await page.keyboard.press('Escape')
  await expect(targetHighlight).toHaveCount(0)

  await selectSource()
  await translatedPane.locator('canvas').click({ position: { x: 5, y: 5 } })
  await expect(targetHighlight).toHaveCount(0)

  await selectSource()
  await translatedPane.getByRole('button', { name: '→' }).click()
  await expect(translatedPane.getByText('стр. 3 / 15')).toBeVisible()
  await expect(targetHighlight).toHaveCount(0)
})
