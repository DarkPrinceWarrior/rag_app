import { expect, test, type Locator, type Page, type Route } from '@playwright/test'

const documentId = '00000000-0000-4000-8000-000000000031'
const segmentId = '00000000-0000-4000-8000-000000000331'
const originalTranslation = 'Расчётное давление — 16 МПа.'
const updatedTranslation = 'Расчётное давление — 16,5 МПа по API 6D.'

const document = {
  id: documentId,
  filename: 'Требования API 6D.pdf',
  status: 'done',
  kind: 'pdf_text',
  size_bytes: 4096,
  page_count: 1,
  segment_count: 1,
  translated_count: 1,
  review_count: 0,
  chunk_count: 1,
  exports: ['pdf'],
  folder_id: null,
  error: null,
  has_view: false,
  source_lang: 'en',
  created_at: '2026-07-15T10:00:00Z',
}

const segment = {
  id: segmentId,
  idx: 0,
  page_idx: 0,
  kind: 'paragraph',
  heading_level: null,
  source_text:
    'API6DTECHNICALSPECIFICATIONIDENTIFIERWITHOUTBREAKS1234567890 requires a design pressure of 16 MPa.',
  translated_text: originalTranslation,
  needs_review: false,
  validation: null,
  bbox: null,
  page_size: null,
}

async function json(route: Route, body: unknown) {
  await route.fulfill({ contentType: 'application/json', body: JSON.stringify(body) })
}

async function expectTouchTarget(locator: Locator) {
  await expect(locator).toBeVisible()
  const box = await locator.boundingBox()
  expect(box, 'touch target must have a layout box').not.toBeNull()
  expect(box?.width ?? 0, 'touch target width').toBeGreaterThanOrEqual(44)
  expect(box?.height ?? 0, 'touch target height').toBeGreaterThanOrEqual(44)
}

async function expectNoHorizontalOverflow(page: Page) {
  const metrics = await page.evaluate(() => ({
    viewport: window.innerWidth,
    root: document.documentElement.scrollWidth,
    body: document.body.scrollWidth,
  }))
  expect(metrics.viewport).toBe(390)
  expect(metrics.root, 'documentElement must not overflow horizontally').toBeLessThanOrEqual(391)
  expect(metrics.body, 'body must not overflow horizontally').toBeLessThanOrEqual(391)
}

test('390px segment editor remains accessible and saves back to the viewer', async ({ page }) => {
  let patchBody: Record<string, unknown> | undefined
  await page.route('**/api/**', async (route) => {
    const request = route.request()
    const path = new URL(request.url()).pathname
    if (path === '/api/config')
      return json(route, { auth_enabled: false, oidc_authority: '', oidc_client_id: '' })
    if (path === `/api/documents/${documentId}`) return json(route, document)
    if (path === `/api/documents/${documentId}/segments`) return json(route, [segment])
    if (path === `/api/segments/${segmentId}` && request.method() === 'PATCH') {
      patchBody = request.postDataJSON() as Record<string, unknown>
      return json(route, {
        ...segment,
        translated_text: patchBody.translated_text,
        needs_review: patchBody.needs_review,
      })
    }
    if (/^\/api\/documents\/[^/]+\/translations$/.test(path)) return json(route, [])
    return json(route, {})
  })

  await page.goto(`/view/${documentId}/segment/${segmentId}`)

  await expect(page.getByRole('textbox', { name: 'Перевод' })).toHaveValue(originalTranslation)
  await expectNoHorizontalOverflow(page)
  await expectTouchTarget(page.getByRole('textbox', { name: 'Перевод' }))
  await expectTouchTarget(page.getByRole('button', { name: 'Требует проверки' }))
  await expectTouchTarget(page.getByRole('button', { name: 'Создать кандидата памяти переводов' }))
  await expectTouchTarget(page.getByRole('button', { name: 'Отмена' }))
  await expectTouchTarget(page.getByRole('button', { name: 'Сохранить и назад' }))

  await page.getByRole('textbox', { name: 'Перевод' }).fill(updatedTranslation)
  await page.getByRole('button', { name: 'Требует проверки' }).click()
  await page.getByRole('button', { name: 'Сохранить и назад' }).click()

  await expect(page).toHaveURL(new RegExp(`/view/${documentId}$`))
  expect(patchBody).toEqual({
    translated_text: updatedTranslation,
    needs_review: true,
    memory_candidate: false,
  })
  await expect(page.getByText(updatedTranslation, { exact: true })).toBeVisible()
  await expectNoHorizontalOverflow(page)
})
