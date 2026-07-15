import { expect, test, type Locator, type Page, type Route } from '@playwright/test'

const documentId = '00000000-0000-4000-8000-000000000001'

const document = {
  id: documentId,
  filename: 'Техническое задание.pdf',
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

async function json(route: Route, body: unknown) {
  await route.fulfill({ contentType: 'application/json', body: JSON.stringify(body) })
}

async function mockApi(page: Page) {
  await page.route('**/api/**', async (route) => {
    const path = new URL(route.request().url()).pathname
    if (path === '/api/config') return json(route, { auth_enabled: false, oidc_authority: '', oidc_client_id: '' })
    if (path === '/api/documents') return json(route, [document])
    if (path === '/api/folders' || path === '/api/chat/sessions') return json(route, [])
    if (path === `/api/documents/${documentId}/translations`) return json(route, [])
    return json(route, {})
  })
}

async function expectTouchTarget(locator: Locator) {
  await expect(locator).toBeVisible()
  const box = await locator.boundingBox()
  expect(box, 'touch target must have a layout box').not.toBeNull()
  expect(box?.width ?? 0, 'touch target width').toBeGreaterThanOrEqual(44)
  expect(box?.height ?? 0, 'touch target height').toBeGreaterThanOrEqual(44)
}

async function expectAccessibleViewport(page: Page) {
  const metrics = await page.evaluate(() => ({
    viewport: window.innerWidth,
    root: document.documentElement.scrollWidth,
    body: document.body.scrollWidth,
  }))
  expect(metrics.root, 'documentElement must not overflow horizontally').toBeLessThanOrEqual(metrics.viewport + 1)
  expect(metrics.body, 'body must not overflow horizontally').toBeLessThanOrEqual(metrics.viewport + 1)

  const unnamed = await page
    .locator('button:visible, a[href]:visible, input:visible, select:visible, textarea:visible')
    .evaluateAll((elements) =>
      elements
        .filter((element) => {
          const labelled = element.getAttribute('aria-label')?.trim()
          const labelledBy = element.getAttribute('aria-labelledby')?.trim()
          const title = element.getAttribute('title')?.trim()
          const text = element.textContent?.trim()
          const placeholder = element.getAttribute('placeholder')?.trim()
          const associatedLabel =
            element.id && document.querySelector(`label[for="${CSS.escape(element.id)}"]`)?.textContent?.trim()
          return !labelled && !labelledBy && !title && !text && !placeholder && !associatedLabel
        })
        .map((element) => element.outerHTML.slice(0, 180)),
    )
  expect(unnamed, 'all visible controls need an accessible name').toEqual([])
}

test('390px core flows have no overflow, named controls, and usable touch targets', async ({ page }) => {
  await mockApi(page)

  await page.goto('/')
  await expectAccessibleViewport(page)
  await expectTouchTarget(page.getByRole('link', { name: 'Документы' }))
  await expectTouchTarget(page.getByRole('link', { name: 'Профиль' }))
  await expectTouchTarget(page.getByRole('button', { name: 'Загрузить ещё' }))
  await expectTouchTarget(page.getByRole('button', { name: 'Действия' }))

  await page.goto('/upload')
  await expectAccessibleViewport(page)
  await expectTouchTarget(page.getByRole('button', { name: 'Выбрать файлы' }))

  await page.goto('/chat')
  await expectAccessibleViewport(page)
  await page.getByPlaceholder('Введите запрос').fill('Проверка')
  await expectTouchTarget(page.getByTitle('Отправить'))
})
