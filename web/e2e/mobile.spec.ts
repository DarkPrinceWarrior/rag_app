import { expect, test, type Locator, type Page, type Route } from '@playwright/test'

const documentId = '00000000-0000-4000-8000-000000000001'
const secondDocumentId = '00000000-0000-4000-8000-000000000002'
const officeDocumentId = '00000000-0000-4000-8000-000000000003'
const segmentId = '00000000-0000-4000-8000-000000000101'

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

const secondDocument = {
  ...document,
  id: secondDocumentId,
  filename: 'Спецификация оборудования.pdf',
}

const officeDocument = {
  ...document,
  id: officeDocumentId,
  filename: 'Техническая спецификация API 6D.docx',
  kind: 'docx',
  exports: ['docx'],
  has_view_orig: true,
  has_view_ru: true,
  parser_backend: 'mineru',
}

const segment = {
  id: segmentId,
  idx: 0,
  page_idx: 0,
  kind: 'paragraph',
  heading_level: null,
  source_text: 'Design pressure is 16 MPa.',
  translated_text: 'Расчётное давление — 16 МПа.',
  needs_review: false,
  validation: null,
  bbox: null,
  page_size: null,
}

async function json(route: Route, body: unknown) {
  await route.fulfill({ contentType: 'application/json', body: JSON.stringify(body) })
}

async function mockApi(page: Page, sessions: unknown[] = []) {
  await page.route('**/api/**', async (route) => {
    const path = new URL(route.request().url()).pathname
    if (path === '/api/config') return json(route, { auth_enabled: false, oidc_authority: '', oidc_client_id: '' })
    if (path === '/api/documents') return json(route, [document, secondDocument, officeDocument])
    if (path === `/api/documents/${documentId}`) return json(route, document)
    if (path === `/api/documents/${documentId}/segments`) return json(route, [segment])
    if (path === `/api/documents/${officeDocumentId}`) return json(route, officeDocument)
    if (path === `/api/documents/${officeDocumentId}/segments`) return json(route, [segment])
    if (path === '/api/folders') return json(route, [])
    if (path === '/api/chat/sessions') return json(route, sessions)
    if (/^\/api\/chat\/sessions\/[^/]+\/messages$/.test(path)) return json(route, [])
    if (/^\/api\/documents\/[^/]+\/translations$/.test(path)) return json(route, [])
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
  await expectTouchTarget(page.getByRole('button', { name: 'Действия' }).first())

  await page.goto('/upload')
  await expectAccessibleViewport(page)
  await expectTouchTarget(page.getByRole('button', { name: 'Выбрать файлы' }))

  await page.goto('/chat')
  await expectAccessibleViewport(page)
  const scopeButton = page.getByRole('button', { name: 'Область чата: Вся библиотека' })
  await expectTouchTarget(scopeButton)
  await scopeButton.click()
  await expect(page.getByRole('heading', { name: 'Область поиска' })).toBeVisible()
  await page.getByRole('dialog').getByText(document.filename, { exact: true }).click()
  await expect(page.getByRole('button', { name: `Область чата: ${document.filename}` })).toBeVisible()

  const sessionsButton = page.getByRole('button', { name: 'История чатов: 0' })
  await expectTouchTarget(sessionsButton)
  await sessionsButton.click()
  await expect(page.getByRole('heading', { name: 'Мои чаты' })).toBeVisible()
  await expectTouchTarget(page.getByRole('button', { name: 'Новый чат' }))
  await page.getByRole('button', { name: 'Закрыть панель' }).click()

  await page.getByPlaceholder('Введите запрос').fill('Проверка')
  await expectTouchTarget(page.getByTitle('Отправить'))
})

test('direct chat link restores a persisted multi-document scope', async ({ page }) => {
  const sessionId = '00000000-0000-4000-8000-000000000010'
  await mockApi(page, [
    {
      id: sessionId,
      title: 'Сравнение требований',
      document_id: null,
      document_ids: [documentId, secondDocumentId],
      folder_id: null,
      created_at: '2026-07-15T10:00:00Z',
      updated_at: '2026-07-15T10:00:00Z',
    },
  ])

  await page.goto(`/chat?sid=${sessionId}`)
  await expect(page.getByRole('button', { name: 'Область чата: 2 документа' })).toBeVisible()
})

test('390px document viewer keeps both source and translation usable', async ({ page }) => {
  await mockApi(page)
  await page.goto(`/view/${documentId}`)

  await expect(page.getByText('Design pressure is 16 MPa.', { exact: true })).toBeVisible()
  await expect(page.getByText('Расчётное давление — 16 МПа.', { exact: true })).toBeVisible()
  await expectAccessibleViewport(page)
  await expectTouchTarget(page.getByRole('button', { name: 'Пересобрать экспорт' }))
  await expectTouchTarget(page.getByRole('button', { name: 'Скачать перевод' }))
})

test('390px Office viewer stacks full-width source and translation panes', async ({ page }) => {
  await mockApi(page)
  await page.goto(`/view/${officeDocumentId}`)

  const sourcePane = page.getByLabel('Оригинал документа')
  const translationPane = page.getByLabel('Перевод документа')
  await expect(sourcePane).toBeVisible()
  await expect(translationPane).toBeVisible()
  const sourceBox = await sourcePane.boundingBox()
  const translationBox = await translationPane.boundingBox()
  expect(sourceBox?.width ?? 0).toBeGreaterThanOrEqual(389)
  expect(translationBox?.width ?? 0).toBeGreaterThanOrEqual(389)
  expect(translationBox?.y ?? 0).toBeGreaterThanOrEqual((sourceBox?.y ?? 0) + (sourceBox?.height ?? 0))
  await expectTouchTarget(page.getByRole('button', { name: 'как в Microsoft' }))
  await expectTouchTarget(page.getByRole('button', { name: 'текст', exact: true }))
  await expectAccessibleViewport(page)
})
