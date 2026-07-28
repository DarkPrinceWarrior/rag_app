import { fileURLToPath } from 'node:url'
import { expect, test, type Locator, type Page, type Route } from '@playwright/test'

const documentId = '00000000-0000-4000-8000-000000000031'
const segmentId = '00000000-0000-4000-8000-000000000331'
const originalTranslation = 'Расчётное давление — 16 МПа.'
const updatedTranslation = 'Расчётное давление — 16,5 МПа по API 6D.'
const pdfFixture = fileURLToPath(
  new URL('../../docs/DocRAGenslate_Руководство_пользователя.pdf', import.meta.url),
)

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

const docxDocument = {
  ...document,
  filename: 'meeting-topics.docx',
  kind: 'docx',
  exports: ['docx'],
  has_view: true,
  has_view_orig: true,
  has_view_ru: true,
  parser_backend: 'native_ooxml',
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

const multiPageSegments = [
  {
    ...segment,
    id: '00000000-0000-4000-8000-000000000381',
    idx: 0,
    source_text: 'Page one selected segment.',
    translated_text: 'Выделенный сегмент первой страницы.',
  },
  {
    ...segment,
    id: '00000000-0000-4000-8000-000000000382',
    idx: 1,
    page_idx: 1,
    source_text: 'Page two first segment.',
    translated_text: 'Первый сегмент второй страницы.',
  },
  {
    ...segment,
    id: '00000000-0000-4000-8000-000000000383',
    idx: 2,
    page_idx: 1,
    source_text: 'Page two second segment.',
    translated_text: 'Второй сегмент второй страницы.',
  },
]

const modeSwitchSegments = [
  multiPageSegments[0],
  {
    ...multiPageSegments[1],
    page_idx: 0,
    source_text: 'Another segment on the same page.',
    translated_text: 'Другой сегмент на той же странице.',
  },
]

const tableSegments = [
  {
    ...segment,
    id: '00000000-0000-4000-8000-000000000332',
    idx: 0,
    source_text: 'Parameter',
    translated_text: 'Параметр',
    location: { t: 0, r: 0, c: 0 },
    table_size: [1, 3],
  },
  {
    ...segment,
    id: '00000000-0000-4000-8000-000000000333',
    idx: 1,
    source_text: 'Value',
    translated_text: 'Значение',
    location: { t: 0, r: 0, c: 1 },
    table_size: [1, 3],
  },
]

const partialTableSegments = [
  {
    ...segment,
    id: '00000000-0000-4000-8000-000000000334',
    idx: 0,
    source_text: 'Third row',
    translated_text: 'Третья строка',
    location: { t: 0, r: 2, c: 0 },
    table_size: [5, 3],
  },
  {
    ...segment,
    id: '00000000-0000-4000-8000-000000000335',
    idx: 1,
    source_text: 'Third value',
    translated_text: 'Третье значение',
    location: { t: 0, r: 2, c: 1 },
    table_size: [5, 3],
  },
  {
    ...segment,
    id: '00000000-0000-4000-8000-000000000336',
    idx: 2,
    source_text: 'Fourth row',
    translated_text: 'Четвёртая строка',
    location: { t: 0, r: 3, c: 0 },
    table_size: [5, 3],
  },
  {
    ...segment,
    id: '00000000-0000-4000-8000-000000000337',
    idx: 3,
    source_text: 'Fourth value',
    translated_text: 'Четвёртое значение',
    location: { t: 0, r: 3, c: 1 },
    table_size: [5, 3],
  },
]

async function json(route: Route, body: unknown) {
  await route.fulfill({ contentType: 'application/json', body: JSON.stringify(body) })
}

async function openDocxTextViewer(page: Page, segments: typeof tableSegments) {
  await page.route('**/api/**', async (route) => {
    const path = new URL(route.request().url()).pathname
    if (path === '/api/config')
      return json(route, { auth_enabled: false, oidc_authority: '', oidc_client_id: '' })
    if (path === `/api/documents/${documentId}`)
      return json(route, { ...docxDocument, segment_count: segments.length })
    if (path === `/api/documents/${documentId}/segments`) return json(route, segments)
    if (/^\/api\/documents\/[^/]+\/translations$/.test(path)) return json(route, [])
    return json(route, {})
  })

  await page.goto(`/view/${documentId}`)
  await page.getByRole('button', { name: 'текст', exact: true }).click()
}

async function openViewerWithSegments(
  page: Page,
  mockedDocument: Record<string, unknown>,
  segments: Array<Record<string, unknown>>,
) {
  await page.route('**/api/**', async (route) => {
    const path = new URL(route.request().url()).pathname
    if (path === '/api/config')
      return json(route, { auth_enabled: false, oidc_authority: '', oidc_client_id: '' })
    if (path === `/api/documents/${documentId}`) return json(route, mockedDocument)
    if (path === `/api/documents/${documentId}/segments`) return json(route, segments)
    if (path.startsWith(`/api/documents/${documentId}/download/`))
      return route.fulfill({ contentType: 'application/pdf', path: pdfFixture })
    if (/^\/api\/documents\/[^/]+\/translations$/.test(path)) return json(route, [])
    return json(route, {})
  })

  await page.goto(`/view/${documentId}`)
}

async function expectSegmentsUnselected(page: Page, ids: string[]) {
  for (const id of ids) {
    const blocks = page.locator(`[data-seg="${id}"]`)
    await expect(blocks).toHaveCount(2)
    for (let side = 0; side < 2; side += 1) {
      await expect(blocks.nth(side)).not.toHaveClass(
        /opacity-30|border-\[#222226\]|border-\[#4b4ce6\]/,
      )
    }
  }
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

test('text viewer toggles a segment from either column and clears it with Escape', async ({ page }) => {
  await page.route('**/api/**', async (route) => {
    const path = new URL(route.request().url()).pathname
    if (path === '/api/config')
      return json(route, { auth_enabled: false, oidc_authority: '', oidc_client_id: '' })
    if (path === `/api/documents/${documentId}`) return json(route, document)
    if (path === `/api/documents/${documentId}/segments`) return json(route, [segment])
    if (path === `/api/segments/${segmentId}/versions`) return json(route, [])
    if (/^\/api\/documents\/[^/]+\/translations$/.test(path)) return json(route, [])
    return json(route, {})
  })

  await page.goto(`/view/${documentId}`)

  const segmentBlocks = page.locator(`[data-seg="${segmentId}"]`)
  const original = segmentBlocks.filter({ hasText: 'requires a design pressure' })
  const translation = segmentBlocks.filter({ hasText: originalTranslation })

  await expect(original).toBeVisible()
  await expect(translation).toBeVisible()

  await original.click()
  await expect(original).toHaveClass(/border-\[#222226\]/)
  await expect(translation).toHaveClass(/border-\[#4b4ce6\]/)

  await translation.click()
  await expect(original).not.toHaveClass(/border-\[#222226\]/)
  await expect(translation).not.toHaveClass(/border-\[#4b4ce6\]/)

  await translation.click()
  await expect(translation).toHaveClass(/border-\[#4b4ce6\]/)
  await translation.click()
  await expect(translation).not.toHaveClass(/border-\[#4b4ce6\]/)

  await original.click()
  await page.keyboard.press('Escape')
  await expect(original).not.toHaveClass(/border-\[#222226\]/)
  await expect(translation).not.toHaveClass(/border-\[#4b4ce6\]/)

  await translation.click()
  await translation.getByTitle('История правок перевода').click()
  await expect(page.getByText('История правок (0)')).toBeVisible()
  await expect(translation).toHaveClass(/border-\[#4b4ce6\]/)
  await page.getByRole('button', { name: 'Закрыть историю правок' }).click()

  await translation.getByTitle('Редактировать перевод').click()
  await expect(page).toHaveURL(new RegExp(`/view/${documentId}/segment/${segmentId}$`))
})

test('text viewer clears selection before showing the next page', async ({ page }) => {
  await openViewerWithSegments(
    page,
    {
      ...document,
      page_count: 2,
      segment_count: multiPageSegments.length,
      translated_count: multiPageSegments.length,
    },
    multiPageSegments,
  )

  const selected = page.locator(`[data-seg="${multiPageSegments[0].id}"]`).first()
  await expect(selected).toBeVisible()
  await selected.click()
  await expect(selected).toHaveClass(/border-\[#222226\]/)

  const pager = page.getByText('стр. 1 / 2', { exact: true }).locator('..')
  await pager.getByRole('button', { name: '→', exact: true }).click()
  await expect(page.getByText('стр. 2 / 2', { exact: true })).toBeVisible()

  await expectSegmentsUnselected(
    page,
    multiPageSegments.slice(1).map((item) => item.id),
  )
})

test('PDF text and document modes clear segment selection when switching', async ({ page }) => {
  await openViewerWithSegments(
    page,
    {
      ...document,
      segment_count: modeSwitchSegments.length,
      translated_count: modeSwitchSegments.length,
    },
    modeSwitchSegments,
  )

  const selected = page.locator(`[data-seg="${modeSwitchSegments[0].id}"]`).first()
  const dimmed = page.locator(`[data-seg="${modeSwitchSegments[1].id}"]`).first()
  await expect(selected).toBeVisible()
  await selected.click()
  await expect(dimmed).toHaveClass(/opacity-30/)

  await page.getByRole('button', { name: 'документ (PDF)', exact: true }).click()
  await expect(selected).toHaveCount(0)
  await page.getByRole('button', { name: 'текст', exact: true }).click()

  await expectSegmentsUnselected(
    page,
    modeSwitchSegments.map((item) => item.id),
  )
})

test('DOCX text and Microsoft modes clear segment selection when switching', async ({ page }) => {
  await openViewerWithSegments(
    page,
    {
      ...docxDocument,
      segment_count: modeSwitchSegments.length,
      translated_count: modeSwitchSegments.length,
    },
    modeSwitchSegments,
  )

  await page.getByRole('button', { name: 'текст', exact: true }).click()
  const selected = page.locator(`[data-seg="${modeSwitchSegments[0].id}"]`).first()
  const dimmed = page.locator(`[data-seg="${modeSwitchSegments[1].id}"]`).first()
  await selected.click()
  await expect(dimmed).toHaveClass(/opacity-30/)

  await page.getByRole('button', { name: 'как в Microsoft', exact: true }).click()
  await expect(selected).toHaveCount(0)
  await page.getByRole('button', { name: 'текст', exact: true }).click()

  await expectSegmentsUnselected(
    page,
    modeSwitchSegments.map((item) => item.id),
  )
})

test('text viewer keeps DOCX table columns that are completely empty', async ({ page }) => {
  await openDocxTextViewer(page, tableSegments)

  const originalTable = page.getByLabel('Оригинал документа').locator('article table')
  await expect(originalTable).toBeVisible()
  await expect(originalTable.locator('tr')).toHaveCount(1)
  await expect(originalTable.locator('tr').first()).toHaveClass(/bg-muted\/50/)
  await expect(originalTable.locator('tr').first()).toHaveClass(/font-medium/)
  await expect(originalTable.locator('td')).toHaveCount(3)
  await expect(originalTable.locator('td').nth(2)).toBeEmpty()

  const translatedTable = page.getByLabel('Перевод документа').locator('article table')
  await expect(translatedTable.locator('tr')).toHaveCount(1)
  await expect(translatedTable.locator('td')).toHaveCount(3)
  await expect(translatedTable.locator('td').nth(2)).toBeEmpty()
})

test('DOCX table page fragment renders only its actual rows and preserves all columns', async ({
  page,
}) => {
  await openDocxTextViewer(page, partialTableSegments)

  const table = page.getByLabel('Оригинал документа').locator('article table')
  const rows = table.locator('tr')
  await expect(rows).toHaveCount(2)
  await expect(rows.first()).not.toHaveClass(/bg-muted\/50/)
  await expect(rows.first()).not.toHaveClass(/font-medium/)
  await expect(rows.nth(0).locator('td')).toHaveCount(3)
  await expect(rows.nth(1).locator('td')).toHaveCount(3)
  await expect(rows.nth(0).locator('td').first()).toHaveText('Third row')
  await expect(rows.nth(1).locator('td').first()).toHaveText('Fourth row')
  await expect(rows.nth(0).locator('td').nth(2)).toBeEmpty()
  await expect(rows.nth(1).locator('td').nth(2)).toBeEmpty()
})
