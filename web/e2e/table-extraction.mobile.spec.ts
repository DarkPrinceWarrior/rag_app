import { expect, test, type Locator, type Page, type Route } from '@playwright/test'

const documentId = '00000000-0000-4000-8000-000000000041'
const query = 'Собери таблицу параметров оборудования'
const retryQuery = 'Повтори извлечение таблицы'

const document = {
  id: documentId,
  filename: 'Спецификация оборудования с длинным названием.pdf',
  status: 'done',
  kind: 'pdf_text',
  size_bytes: 4096,
  page_count: 12,
  segment_count: 20,
  translated_count: 20,
  review_count: 0,
  chunk_count: 20,
  exports: ['pdf'],
  folder_id: null,
  error: null,
  has_view: false,
  source_lang: 'en',
  created_at: '2026-07-15T10:00:00Z',
}

const extractedTable = {
  title: 'Сводная таблица технических параметров',
  columns: [
    'Идентификатор оборудования без пробелов',
    'Расчётное давление',
    'Нормативный документ',
  ],
  rows: [
    ['API6DTECHNICALSPECIFICATIONIDENTIFIERWITHOUTBREAKS1234567890', '16,5 МПа', 'API 6D'],
    ['PIPELINEVALVEIDENTIFIERWITHOUTBREAKS0987654321', '10 МПа', 'ГОСТ 9544'],
  ],
  sources: [
    {
      n: 1,
      document_id: documentId,
      filename: document.filename,
      heading_path: 'Технические параметры / Расчётные значения',
      page: 7,
      segment_ids: ['00000000-0000-4000-8000-000000000441'],
    },
  ],
}

async function json(route: Route, body: unknown, status = 200) {
  await route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) })
}

async function expectTouchTarget(locator: Locator) {
  await expect(locator).toBeVisible()
  const box = await locator.boundingBox()
  expect(box, 'touch target must have a layout box').not.toBeNull()
  expect(box?.width ?? 0, 'touch target width').toBeGreaterThanOrEqual(44)
  expect(box?.height ?? 0, 'touch target height').toBeGreaterThanOrEqual(44)
}

async function expectNoHorizontalPageOverflow(page: Page) {
  const metrics = await page.evaluate(() => ({
    viewport: window.innerWidth,
    root: document.documentElement.scrollWidth,
    body: document.body.scrollWidth,
  }))
  expect(metrics.viewport).toBe(390)
  expect(metrics.root, 'documentElement must not overflow horizontally').toBeLessThanOrEqual(391)
  expect(metrics.body, 'body must not overflow horizontally').toBeLessThanOrEqual(391)
}

test('390px table extraction shows progress, result/XLSX, and a recoverable error', async ({ page }) => {
  let extractCalls = 0
  let firstExtractBody: Record<string, unknown> | undefined
  let xlsxBody: Record<string, unknown> | undefined
  await page.route('**/api/**', async (route) => {
    const request = route.request()
    const path = new URL(request.url()).pathname
    if (path === '/api/config')
      return json(route, { auth_enabled: false, oidc_authority: '', oidc_client_id: '' })
    if (path === '/api/documents') return json(route, [document])
    if (path === '/api/folders') return json(route, [])
    if (path === '/api/chat/sessions') return json(route, [])
    if (path === '/api/extract/table') {
      extractCalls += 1
      if (extractCalls === 1) {
        firstExtractBody = request.postDataJSON() as Record<string, unknown>
        await new Promise((resolve) => setTimeout(resolve, 250))
        return json(route, extractedTable)
      }
      return json(route, { detail: 'Не удалось собрать таблицу — уточните запрос' }, 422)
    }
    if (path === '/api/extract/xlsx') {
      xlsxBody = request.postDataJSON() as Record<string, unknown>
      return route.fulfill({
        status: 200,
        contentType: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        body: 'mock-xlsx',
      })
    }
    return json(route, {})
  })

  await page.goto('/chat')
  const composer = page.getByPlaceholder('Введите запрос')
  await composer.fill(query)
  const tableButton = page.getByRole('button', { name: 'Таблица' })
  await expectTouchTarget(tableButton)
  await tableButton.click()

  await expect(page.getByText('⊞ извлекаю таблицу из источников…', { exact: true })).toBeVisible()
  await expect(tableButton).toBeDisabled()
  await expect(page.getByText(extractedTable.title, { exact: true })).toBeVisible()
  expect(firstExtractBody).toEqual({ query })

  await expect(page.getByRole('columnheader', { name: extractedTable.columns[0] })).toBeVisible()
  await expect(page.getByRole('cell', { name: extractedTable.rows[0][0] })).toBeVisible()
  await expectNoHorizontalPageOverflow(page)

  const xlsxButton = page.getByRole('button', { name: 'XLSX' })
  await expectTouchTarget(xlsxButton)
  const downloadPromise = page.waitForEvent('download')
  await xlsxButton.click()
  const download = await downloadPromise
  expect(download.suggestedFilename()).toBe(`${extractedTable.title}.xlsx`)
  expect(xlsxBody).toEqual(extractedTable)

  const followupComposer = page.getByPlaceholder('Спросите ещё что-нибудь или соберите таблицу…')
  await followupComposer.fill(retryQuery)
  await tableButton.click()
  await expect(page.getByText('Ошибка: Error: 422: Не удалось собрать таблицу — уточните запрос')).toBeVisible()
  await expectNoHorizontalPageOverflow(page)
  await expect(followupComposer).toHaveValue(retryQuery)
  await expect(tableButton).toBeEnabled()
})
