import { expect, test, type Route } from '@playwright/test'

interface UploadedRequest {
  filename: string
  body: string
}

async function json(route: Route, body: unknown) {
  await route.fulfill({ contentType: 'application/json', body: JSON.stringify(body) })
}

test('upload applies the selected parser to PDF and images but keeps native Office parsing', async ({
  page,
}) => {
  const uploads: UploadedRequest[] = []
  const documents = new Map<string, Record<string, unknown>>()
  let sequence = 0

  await page.route('**/api/**', async (route) => {
    const request = route.request()
    const path = new URL(request.url()).pathname
    if (path === '/api/config')
      return json(route, { auth_enabled: false, oidc_authority: '', oidc_client_id: '' })
    if (path === '/api/memory' || path === '/api/memory/candidates' || path === '/api/folders')
      return json(route, [])

    if (path === '/api/documents' && request.method() === 'POST') {
      const body = request.postDataBuffer()?.toString('utf8') ?? ''
      const filename =
        ['drawing.pdf', 'scan.png', 'minutes.docx', 'register.xlsx', 'deck.pptx'].find((name) =>
          body.includes(name),
        ) ?? 'unknown'
      sequence += 1
      const id = `00000000-0000-4000-8000-${String(sequence).padStart(12, '0')}`
      const document = {
        id,
        filename,
        status: 'done',
        kind: filename.endsWith('.pdf')
          ? 'pdf_text'
          : filename.endsWith('.png')
            ? 'pdf_scan'
            : filename.slice(filename.lastIndexOf('.') + 1),
        size_bytes: 100,
        page_count: 1,
        segment_count: 1,
        translated_count: 1,
        review_count: 0,
        chunk_count: 1,
        exports: [],
        folder_id: null,
        error: null,
        created_at: '2026-07-28T10:00:00Z',
      }
      uploads.push({ filename, body })
      documents.set(id, document)
      return json(route, document)
    }

    const documentMatch = /^\/api\/documents\/([^/]+)$/.exec(path)
    if (documentMatch) return json(route, documents.get(documentMatch[1]) ?? {})
    if (path === '/api/documents') return json(route, [])
    return json(route, {})
  })

  await page.goto('/upload')
  await page.locator('input[type=file]').first().setInputFiles([
    { name: 'drawing.pdf', mimeType: 'application/pdf', buffer: Buffer.from('%PDF-1.4') },
    { name: 'scan.png', mimeType: 'image/png', buffer: Buffer.from('png') },
    {
      name: 'minutes.docx',
      mimeType: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      buffer: Buffer.from('docx'),
    },
    {
      name: 'register.xlsx',
      mimeType: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      buffer: Buffer.from('xlsx'),
    },
    {
      name: 'deck.pptx',
      mimeType: 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
      buffer: Buffer.from('pptx'),
    },
  ])

  await expect(page.getByText('Парсер PDF и изображений', { exact: true })).toBeVisible()
  const officeInfo = page.getByTestId('office-parser-info')
  await expect(officeInfo).toContainText('Встроенный OOXML-разбор')
  await expect(officeInfo).toContainText('сохраняются абзацы, таблицы, ячейки и слайды')

  await page.getByRole('button', { name: 'Автоматически (по умолчанию)' }).click()
  await page.getByRole('button', { name: 'PaddleOCR-VL 1.6', exact: true }).click()
  await page.getByRole('button', { name: 'Обработать' }).click()

  await expect.poll(() => uploads.length).toBe(5)
  for (const filename of ['drawing.pdf', 'scan.png']) {
    const parserUpload = uploads.find((upload) => upload.filename === filename)
    expect(parserUpload?.body).toContain('name="parser_backend"')
    expect(parserUpload?.body).toContain('paddle_vl')
  }

  for (const filename of ['minutes.docx', 'register.xlsx', 'deck.pptx']) {
    const officeUpload = uploads.find((upload) => upload.filename === filename)
    expect(officeUpload?.body).not.toContain('name="parser_backend"')
  }
})

test('viewer does not offer an OCR/VLM parser for a DOCX document', async ({ page }) => {
  const documentId = '00000000-0000-4000-8000-000000000071'
  await page.route('**/api/**', async (route) => {
    const path = new URL(route.request().url()).pathname
    if (path === '/api/config')
      return json(route, { auth_enabled: false, oidc_authority: '', oidc_client_id: '' })
    if (path === `/api/documents/${documentId}`)
      return json(route, {
        id: documentId,
        filename: 'meeting-topics.docx',
        status: 'done',
        kind: 'docx',
        size_bytes: 100,
        page_count: 1,
        segment_count: 1,
        translated_count: 1,
        review_count: 0,
        chunk_count: 1,
        exports: ['docx'],
        folder_id: null,
        error: null,
        has_view: false,
        has_view_orig: false,
        has_view_ru: false,
        parser_backend: 'native_ooxml',
        source_lang: 'en',
        created_at: '2026-07-28T10:00:00Z',
      })
    if (path === `/api/documents/${documentId}/segments`)
      return json(route, [
        {
          id: '00000000-0000-4000-8000-000000000072',
          idx: 0,
          page_idx: 0,
          kind: 'paragraph',
          heading_level: null,
          source_text: 'Meeting topics',
          translated_text: 'Темы встречи',
          needs_review: false,
          validation: null,
          bbox: null,
          page_size: null,
        },
      ])
    if (path === `/api/documents/${documentId}/translations`) return json(route, [])
    return json(route, {})
  })

  await page.goto(`/view/${documentId}`)
  await expect(page.getByText('meeting-topics.docx · done')).toBeVisible()
  await expect(page.locator('select[title^="Движок парсинга PDF"]')).toHaveCount(0)
})
