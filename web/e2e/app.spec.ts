import { expect, test, type Page, type Route } from '@playwright/test'

const documentId = '00000000-0000-4000-8000-000000000001'
const segmentId = '00000000-0000-4000-8000-000000000101'

const doneDocument = {
  id: documentId,
  filename: 'Техническое задание.pdf',
  status: 'done',
  kind: 'pdf_text',
  size_bytes: 4096,
  page_count: 3,
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
  target_lang: 'ru',
  created_at: '2026-07-15T10:00:00Z',
}

const citation = {
  n: 1,
  chunk_id: 'chunk-1',
  document_id: documentId,
  filename: doneDocument.filename,
  heading_path: '2.1 Расчётные параметры',
  page_start: 0,
  page_end: 0,
  segment_ids: [segmentId],
  bboxes: [],
}

interface ApiState {
  documentStatus: 'done' | 'error' | 'parsing'
  documentPolls: number
  translatedText: string
  requestedTranslation: string | null
}

interface ChatMockOptions {
  answer?: string
  sessions?: unknown[]
  messages?: Record<string, unknown[]>
  documents?: unknown[]
  folders?: unknown[]
}

async function fulfillJson(route: Route, body: unknown, status = 200) {
  await route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) })
}

function documentFor(state: ApiState) {
  return {
    ...doneDocument,
    status: state.documentStatus,
    translated_count: state.documentStatus === 'done' ? 1 : 0,
    error: state.documentStatus === 'error' ? 'Сбой тестового обработчика' : null,
  }
}

function segmentFor(state: ApiState) {
  return {
    id: segmentId,
    idx: 0,
    page_idx: 0,
    kind: 'paragraph',
    heading_level: null,
    source_text: 'Design pressure is 16 MPa.',
    translated_text: state.translatedText,
    needs_review: false,
    validation: null,
    bbox: null,
    page_size: null,
  }
}

async function mockApi(
  page: Page,
  initialStatus: ApiState['documentStatus'] = 'done',
  chat: ChatMockOptions = {},
) {
  const state: ApiState = {
    documentStatus: initialStatus,
    documentPolls: 0,
    translatedText: 'Расчётное давление — 16 МПа.',
    requestedTranslation: null,
  }

  await page.route('**/api/**', async (route) => {
    const request = route.request()
    const url = new URL(request.url())
    const path = url.pathname
    const method = request.method()

    if (path === '/api/config') {
      return fulfillJson(route, { auth_enabled: false, oidc_authority: '', oidc_client_id: '' })
    }
    if (path === '/api/memory' || path === '/api/memory/candidates') return fulfillJson(route, [])
    if (path === '/api/folders') return fulfillJson(route, chat.folders ?? [])
    const historyMatch = /^\/api\/chat\/sessions\/([^/]+)\/messages$/.exec(path)
    if (historyMatch) return fulfillJson(route, chat.messages?.[historyMatch[1]] ?? [])
    if (path === '/api/chat/sessions') return fulfillJson(route, chat.sessions ?? [])

    if (path === `/api/documents/${documentId}/translations`) {
      if (method === 'POST') {
        const payload = request.postDataJSON() as { target_lang: string }
        state.requestedTranslation = payload.target_lang
        return fulfillJson(route, { target_lang: payload.target_lang, status: 'translating' })
      }
      return fulfillJson(route, [])
    }

    if (path === `/api/documents/${documentId}/retry` && method === 'POST') {
      state.documentStatus = 'parsing'
      return fulfillJson(route, { status: 'queued' })
    }
    if (path === `/api/documents/${documentId}/reexport` && method === 'POST') {
      return fulfillJson(route, { status: 'queued' })
    }
    if (path === `/api/documents/${documentId}/download/pdf`) {
      return route.fulfill({
        status: 200,
        contentType: 'application/pdf',
        headers: { 'Content-Disposition': 'attachment; filename="translated.pdf"' },
        body: '%PDF-1.4 mocked export',
      })
    }
    if (path === `/api/documents/${documentId}/segments`) {
      return fulfillJson(route, [segmentFor(state)])
    }
    if (path === `/api/segments/${segmentId}` && method === 'PATCH') {
      const payload = request.postDataJSON() as { translated_text: string }
      state.translatedText = payload.translated_text
      return fulfillJson(route, segmentFor(state))
    }
    if (path === `/api/segments/${segmentId}/versions`) return fulfillJson(route, [])

    if (path === '/api/documents' && method === 'POST') {
      state.documentStatus = 'parsing'
      state.documentPolls = 0
      return fulfillJson(route, documentFor(state))
    }
    if (path === `/api/documents/${documentId}`) {
      if (state.documentStatus === 'parsing') {
        state.documentPolls += 1
        if (state.documentPolls >= 2) state.documentStatus = 'done'
      }
      return fulfillJson(route, documentFor(state))
    }
    if (path === '/api/documents') return fulfillJson(route, chat.documents ?? [documentFor(state)])

    if (path === '/api/chat' && method === 'POST') {
      return route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: [
          `data: ${JSON.stringify({ type: 'session', session_id: 'chat-1' })}\n\n`,
          `data: ${JSON.stringify({ type: 'delta', text: chat.answer ?? 'Расчётное давление — 16 МПа [1].' })}\n\n`,
          `data: ${JSON.stringify({ type: 'done', citations: [citation] })}\n\n`,
        ].join(''),
      })
    }

    return fulfillJson(route, {})
  })
  return state
}

test('retry updates the library card after a successful enqueue', async ({ page }) => {
  await mockApi(page, 'error')
  await page.goto('/')
  await expect(page.getByText(doneDocument.filename)).toBeVisible()
  await page.getByRole('button', { name: 'Повторить' }).click()
  await expect(page.getByText('Обработка…')).toBeVisible()
})

test('header keeps two library rows and a compact branded navigation elsewhere', async ({ page }) => {
  await mockApi(page)

  await page.goto('/')
  const navigation = page.getByRole('navigation', { name: 'Основная навигация' })
  const navigationRow = page.getByTestId('header-navigation-row')
  const primaryRow = page.getByTestId('header-primary-row')
  await expect(primaryRow).toBeVisible()
  await expect(navigationRow).toBeVisible()
  await expect(page.getByPlaceholder('Найти среди документов')).toBeVisible()
  const profile = page.getByRole('link', { name: 'Профиль' })
  await expect(profile).toBeVisible()
  await expect(profile.locator('svg')).toHaveCount(1)
  await expect(navigationRow.getByRole('link', { name: 'Профиль' })).toHaveCount(0)

  const documentsTab = navigation.getByRole('link', { name: 'Документы', exact: true })
  await expect(documentsTab).toHaveAttribute('aria-current', 'page')
  await expect(documentsTab).toHaveCSS('border-bottom-color', 'rgb(75, 76, 230)')
  await expect(documentsTab).toHaveCSS('border-bottom-width', '3px')
  await expect(documentsTab).toHaveCSS('background-image', /linear-gradient/)

  for (const { path, activeTab } of [
    { path: '/upload', activeTab: 'Загрузка' },
    { path: '/chat', activeTab: 'ИИ-консультант' },
    { path: '/account', activeTab: null },
  ]) {
    await page.goto(path)
    await expect(primaryRow).toHaveCount(0)
    await expect(page.getByPlaceholder('Найти среди документов')).toHaveCount(0)
    const compactProfile = navigationRow.getByRole('link', { name: 'Профиль' })
    await expect(compactProfile).toBeVisible()

    const [rowBox, profileBox] = await Promise.all([navigationRow.boundingBox(), compactProfile.boundingBox()])
    expect(Math.abs((rowBox?.y ?? 0) + (rowBox?.height ?? 0) / 2 - ((profileBox?.y ?? 0) + (profileBox?.height ?? 0) / 2))).toBeLessThanOrEqual(1)
    expect((rowBox?.x ?? 0) + (rowBox?.width ?? 0) - ((profileBox?.x ?? 0) + (profileBox?.width ?? 0))).toBeLessThanOrEqual(33)

    if (activeTab) {
      const activeLink = navigation.getByRole('link', { name: activeTab, exact: true })
      await expect(activeLink).toHaveAttribute('aria-current', 'page')
      await expect(activeLink).toHaveCSS('border-bottom-color', 'rgb(75, 76, 230)')
      await expect(activeLink).toHaveCSS('border-bottom-width', '3px')
      await expect(activeLink).toHaveCSS('background-image', /linear-gradient/)
    } else {
      await expect(navigation.locator('[aria-current="page"]')).toHaveCount(0)
      const extensionDownload = page.getByRole('link', { name: 'Скачать расширение для Google Chrome' })
      await expect(extensionDownload).toHaveAttribute('href', '/downloads/DocRAGenslate-Chrome.zip')
    }

    const inactiveStyles = await navigation.locator('a:not([aria-current="page"])').evaluateAll(
      (links) => links.map((link) => ({
        borderBottomColor: getComputedStyle(link).borderBottomColor,
        backgroundImage: getComputedStyle(link).backgroundImage,
      })),
    )
    expect(inactiveStyles.every((style) => style.borderBottomColor === 'rgba(0, 0, 0, 0)')).toBe(true)
    expect(inactiveStyles.every((style) => style.backgroundImage === 'none')).toBe(true)
  }

  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/account')
  const mobileDownload = page.getByRole('link', { name: 'Скачать расширение для Google Chrome' })
  const mobileDownloadBox = await mobileDownload.boundingBox()
  expect(mobileDownloadBox?.width ?? Infinity).toBeLessThanOrEqual(358)
  expect(mobileDownloadBox?.height ?? 0).toBeGreaterThanOrEqual(44)
  expect(await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)).toBeLessThanOrEqual(1)

  await page.goto('/chat')
  for (const control of [
    navigation.getByRole('link', { name: 'ИИ-консультант', exact: true }),
    navigationRow.getByRole('link', { name: 'Профиль' }),
  ]) {
    const box = await control.boundingBox()
    expect(box?.width ?? 0).toBeGreaterThanOrEqual(44)
    expect(box?.height ?? 0).toBeGreaterThanOrEqual(44)
  }
  const chatViewport = await page.evaluate(() => {
    const app = document.querySelector('#root > div')
    const box = app?.getBoundingClientRect()
    return {
      position: app ? getComputedStyle(app).position : '',
      top: box?.top ?? Infinity,
      bottom: box?.bottom ?? -Infinity,
      viewportHeight: window.innerHeight,
    }
  })
  expect(chatViewport.position).toBe('fixed')
  expect(chatViewport.top).toBe(0)
  expect(chatViewport.bottom).toBeCloseTo(chatViewport.viewportHeight, 0)
})

test('account explains Chrome extension installation and serves the archive', async ({ page }) => {
  await mockApi(page)
  await page.setViewportSize({ width: 1440, height: 1000 })
  await page.goto('/account')

  const card = page.locator('section[aria-labelledby="chrome-extension-title"]')
  await expect(card).toBeVisible()
  await expect(card.getByRole('heading', { name: 'Как установить расширение' })).toBeVisible()
  const installGuide = card.locator('#chrome-install-guide-content')
  const installGuideToggle = card.locator('button[aria-controls="chrome-install-guide-content"]')
  await expect(installGuideToggle).toHaveAttribute('aria-expanded', 'false')
  await expect(installGuideToggle).toHaveAttribute('aria-controls', 'chrome-install-guide-content')
  await expect(installGuideToggle).toHaveAccessibleName('Развернуть инструкцию «Как установить расширение»')
  await expect(installGuide).toBeHidden()

  await installGuideToggle.click()
  await expect(installGuideToggle).toHaveAttribute('aria-expanded', 'true')
  await expect(installGuideToggle).toHaveAccessibleName('Свернуть инструкцию «Как установить расширение»')
  await expect(installGuide).toBeVisible()
  await expect(installGuide.locator('ol > li')).toHaveCount(8)

  await installGuideToggle.focus()
  await page.keyboard.press('Enter')
  await expect(installGuide).toBeHidden()
  await page.keyboard.press('Space')
  await expect(installGuide).toBeVisible()

  expect(await installGuide.locator('ol > li > div > p:first-child').allTextContents()).toEqual([
    'Скачайте расширение',
    'Распакуйте архив',
    'Откройте страницу расширений Chrome',
    'Включите режим разработчика',
    'Установите расширение',
    'Закрепите DocRAGenslate',
    'Войдите под своей учётной записью',
    'Обновите страницу и проверьте перевод',
  ])

  for (const text of [
    'Извлечь всё',
    'Режим разработчика',
    'Загрузить распакованное',
    'значок пазла',
    'нажмите «Войти»',
    'Ctrl+R',
    'не удаляйте и не перемещайте',
    'Выбирайте папку, а не ZIP-архив',
    'булавку рядом с DocRAGenslate',
    'выданные вам логин и пароль',
    'кнопку «Перевести»',
  ]) {
    await expect(card.getByText(text, { exact: false }).first()).toBeVisible()
  }
  await expect(card.getByText('chrome://extensions', { exact: true }).first()).toBeVisible()
  const copyAddress = card.getByRole('button', { name: 'Скопировать адрес страницы расширений Chrome' })
  await expect(copyAddress).toBeVisible()
  await page.context().grantPermissions(['clipboard-read', 'clipboard-write'])
  await copyAddress.click()
  await expect(copyAddress).toContainText(/Скопировано|Адрес выделен/)
  expect(await page.evaluate(() => navigator.clipboard.readText())).toBe('chrome://extensions')

  const download = card.getByRole('link', { name: 'Скачать расширение для Google Chrome' })
  await expect(download).toHaveAttribute('href', '/downloads/DocRAGenslate-Chrome.zip')
  await expect(download).toHaveAttribute('download', '')
  const archive = await page.request.get(new URL('/downloads/DocRAGenslate-Chrome.zip', page.url()).toString())
  expect(archive.ok()).toBe(true)
  expect(archive.headers()['content-type']).toContain('application/zip')
  const archiveBody = await archive.body()
  expect(archiveBody.byteLength).toBeGreaterThan(0)
  expect([...archiveBody.subarray(0, 4)]).toEqual([0x50, 0x4b, 0x03, 0x04])

  const [cardBox, introBox, desktopDownloadBox] = await Promise.all([
    card.boundingBox(),
    card.getByRole('heading', { name: 'Расширение для Google Chrome' }).locator('..').boundingBox(),
    download.boundingBox(),
  ])
  expect(await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)).toBeLessThanOrEqual(1)
  expect((cardBox?.x ?? 0) + (cardBox?.width ?? Infinity)).toBeLessThanOrEqual(1440)
  expect(desktopDownloadBox?.height ?? 0).toBeGreaterThanOrEqual(44)
  expect((introBox?.x ?? 0) + (introBox?.width ?? Infinity)).toBeLessThanOrEqual(desktopDownloadBox?.x ?? 0)

  await page.setViewportSize({ width: 390, height: 844 })
  const downloadBox = await download.boundingBox()
  expect(downloadBox?.width ?? Infinity).toBeLessThanOrEqual(358)
  expect(downloadBox?.height ?? 0).toBeGreaterThanOrEqual(44)
  const copyBox = await copyAddress.boundingBox()
  expect(copyBox?.height ?? 0).toBeGreaterThanOrEqual(44)
  const toggleBox = await installGuideToggle.boundingBox()
  expect(toggleBox?.height ?? 0).toBeGreaterThanOrEqual(44)
  const layout = await page.evaluate(() => ({
    pageOverflow: document.documentElement.scrollWidth - window.innerWidth,
    stepOverflows: [...document.querySelectorAll('section[aria-labelledby="chrome-extension-title"] ol > li')].map(
      (item) => item.scrollWidth - item.clientWidth,
    ),
  }))
  expect(layout.pageOverflow).toBeLessThanOrEqual(1)
  expect(layout.stepOverflows.every((overflow) => overflow <= 1)).toBe(true)

  await installGuideToggle.click()
  await expect(installGuide).toBeHidden()
  expect(await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)).toBeLessThanOrEqual(1)
})

test('document cards stay compact on wide screens', async ({ page }) => {
  await mockApi(page)

  for (const width of [1024, 1920, 2560]) {
    await page.setViewportSize({ width, height: 1000 })
    await page.goto('/')
    const card = page.getByTestId('document-card').first()
    await expect(card).toBeVisible()
    const box = await card.boundingBox()
    expect(box?.width ?? 0).toBeGreaterThanOrEqual(237)
    expect(box?.width ?? Infinity).toBeLessThanOrEqual(281)
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)
    expect(overflow).toBeLessThanOrEqual(1)
  }
})

test('chat history widens its desktop panel without squeezing search controls', async ({ page }) => {
  await mockApi(page)
  await page.setViewportSize({ width: 1440, height: 800 })
  await page.goto('/chat')

  const sidebar = page.getByTestId('chat-sidebar')
  await expect(sidebar).toHaveCSS('width', '320px')
  await page.getByRole('button', { name: 'Мои чаты' }).click()
  await expect(sidebar).toHaveCSS('width', '352px')

  const search = page.getByRole('searchbox', { name: 'Поиск по чатам' })
  const newChat = page.getByRole('button', { name: 'Новый чат' })
  const scrollArea = page.getByTestId('chat-sidebar-scroll')
  await search.focus()
  const [sidebarBox, searchBox, newChatBox, scrollBox] = await Promise.all([
    sidebar.boundingBox(),
    search.boundingBox(),
    newChat.boundingBox(),
    scrollArea.boundingBox(),
  ])
  expect(searchBox?.width ?? 0).toBeGreaterThanOrEqual(303)
  expect(Math.abs((searchBox?.width ?? 0) - (newChatBox?.width ?? 0))).toBeLessThanOrEqual(1)
  expect((searchBox?.x ?? 0) - (sidebarBox?.x ?? 0)).toBeGreaterThanOrEqual(23)
  expect((sidebarBox?.x ?? 0) + (sidebarBox?.width ?? 0) - ((searchBox?.x ?? 0) + (searchBox?.width ?? 0))).toBeGreaterThanOrEqual(23)
  expect((searchBox?.x ?? 0) - (scrollBox?.x ?? 0)).toBeGreaterThanOrEqual(11)
  expect((scrollBox?.x ?? 0) + (scrollBox?.width ?? 0) - ((searchBox?.x ?? 0) + (searchBox?.width ?? 0))).toBeGreaterThanOrEqual(11)
})

test('library sections share one left edge and upload uses the wider workspace', async ({ page }) => {
  await mockApi(page, 'done', {
    folders: [{ id: 'folder-a', name: 'Проект А', documents: 1 }],
  })
  await page.setViewportSize({ width: 1440, height: 1000 })
  await page.goto('/')

  const geometry = await Promise.all([
    page.getByRole('heading', { name: 'Папки', exact: true }).boundingBox(),
    page.getByTestId('folder-card').first().boundingBox(),
    page.getByRole('heading', { name: 'Документы', exact: true }).boundingBox(),
    page.getByTestId('document-card').first().boundingBox(),
  ])
  const leftEdges = geometry.map((box) => box?.x ?? Infinity)
  expect(Math.max(...leftEdges) - Math.min(...leftEdges)).toBeLessThanOrEqual(1)

  await page.goto('/upload')
  const dropzone = await page.getByTestId('upload-dropzone').boundingBox()
  expect(dropzone?.width ?? 0).toBeGreaterThanOrEqual(790)
})

test('upload shows processing before the document reaches the completed state', async ({ page }) => {
  await mockApi(page)
  await page.goto('/upload')
  await page.locator('input[type=file]').first().setInputFiles({
    name: doneDocument.filename,
    mimeType: 'application/pdf',
    buffer: Buffer.from('%PDF-1.4 test'),
  })
  await page.getByRole('button', { name: 'Обработать' }).click()
  await expect(page.getByText('Разбираю документ')).toBeVisible()
  await expect(page.getByText('Успешно')).toBeVisible({ timeout: 5_000 })
})

test('chat renders SSE deltas, citations, and the cited source', async ({ page }) => {
  await mockApi(page)
  await page.goto('/chat')
  await page.getByPlaceholder('Введите запрос').fill('Какое расчётное давление?')
  await page.getByTitle('Отправить').click()
  await expect(page.getByText(/Расчётное давление — 16 МПа/)).toBeVisible()

  await page.getByRole('button', { name: /\[1\] Техническое задание\.pdf/ }).click()
  await expect(page.getByText('Источник [1]')).toBeVisible()
  await expect(page.getByText('Расчётное давление — 16 МПа.', { exact: true })).toBeVisible()
  await expect(page.getByRole('link', { name: 'Открыть во вьювере' })).toBeVisible()
})

test('chat composer and footer fit in the initial desktop viewport', async ({ page }) => {
  await mockApi(page)
  for (const height of [600, 720, 900]) {
    await page.setViewportSize({ width: 1440, height })
    await page.goto('/chat')

    const composer = page.getByPlaceholder('Введите запрос')
    const footer = page.locator('footer')
    await expect(composer).toBeVisible()
    await expect(footer).toBeVisible()
    const geometry = await page.evaluate(() => {
      const input = document.querySelector('textarea[placeholder="Введите запрос"]')?.getBoundingClientRect()
      const pageFooter = document.querySelector('footer')?.getBoundingClientRect()
      const aside = document.querySelector('aside')?.getBoundingClientRect()
      const chatColumn = document.querySelector('aside')?.nextElementSibling?.getBoundingClientRect()
      return {
        viewportHeight: window.innerHeight,
        documentHeight: document.documentElement.scrollHeight,
        documentWidth: document.documentElement.scrollWidth,
        viewportWidth: window.innerWidth,
        inputBottom: input?.bottom ?? Infinity,
        footerTop: pageFooter?.top ?? -Infinity,
        footerBottom: pageFooter?.bottom ?? Infinity,
        asideTop: aside?.top ?? Infinity,
        asideBottom: aside?.bottom ?? -Infinity,
        chatTop: chatColumn?.top ?? -Infinity,
        chatBottom: chatColumn?.bottom ?? Infinity,
      }
    })
    expect(geometry.documentHeight).toBeLessThanOrEqual(geometry.viewportHeight + 1)
    expect(geometry.documentWidth).toBeLessThanOrEqual(geometry.viewportWidth + 1)
    expect(geometry.inputBottom).toBeLessThanOrEqual(geometry.footerTop)
    expect(geometry.footerBottom).toBeLessThanOrEqual(geometry.viewportHeight + 1)
    expect(Math.abs(geometry.asideTop - geometry.chatTop)).toBeLessThanOrEqual(1)
    expect(Math.abs(geometry.asideBottom - geometry.chatBottom)).toBeLessThanOrEqual(1)
  }
})

test('chat expands folders and combines several folders into one document scope', async ({ page }) => {
  const folderA = { id: 'folder-a', name: 'Проект А', documents: 1 }
  const folderB = { id: 'folder-b', name: 'Проект Б', documents: 1 }
  const documentA = {
    ...doneDocument,
    id: '00000000-0000-4000-8000-000000000011',
    filename: 'Документ А.pdf',
    folder_id: folderA.id,
  }
  const documentB = {
    ...doneDocument,
    id: '00000000-0000-4000-8000-000000000012',
    filename: 'Документ Б.pdf',
    folder_id: folderB.id,
  }
  await mockApi(page, 'done', { documents: [documentA, documentB], folders: [folderA, folderB] })
  await page.goto('/chat')

  await expect(page.getByText('Папки', { exact: true })).toBeVisible()
  await expect(page.getByText('Документы без папки', { exact: true })).toBeVisible()
  await page.getByRole('checkbox', { name: 'Выбрать папку Проект А' }).click()
  await page.getByRole('button', { name: 'Раскрыть папку Проект А' }).click()
  await expect(page.getByRole('checkbox', { name: 'Документ А.pdf' })).toBeChecked()
  await page.getByRole('checkbox', { name: 'Выбрать папку Проект Б' }).click()

  const requestPromise = page.waitForRequest(
    (request) => new URL(request.url()).pathname === '/api/chat' && request.method() === 'POST',
  )
  await page.getByPlaceholder('Введите запрос').fill('Сравни требования')
  await page.getByTitle('Отправить').click()
  const request = await requestPromise
  const body = request.postDataJSON() as { folder_ids?: string[]; document_ids?: string[] }
  expect(body.folder_ids).toEqual([folderA.id, folderB.id])
  expect(body.document_ids).toBeUndefined()
})

test('chat renders a persisted or streamed quantity warning outside malformed Markdown', async ({ page }) => {
  const warningSuffix =
    '\n\n<!-- docragenslate:quantity-warning -->\n> ⚠️ **Проверьте числовые значения.** Часть числовых значений ответа не найдена в использованных фрагментах. Сверьте их с первоисточником.'
  const malformedAnswer = 'Расчёт:\n```text\n16 МПа'
  const warnedAnswer = `${malformedAnswer}${warningSuffix}`
  const persistedSession = {
    id: 'persisted-quantity',
    title: 'Проверка чисел',
    document_id: null,
    document_ids: null,
    folder_id: null,
    created_at: '2026-07-16T08:00:00Z',
    updated_at: '2026-07-16T08:00:00Z',
  }
  await mockApi(page, 'done', {
    answer: warnedAnswer,
    sessions: [persistedSession],
    messages: {
      'persisted-quantity': [
        {
          id: 'persisted-answer',
          role: 'assistant',
          content: warnedAnswer,
          citations: [],
          created_at: '2026-07-16T08:00:00Z',
        },
        {
          id: 'marker-in-body',
          role: 'assistant',
          content: `Текст${warningSuffix}\n\nПродолжение ответа.`,
          citations: [],
          created_at: '2026-07-16T08:01:00Z',
        },
      ],
    },
  })

  await page.goto('/chat')
  await page.getByPlaceholder('Введите запрос').fill('Проверь расчёт')
  await page.getByTitle('Отправить').click()

  const streamedWarning = page.getByRole('status', { name: 'Проверьте числовые значения' })
  await expect(streamedWarning).toBeVisible()
  await expect(streamedWarning.getByText('Контроль источников')).toBeVisible()
  await expect(page.locator('code')).toContainText('16 МПа')
  await expect(page.locator('body')).not.toContainText('docragenslate:quantity-warning')

  await page.goto('/chat?sid=persisted-quantity')
  const persistedWarning = page.getByTestId('quantity-warning')
  await expect(persistedWarning).toHaveCount(1)
  await expect(persistedWarning).toContainText('Часть числовых значений ответа не найдена')
  await expect(page.locator('code')).toContainText('16 МПа')
  await expect(page.getByText('Продолжение ответа.')).toBeVisible()
})

test('viewer edits a segment and downloads the translated export', async ({ page }) => {
  const state = await mockApi(page)
  await page.goto(`/view/${documentId}`)

  const translated = page.getByText(state.translatedText, { exact: true })
  await translated.hover()
  await page.getByTitle('Редактировать перевод').click()
  await expect(page).toHaveURL(new RegExp(`/view/${documentId}/segment/${segmentId}$`))

  await page.getByRole('textbox', { name: 'Перевод' }).fill('Расчётное давление — 18 МПа.')
  await page.getByRole('button', { name: 'Сохранить и назад' }).click()
  await expect(page).toHaveURL(new RegExp(`/view/${documentId}$`))
  await expect(page.getByText('Расчётное давление — 18 МПа.', { exact: true })).toBeVisible()

  await page.getByRole('button', { name: 'Скачать перевод' }).click()
  const exportResponse = page.waitForResponse(
    (response) => response.url().endsWith(`/api/documents/${documentId}/download/pdf`) && response.ok(),
  )
  await page.getByRole('button', { name: 'Перевод — PDF (рус.)' }).click()
  await exportResponse
})

test('library can enqueue an additional language translation', async ({ page }) => {
  const state = await mockApi(page)
  await page.goto('/')

  const actions = page.getByRole('button', { name: 'Действия' })
  await actions.scrollIntoViewIfNeeded()
  await expect(actions).toBeVisible()
  await actions.click()
  await page.getByRole('button', { name: /中文/ }).click()
  await expect.poll(() => state.requestedTranslation).toBe('zh')
})
