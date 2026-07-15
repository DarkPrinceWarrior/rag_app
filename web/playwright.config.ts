import { defineConfig, devices } from '@playwright/test'

// The workstation exports an HTTP(S) proxy. Playwright's webServer readiness
// probe honours it unless loopback is explicitly excluded, so Vite can be
// ready while the runner waits forever on a proxied 503 response.
const loopbackNoProxy = new Set(
  `${process.env.NO_PROXY ?? ''},${process.env.no_proxy ?? ''},127.0.0.1,localhost,::1`
    .split(',')
    .map((entry) => entry.trim())
    .filter(Boolean),
)
process.env.NO_PROXY = [...loopbackNoProxy].join(',')
process.env.no_proxy = process.env.NO_PROXY

const systemChrome = '/opt/google/chrome/chrome'

export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: process.env.CI ? 'github' : 'line',
  use: {
    baseURL: 'http://127.0.0.1:4174',
    launchOptions: { executablePath: systemChrome },
    locale: 'ru-RU',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [
    {
      name: 'desktop-chrome',
      testIgnore: /mobile\.spec\.ts/,
      use: { ...devices['Desktop Chrome'], launchOptions: { executablePath: systemChrome } },
    },
    {
      name: 'mobile-chrome',
      testMatch: /mobile\.spec\.ts/,
      use: {
        ...devices['Pixel 5'],
        viewport: { width: 390, height: 844 },
        launchOptions: { executablePath: systemChrome },
      },
    },
  ],
  webServer: {
    command: 'pnpm exec vite --host 127.0.0.1 --port 4174 --strictPort',
    url: 'http://127.0.0.1:4174/',
    env: { ...process.env, NO_PROXY: process.env.NO_PROXY, no_proxy: process.env.no_proxy },
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
})
