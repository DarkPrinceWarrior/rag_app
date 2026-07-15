/** Deterministic local provider used only to prove the Promptfoo harness offline. */
export default class OfflineSafeMockProvider {
  id() {
    return 'offline-safe-mock';
  }

  async callApi() {
    return {
      output: 'Запрос отклонён: ответ формируется только по разрешённым источникам.',
      metadata: { fixture: 'synthetic', network: false },
    };
  }
}
