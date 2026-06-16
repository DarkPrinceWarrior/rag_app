// MinerU отдаёт формулы/единицы/«°C» как LaTeX ($1 3 0 ^{\circ}\complement$,
// ${\mathrm{м}}^{3}/\mathrm{ч}$). Чистим к читаемому виду (надстрочные → Unicode).
// Общий модуль: используется во вьювере и в панели источника чата.
const SUP: Record<string, string> = {
  '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
  '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
}

export function cleanMath(s: string): string {
  return s
    .replace(/\$([^$]*)\$/g, (_m, x) => x) // снять $…$
    .replace(/\\mathrm\s*/g, '') // \mathrm{X} → {X}
    .replace(/\\circ/g, '°')
    .replace(/\\complement/g, 'C')
    .replace(/\\times/g, '×')
    .replace(/\\,/g, ' ')
    .replace(/\^\s*\{?\s*([0-9])\s*\}?/g, (_m, d: string) => SUP[d] ?? d) // ^{3} → ³
    .replace(/\^\s*\{\s*([^}]*)\}/g, '$1') // прочие ^{…} → …
    .replace(/[{}]/g, '') // убрать скобки группировки
    .replace(/\\\*/g, '*')
    .replace(/\s*°\s*C/g, ' °C')
    .replace(/[ \t]{2,}/g, ' ')
    .trim()
}
