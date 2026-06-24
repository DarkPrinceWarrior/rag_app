// Логотип DocRAGenslate: градиентная плитка + документ + cyan-бейдж с
// двусторонней стрелкой «перевод» (⇄). Только фигуры, без зависимости от
// шрифтов — чёткий на любом размере, включая favicon 16px.
export function Logo({ size = 26, className }: { size?: number; className?: string }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
      className={className}
    >
      <defs>
        <linearGradient id="dragTile" x1="0" y1="0" x2="32" y2="32" gradientUnits="userSpaceOnUse">
          <stop stopColor="#4F46E5" />
          <stop offset="1" stopColor="#2563EB" />
        </linearGradient>
        <linearGradient id="dragBadge" x1="17" y1="17" x2="28" y2="28" gradientUnits="userSpaceOnUse">
          <stop stopColor="#22D3EE" />
          <stop offset="1" stopColor="#38BDF8" />
        </linearGradient>
      </defs>
      <rect width="32" height="32" rx="8.5" fill="url(#dragTile)" />
      {/* документ-источник (позади) */}
      <rect x="12.2" y="6.4" width="12.4" height="15.4" rx="2.4" fill="#fff" opacity="0.4" />
      {/* документ (впереди) со строками текста */}
      <rect x="7.4" y="7.9" width="13" height="16.4" rx="2.6" fill="#fff" />
      <rect x="9.9" y="11.2" width="8" height="1.7" rx="0.85" fill="#2563EB" />
      <rect x="9.9" y="14.3" width="8" height="1.7" rx="0.85" fill="#9AB0F6" />
      <rect x="9.9" y="17.4" width="5.2" height="1.7" rx="0.85" fill="#9AB0F6" />
      {/* бейдж «перевод» с двусторонней стрелкой ⇄ */}
      <circle cx="22.6" cy="22.6" r="6.4" fill="#fff" />
      <circle cx="22.6" cy="22.6" r="5.5" fill="url(#dragBadge)" />
      <g stroke="#fff" strokeWidth="1.25" strokeLinecap="round" strokeLinejoin="round" fill="none">
        <path d="M19.7 21.2 H24.9 M23.5 19.9 L25.4 21.2 L23.5 22.5" />
        <path d="M25.5 24 H20.3 M21.7 22.7 L19.8 24 L21.7 25.3" />
      </g>
    </svg>
  )
}
