// Логотип приложения: градиентная плитка с двумя наложенными «страницами»
// (оригинал + перевод) — без зависимости от шрифтов, чёткий на любом размере.
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
        <linearGradient id="tolmachLg" x1="0" y1="0" x2="32" y2="32" gradientUnits="userSpaceOnUse">
          <stop stopColor="#6366F1" />
          <stop offset="1" stopColor="#2563EB" />
        </linearGradient>
      </defs>
      <rect width="32" height="32" rx="8.5" fill="url(#tolmachLg)" />
      {/* документ-оригинал (позади) */}
      <rect x="12" y="7.3" width="12.4" height="15" rx="2.4" fill="#fff" opacity="0.42" />
      {/* документ-перевод (впереди) */}
      <rect x="7.6" y="9.4" width="12.4" height="15" rx="2.4" fill="#fff" />
      {/* строки текста на переднем документе */}
      <rect x="10" y="12.6" width="7.6" height="1.7" rx="0.85" fill="#2563EB" />
      <rect x="10" y="15.7" width="7.6" height="1.7" rx="0.85" fill="#9AB0F6" />
      <rect x="10" y="18.8" width="4.8" height="1.7" rx="0.85" fill="#9AB0F6" />
    </svg>
  )
}
