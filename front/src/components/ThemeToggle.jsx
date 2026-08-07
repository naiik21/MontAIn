import { useEffect, useState } from 'react'

/**
 * Alterna entre tema claro y oscuro. El script inline de Layout.astro decide
 * el tema inicial antes del primer pintado; aquí solo se lee y se cambia.
 */
export function ThemeToggle() {
  const [dark, setDark] = useState(false)

  useEffect(() => {
    setDark(document.documentElement.dataset.theme === 'dark')
  }, [])

  const toggle = () => {
    const next = !dark
    setDark(next)
    if (next) {
      document.documentElement.dataset.theme = 'dark'
    } else {
      delete document.documentElement.dataset.theme
    }
    localStorage.setItem('montain-theme', next ? 'dark' : 'light')
  }

  return (
    <button
      type='button'
      className='theme-toggle'
      onClick={toggle}
      aria-label={dark ? 'Cambiar a tema claro' : 'Cambiar a tema oscuro'}
    >
      {dark ? (
        /* sol */
        <svg width='16' height='16' viewBox='0 0 16 16' fill='none' aria-hidden='true'>
          <circle cx='8' cy='8' r='3.2' stroke='currentColor' strokeWidth='1.4' />
          <path
            d='M8 1.2v1.8M8 13v1.8M1.2 8H3M13 8h1.8M3.2 3.2l1.3 1.3M11.5 11.5l1.3 1.3M12.8 3.2l-1.3 1.3M4.5 11.5l-1.3 1.3'
            stroke='currentColor'
            strokeWidth='1.4'
            strokeLinecap='round'
          />
        </svg>
      ) : (
        /* luna */
        <svg width='16' height='16' viewBox='0 0 16 16' fill='none' aria-hidden='true'>
          <path
            d='M13.5 9.5A6 6 0 0 1 6.5 2.5a6 6 0 1 0 7 7Z'
            stroke='currentColor'
            strokeWidth='1.4'
            strokeLinejoin='round'
          />
        </svg>
      )}
    </button>
  )
}
