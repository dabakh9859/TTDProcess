import { useEffect } from 'react'

import { useAppStore } from '../store/useAppStore'

/**
 * Applies the theme by stamping `data-theme` on <html>; the palettes
 * themselves live in theme.css, imported globally from index.css.
 *
 * Renders nothing — it exists only for that side effect, exactly as the
 * original did. The choice is persisted by the store, not from here.
 */
export default function ThemeProvider() {
  const theme = useAppStore((s) => s.theme)

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
  }, [theme])

  return null
}
