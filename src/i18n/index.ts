import { useAppStore } from '../store/useAppStore'

import en from './en.json'
import fr from './fr.json'

export type Lang = 'fr' | 'en'

/** Every key present in the French dictionary — the reference locale. */
export type TKey = keyof typeof fr

const dictionaries: Record<Lang, Record<string, string>> = { fr, en }

/**
 * Look up a translation, falling back the same way the original app did:
 * requested locale, then French, then the key itself so a missing entry is
 * visible in the UI rather than blank.
 *
 * Placeholders are written `{name}` and replaced from `params`.
 */
export function translate(
  lang: Lang,
  key: TKey,
  params?: Record<string, string | number>,
): string {
  const table = dictionaries[lang] ?? dictionaries.fr
  let out = table[key] ?? dictionaries.fr[key] ?? key

  if (params) {
    for (const p of Object.keys(params)) {
      out = out.replaceAll(`{${p}}`, String(params[p]))
    }
  }
  return out
}

/** Hook form — re-renders when the user switches language. */
export function useT() {
  const language = useAppStore((s) => s.language)
  return {
    t: (key: TKey, params?: Record<string, string | number>) =>
      translate(language, key, params),
    language,
  }
}
