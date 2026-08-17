import { invoke as tauriInvoke } from '@tauri-apps/api/core'

import type { Command } from './commands'

/**
 * Thin wrapper around Tauri's `invoke`.
 *
 * Two reasons it exists rather than importing `invoke` everywhere:
 *   - `Command` constrains the name to a command the Rust side actually
 *     exposes, so a typo fails at compile time instead of at runtime.
 *   - Rust returns `Result<T, String>`; a rejected promise here always carries
 *     a string message, which `toErrorMessage` normalises for the UI.
 */
export async function invoke<T = unknown>(
  cmd: Command,
  args?: Record<string, unknown>,
): Promise<T> {
  return tauriInvoke<T>(cmd, args)
}

/** Turn whatever a rejected `invoke` threw into something displayable. */
export function toErrorMessage(e: unknown): string {
  if (typeof e === 'string') return e
  if (e instanceof Error) return e.message
  return String(e)
}
