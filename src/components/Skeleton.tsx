import type { CSSProperties } from 'react'

interface SkeletonProps {
  width: number | string
  height: number | string
  /** Render as a circle — used for avatar-sized placeholders. */
  round?: boolean
  /** Stagger the shimmer so a list of bars ripples instead of pulsing as one. */
  delay?: number
}

/** Shimmering placeholder bar. The `shimmer` keyframes live in index.css. */
export default function Skeleton({ width, height, round, delay = 0 }: SkeletonProps) {
  const style: CSSProperties = {
    width,
    height,
    borderRadius: round ? '50%' : 4,
    background:
      'linear-gradient(90deg, var(--border-1) 25%, var(--border-2) 37%, var(--border-1) 63%)',
    backgroundSize: '600px 100%',
    animation: 'shimmer 1.6s ease-in-out infinite',
    animationDelay: `${delay}ms`,
  }
  return <div style={style} />
}

/** Shared card chrome, used by most panels across the app. */
export const card: CSSProperties = {
  background: 'var(--bg-4)',
  border: '1px solid var(--border-2)',
  borderRadius: 8,
}
