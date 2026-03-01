/**
 * Lightweight event bus for cross-component hover coordination.
 *
 * Components emit/listen for the trajectory point index being hovered.
 * `null` means "nothing hovered" (clear highlight).
 */
type HoverListener = (trajectoryIndex: number | null, source: string) => void;

const listeners: HoverListener[] = [];

export function onHover(fn: HoverListener): void {
  listeners.push(fn);
}

export function emitHover(trajectoryIndex: number | null, source: string): void {
  for (const fn of listeners) fn(trajectoryIndex, source);
}
